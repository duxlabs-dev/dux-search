package service

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// DownloadState is the progress of the current or last model download.
type DownloadState struct {
	Active bool                                `json:"active"`
	Towers map[semantic.Modality]ModelProgress `json:"towers,omitempty"`
	Error  string                              `json:"error,omitempty"`
}

type downloadState struct {
	active bool
	towers map[semantic.Modality]ModelProgress
	err    string
}

// Status is the answer to semantic.status.
type Status struct {
	Enabled     bool                    `json:"enabled"`
	Embedder    semantic.EmbedderStatus `json:"embedder"`
	Model       any                     `json:"model,omitempty"`
	Collections map[string]int          `json:"collections"`
	// Pending counts items stored but not embedded yet, per collection.
	Pending  map[string]int `json:"pending,omitempty"`
	Indexing IndexProgress  `json:"indexing"`
	Download DownloadState  `json:"download"`
}

// Status reports everything a settings page needs in one call.
func (s *Service) Status() Status {
	st := Status{
		Enabled:     s.opt.Enabled,
		Collections: map[string]int{},
		Embedder:    semantic.EmbedderStatus{Reason: s.disabledErr().Error(), Loaded: []semantic.Modality{}},
	}
	if s.opt.Enabled {
		st.Embedder = s.opt.Embedder.Status()
		if st.Embedder.Loaded == nil {
			st.Embedder.Loaded = []semantic.Modality{}
		}
		for _, c := range s.opt.Store.Collections() {
			st.Collections[c.Name] = c.Count
		}
		if s.opt.Files != nil {
			st.Indexing = s.opt.Files.Progress()
		}
	}
	if s.opt.Models != nil {
		st.Model = s.opt.Models.Status()
	}
	s.mu.Lock()
	if len(s.pending) > 0 {
		st.Pending = make(map[string]int, len(s.pending))
		for k, v := range s.pending {
			st.Pending[k] = v
		}
	}
	st.Download = s.downloadLocked()
	s.mu.Unlock()
	return st
}

func (s *Service) downloadLocked() DownloadState {
	d := DownloadState{Active: s.dl.active, Error: s.dl.err}
	if len(s.dl.towers) > 0 {
		d.Towers = make(map[semantic.Modality]ModelProgress, len(s.dl.towers))
		for k, v := range s.dl.towers {
			d.Towers[k] = v
		}
	}
	return d
}

// Warm loads the given towers ahead of use (default: text) and resets the
// embedder's idle timer. A launcher calls it when it opens.
func (s *Service) Warm(ctx context.Context, mods ...semantic.Modality) error {
	if !s.opt.Enabled {
		return s.disabledErr()
	}
	if len(mods) == 0 {
		mods = []semantic.Modality{semantic.ModalityText}
	}
	for _, m := range mods {
		if err := validModality(m); err != nil {
			return err
		}
	}
	return s.opt.Embedder.Warm(ctx, mods...)
}

func validModality(m semantic.Modality) error {
	switch m {
	case semantic.ModalityText, semantic.ModalityImage, semantic.ModalityAudio:
		return nil
	}
	return fmt.Errorf("unknown modality %q (use text, image or audio)", m)
}

// ParseModalities validates a list of tower names from a request.
func ParseModalities(names []string) ([]semantic.Modality, error) {
	out := make([]semantic.Modality, 0, len(names))
	seen := map[semantic.Modality]bool{}
	for _, n := range names {
		m := semantic.Modality(n)
		if err := validModality(m); err != nil {
			return nil, err
		}
		if !seen[m] {
			seen[m] = true
			out = append(out, m)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out, nil
}

// OnModelProgress registers the function that receives every download progress
// report (the server publishes them as model.progress events).
func (s *Service) OnModelProgress(fn func(ModelProgress)) {
	s.mu.Lock()
	s.sink = fn
	s.mu.Unlock()
}

// ModelStatus is the installed-model status as the model manager reports it.
func (s *Service) ModelStatus() (any, error) {
	if s.opt.Models == nil {
		return nil, ErrNoModelManager
	}
	return s.opt.Models.Status(), nil
}

// ModelDownload starts downloading the towers in the background and returns at
// once; progress arrives through OnModelProgress and Status. Model management
// works whether or not semantic search is enabled, so a model can be installed
// before turning the feature on.
func (s *Service) ModelDownload(towers []semantic.Modality) error {
	if s.opt.Models == nil {
		return ErrNoModelManager
	}
	if len(towers) == 0 {
		return fmt.Errorf("no towers given (use text, image or audio)")
	}
	s.mu.Lock()
	if s.dl.active {
		s.mu.Unlock()
		return ErrDownloadRunning
	}
	s.dl = downloadState{active: true, towers: map[semantic.Modality]ModelProgress{}}
	s.mu.Unlock()

	ctx := s.base
	if ctx == nil {
		ctx = context.Background()
	}
	go func() {
		err := s.opt.Models.Download(ctx, towers, s.reportProgress)
		final := ModelProgress{State: StateFinished}
		s.mu.Lock()
		s.dl.active = false
		if err != nil {
			s.dl.err = err.Error()
			final = ModelProgress{State: StateFailed, Error: err.Error()}
		}
		sink := s.sink
		s.mu.Unlock()
		if sink != nil {
			sink(final)
		}
		if err == nil && s.opt.Enabled {
			s.wake() // embed whatever was waiting for the model
		}
	}()
	return nil
}

// progressInterval throttles the progress stream: a downloader reports every
// chunk, subscribers need a few updates a second.
const progressInterval = 100 * time.Millisecond

func (s *Service) reportProgress(p ModelProgress) {
	s.mu.Lock()
	prev, seen := s.dl.towers[p.Tower]
	if s.dl.towers != nil {
		s.dl.towers[p.Tower] = p // status always has the latest numbers
	}
	sink := s.sink
	send := !seen || prev.State != p.State || time.Since(s.lastSent[p.Tower]) >= progressInterval
	if send {
		if s.lastSent == nil {
			s.lastSent = map[semantic.Modality]time.Time{}
		}
		s.lastSent[p.Tower] = time.Now()
	}
	s.mu.Unlock()
	if send && sink != nil {
		sink(p)
	}
}

// ModelRemove deletes the towers' files. A running download blocks it.
func (s *Service) ModelRemove(towers []semantic.Modality) error {
	if s.opt.Models == nil {
		return ErrNoModelManager
	}
	if len(towers) == 0 {
		return fmt.Errorf("no towers given (use text, image or audio)")
	}
	s.mu.Lock()
	active := s.dl.active
	s.mu.Unlock()
	if active {
		return ErrDownloadRunning
	}
	return s.opt.Models.Remove(towers)
}

// FileChanged forwards a file event from the lexical indexer to the file
// embedding queue. It is a no-op while semantic search is off.
func (s *Service) FileChanged(path string, deleted bool) {
	if s.opt.Enabled && s.opt.Files != nil {
		s.opt.Files.Enqueue(path, deleted)
	}
}
