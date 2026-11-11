package service

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// SetResult is the answer to collections.set.
type SetResult struct {
	Collection string `json:"collection"`
	Total      int    `json:"total"`    // items in the collection now
	Embedded   int    `json:"embedded"` // embedded during this call
	Removed    int    `json:"removed"`  // vectors of items that are no longer in the set
	Pending    int    `json:"pending"`  // still to embed; the background worker finishes them
}

// DeleteResult is the answer to collections.delete.
type DeleteResult struct {
	Collection string `json:"collection"`
	Removed    int    `json:"removed"`
}

// SetCollection replaces the whole content of a caller-owned collection. Items
// whose hash is unchanged keep their vector. The set is stored first, so with no
// model (or on any embedding failure) nothing is lost: the background worker
// embeds the remainder as soon as the embedder works.
func (s *Service) SetCollection(ctx context.Context, name string, items []semantic.Item) (SetResult, error) {
	if !s.opt.Enabled {
		return SetResult{}, s.disabledErr()
	}
	if err := validCollectionName(name); err != nil {
		return SetResult{}, err
	}
	if isFilesCollection(name) {
		return SetResult{}, fmt.Errorf("collection %q is reserved for the file indexer", name)
	}
	if len(items) > maxItemsPerCollection {
		return SetResult{}, fmt.Errorf("collection %q has %d items, the limit is %d", name, len(items), maxItemsPerCollection)
	}
	seen := make(map[string]struct{}, len(items))
	for i, it := range items {
		if it.ID == "" {
			return SetResult{}, fmt.Errorf("item %d has no id", i)
		}
		if it.Title == "" && it.Text == "" && len(it.Keywords) == 0 {
			return SetResult{}, fmt.Errorf("item %q has no title, text or keywords to embed", it.ID)
		}
		if _, dup := seen[it.ID]; dup {
			return SetResult{}, fmt.Errorf("duplicate item id %q", it.ID)
		}
		seen[it.ID] = struct{}{}
	}

	l := s.lockFor(name)
	l.want.Add(1)
	l.mu.Lock()
	l.want.Add(-1)
	defer l.mu.Unlock()

	st := s.opt.Store
	if err := st.EnsureCollection(name, DimShell, s.opt.ModelID); err != nil {
		return SetResult{}, err
	}
	if err := st.PutItems(name, items); err != nil {
		return SetResult{}, err
	}
	stored, err := st.Hashes(name)
	if err != nil {
		return SetResult{}, err
	}
	res := SetResult{Collection: name, Total: len(items)}
	var gone []string
	for id := range stored {
		if _, keep := seen[id]; !keep {
			gone = append(gone, id)
		}
	}
	if len(gone) > 0 {
		if err := st.Delete(name, gone); err != nil {
			return SetResult{}, err
		}
		res.Removed = len(gone)
	}
	var todo []semantic.Item
	for _, it := range items {
		if stored[it.ID] != itemHash(it, s.opt.ModelID, DimShell) {
			todo = append(todo, it)
		}
	}

	deadline := time.Now().Add(s.opt.SetBudget)
	for len(todo) > 0 && time.Now().Before(deadline) && ctx.Err() == nil {
		n, err := s.embedItems(ctx, name, todo[:min(embedBatch, len(todo))])
		if err != nil {
			s.noteEmbedError(err)
			break
		}
		res.Embedded += n
		todo = todo[n:]
	}
	res.Pending = len(todo)
	s.setPending(name, res.Pending)
	if res.Pending > 0 {
		s.wake()
	}
	return res, nil
}

// DeleteCollection removes a caller-owned collection with its items and vectors.
func (s *Service) DeleteCollection(name string) (DeleteResult, error) {
	if !s.opt.Enabled {
		return DeleteResult{}, s.disabledErr()
	}
	if err := validCollectionName(name); err != nil {
		return DeleteResult{}, err
	}
	if isFilesCollection(name) {
		return DeleteResult{}, fmt.Errorf("collection %q is reserved for the file indexer", name)
	}
	l := s.lockFor(name)
	l.want.Add(1)
	l.mu.Lock()
	l.want.Add(-1)
	defer l.mu.Unlock()

	n, err := s.opt.Store.Count(name)
	if err != nil {
		return DeleteResult{}, err
	}
	if err := s.opt.Store.DeleteItems(name); err != nil {
		return DeleteResult{}, err
	}
	if err := s.opt.Store.DropCollection(name); err != nil {
		return DeleteResult{}, err
	}
	s.setPending(name, 0)
	return DeleteResult{Collection: name, Removed: n}, nil
}

// embedItems embeds one batch and stores the vectors. It returns how many were stored.
func (s *Service) embedItems(ctx context.Context, name string, batch []semantic.Item) (int, error) {
	docs := make([]semantic.TextDoc, len(batch))
	for i, it := range batch {
		docs[i] = docFor(it)
	}
	vecs, err := s.opt.Embedder.EmbedDocuments(ctx, docs)
	if err != nil {
		return 0, err
	}
	if len(vecs) != len(batch) {
		return 0, fmt.Errorf("embedder returned %d vectors for %d documents", len(vecs), len(batch))
	}
	out := make([]semantic.Vector, len(batch))
	for i, it := range batch {
		v, err := fit(vecs[i], DimShell)
		if err != nil {
			return 0, err
		}
		out[i] = semantic.Vector{ID: it.ID, Hash: itemHash(it, s.opt.ModelID, DimShell), Vec: v}
	}
	if err := s.opt.Store.Upsert(name, out); err != nil {
		return 0, err
	}
	return len(batch), nil
}

// countPending compares the stored items of a collection with its vectors.
func (s *Service) countPending(name string) (int, error) {
	items, err := s.opt.Store.Items(name)
	if err != nil {
		return 0, err
	}
	stored, err := s.opt.Store.Hashes(name)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, it := range items {
		if stored[it.ID] != itemHash(it, s.opt.ModelID, DimShell) {
			n++
		}
	}
	return n, nil
}

func (s *Service) noteEmbedError(err error) {
	msg := err.Error()
	s.mu.Lock()
	changed := msg != s.lastErr
	s.lastErr = msg
	s.mu.Unlock()
	if !changed {
		return
	}
	if errors.Is(err, semantic.ErrModelUnavailable) {
		log.Infof("semantic: embedding is waiting for the model: %v", err)
		return
	}
	log.Warnf("semantic: embedding failed, will retry: %v", err)
}

// run is the background worker. It only does work for collections with pending
// items, and sleeps otherwise: no polling cost in steady state.
func (s *Service) run() {
	defer close(s.done)
	tick := time.NewTicker(s.opt.RetryInterval)
	defer tick.Stop()
	for {
		select {
		case <-s.base.Done():
			return
		case <-s.kick:
		case <-tick.C:
		}
		s.mu.Lock()
		var names []string
		for n := range s.pending {
			names = append(names, n)
		}
		s.mu.Unlock()
		for _, name := range names {
			if s.base.Err() != nil {
				return
			}
			s.syncCollection(name)
		}
	}
}

// syncCollection embeds everything missing from one collection, batch by batch.
// A writer waiting for the collection (SetCollection/DeleteCollection) makes the
// pass yield; the writer recomputes the pending set itself.
func (s *Service) syncCollection(name string) {
	l := s.lockFor(name)
	l.mu.Lock()
	defer l.mu.Unlock()

	items, err := s.opt.Store.Items(name)
	if err != nil {
		s.noteEmbedError(err)
		return
	}
	stored, err := s.opt.Store.Hashes(name)
	if err != nil {
		s.noteEmbedError(err)
		return
	}
	var todo []semantic.Item
	for _, it := range items {
		if stored[it.ID] != itemHash(it, s.opt.ModelID, DimShell) {
			todo = append(todo, it)
		}
	}
	for len(todo) > 0 {
		if s.base.Err() != nil || l.want.Load() > 0 {
			s.setPending(name, len(todo))
			return
		}
		n, err := s.embedItems(s.base, name, todo[:min(embedBatch, len(todo))])
		if err != nil {
			s.setPending(name, len(todo))
			s.noteEmbedError(err)
			return
		}
		todo = todo[n:]
		s.setPending(name, len(todo))
	}
	s.mu.Lock()
	s.lastErr = ""
	s.mu.Unlock()
}
