package main

// This is the only file of the daemon that touches the embedding runtime
// (internal/semantic/embed), the model manager (internal/semantic/model) and the
// file embedding queue (internal/semantic/filesidx). Everything else talks to
// internal/semantic/service, which knows them through small interfaces.

import (
	"context"
	"io"
	"path/filepath"
	"slices"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed"
	"github.com/duxlabs-dev/dux-search/internal/semantic/filesidx"
	"github.com/duxlabs-dev/dux-search/internal/semantic/model"
	"github.com/duxlabs-dev/dux-search/internal/semantic/preprocess"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
	"github.com/duxlabs-dev/dux-search/internal/semantic/vecstore"
)

// modelID names the model build in the vector database; vectors made with
// another build are never mixed with these. It follows the pinned revision in
// the model package, so moving the pin embeds every collection again.
const modelID = model.ModelID

// semanticRuntime is the semantic half of a running daemon.
type semanticRuntime struct {
	svc    *service.Service
	cancel context.CancelFunc
	store  *vecstore.Store
	files  *filesidx.Indexer
}

// startSemantic builds the semantic service for cfg. It never fails the daemon:
// whatever goes wrong, lexical search keeps working and semantic.status says why
// the semantic side is off. With [semantic].enabled = false it opens nothing
// (no vector database, no helper process) and only keeps the model manager.
func startSemantic(cfg *config.Config, idx *indexer.Indexer) *semanticRuntime {
	ctx, cancel := context.WithCancel(context.Background())
	rt := &semanticRuntime{cancel: cancel}
	opts := service.Options{Models: newModelManager(cfg), ModelID: modelID}

	finish := func() *semanticRuntime {
		svc, err := service.New(opts)
		if err != nil {
			// only possible for an enabled service with missing parts: a wiring bug
			log.Errorf("semantic search disabled: %v", err)
			opts.Enabled, opts.DisabledReason = false, err.Error()
			svc, _ = service.New(opts)
		}
		if err := svc.Start(ctx); err != nil {
			log.Errorf("semantic service: %v", err)
		}
		rt.svc = svc
		return rt
	}

	if !cfg.Semantic.Enabled {
		return finish()
	}

	store, err := vecstore.Open(cfg.SemanticDBPath(), vecstore.Options{RecoverCorrupt: true})
	if err != nil {
		log.Errorf("semantic search is off, cannot open %s: %v", cfg.SemanticDBPath(), err)
		opts.DisabledReason = "vector database: " + err.Error()
		return finish()
	}
	rt.store = store
	if rep := store.Report(); rep.Recovered {
		log.Warnf("semantic: the vector database was damaged and was replaced (old file: %s); everything is embedded again", rep.RecoveredFromBackup)
	}

	emb, err := embed.New(embed.Config{
		ModelDir:   cfg.Semantic.ModelDir,
		ORTLibrary: cfg.Semantic.OnnxRuntimeLib,
		Threads:    cfg.Semantic.Threads,
		IdleUnload: cfg.Semantic.IdleUnload(),
	})
	if err != nil {
		log.Errorf("semantic: embedding runtime unavailable: %v", err)
		emb = service.UnavailableEmbedder(err.Error())
	}

	fcfg := filesidx.DefaultConfig(filepath.Join(filepath.Dir(cfg.SemanticDBPath()), "semantic-files.db"))
	fcfg.ModelID = modelID
	fcfg.IndexText = cfg.Semantic.IndexText
	fcfg.IndexImages = cfg.Semantic.IndexImages
	fcfg.IndexAudio = cfg.Semantic.IndexAudio
	fcfg.MaxChunksPerFile = cfg.Semantic.MaxChunksPerFile
	fcfg.PauseOnBattery = cfg.Semantic.PauseOnBattery
	fcfg.IsText = cfg.IsTextFile
	// ShouldIndexFile also answers index_all_files, which is about lexical
	// indexing of unknown types; only the exclusion rules apply here.
	fcfg.Allow = func(p string) bool { return cfg.ExclusionReason(p) == "" }
	fcfg.Preprocessor = mediaPreprocessor{}
	files := filesidx.New(fcfg, emb, store)
	rt.files = files

	opts.Enabled, opts.Embedder, opts.Store, opts.Files = true, emb, store, filesAdapter{files}
	finish()

	if err := files.Start(ctx); err != nil {
		log.Errorf("semantic: file indexing is off: %v", err)
	} else {
		// dsearch reports only changes, so files that were indexed before the
		// embedding queue existed (or while it was off) are found here.
		go func() {
			paths, err := idx.Paths()
			if err == nil {
				err = files.Reconcile(ctx, slices.Values(paths))
			}
			if err != nil && ctx.Err() == nil {
				log.Errorf("semantic: comparing indexed files with the embedding queue: %v", err)
			}
		}()
	}
	idx.SetChangeListener(func(path string, deleted bool) { rt.svc.FileChanged(path, deleted) })
	return rt
}

// close stops the semantic side; the store is closed last so nothing writes to it.
func (rt *semanticRuntime) close() {
	rt.cancel()
	// The queue goes first so no worker embeds into a closed helper or store.
	if rt.files != nil {
		if err := rt.files.Close(); err != nil {
			log.Warnf("semantic: closing the file embedding queue: %v", err)
		}
	}
	if err := rt.svc.Close(); err != nil {
		log.Warnf("semantic: closing the embedder: %v", err)
	}
	if rt.store != nil {
		if err := rt.store.Close(); err != nil {
			log.Warnf("semantic: closing the vector database: %v", err)
		}
	}
}

type filesAdapter struct{ x *filesidx.Indexer }

func (a filesAdapter) Enqueue(path string, deleted bool) {
	op := filesidx.OpIndex
	if deleted {
		op = filesidx.OpDelete
	}
	a.x.Enqueue(path, op)
}

func (a filesAdapter) Progress() service.IndexProgress {
	p := a.x.Progress()
	return service.IndexProgress{Pending: int(p.Pending), Done: int(p.Done), PausedReason: string(p.PausedReason)}
}

// modelAdapter exposes internal/semantic/model for one model directory.
type modelAdapter struct{ dir string }

// newModelManager manages the model files of cfg. It does not need the daemon,
// so the CLI uses it too when none is running.
func newModelManager(cfg *config.Config) service.ModelManager {
	return modelAdapter{dir: cfg.Semantic.ModelDir}
}

func (m modelAdapter) Status() any { return model.Status(m.dir) }

func (m modelAdapter) Download(ctx context.Context, towers []semantic.Modality, progress func(service.ModelProgress)) error {
	return model.Download(ctx, m.dir, towers, func(p model.Progress) {
		progress(service.ModelProgress{
			Tower: semantic.Modality(p.Tower),
			Bytes: int64(p.Bytes),
			Total: int64(p.Total),
			State: string(p.State),
		})
	})
}

func (m modelAdapter) Remove(towers []semantic.Modality) error { return model.Remove(m.dir, towers) }

type mediaPreprocessor struct{}

func (mediaPreprocessor) Image(r io.Reader) (semantic.ImageInput, error) { return preprocess.Image(r) }
func (mediaPreprocessor) Audio(path string) (semantic.AudioInput, error) {
	return preprocess.Audio(path)
}
