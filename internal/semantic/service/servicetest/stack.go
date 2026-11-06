// Package servicetest builds a complete semantic stack for tests of the layers
// above the service (socket server, HTTP API, CLI): a real lexical index over
// real files, a real vector store, the real service, and the fixture embedder
// that answers with real model vectors.
package servicetest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service/fixture"
	"github.com/duxlabs-dev/dux-search/internal/semantic/vecstore"
)

// Stack is the assembled stack. Files maps a Go package to the .txt file that
// holds its documentation (the corpus the fixture vectors were made from).
type Stack struct {
	Root     string
	Config   *config.Config
	Indexer  *indexer.Indexer
	Store    *vecstore.Store
	Embedder *fixture.Embedder
	Service  *service.Service
	Data     *fixture.Data
	Files    map[string]string
}

// Options tune NewStack.
type Options struct {
	Disabled bool // build a disabled service (no store, no embedder)
	Models   service.ModelManager
	Indexing service.FileIndexer
	// NoFileVectors skips loading the chunk vectors into files.text.
	NoFileVectors bool
}

// NewStack builds the stack in temporary directories; everything is cleaned up
// with the test.
func NewStack(t testing.TB, o Options) *Stack {
	t.Helper()
	root := t.TempDir()
	cfg := config.Default()
	cfg.IndexPath = filepath.Join(root, ".dsearch", "index")
	cfg.IndexPaths = []config.IndexPath{{Path: root, MaxDepth: 10, ExcludeDirs: []string{}}}
	cfg.Semantic.Enabled = !o.Disabled
	cfg.Semantic.ModelDir = filepath.Join(root, ".dsearch", "model")
	cfg.BuildMaps()

	idx, err := indexer.New(cfg)
	must(t, err)
	t.Cleanup(func() { idx.Close() })

	s := &Stack{Root: root, Config: cfg, Indexer: idx, Data: fixture.MustLoad(), Files: map[string]string{}}
	chunks := map[string][]string{}
	var pkgs []string
	for _, d := range s.Data.FileDocs() {
		if _, ok := chunks[d.Pkg]; !ok {
			pkgs = append(pkgs, d.Pkg)
		}
		chunks[d.Pkg] = append(chunks[d.Pkg], d.Text)
	}
	for _, p := range pkgs {
		path := filepath.Join(root, strings.ReplaceAll(p, "/", "_")+".txt")
		must(t, os.WriteFile(path, []byte(strings.Join(chunks[p], "\n\n")), 0o644))
		must(t, idx.Index(path))
		s.Files[p] = path
	}

	opts := service.Options{Enabled: !o.Disabled, Models: o.Models, Files: o.Indexing, RetryInterval: 20 * time.Millisecond}
	if !o.Disabled {
		s.Embedder = fixture.NewEmbedder()
		s.Store, err = vecstore.Open(cfg.SemanticDBPath(), vecstore.Options{})
		must(t, err)
		t.Cleanup(func() { s.Store.Close() })
		opts.ModelID, opts.Embedder, opts.Store = "embeddinggemma-2-test", s.Embedder, s.Store
		if !o.NoFileVectors {
			must(t, s.Store.EnsureCollection(service.CollFilesText, service.DimFilesText, "files-model"))
			var vecs []semantic.Vector
			for _, d := range s.Data.FileDocs() {
				vecs = append(vecs, semantic.Vector{
					ID: fmt.Sprintf("%s#%d", s.Files[d.Pkg], d.Chunk), Hash: "h",
					Vec: semantic.Truncate(d.Vec, service.DimFilesText),
				})
			}
			must(t, s.Store.Upsert(service.CollFilesText, vecs))
		}
	}
	s.Service, err = service.New(opts)
	must(t, err)
	must(t, s.Service.Start(context.Background()))
	t.Cleanup(func() { s.Service.Close() })
	return s
}

// ActionItems are the 20 action descriptions of the fixture as collection items.
func (s *Stack) ActionItems() []semantic.Item {
	var items []semantic.Item
	for _, d := range s.Data.ActionDocs() {
		items = append(items, semantic.Item{ID: d.ID, Title: d.Title, Text: d.Text, Payload: map[string]any{"category": "system"}})
	}
	return items
}

func must(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
