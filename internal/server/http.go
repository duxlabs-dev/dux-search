package server

import (
	"net/http"
	"time"

	"github.com/AvengeMedia/dankgo/httpapi"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/duxlabs-dev/dux-search/internal/api"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

func NewHTTP(addr string, indexer api.IndexerInterface, watcher api.WatcherInterface, opts ...RouterOption) *httpapi.Server {
	return httpapi.NewServer(addr, newHTTPHandler(indexer, watcher, opts...))
}

func newHTTPHandler(indexer api.IndexerInterface, watcher api.WatcherInterface, opts ...RouterOption) http.Handler {
	var sem Router // only used to collect the semantic option
	for _, opt := range opts {
		opt(&sem)
	}

	r := chi.NewRouter()

	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	r.Group(func(r chi.Router) {
		r.Use(middleware.RequestID)
		r.Use(middleware.Logger)
		r.Use(middleware.Recoverer)
		r.Use(middleware.Timeout(30 * time.Second))

		config := httpapi.NewHumaConfig("Dux Search API", "1.0.0",
			httpapi.WithDescription("Desktop filesystem search service with Bleve indexing"))
		config.DocsPath = ""
		humaAPI := humachi.New(r, config)

		r.Get("/docs", httpapi.DocsHandler("Dux Search API Reference"))

		srv := &api.Server{
			Indexer:  indexer,
			Watcher:  watcher,
			Semantic: sem.semantic,
		}

		api.RegisterHandlers(srv, humaAPI)
	})

	return r
}
