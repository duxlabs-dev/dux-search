package server

import (
	"context"
	"fmt"

	"github.com/AvengeMedia/dankgo/ipc"
	"github.com/AvengeMedia/dankgo/ipc/params"
	"github.com/AvengeMedia/dankgo/log"
	bleve "github.com/blevesearch/bleve/v2"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
)

type IndexerInterface interface {
	Search(query string, limit int) (*bleve.SearchResult, error)
	SearchWithOptions(opts *indexer.SearchOptions) (*bleve.SearchResult, error)
	SearchAll(opts *indexer.SearchOptions) (*indexer.SearchResult, error)
	ReindexAll() error
	SyncIncremental() error
	Stats() *config.IndexStats
	ListFiles(prefix string, limit int) ([]indexer.FileEntry, int, error)
}

type WatcherInterface interface {
	Start() error
	Stop() error
	IsRunning() bool
}

type Router struct {
	indexer  IndexerInterface
	watcher  WatcherInterface
	config   *config.Config
	semantic *service.Service
	bus      *ipc.EventBus
}

// RouterOption customizes NewRouter.
type RouterOption func(*Router)

// WithSemantic attaches the semantic service. Without it the router behaves as
// if semantic search were disabled.
func WithSemantic(svc *service.Service) RouterOption {
	return func(r *Router) { r.semantic = svc }
}

func NewRouter(indexer IndexerInterface, watcher WatcherInterface, cfg *config.Config, opts ...RouterOption) *Router {
	r := &Router{
		indexer: indexer,
		watcher: watcher,
		config:  cfg,
		bus:     ipc.NewEventBus(),
	}
	for _, opt := range opts {
		opt(r)
	}
	if r.semantic == nil {
		r.semantic, _ = service.New(service.Options{}) // disabled: cannot fail
	}
	r.semantic.OnModelProgress(func(p service.ModelProgress) { r.bus.Publish(TopicModelProgress, p) })
	return r
}

// Bus is the event bus subscribers listen on (see TopicModelProgress); NewUnix
// hands it to the socket server.
func (r *Router) Bus() *ipc.EventBus { return r.bus }

type IndexPathInfo struct {
	Path          string   `json:"path"`
	MaxDepth      int      `json:"max_depth"`
	ExcludeHidden bool     `json:"exclude_hidden"`
	ExcludeDirs   []string `json:"exclude_dirs"`
}

// Info is what a client needs to decide whether a search is covered by the
// index before asking it.
type Info struct {
	SchemaVersion         int             `json:"schema_version"`
	ExpectedSchemaVersion int             `json:"expected_schema_version"`
	Phase                 string          `json:"phase"`
	IndexPaths            []IndexPathInfo `json:"index_paths"`
}

func (r *Router) Handle(ctx context.Context, w *ipc.ConnWriter, req ipc.Request, _ *ipc.Subscriber) {
	log.Debugf("DMS API Request: method=%s id=%d", req.Method, req.ID)

	switch req.Method {
	case "search":
		r.handleSearch(ctx, w, req)
	case "reindex":
		r.handleReindex(w, req)
	case "sync":
		r.handleSync(w, req)
	case "stats":
		r.handleStats(w, req)
	case "info":
		r.handleInfo(w, req)
	case "index.files":
		r.handleIndexFiles(w, req)
	case "watch.start":
		r.handleWatchStart(w, req)
	case "watch.stop":
		r.handleWatchStop(w, req)
	case "watch.status":
		r.handleWatchStatus(w, req)
	case "collections.set":
		r.handleCollectionsSet(ctx, w, req)
	case "collections.delete":
		r.handleCollectionsDelete(w, req)
	case "semantic.status":
		r.handleSemanticStatus(w, req)
	case "semantic.warm":
		r.handleSemanticWarm(ctx, w, req)
	case "model.status":
		r.handleModelStatus(w, req)
	case "model.download":
		r.handleModelDownload(w, req)
	case "model.remove":
		r.handleModelRemove(w, req)
	default:
		ipc.RespondError(w, req.ID, fmt.Sprintf("unknown method: %s", req.Method))
	}
}

func (r *Router) handleSearch(ctx context.Context, w *ipc.ConnWriter, req ipc.Request) {
	query, err := params.String(req.Params, "query")
	if err != nil {
		ipc.RespondError(w, req.ID, "query parameter required")
		return
	}

	opts := &indexer.SearchOptions{
		Query:         query,
		Limit:         params.IntOpt(req.Params, "limit", 10),
		Field:         params.StringOpt(req.Params, "field", ""),
		ContentType:   params.StringOpt(req.Params, "content_type", ""),
		Extension:     params.StringOpt(req.Params, "extension", ""),
		Fuzzy:         params.BoolOpt(req.Params, "fuzzy", false),
		SortBy:        params.StringOpt(req.Params, "sort_by", ""),
		SortDesc:      params.BoolOpt(req.Params, "sort_desc", false),
		MinSize:       int64(params.FloatOpt(req.Params, "min_size", 0)),
		MaxSize:       int64(params.FloatOpt(req.Params, "max_size", 0)),
		ModifiedAfter: params.StringOpt(req.Params, "modified_after", ""),
		Folder:        params.StringOpt(req.Params, "folder", ""),
		Type:          params.StringOpt(req.Params, "type", ""),
		Facets:        params.StringSlice(req.Params, "facets"),
	}

	if semantic := params.BoolOpt(req.Params, "semantic", false); semantic {
		r.handleSemanticSearch(ctx, w, req, opts)
		return
	} else if len(params.StringSlice(req.Params, "collections")) > 0 {
		ipc.RespondError(w, req.ID, "collections requires semantic=true")
		return
	}

	if opts.Type == "all" {
		result, err := r.indexer.SearchAll(opts)
		if err != nil {
			ipc.RespondError(w, req.ID, fmt.Sprintf("search failed: %v", err))
			return
		}
		ipc.Respond(w, req.ID, result)
		return
	}

	result, err := r.indexer.SearchWithOptions(opts)
	if err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("search failed: %v", err))
		return
	}

	ipc.Respond(w, req.ID, &indexer.SearchResult{SearchResult: result})
}

func (r *Router) busy() bool {
	stats := r.indexer.Stats()
	return stats != nil && stats.Phase != "" && stats.Phase != indexer.PhaseIdle
}

func (r *Router) handleReindex(w *ipc.ConnWriter, req ipc.Request) {
	if r.busy() {
		ipc.RespondError(w, req.ID, "index operation already in progress")
		return
	}
	go func() {
		if err := r.indexer.ReindexAll(); err != nil {
			log.Errorf("reindex failed: %v", err)
		}
	}()

	ipc.Respond(w, req.ID, map[string]string{"status": "reindexing started"})
}

func (r *Router) handleSync(w *ipc.ConnWriter, req ipc.Request) {
	if r.busy() {
		ipc.RespondError(w, req.ID, "index operation already in progress")
		return
	}
	go func() {
		if err := r.indexer.SyncIncremental(); err != nil {
			log.Errorf("sync failed: %v", err)
		}
	}()

	ipc.Respond(w, req.ID, map[string]string{"status": "incremental sync started"})
}

func (r *Router) handleStats(w *ipc.ConnWriter, req ipc.Request) {
	ipc.Respond(w, req.ID, r.indexer.Stats())
}

func (r *Router) handleInfo(w *ipc.ConnWriter, req ipc.Request) {
	info := Info{ExpectedSchemaVersion: indexer.SchemaVersion, Phase: indexer.PhaseIdle, IndexPaths: []IndexPathInfo{}}
	if stats := r.indexer.Stats(); stats != nil {
		info.SchemaVersion = stats.SchemaVersion
		if stats.Phase != "" {
			info.Phase = stats.Phase
		}
	}
	if r.config != nil {
		for i := range r.config.IndexPaths {
			ip := &r.config.IndexPaths[i]
			excludes := ip.ResolvedExcludeDirs()
			if excludes == nil {
				excludes = []string{}
			}
			info.IndexPaths = append(info.IndexPaths, IndexPathInfo{Path: ip.Path, MaxDepth: ip.MaxDepth, ExcludeHidden: ip.ExcludeHidden, ExcludeDirs: excludes})
		}
	}
	ipc.Respond(w, req.ID, info)
}

func (r *Router) handleWatchStart(w *ipc.ConnWriter, req ipc.Request) {
	if r.watcher.IsRunning() {
		ipc.RespondError(w, req.ID, "watcher already running")
		return
	}

	if err := r.watcher.Start(); err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("failed to start watcher: %v", err))
		return
	}

	ipc.Respond(w, req.ID, map[string]string{"status": "watcher started"})
}

func (r *Router) handleWatchStop(w *ipc.ConnWriter, req ipc.Request) {
	if !r.watcher.IsRunning() {
		ipc.RespondError(w, req.ID, "watcher not running")
		return
	}

	if err := r.watcher.Stop(); err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("failed to stop watcher: %v", err))
		return
	}

	ipc.Respond(w, req.ID, map[string]string{"status": "watcher stopped"})
}

func (r *Router) handleWatchStatus(w *ipc.ConnWriter, req ipc.Request) {
	status := "stopped"
	if r.watcher.IsRunning() {
		status = "running"
	}

	ipc.Respond(w, req.ID, map[string]string{"status": status})
}

func (r *Router) handleIndexFiles(w *ipc.ConnWriter, req ipc.Request) {
	prefix := params.StringOpt(req.Params, "prefix", "")

	limit := 100
	if l := params.IntOpt(req.Params, "limit", 0); l > 0 {
		limit = l
	}

	files, total, err := r.indexer.ListFiles(prefix, limit)
	if err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("list files failed: %v", err))
		return
	}

	ipc.Respond(w, req.ID, map[string]any{
		"files": files,
		"total": total,
	})
}
