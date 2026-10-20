package api

import (
	"context"

	"github.com/AvengeMedia/dankgo/log"
	bleve "github.com/blevesearch/bleve/v2"
	"github.com/danielgtaylor/huma/v2"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
)

type IndexerInterface interface {
	Search(query string, limit int) (*bleve.SearchResult, error)
	SearchWithOptions(opts *indexer.SearchOptions) (*bleve.SearchResult, error)
	SearchAll(opts *indexer.SearchOptions) (*indexer.SearchResult, error)
	ReindexAll() error
	SyncIncremental() error
	Stats() *config.IndexStats
}

type WatcherInterface interface {
	Start() error
	Stop() error
	IsRunning() bool
}

type Server struct {
	Indexer IndexerInterface
	Watcher WatcherInterface
}

func (s *Server) busy() bool {
	stats := s.Indexer.Stats()
	return stats != nil && stats.Phase != "" && stats.Phase != indexer.PhaseIdle
}

type SearchInput struct {
	Query           string   `query:"q" minLength:"1" doc:"Search query" example:"mountain"`
	Limit           int      `query:"limit" default:"10" minimum:"1" maximum:"500" doc:"Maximum results"`
	Field           string   `query:"field" enum:"filename,body" doc:"Search specific field (empty for all)"`
	ContentType     string   `query:"content_type" doc:"Filter by MIME type" example:"image/jpeg"`
	Extension       string   `query:"ext" doc:"Filter by file extension" example:".jpg"`
	Fuzzy           bool     `query:"fuzzy" doc:"Enable fuzzy matching for typos"`
	SortBy          string   `query:"sort" enum:"score,mtime,size,filename,exif_date,iso,focal_length,aperture," default:"score" doc:"Sort results by field"`
	SortDesc        bool     `query:"desc" default:"true" doc:"Sort descending"`
	MinSize         int64    `query:"min_size" doc:"Minimum file size in bytes"`
	MaxSize         int64    `query:"max_size" doc:"Maximum file size in bytes"`
	ModifiedAfter   string   `query:"modified_after" doc:"Filter by modification date (RFC3339)" example:"2024-01-01T00:00:00Z"`
	Facets          []string `query:"facets" doc:"Enable facets for fields" example:"content_type"`
	Folder          string   `query:"folder" doc:"Filter by folder path" example:"/home/user/Pictures"`
	ExifMake        string   `query:"exif_make" doc:"Filter by camera make" example:"Canon"`
	ExifModel       string   `query:"exif_model" doc:"Filter by camera model" example:"Canon EOS 5D"`
	ExifDateAfter   string   `query:"exif_date_after" doc:"Photos taken after date" example:"2024-01-01T00:00:00Z"`
	ExifDateBefore  string   `query:"exif_date_before" doc:"Photos taken before date" example:"2024-12-31T23:59:59Z"`
	ExifMinISO      int      `query:"exif_min_iso" doc:"Minimum ISO value" example:"100"`
	ExifMaxISO      int      `query:"exif_max_iso" doc:"Maximum ISO value" example:"3200"`
	ExifMinAperture float64  `query:"exif_min_aperture" doc:"Minimum aperture (f-number)" example:"1.8"`
	ExifMaxAperture float64  `query:"exif_max_aperture" doc:"Maximum aperture (f-number)" example:"16"`
	ExifMinFocalLen float64  `query:"exif_min_focal_len" doc:"Minimum focal length in mm" example:"24"`
	ExifMaxFocalLen float64  `query:"exif_max_focal_len" doc:"Maximum focal length in mm" example:"200"`
	ExifLatMin      float64  `query:"exif_lat_min" doc:"Minimum GPS latitude" example:"40.0"`
	ExifLatMax      float64  `query:"exif_lat_max" doc:"Maximum GPS latitude" example:"41.0"`
	ExifLonMin      float64  `query:"exif_lon_min" doc:"Minimum GPS longitude" example:"-74.0"`
	ExifLonMax      float64  `query:"exif_lon_max" doc:"Maximum GPS longitude" example:"-73.0"`
	XattrTags       string   `query:"xattr_tags" doc:"Tags" example:"+must,should,-must-not"`
	Type            string   `query:"type" enum:"file,dir,all," doc:"Filter by type" example:"dir"`
}

type SearchOutput struct {
	Body *indexer.SearchResult
}

type ReindexOutput struct {
	Body struct {
		Status string `json:"status" example:"reindexing started"`
	}
}

type StatsOutput struct {
	Body *config.IndexStats
}

type WatchStatusOutput struct {
	Body struct {
		Status string `json:"status" enum:"running,stopped" example:"running"`
	}
}

type WatchActionOutput struct {
	Body struct {
		Status string `json:"status" example:"watcher started"`
	}
}

func RegisterHandlers(srv *Server, api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "search",
		Summary:     "Search indexed files",
		Description: "Advanced search with filters, facets, sorting, and metadata queries",
		Method:      "GET",
		Path:        "/search",
		Tags:        []string{"Search"},
	}, func(ctx context.Context, input *SearchInput) (*SearchOutput, error) {
		opts := &indexer.SearchOptions{
			Query:           input.Query,
			Limit:           input.Limit,
			Field:           input.Field,
			ContentType:     input.ContentType,
			Extension:       input.Extension,
			Fuzzy:           input.Fuzzy,
			SortBy:          input.SortBy,
			SortDesc:        input.SortDesc,
			MinSize:         input.MinSize,
			MaxSize:         input.MaxSize,
			ModifiedAfter:   input.ModifiedAfter,
			Facets:          input.Facets,
			Folder:          input.Folder,
			ExifMake:        input.ExifMake,
			ExifModel:       input.ExifModel,
			ExifDateAfter:   input.ExifDateAfter,
			ExifDateBefore:  input.ExifDateBefore,
			ExifMinISO:      input.ExifMinISO,
			ExifMaxISO:      input.ExifMaxISO,
			ExifMinAperture: input.ExifMinAperture,
			ExifMaxAperture: input.ExifMaxAperture,
			ExifMinFocalLen: input.ExifMinFocalLen,
			ExifMaxFocalLen: input.ExifMaxFocalLen,
			ExifLatMin:      input.ExifLatMin,
			ExifLatMax:      input.ExifLatMax,
			ExifLonMin:      input.ExifLonMin,
			ExifLonMax:      input.ExifLonMax,
			XattrTags:       input.XattrTags,
			Type:            input.Type,
		}

		if input.Type == "all" {
			result, err := srv.Indexer.SearchAll(opts)
			if err != nil {
				return nil, huma.Error400BadRequest("search failed", err)
			}
			return &SearchOutput{Body: result}, nil
		}

		result, err := srv.Indexer.SearchWithOptions(opts)
		if err != nil {
			return nil, huma.Error400BadRequest("search failed", err)
		}

		return &SearchOutput{Body: &indexer.SearchResult{SearchResult: result}}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "reindex",
		Summary:     "Trigger full reindex",
		Description: "Start a full reindex of all files (async operation)",
		Method:      "POST",
		Path:        "/reindex",
		Tags:        []string{"Index"},
	}, func(ctx context.Context, input *struct{}) (*ReindexOutput, error) {
		if srv.busy() {
			return nil, huma.Error409Conflict("index operation already in progress")
		}
		go func() {
			if err := srv.Indexer.ReindexAll(); err != nil {
				log.Errorf("reindex failed: %v", err)
			}
		}()

		return &ReindexOutput{
			Body: struct {
				Status string `json:"status" example:"reindexing started"`
			}{
				Status: "reindexing started",
			},
		}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "sync",
		Summary:     "Incremental sync",
		Description: "Add new/updated files, remove deleted files (async operation)",
		Method:      "POST",
		Path:        "/sync",
		Tags:        []string{"Index"},
	}, func(ctx context.Context, input *struct{}) (*ReindexOutput, error) {
		if srv.busy() {
			return nil, huma.Error409Conflict("index operation already in progress")
		}
		go func() {
			if err := srv.Indexer.SyncIncremental(); err != nil {
				log.Errorf("sync failed: %v", err)
			}
		}()

		return &ReindexOutput{
			Body: struct {
				Status string `json:"status" example:"reindexing started"`
			}{
				Status: "incremental sync started",
			},
		}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "stats",
		Summary:     "Get index statistics",
		Description: "Returns statistics about the indexed files",
		Method:      "GET",
		Path:        "/stats",
		Tags:        []string{"Index"},
	}, func(ctx context.Context, input *struct{}) (*StatsOutput, error) {
		stats := srv.Indexer.Stats()
		return &StatsOutput{Body: stats}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "watchStart",
		Summary:     "Start file watcher",
		Description: "Enable real-time file watching for incremental updates",
		Method:      "POST",
		Path:        "/watch/start",
		Tags:        []string{"Watch"},
	}, func(ctx context.Context, input *struct{}) (*WatchActionOutput, error) {
		if srv.Watcher.IsRunning() {
			return nil, huma.Error409Conflict("watcher already running")
		}

		if err := srv.Watcher.Start(); err != nil {
			return nil, huma.Error500InternalServerError("failed to start watcher", err)
		}

		return &WatchActionOutput{
			Body: struct {
				Status string `json:"status" example:"watcher started"`
			}{
				Status: "watcher started",
			},
		}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "watchStop",
		Summary:     "Stop file watcher",
		Description: "Disable real-time file watching",
		Method:      "POST",
		Path:        "/watch/stop",
		Tags:        []string{"Watch"},
	}, func(ctx context.Context, input *struct{}) (*WatchActionOutput, error) {
		if !srv.Watcher.IsRunning() {
			return nil, huma.Error409Conflict("watcher not running")
		}

		if err := srv.Watcher.Stop(); err != nil {
			return nil, huma.Error500InternalServerError("failed to stop watcher", err)
		}

		return &WatchActionOutput{
			Body: struct {
				Status string `json:"status" example:"watcher started"`
			}{
				Status: "watcher stopped",
			},
		}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "watchStatus",
		Summary:     "Get watcher status",
		Description: "Check if file watcher is currently running",
		Method:      "GET",
		Path:        "/watch/status",
		Tags:        []string{"Watch"},
	}, func(ctx context.Context, input *struct{}) (*WatchStatusOutput, error) {
		status := "stopped"
		if srv.Watcher.IsRunning() {
			status = "running"
		}

		return &WatchStatusOutput{
			Body: struct {
				Status string `json:"status" enum:"running,stopped" example:"running"`
			}{
				Status: status,
			},
		}, nil
	})
}
