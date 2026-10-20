package client

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/AvengeMedia/dankgo/ipc"
	bleve "github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/search"
	"github.com/duxlabs-dev/dux-search/internal/config"
)

const (
	socketWaitTime = 2 * time.Second
	socketPollGap  = 100 * time.Millisecond
)

var ErrServiceNotRunning = errors.New("service not running")

type SearchResult struct {
	*bleve.SearchResult
	DirectoryHits search.DocumentMatchCollection `json:"directory_hits,omitempty"`
}

func findRunningSocket() (string, error) {
	deadline := time.Now().Add(socketWaitTime)
	for {
		for _, name := range []string{"dsearch", "danksearch"} {
			if socketPath, err := ipc.FindRunningSocket(name); err == nil {
				return socketPath, nil
			}
		}
		if time.Now().After(deadline) {
			return "", ErrServiceNotRunning
		}
		time.Sleep(socketPollGap)
	}
}

func sendRequest(method string, params map[string]any) (json.RawMessage, error) {
	socketPath, err := findRunningSocket()
	if err != nil {
		return nil, err
	}

	c, err := ipc.Dial(socketPath)
	if err != nil {
		return nil, ErrServiceNotRunning
	}
	defer c.Close()

	resp, err := c.Call(ipc.Request{ID: 1, Method: method, Params: params})
	if err != nil {
		return nil, err
	}

	if resp.Error != "" {
		return nil, errors.New(resp.Error)
	}

	return json.Marshal(resp.Result)
}

type SearchOptions struct {
	Query           string
	Limit           int
	Field           string
	ContentType     string
	Extension       string
	Fuzzy           bool
	SortBy          string
	SortDesc        bool
	MinSize         int64
	MaxSize         int64
	ModifiedAfter   string
	Facets          []string
	Folder          string
	ExifMake        string
	ExifModel       string
	ExifDateAfter   string
	ExifDateBefore  string
	ExifMinISO      int
	ExifMaxISO      int
	ExifMinAperture float64
	ExifMaxAperture float64
	ExifMinFocalLen float64
	ExifMaxFocalLen float64
	ExifLatMin      float64
	ExifLatMax      float64
	ExifLonMin      float64
	ExifLonMax      float64
	XattrTags       string
	Type            string
}

func Search(query string, limit int) (*SearchResult, error) {
	return SearchWithOptions(&SearchOptions{
		Query: query,
		Limit: limit,
	})
}

func SearchWithOptions(opts *SearchOptions) (*SearchResult, error) {
	params := map[string]any{
		"query": opts.Query,
		"limit": opts.Limit,
	}

	// Add optional parameters
	if opts.Field != "" {
		params["field"] = opts.Field
	}
	if opts.ContentType != "" {
		params["content_type"] = opts.ContentType
	}
	if opts.Extension != "" {
		params["extension"] = opts.Extension
	}
	if opts.Fuzzy {
		params["fuzzy"] = opts.Fuzzy
	}
	if opts.SortBy != "" {
		params["sort_by"] = opts.SortBy
	}
	if opts.SortDesc {
		params["sort_desc"] = opts.SortDesc
	}
	if opts.MinSize > 0 {
		params["min_size"] = opts.MinSize
	}
	if opts.MaxSize > 0 {
		params["max_size"] = opts.MaxSize
	}
	if opts.ModifiedAfter != "" {
		params["modified_after"] = opts.ModifiedAfter
	}
	if len(opts.Facets) > 0 {
		params["facets"] = opts.Facets
	}
	if opts.Folder != "" {
		params["folder"] = opts.Folder
	}
	if opts.ExifMake != "" {
		params["exif_make"] = opts.ExifMake
	}
	if opts.ExifModel != "" {
		params["exif_model"] = opts.ExifModel
	}
	if opts.ExifDateAfter != "" {
		params["exif_date_after"] = opts.ExifDateAfter
	}
	if opts.ExifDateBefore != "" {
		params["exif_date_before"] = opts.ExifDateBefore
	}
	if opts.ExifMinISO > 0 {
		params["exif_min_iso"] = opts.ExifMinISO
	}
	if opts.ExifMaxISO > 0 {
		params["exif_max_iso"] = opts.ExifMaxISO
	}
	if opts.ExifMinAperture > 0 {
		params["exif_min_aperture"] = opts.ExifMinAperture
	}
	if opts.ExifMaxAperture > 0 {
		params["exif_max_aperture"] = opts.ExifMaxAperture
	}
	if opts.ExifMinFocalLen > 0 {
		params["exif_min_focal_len"] = opts.ExifMinFocalLen
	}
	if opts.ExifMaxFocalLen > 0 {
		params["exif_max_focal_len"] = opts.ExifMaxFocalLen
	}
	if opts.ExifLatMin != 0 {
		params["exif_lat_min"] = opts.ExifLatMin
	}
	if opts.ExifLatMax != 0 {
		params["exif_lat_max"] = opts.ExifLatMax
	}
	if opts.ExifLonMin != 0 {
		params["exif_lon_min"] = opts.ExifLonMin
	}
	if opts.ExifLonMax != 0 {
		params["exif_lon_max"] = opts.ExifLonMax
	}
	if opts.XattrTags != "" {
		params["xattr_tags"] = opts.XattrTags
	}
	if opts.Type != "" {
		params["type"] = opts.Type
	}

	result, err := sendRequest("search", params)
	if err != nil {
		return nil, err
	}

	searchResult := SearchResult{SearchResult: &bleve.SearchResult{}}
	if err := json.Unmarshal(result, &searchResult); err != nil {
		return nil, err
	}

	return &searchResult, nil
}

func Reindex() (string, error) {
	result, err := sendRequest("reindex", nil)
	if err != nil {
		return "", err
	}

	var resp struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		return "", err
	}

	return resp.Status, nil
}

func Sync() (string, error) {
	result, err := sendRequest("sync", nil)
	if err != nil {
		return "", err
	}

	var resp struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		return "", err
	}

	return resp.Status, nil
}

func Stats() (*config.IndexStats, error) {
	result, err := sendRequest("stats", nil)
	if err != nil {
		return nil, err
	}

	stats := &config.IndexStats{}
	if err := json.Unmarshal(result, stats); err != nil {
		return nil, err
	}

	return stats, nil
}

type FileEntry struct {
	Path    string `json:"path"`
	ModTime string `json:"mod_time"`
	Size    int64  `json:"size"`
}

type FilesResult struct {
	Files []FileEntry `json:"files"`
	Total int         `json:"total"`
}

func Files(prefix string, limit int) (*FilesResult, error) {
	params := map[string]any{
		"limit": limit,
	}
	if prefix != "" {
		params["prefix"] = prefix
	}

	result, err := sendRequest("index.files", params)
	if err != nil {
		return nil, err
	}

	var filesResult FilesResult
	if err := json.Unmarshal(result, &filesResult); err != nil {
		return nil, err
	}

	return &filesResult, nil
}

func WatchStatus() (string, error) {
	result, err := sendRequest("watch.status", nil)
	if err != nil {
		return "", err
	}

	var resp struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		return "", err
	}

	return resp.Status, nil
}

func WatchStart() (string, error) {
	result, err := sendRequest("watch.start", nil)
	if err != nil {
		return "", err
	}

	var resp struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		return "", err
	}

	return resp.Status, nil
}

func WatchStop() (string, error) {
	result, err := sendRequest("watch.stop", nil)
	if err != nil {
		return "", err
	}

	var resp struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		return "", err
	}

	return resp.Status, nil
}
