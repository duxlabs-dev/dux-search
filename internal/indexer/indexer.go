package indexer

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	bleve "github.com/blevesearch/bleve/v2"
	_ "github.com/blevesearch/bleve/v2/analysis/analyzer/custom"
	_ "github.com/blevesearch/bleve/v2/analysis/token/edgengram"
	_ "github.com/blevesearch/bleve/v2/analysis/token/lowercase"
	_ "github.com/blevesearch/bleve/v2/analysis/token/ngram"
	_ "github.com/blevesearch/bleve/v2/analysis/tokenizer/regexp"
	_ "github.com/blevesearch/bleve/v2/analysis/tokenizer/single"
	"github.com/blevesearch/bleve/v2/index/scorch"
	"github.com/blevesearch/bleve/v2/mapping"
	"github.com/blevesearch/bleve/v2/search"
	query "github.com/blevesearch/bleve/v2/search/query"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/errdefs"
	"github.com/duxlabs-dev/dux-search/internal/metastore"
	"github.com/pkg/xattr"
	"github.com/rwcarlsen/goexif/exif"
)

const SchemaVersion = 5

const (
	statsMetaKey      = "stats"
	rebuildReasonKey  = "rebuild_reason"
	asyncErrorHandler = "dsearch"
	exifTimeLayout    = "2006:01:02 15:04:05"
	deleteChunk       = 500
)

var openIndexers sync.Map

func init() {
	scorch.RegistryAsyncErrorCallbacks[asyncErrorHandler] = func(err error, path string) {
		log.Errorf("index background error at %s: %v", path, err)
		v, ok := openIndexers.Load(path)
		if !ok {
			return
		}
		v.(*Indexer).failFatally(err)
	}
}

type Document struct {
	Path           string    `json:"path"`
	Filename       string    `json:"filename"`
	FilenameSub    string    `json:"filename_sub"`
	FilenamePrefix string    `json:"filename_prefix"`
	FilenameWords  string    `json:"filename_words"`
	Body           string    `json:"body"`
	ContentType    string    `json:"content_type"`
	ModTime        time.Time `json:"mtime"`
	Size           int64     `json:"size"`
	Hash           string    `json:"hash"`
	ExifMake       string    `json:"exif_make,omitempty"`
	ExifModel      string    `json:"exif_model,omitempty"`
	ExifDateTime   time.Time `json:"exif_datetime,omitzero"`
	ExifLatitude   float64   `json:"exif_latitude,omitempty"`
	ExifLongitude  float64   `json:"exif_longitude,omitempty"`
	ExifISO        int       `json:"exif_iso,omitempty"`
	ExifFNumber    float64   `json:"exif_fnumber,omitempty"`
	ExifExposure   string    `json:"exif_exposure,omitempty"`
	ExifFocalLen   float64   `json:"exif_focal_length,omitempty"`
	XattrTags      []string  `json:"xattr_tags,omitempty"`
	DocType        string    `json:"doc_type"`
	Parents        []string  `json:"parents"`
}

type SearchResult struct {
	*bleve.SearchResult
	DirectoryHits search.DocumentMatchCollection `json:"directory_hits,omitempty"`
}

type Indexer struct {
	index         bleve.Index
	config        *config.Config
	meta          *metastore.Store
	mu            sync.RWMutex
	opMu          sync.Mutex
	indexComplete atomic.Bool

	phaseState     atomic.Pointer[phaseState]
	filesProcessed atomic.Int64
	bytesProcessed atomic.Int64

	fatal     chan error
	fatalOnce sync.Once
}

type phaseState struct {
	Name    string
	Started time.Time
}

const (
	PhaseIdle       = "idle"
	PhaseReindexing = "reindexing"
	PhaseSyncing    = "syncing"
)

var ErrBusy = errdefs.NewCustomError(errdefs.ErrTypeIndexBusy, "index operation already in progress", nil)

func (i *Indexer) setPhase(name string) {
	i.phaseState.Store(&phaseState{Name: name, Started: time.Now()})
	i.filesProcessed.Store(0)
	i.bytesProcessed.Store(0)
}

func (i *Indexer) clearPhase() {
	i.phaseState.Store(&phaseState{Name: PhaseIdle})
}

func (i *Indexer) Phase() (string, time.Time) {
	state := i.phaseState.Load()
	if state == nil {
		return PhaseIdle, time.Time{}
	}
	return state.Name, state.Started
}

func (i *Indexer) Busy() bool {
	phase, _ := i.Phase()
	return phase != PhaseIdle
}

func (i *Indexer) Progress() (files int64, bytes int64) {
	return i.filesProcessed.Load(), i.bytesProcessed.Load()
}

func (i *Indexer) CurrentSchemaVersion() (int, error) {
	val, err := i.meta.GetMeta("schema_version")
	if err != nil || val == "" {
		return 0, err
	}
	return strconv.Atoi(val)
}

type SearchOptions struct {
	Query           string   `json:"query"`
	Limit           int      `json:"limit"`
	Field           string   `json:"field,omitempty"`
	ContentType     string   `json:"content_type,omitempty"`
	Extension       string   `json:"extension,omitempty"`
	Fuzzy           bool     `json:"fuzzy,omitempty"`
	SortBy          string   `json:"sort_by,omitempty"`
	SortDesc        bool     `json:"sort_desc,omitempty"`
	MinSize         int64    `json:"min_size,omitempty"`
	MaxSize         int64    `json:"max_size,omitempty"`
	ModifiedAfter   string   `json:"modified_after,omitempty"`
	Facets          []string `json:"facets,omitempty"`
	Folder          string   `json:"folder,omitempty"`
	ExifMake        string   `json:"exif_make,omitempty"`
	ExifModel       string   `json:"exif_model,omitempty"`
	ExifDateAfter   string   `json:"exif_date_after,omitempty"`
	ExifDateBefore  string   `json:"exif_date_before,omitempty"`
	ExifMinISO      int      `json:"exif_min_iso,omitempty"`
	ExifMaxISO      int      `json:"exif_max_iso,omitempty"`
	ExifMinAperture float64  `json:"exif_min_aperture,omitempty"`
	ExifMaxAperture float64  `json:"exif_max_aperture,omitempty"`
	ExifMinFocalLen float64  `json:"exif_min_focal_len,omitempty"`
	ExifMaxFocalLen float64  `json:"exif_max_focal_len,omitempty"`
	ExifLatMin      float64  `json:"exif_lat_min,omitempty"`
	ExifLatMax      float64  `json:"exif_lat_max,omitempty"`
	ExifLonMin      float64  `json:"exif_lon_min,omitempty"`
	ExifLonMax      float64  `json:"exif_lon_max,omitempty"`
	XattrTags       string   `json:"xattr_tags,omitempty"`
	Type            string   `json:"type,omitempty"`
}

func New(cfg *config.Config) (*Indexer, error) {
	idx, err := openOrCreateIndex(cfg.IndexPath)
	if err != nil {
		return nil, errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "failed to open index", err)
	}

	meta, err := metastore.New(cfg.IndexPath)
	if err != nil {
		idx.Close()
		return nil, errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "failed to open metastore", err)
	}

	i := &Indexer{
		index:  idx,
		config: cfg,
		meta:   meta,
		fatal:  make(chan error, 1),
	}
	openIndexers.Store(cfg.IndexPath, i)

	count, err := idx.DocCount()
	if err == nil && count > 0 {
		i.indexComplete.Store(true)
		log.Infof("loaded existing index with %d documents", count)
	}

	return i, nil
}

// bleve's background merger and persister swallow panics and just exit, after
// which every write blocks forever; the callback is the only way to hear about it.
func (i *Indexer) Fatal() <-chan error {
	return i.fatal
}

func (i *Indexer) failFatally(err error) {
	i.fatalOnce.Do(func() {
		if putErr := i.meta.PutMeta(rebuildReasonKey, err.Error()); putErr != nil {
			log.Errorf("failed to mark index for rebuild: %v", putErr)
		}
		i.fatal <- errdefs.NewCustomError(errdefs.ErrTypeIndexCorrupted, "index background task failed, rebuild scheduled on next start", err)
	})
}

func openOrCreateIndex(path string) (bleve.Index, error) {
	idx, err := bleve.OpenUsing(path, getIndexConfig())
	switch {
	case err == bleve.ErrorIndexPathDoesNotExist:
		mapping := buildIndexMapping()
		idx, err = bleve.NewUsing(path, mapping, "scorch", "scorch", getIndexConfig())
		if err != nil {
			return nil, err
		}
		log.Infof("created new index at %s", path)
		return idx, nil
	case err != nil:
		return nil, fmt.Errorf("failed to open index at %s (try 'dsearch index generate' to rebuild): %w", path, err)
	}
	log.Infof("opened existing index at %s", path)
	return idx, nil
}

func getIndexConfig() map[string]any {
	return map[string]any{
		"create_if_missing":      true,
		"error_if_exists":        false,
		"unsafe_batch":           false,
		"asyncErrorCallbackName": asyncErrorHandler,
		"store":                  getStoreConfig(),
	}
}

func getStoreConfig() map[string]any {
	return map[string]any{
		"mmap":              false,
		"metrics":           false,
		"create_if_missing": true,
		"error_if_exists":   false,
	}
}

func mustAdd(err error) {
	if err != nil {
		panic(err)
	}
}

func buildIndexMapping() mapping.IndexMapping {
	m := bleve.NewIndexMapping()

	mustAdd(m.AddCustomAnalyzer("keyword_lc", map[string]any{
		"type":          "custom",
		"tokenizer":     "single",
		"token_filters": []string{"to_lower"},
	}))
	mustAdd(m.AddCustomTokenFilter("ngram_2_15", map[string]any{
		"type": "ngram",
		"min":  float64(2),
		"max":  float64(15),
	}))
	mustAdd(m.AddCustomTokenFilter("edge_ngram_2_30", map[string]any{
		"type": "edge_ngram",
		"min":  float64(2),
		"max":  float64(30),
	}))
	mustAdd(m.AddCustomAnalyzer("filename_ngram", map[string]any{
		"type":          "custom",
		"tokenizer":     "single",
		"token_filters": []string{"to_lower", "ngram_2_15"},
	}))
	mustAdd(m.AddCustomAnalyzer("filename_edge", map[string]any{
		"type":          "custom",
		"tokenizer":     "single",
		"token_filters": []string{"to_lower", "edge_ngram_2_30"},
	}))
	mustAdd(m.AddCustomTokenizer("filename_word_tok", map[string]any{
		"type":   "regexp",
		"regexp": `[\p{L}\p{N}]+`,
	}))
	mustAdd(m.AddCustomAnalyzer("filename_words", map[string]any{
		"type":          "custom",
		"tokenizer":     "filename_word_tok",
		"token_filters": []string{"to_lower"},
	}))

	docMapping := bleve.NewDocumentMapping()

	storedKeyword := func(name string, analyzer string, includeInAll bool) {
		f := bleve.NewTextFieldMapping()
		f.Analyzer = analyzer
		f.Store = true
		f.IncludeInAll = includeInAll
		docMapping.AddFieldMappingsAt(name, f)
	}
	unstoredText := func(name string, analyzer string) {
		f := bleve.NewTextFieldMapping()
		f.Analyzer = analyzer
		f.Store = false
		f.IncludeTermVectors = false
		docMapping.AddFieldMappingsAt(name, f)
	}
	storedNumeric := func(name string) {
		f := bleve.NewNumericFieldMapping()
		f.Store = true
		docMapping.AddFieldMappingsAt(name, f)
	}
	storedDateTime := func(name string) {
		f := bleve.NewDateTimeFieldMapping()
		f.Store = true
		docMapping.AddFieldMappingsAt(name, f)
	}

	storedKeyword("path", "keyword_lc", false)
	storedKeyword("filename", "keyword_lc", true)
	unstoredText("filename_sub", "filename_ngram")
	unstoredText("filename_prefix", "filename_edge")
	unstoredText("filename_words", "filename_words")
	unstoredText("body", "")
	storedKeyword("content_type", "", true)
	storedDateTime("mtime")
	storedNumeric("size")
	storedKeyword("hash", "keyword", true)
	storedKeyword("exif_make", "keyword_lc", false)
	storedKeyword("exif_model", "keyword_lc", false)
	storedDateTime("exif_datetime")
	storedNumeric("exif_latitude")
	storedNumeric("exif_longitude")
	storedNumeric("exif_iso")
	storedNumeric("exif_fnumber")
	storedKeyword("exif_exposure", "", true)
	storedNumeric("exif_focal_length")
	storedKeyword("doc_type", "keyword", false)
	unstoredText("parents", "keyword")

	xattrTagsField := bleve.NewKeywordFieldMapping()
	xattrTagsField.Store = true
	docMapping.AddFieldMappingsAt("xattr_tags", xattrTagsField)

	m.DefaultMapping = docMapping
	return m
}

func (i *Indexer) currentIndex() bleve.Index {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return i.index
}

func fileMeta(info os.FileInfo) metastore.FileMeta {
	meta := metastore.FileMeta{ModTime: info.ModTime()}
	if !info.IsDir() {
		meta.Size = info.Size()
	}
	return meta
}

func (i *Indexer) unchanged(path string, info os.FileInfo) bool {
	existing, found, err := i.meta.Get(path)
	if err != nil || !found {
		return false
	}
	current := fileMeta(info)
	return existing.ModTime.Equal(current.ModTime) && existing.Size == current.Size
}

func (i *Indexer) Index(path string) error {
	if !i.config.ShouldIndexFile(path) {
		return nil
	}

	info, err := os.Stat(path)
	if err != nil {
		if os.IsPermission(err) {
			return errdefs.NewCustomError(errdefs.ErrTypeFileAccessDenied, path, err)
		}
		return err
	}

	if i.unchanged(path, info) {
		return nil
	}

	doc, err := i.buildDocument(path, info)
	if err != nil {
		return err
	}

	if err := i.currentIndex().Index(path, doc); err != nil {
		return errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, path, err)
	}

	if err := i.meta.Put(path, fileMeta(info)); err != nil {
		log.Debugf("failed to update metastore for %s: %v", path, err)
	}
	i.indexComplete.Store(true)
	log.Debugf("indexed %s", path)
	return nil
}

func (i *Indexer) buildDocument(path string, info os.FileInfo) (doc *Document, err error) {
	defer func() {
		if r := recover(); r != nil {
			doc, err = nil, fmt.Errorf("panic while reading %s: %v", path, r)
		}
	}()

	if !info.IsDir() {
		return i.readDocument(path, info), nil
	}

	base := filepath.Base(path)
	return &Document{
		Path:           path,
		Filename:       base,
		FilenameSub:    base,
		FilenamePrefix: base,
		FilenameWords:  base,
		ContentType:    "inode/directory",
		ModTime:        info.ModTime(),
		DocType:        "dir",
		Parents:        ParentDirs(path),
	}, nil
}

func (i *Indexer) readDocument(path string, info os.FileInfo) *Document {
	filename := filepath.Base(path)
	contentType := mime.TypeByExtension(filepath.Ext(path))
	if contentType == "" {
		contentType = "application/octet-stream"
	}

	doc := &Document{
		Path:           path,
		Filename:       filename,
		FilenameSub:    filename,
		FilenamePrefix: filename,
		FilenameWords:  filename,
		ContentType:    contentType,
		ModTime:        info.ModTime(),
		Size:           info.Size(),
		DocType:        "file",
		Parents:        ParentDirs(path),
	}

	if i.config.IsTextFile(path) {
		i.readBody(path, doc)
	}

	if isImageFile(contentType) && i.config.ExtractExif(path) {
		i.extractExifData(path, doc)
	}

	if i.config.IndexXattrTags {
		i.extractXattrTags(path, doc)
	}

	return doc
}

func (i *Indexer) readBody(path string, doc *Document) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	content, err := io.ReadAll(io.LimitReader(f, i.config.MaxFileBytes))
	if err != nil {
		return
	}

	hash := sha256.Sum256(content)
	doc.Body = string(content)
	doc.Hash = hex.EncodeToString(hash[:])
}

func (i *Indexer) extractXattrTags(path string, doc *Document) {
	tags, err := xattr.Get(path, "user.xdg.tags")
	if err != nil || len(tags) == 0 {
		return
	}
	parsedTags, _ := csv.NewReader(bytes.NewReader(tags)).Read()
	if len(parsedTags) == 0 {
		return
	}
	slices.Sort(parsedTags)
	doc.XattrTags = slices.Compact(parsedTags)
}

func isImageFile(contentType string) bool {
	return strings.HasPrefix(contentType, "image/")
}

func (i *Indexer) extractExifData(path string, doc *Document) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	x, err := exif.Decode(f)
	if err != nil {
		return
	}

	if make, err := x.Get(exif.Make); err == nil {
		if makeStr, err := make.StringVal(); err == nil {
			doc.ExifMake = makeStr
		}
	}

	if model, err := x.Get(exif.Model); err == nil {
		if modelStr, err := model.StringVal(); err == nil {
			doc.ExifModel = modelStr
		}
	}

	if dt, err := x.DateTime(); err == nil {
		doc.ExifDateTime = dt
	}

	if lat, lon, err := x.LatLong(); err == nil {
		doc.ExifLatitude = lat
		doc.ExifLongitude = lon
	}

	if isoSpeed, err := x.Get(exif.ISOSpeedRatings); err == nil {
		if isoInt, err := isoSpeed.Int(0); err == nil {
			doc.ExifISO = isoInt
		}
	}

	if fNumber, err := x.Get(exif.FNumber); err == nil {
		if num, denom, err := fNumber.Rat2(0); err == nil && denom != 0 {
			doc.ExifFNumber = float64(num) / float64(denom)
		}
	}

	if expTime, err := x.Get(exif.ExposureTime); err == nil {
		if _, _, err := expTime.Rat2(0); err == nil {
			doc.ExifExposure = expTime.String()
		}
	}

	if focalLen, err := x.Get(exif.FocalLength); err == nil {
		if num, denom, err := focalLen.Rat2(0); err == nil && denom != 0 {
			doc.ExifFocalLen = float64(num) / float64(denom)
		}
	}
}

func (i *Indexer) Delete(path string) error {
	paths := []string{path}
	prefix := strings.TrimRight(path, "/") + "/"
	err := i.meta.ForEachPrefix(prefix, func(child string, _ metastore.FileMeta) error {
		paths = append(paths, child)
		return nil
	})
	if err != nil {
		log.Debugf("failed to list children of %s in metastore: %v", path, err)
	}

	if err := i.deletePaths(paths); err != nil {
		return err
	}
	log.Debugf("deleted %s from index (%d entries)", path, len(paths))
	return nil
}

func (i *Indexer) deletePaths(paths []string) error {
	idx := i.currentIndex()
	for chunk := range slices.Chunk(paths, deleteChunk) {
		batch := idx.NewBatch()
		for _, p := range chunk {
			batch.Delete(p)
		}
		if err := idx.Batch(batch); err != nil {
			return errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "delete failed", err)
		}
		if err := i.meta.DeleteBatch(chunk); err != nil {
			log.Debugf("failed to delete %d entries from metastore: %v", len(chunk), err)
		}
	}
	return nil
}

func (i *Indexer) Search(query string, limit int) (*bleve.SearchResult, error) {
	return i.SearchWithOptions(&SearchOptions{
		Query: query,
		Limit: limit,
	})
}

func parseExifDate(s string) (time.Time, bool) {
	for _, layout := range []string{exifTimeLayout, time.RFC3339} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

func numericRange(field string, lo, hi float64) query.Query {
	var minVal, maxVal *float64
	if lo != 0 {
		minVal = &lo
	}
	if hi != 0 {
		maxVal = &hi
	}
	q := bleve.NewNumericRangeInclusiveQuery(minVal, maxVal, nil, nil)
	q.SetField(field)
	return q
}

// ParentDirs lists every ancestor directory of path as the lowercased term the
// folder filter looks up, so narrowing to a folder is one term query instead
// of a prefix scan over every indexed path.
func ParentDirs(path string) []string {
	var out []string
	for dir := filepath.Dir(path); ; dir = filepath.Dir(dir) {
		out = append(out, folderTerm(dir))
		if dir == filepath.Dir(dir) {
			return out
		}
	}
}

func folderTerm(dir string) string {
	trimmed := strings.TrimRight(dir, "/")
	if trimmed == "" {
		return "/"
	}
	return strings.ToLower(trimmed)
}

func termQuery(field, term string) query.Query {
	q := bleve.NewTermQuery(term)
	q.SetField(field)
	return q
}

func (i *Indexer) buildMainQuery(opts *SearchOptions) query.Query {
	switch {
	case opts.Query == "*" || opts.Query == "":
		return bleve.NewMatchAllQuery()
	case opts.Field != "":
		return i.buildFieldQuery(opts.Query, opts.Field, opts.Fuzzy)
	}

	filenameQuery := i.buildFilenameQuery(opts.Query, 20.0, 10.0, opts.Fuzzy)
	bodyQuery := bleve.NewMatchQuery(opts.Query)
	bodyQuery.SetField("body")
	bodyQuery.SetBoost(1.0)
	return bleve.NewDisjunctionQuery(filenameQuery, bodyQuery)
}

func (i *Indexer) buildFilters(opts *SearchOptions) ([]query.Query, error) {
	filters := []query.Query{}

	if opts.ContentType != "" {
		ctQuery := bleve.NewMatchQuery(opts.ContentType)
		ctQuery.SetField("content_type")
		filters = append(filters, ctQuery)
	}

	if opts.Extension != "" {
		extQuery := bleve.NewWildcardQuery("*" + strings.ToLower(opts.Extension))
		extQuery.SetField("filename")
		filters = append(filters, extQuery)
	}

	if opts.MinSize > 0 || opts.MaxSize > 0 {
		filters = append(filters, numericRange("size", float64(opts.MinSize), float64(opts.MaxSize)))
	}

	if opts.ModifiedAfter != "" {
		if t, err := time.Parse(time.RFC3339, opts.ModifiedAfter); err == nil {
			dateQuery := bleve.NewDateRangeInclusiveQuery(t, time.Time{}, nil, nil)
			dateQuery.SetField("mtime")
			filters = append(filters, dateQuery)
		}
	}

	if opts.Folder != "" {
		info, err := os.Stat(opts.Folder)
		if err != nil || !info.IsDir() {
			return nil, errdefs.NewCustomError(errdefs.ErrTypeSearchFailed, fmt.Sprintf("folder not found: %s", opts.Folder), nil)
		}
		filters = append(filters, termQuery("parents", folderTerm(opts.Folder)))
	}

	switch opts.Type {
	case "dir", "file":
		filters = append(filters, termQuery("doc_type", opts.Type))
	case "all":
	default:
		excludeDirs := bleve.NewBooleanQuery()
		excludeDirs.AddMustNot(termQuery("doc_type", "dir"))
		filters = append(filters, excludeDirs)
	}

	if opts.ExifMake != "" {
		filters = append(filters, termQuery("exif_make", strings.ToLower(opts.ExifMake)))
	}

	if opts.ExifModel != "" {
		filters = append(filters, termQuery("exif_model", strings.ToLower(opts.ExifModel)))
	}

	if opts.ExifDateAfter != "" || opts.ExifDateBefore != "" {
		after, _ := parseExifDate(opts.ExifDateAfter)
		before, _ := parseExifDate(opts.ExifDateBefore)
		if !after.IsZero() || !before.IsZero() {
			dateQuery := bleve.NewDateRangeInclusiveQuery(after, before, nil, nil)
			dateQuery.SetField("exif_datetime")
			filters = append(filters, dateQuery)
		}
	}

	if opts.ExifMinISO > 0 || opts.ExifMaxISO > 0 {
		filters = append(filters, numericRange("exif_iso", float64(opts.ExifMinISO), float64(opts.ExifMaxISO)))
	}
	if opts.ExifMinAperture > 0 || opts.ExifMaxAperture > 0 {
		filters = append(filters, numericRange("exif_fnumber", opts.ExifMinAperture, opts.ExifMaxAperture))
	}
	if opts.ExifMinFocalLen > 0 || opts.ExifMaxFocalLen > 0 {
		filters = append(filters, numericRange("exif_focal_length", opts.ExifMinFocalLen, opts.ExifMaxFocalLen))
	}
	if opts.ExifLatMin != 0 || opts.ExifLatMax != 0 {
		filters = append(filters, numericRange("exif_latitude", opts.ExifLatMin, opts.ExifLatMax))
	}
	if opts.ExifLonMin != 0 || opts.ExifLonMax != 0 {
		filters = append(filters, numericRange("exif_longitude", opts.ExifLonMin, opts.ExifLonMax))
	}

	if i.config.IndexXattrTags && opts.XattrTags != "" {
		if tagsQuery := buildXattrTagsQuery(opts.XattrTags); tagsQuery != nil {
			filters = append(filters, tagsQuery)
		}
	}

	return filters, nil
}

func buildXattrTagsQuery(spec string) query.Query {
	tags, _ := csv.NewReader(strings.NewReader(spec)).Read()
	tagsQuery := bleve.NewBooleanQuery()
	added := 0
	for _, tag := range tags {
		if tag == "" {
			continue
		}

		addFn := tagsQuery.AddShould
		switch tag[0] {
		case '-':
			tag = tag[1:]
			addFn = tagsQuery.AddMustNot
		case '+':
			tag = tag[1:]
			addFn = tagsQuery.AddMust
		}
		if tag == "" {
			continue
		}

		addFn(termQuery("xattr_tags", tag))
		added++
	}
	if added == 0 {
		return nil
	}
	return tagsQuery
}

var sortFields = map[string]string{
	"mtime":             "mtime",
	"size":              "size",
	"filename":          "filename",
	"exif_date":         "exif_datetime",
	"exif_datetime":     "exif_datetime",
	"exif_iso":          "exif_iso",
	"iso":               "exif_iso",
	"exif_focal_length": "exif_focal_length",
	"focal_length":      "exif_focal_length",
	"exif_fnumber":      "exif_fnumber",
	"aperture":          "exif_fnumber",
}

func sortSpec(sortBy string, desc bool) string {
	field, ok := sortFields[sortBy]
	if !ok {
		return "-_score"
	}
	if desc {
		return "-" + field
	}
	return field
}

func (i *Indexer) SearchWithOptions(opts *SearchOptions) (*bleve.SearchResult, error) {
	if !i.indexComplete.Load() {
		return nil, errdefs.NewCustomError(errdefs.ErrTypeSearchFailed, "index not ready", nil)
	}

	if opts.Limit <= 0 {
		opts.Limit = 10
	}

	filters, err := i.buildFilters(opts)
	if err != nil {
		return nil, err
	}

	finalQuery := i.buildMainQuery(opts)
	if len(filters) > 0 {
		finalQuery = bleve.NewConjunctionQuery(append([]query.Query{finalQuery}, filters...)...)
	}

	req := bleve.NewSearchRequest(finalQuery)
	req.Size = opts.Limit
	req.Highlight = bleve.NewHighlight()
	for _, facet := range opts.Facets {
		req.AddFacet(facet, bleve.NewFacetRequest(facet, 10))
	}
	req.SortBy([]string{sortSpec(opts.SortBy, opts.SortDesc)})

	result, err := i.currentIndex().Search(req)
	if err != nil {
		return nil, errdefs.NewCustomError(errdefs.ErrTypeSearchFailed, opts.Query, err)
	}

	return result, nil
}

func (i *Indexer) SearchAll(opts *SearchOptions) (*SearchResult, error) {
	fileOpts := *opts
	fileOpts.Type = "file"
	fileResult, err := i.SearchWithOptions(&fileOpts)
	if err != nil {
		return nil, err
	}

	dirOpts := *opts
	dirOpts.Type = "dir"
	dirResult, err := i.SearchWithOptions(&dirOpts)
	if err != nil {
		return nil, err
	}

	return &SearchResult{
		SearchResult:  fileResult,
		DirectoryHits: dirResult.Hits,
	}, nil
}

func (i *Indexer) buildFilenameQuery(queryStr string, boostPrefix, boostContains float64, fuzzy bool) query.Query {
	q := strings.TrimSpace(queryStr)
	if q == "" {
		return bleve.NewMatchNoneQuery()
	}

	if strings.ContainsAny(q, "*?") {
		wildcardQuery := bleve.NewWildcardQuery(strings.ToLower(q))
		wildcardQuery.SetField("filename")
		return wildcardQuery
	}

	disj := bleve.NewDisjunctionQuery()

	prefixQuery := bleve.NewPrefixQuery(strings.ToLower(q))
	prefixQuery.SetField("filename_prefix")
	prefixQuery.SetBoost(boostPrefix)
	disj.AddQuery(prefixQuery)

	wordsQuery := bleve.NewMatchQuery(q)
	wordsQuery.SetField("filename_words")
	wordsQuery.SetBoost((boostPrefix + boostContains) / 2)
	disj.AddQuery(wordsQuery)

	if len(q) >= 2 {
		matchQuery := bleve.NewMatchQuery(q)
		matchQuery.SetField("filename_sub")
		matchQuery.SetBoost(boostContains)
		if !fuzzy {
			matchQuery.SetOperator(query.MatchQueryOperatorAnd)
		}
		disj.AddQuery(matchQuery)
	}

	return disj
}

func (i *Indexer) buildFieldQuery(queryStr, field string, fuzzy bool) query.Query {
	switch {
	case field == "filename":
		return i.buildFilenameQuery(queryStr, 2.0, 1.0, fuzzy)
	case field == "body" && fuzzy:
		q := bleve.NewFuzzyQuery(queryStr)
		q.SetField("body")
		return q
	}

	q := bleve.NewMatchQuery(queryStr)
	q.SetField(field)
	return q
}

type walkStats struct {
	files int64
	bytes int64
}

type fileJob struct {
	path string
	info os.FileInfo
}

func (i *Indexer) walkIndexPaths(bat *batcher, shouldIndex func(path string, info os.FileInfo) bool) walkStats {
	var stats walkStats
	jobs := make(chan fileJob, i.config.WorkerCount*2)
	var wg sync.WaitGroup

	for range max(i.config.WorkerCount, 1) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				doc, err := i.buildDocument(job.path, job.info)
				if err != nil {
					log.Warnf("failed to build doc %s: %v", job.path, err)
					continue
				}
				bat.submit(batchJob{path: job.path, doc: doc, meta: fileMeta(job.info)})

				atomic.AddInt64(&stats.files, 1)
				atomic.AddInt64(&stats.bytes, job.info.Size())
				i.filesProcessed.Add(1)
				i.bytesProcessed.Add(job.info.Size())
			}
		}()
	}

	for _, idxPath := range i.config.IndexPaths {
		log.Infof("walking %s (max_depth: %d)", idxPath.Path, idxPath.MaxDepth)

		walkFollowSymlinks(idxPath.Path, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				log.Debugf("skipping %s: %v", path, err)
				return nil
			}

			if info.IsDir() {
				if !i.walkableDir(path) {
					return filepath.SkipDir
				}
				if !shouldIndex(path, info) {
					return nil
				}
				doc, err := i.buildDocument(path, info)
				if err != nil {
					log.Debugf("failed to build directory doc %s: %v", path, err)
					return nil
				}
				bat.submit(batchJob{path: path, doc: doc, meta: fileMeta(info)})
				return nil
			}

			if !i.config.ShouldIndexFile(path) || !shouldIndex(path, info) {
				return nil
			}
			jobs <- fileJob{path: path, info: info}
			return nil
		})
	}

	close(jobs)
	wg.Wait()
	return stats
}

func (i *Indexer) walkableDir(path string) bool {
	if !i.config.ShouldIndexDir(path) {
		return false
	}
	maxDepth := i.config.GetMaxDepth(path)
	return maxDepth <= 0 || i.config.GetDepth(path) < maxDepth
}

func (i *Indexer) ReindexAll() error {
	if !i.opMu.TryLock() {
		return ErrBusy
	}
	defer i.opMu.Unlock()

	i.setPhase(PhaseReindexing)
	defer i.clearPhase()

	start := time.Now()

	if err := i.meta.Clear(); err != nil {
		return errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "failed to clear metastore", err)
	}

	if err := i.replaceIndex(); err != nil {
		return err
	}

	bat := newBatcher(i, defaultBatchSize, defaultBatchInterval)
	stats := i.walkIndexPaths(bat, func(string, os.FileInfo) bool { return true })
	bat.close()

	duration := time.Since(start)
	i.indexComplete.Store(true)
	i.finishOperation(int(stats.files), stats.bytes, duration)
	if err := i.meta.PutMeta(rebuildReasonKey, ""); err != nil {
		log.Warnf("failed to clear rebuild marker: %v", err)
	}

	log.Infof("reindex complete: %d files, %d bytes, took %s", stats.files, stats.bytes, duration)
	return nil
}

func (i *Indexer) replaceIndex() error {
	i.mu.Lock()
	defer i.mu.Unlock()

	indexPath := i.config.IndexPath
	if err := i.index.Close(); err != nil {
		return errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "failed to close index", err)
	}

	if err := removeIndexDir(indexPath); err != nil {
		return errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "failed to remove index", err)
	}

	newIndex, err := openOrCreateIndex(indexPath)
	if err != nil {
		return errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "failed to create new index", err)
	}

	i.index = newIndex
	i.indexComplete.Store(false)
	return nil
}

func (i *Indexer) finishOperation(totalFiles int, totalBytes int64, duration time.Duration) {
	if err := i.saveStatsDocument(totalFiles, totalBytes, duration); err != nil {
		log.Warnf("failed to save stats: %v", err)
	}
	if err := i.SaveSchemaVersion(); err != nil {
		log.Warnf("failed to save schema version: %v", err)
	}
}

func (i *Indexer) SyncIncremental() error {
	if !i.opMu.TryLock() {
		return ErrBusy
	}
	defer i.opMu.Unlock()

	i.setPhase(PhaseSyncing)
	defer i.clearPhase()

	start := time.Now()
	log.Infof("starting incremental sync")

	indexedPaths := make(map[string]metastore.FileMeta)
	if err := i.meta.ForEach(func(path string, meta metastore.FileMeta) error {
		indexedPaths[path] = meta
		return nil
	}); err != nil {
		return errdefs.NewCustomError(errdefs.ErrTypeIndexingFailed, "failed to read metastore", err)
	}

	var added, updated, unchanged int64
	bat := newBatcher(i, defaultBatchSize, defaultBatchInterval)
	stats := i.walkIndexPaths(bat, func(path string, info os.FileInfo) bool {
		existing, exists := indexedPaths[path]
		delete(indexedPaths, path)
		current := fileMeta(info)
		switch {
		case !exists:
			added++
			return true
		case !existing.ModTime.Equal(current.ModTime) || existing.Size != current.Size:
			updated++
			return true
		default:
			unchanged++
			return false
		}
	})
	bat.close()

	stale := make([]string, 0, len(indexedPaths))
	for path := range indexedPaths {
		stale = append(stale, path)
	}
	if err := i.deletePaths(stale); err != nil {
		log.Warnf("failed to delete stale entries: %v", err)
	}

	duration := time.Since(start)
	i.indexComplete.Store(true)

	count, _ := i.GetDocCount()
	i.finishOperation(int(count), stats.bytes, duration)

	log.Infof("incremental sync complete: +%d new, ~%d updated, -%d deleted, =%d unchanged, took %s",
		added, updated, len(stale), unchanged, duration)

	return nil
}

func (i *Indexer) Stats() *config.IndexStats {
	stats, err := i.calculateStats()
	if err != nil {
		log.Warnf("failed to calculate stats: %v", err)
		stats = &config.IndexStats{}
	}
	i.attachRuntimeStats(stats)
	return stats
}

func (i *Indexer) calculateStats() (*config.IndexStats, error) {
	count, err := i.GetDocCount()
	if err != nil {
		return nil, err
	}

	statsDoc, err := i.loadStatsDocument()
	if err != nil || statsDoc == nil {
		return &config.IndexStats{TotalFiles: int(count)}, nil
	}

	return statsDoc, nil
}

func (i *Indexer) attachRuntimeStats(stats *config.IndexStats) {
	phase, started := i.Phase()
	stats.Phase = phase
	stats.PhaseStartedAt = started
	stats.FilesProcessed, stats.BytesProcessed = i.Progress()
	stats.ExpectedSchemaVersion = SchemaVersion
	if v, err := i.CurrentSchemaVersion(); err == nil {
		stats.SchemaVersion = v
	}
}

type statsMetadata struct {
	TotalFiles    int       `json:"total_files"`
	TotalBytes    int64     `json:"total_bytes"`
	LastIndexTime time.Time `json:"last_index_time"`
	IndexDuration string    `json:"index_duration"`
}

func (i *Indexer) loadStatsDocument() (*config.IndexStats, error) {
	raw, err := i.meta.GetMeta(statsMetaKey)
	if err != nil || raw == "" {
		return nil, err
	}

	var stats statsMetadata
	if err := json.Unmarshal([]byte(raw), &stats); err != nil {
		return nil, err
	}

	return &config.IndexStats{
		TotalFiles:    stats.TotalFiles,
		TotalBytes:    stats.TotalBytes,
		LastIndexTime: stats.LastIndexTime,
		IndexDuration: stats.IndexDuration,
	}, nil
}

func (i *Indexer) saveStatsDocument(totalFiles int, totalBytes int64, duration time.Duration) error {
	raw, err := json.Marshal(statsMetadata{
		TotalFiles:    totalFiles,
		TotalBytes:    totalBytes,
		LastIndexTime: time.Now(),
		IndexDuration: duration.String(),
	})
	if err != nil {
		return err
	}
	return i.meta.PutMeta(statsMetaKey, string(raw))
}

func (i *Indexer) GetDocCount() (uint64, error) {
	return i.currentIndex().DocCount()
}

type FileEntry struct {
	Path    string    `json:"path"`
	ModTime time.Time `json:"mod_time"`
	Size    int64     `json:"size"`
}

func (i *Indexer) ListFiles(prefix string, limit int) ([]FileEntry, int, error) {
	var files []FileEntry
	total := 0

	iterFn := func(path string, meta metastore.FileMeta) error {
		total++
		if len(files) < limit {
			files = append(files, FileEntry{
				Path:    path,
				ModTime: meta.ModTime,
				Size:    meta.Size,
			})
		}
		return nil
	}

	var err error
	switch {
	case prefix != "":
		err = i.meta.ForEachPrefix(prefix, iterFn)
	default:
		err = i.meta.ForEach(iterFn)
	}

	return files, total, err
}

func (i *Indexer) NeedsReindex() bool {
	reason, _ := i.RebuildReason()
	if reason != "" {
		return true
	}
	v, err := i.CurrentSchemaVersion()
	return err != nil || v != SchemaVersion
}

func (i *Indexer) RebuildReason() (string, error) {
	return i.meta.GetMeta(rebuildReasonKey)
}

func (i *Indexer) SaveSchemaVersion() error {
	return i.meta.PutMeta("schema_version", strconv.Itoa(SchemaVersion))
}

func (i *Indexer) Close() error {
	openIndexers.Delete(i.config.IndexPath)
	i.mu.Lock()
	defer i.mu.Unlock()
	i.meta.Close()
	return i.index.Close()
}

func removeIndexDir(path string) error {
	if path == "" {
		return fmt.Errorf("index path is empty")
	}

	_, err := os.Stat(filepath.Join(path, "index_meta.json"))
	switch {
	case os.IsNotExist(err):
		if _, statErr := os.Stat(path); os.IsNotExist(statErr) {
			return nil
		}
		return fmt.Errorf("refusing to remove %s: not a bleve index directory", path)
	case err != nil:
		return fmt.Errorf("failed to check index directory %s: %w", path, err)
	}

	return os.RemoveAll(path)
}

func walkFollowSymlinks(root string, fn filepath.WalkFunc) {
	info, err := os.Stat(root)
	if err != nil {
		log.Warnf("skipping inaccessible root %s: %v", root, err)
		return
	}
	symWalk(root, info, fn, make(map[string]bool))
}

func symWalk(path string, info os.FileInfo, fn filepath.WalkFunc, visited map[string]bool) {
	if !info.IsDir() {
		_ = fn(path, info, nil)
		return
	}

	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		_ = fn(path, info, err)
		return
	}

	if visited[resolved] {
		return
	}
	visited[resolved] = true

	if err := fn(path, info, nil); err != nil {
		return
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		_ = fn(path, info, err)
		return
	}

	for _, e := range entries {
		child := filepath.Join(path, e.Name())
		childInfo, err := os.Stat(child)
		if err != nil {
			log.Debugf("skipping inaccessible path %s: %v", child, err)
			continue
		}
		symWalk(child, childInfo, fn, visited)
	}
}
