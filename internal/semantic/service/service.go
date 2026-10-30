// Package service is the semantic layer of the dsearch daemon: it owns the
// embedder, the vector store and the collections, and answers queries against
// them. It depends on the embedder, model manager and file indexer only through
// the small interfaces below, so the daemon wires the real ones in one place
// (cmd/dsearch/semantic_wire.go) and tests use the fixture embedder.
package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/vecstore"
)

// Collection dimensions (Matryoshka truncation of the 768-d model output).
const (
	DimShell      = semantic.NativeDim
	DimFilesText  = 256
	DimFilesImage = 512
	DimFilesAudio = 256
)

// Names of the collections the file indexer owns. Clients cannot write them.
const (
	CollFilesText  = "files.text"
	CollFilesImage = "files.image"
	CollFilesAudio = "files.audio"
	filesPrefix    = "files."
)

const (
	maxItemsPerCollection = 100_000
	defaultCollectionHits = 8
	maxCollectionHits     = 1000
	embedBatch            = 16
	// embedFormat versions how an Item becomes embedder input; it is part of the
	// item hash so changing the composition re-embeds everything.
	embedFormat = "1"
)

var (
	// ErrDisabled is returned when semantic search is switched off in the config.
	ErrDisabled = errors.New("semantic search is disabled")
	// ErrDownloadRunning is returned when a model download is already in progress.
	ErrDownloadRunning = errors.New("a model download is already running")
	// ErrNoModelManager is returned when model management is not wired.
	ErrNoModelManager = errors.New("model management is not available")

	collectionName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$`)
)

// Store is what the service needs from the vector store: semantic.VectorStore
// plus collection management and the persisted item sets. *vecstore.Store
// implements it.
type Store interface {
	semantic.VectorStore
	DropCollection(name string) error
	Info(name string) (vecstore.CollectionInfo, bool)
	SearchCentered(name string, query []float32, k int) ([]semantic.Hit, vecstore.ScoreStats, error)
	Collections() []vecstore.CollectionInfo
	PutItems(name string, items []semantic.Item) error
	Items(name string) ([]semantic.Item, error)
	DeleteItems(name string) error
	ItemCollections() ([]string, error)
}

// ModelProgress is one download progress report (event model.progress).
type ModelProgress struct {
	Tower semantic.Modality `json:"tower"`
	Bytes int64             `json:"bytes"`
	Total int64             `json:"total"`
	State string            `json:"state"`
	Error string            `json:"error,omitempty"`
}

// Terminal states the service itself reports when a download call returns.
const (
	StateFinished = "finished"
	StateFailed   = "failed"
)

// ModelManager manages the model files (backed by internal/semantic/model).
type ModelManager interface {
	// Status describes what is installed; it is passed to clients unchanged.
	Status() any
	Download(ctx context.Context, towers []semantic.Modality, progress func(ModelProgress)) error
	Remove(towers []semantic.Modality) error
}

// IndexProgress is the file indexer's backlog.
type IndexProgress struct {
	Pending      int    `json:"pending"`
	Done         int    `json:"done"`
	PausedReason string `json:"paused_reason,omitempty"`
}

// FileIndexer is the background file embedding queue (internal/semantic/filesidx).
type FileIndexer interface {
	// Enqueue schedules a file for (re-)indexing or, with deleted, removal.
	Enqueue(path string, deleted bool)
	Progress() IndexProgress
}

// Options configure New.
type Options struct {
	Enabled bool
	// DisabledReason explains why a service that is not Enabled is off when the
	// config did not ask for that (e.g. the vector database could not be opened);
	// it is appended to the disabled error and shown by Status.
	DisabledReason string
	ModelID        string // identity of the model files, part of every item hash
	Embedder       semantic.Embedder
	Store          Store
	Models         ModelManager
	Files          FileIndexer

	// SetBudget bounds how long collections.set embeds synchronously before the
	// rest moves to the background (default 2s).
	SetBudget time.Duration
	// SearchTimeout bounds the embedding side of a search (default 10s). A cold
	// model can take seconds to load; past the timeout the search answers with its
	// lexical results and says why.
	SearchTimeout time.Duration
	// RetryInterval is how often the background worker retries while the model
	// is unavailable or failing (default 30s).
	RetryInterval time.Duration
}

// Service is safe for concurrent use.
type Service struct {
	opt Options

	locks sync.Map // collection name -> *collLock

	mu       sync.Mutex
	pending  map[string]int // collection -> items still to embed
	dl       downloadState
	lastSent map[semantic.Modality]time.Time
	sink     func(ModelProgress)
	lastErr  string

	kick   chan struct{}
	cancel context.CancelFunc
	done   chan struct{}
	base   context.Context
}

type collLock struct {
	mu sync.Mutex
	// want is raised by a writer that is waiting for mu, so a long background
	// embedding pass yields between batches instead of making the writer wait it out.
	want atomic.Int32
}

// New builds the service. It does not start any goroutine; call Start.
func New(opt Options) (*Service, error) {
	if opt.SetBudget <= 0 {
		opt.SetBudget = 2 * time.Second
	}
	if opt.SearchTimeout <= 0 {
		opt.SearchTimeout = 10 * time.Second
	}
	if opt.RetryInterval <= 0 {
		opt.RetryInterval = 30 * time.Second
	}
	if opt.Enabled {
		if opt.Embedder == nil || opt.Store == nil {
			return nil, errors.New("service: an enabled service needs an embedder and a store")
		}
		if opt.ModelID == "" {
			return nil, errors.New("service: ModelID is required")
		}
	}
	return &Service{
		opt:     opt,
		pending: map[string]int{},
		kick:    make(chan struct{}, 1),
		done:    make(chan struct{}),
	}, nil
}

// Enabled reports whether semantic search is on.
func (s *Service) Enabled() bool { return s.opt.Enabled }

// disabledErr is ErrDisabled, with the reason when there is one beyond the config.
func (s *Service) disabledErr() error {
	if s.opt.DisabledReason == "" {
		return ErrDisabled
	}
	return fmt.Errorf("%w: %s", ErrDisabled, s.opt.DisabledReason)
}

// Start prepares the stored collections and starts the background worker.
func (s *Service) Start(ctx context.Context) error {
	s.base, s.cancel = context.WithCancel(ctx)
	if !s.opt.Enabled {
		close(s.done)
		return nil
	}
	names, err := s.opt.Store.ItemCollections()
	if err != nil {
		return fmt.Errorf("service: listing stored collections: %w", err)
	}
	for _, name := range names {
		// A changed model or dimension empties the collection here, which is what
		// makes the worker embed its stored items again.
		if err := s.opt.Store.EnsureCollection(name, DimShell, s.opt.ModelID); err != nil {
			log.Warnf("semantic: collection %s: %v", name, err)
			continue
		}
		n, err := s.countPending(name)
		if err != nil {
			log.Warnf("semantic: collection %s: %v", name, err)
			continue
		}
		s.setPending(name, n)
	}
	go s.run()
	s.wake()
	return nil
}

// Close stops the worker and releases the embedder. The store belongs to the caller.
func (s *Service) Close() error {
	if s.cancel != nil {
		s.cancel()
		<-s.done
	}
	if s.opt.Enabled && s.opt.Embedder != nil {
		return s.opt.Embedder.Close()
	}
	return nil
}

func (s *Service) wake() {
	select {
	case s.kick <- struct{}{}:
	default:
	}
}

func (s *Service) lockFor(name string) *collLock {
	l, _ := s.locks.LoadOrStore(name, &collLock{})
	return l.(*collLock)
}

func (s *Service) setPending(name string, n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if n <= 0 {
		delete(s.pending, name)
		return
	}
	s.pending[name] = n
}

// ItemHash is the content hash of an item for this model and dimension: an
// unchanged hash is not embedded again.
func (s *Service) ItemHash(it semantic.Item, dim int) string {
	return itemHash(it, s.opt.ModelID, dim)
}

func itemHash(it semantic.Item, model string, dim int) string {
	h := sha256.New()
	for _, part := range []string{it.Title, it.Text, strings.Join(it.Keywords, "\x1f"), model, strconv.Itoa(dim), embedFormat} {
		h.Write([]byte(part))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// docFor is the embedder input for an item. Keywords are appended to the text
// because they are the synonyms that make a short title findable.
func docFor(it semantic.Item) semantic.TextDoc {
	text := it.Text
	if len(it.Keywords) > 0 {
		if text != "" {
			text += " "
		}
		text += "Keywords: " + strings.Join(it.Keywords, ", ")
	}
	return semantic.TextDoc{Title: it.Title, Text: text}
}

// fit cuts an embedding to the collection dimension.
func fit(v []float32, dim int) ([]float32, error) {
	if len(v) < dim {
		return nil, fmt.Errorf("embedder returned %d values, collection needs %d", len(v), dim)
	}
	if len(v) == dim {
		return v, nil
	}
	return semantic.Truncate(v, dim), nil
}

func validCollectionName(name string) error {
	if !collectionName.MatchString(name) {
		return fmt.Errorf("invalid collection name %q: use letters, digits and . _ : - (max 64)", name)
	}
	return nil
}

func isFilesCollection(name string) bool { return strings.HasPrefix(name, filesPrefix) }

// UnavailableEmbedder is the embedder of a daemon whose model runtime could not
// be set up: every call fails with ErrModelUnavailable and the reason, so stored
// collections wait and searches fall back to lexical results instead of the
// whole semantic layer being off.
func UnavailableEmbedder(reason string) semantic.Embedder { return unavailable{reason} }

type unavailable struct{ reason string }

func (u unavailable) err() error { return fmt.Errorf("%w: %s", semantic.ErrModelUnavailable, u.reason) }
func (u unavailable) EmbedQueries(context.Context, []string) ([][]float32, error) {
	return nil, u.err()
}
func (u unavailable) EmbedDocuments(context.Context, []semantic.TextDoc) ([][]float32, error) {
	return nil, u.err()
}
func (u unavailable) EmbedImages(context.Context, []semantic.ImageInput) ([][]float32, error) {
	return nil, u.err()
}
func (u unavailable) EmbedAudio(context.Context, []semantic.AudioInput) ([][]float32, error) {
	return nil, u.err()
}
func (u unavailable) Warm(context.Context, ...semantic.Modality) error { return u.err() }
func (u unavailable) Status() semantic.EmbedderStatus {
	return semantic.EmbedderStatus{Reason: u.reason, Loaded: []semantic.Modality{}}
}
func (u unavailable) Close() error { return nil }
