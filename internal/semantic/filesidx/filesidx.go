// Package filesidx embeds the content of the files dsearch indexes and keeps
// the vectors in the vector store: text and code files are split into chunks
// (collection files.text, 256 dimensions, ids "path#n"), images go to
// files.image (512 dimensions) and audio to files.audio (256 dimensions), both
// under the file path.
//
// File events arrive through Enqueue (or Reconcile, which compares a full file
// list with what was embedded). They are persisted in a bbolt queue, so work
// survives a restart, and are processed on one low priority thread. The
// indexer pauses while the machine runs on battery or the user is at the
// keyboard, and waits while the embedding model is not installed.
package filesidx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/model"
	"golang.org/x/text/encoding"
)

// Vector collections and their dimensions (matryoshka truncation of the 768
// dimensional model output).
const (
	CollectionText  = "files.text"
	CollectionImage = "files.image"
	CollectionAudio = "files.audio"

	DimText  = 256
	DimImage = 512
	DimAudio = 256
)

// Preprocessor turns media files into model inputs. It is implemented by the
// preprocess package; any error it returns marks the file as not indexable
// until the file changes.
type Preprocessor interface {
	Image(r io.Reader) (semantic.ImageInput, error)
	Audio(path string) (semantic.AudioInput, error)
}

// Config configures an Indexer. DefaultConfig fills in sensible values.
type Config struct {
	// StatePath is the bbolt file holding the queue and the per-file records.
	// Keep it next to the dsearch index, not inside it.
	StatePath string
	// ModelID names the embedding model build; a change re-embeds everything.
	ModelID string

	IndexText, IndexImages, IndexAudio bool

	// MaxChunksPerFile caps the vectors of one text file (the head of the file
	// is kept); MaxTextBytes caps the bytes read of a text file; the media
	// limits skip files above them.
	MaxChunksPerFile int
	MaxTextBytes     int64
	MaxImageBytes    int64
	MaxAudioBytes    int64

	// IsText says whether dsearch treats the path as a text or code file
	// (config.Config.IsTextFile). Allow applies dsearch's exclusion rules
	// (config.Config.ShouldIndexFile); a path it refuses is never embedded and
	// its vectors are removed. Nil funcs accept everything / use the default
	// text extension list.
	IsText func(path string) bool
	Allow  func(path string) bool
	// Preprocessor enables image and audio indexing; nil turns both off.
	Preprocessor Preprocessor
	// LegacyEncoding decodes text files that are not valid UTF-8. Nil picks a
	// Windows code page from the locale.
	LegacyEncoding encoding.Encoding

	// Background policy.
	PauseOnBattery  bool
	PauseWhenActive bool           // pause while the login manager reports the user as active
	IdleHintGrace   time.Duration  // see logindActivity
	Power           PowerSource    // nil: UPower, then /sys/class/power_supply
	Activity        ActivitySource // nil: logind IdleHint
	PolicyInterval  time.Duration  // how often the policy is re-evaluated
	ModelRetry      time.Duration  // how often an unavailable model is retried
	TextBatch       int            // chunks per EmbedDocuments call
}

// DefaultConfig returns the production defaults for a state file.
func DefaultConfig(statePath string) Config {
	return Config{
		StatePath:        statePath,
		IndexText:        true,
		IndexImages:      true,
		IndexAudio:       true,
		MaxChunksPerFile: 64,
		MaxTextBytes:     2 << 20,
		MaxImageBytes:    64 << 20,
		MaxAudioBytes:    256 << 20,
		PauseOnBattery:   true,
		PauseWhenActive:  true,
	}
}

func (c Config) withDefaults() Config {
	if c.ModelID == "" {
		c.ModelID = model.ModelID
	}
	if c.MaxChunksPerFile <= 0 {
		c.MaxChunksPerFile = 64
	}
	if c.MaxTextBytes <= 0 {
		c.MaxTextBytes = 2 << 20
	}
	if c.MaxImageBytes <= 0 {
		c.MaxImageBytes = 64 << 20
	}
	if c.MaxAudioBytes <= 0 {
		c.MaxAudioBytes = 256 << 20
	}
	if c.IdleHintGrace <= 0 {
		c.IdleHintGrace = 30 * time.Minute
	}
	if c.PolicyInterval <= 0 {
		c.PolicyInterval = 10 * time.Second
	}
	if c.ModelRetry <= 0 {
		c.ModelRetry = 30 * time.Second
	}
	if c.TextBatch <= 0 {
		c.TextBatch = 16
	}
	if c.IsText == nil {
		c.IsText = defaultIsText
	}
	if c.Allow == nil {
		c.Allow = func(string) bool { return true }
	}
	if c.LegacyEncoding == nil {
		c.LegacyEncoding = legacyEncodingFromLocale()
	}
	return c
}

// Progress is a snapshot of the indexing work.
type Progress struct {
	Pending      int    `json:"pending"`                 // files queued, not yet processed
	Done         int64  `json:"done"`                    // files processed since Start
	Failed       int64  `json:"failed"`                  // of those, files that cannot be indexed: binary, unreadable, unsupported, too large, or embedding failed for good
	Indexed      int    `json:"indexed"`                 // files the indexer knows (embedded or recorded as not indexable)
	PausedReason string `json:"paused_reason,omitempty"` // PauseBattery, PauseUserActive, PauseModelUnavailable
}

var (
	maxBuffered = 100_000         // events held in memory before Enqueue waits for the writer
	enqueueWait = 5 * time.Second // longest Enqueue waits for room before it drops an event
)

// Indexer maintains the file vectors.
type Indexer struct {
	cfg   Config
	emb   semantic.Embedder
	store semantic.VectorStore
	kinds map[string]bool // enabled kinds, after Preprocessor availability

	mu      sync.Mutex
	cond    *sync.Cond
	buf     map[string]Op
	started bool
	stopped bool
	st      *stateDB
	cancel  context.CancelFunc
	stopC   chan struct{} // closed when everything has shut down

	wake  chan struct{}
	flush chan struct{}

	queued       atomic.Int64 // items in the state queue
	draining     atomic.Int64 // events on their way from the buffer into the queue
	indexed      atomic.Int64 // records in the state file
	done, failed atomic.Int64
	policyReason atomic.Pointer[string]
	modelRetryAt atomic.Int64 // unix nanoseconds
	power        PowerSource
	activity     ActivitySource
	closers      []func()
}

// New creates an indexer. Nothing is opened until Start; events queued before
// Start are kept in memory and persisted once it runs.
func New(cfg Config, emb semantic.Embedder, store semantic.VectorStore) *Indexer {
	cfg = cfg.withDefaults()
	x := &Indexer{
		cfg:   cfg,
		emb:   emb,
		store: store,
		buf:   map[string]Op{},
		wake:  make(chan struct{}, 1),
		flush: make(chan struct{}, 1),
		stopC: make(chan struct{}),
		kinds: map[string]bool{
			kindText:  cfg.IndexText,
			kindImage: cfg.IndexImages && cfg.Preprocessor != nil,
			kindAudio: cfg.IndexAudio && cfg.Preprocessor != nil,
		},
	}
	x.cond = sync.NewCond(&x.mu)
	return x
}

// Enqueue records a file event. It returns quickly: events are batched into
// the persistent queue by a writer goroutine. The last event for a path wins.
// After a crash the last few hundred milliseconds of events can be missing;
// call Reconcile at startup to catch up.
func (x *Indexer) Enqueue(path string, op Op) {
	if (op != OpIndex && op != OpDelete) || !filepath.IsAbs(path) {
		return
	}
	path = filepath.Clean(path)
	x.mu.Lock()
	// Backpressure: wait for the writer, but never block the caller for long
	// when the state file cannot be written (Reconcile recovers dropped events).
	deadline := time.Now().Add(enqueueWait)
	for x.started && !x.stopped && len(x.buf) >= maxBuffered {
		if time.Now().After(deadline) {
			x.mu.Unlock()
			log.Warnf("semantic files: event queue is full, dropping %s event for %s", op, path)
			return
		}
		wake := time.AfterFunc(100*time.Millisecond, x.cond.Broadcast)
		x.cond.Wait()
		wake.Stop()
	}
	if x.stopped {
		x.mu.Unlock()
		return
	}
	x.buf[path] = op
	n := len(x.buf)
	x.mu.Unlock()
	if n >= 2000 {
		select {
		case x.flush <- struct{}{}:
		default:
		}
	}
}

// Progress reports the queue and the pause reason. Pending is exact when the
// indexer is quiet; while events are still being written to the queue it can
// count a path twice but never misses one.
func (x *Indexer) Progress() Progress {
	x.mu.Lock()
	buffered := len(x.buf)
	x.mu.Unlock()
	p := Progress{Pending: int(x.queued.Load()+x.draining.Load()) + buffered, Done: x.done.Load(), Failed: x.failed.Load(), Indexed: int(x.indexed.Load())}
	if p.Pending > 0 {
		p.PausedReason = x.pauseReason()
	}
	return p
}

func (x *Indexer) pauseReason() string {
	if r := x.policyReason.Load(); r != nil && *r != "" {
		return *r
	}
	if time.Now().UnixNano() < x.modelRetryAt.Load() {
		return PauseModelUnavailable
	}
	return ""
}

func (x *Indexer) kick() {
	select {
	case x.wake <- struct{}{}:
	default:
	}
}

// Start opens the state, prepares the collections, repairs inconsistencies
// between the state and the vector store, and launches the background work.
// It returns as soon as the setup is done; the indexer runs until ctx ends or
// Close is called.
func (x *Indexer) Start(ctx context.Context) error {
	x.mu.Lock()
	if x.started || x.stopped {
		x.mu.Unlock()
		return errors.New("file indexer was already started")
	}
	x.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(x.cfg.StatePath), 0o755); err != nil {
		return err
	}
	st, err := openState(x.cfg.StatePath, x.cfg.ModelID)
	if err != nil {
		return err
	}
	if err := x.prepare(st); err != nil {
		return errors.Join(err, st.close())
	}
	queued, err := st.pendingCount()
	if err != nil {
		return errors.Join(err, st.close())
	}
	indexed, err := st.fileCount()
	if err != nil {
		return errors.Join(err, st.close())
	}

	ctx, cancel := context.WithCancel(ctx)
	x.mu.Lock()
	x.st, x.cancel, x.started = st, cancel, true
	x.mu.Unlock()
	x.queued.Store(int64(queued))
	x.indexed.Store(int64(indexed))

	if x.cfg.Power == nil {
		p := newSystemPower()
		x.closers = append(x.closers, p.close)
		x.power = p
	} else {
		x.power = x.cfg.Power
	}
	if x.cfg.Activity == nil {
		a := newLogindActivity(x.cfg.IdleHintGrace)
		x.closers = append(x.closers, a.close)
		x.activity = a
	} else {
		x.activity = x.cfg.Activity
	}

	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); x.writer(ctx) }()
	go func() { defer wg.Done(); x.monitor(ctx) }()
	go func() { defer wg.Done(); x.worker(ctx) }()
	go func() {
		<-ctx.Done()
		wg.Wait()
		x.shutdown()
	}()
	x.kick()
	return nil
}

// Close stops the indexer and waits until the state file is closed.
func (x *Indexer) Close() error {
	x.mu.Lock()
	cancel, started := x.cancel, x.started
	x.mu.Unlock()
	if !started {
		x.mu.Lock()
		x.stopped = true
		x.mu.Unlock()
		x.cond.Broadcast()
		return nil
	}
	cancel()
	<-x.stopC
	return nil
}

// Wait blocks until the indexer has shut down.
func (x *Indexer) Wait() { <-x.stopC }

func (x *Indexer) shutdown() {
	x.mu.Lock()
	x.stopped = true
	st := x.st
	x.mu.Unlock()
	x.cond.Broadcast()
	for _, c := range x.closers {
		c()
	}
	if st != nil {
		if err := st.close(); err != nil {
			log.Warnf("semantic files: closing state: %v", err)
		}
	}
	close(x.stopC)
}

// writer moves buffered events into the persistent queue in batches.
func (x *Indexer) writer(ctx context.Context) {
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			x.drain()
			return
		case <-tick.C:
		case <-x.flush:
		}
		x.drain()
	}
}

func (x *Indexer) drain() {
	for {
		x.mu.Lock()
		if len(x.buf) == 0 {
			x.mu.Unlock()
			return
		}
		batch := x.buf
		x.buf = map[string]Op{}
		x.draining.Add(int64(len(batch)))
		x.mu.Unlock()
		x.cond.Broadcast()

		added, err := x.st.enqueueBatch(batch)
		x.queued.Add(int64(added))
		x.draining.Add(-int64(len(batch)))
		if err != nil {
			log.Errorf("semantic files: queueing %d events: %v", len(batch), err)
			x.mu.Lock()
			for p, op := range batch {
				if _, newer := x.buf[p]; !newer {
					x.buf[p] = op
				}
			}
			x.mu.Unlock()
			return // retried on the next tick
		}
		x.kick()
	}
}

// monitor re-evaluates the battery and user activity policy.
func (x *Indexer) monitor(ctx context.Context) {
	tick := time.NewTicker(x.cfg.PolicyInterval)
	defer tick.Stop()
	for {
		reason := x.evalPolicy(ctx)
		if old := x.policyReason.Swap(&reason); old == nil || *old != reason {
			if reason != "" {
				log.Infof("semantic files: indexing paused (%s)", reason)
			} else if old != nil {
				log.Infof("semantic files: indexing resumed")
			}
			x.kick()
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}

func (x *Indexer) evalPolicy(ctx context.Context) string {
	if x.cfg.PauseOnBattery {
		if on, err := x.power.OnBattery(ctx); err != nil {
			log.Debugf("semantic files: power state unknown: %v", err)
		} else if on {
			return PauseBattery
		}
	}
	if x.cfg.PauseWhenActive {
		if active, known, err := x.activity.UserActive(ctx); err != nil {
			log.Debugf("semantic files: user activity unknown: %v", err)
		} else if known && active {
			return PauseUserActive
		}
	}
	return ""
}

// Reconcile aligns the queue with the complete list of files dsearch has
// indexed: files that are new or changed since they were embedded are queued
// for indexing, and records of files missing from the list are queued for
// deletion. Run it after Start (in a goroutine of its own; it stats every
// file). Paths must be absolute.
func (x *Indexer) Reconcile(ctx context.Context, files iter.Seq[string]) error {
	x.mu.Lock()
	st := x.st
	x.mu.Unlock()
	if st == nil {
		return errClosed
	}
	seen := map[string]struct{}{}
	n := 0
	for p := range files {
		if n++; n%256 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if !filepath.IsAbs(p) {
			continue
		}
		p = filepath.Clean(p)
		seen[p] = struct{}{}
		work, err := x.needsWork(st, p)
		if err != nil {
			return err
		}
		if work {
			x.Enqueue(p, OpIndex)
		}
	}
	var gone []string
	err := st.forEachRecord(func(p string, _ record) error {
		if _, ok := seen[p]; !ok {
			gone = append(gone, p)
		}
		return ctx.Err()
	})
	if err != nil {
		return err
	}
	for _, p := range gone {
		x.Enqueue(p, OpDelete)
	}
	return nil
}

// needsWork says whether a listed file must be (re)processed.
func (x *Indexer) needsWork(st *stateDB, path string) (bool, error) {
	rec, has, err := st.record(path)
	if err != nil {
		return false, err
	}
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return has, nil
	}
	kind := x.kindOf(path)
	if kind == "" || !x.kinds[kind] || !x.cfg.Allow(path) {
		return has && rec.Kind != "", nil
	}
	return !has || !rec.current(info, kind), nil
}

// current says that nothing has to be done for the file as it is on disk.
func (r record) current(info os.FileInfo, kind string) bool {
	return r.Ver == pipelineVersion && r.Size == info.Size() && r.MTime == info.ModTime().UnixNano() &&
		(r.Kind == kind || (r.Kind == "" && r.Err != ""))
}

var imageExts = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".gif": true, ".webp": true, ".bmp": true}
var audioExts = map[string]bool{".wav": true, ".flac": true, ".mp3": true, ".ogg": true, ".oga": true}

// kindOf classifies a path by its extension.
func (x *Indexer) kindOf(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch {
	case imageExts[ext]:
		return kindImage
	case audioExts[ext]:
		return kindAudio
	case x.cfg.IsText(path):
		return kindText
	}
	return ""
}

func chunkID(path string, n int) string { return fmt.Sprintf("%s#%d", path, n) }
