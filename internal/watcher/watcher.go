package watcher

import (
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/errdefs"
	"github.com/fsnotify/fsnotify"
)

const (
	defaultDebounce = 750 * time.Millisecond
	maxDebounceWait = 10 * time.Second
	stallWarnAfter  = 30 * time.Second
)

type pendingIndex struct {
	timer *time.Timer
	since time.Time
}

type Indexer interface {
	Index(path string) error
	Delete(path string) error
}

type Watcher struct {
	indexer  Indexer
	config   *config.Config
	debounce time.Duration
	workers  chan struct{}

	mu      sync.Mutex
	watcher *fsnotify.Watcher
	running bool
	done    chan struct{}
	pending map[string]*pendingIndex
}

func New(indexer Indexer, cfg *config.Config) *Watcher {
	return &Watcher{
		indexer:  indexer,
		config:   cfg,
		debounce: defaultDebounce,
		workers:  make(chan struct{}, max(cfg.WorkerCount, 1)),
		pending:  make(map[string]*pendingIndex),
	}
}

func (w *Watcher) Start() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.running {
		return nil
	}

	fw, err := fsnotify.NewWatcher()
	if err != nil {
		return errdefs.NewCustomError(errdefs.ErrTypeWatcherFailed, "failed to create watcher", err)
	}

	for _, idxPath := range w.config.IndexPaths {
		if !idxPath.ShouldWatch() {
			log.Infof("skipping watch for %s (watch disabled)", idxPath.Path)
			continue
		}
		if err := w.addWatches(fw, idxPath.Path); err != nil {
			fw.Close()
			return errdefs.NewCustomError(errdefs.ErrTypeWatcherFailed, "failed to add watches", err)
		}
	}

	w.watcher = fw
	w.done = make(chan struct{})
	w.running = true

	go w.eventLoop(fw, w.done)
	log.Infof("watcher started")
	return nil
}

func (w *Watcher) Stop() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.running {
		return nil
	}

	w.running = false
	close(w.done)
	for path, p := range w.pending {
		p.timer.Stop()
		delete(w.pending, path)
	}
	err := w.watcher.Close()
	w.watcher = nil
	log.Infof("watcher stopped")
	return err
}

func (w *Watcher) IsRunning() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.running
}

func (w *Watcher) watchableDir(path string) bool {
	if !w.config.ShouldIndexDir(path) {
		return false
	}
	maxDepth := w.config.GetMaxDepth(path)
	return maxDepth <= 0 || w.config.GetDepth(path) < maxDepth
}

func (w *Watcher) addWatches(fw *fsnotify.Watcher, root string) error {
	watchCount := 0
	errorCount := 0

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Debugf("skipping %s: %v", path, err)
			if info != nil && info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if !info.IsDir() {
			return nil
		}

		if !w.watchableDir(path) {
			return filepath.SkipDir
		}

		if err := fw.Add(path); err != nil {
			errorCount++
			if errorCount == 1 {
				log.Warnf("failed to add watch for %s: %v", path, err)
			}
			return nil
		}

		watchCount++
		return nil
	})

	if errorCount > 0 {
		log.Warnf("failed to add %d watches (added %d successfully)", errorCount, watchCount)
		log.Infof("if you hit inotify limits, increase with: sudo sysctl fs.inotify.max_user_watches=524288")
	} else {
		log.Infof("added %d directory watches under %s", watchCount, root)
	}

	return err
}

func (w *Watcher) eventLoop(fw *fsnotify.Watcher, done <-chan struct{}) {
	for {
		select {
		case <-done:
			return
		case event, ok := <-fw.Events:
			if !ok {
				return
			}
			w.handleEvent(fw, event)
		case err, ok := <-fw.Errors:
			if !ok {
				return
			}
			log.Errorf("watcher error: %v", err)
		}
	}
}

func (w *Watcher) handleEvent(fw *fsnotify.Watcher, event fsnotify.Event) {
	path := event.Name

	switch {
	case event.Has(fsnotify.Remove), event.Has(fsnotify.Rename):
		w.cancelPending(path)
		if err := w.indexer.Delete(path); err != nil {
			log.Debugf("failed to delete %s: %v", path, err)
		}
	case event.Has(fsnotify.Create):
		w.handleCreate(fw, path)
	case event.Has(fsnotify.Write):
		w.scheduleIndex(path)
	}
}

func (w *Watcher) handleCreate(fw *fsnotify.Watcher, path string) {
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	if !info.IsDir() {
		w.scheduleIndex(path)
		return
	}
	if !w.watchableDir(path) {
		return
	}

	if err := w.addWatches(fw, path); err != nil {
		log.Debugf("failed to watch new dir %s: %v", path, err)
	}
	w.indexTree(path)
}

func (w *Watcher) indexTree(root string) {
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() && path != root && !w.watchableDir(path) {
			return filepath.SkipDir
		}
		w.scheduleIndex(path)
		return nil
	})
}

func (w *Watcher) scheduleIndex(path string) {
	if !w.config.ShouldIndexFile(path) {
		return
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.running {
		return
	}
	if p, ok := w.pending[path]; ok {
		if time.Since(p.since) < maxDebounceWait {
			p.timer.Reset(w.debounce)
		}
		return
	}
	w.pending[path] = &pendingIndex{
		timer: time.AfterFunc(w.debounce, func() { w.indexNow(path) }),
		since: time.Now(),
	}
}

func (w *Watcher) cancelPending(path string) {
	w.mu.Lock()
	defer w.mu.Unlock()

	p, ok := w.pending[path]
	if !ok {
		return
	}
	p.timer.Stop()
	delete(w.pending, path)
}

func (w *Watcher) indexNow(path string) {
	w.mu.Lock()
	delete(w.pending, path)
	running := w.running
	w.mu.Unlock()
	if !running {
		return
	}

	w.workers <- struct{}{}
	defer func() { <-w.workers }()

	start := time.Now()
	if err := w.indexer.Index(path); err != nil {
		log.Debugf("failed to index %s: %v", path, err)
	}
	if elapsed := time.Since(start); elapsed > stallWarnAfter {
		log.Warnf("indexing %s took %s, index updates may be stalled", path, elapsed.Round(time.Second))
	}
}
