package watcher

import (
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/config"
	mocks_watcher "github.com/duxlabs-dev/dux-search/internal/mocks/watcher"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type countingIndexer struct {
	*mocks_watcher.MockIndexer
	indexed atomic.Int64
	deleted atomic.Int64
}

func newCountingIndexer(t *testing.T) *countingIndexer {
	ci := &countingIndexer{MockIndexer: mocks_watcher.NewMockIndexer(t)}
	ci.EXPECT().Index(mock.Anything).RunAndReturn(func(string) error {
		ci.indexed.Add(1)
		return nil
	}).Maybe()
	ci.EXPECT().Delete(mock.Anything).RunAndReturn(func(string) error {
		ci.deleted.Add(1)
		return nil
	}).Maybe()
	return ci
}

type WatcherSuite struct {
	suite.Suite
}

func newTestWatcher(idx Indexer, cfg *config.Config) *Watcher {
	w := New(idx, cfg)
	w.debounce = 20 * time.Millisecond
	return w
}

func testConfig(root string) *config.Config {
	cfg := config.Default()
	cfg.IndexPaths = []config.IndexPath{{Path: root, MaxDepth: 10}}
	cfg.BuildMaps()
	return cfg
}

func (s *WatcherSuite) eventually(cond func() bool) {
	s.Eventually(cond, 2*time.Second, 10*time.Millisecond)
}

func TestWatcherSuite(t *testing.T) {
	suite.Run(t, new(WatcherSuite))
}

func (s *WatcherSuite) TestNew() {
	cfg := config.Default()
	idx := newCountingIndexer(s.T())

	w := New(idx, cfg)
	s.Nil(w.watcher)
	s.Equal(idx, w.indexer)
	s.Equal(cfg, w.config)
}

func (s *WatcherSuite) TestStartStop() {
	tmpDir := s.T().TempDir()
	cfg := config.Default()
	cfg.IndexPaths = []config.IndexPath{
		{
			Path:          tmpDir,
			MaxDepth:      10,
			ExcludeHidden: false,
			ExcludeDirs:   []string{},
		},
	}
	cfg.BuildMaps()

	w := New(newCountingIndexer(s.T()), cfg)

	s.False(w.IsRunning())

	s.Require().NoError(w.Start())
	s.True(w.IsRunning())

	s.NoError(w.Start(), "Start() should be idempotent")

	s.NoError(w.Stop())
	s.False(w.IsRunning())
}

func (s *WatcherSuite) TestFileEvents() {
	tmpDir := s.T().TempDir()
	cfg := config.Default()
	cfg.IndexPaths = []config.IndexPath{
		{
			Path:          tmpDir,
			MaxDepth:      10,
			ExcludeHidden: false,
			ExcludeDirs:   []string{},
		},
	}
	cfg.BuildMaps()
	idx := newCountingIndexer(s.T())

	w := newTestWatcher(idx, cfg)
	s.Require().NoError(w.Start())
	defer w.Stop()

	testFile := filepath.Join(tmpDir, "test.txt")
	s.Require().NoError(os.WriteFile(testFile, []byte("hello"), 0644))
	s.eventually(func() bool { return idx.indexed.Load() == 1 })

	s.Require().NoError(os.WriteFile(testFile, []byte("world"), 0644))
	s.eventually(func() bool { return idx.indexed.Load() == 2 })

	s.Require().NoError(os.Remove(testFile))
	s.eventually(func() bool { return idx.deleted.Load() == 1 })
}

func (s *WatcherSuite) TestBurstOfWritesCoalesces() {
	tmpDir := s.T().TempDir()
	idx := newCountingIndexer(s.T())

	w := newTestWatcher(idx, testConfig(tmpDir))
	w.debounce = 150 * time.Millisecond
	s.Require().NoError(w.Start())
	defer w.Stop()

	testFile := filepath.Join(tmpDir, "download.bin")
	f, err := os.Create(testFile)
	s.Require().NoError(err)
	for range 50 {
		_, err := f.Write([]byte("chunk"))
		s.Require().NoError(err)
		time.Sleep(2 * time.Millisecond)
	}
	s.Require().NoError(f.Close())

	s.eventually(func() bool { return idx.indexed.Load() >= 1 })
	time.Sleep(300 * time.Millisecond)
	// a scheduler stall longer than the debounce can split the burst in two
	s.LessOrEqual(idx.indexed.Load(), int64(2))
}

func (s *WatcherSuite) TestRemoveCancelsPendingIndex() {
	tmpDir := s.T().TempDir()
	idx := newCountingIndexer(s.T())

	w := newTestWatcher(idx, testConfig(tmpDir))
	w.debounce = 200 * time.Millisecond
	s.Require().NoError(w.Start())
	defer w.Stop()

	testFile := filepath.Join(tmpDir, "temp.txt")
	s.Require().NoError(os.WriteFile(testFile, []byte("x"), 0644))
	s.Require().NoError(os.Remove(testFile))

	s.eventually(func() bool { return idx.deleted.Load() == 1 })
	time.Sleep(300 * time.Millisecond)
	s.Zero(idx.indexed.Load())
}

func (s *WatcherSuite) TestNewDirectoryIsWatchedAndIndexed() {
	tmpDir := s.T().TempDir()
	idx := newCountingIndexer(s.T())

	w := newTestWatcher(idx, testConfig(tmpDir))
	s.Require().NoError(w.Start())
	defer w.Stop()

	staging := filepath.Join(s.T().TempDir(), "tree")
	s.Require().NoError(os.MkdirAll(filepath.Join(staging, "sub"), 0755))
	s.Require().NoError(os.WriteFile(filepath.Join(staging, "a.txt"), []byte("a"), 0644))
	s.Require().NoError(os.WriteFile(filepath.Join(staging, "sub", "b.txt"), []byte("b"), 0644))

	moved := filepath.Join(tmpDir, "tree")
	s.Require().NoError(os.Rename(staging, moved))
	s.eventually(func() bool { return idx.indexed.Load() == 4 })

	s.Require().NoError(os.WriteFile(filepath.Join(moved, "sub", "c.txt"), []byte("c"), 0644))
	s.eventually(func() bool { return idx.indexed.Load() == 5 })
}

func (s *WatcherSuite) TestStopDropsPendingWork() {
	tmpDir := s.T().TempDir()
	idx := newCountingIndexer(s.T())

	w := newTestWatcher(idx, testConfig(tmpDir))
	w.debounce = 200 * time.Millisecond
	s.Require().NoError(w.Start())

	s.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "late.txt"), []byte("x"), 0644))
	s.eventually(func() bool {
		w.mu.Lock()
		defer w.mu.Unlock()
		return len(w.pending) == 1
	})
	s.Require().NoError(w.Stop())

	time.Sleep(300 * time.Millisecond)
	s.Zero(idx.indexed.Load())
	s.Require().NoError(w.Start())
	s.Require().NoError(w.Stop())
}

func (s *WatcherSuite) TestExcludedDirs() {
	tmpDir := s.T().TempDir()
	cfg := config.Default()
	cfg.IndexPaths = []config.IndexPath{
		{
			Path:          tmpDir,
			MaxDepth:      10,
			ExcludeHidden: false,
			ExcludeDirs:   []string{".git"},
		},
	}
	cfg.BuildMaps()
	idx := newCountingIndexer(s.T())

	excludedDir := filepath.Join(tmpDir, ".git")
	s.Require().NoError(os.Mkdir(excludedDir, 0755))

	w := newTestWatcher(idx, cfg)
	s.Require().NoError(w.Start())
	defer w.Stop()

	testFile := filepath.Join(excludedDir, "config.txt")
	s.Require().NoError(os.WriteFile(testFile, []byte("test"), 0644))
	time.Sleep(100 * time.Millisecond)

	s.Zero(idx.indexed.Load())
}
