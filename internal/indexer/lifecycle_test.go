package indexer

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/blevesearch/bleve/v2/index/scorch"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/stretchr/testify/suite"
)

type LifecycleSuite struct {
	suite.Suite
	tmpDir string
	cfg    *config.Config
}

func TestLifecycleSuite(t *testing.T) {
	suite.Run(t, new(LifecycleSuite))
}

func (s *LifecycleSuite) SetupTest() {
	s.tmpDir = s.T().TempDir()
	s.cfg = config.Default()
	s.cfg.IndexPath = filepath.Join(s.tmpDir, "index")
	s.cfg.IndexPaths = []config.IndexPath{{Path: s.tmpDir, MaxDepth: 10}}
	s.cfg.WorkerCount = 2
	s.cfg.BuildMaps()
}

func (s *LifecycleSuite) newIndexer() *Indexer {
	idx, err := New(s.cfg)
	s.Require().NoError(err)
	s.T().Cleanup(func() { idx.Close() })
	return idx
}

func (s *LifecycleSuite) write(rel, content string) string {
	path := filepath.Join(s.tmpDir, rel)
	s.Require().NoError(os.MkdirAll(filepath.Dir(path), 0755))
	s.Require().NoError(os.WriteFile(path, []byte(content), 0644))
	return path
}

func (s *LifecycleSuite) TestDeleteRemovesSubtree() {
	idx := s.newIndexer()
	dir := filepath.Join(s.tmpDir, "photos")
	inside := s.write("photos/a.txt", "alpha")
	nested := s.write("photos/2024/b.txt", "beta")
	sibling := s.write("photos-archive/c.txt", "gamma")
	s.Require().NoError(idx.ReindexAll())

	s.Require().NoError(os.RemoveAll(dir))
	s.Require().NoError(idx.Delete(dir))

	for _, gone := range []string{dir, inside, nested, filepath.Join(dir, "2024")} {
		_, found, err := idx.meta.Get(gone)
		s.Require().NoError(err)
		s.False(found, gone)
	}
	_, found, err := idx.meta.Get(sibling)
	s.Require().NoError(err)
	s.True(found)

	result, err := idx.Search("gamma", 5)
	s.Require().NoError(err)
	s.Equal(uint64(1), result.Total)
	result, err = idx.Search("alpha", 5)
	s.Require().NoError(err)
	s.Zero(result.Total)
}

func (s *LifecycleSuite) TestIndexSkipsUnchangedFile() {
	idx := s.newIndexer()
	path := s.write("doc.txt", "first")
	s.Require().NoError(idx.Index(path))

	s.Require().NoError(os.Chtimes(path, time.Unix(1_700_000_000, 0), time.Unix(1_700_000_000, 0)))
	s.Require().NoError(idx.Index(path))
	s.Require().NoError(os.WriteFile(path, []byte("second"), 0644))
	s.Require().NoError(os.Chtimes(path, time.Unix(1_700_000_000, 0), time.Unix(1_700_000_000, 0)))
	s.Require().NoError(idx.Index(path))

	result, err := idx.Search("second", 5)
	s.Require().NoError(err)
	s.Equal(uint64(1), result.Total, "size change must trigger reindex even with equal mtime")

	s.Require().NoError(os.WriteFile(path, []byte("third!"), 0644))
	s.Require().NoError(os.Chtimes(path, time.Unix(1_700_000_000, 0), time.Unix(1_700_000_000, 0)))
	s.Require().NoError(idx.Index(path))
	result, err = idx.Search("third", 5)
	s.Require().NoError(err)
	s.Zero(result.Total, "same mtime and size is treated as unchanged")
}

func (s *LifecycleSuite) TestSyncIncrementalTracksChanges() {
	idx := s.newIndexer()
	keep := s.write("keep.txt", "keep")
	drop := s.write("drop.txt", "drop")
	s.Require().NoError(idx.ReindexAll())

	s.Require().NoError(os.Remove(drop))
	added := s.write("nested/added.txt", "added")
	s.Require().NoError(os.WriteFile(keep, []byte("keep changed"), 0644))
	s.Require().NoError(os.Chtimes(keep, time.Now().Add(time.Hour), time.Now().Add(time.Hour)))
	s.Require().NoError(idx.SyncIncremental())

	_, found, err := idx.meta.Get(drop)
	s.Require().NoError(err)
	s.False(found)
	_, found, err = idx.meta.Get(added)
	s.Require().NoError(err)
	s.True(found)

	for query, want := range map[string]uint64{"drop": 0, "added": 1, "changed": 1} {
		result, err := idx.Search(query, 5)
		s.Require().NoError(err)
		s.Equal(want, result.Total, query)
	}
}

func (s *LifecycleSuite) TestConcurrentOperationsAreRejected() {
	idx := s.newIndexer()
	s.write("a.txt", "a")

	idx.opMu.Lock()
	s.ErrorIs(idx.ReindexAll(), ErrBusy)
	s.ErrorIs(idx.SyncIncremental(), ErrBusy)
	idx.opMu.Unlock()

	s.Require().NoError(idx.ReindexAll())
	s.False(idx.Busy())
}

func (s *LifecycleSuite) TestReindexSurvivesVanishingDirectory() {
	idx := s.newIndexer()
	s.write("stable/a.txt", "a")
	vanishing := filepath.Join(s.tmpDir, "vanishing")
	s.Require().NoError(os.Mkdir(vanishing, 0755))
	s.Require().NoError(os.Symlink(filepath.Join(s.tmpDir, "missing-target"), filepath.Join(s.tmpDir, "dangling")))
	s.Require().NoError(os.Chmod(vanishing, 0))
	s.T().Cleanup(func() { os.Chmod(vanishing, 0755) })

	s.Require().NoError(idx.ReindexAll())

	result, err := idx.Search("a.txt", 5)
	s.Require().NoError(err)
	s.Equal(uint64(1), result.Total)
}

func (s *LifecycleSuite) TestExtractExifHonoursIndexPathSetting() {
	off := false
	s.cfg.IndexPaths[0].ExtractExif = &off
	s.cfg.BuildMaps()
	idx := s.newIndexer()

	s.False(idx.config.ExtractExif(filepath.Join(s.tmpDir, "x.jpg")))
	s.cfg.IndexPaths[0].ExtractExif = nil
	s.True(idx.config.ExtractExif(filepath.Join(s.tmpDir, "x.jpg")))
}

func (s *LifecycleSuite) TestMatchAllReturnsOnlyRealEntries() {
	idx := s.newIndexer()
	s.write("a.txt", "a")
	s.write("dir/b.txt", "b")
	s.Require().NoError(idx.ReindexAll())

	files, err := idx.SearchWithOptions(&SearchOptions{Query: "*", Limit: 100})
	s.Require().NoError(err)
	s.Equal(uint64(2), files.Total)

	all, err := idx.SearchWithOptions(&SearchOptions{Query: "*", Limit: 100, Type: "all"})
	s.Require().NoError(err)
	s.Equal(uint64(4), all.Total, "2 files, the dir and the root")

	stats := idx.Stats()
	s.Equal(2, stats.TotalFiles)
	s.False(stats.LastIndexTime.IsZero())
	s.NotEmpty(stats.IndexDuration)
}

func (s *LifecycleSuite) TestBackgroundFailureSchedulesRebuild() {
	idx := s.newIndexer()
	s.write("a.txt", "a")
	s.Require().NoError(idx.ReindexAll())
	s.False(idx.NeedsReindex())

	scorch.RegistryAsyncErrorCallbacks[asyncErrorHandler](errors.New("merger panic"), s.cfg.IndexPath)
	scorch.RegistryAsyncErrorCallbacks[asyncErrorHandler](errors.New("second"), s.cfg.IndexPath)

	select {
	case err := <-idx.Fatal():
		s.ErrorContains(err, "merger panic")
	case <-time.After(time.Second):
		s.Fail("fatal error was not reported")
	}
	s.True(idx.NeedsReindex())
	reason, err := idx.RebuildReason()
	s.Require().NoError(err)
	s.Equal("merger panic", reason)

	s.Require().NoError(idx.ReindexAll())
	s.False(idx.NeedsReindex())
}

func (s *LifecycleSuite) TestClosedIndexerIgnoresBackgroundErrors() {
	idx, err := New(s.cfg)
	s.Require().NoError(err)
	s.Require().NoError(idx.Close())

	scorch.RegistryAsyncErrorCallbacks[asyncErrorHandler](errors.New("late"), s.cfg.IndexPath)
	select {
	case <-idx.Fatal():
		s.Fail("closed indexer should not receive errors")
	default:
	}
}
