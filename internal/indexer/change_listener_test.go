package indexer

import (
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type changeEvent struct {
	path    string
	deleted bool
}

type changeLog struct {
	mu     sync.Mutex
	events []changeEvent
}

func (c *changeLog) record(path string, deleted bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, changeEvent{path, deleted})
}

// take returns and clears the events seen so far.
func (c *changeLog) take() []changeEvent {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := c.events
	c.events = nil
	sort.Slice(out, func(i, j int) bool {
		if out[i].path != out[j].path {
			return out[i].path < out[j].path
		}
		return !out[i].deleted && out[j].deleted
	})
	return out
}

func (s *LifecycleSuite) TestChangeListenerSeesIndexAndDelete() {
	idx := s.newIndexer()
	var log changeLog
	idx.SetChangeListener(log.record)

	a := s.write("a.txt", "alpha")
	s.Require().NoError(idx.Index(a))
	s.Equal([]changeEvent{{a, false}}, log.take())

	s.Require().NoError(idx.Index(a))
	s.Empty(log.take(), "an unchanged file raises no event")

	s.Require().NoError(os.WriteFile(a, []byte("alpha, edited and longer"), 0o644))
	future := time.Now().Add(time.Minute)
	s.Require().NoError(os.Chtimes(a, future, future))
	s.Require().NoError(idx.Index(a))
	s.Equal([]changeEvent{{a, false}}, log.take())

	dir := filepath.Join(s.tmpDir, "photos")
	inside := s.write("photos/b.txt", "beta")
	nested := s.write("photos/2024/c.txt", "gamma")
	s.Require().NoError(idx.Index(dir))
	s.Empty(log.take(), "a directory is not a file event")
	s.Require().NoError(idx.Index(inside))
	s.Require().NoError(idx.Index(nested))
	log.take()

	s.Require().NoError(os.RemoveAll(dir))
	s.Require().NoError(idx.Delete(dir))
	got := log.take()
	var paths []string
	for _, e := range got {
		s.True(e.deleted, e.path)
		paths = append(paths, e.path)
	}
	s.Contains(paths, inside)
	s.Contains(paths, nested)
	s.Contains(paths, dir)
	s.NotContains(paths, a)

	idx.SetChangeListener(nil)
	s.Require().NoError(idx.Delete(a))
	s.Empty(log.take())
}

func (s *LifecycleSuite) TestChangeListenerOnFullReindex() {
	idx := s.newIndexer()
	keep := s.write("keep.txt", "stays")
	vanish := s.write("vanish.txt", "goes away")
	nested := s.write("sub/deep.go", "package deep")
	s.Require().NoError(idx.ReindexAll())

	var log changeLog
	idx.SetChangeListener(log.record)
	s.Require().NoError(os.Remove(vanish))
	added := s.write("added.txt", "new")
	s.Require().NoError(idx.ReindexAll())

	want := []changeEvent{{added, false}, {keep, false}, {nested, false}, {vanish, true}}
	sort.Slice(want, func(i, j int) bool { return want[i].path < want[j].path })
	s.Equal(want, log.take(), "every file is reported again and the vanished one as deleted")
}

func (s *LifecycleSuite) TestChangeListenerOnIncrementalSync() {
	idx := s.newIndexer()
	unchanged := s.write("same.txt", "same")
	edited := s.write("edit.txt", "before")
	removed := s.write("gone.txt", "bye")
	s.Require().NoError(idx.ReindexAll())

	var log changeLog
	idx.SetChangeListener(log.record)
	s.Require().NoError(os.WriteFile(edited, []byte("after the edit"), 0o644))
	future := time.Now().Add(time.Minute)
	s.Require().NoError(os.Chtimes(edited, future, future))
	s.Require().NoError(os.Remove(removed))
	fresh := s.write("fresh.txt", "hello")
	s.Require().NoError(idx.SyncIncremental())

	want := []changeEvent{{edited, false}, {fresh, false}, {removed, true}}
	sort.Slice(want, func(i, j int) bool { return want[i].path < want[j].path })
	got := log.take()
	s.Equal(want, got)
	for _, e := range got {
		s.NotEqual(unchanged, e.path)
	}
}

func (s *LifecycleSuite) TestPanickingListenerDoesNotBreakIndexing() {
	idx := s.newIndexer()
	idx.SetChangeListener(func(string, bool) { panic("listener bug") })
	a := s.write("a.txt", "searchable words")
	s.Require().NoError(idx.Index(a))
	s.Require().NoError(idx.ReindexAll())
	res, err := idx.Search("searchable", 5)
	s.Require().NoError(err)
	s.Equal(uint64(1), res.Total)
}
