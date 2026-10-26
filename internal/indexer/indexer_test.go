package indexer

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/stretchr/testify/suite"
)

type IndexerSuite struct {
	suite.Suite
	tmpDir string
	cfg    *config.Config
}

func TestIndexerSuite(t *testing.T) {
	suite.Run(t, new(IndexerSuite))
}

func (s *IndexerSuite) SetupTest() {
	s.tmpDir = s.T().TempDir()
	s.cfg = config.Default()
	s.cfg.IndexPath = filepath.Join(s.tmpDir, "index")
	s.cfg.IndexPaths = []config.IndexPath{
		{
			Path:          s.tmpDir,
			MaxDepth:      10,
			ExcludeHidden: false,
			ExcludeDirs:   []string{},
		},
	}
	s.cfg.BuildMaps()
}

func (s *IndexerSuite) newIndexer() *Indexer {
	idx, err := New(s.cfg)
	s.Require().NoError(err)
	s.T().Cleanup(func() { idx.Close() })
	return idx
}

func (s *IndexerSuite) writeFile(name, content string) string {
	path := filepath.Join(s.tmpDir, name)
	s.Require().NoError(os.WriteFile(path, []byte(content), 0644))
	return path
}

func (s *IndexerSuite) TestNew() {
	idx := s.newIndexer()
	s.NotNil(idx.index)
	s.Equal(s.cfg, idx.config)
}

func (s *IndexerSuite) TestIndex() {
	idx := s.newIndexer()
	testFile := s.writeFile("test.txt", "hello world")

	s.NoError(idx.Index(testFile))

	result, err := idx.Search("hello", 10)
	s.Require().NoError(err)
	s.NotZero(result.Total)
}

func (s *IndexerSuite) TestDelete() {
	idx := s.newIndexer()
	testFile := s.writeFile("test.txt", "hello world")

	s.Require().NoError(idx.Index(testFile))
	s.NoError(idx.Delete(testFile))

	result, err := idx.Search("hello", 10)
	s.Require().NoError(err)
	s.Zero(result.Total)
}

func (s *IndexerSuite) TestSearch() {
	idx := s.newIndexer()
	s.writeFile("test1.txt", "golang programming")
	s.writeFile("test2.txt", "python scripting")

	idx.Index(filepath.Join(s.tmpDir, "test1.txt"))
	idx.Index(filepath.Join(s.tmpDir, "test2.txt"))

	result, err := idx.Search("golang", 10)
	s.Require().NoError(err)
	s.NotZero(result.Total)

	result, err = idx.Search("python", 10)
	s.Require().NoError(err)
	s.NotZero(result.Total)
}

func (s *IndexerSuite) TestStats() {
	idx := s.newIndexer()
	s.NotNil(idx.Stats())
}

func (s *IndexerSuite) TestStatsReportsSchemaAndPhase() {
	idx := s.newIndexer()
	stats := idx.Stats()
	s.Equal(SchemaVersion, stats.ExpectedSchemaVersion)
	s.Equal(PhaseIdle, stats.Phase)

	s.Require().NoError(idx.SaveSchemaVersion())
	stats = idx.Stats()
	s.Equal(SchemaVersion, stats.SchemaVersion)
}

func (s *IndexerSuite) TestPhaseTracksReindex() {
	s.writeFile("a.txt", "x")
	s.writeFile("b.txt", "y")
	idx := s.newIndexer()
	s.Require().NoError(idx.ReindexAll())

	phase, _ := idx.Phase()
	s.Equal(PhaseIdle, phase)
	files, _ := idx.Progress()
	s.GreaterOrEqual(files, int64(2))
}

func (s *IndexerSuite) TestBatchedReindexProducesSearchableDocs() {
	for i := range 50 {
		s.writeFile(fmt.Sprintf("doc_%03d.txt", i), fmt.Sprintf("content-%d unique-token-%d", i, i))
	}
	idx := s.newIndexer()
	s.Require().NoError(idx.ReindexAll())

	for _, q := range []string{"doc_017", "doc_042", "unique-token-9"} {
		result, err := idx.Search(q, 5)
		s.Require().NoError(err)
		s.NotZero(result.Total, "expected hits for %q", q)
	}
}

func (s *IndexerSuite) TestRemoveIndexDir() {
	idx, err := New(s.cfg)
	s.Require().NoError(err)
	idx.Close()

	s.NoError(removeIndexDir(s.cfg.IndexPath))
	_, statErr := os.Stat(s.cfg.IndexPath)
	s.True(os.IsNotExist(statErr))
}

func (s *IndexerSuite) TestRemoveIndexDirRefusesNonIndex() {
	plainDir := filepath.Join(s.tmpDir, "not-an-index")
	s.Require().NoError(os.MkdirAll(plainDir, 0755))
	s.Require().NoError(os.WriteFile(filepath.Join(plainDir, "important.txt"), []byte("data"), 0644))

	err := removeIndexDir(plainDir)
	s.Error(err)
	s.Contains(err.Error(), "not a bleve index directory")

	_, statErr := os.Stat(filepath.Join(plainDir, "important.txt"))
	s.NoError(statErr)
}

func (s *IndexerSuite) TestRemoveIndexDirNonExistent() {
	s.NoError(removeIndexDir(filepath.Join(s.tmpDir, "does-not-exist")))
}

func (s *IndexerSuite) TestRemoveIndexDirEmpty() {
	s.Error(removeIndexDir(""))
}

func (s *IndexerSuite) TestFilenameWordSeparators() {
	idx := s.newIndexer()

	target := s.writeFile("My.Show.S01E02.1080p.WEB-DL.mkv", "")
	s.writeFile("unrelated_video.mp4", "")
	s.writeFile("notes.txt", "")
	s.writeFile("report_2024_q4.pdf", "")
	s.writeFile("vacation-photos-summer.zip", "")

	for _, name := range []string{
		"My.Show.S01E02.1080p.WEB-DL.mkv",
		"unrelated_video.mp4",
		"notes.txt",
		"report_2024_q4.pdf",
		"vacation-photos-summer.zip",
	} {
		s.Require().NoError(idx.Index(filepath.Join(s.tmpDir, name)))
	}

	cases := []struct {
		query string
		want  string
	}{
		{"show", target},
		{"s01e02", target},
		{"web", target},
		{"1080p", target},
		{"q4", filepath.Join(s.tmpDir, "report_2024_q4.pdf")},
		{"2024", filepath.Join(s.tmpDir, "report_2024_q4.pdf")},
		{"summer", filepath.Join(s.tmpDir, "vacation-photos-summer.zip")},
		{"photos", filepath.Join(s.tmpDir, "vacation-photos-summer.zip")},
	}

	for _, tc := range cases {
		s.Run(tc.query, func() {
			result, err := idx.Search(tc.query, 5)
			s.Require().NoError(err)
			s.Require().NotZero(result.Total, "no hits for %q", tc.query)
			s.Equal(tc.want, result.Hits[0].ID, "top hit for %q should be %s", tc.query, tc.want)
		})
	}
}

func (s *IndexerSuite) TestFilenameQueryWithDots() {
	idx := s.newIndexer()
	target := s.writeFile("My.Show.S01E02.mkv", "")
	s.writeFile("other.mkv", "")
	s.Require().NoError(idx.Index(target))
	s.Require().NoError(idx.Index(filepath.Join(s.tmpDir, "other.mkv")))

	result, err := idx.Search("my.show", 5)
	s.Require().NoError(err)
	s.Require().NotZero(result.Total)
	s.Equal(target, result.Hits[0].ID)
}

func (s *IndexerSuite) TestReadDocument() {
	idx := s.newIndexer()
	content := "package main\n\nfunc main() {}\n"
	testFile := s.writeFile("test.go", content)

	info, err := os.Stat(testFile)
	s.Require().NoError(err)

	doc := idx.readDocument(testFile, info)
	s.Equal(testFile, doc.Path)
	s.Equal("test.go", doc.Filename)
	s.Equal(content, doc.Body)
	s.NotEmpty(doc.Hash)
}

func (s *IndexerSuite) TestWalkSkipsSymlinkToAncestor() {
	root := filepath.Join(s.tmpDir, "root")
	s.Require().NoError(os.MkdirAll(filepath.Join(root, "sub"), 0755))
	s.Require().NoError(os.MkdirAll(filepath.Join(s.tmpDir, "outside"), 0755))
	s.Require().NoError(os.WriteFile(filepath.Join(root, "a.txt"), []byte("a"), 0644))
	s.Require().NoError(os.WriteFile(filepath.Join(s.tmpDir, "outside", "b.txt"), []byte("b"), 0644))
	s.Require().NoError(os.Symlink(s.tmpDir, filepath.Join(root, "sub", "up")))
	s.Require().NoError(os.Symlink(filepath.Join(root, "sub"), filepath.Join(root, "sub", "self")))

	var seen []string
	walkFollowSymlinks(root, func(path string, info os.FileInfo, err error) error {
		s.NoError(err)
		rel, _ := filepath.Rel(root, path)
		seen = append(seen, rel)
		return nil
	})

	s.ElementsMatch([]string{".", "a.txt", "sub"}, seen)
}

func (s *IndexerSuite) TestBatcherFlushesOnBodyBytes() {
	idx := s.newIndexer()
	bat := newBatcher(idx, 500, 1<<20, time.Hour)

	for n := range 3 {
		path := filepath.Join(s.tmpDir, fmt.Sprintf("big%d.txt", n))
		bat.submit(batchJob{path: path, doc: &Document{Path: path, Body: strings.Repeat("x ", 300*1024)}})
	}

	s.Eventually(func() bool {
		count, err := idx.GetDocCount()
		return err == nil && count >= 2
	}, 3*time.Second, 50*time.Millisecond, "batch should flush once pending bodies exceed the byte budget")

	bat.close()
	count, err := idx.GetDocCount()
	s.Require().NoError(err)
	s.Equal(uint64(3), count)
}
