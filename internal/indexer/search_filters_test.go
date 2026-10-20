package indexer

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/stretchr/testify/suite"
)

type SearchFiltersSuite struct {
	suite.Suite
	tmpDir string
	idx    *Indexer
}

func TestSearchFiltersSuite(t *testing.T) {
	suite.Run(t, new(SearchFiltersSuite))
}

func (s *SearchFiltersSuite) SetupTest() {
	s.tmpDir = s.T().TempDir()
	cfg := config.Default()
	cfg.IndexPath = filepath.Join(s.tmpDir, "index")
	cfg.IndexPaths = []config.IndexPath{{Path: s.tmpDir, MaxDepth: 10}}
	cfg.BuildMaps()

	idx, err := New(cfg)
	s.Require().NoError(err)
	s.T().Cleanup(func() { idx.Close() })
	s.idx = idx
}

func (s *SearchFiltersSuite) indexPhoto(name string, taken time.Time, iso int) string {
	path := filepath.Join(s.tmpDir, name)
	s.Require().NoError(os.WriteFile(path, []byte("x"), 0644))
	doc := &Document{
		Path:         path,
		Filename:     name,
		ExifDateTime: taken,
		ExifISO:      iso,
		DocType:      "file",
	}
	s.Require().NoError(s.idx.currentIndex().Index(path, doc))
	s.idx.indexComplete.Store(true)
	return path
}

func (s *SearchFiltersSuite) ids(opts *SearchOptions) []string {
	result, err := s.idx.SearchWithOptions(opts)
	s.Require().NoError(err)
	ids := make([]string, 0, len(result.Hits))
	for _, hit := range result.Hits {
		ids = append(ids, hit.ID)
	}
	return ids
}

func (s *SearchFiltersSuite) TestExifDateRange() {
	old := s.indexPhoto("old.jpg", time.Date(2020, 3, 1, 12, 0, 0, 0, time.UTC), 100)
	mid := s.indexPhoto("mid.jpg", time.Date(2024, 6, 1, 12, 0, 0, 0, time.UTC), 400)
	recent := s.indexPhoto("new.jpg", time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC), 1600)

	s.ElementsMatch([]string{mid}, s.ids(&SearchOptions{
		Query:          "*",
		ExifDateAfter:  "2024-01-01T00:00:00Z",
		ExifDateBefore: "2024-12-31T23:59:59Z",
	}))
	s.ElementsMatch([]string{mid, recent}, s.ids(&SearchOptions{Query: "*", ExifDateAfter: "2024:01:01 00:00:00"}))
	s.ElementsMatch([]string{old}, s.ids(&SearchOptions{Query: "*", ExifDateBefore: "2021-01-01T00:00:00Z"}))
	s.ElementsMatch([]string{old, mid, recent}, s.ids(&SearchOptions{Query: "*", ExifDateAfter: "garbage"}))

	plain := s.indexPhoto("notes.txt", time.Time{}, 0)
	s.ElementsMatch([]string{old}, s.ids(&SearchOptions{Query: "*", ExifDateBefore: "2021-01-01T00:00:00Z"}))
	s.ElementsMatch([]string{old, mid, recent, plain}, s.ids(&SearchOptions{Query: "*"}))
}

func (s *SearchFiltersSuite) TestSortByExifDate() {
	old := s.indexPhoto("old.jpg", time.Date(2020, 3, 1, 12, 0, 0, 0, time.UTC), 100)
	mid := s.indexPhoto("mid.jpg", time.Date(2024, 6, 1, 12, 0, 0, 0, time.UTC), 400)
	recent := s.indexPhoto("new.jpg", time.Date(2025, 1, 1, 12, 0, 0, 0, time.UTC), 1600)

	s.Equal([]string{recent, mid, old}, s.ids(&SearchOptions{Query: "*", SortBy: "exif_date", SortDesc: true}))
	s.Equal([]string{old, mid, recent}, s.ids(&SearchOptions{Query: "*", SortBy: "iso"}))
}

func (s *SearchFiltersSuite) TestISORange() {
	s.indexPhoto("old.jpg", time.Time{}, 100)
	mid := s.indexPhoto("mid.jpg", time.Time{}, 400)
	recent := s.indexPhoto("new.jpg", time.Time{}, 1600)

	s.ElementsMatch([]string{mid, recent}, s.ids(&SearchOptions{Query: "*", ExifMinISO: 200}))
	s.ElementsMatch([]string{mid}, s.ids(&SearchOptions{Query: "*", ExifMinISO: 200, ExifMaxISO: 800}))
}

func (s *SearchFiltersSuite) TestCaseInsensitiveWildcardAndExtension() {
	upper := filepath.Join(s.tmpDir, "README.MD")
	lower := filepath.Join(s.tmpDir, "notes.md")
	for _, p := range []string{upper, lower} {
		s.Require().NoError(os.WriteFile(p, []byte("x"), 0644))
		s.Require().NoError(s.idx.Index(p))
	}

	s.ElementsMatch([]string{upper}, s.ids(&SearchOptions{Query: "README*"}))
	s.ElementsMatch([]string{upper}, s.ids(&SearchOptions{Query: "readme*"}))
	s.ElementsMatch([]string{upper, lower}, s.ids(&SearchOptions{Query: "*", Extension: ".md"}))
	s.ElementsMatch([]string{upper, lower}, s.ids(&SearchOptions{Query: "*", Extension: ".MD"}))
}

func (s *SearchFiltersSuite) TestSubstringRequiresWholeQuery() {
	target := filepath.Join(s.tmpDir, "dropbox-sync.log")
	noise := filepath.Join(s.tmpDir, "root.bolt")
	for _, p := range []string{target, noise} {
		s.Require().NoError(os.WriteFile(p, []byte("x"), 0644))
		s.Require().NoError(s.idx.Index(p))
	}

	s.ElementsMatch([]string{target}, s.ids(&SearchOptions{Query: "drop"}))
	s.ElementsMatch([]string{target}, s.ids(&SearchOptions{Query: "box-sy"}))
	s.ElementsMatch([]string{target, noise}, s.ids(&SearchOptions{Query: "drop", Fuzzy: true}))
}
