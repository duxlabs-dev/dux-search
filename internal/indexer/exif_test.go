package indexer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/stretchr/testify/suite"
)

type ExifSuite struct {
	suite.Suite
	tmpDir string
	cfg    *config.Config
}

func TestExifSuite(t *testing.T) {
	suite.Run(t, new(ExifSuite))
}

func (s *ExifSuite) SetupTest() {
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

func (s *ExifSuite) newIndexer() *Indexer {
	idx, err := New(s.cfg)
	s.Require().NoError(err)
	s.T().Cleanup(func() { idx.Close() })
	return idx
}

func (s *ExifSuite) TestExifExtraction() {
	idx := s.newIndexer()

	testJpegData := []byte{
		0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x18, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00,
		0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xD9,
	}
	testFile := filepath.Join(s.tmpDir, "test.jpg")
	s.Require().NoError(os.WriteFile(testFile, testJpegData, 0644))

	info, err := os.Stat(testFile)
	s.Require().NoError(err)

	doc := idx.readDocument(testFile, info)
	s.Equal("image/jpeg", doc.ContentType)
}

func (s *ExifSuite) TestIsImageFile() {
	tests := []struct {
		contentType string
		want        bool
	}{
		{"image/jpeg", true},
		{"image/png", true},
		{"image/gif", true},
		{"text/plain", false},
		{"application/pdf", false},
	}

	for _, tt := range tests {
		s.Equal(tt.want, isImageFile(tt.contentType), "isImageFile(%s)", tt.contentType)
	}
}

func (s *ExifSuite) TestSearchWithExifFilters() {
	idx := s.newIndexer()

	testFile := filepath.Join(s.tmpDir, "test.txt")
	s.Require().NoError(os.WriteFile(testFile, []byte("test content"), 0644))
	s.Require().NoError(idx.Index(testFile))

	opts := &SearchOptions{
		Query:     "*",
		Limit:     10,
		ExifMake:  "Canon",
		ExifModel: "EOS 5D",
	}

	_, err := idx.SearchWithOptions(opts)
	s.NoError(err)
}

func (s *ExifSuite) TestSearchWithFolderFilter() {
	subDir := filepath.Join(s.tmpDir, "subdir")
	s.Require().NoError(os.Mkdir(subDir, 0755))

	idx := s.newIndexer()

	testFile1 := filepath.Join(s.tmpDir, "root.txt")
	s.Require().NoError(os.WriteFile(testFile1, []byte("root file"), 0644))

	testFile2 := filepath.Join(subDir, "sub.txt")
	s.Require().NoError(os.WriteFile(testFile2, []byte("sub file"), 0644))

	s.Require().NoError(idx.Index(testFile1))
	s.Require().NoError(idx.Index(testFile2))

	opts := &SearchOptions{
		Query:  "*",
		Limit:  10,
		Folder: subDir,
	}

	result, err := idx.SearchWithOptions(opts)
	s.Require().NoError(err)
	s.NotZero(result.Total)

	found := false
	for _, hit := range result.Hits {
		if hit.ID == testFile2 {
			found = true
		}
		s.NotEqual(testFile1, hit.ID, "should not find root.txt when filtering by subdir")
	}
	s.True(found, "expected to find sub.txt")
}
