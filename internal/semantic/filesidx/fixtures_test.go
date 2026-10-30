package filesidx

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type queryCase struct {
	Collection string   `json:"collection"`
	Query      string   `json:"query"`
	Want       []string `json:"want"` // file names, any of which may rank first
}

func loadQueries(t testing.TB) []queryCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "queries.json"))
	if err != nil {
		t.Fatal(err)
	}
	var qs []queryCase
	if err := json.Unmarshal(raw, &qs); err != nil {
		t.Fatal(err)
	}
	return qs
}

// TestRecordFixtureInputs writes the text inputs of every fixture file, exactly
// as the indexer would embed them, for testdata/gen_vectors.py. It only runs
// when DUX_FIXTURE_MISSING names the output file (see gen_vectors.py).
func TestRecordFixtureInputs(t *testing.T) {
	out := os.Getenv("DUX_FIXTURE_MISSING")
	if out == "" {
		t.Skip("set DUX_FIXTURE_MISSING to record the fixture inputs")
	}
	cfg := DefaultConfig("").withDefaults()
	cfg.LegacyEncoding = legacyEncodingFor("tr_TR.UTF-8")
	record := func(title, path string) {
		text, _, err := readText(path, cfg.MaxTextBytes, cfg.LegacyEncoding)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		chunks, _ := splitText(text, cfg.MaxChunksPerFile)
		for _, c := range chunks {
			recordMissing("text", map[string]string{"title": title, "text": c})
		}
	}
	files, _ := filepath.Glob(filepath.Join("testdata", "files", "*"))
	for _, f := range files {
		if cfg.IsText(f) {
			record(filepath.Base(f), f)
		}
	}
	// variants/<working name>/<version>: the same title for every version.
	variants, _ := filepath.Glob(filepath.Join("testdata", "variants", "*", "*"))
	for _, v := range variants {
		record(filepath.Base(filepath.Dir(v)), v)
	}
}
