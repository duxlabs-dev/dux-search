package preprocess

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Helpers for the reference tests: testdata/ref is produced by
// testdata/gen_reference.py with the Hugging Face transformers processors.

func testdataPath(parts ...string) string {
	return filepath.Join(append([]string{"testdata"}, parts...)...)
}

func readGz(t testing.TB, name string) []byte {
	t.Helper()
	f, err := os.Open(testdataPath("ref", name))
	if err != nil {
		t.Fatalf("reference data missing (run testdata/gen_reference.py): %v", err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func readJSON(t testing.TB, name string, v any) {
	t.Helper()
	b, err := os.ReadFile(testdataPath("ref", name))
	if err != nil {
		t.Fatalf("reference data missing (run testdata/gen_reference.py): %v", err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func i16s(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return out
}

var _ = bytes.NewReader
