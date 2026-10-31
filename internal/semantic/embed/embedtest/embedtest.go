// Package embedtest locates the real model and runtime for tests. Tests that
// need them are skipped, with the reason, when they are not installed; nothing
// is faked.
package embedtest

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/wire"
)

// ModelDir returns the model directory or skips the test.
func ModelDir(t testing.TB, towers ...string) string {
	t.Helper()
	// Default: where `dsearch model download` puts the model.
	dir := os.Getenv("DSEARCH_TEST_MODEL_DIR")
	if dir == "" {
		dir = config.DefaultSemantic().ModelDir
	}
	need := []string{"tokenizer.model", "config.json", filepath.Join("onnx", "model_quantized.onnx")}
	for _, tw := range towers {
		need = append(need, filepath.Join("onnx", tw+"_quantized.onnx"))
	}
	for _, f := range need {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Skipf("model not available (set DSEARCH_TEST_MODEL_DIR): %v", err)
		}
	}
	return dir
}

// ORTLibrary returns DSEARCH_TEST_ORT_LIB or "" (standard locations).
func ORTLibrary() string { return os.Getenv("DSEARCH_TEST_ORT_LIB") }

// Ref is a reference container written by testdata/gen_reference.py.
type Ref struct {
	Header map[string]any
	arrays map[string]any
}

// LoadRef reads a .dref file (gzip: u32le header length, JSON header naming
// every array, then the arrays, little endian).
func LoadRef(t testing.TB, path string) *Ref {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) < 4 {
		t.Fatalf("%s: truncated", path)
	}
	hl := int(binary.LittleEndian.Uint32(raw))
	var hdr struct {
		Arrays []struct {
			Name  string `json:"name"`
			DType string `json:"dtype"`
			Count int    `json:"count"`
		} `json:"arrays"`
	}
	ref := &Ref{arrays: map[string]any{}}
	if err := json.Unmarshal(raw[4:4+hl], &ref.Header); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw[4:4+hl], &hdr); err != nil {
		t.Fatal(err)
	}
	r := wire.NewReader(raw[4+hl:])
	for _, a := range hdr.Arrays {
		switch a.DType {
		case "float32":
			ref.arrays[a.Name] = r.Float32(a.Count)
		case "int64":
			ref.arrays[a.Name] = r.Int64(a.Count)
		case "uint8":
			ref.arrays[a.Name] = r.Bool(a.Count)
		default:
			t.Fatalf("%s: unsupported dtype %s", path, a.DType)
		}
	}
	if err := r.Done(); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return ref
}

// Float32, Int64 and Bool return a named array.
func (r *Ref) Float32(name string) []float32 { return r.arrays[name].([]float32) }
func (r *Ref) Int64(name string) []int64     { return r.arrays[name].([]int64) }
func (r *Ref) Bool(name string) []bool       { return r.arrays[name].([]bool) }

// Int returns an integer header field.
func (r *Ref) Int(name string) int { return int(r.Header[name].(float64)) }

// Cosine is the cosine similarity of two vectors.
func Cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

var (
	helperOnce sync.Once
	helperPath string
	helperErr  error
)

// HelperBinary builds cmd/dsearch-embed with CGO_ENABLED=0 once per test
// process and returns its path.
func HelperBinary(t testing.TB) string {
	t.Helper()
	helperOnce.Do(func() {
		out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}").Output()
		if err != nil {
			helperErr = fmt.Errorf("go list: %w", err)
			return
		}
		root := strings.TrimSpace(string(out))
		dir, err := os.MkdirTemp("", "dsearch-embed-test")
		if err != nil {
			helperErr = err
			return
		}
		helperPath = filepath.Join(dir, "dsearch-embed")
		cmd := exec.Command("go", "build", "-o", helperPath, "./cmd/dsearch-embed")
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
		if b, err := cmd.CombinedOutput(); err != nil {
			helperErr = fmt.Errorf("build helper: %v\n%s", err, b)
		}
	})
	if helperErr != nil {
		t.Fatal(helperErr)
	}
	return helperPath
}

// CleanupHelper removes the built helper; call it from TestMain.
func CleanupHelper() {
	if helperPath != "" {
		os.RemoveAll(filepath.Dir(helperPath))
	}
}
