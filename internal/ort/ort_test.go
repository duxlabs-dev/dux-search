package ort

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/embedtest"
)

func loadOrSkip(t *testing.T) *Runtime {
	t.Helper()
	rt, err := Load(embedtest.ORTLibrary())
	if errors.Is(err, ErrLibraryNotFound) {
		t.Skipf("onnxruntime not installed: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(rt.Close)
	return rt
}

func TestLoadExplicitMissing(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "nope.so"))
	if !errors.Is(err, ErrLibraryNotFound) {
		t.Fatalf("want ErrLibraryNotFound, got %v", err)
	}
}

func TestLoadNotAnORTLibrary(t *testing.T) {
	// A real shared library that is not onnxruntime must give a clear error.
	libc := "/usr/lib/libc.so.6"
	if _, err := os.Stat(libc); err != nil {
		t.Skip("no libc.so.6 at the usual place")
	}
	if _, err := Load(libc); err == nil {
		t.Fatal("expected an error for a library without OrtGetApiBase")
	}
}

func TestVersionAndErrors(t *testing.T) {
	rt := loadOrSkip(t)
	if rt.Version() == "" || rt.Path() == "" {
		t.Fatalf("version %q path %q", rt.Version(), rt.Path())
	}
	t.Logf("onnxruntime %s at %s", rt.Version(), rt.Path())
	// A file that exists but is not a model: the runtime's own message must come back.
	bad := filepath.Join(t.TempDir(), "bad.onnx")
	if err := os.WriteFile(bad, []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := rt.NewSession(bad, Options{})
	var oe *Error
	if !errors.As(err, &oe) || oe.Msg == "" || oe.Code == 0 {
		t.Fatalf("want *Error with message and code, got %#v", err)
	}
	if _, err := rt.NewSession(filepath.Join(t.TempDir(), "missing.onnx"), Options{}); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("want not-exist, got %v", err)
	}
}

func TestTensors(t *testing.T) {
	rt := loadOrSkip(t)
	f, err := rt.NewFloat32([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	data, shape, err := f.Float32()
	if err != nil || len(shape) != 2 || shape[0] != 2 || shape[1] != 3 || data[5] != 6 {
		t.Fatalf("float32 round trip: %v %v %v", data, shape, err)
	}
	empty, err := rt.NewFloat32([]int64{0, 512}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	ed, es, err := empty.Float32()
	if err != nil || len(ed) != 0 || es[0] != 0 || es[1] != 512 {
		t.Fatalf("empty tensor: %v %v %v", ed, es, err)
	}
	if _, err := rt.NewFloat32([]int64{2, 2}, []float32{1}); err == nil {
		t.Fatal("expected a length mismatch error")
	}
	if _, err := rt.NewInt64([]int64{-1}, nil); err == nil {
		t.Fatal("expected a negative dimension error")
	}
	i, err := rt.NewInt64([]int64{3}, []int64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer i.Close()
	if _, _, err := i.Float32(); err == nil {
		t.Fatal("reading int64 as float32 must fail")
	}
	b, err := rt.NewBool([]int64{2}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	b.Close()
	b.Close() // idempotent
}

// TestTextModel runs the real text tower end to end on one hand-made input
// and checks the structural properties the embedding must have.
func TestTextModel(t *testing.T) {
	rt := loadOrSkip(t)
	dir := embedtest.ModelDir(t)
	s, err := rt.NewSession(filepath.Join(dir, "onnx", "model_quantized.onnx"), Options{IntraOpThreads: 2, DisableCPUMemArena: true, Config: map[string]string{"session.use_env_allocators": "0"}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	wantIn := []string{"input_ids", "attention_mask", "image_features", "video_features", "audio_features"}
	got := s.InputNames()
	if len(got) != len(wantIn) {
		t.Fatalf("inputs %v", got)
	}
	for i := range wantIn {
		if got[i] != wantIn[i] {
			t.Fatalf("inputs %v", got)
		}
	}
	mk := func(shape []int64, data []int64) *Tensor {
		x, err := rt.NewInt64(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	empty := func() *Tensor {
		x, err := rt.NewFloat32([]int64{0, 512}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	ids := mk([]int64{1, 4}, []int64{2, 2364, 3000, 1})
	am := mk([]int64{1, 4}, []int64{1, 1, 1, 1})
	ins := []Input{{"input_ids", ids}, {"attention_mask", am}, {"image_features", empty()}, {"video_features", empty()}, {"audio_features", empty()}}
	defer func() {
		for _, in := range ins {
			in.Tensor.Close()
		}
	}()
	out, err := s.Run(ins, []string{"sentence_embedding"})
	if err != nil {
		t.Fatal(err)
	}
	defer out[0].Close()
	v, shape, err := out[0].Float32()
	if err != nil || len(shape) != 2 || shape[0] != 1 || shape[1] != 768 {
		t.Fatalf("output %v %v", shape, err)
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1) > 1e-3 {
		t.Fatalf("embedding not L2-normalized: |v|^2 = %v", sum)
	}
	if _, err := s.Run(ins, []string{"no_such_output"}); err == nil {
		t.Fatal("unknown output must fail with the runtime's message")
	}
}
