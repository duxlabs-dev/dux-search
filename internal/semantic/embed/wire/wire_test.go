package wire

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"slices"
	"strings"
	"testing"
)

func TestFrameRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	c := NewConn(&buf, &buf)
	req := Request{ID: 7, Op: OpText, Texts: []string{"karanlık", "emoji 😀", ""}}
	blob := AppendFloat32(nil, []float32{1.5, -2, float32(math.Inf(1))})
	blob = AppendInt64(blob, []int64{-1, 1 << 40})
	blob = AppendBool(blob, []bool{true, false, true})
	if err := c.Write(req, blob); err != nil {
		t.Fatal(err)
	}
	if err := c.Write(Response{ID: 7, OK: true}, nil); err != nil {
		t.Fatal(err)
	}
	var got Request
	gb, err := c.Read(&got)
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != 7 || got.Op != OpText || !slices.Equal(got.Texts, req.Texts) {
		t.Fatalf("header %+v", got)
	}
	r := NewReader(gb)
	f := r.Float32(3)
	i := r.Int64(2)
	b := r.Bool(3)
	if err := r.Done(); err != nil {
		t.Fatal(err)
	}
	if f[0] != 1.5 || f[1] != -2 || !math.IsInf(float64(f[2]), 1) || i[0] != -1 || i[1] != 1<<40 || !b[0] || b[1] || !b[2] {
		t.Fatalf("blob %v %v %v", f, i, b)
	}
	var resp Response
	if blob, err := c.Read(&resp); err != nil || len(blob) != 0 || !resp.OK {
		t.Fatalf("second frame: %v %v %+v", blob, err, resp)
	}
	if _, err := c.Read(&resp); err != io.EOF {
		t.Fatalf("want EOF at end of stream, got %v", err)
	}
}

func TestReaderErrors(t *testing.T) {
	r := NewReader(make([]byte, 7))
	if r.Float32(2) != nil || r.Err == nil {
		t.Fatal("short blob must fail")
	}
	// The first error sticks and later reads stay nil.
	if r.Bool(1) != nil {
		t.Fatal("reads after an error must return nil")
	}
	r = NewReader(make([]byte, 9))
	r.Float32(2)
	if r.Done() == nil {
		t.Fatal("trailing bytes must be reported")
	}
	if NewReader(nil).Float32(-1) != nil {
		t.Fatal("negative count must fail")
	}
}

func TestOversizeAndTruncatedFramesAreRejected(t *testing.T) {
	var pre [8]byte
	binary.LittleEndian.PutUint32(pre[0:], maxHeader+1)
	if _, err := NewConn(bytes.NewReader(pre[:]), io.Discard).Read(&Request{}); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("oversize header: %v", err)
	}
	binary.LittleEndian.PutUint32(pre[0:], 2)
	binary.LittleEndian.PutUint32(pre[4:], 10)
	if _, err := NewConn(bytes.NewReader(append(pre[:], '{', '}')), io.Discard).Read(&Request{}); err != io.ErrUnexpectedEOF {
		t.Fatalf("truncated blob: %v", err)
	}
	binary.LittleEndian.PutUint32(pre[4:], 0)
	if _, err := NewConn(bytes.NewReader(append(pre[:], 'x', 'y')), io.Discard).Read(&Request{}); err == nil || !strings.Contains(err.Error(), "bad header") {
		t.Fatalf("garbage header: %v", err)
	}
}
