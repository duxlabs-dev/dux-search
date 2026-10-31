// Package wire is the framing and message vocabulary spoken between the
// embedder client (dsearch) and the dsearch-embed helper over the helper's
// stdin and stdout. It is private to this repository: both ends are built
// from the same tree, so there is no version negotiation beyond Hello.
package wire

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
)

// Protocol is bumped when the message layout changes.
const Protocol = 1

// Operations.
const (
	OpHello  = "hello"  // handshake; response carries versions
	OpLoad   = "load"   // load towers (Towers)
	OpUnload = "unload" // unload towers (Towers)
	OpText   = "text"   // embed Texts (already prefixed)
	OpImage  = "image"  // embed Images
	OpAudio  = "audio"  // embed Clips
	OpStatus = "status" // loaded towers
	OpQuit   = "quit"   // exit after replying
	// OpCancel asks the helper to skip request Cancel if it has not started.
	// It has no reply.
	OpCancel = "cancel"
)

// Error codes in Response.Code.
const (
	CodeUnavailable = "unavailable" // model files or runtime library missing/unusable
	CodeBadRequest  = "bad_request"
	CodeInternal    = "internal"
	CodeCancelled   = "cancelled" // skipped because the client gave up
)

// Limits on one frame; a peer exceeding them is broken or hostile.
const (
	maxHeader = 64 << 20
	maxBlob   = 1 << 30
)

// Request is the JSON header of a client frame.
type Request struct {
	ID     uint64      `json:"id"`
	Op     string      `json:"op"`
	Towers []string    `json:"towers,omitempty"`
	Texts  []string    `json:"texts,omitempty"`
	Images []ImageMeta `json:"images,omitempty"`
	Clips  []AudioMeta `json:"clips,omitempty"`
	Cancel uint64      `json:"cancel,omitempty"` // OpCancel: id of the request to skip
}

// ImageMeta describes one image in the blob: Patches*768 float32 pixels then
// Patches*2 int64 positions.
type ImageMeta struct {
	Patches    int `json:"patches"`
	SoftTokens int `json:"soft_tokens"`
}

// AudioMeta describes one clip in the blob: Frames*128 float32 log-mel then
// Frames mask bytes (0 or 1).
type AudioMeta struct {
	Frames int `json:"frames"`
}

// Response is the JSON header of a helper frame. Vectors of an embedding
// reply are N*Dim float32 in the blob.
type Response struct {
	ID      uint64   `json:"id"`
	OK      bool     `json:"ok"`
	Code    string   `json:"code,omitempty"`
	Error   string   `json:"error,omitempty"`
	N       int      `json:"n,omitempty"`
	Dim     int      `json:"dim,omitempty"`
	Loaded  []string `json:"loaded,omitempty"`
	Device  string   `json:"device,omitempty"`  // with Loaded: "cpu" or "cuda"
	Helper  string   `json:"helper,omitempty"`  // hello: protocol and build
	Runtime string   `json:"runtime,omitempty"` // hello: onnxruntime version
	Proto   int      `json:"proto,omitempty"`
}

// Conn reads and writes frames. A frame is
//
//	uint32le header length | uint32le blob length | header JSON | blob
type Conn struct {
	r *bufio.Reader
	w io.Writer
}

// NewConn wraps a duplex pair.
func NewConn(r io.Reader, w io.Writer) *Conn {
	return &Conn{r: bufio.NewReaderSize(r, 1<<16), w: w}
}

// Write sends one frame. It is not safe for concurrent use; callers
// serialize.
func (c *Conn) Write(header any, blob []byte) error {
	h, err := json.Marshal(header)
	if err != nil {
		return err
	}
	if len(h) > maxHeader || len(blob) > maxBlob {
		return fmt.Errorf("wire: frame too large (%d + %d bytes)", len(h), len(blob))
	}
	var pre [8]byte
	binary.LittleEndian.PutUint32(pre[0:], uint32(len(h)))
	binary.LittleEndian.PutUint32(pre[4:], uint32(len(blob)))
	buf := make([]byte, 0, 8+len(h))
	buf = append(buf, pre[:]...)
	buf = append(buf, h...)
	if _, err := c.w.Write(buf); err != nil {
		return err
	}
	if len(blob) > 0 {
		if _, err := c.w.Write(blob); err != nil {
			return err
		}
	}
	return nil
}

// Read receives one frame and decodes its header into header.
func (c *Conn) Read(header any) ([]byte, error) {
	var pre [8]byte
	if _, err := io.ReadFull(c.r, pre[:]); err != nil {
		return nil, err
	}
	hl, bl := binary.LittleEndian.Uint32(pre[0:]), binary.LittleEndian.Uint32(pre[4:])
	if hl > maxHeader || bl > maxBlob {
		return nil, fmt.Errorf("wire: frame too large (%d + %d bytes)", hl, bl)
	}
	h := make([]byte, hl)
	if _, err := io.ReadFull(c.r, h); err != nil {
		return nil, midFrame(err)
	}
	var blob []byte
	if bl > 0 {
		blob = make([]byte, bl)
		if _, err := io.ReadFull(c.r, blob); err != nil {
			return nil, midFrame(err)
		}
	}
	if err := json.Unmarshal(h, header); err != nil {
		return nil, fmt.Errorf("wire: bad header: %w", err)
	}
	return blob, nil
}

// midFrame turns a clean EOF inside a frame into an unexpected one, so a
// peer that died mid-message is not mistaken for an orderly close.
func midFrame(err error) error {
	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	return err
}

// AppendFloat32 appends little-endian float32 values.
func AppendFloat32(b []byte, v []float32) []byte {
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

// AppendInt64 appends little-endian int64 values.
func AppendInt64(b []byte, v []int64) []byte {
	for _, x := range v {
		b = binary.LittleEndian.AppendUint64(b, uint64(x))
	}
	return b
}

// AppendBool appends one byte per value.
func AppendBool(b []byte, v []bool) []byte {
	for _, x := range v {
		if x {
			b = append(b, 1)
		} else {
			b = append(b, 0)
		}
	}
	return b
}

// Reader consumes typed values from a blob and remembers the first
// short-read error, so decoding code stays linear.
type Reader struct {
	b   []byte
	Err error
}

// NewReader wraps a blob.
func NewReader(b []byte) *Reader { return &Reader{b: b} }

func (r *Reader) take(n int) []byte {
	if r.Err != nil {
		return nil
	}
	if n < 0 || n > len(r.b) {
		r.Err = fmt.Errorf("wire: blob too short (need %d, have %d)", n, len(r.b))
		return nil
	}
	out := r.b[:n]
	r.b = r.b[n:]
	return out
}

// Float32 reads n float32 values.
func (r *Reader) Float32(n int) []float32 {
	b := r.take(n * 4)
	if b == nil {
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// Int64 reads n int64 values.
func (r *Reader) Int64(n int) []int64 {
	b := r.take(n * 8)
	if b == nil {
		return nil
	}
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64(b[i*8:]))
	}
	return out
}

// Bool reads n bool values.
func (r *Reader) Bool(n int) []bool {
	b := r.take(n)
	if b == nil {
		return nil
	}
	out := make([]bool, n)
	for i := range out {
		out[i] = b[i] != 0
	}
	return out
}

// Done reports leftover bytes as an error.
func (r *Reader) Done() error {
	if r.Err == nil && len(r.b) != 0 {
		r.Err = fmt.Errorf("wire: %d unexpected trailing blob bytes", len(r.b))
	}
	return r.Err
}
