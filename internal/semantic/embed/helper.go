package embed

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/wire"
)

// errGone reports that the helper process ended (crash, kill, or its own
// exit) while a request was waiting; the request may be retried on a fresh
// helper.
type errGone struct {
	cause  error
	stderr string
}

func (e *errGone) Error() string {
	msg := "embedding helper exited"
	if e.cause != nil && !errors.Is(e.cause, io.EOF) {
		msg += ": " + e.cause.Error()
	}
	if e.stderr != "" {
		msg += " (" + e.stderr + ")"
	}
	return msg
}

// tail keeps the last bytes written to it; the helper's stderr is shown in
// errors when it dies.
type tail struct {
	mu  sync.Mutex
	buf []byte
}

const tailSize = 2048

func (t *tail) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > tailSize {
		t.buf = t.buf[len(t.buf)-tailSize:]
	}
	return len(p), nil
}

func (t *tail) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(bytes.TrimSpace(t.buf))
}

type reply struct {
	resp wire.Response
	blob []byte
	err  error
}

// helper is one running dsearch-embed process.
type helper struct {
	cmd    *exec.Cmd
	conn   *wire.Conn
	stdin  io.WriteCloser
	stderr *tail
	done   chan struct{} // closed when the process has been reaped
	onExit func(*helper)

	wmu     sync.Mutex // serializes frame writes
	mu      sync.Mutex
	nextID  uint64
	pending map[uint64]chan reply
	exitErr error
	dead    bool
}

// startHelper launches the helper and waits for its hello.
func startHelper(path string, args []string, onExit func(*helper)) (*helper, wire.Response, error) {
	cmd := exec.Command(path, args...)
	cmd.Env = os.Environ()
	if os.Getenv("GOMAXPROCS") == "" {
		// The helper only marshals buffers; the work happens in ORT's threads.
		cmd.Env = append(cmd.Env, "GOMAXPROCS=2")
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, wire.Response{}, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, wire.Response{}, err
	}
	h := &helper{cmd: cmd, stdin: stdin, stderr: &tail{}, done: make(chan struct{}), pending: map[uint64]chan reply{}, onExit: onExit}
	cmd.Stderr = h.stderr
	h.conn = wire.NewConn(stdout, stdin)
	if err := cmd.Start(); err != nil {
		return nil, wire.Response{}, fmt.Errorf("start %s: %w", path, err)
	}
	go h.readLoop(stdout)
	ctx, cancel := context.WithTimeout(context.Background(), helloTimeout)
	defer cancel()
	resp, _, err := h.roundTrip(ctx, wire.Request{Op: wire.OpHello}, nil)
	if err != nil {
		h.kill()
		return nil, resp, err
	}
	return h, resp, nil
}

func (h *helper) readLoop(r io.Reader) {
	for {
		var resp wire.Response
		blob, err := h.conn.Read(&resp)
		if err != nil {
			h.finish(err)
			return
		}
		h.mu.Lock()
		ch := h.pending[resp.ID]
		delete(h.pending, resp.ID)
		h.mu.Unlock()
		if ch != nil {
			ch <- reply{resp: resp, blob: blob}
		}
	}
}

// finish reaps the process and fails everything still waiting.
func (h *helper) finish(cause error) {
	_ = h.stdin.Close()
	h.kill()         // a corrupt stream leaves the process running
	_ = h.cmd.Wait() // also drains the stderr copier
	h.mu.Lock()
	if h.dead {
		h.mu.Unlock()
		return
	}
	h.dead, h.exitErr = true, cause
	pending := h.pending
	h.pending = map[uint64]chan reply{}
	h.mu.Unlock()
	gone := &errGone{cause: cause, stderr: lastLine(h.stderr.String())}
	for _, ch := range pending {
		ch <- reply{err: gone}
	}
	close(h.done)
	if h.onExit != nil {
		h.onExit(h)
	}
}

func lastLine(s string) string {
	if i := bytes.LastIndexByte([]byte(s), '\n'); i >= 0 {
		s = s[i+1:]
	}
	return s
}

// roundTrip sends one request and waits for its reply, the context, or the
// helper's death.
func (h *helper) roundTrip(ctx context.Context, req wire.Request, blob []byte) (wire.Response, []byte, error) {
	ch := make(chan reply, 1)
	h.mu.Lock()
	if h.dead {
		h.mu.Unlock()
		return wire.Response{}, nil, &errGone{cause: h.exitErr, stderr: lastLine(h.stderr.String())}
	}
	h.nextID++
	req.ID = h.nextID
	h.pending[req.ID] = ch
	h.mu.Unlock()
	h.wmu.Lock()
	err := h.conn.Write(req, blob)
	h.wmu.Unlock()
	if err != nil {
		h.mu.Lock()
		delete(h.pending, req.ID)
		h.mu.Unlock()
		h.kill()
		return wire.Response{}, nil, &errGone{cause: err, stderr: lastLine(h.stderr.String())}
	}
	select {
	case r := <-ch:
		return r.resp, r.blob, r.err
	case <-ctx.Done():
		h.mu.Lock()
		delete(h.pending, req.ID)
		h.mu.Unlock()
		// Tell the helper not to start the work if it has not yet.
		h.wmu.Lock()
		_ = h.conn.Write(wire.Request{Op: wire.OpCancel, Cancel: req.ID}, nil)
		h.wmu.Unlock()
		return wire.Response{}, nil, ctx.Err()
	}
}

// alive reports whether the process is still running.
func (h *helper) alive() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return !h.dead
}

// kill ends the process at once.
func (h *helper) kill() {
	if h.cmd.Process != nil {
		_ = h.cmd.Process.Kill()
	}
}

// stop asks the helper to quit and kills it if it does not leave in time.
func (h *helper) stop(grace time.Duration) {
	if !h.alive() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	_, _, _ = h.roundTrip(ctx, wire.Request{Op: wire.OpQuit}, nil)
	_ = h.stdin.Close()
	select {
	case <-h.done:
	case <-time.After(grace):
		h.kill()
		<-h.done
	}
}
