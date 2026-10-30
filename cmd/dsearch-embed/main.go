// Command dsearch-embed is the embedding helper of dsearch. It loads the
// EmbeddingGemma 2 towers through ONNX Runtime and answers requests framed by
// internal/semantic/embed/wire on stdin/stdout. dsearch starts it on demand
// and closes it when idle; it exits when stdin closes, so it never outlives
// its parent. Built with CGO_ENABLED=0: the runtime library is opened at run
// time with dlopen.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"sync"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/ort"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/engine"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/wire"
)

// version identifies the helper build in the hello reply.
var version = "dev"

func main() {
	var cfg engine.Config
	showVersion := false
	flag.StringVar(&cfg.ModelDir, "model-dir", "", "model directory (onnx/, tokenizer.model, config.json)")
	flag.StringVar(&cfg.ORTLibrary, "ort-lib", "", "path to libonnxruntime.so (default: standard locations)")
	flag.IntVar(&cfg.Threads, "threads", 0, "intra-op threads (0 = min(4, CPUs))")
	flag.BoolVar(&showVersion, "version", false, "print the version and exit")
	flag.Parse()
	if showVersion {
		fmt.Println("dsearch-embed", version)
		return
	}
	if cfg.ModelDir == "" {
		fmt.Fprintln(os.Stderr, "dsearch-embed: --model-dir is required")
		os.Exit(2)
	}
	// stdout carries the protocol; keep every other writer off it.
	out := os.Stdout
	os.Stdout = os.Stderr
	// The Go heap only holds request blobs; return freed pages promptly.
	debug.SetGCPercent(50)
	serve(cfg, wire.NewConn(os.Stdin, out))
}

type job struct {
	req  wire.Request
	blob []byte
}

func serve(cfg engine.Config, conn *wire.Conn) {
	eng, initErr := engine.New(cfg)
	jobs := make(chan job, 64)
	var cancelled sync.Map // request ids the client abandoned while they were queued
	go func() {
		// EOF on stdin means the parent is gone (or closed us): leave at once,
		// even in the middle of a run.
		for {
			var req wire.Request
			blob, err := conn.Read(&req)
			if err != nil {
				os.Exit(0)
			}
			if req.Op == wire.OpCancel {
				cancelled.Store(req.Cancel, struct{}{})
				continue
			}
			jobs <- job{req, blob}
		}
	}()
	for j := range jobs {
		if _, gone := cancelled.LoadAndDelete(j.req.ID); gone {
			if err := conn.Write(fail(j.req.ID, wire.CodeCancelled, errors.New("cancelled")), nil); err != nil {
				os.Exit(0)
			}
			continue
		}
		trimMu.Lock()
		if trimTimer != nil {
			trimTimer.Stop()
		}
		resp, blob := handle(eng, initErr, j)
		trimMu.Unlock()
		if err := conn.Write(resp, blob); err != nil {
			os.Exit(0)
		}
		if j.req.Op == wire.OpQuit {
			if eng != nil {
				eng.Close()
			}
			os.Exit(0)
		}
		trimSoon()
	}
}

// trimDelay is how long the helper must be quiet before it returns memory
// to the system. Doing it after every request would put a full collection of
// the tokenizer's heap into each query's latency.
const trimDelay = 2 * time.Second

var (
	trimMu    sync.Mutex // held while a request runs or memory is being trimmed
	trimTimer *time.Timer
)

// trimSoon arms the trim timer after a request.
func trimSoon() {
	if trimTimer != nil {
		trimTimer.Stop()
	}
	trimTimer = time.AfterFunc(trimDelay, func() {
		if !trimMu.TryLock() {
			return // a request started meanwhile; its end re-arms the timer
		}
		defer trimMu.Unlock()
		// Request-sized buffers are garbage now; hand the pages back so the
		// idle helper stays small.
		debug.FreeOSMemory()
		ort.TrimMemory()
	})
}

func handle(eng *engine.Engine, initErr error, j job) (resp wire.Response, blob []byte) {
	resp = wire.Response{ID: j.req.ID}
	defer func() {
		if r := recover(); r != nil {
			resp = fail(j.req.ID, wire.CodeInternal, fmt.Errorf("panic: %v", r))
			blob = nil
		}
	}()
	if j.req.Op == wire.OpQuit {
		resp.OK = true
		return
	}
	if initErr != nil {
		if j.req.Op == wire.OpHello {
			resp = fail(j.req.ID, wire.CodeUnavailable, initErr)
			resp.Proto = wire.Protocol
			return
		}
		return fail(j.req.ID, wire.CodeUnavailable, initErr), nil
	}
	switch j.req.Op {
	case wire.OpHello:
		resp.OK, resp.Proto = true, wire.Protocol
		resp.Helper, resp.Runtime = version, eng.RuntimeVersion()
	case wire.OpStatus:
		resp.OK, resp.Loaded = true, towerNames(eng)
	case wire.OpLoad, wire.OpUnload:
		mods, err := parseTowers(j.req.Towers)
		if err == nil && j.req.Op == wire.OpLoad {
			err = eng.Load(mods...)
		} else if err == nil {
			eng.Unload(mods...)
		}
		if err != nil {
			return fail(j.req.ID, code(err), err), nil
		}
		resp.OK, resp.Loaded = true, towerNames(eng)
	case wire.OpText, wire.OpImage, wire.OpAudio:
		vecs, err := embed(eng, j)
		if err != nil {
			return fail(j.req.ID, code(err), err), nil
		}
		resp.OK, resp.N, resp.Dim = true, len(vecs), engineDim(vecs)
		blob = make([]byte, 0, len(vecs)*resp.Dim*4)
		for _, v := range vecs {
			blob = wire.AppendFloat32(blob, v)
		}
		resp.Loaded = towerNames(eng)
	default:
		return fail(j.req.ID, wire.CodeBadRequest, fmt.Errorf("unknown op %q", j.req.Op)), nil
	}
	return
}

func engineDim(v [][]float32) int {
	if len(v) == 0 {
		return 0
	}
	return len(v[0])
}

func fail(id uint64, code string, err error) wire.Response {
	return wire.Response{ID: id, Code: code, Error: err.Error()}
}

func code(err error) string {
	switch {
	case errors.Is(err, engine.ErrUnavailable):
		return wire.CodeUnavailable
	case errors.Is(err, engine.ErrInvalidInput):
		return wire.CodeBadRequest
	}
	return wire.CodeInternal
}
