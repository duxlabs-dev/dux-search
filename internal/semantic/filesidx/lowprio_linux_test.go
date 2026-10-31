//go:build linux

package filesidx

import (
	"context"
	"sync/atomic"
	"testing"

	"golang.org/x/sys/unix"
)

func TestWorkerThreadRunsAtLowestPriority(t *testing.T) {
	type result struct {
		nice, class int
		ok          bool
	}
	ch := make(chan result, 1)
	go func() {
		ok := lowerThreadPriority()
		tid := unix.Gettid()
		nice, _ := unix.Getpriority(unix.PRIO_PROCESS, tid)
		// The kernel returns 20 - nice from getpriority(2).
		v, _, errno := unix.Syscall(unix.SYS_IOPRIO_GET, 1, uintptr(tid), 0)
		if errno != 0 {
			v = 0
		}
		ch <- result{nice: 20 - nice, class: int(v) >> 13, ok: ok}
	}()
	got := <-ch
	if !got.ok || got.nice != 19 || got.class != 3 {
		t.Fatalf("nice=%d ioprio class=%d applied=%v, want 19 / 3 (idle)", got.nice, got.class, got.ok)
	}
	if n, _ := unix.Getpriority(unix.PRIO_PROCESS, unix.Gettid()); 20-n == 19 {
		t.Log("note: the test goroutine's own thread is niced as well")
	}
}

func TestWorkerGoroutineIsLowPriorityInPractice(t *testing.T) {
	// The embedder is called from the worker: look at the thread that does it.
	var nice atomic.Int32
	nice.Store(-100)
	r := newRig(t, nil)
	r.emb.gate = func(context.Context) error {
		n, _ := unix.Getpriority(unix.PRIO_PROCESS, unix.Gettid())
		nice.Store(int32(20 - n))
		return nil
	}
	r.start()
	r.x.Enqueue(r.copyFixture("files/builder.go", "builder.go"), OpIndex)
	r.settle()
	if nice.Load() != 19 {
		t.Fatalf("embedder was called from a thread with nice %d, want 19", nice.Load())
	}
}
