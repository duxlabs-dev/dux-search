//go:build linux

package filesidx

import (
	"runtime"

	"golang.org/x/sys/unix"
)

const (
	ioprioWhoProcess = 1 // IOPRIO_WHO_PROCESS: a single thread when given a TID
	ioprioClassIdle  = 3
	ioprioClassShift = 13
)

// lowerThreadPriority locks the calling goroutine to its OS thread and puts
// that thread at the lowest CPU priority (nice 19) and the idle I/O class, so
// background indexing only uses what the desktop leaves over. The thread is
// never unlocked: it dies with the goroutine instead of rejoining the pool
// niced. It returns whether both settings were applied.
func lowerThreadPriority() bool {
	runtime.LockOSThread()
	tid := unix.Gettid()
	okCPU := unix.Setpriority(unix.PRIO_PROCESS, tid, 19) == nil
	_, _, errno := unix.Syscall(unix.SYS_IOPRIO_SET, ioprioWhoProcess, uintptr(tid), ioprioClassIdle<<ioprioClassShift)
	return okCPU && errno == 0
}
