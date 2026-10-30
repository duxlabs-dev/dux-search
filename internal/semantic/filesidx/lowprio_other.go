//go:build !linux

package filesidx

import "runtime"

// lowerThreadPriority only pins the goroutine on systems without the Linux
// priority syscalls.
func lowerThreadPriority() bool {
	runtime.LockOSThread()
	return false
}
