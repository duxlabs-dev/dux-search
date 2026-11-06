package ort

import (
	"sync"

	"github.com/ebitengine/purego"
)

var (
	trimOnce sync.Once
	trimFn   func(pad uintptr) int32
)

// TrimMemory asks glibc to return free heap pages to the kernel. ONNX Runtime
// allocates and frees large activation buffers on every run; without this the
// freed pages stay resident in the process. It does nothing on a C library
// without malloc_trim (musl).
func TrimMemory() {
	trimOnce.Do(func() {
		for _, name := range []string{"libc.so.6"} {
			lib, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err != nil {
				continue
			}
			if sym, err := purego.Dlsym(lib, "malloc_trim"); err == nil {
				purego.RegisterFunc(&trimFn, sym)
				return
			}
		}
	})
	if trimFn != nil {
		trimFn(0)
	}
}
