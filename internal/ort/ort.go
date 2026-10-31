// Package ort is a small binding to the ONNX Runtime C API that loads
// libonnxruntime.so at run time with purego, so binaries keep building with
// CGO_ENABLED=0 and use the runtime the distribution ships.
package ort

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ErrLibraryNotFound means no usable libonnxruntime was found; the wrapped
// message lists what was tried.
var ErrLibraryNotFound = errors.New("onnxruntime library not found")

// ErrUnsupportedVersion means the library is older than API version 22.
var ErrUnsupportedVersion = errors.New("onnxruntime library too old")

// Error is a failure reported by the runtime through an OrtStatus.
type Error struct {
	Code int
	Msg  string
}

func (e *Error) Error() string { return fmt.Sprintf("onnxruntime: %s (code %d)", e.Msg, e.Code) }

// api holds the OrtApi entries used, bound to Go function values.
type api struct {
	getErrorCode     func(status uintptr) int32
	getErrorMessage  func(status uintptr) uintptr
	releaseStatus    func(status uintptr)
	createEnv        func(level int32, logid string, out *uintptr) uintptr
	releaseEnv       func(env uintptr)
	createSessOpts   func(out *uintptr) uintptr
	releaseSessOpts  func(opts uintptr)
	setIntraThreads  func(opts uintptr, n int32) uintptr
	setInterThreads  func(opts uintptr, n int32) uintptr
	setGraphOptLevel func(opts uintptr, level int32) uintptr
	disableCPUArena  func(opts uintptr) uintptr
	addConfigEntry   func(opts uintptr, key, value string) uintptr
	createSession    func(env uintptr, path string, opts uintptr, out *uintptr) uintptr
	releaseSession   func(sess uintptr)
	inputCount       func(sess uintptr, out *uintptr) uintptr
	outputCount      func(sess uintptr, out *uintptr) uintptr
	inputName        func(sess uintptr, i uintptr, alloc uintptr, out *uintptr) uintptr
	outputName       func(sess uintptr, i uintptr, alloc uintptr, out *uintptr) uintptr
	defaultAllocator func(out *uintptr) uintptr
	allocatorFree    func(alloc uintptr, p uintptr) uintptr
	createTensor     func(alloc uintptr, shape *int64, rank uintptr, typ int32, out *uintptr) uintptr
	tensorData       func(value uintptr, out *uintptr) uintptr
	releaseValue     func(value uintptr)
	getValueType     func(value uintptr, out *int32) uintptr
	typeAndShape     func(value uintptr, out *uintptr) uintptr
	releaseShape     func(info uintptr)
	elemType         func(info uintptr, out *int32) uintptr
	dimsCount        func(info uintptr, out *uintptr) uintptr
	dims             func(info uintptr, out *int64, n uintptr) uintptr
	elemCount        func(info uintptr, out *uintptr) uintptr
	run              func(sess, runOpts uintptr, inNames **byte, ins *uintptr, nIn uintptr, outNames **byte, nOut uintptr, outs *uintptr) uintptr
	providers        func(out *uintptr, n *int32) uintptr
	releaseProviders func(list uintptr, n int32) uintptr
	createCUDAOpts   func(out *uintptr) uintptr
	updateCUDAOpts   func(opts uintptr, keys, values **byte, n uintptr) uintptr
	releaseCUDAOpts  func(opts uintptr)
	appendCUDA       func(sessOpts, cudaOpts uintptr) uintptr
	createRunOpts    func(out *uintptr) uintptr
	releaseRunOpts   func(opts uintptr)
	addRunConfig     func(opts uintptr, key, value string) uintptr
}

// Runtime is a loaded libonnxruntime with one shared environment.
type Runtime struct {
	lib     uintptr
	path    string
	version string
	a       api
	env     uintptr
	alloc   uintptr // default CPU allocator, owned by the runtime

	mu     sync.Mutex
	closed bool
}

// Load opens the runtime library. path may be empty to search the standard
// locations (see Candidates). Libraries are never unloaded: ORT keeps worker
// threads and static state that dlclose cannot tear down safely.
func Load(path string) (*Runtime, error) {
	cands, err := Candidates(path)
	if err != nil {
		return nil, err
	}
	var tried []string
	tooOld := false
	for _, c := range cands {
		lib, derr := purego.Dlopen(c, purego.RTLD_NOW|purego.RTLD_LOCAL)
		if derr != nil {
			tried = append(tried, fmt.Sprintf("%s (%v)", c, derr))
			continue
		}
		rt := &Runtime{lib: lib, path: c}
		berr := rt.bind()
		if berr == nil {
			return rt, nil
		}
		tooOld = tooOld || errors.Is(berr, ErrUnsupportedVersion)
		tried = append(tried, fmt.Sprintf("%s (%v)", c, berr))
	}
	kind := ErrLibraryNotFound
	if tooOld {
		kind = ErrUnsupportedVersion
	}
	return nil, fmt.Errorf("%w; tried: %s", kind, strings.Join(tried, "; "))
}

// Candidates returns the library paths Load tries. An explicit path is the
// only candidate; it must exist.
func Candidates(explicit string) ([]string, error) {
	if explicit != "" {
		if _, err := os.Stat(explicit); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrLibraryNotFound, err)
		}
		return []string{explicit}, nil
	}
	var out []string
	dirs := []string{
		"/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
		"/usr/local/lib", "/usr/local/lib64", "/usr/lib/onnxruntime", "/opt/onnxruntime/lib",
	}
	for _, d := range dirs {
		if p := filepath.Join(d, "libonnxruntime.so"); fileExists(p) {
			out = append(out, p)
		}
		m, _ := filepath.Glob(filepath.Join(d, "libonnxruntime.so.*"))
		sort.Sort(sort.Reverse(sort.StringSlice(m)))
		for _, p := range m {
			if fileExists(p) {
				out = append(out, p)
			}
		}
	}
	// Last resort: the dynamic loader's own search (LD_LIBRARY_PATH, ldconfig).
	out = append(out, "libonnxruntime.so.1", "libonnxruntime.so")
	return out, nil
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

func (r *Runtime) bind() error {
	sym, err := purego.Dlsym(r.lib, "OrtGetApiBase")
	if err != nil {
		return fmt.Errorf("not an onnxruntime library: %w", err)
	}
	var getBase func() uintptr
	purego.RegisterFunc(&getBase, sym)
	base := getBase()
	if base == 0 {
		return errors.New("OrtGetApiBase returned null")
	}
	baseTab := ptr(base)
	var getAPI func(version uint32) uintptr
	var getVersion func() uintptr
	purego.RegisterFunc(&getAPI, *(*uintptr)(baseTab))
	purego.RegisterFunc(&getVersion, *(*uintptr)(unsafe.Add(baseTab, 8)))
	r.version = cString(getVersion())
	tab := getAPI(APIVersion)
	if tab == 0 {
		return fmt.Errorf("%w: runtime %s does not implement API %d", ErrUnsupportedVersion, r.version, APIVersion)
	}
	tp := ptr(tab)
	reg := func(fn any, idx int) {
		purego.RegisterFunc(fn, *(*uintptr)(unsafe.Add(tp, idx*int(unsafe.Sizeof(uintptr(0))))))
	}
	a := &r.a
	reg(&a.getErrorCode, idxGetErrorCode)
	reg(&a.getErrorMessage, idxGetErrorMessage)
	reg(&a.releaseStatus, idxReleaseStatus)
	reg(&a.createEnv, idxCreateEnv)
	reg(&a.releaseEnv, idxReleaseEnv)
	reg(&a.createSessOpts, idxCreateSessionOptions)
	reg(&a.releaseSessOpts, idxReleaseSessionOptions)
	reg(&a.setIntraThreads, idxSetIntraOpNumThreads)
	reg(&a.setInterThreads, idxSetInterOpNumThreads)
	reg(&a.setGraphOptLevel, idxSetSessionGraphOptLevel)
	reg(&a.disableCPUArena, idxDisableCpuMemArena)
	reg(&a.addConfigEntry, idxAddSessionConfigEntry)
	reg(&a.createSession, idxCreateSession)
	reg(&a.releaseSession, idxReleaseSession)
	reg(&a.inputCount, idxSessionGetInputCount)
	reg(&a.outputCount, idxSessionGetOutputCount)
	reg(&a.inputName, idxSessionGetInputName)
	reg(&a.outputName, idxSessionGetOutputName)
	reg(&a.defaultAllocator, idxGetAllocatorWithDefaultOptions)
	reg(&a.allocatorFree, idxAllocatorFree)
	reg(&a.createTensor, idxCreateTensorAsOrtValue)
	reg(&a.tensorData, idxGetTensorMutableData)
	reg(&a.releaseValue, idxReleaseValue)
	reg(&a.getValueType, idxGetValueType)
	reg(&a.typeAndShape, idxGetTensorTypeAndShape)
	reg(&a.releaseShape, idxReleaseTensorTypeAndShape)
	reg(&a.elemType, idxGetTensorElementType)
	reg(&a.dimsCount, idxGetDimensionsCount)
	reg(&a.dims, idxGetDimensions)
	reg(&a.elemCount, idxGetTensorShapeElementCount)
	reg(&a.run, idxRun)
	reg(&a.providers, idxGetAvailableProviders)
	reg(&a.releaseProviders, idxReleaseAvailableProviders)
	reg(&a.createCUDAOpts, idxCreateCUDAProviderOptions)
	reg(&a.updateCUDAOpts, idxUpdateCUDAProviderOptions)
	reg(&a.releaseCUDAOpts, idxReleaseCUDAProviderOptions)
	reg(&a.appendCUDA, idxAppendCUDAProviderV2)
	reg(&a.createRunOpts, idxCreateRunOptions)
	reg(&a.releaseRunOpts, idxReleaseRunOptions)
	reg(&a.addRunConfig, idxAddRunConfigEntry)

	if err := r.check(a.createEnv(logError, "dsearch", &r.env)); err != nil {
		return err
	}
	return r.check(a.defaultAllocator(&r.alloc))
}

// Version returns the runtime's version string, e.g. "1.29.0".
func (r *Runtime) Version() string { return r.version }

// Path returns the library file that was loaded.
func (r *Runtime) Path() string { return r.path }

// Providers lists the execution providers the library was built with, in
// the runtime's order of preference; "CPUExecutionProvider" is always there.
// A listed provider can still fail to start, for example without a GPU or
// driver: NewSession reports that.
func (r *Runtime) Providers() ([]string, error) {
	var list uintptr
	var n int32
	if err := r.check(r.a.providers(&list, &n)); err != nil {
		return nil, err
	}
	out := make([]string, n)
	for i := range out {
		out[i] = cString(*(*uintptr)(unsafe.Add(ptr(list), i*int(unsafe.Sizeof(uintptr(0))))))
	}
	return out, r.check(r.a.releaseProviders(list, n))
}

// HasProvider says whether the library was built with the named provider.
func (r *Runtime) HasProvider(name string) bool {
	ps, err := r.Providers()
	return err == nil && slices.Contains(ps, name)
}

// check converts an OrtStatus into an error and releases it.
func (r *Runtime) check(status uintptr) error {
	if status == 0 {
		return nil
	}
	e := &Error{Code: int(r.a.getErrorCode(status)), Msg: cString(r.a.getErrorMessage(status))}
	r.a.releaseStatus(status)
	return e
}

// Close releases the environment. Sessions and tensors must be closed first.
// The library stays mapped.
func (r *Runtime) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	r.closed = true
	r.a.releaseEnv(r.env)
	r.env = 0
}

// Options configure a Session.
type Options struct {
	IntraOpThreads int // 0 = runtime default
	InterOpThreads int // 0 = 1; the models here are sequential graphs
	// DisableCPUMemArena frees intermediate buffers after every run instead of
	// caching them, trading a little speed for a lower resident size.
	DisableCPUMemArena bool
	// Config holds raw session config entries ("session.xxx" keys).
	Config map[string]string
	// CUDA runs the graph on an NVIDIA GPU through the CUDA execution
	// provider; operators it does not implement stay on the CPU. Nil runs
	// everything on the CPU.
	CUDA *CUDAOptions
}

// CUDAOptions configure the CUDA execution provider.
type CUDAOptions struct {
	DeviceID int    // GPU index as CUDA numbers them
	MemLimit uint64 // bytes the provider's arena may hold; 0 = no limit
}

// Session is a loaded model. Run is safe for concurrent use.
type Session struct {
	rt      *Runtime
	h       uintptr
	runOpts uintptr // 0 = defaults
	inputs  []string
	outputs []string
}

// NewSession loads the model at path. External weight files (.onnx_data)
// next to it are resolved by the runtime.
func (r *Runtime) NewSession(path string, o Options) (*Session, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	var opts uintptr
	if err := r.check(r.a.createSessOpts(&opts)); err != nil {
		return nil, err
	}
	defer r.a.releaseSessOpts(opts)
	if o.IntraOpThreads > 0 {
		if err := r.check(r.a.setIntraThreads(opts, int32(o.IntraOpThreads))); err != nil {
			return nil, err
		}
	}
	inter := o.InterOpThreads
	if inter <= 0 {
		inter = 1
	}
	if err := r.check(r.a.setInterThreads(opts, int32(inter))); err != nil {
		return nil, err
	}
	if err := r.check(r.a.setGraphOptLevel(opts, optAll)); err != nil {
		return nil, err
	}
	if o.DisableCPUMemArena {
		if err := r.check(r.a.disableCPUArena(opts)); err != nil {
			return nil, err
		}
	}
	for k, v := range o.Config {
		if err := r.check(r.a.addConfigEntry(opts, k, v)); err != nil {
			return nil, fmt.Errorf("config %s: %w", k, err)
		}
	}
	if o.CUDA != nil {
		if err := r.appendCUDA(opts, *o.CUDA); err != nil {
			return nil, fmt.Errorf("CUDA execution provider: %w", err)
		}
	}
	s := &Session{rt: r}
	if o.CUDA != nil {
		// The GPU arena keeps what the largest batch needed; shrink it after
		// every run so an idle helper does not hold video memory other
		// programs need.
		if err := r.check(r.a.createRunOpts(&s.runOpts)); err != nil {
			return nil, err
		}
		if err := r.check(r.a.addRunConfig(s.runOpts, "memory.enable_memory_arena_shrinkage", "gpu:"+strconv.Itoa(o.CUDA.DeviceID))); err != nil {
			s.Close()
			return nil, err
		}
	}
	if err := r.check(r.a.createSession(r.env, path, opts, &s.h)); err != nil {
		s.Close()
		return nil, fmt.Errorf("load %s: %w", filepath.Base(path), err)
	}
	var err error
	if s.inputs, err = r.names(s.h, r.a.inputCount, r.a.inputName); err == nil {
		s.outputs, err = r.names(s.h, r.a.outputCount, r.a.outputName)
	}
	if err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

// appendCUDA adds the CUDA execution provider to session options. A runtime
// built without it fails here; one without a usable GPU fails when the
// session is created.
func (r *Runtime) appendCUDA(opts uintptr, o CUDAOptions) error {
	var cuda uintptr
	if err := r.check(r.a.createCUDAOpts(&cuda)); err != nil {
		return err
	}
	defer r.a.releaseCUDAOpts(cuda)
	// The arena grows by what a run asks for instead of doubling, so a long
	// document does not reserve twice the memory it needs.
	keys := []*byte{cStringBytes("device_id"), cStringBytes("arena_extend_strategy")}
	values := []*byte{cStringBytes(strconv.Itoa(o.DeviceID)), cStringBytes("kSameAsRequested")}
	if o.MemLimit > 0 {
		keys = append(keys, cStringBytes("gpu_mem_limit"))
		values = append(values, cStringBytes(strconv.FormatUint(o.MemLimit, 10)))
	}
	err := r.check(r.a.updateCUDAOpts(cuda, &keys[0], &values[0], uintptr(len(keys))))
	runtime.KeepAlive(keys)
	runtime.KeepAlive(values)
	if err != nil {
		return err
	}
	return r.check(r.a.appendCUDA(opts, cuda))
}

func (r *Runtime) names(sess uintptr, count func(uintptr, *uintptr) uintptr, name func(uintptr, uintptr, uintptr, *uintptr) uintptr) ([]string, error) {
	var n uintptr
	if err := r.check(count(sess, &n)); err != nil {
		return nil, err
	}
	out := make([]string, n)
	for i := range out {
		var p uintptr
		if err := r.check(name(sess, uintptr(i), r.alloc, &p)); err != nil {
			return nil, err
		}
		out[i] = cString(p)
		if err := r.check(r.a.allocatorFree(r.alloc, p)); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// InputNames and OutputNames list the graph's tensors in declaration order.
func (s *Session) InputNames() []string  { return append([]string(nil), s.inputs...) }
func (s *Session) OutputNames() []string { return append([]string(nil), s.outputs...) }

// Close releases the session and its weights.
func (s *Session) Close() {
	if s.h != 0 {
		s.rt.a.releaseSession(s.h)
		s.h = 0
	}
	if s.runOpts != 0 {
		s.rt.a.releaseRunOpts(s.runOpts)
		s.runOpts = 0
	}
}

// Input is one named model input.
type Input struct {
	Name   string
	Tensor *Tensor
}

// Run executes the graph and returns the requested outputs in order. The
// caller owns the returned tensors and must Close them.
func (s *Session) Run(inputs []Input, outputs []string) ([]*Tensor, error) {
	if s.h == 0 {
		return nil, errors.New("ort: session closed")
	}
	inNames := make([]*byte, len(inputs))
	inVals := make([]uintptr, len(inputs))
	for i, in := range inputs {
		if in.Tensor == nil || in.Tensor.h == 0 {
			return nil, fmt.Errorf("ort: input %q has no tensor", in.Name)
		}
		inNames[i] = cStringBytes(in.Name)
		inVals[i] = in.Tensor.h
	}
	outNames := make([]*byte, len(outputs))
	for i, n := range outputs {
		outNames[i] = cStringBytes(n)
	}
	outVals := make([]uintptr, len(outputs))
	var inN, outN **byte
	var inV, outV *uintptr
	if len(inputs) > 0 {
		inN, inV = &inNames[0], &inVals[0]
	}
	if len(outputs) > 0 {
		outN, outV = &outNames[0], &outVals[0]
	}
	status := s.rt.a.run(s.h, s.runOpts, inN, inV, uintptr(len(inputs)), outN, uintptr(len(outputs)), outV)
	runtime.KeepAlive(inNames)
	runtime.KeepAlive(outNames)
	runtime.KeepAlive(inputs)
	runtime.KeepAlive(inVals)
	if err := s.rt.check(status); err != nil {
		return nil, err
	}
	res := make([]*Tensor, len(outVals))
	for i, h := range outVals {
		res[i] = &Tensor{rt: s.rt, h: h}
	}
	return res, nil
}

// ptr reinterprets an address owned by C as an unsafe.Pointer without the
// uintptr conversion go vet rejects.
func ptr(u uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&u)) }

// cString reads a NUL-terminated string from C memory.
func cString(p uintptr) string {
	if p == 0 {
		return ""
	}
	base := ptr(p)
	n := 0
	for *(*byte)(unsafe.Add(base, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(base), n))
}

func cStringBytes(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}
