package ort

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Tensor is a runtime-owned dense tensor. Memory lives in the runtime, so
// nothing in Go memory is retained by C after a call returns.
type Tensor struct {
	rt *Runtime
	h  uintptr
}

func (r *Runtime) newTensor(shape []int64, typ int32) (*Tensor, unsafe.Pointer, int, error) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			return nil, nil, 0, fmt.Errorf("ort: negative dimension in shape %v", shape)
		}
		n *= int(d)
	}
	var sp *int64
	if len(shape) > 0 {
		sp = &shape[0]
	}
	t := &Tensor{rt: r}
	if err := r.check(r.a.createTensor(r.alloc, sp, uintptr(len(shape)), typ, &t.h)); err != nil {
		return nil, nil, 0, err
	}
	runtime.KeepAlive(shape)
	if n == 0 {
		return t, nil, 0, nil
	}
	var data uintptr
	if err := r.check(r.a.tensorData(t.h, &data)); err != nil {
		t.Close()
		return nil, nil, 0, err
	}
	return t, ptr(data), n, nil
}

func checkLen(n, have int, shape []int64) error {
	if n != have {
		return fmt.Errorf("ort: shape %v needs %d elements, got %d", shape, n, have)
	}
	return nil
}

// NewFloat32 creates a float32 tensor holding a copy of data.
func (r *Runtime) NewFloat32(shape []int64, data []float32) (*Tensor, error) {
	t, p, n, err := r.newTensor(shape, elemFloat32)
	if err != nil {
		return nil, err
	}
	if err := checkLen(n, len(data), shape); err != nil {
		t.Close()
		return nil, err
	}
	if n > 0 {
		copy(unsafe.Slice((*float32)(p), n), data)
	}
	return t, nil
}

// NewInt64 creates an int64 tensor holding a copy of data.
func (r *Runtime) NewInt64(shape []int64, data []int64) (*Tensor, error) {
	t, p, n, err := r.newTensor(shape, elemInt64)
	if err != nil {
		return nil, err
	}
	if err := checkLen(n, len(data), shape); err != nil {
		t.Close()
		return nil, err
	}
	if n > 0 {
		copy(unsafe.Slice((*int64)(p), n), data)
	}
	return t, nil
}

// NewBool creates a bool tensor holding a copy of data.
func (r *Runtime) NewBool(shape []int64, data []bool) (*Tensor, error) {
	t, p, n, err := r.newTensor(shape, elemBool)
	if err != nil {
		return nil, err
	}
	if err := checkLen(n, len(data), shape); err != nil {
		t.Close()
		return nil, err
	}
	if n > 0 {
		copy(unsafe.Slice((*bool)(p), n), data)
	}
	return t, nil
}

// info reads element type, shape and element count.
func (t *Tensor) info() (typ int32, shape []int64, n int, err error) {
	a := &t.rt.a
	var vt int32
	if err = t.rt.check(a.getValueType(t.h, &vt)); err != nil {
		return
	}
	if vt != onnxTypeTensor {
		err = fmt.Errorf("ort: value is not a tensor (type %d)", vt)
		return
	}
	var ti uintptr
	if err = t.rt.check(a.typeAndShape(t.h, &ti)); err != nil {
		return
	}
	defer a.releaseShape(ti)
	if err = t.rt.check(a.elemType(ti, &typ)); err != nil {
		return
	}
	var rank, count uintptr
	if err = t.rt.check(a.dimsCount(ti, &rank)); err != nil {
		return
	}
	shape = make([]int64, rank)
	if rank > 0 {
		if err = t.rt.check(a.dims(ti, &shape[0], rank)); err != nil {
			return
		}
	}
	if err = t.rt.check(a.elemCount(ti, &count)); err != nil {
		return
	}
	return typ, shape, int(count), nil
}

// Shape returns the tensor dimensions.
func (t *Tensor) Shape() ([]int64, error) {
	_, s, _, err := t.info()
	return s, err
}

// Float32 returns the shape and a copy of the data of a float32 tensor.
func (t *Tensor) Float32() ([]float32, []int64, error) {
	typ, shape, n, err := t.info()
	if err != nil {
		return nil, nil, err
	}
	if typ != elemFloat32 {
		return nil, nil, fmt.Errorf("ort: tensor element type %d is not float32", typ)
	}
	out := make([]float32, n)
	if n == 0 {
		return out, shape, nil
	}
	var data uintptr
	if err := t.rt.check(t.rt.a.tensorData(t.h, &data)); err != nil {
		return nil, nil, err
	}
	copy(out, unsafe.Slice((*float32)(ptr(data)), n))
	return out, shape, nil
}

// Close releases the tensor.
func (t *Tensor) Close() {
	if t != nil && t.h != 0 {
		t.rt.a.releaseValue(t.h)
		t.h = 0
	}
}
