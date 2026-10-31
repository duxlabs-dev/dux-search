package preprocess

import "math"

// PIL's separable two-pass resampler (libImaging/Resample.c) for 8-bit
// three-channel data, bicubic kernel (a = -0.5) with antialiasing: the filter
// support grows with the downscale factor, coefficients are normalised per
// output pixel and applied in 22-bit fixed point. transformers' Gemma4 image
// processor calls PIL.Image.resize(BICUBIC) on the PIL backend and
// torchvision.transforms.v2.functional.resize(..., antialias=True), which is
// modelled on the same algorithm, on the torch backend.

const resamplePrecision = 22 // PRECISION_BITS = 32 - 8 - 2

// pixView is a read-only window on an RGB8 buffer, addressed as
// pix[base + x*sx + y*sy] (three consecutive bytes per pixel). Strides let the
// eight EXIF orientations be expressed without copying the picture.
type pixView struct {
	pix    []uint8
	base   int
	sx, sy int // in bytes
	w, h   int
}

func bicubicFilter(x float64) float64 {
	const a = -0.5
	if x < 0 {
		x = -x
	}
	if x < 1 {
		return ((a+2)*x-(a+3))*x*x + 1
	}
	if x < 2 {
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

type coeffs struct {
	ksize  int
	bounds []int32 // per output: xmin, count
	k      []int32 // per output: ksize fixed-point weights
}

func precomputeCoeffs(inSize, outSize int) coeffs {
	const support0 = 2.0
	scale := float64(inSize) / float64(outSize)
	filterscale := scale
	if filterscale < 1 {
		filterscale = 1
	}
	support := support0 * filterscale
	ksize := int(math.Ceil(support))*2 + 1
	c := coeffs{ksize: ksize, bounds: make([]int32, outSize*2), k: make([]int32, outSize*ksize)}
	tmp := make([]float64, ksize)
	ss := 1 / filterscale
	for xx := 0; xx < outSize; xx++ {
		center := (float64(xx) + 0.5) * scale
		xmin := int(center - support + 0.5)
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + support + 0.5)
		if xmax > inSize {
			xmax = inSize
		}
		xmax -= xmin
		var ww float64
		for x := 0; x < xmax; x++ {
			w := bicubicFilter((float64(x+xmin) - center + 0.5) * ss)
			tmp[x] = w
			ww += w
		}
		for x := 0; x < ksize; x++ {
			if x < xmax {
				if ww != 0 {
					tmp[x] /= ww
				}
			} else {
				tmp[x] = 0
			}
		}
		for x := 0; x < ksize; x++ {
			v := tmp[x] * (1 << resamplePrecision)
			if tmp[x] < 0 {
				c.k[xx*ksize+x] = int32(-0.5 + v)
			} else {
				c.k[xx*ksize+x] = int32(0.5 + v)
			}
		}
		c.bounds[xx*2] = int32(xmin)
		c.bounds[xx*2+1] = int32(xmax)
	}
	return c
}

func clip8(v int32) uint8 {
	v >>= resamplePrecision
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// resizeBicubic resizes the view to outW x outH. The horizontal pass runs
// first (only over the source rows the vertical pass reads), then the
// vertical one, exactly like ImagingResample; a pass whose size does not
// change is skipped. The result is a tightly packed RGB8 buffer.
func resizeBicubic(src pixView, outW, outH int) []uint8 {
	needH := outW != src.w
	needV := outH != src.h
	var cv coeffs
	rowLo, rowHi := 0, src.h
	if needV {
		cv = precomputeCoeffs(src.h, outH)
		rowLo = int(cv.bounds[0])
		rowHi = int(cv.bounds[(outH-1)*2]) + int(cv.bounds[(outH-1)*2+1])
	}
	// Horizontal pass into tmp (rows rowLo..rowHi of the source, outW wide).
	var tmp []uint8
	if needH {
		ch := precomputeCoeffs(src.w, outW)
		tmp = make([]uint8, (rowHi-rowLo)*outW*3)
		for y := rowLo; y < rowHi; y++ {
			row := src.base + y*src.sy
			dst := tmp[(y-rowLo)*outW*3:]
			for xx := 0; xx < outW; xx++ {
				xmin := int(ch.bounds[xx*2])
				n := int(ch.bounds[xx*2+1])
				k := ch.k[xx*ch.ksize : xx*ch.ksize+n]
				var s0, s1, s2 int32 = 1 << (resamplePrecision - 1), 1 << (resamplePrecision - 1), 1 << (resamplePrecision - 1)
				p := row + xmin*src.sx
				for _, kv := range k {
					s0 += int32(src.pix[p]) * kv
					s1 += int32(src.pix[p+1]) * kv
					s2 += int32(src.pix[p+2]) * kv
					p += src.sx
				}
				o := xx * 3
				dst[o], dst[o+1], dst[o+2] = clip8(s0), clip8(s1), clip8(s2)
			}
		}
	} else {
		// No horizontal resampling: read the rows straight from the view.
		tmp = make([]uint8, (rowHi-rowLo)*outW*3)
		for y := rowLo; y < rowHi; y++ {
			row := src.base + y*src.sy
			dst := tmp[(y-rowLo)*outW*3:]
			for x := 0; x < outW; x++ {
				p := row + x*src.sx
				dst[x*3], dst[x*3+1], dst[x*3+2] = src.pix[p], src.pix[p+1], src.pix[p+2]
			}
		}
	}
	if !needV {
		return tmp
	}
	out := make([]uint8, outH*outW*3)
	stride := outW * 3
	acc := make([]int32, stride)
	for yy := 0; yy < outH; yy++ {
		ymin := int(cv.bounds[yy*2])
		n := int(cv.bounds[yy*2+1])
		k := cv.k[yy*cv.ksize : yy*cv.ksize+n]
		for i := range acc {
			acc[i] = 1 << (resamplePrecision - 1)
		}
		for j, kv := range k {
			row := tmp[(ymin+j-rowLo)*stride : (ymin+j-rowLo+1)*stride]
			for i, v := range row {
				acc[i] += int32(v) * kv
			}
		}
		dst := out[yy*stride : (yy+1)*stride]
		for i, a := range acc {
			dst[i] = clip8(a)
		}
	}
	return out
}
