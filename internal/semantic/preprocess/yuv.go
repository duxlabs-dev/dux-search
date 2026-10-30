package preprocess

import "image"

// libjpeg-turbo compatible YCbCr -> RGB (jdcolor.c, jdsample.c). PIL decodes
// JPEG with libjpeg-turbo: chroma is upsampled with the "fancy" triangle
// filter and converted with 16-bit fixed point tables. Go's image/jpeg hands
// out the raw planes, so both steps are done here to stay close to the
// reference.

const (
	jpegScaleBits = 16
	jpegHalf      = 1 << (jpegScaleBits - 1)
)

func jfix(x float64) int32 { return int32(x*(1<<jpegScaleBits) + 0.5) }

var (
	crR, cbB, crG, cbG [256]int32
)

func init() {
	for i := 0; i < 256; i++ {
		x := int32(i - 128)
		crR[i] = (jfix(1.40200)*x + jpegHalf) >> jpegScaleBits
		cbB[i] = (jfix(1.77200)*x + jpegHalf) >> jpegScaleBits
		crG[i] = -jfix(0.71414) * x
		cbG[i] = -jfix(0.34414)*x + jpegHalf
	}
}

func clampU8(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func yccToRGB(dst []uint8, y []uint8, cb, cr []uint8, w int) {
	for x := 0; x < w; x++ {
		yy := int32(y[x])
		cbv, crv := cb[x], cr[x]
		dst[x*3] = clampU8(yy + crR[crv])
		dst[x*3+1] = clampU8(yy + ((cbG[cbv] + crG[crv]) >> jpegScaleBits))
		dst[x*3+2] = clampU8(yy + cbB[cbv])
	}
}

// ycbcrToRGB converts a decoded JPEG to packed RGB8.
func ycbcrToRGB(img *image.YCbCr) []uint8 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h*3)
	cw := w
	ch := h
	var hs, vs int // horizontal / vertical chroma subsampling factors
	switch img.SubsampleRatio {
	case image.YCbCrSubsampleRatio444:
		hs, vs = 1, 1
	case image.YCbCrSubsampleRatio422:
		hs, vs = 2, 1
	case image.YCbCrSubsampleRatio420:
		hs, vs = 2, 2
	case image.YCbCrSubsampleRatio440:
		hs, vs = 1, 2
	case image.YCbCrSubsampleRatio411:
		hs, vs = 4, 1
	case image.YCbCrSubsampleRatio410:
		hs, vs = 4, 2
	}
	cw = (w + hs - 1) / hs
	ch = (h + vs - 1) / vs
	cbRow := make([]uint8, w)
	crRow := make([]uint8, w)
	crow := func(plane []uint8, r int) []uint8 {
		if r < 0 {
			r = 0
		}
		if r >= ch {
			r = ch - 1
		}
		return plane[r*img.CStride : r*img.CStride+cw]
	}
	for y := 0; y < h; y++ {
		yrow := img.Y[y*img.YStride : y*img.YStride+w]
		switch {
		case hs == 1 && vs == 1:
			copy(cbRow, img.Cb[y*img.CStride:y*img.CStride+w])
			copy(crRow, img.Cr[y*img.CStride:y*img.CStride+w])
		case hs == 2 && vs == 1 && cw > 2:
			upsampleH2V1(cbRow, crow(img.Cb, y), w)
			upsampleH2V1(crRow, crow(img.Cr, y), w)
		case hs == 2 && vs == 2 && cw > 2:
			near := y / 2
			far := near - 1
			if y&1 == 1 {
				far = near + 1
			}
			upsampleH2V2(cbRow, crow(img.Cb, near), crow(img.Cb, far), w)
			upsampleH2V2(crRow, crow(img.Cr, near), crow(img.Cr, far), w)
		case hs == 1 && vs == 2:
			near := y / 2
			far := near - 1
			bias := int32(1)
			if y&1 == 1 {
				far = near + 1
				bias = 2
			}
			upsampleH1V2(cbRow, crow(img.Cb, near), crow(img.Cb, far), w, bias)
			upsampleH1V2(crRow, crow(img.Cr, near), crow(img.Cr, far), w, bias)
		default:
			// Box replication (4:1:1, 4:1:0 and narrow pictures).
			cbr, crr := crow(img.Cb, y/vs), crow(img.Cr, y/vs)
			for x := 0; x < w; x++ {
				cbRow[x], crRow[x] = cbr[x/hs], crr[x/hs]
			}
		}
		yccToRGB(out[y*w*3:], yrow, cbRow, crRow, w)
	}
	return out
}

// upsampleH2V1 is h2v1_fancy_upsample: out[2i] = (3*in[i] + in[i-1] + 1) >> 2,
// out[2i+1] = (3*in[i] + in[i+1] + 2) >> 2, edges copy the border sample.
func upsampleH2V1(dst, in []uint8, w int) {
	n := len(in)
	set := func(i int, v int32) {
		if i < w {
			dst[i] = uint8(v)
		}
	}
	set(0, int32(in[0]))
	set(1, (int32(in[0])*3+int32(in[1])+2)>>2)
	for i := 1; i < n-1; i++ {
		v := int32(in[i]) * 3
		set(2*i, (v+int32(in[i-1])+1)>>2)
		set(2*i+1, (v+int32(in[i+1])+2)>>2)
	}
	v := int32(in[n-1]) * 3
	set(2*(n-1), (v+int32(in[n-2])+1)>>2)
	set(2*(n-1)+1, int32(in[n-1]))
}

// upsampleH2V2 is h2v2_fancy_upsample for one output row: near is the chroma
// row nearest to the output row, far the neighbour on the output row's side.
func upsampleH2V2(dst, near, far []uint8, w int) {
	n := len(near)
	set := func(i int, v int32) {
		if i < w {
			dst[i] = uint8(v)
		}
	}
	this := int32(near[0])*3 + int32(far[0])
	next := int32(near[1])*3 + int32(far[1])
	set(0, (this*4+8)>>4)
	set(1, (this*3+next+7)>>4)
	last := this
	this = next
	for i := 1; i < n-1; i++ {
		next = int32(near[i+1])*3 + int32(far[i+1])
		set(2*i, (this*3+last+8)>>4)
		set(2*i+1, (this*3+next+7)>>4)
		last = this
		this = next
	}
	set(2*(n-1), (this*3+last+8)>>4)
	set(2*(n-1)+1, (this*4+7)>>4)
}

// upsampleH1V2 is h1v2_fancy_upsample (libjpeg-turbo): vertical triangle
// filter; bias is 1 for the upper and 2 for the lower output row.
func upsampleH1V2(dst, near, far []uint8, w int, bias int32) {
	for i := 0; i < w; i++ {
		dst[i] = uint8((int32(near[i])*3 + int32(far[i]) + bias) >> 2)
	}
}

// WebP: libwebp's fancy upsampler and YUV->RGB (upsampling.c, yuv.h).

func mulHi(v, coeff int32) int32 { return (v * coeff) >> 8 }

func webpClip8(v int32) uint8 {
	if v&^((256<<6)-1) == 0 {
		return uint8(v >> 6)
	}
	if v < 0 {
		return 0
	}
	return 255
}

func webpPixel(dst []uint8, y, u, v int32) {
	dst[0] = webpClip8(mulHi(y, 19077) + mulHi(v, 26149) - 14234)
	dst[1] = webpClip8(mulHi(y, 19077) - mulHi(u, 6419) - mulHi(v, 13320) + 8708)
	dst[2] = webpClip8(mulHi(y, 19077) + mulHi(u, 33050) - 17685)
}

// webpUpsampleLine is UpsampleRgbLinePair for one pair of rows; bottom may be
// nil. tu/tv are the chroma rows above the pair, cu/cv the chroma rows below.
func webpUpsampleLine(topY, botY, tu, tv, cu, cv, topDst, botDst []uint8, w int) {
	last := (w - 1) >> 1
	tlu, tlv := int32(tu[0]), int32(tv[0])
	lu, lv := int32(cu[0]), int32(cv[0])
	webpPixel(topDst[0:], int32(topY[0]), (3*tlu+lu+2)>>2, (3*tlv+lv+2)>>2)
	if botY != nil {
		webpPixel(botDst[0:], int32(botY[0]), (3*lu+tlu+2)>>2, (3*lv+tlv+2)>>2)
	}
	for x := 1; x <= last; x++ {
		tu1, tv1 := int32(tu[x]), int32(tv[x])
		u1, v1 := int32(cu[x]), int32(cv[x])
		avgU := tlu + tu1 + lu + u1 + 8
		avgV := tlv + tv1 + lv + v1 + 8
		d12u := (avgU + 2*(tu1+lu)) >> 3
		d12v := (avgV + 2*(tv1+lv)) >> 3
		d03u := (avgU + 2*(tlu+u1)) >> 3
		d03v := (avgV + 2*(tlv+v1)) >> 3
		webpPixel(topDst[(2*x-1)*3:], int32(topY[2*x-1]), (d12u+tlu)>>1, (d12v+tlv)>>1)
		webpPixel(topDst[(2*x)*3:], int32(topY[2*x]), (d03u+tu1)>>1, (d03v+tv1)>>1)
		if botY != nil {
			webpPixel(botDst[(2*x-1)*3:], int32(botY[2*x-1]), (d03u+lu)>>1, (d03v+lv)>>1)
			webpPixel(botDst[(2*x)*3:], int32(botY[2*x]), (d12u+u1)>>1, (d12v+v1)>>1)
		}
		tlu, tlv = tu1, tv1
		lu, lv = u1, v1
	}
	if w&1 == 0 {
		webpPixel(topDst[(w-1)*3:], int32(topY[w-1]), (3*tlu+lu+2)>>2, (3*tlv+lv+2)>>2)
		if botY != nil {
			webpPixel(botDst[(w-1)*3:], int32(botY[w-1]), (3*lu+tlu+2)>>2, (3*lv+tlv+2)>>2)
		}
	}
}

// webpYCbCrToRGB converts a decoded VP8 frame (4:2:0) like libwebp does with
// its default options (fancy upsampling on).
func webpYCbCrToRGB(img *image.YCbCr) []uint8 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h*3)
	yrow := func(r int) []uint8 { return img.Y[r*img.YStride : r*img.YStride+w] }
	cw := (w + 1) / 2
	urow := func(r int) []uint8 { return img.Cb[r*img.CStride : r*img.CStride+cw] }
	vrow := func(r int) []uint8 { return img.Cr[r*img.CStride : r*img.CStride+cw] }
	dst := func(r int) []uint8 { return out[r*w*3 : (r+1)*w*3] }
	webpUpsampleLine(yrow(0), nil, urow(0), vrow(0), urow(0), vrow(0), dst(0), nil, w)
	k := 1
	for ; 2*k < h; k++ {
		webpUpsampleLine(yrow(2*k-1), yrow(2*k), urow(k-1), vrow(k-1), urow(k), vrow(k), dst(2*k-1), dst(2*k), w)
	}
	if h > 1 && h%2 == 0 {
		webpUpsampleLine(yrow(h-1), nil, urow(k-1), vrow(k-1), urow(k-1), vrow(k-1), dst(h-1), nil, w)
	}
	return out
}
