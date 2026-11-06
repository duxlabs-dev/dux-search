package preprocess

import (
	"fmt"
	"io"
	"math"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// Image processor constants, from processor_config.json (image_processor).
const (
	PatchSize     = 16
	PoolingKernel = 3
	MaxSoftTokens = 280
	// MaxPatches is the patch budget: max_soft_tokens * pooling_kernel_size^2.
	MaxPatches = MaxSoftTokens * PoolingKernel * PoolingKernel
	// PatchDim is the length of one flattened patch (16 x 16 x 3).
	PatchDim = PatchSize * PatchSize * 3

	// rescaleFactor is the JSON value 0.00392156862745098; transformers
	// multiplies the uint8 pixels (as float32) by it.
	rescaleFactor = 0.00392156862745098
)

// Limits applied before any pixel is decoded.
var (
	// MaxPixels is the largest picture (width*height) Image accepts. Decoding
	// needs about 4.5 bytes per pixel at peak (Go's YCbCr planes plus the RGB
	// buffer), so 100 megapixels is roughly 450 MB; larger pictures return
	// ErrTooLarge. PIL's own decompression-bomb limit is 179 megapixels.
	MaxPixels = 100_000_000
	// MaxDecodeBytes bounds the memory a single decode may need, estimated from
	// the header before decoding (decoder planes plus the RGB buffer).
	MaxDecodeBytes int64 = 768 << 20
	// MaxImageBytes bounds the file size Image reads into memory.
	MaxImageBytes int64 = 256 << 20
)

// Image decodes r (jpeg, png, gif first frame, webp, bmp) and builds the
// vision tower input exactly like Gemma4ImageProcessor: the EXIF orientation is
// applied (transformers.load_image does), the picture is converted to RGB
// (alpha is dropped, not composited, like PIL's convert("RGB")), resized with
// an aspect-preserving bicubic antialiased resize to the largest size whose
// sides are multiples of PatchSize*PoolingKernel and that fits MaxPatches
// patches, rescaled to [0,1], and cut into 16x16x3 patches in row-major order.
//
// The returned input holds only the real patches (NumPatches <= MaxPatches);
// Padded produces the MaxPatches-long layout transformers returns.
func Image(r io.Reader) (semantic.ImageInput, error) {
	data, err := readLimited(r, MaxImageBytes)
	if err != nil {
		return semantic.ImageInput{}, err
	}
	dec, err := decodeImage(data)
	if err != nil {
		return semantic.ImageInput{}, err
	}
	return buildImageInput(dec.view(), MaxSoftTokens)
}

func readLimited(r io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("%w: file larger than %d bytes", ErrTooLarge, limit)
	}
	return data, nil
}

// TargetSize returns the resized height and width get_aspect_ratio_preserving_size
// computes: the largest size that fits maxPatches patches of patchSize and is
// divisible by poolingKernel*patchSize on both sides.
func TargetSize(height, width, patchSize, maxPatches, poolingKernel int) (int, int, error) {
	if height <= 0 || width <= 0 {
		return 0, 0, fmt.Errorf("preprocess: invalid image size %dx%d", width, height)
	}
	totalPx := float64(height) * float64(width)
	targetPx := float64(maxPatches * patchSize * patchSize)
	factor := math.Sqrt(targetPx / totalPx)
	idealH := factor * float64(height)
	idealW := factor * float64(width)
	sideMult := poolingKernel * patchSize

	th := int(math.Floor(idealH/float64(sideMult))) * sideMult
	tw := int(math.Floor(idealW/float64(sideMult))) * sideMult
	if th == 0 && tw == 0 {
		return 0, 0, fmt.Errorf("preprocess: image %dx%d resizes to 0x0 (sides must be multiples of %d)", width, height, sideMult)
	}
	maxSide := (maxPatches / (poolingKernel * poolingKernel)) * sideMult
	if th == 0 {
		th = sideMult
		tw = min(int(math.Floor(float64(width)/float64(height)))*sideMult, maxSide)
	} else if tw == 0 {
		tw = sideMult
		th = min(int(math.Floor(float64(height)/float64(width)))*sideMult, maxSide)
	}
	if float64(th)*float64(tw) > targetPx {
		return 0, 0, fmt.Errorf("preprocess: resizing %dx%d to %dx%d exceeds %d patches", width, height, tw, th, maxPatches)
	}
	return th, tw, nil
}

func buildImageInput(src pixView, maxSoftTokens int) (semantic.ImageInput, error) {
	maxPatches := maxSoftTokens * PoolingKernel * PoolingKernel
	th, tw, err := TargetSize(src.h, src.w, PatchSize, maxPatches, PoolingKernel)
	if err != nil {
		return semantic.ImageInput{}, err
	}
	var rgb []uint8
	if th == src.h && tw == src.w {
		rgb = packView(src)
	} else {
		rgb = resizeBicubic(src, tw, th)
	}
	return patchify(rgb, tw, th), nil
}

// packView copies a view into a tightly packed row-major RGB8 buffer.
func packView(v pixView) []uint8 {
	out := make([]uint8, v.w*v.h*3)
	for y := 0; y < v.h; y++ {
		row := v.base + y*v.sy
		for x := 0; x < v.w; x++ {
			p := row + x*v.sx
			o := (y*v.w + x) * 3
			out[o], out[o+1], out[o+2] = v.pix[p], v.pix[p+1], v.pix[p+2]
		}
	}
	return out
}

// patchify rescales rgb (w x h, both multiples of PatchSize) and flattens it
// into patches: the patch grid is row-major, inside a patch the order is
// (row, column, channel) - reshape(C, nh, p, nw, p).permute(1, 3, 2, 4, 0).
func patchify(rgb []uint8, w, h int) semantic.ImageInput {
	nw, nh := w/PatchSize, h/PatchSize
	n := nw * nh
	pix := make([]float32, n*PatchDim)
	pos := make([]int64, n*2)
	const scale = float32(rescaleFactor)
	for py := 0; py < nh; py++ {
		for px := 0; px < nw; px++ {
			pi := py*nw + px
			pos[pi*2], pos[pi*2+1] = int64(px), int64(py)
			dst := pix[pi*PatchDim : (pi+1)*PatchDim]
			for ky := 0; ky < PatchSize; ky++ {
				src := rgb[((py*PatchSize+ky)*w+px*PatchSize)*3:]
				d := dst[ky*PatchSize*3 : (ky+1)*PatchSize*3]
				for i := range d {
					d[i] = float32(src[i]) * scale
				}
			}
		}
	}
	return semantic.ImageInput{Pixels: pix, Positions: pos, NumPatches: n, SoftTokens: n / (PoolingKernel * PoolingKernel)}
}

// Padded returns in padded to MaxPatches patches the way the transformers
// processor emits it: zero pixels and (-1, -1) positions for the padding.
// NumPatches and SoftTokens keep the real counts.
func Padded(in semantic.ImageInput) semantic.ImageInput {
	if in.NumPatches >= MaxPatches {
		return in
	}
	out := in
	out.Pixels = make([]float32, MaxPatches*PatchDim)
	copy(out.Pixels, in.Pixels)
	out.Positions = make([]int64, MaxPatches*2)
	copy(out.Positions, in.Positions)
	for i := in.NumPatches * 2; i < len(out.Positions); i++ {
		out.Positions[i] = -1
	}
	return out
}
