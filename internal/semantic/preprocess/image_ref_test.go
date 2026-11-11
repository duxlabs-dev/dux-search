package preprocess

import (
	"bytes"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

type imgRef struct {
	File       string `json:"file"`
	Level      string `json:"level"` // "full": whole processor, "decode": decoded RGB picture only
	SrcSize    [2]int `json:"src_size"`
	NumPatches int    `json:"num_patches"`
	SoftTokens int    `json:"soft_tokens"`
	TargetSize [2]int `json:"target_size"` // width, height of the resized picture
	ResizedPNG string `json:"resized_png"`
	DecodedPNG string `json:"decoded_png"`
	PixelFile  string `json:"pixel_file"` // raw padded pixel_values (a few pictures)
	PosFile    string `json:"pos_file"`
}

type sizeRef struct {
	H     int    `json:"h"`
	W     int    `json:"w"`
	TH    int    `json:"th"`
	TW    int    `json:"tw"`
	Error string `json:"error"`
}

type imagesRef struct {
	Versions map[string]string `json:"versions"`
	Images   []imgRef          `json:"images"`
	Sizes    []sizeRef         `json:"sizes"`
}

// TestTargetSizeReference compares TargetSize with
// get_aspect_ratio_preserving_size on shapes including the degenerate ones.
func TestTargetSizeReference(t *testing.T) {
	var ref imagesRef
	readJSON(t, "images.json", &ref)
	if len(ref.Sizes) == 0 {
		t.Fatal("no size references")
	}
	for _, s := range ref.Sizes {
		th, tw, err := TargetSize(s.H, s.W, PatchSize, MaxPatches, PoolingKernel)
		if s.Error != "" {
			if err == nil {
				t.Errorf("%dx%d: want error (%s), got %dx%d", s.W, s.H, s.Error, tw, th)
			}
			continue
		}
		if err != nil || th != s.TH || tw != s.TW {
			t.Errorf("%dx%d: got %dx%d (%v), want %dx%d", s.W, s.H, tw, th, err, s.TW, s.TH)
		}
	}
}

func isJPEG(file string) bool {
	ext := strings.ToLower(filepath.Ext(file))
	return ext == ".jpg" || ext == ".jpeg"
}

// Tolerances, in 8-bit levels (pixel_values * 255); one level is 0.0039.
//
// Lossless formats and lossy WebP decode bit-exactly (decode level: 0 levels).
// The resize follows PIL's algorithm exactly, but the reference processor
// resizes with torchvision, whose antialiased uint8 kernel quantises the
// filter weights to 16 bits instead of 22 and so differs from PIL by up to 2
// levels on about 0.1 % of the values (measured in gen_reference.py as
// torchvision_vs_pil_*). JPEG adds the IDCT difference between Go's decoder
// and libjpeg-turbo (up to 3 levels on about 2 % of the samples), which the
// chroma upsampling and the resize can carry to 4.
const (
	fullLosslessMax, fullLosslessMean = 2.0, 0.01
	fullJPEGMax, fullJPEGMean         = 5.0, 0.1
	decodeJPEGMax, decodeJPEGMean     = 3.0, 0.06
)

func decodePNGRGB(t testing.TB, name string) (rgb []byte, w, h int) {
	t.Helper()
	f, err := os.Open(testdataPath("ref", name))
	if err != nil {
		t.Fatalf("reference data missing (run testdata/gen_reference.py): %v", err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	w, h = b.Dx(), b.Dy()
	rgb = make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			o := (y*w + x) * 3
			rgb[o], rgb[o+1], rgb[o+2] = byte(r>>8), byte(g>>8), byte(bl>>8)
		}
	}
	return rgb, w, h
}

// refPatches flattens a resized picture into patches by the index formula of
// reshape(C, nh, p, nw, p).permute(1, 3, 2, 4, 0): element (py, px, ky, kx, c)
// comes from pixel (py*16+ky, px*16+kx), channel c. The reference script
// asserts that transformers' own convert_image_to_patches agrees with this
// layout for every picture before storing it.
func refPatches(rgb []byte, w, h int) []byte {
	nw, nh := w/PatchSize, h/PatchSize
	out := make([]byte, nw*nh*PatchDim)
	for py := 0; py < nh; py++ {
		for px := 0; px < nw; px++ {
			for ky := 0; ky < PatchSize; ky++ {
				for kx := 0; kx < PatchSize; kx++ {
					for c := 0; c < 3; c++ {
						out[(((py*nw+px)*PatchSize+ky)*PatchSize+kx)*3+c] = rgb[((py*PatchSize+ky)*w+px*PatchSize+kx)*3+c]
					}
				}
			}
		}
	}
	return out
}

func TestImageReference(t *testing.T) {
	var ref imagesRef
	readJSON(t, "images.json", &ref)
	full := 0
	for _, e := range ref.Images {
		if e.Level == "full" {
			full++
		}
	}
	if full < 12 {
		t.Fatalf("only %d full-pipeline reference images", full)
	}
	for _, e := range ref.Images {
		e := e
		t.Run(e.File, func(t *testing.T) {
			data, err := os.ReadFile(testdataPath("images", e.File))
			if err != nil {
				t.Fatal(err)
			}
			if e.Level == "decode" {
				checkDecode(t, e, data)
				return
			}
			got, err := Image(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			rgb, w, h := decodePNGRGB(t, e.ResizedPNG)
			if w != e.TargetSize[0] || h != e.TargetSize[1] {
				t.Fatalf("reference picture is %dx%d, expected %v", w, h, e.TargetSize)
			}
			wantPix := refPatches(rgb, w, h)
			if got.NumPatches != e.NumPatches || got.SoftTokens != e.SoftTokens || len(wantPix) != e.NumPatches*PatchDim {
				t.Fatalf("patches/soft tokens: got %d/%d, want %d/%d", got.NumPatches, got.SoftTokens, e.NumPatches, e.SoftTokens)
			}
			if len(got.Pixels) != got.NumPatches*PatchDim || len(got.Positions) != got.NumPatches*2 {
				t.Fatalf("output sizes %d %d for %d patches", len(got.Pixels), len(got.Positions), got.NumPatches)
			}
			// pixel_position_ids: (x, y) of each patch, row-major over the grid.
			nw := w / PatchSize
			for i := 0; i < got.NumPatches; i++ {
				if got.Positions[2*i] != int64(i%nw) || got.Positions[2*i+1] != int64(i/nw) {
					t.Fatalf("position %d = (%d,%d), want (%d,%d)", i, got.Positions[2*i], got.Positions[2*i+1], i%nw, i/nw)
				}
			}
			var maxd, sum float64
			var differing int
			scale := float32(rescaleFactor)
			for i, v := range got.Pixels {
				d := math.Abs(float64(v)-float64(float32(wantPix[i])*scale)) * 255
				maxd = math.Max(maxd, d)
				if d > 1e-3 {
					differing++
				}
				sum += d
			}
			mean := sum / float64(len(got.Pixels))
			tolMax, tolMean := fullLosslessMax, fullLosslessMean
			if isJPEG(e.File) {
				tolMax, tolMean = fullJPEGMax, fullJPEGMean
			}
			t.Logf("%dx%d -> %dx%d, %d patches / %d soft tokens: max diff %.0f levels, mean %.5f, %.3f%% of values differ", e.SrcSize[0], e.SrcSize[1], w, h, got.NumPatches, got.SoftTokens, maxd, mean, 100*float64(differing)/float64(len(got.Pixels)))
			if maxd > tolMax || mean > tolMean {
				t.Errorf("pixel_values differ from transformers: max %.0f levels (limit %.0f), mean %.5f (limit %.5f)", maxd, tolMax, mean, tolMean)
			}
			if e.PixelFile != "" {
				checkRawPadded(t, e, got)
			}
		})
	}
}

// checkRawPadded compares Padded(got) with transformers' complete padded
// tensors, which pins down the patch layout, position ids and padding rule
// independently of refPatches.
func checkRawPadded(t *testing.T, e imgRef, got semantic.ImageInput) {
	t.Helper()
	wantPix := readGz(t, e.PixelFile)
	wantPos := i16s(readGz(t, e.PosFile))
	pad := Padded(got)
	if len(wantPix) != MaxPatches*PatchDim || len(wantPos) != MaxPatches*2 || len(pad.Pixels) != len(wantPix) || len(pad.Positions) != len(wantPos) {
		t.Fatalf("padded sizes %d/%d, reference %d/%d", len(pad.Pixels), len(pad.Positions), len(wantPix), len(wantPos))
	}
	for i, p := range pad.Positions {
		if int64(wantPos[i]) != p {
			t.Fatalf("padded position %d: got %d want %d", i, p, wantPos[i])
		}
	}
	var maxd float64
	for i, v := range pad.Pixels {
		maxd = math.Max(maxd, math.Abs(float64(v)-float64(float32(wantPix[i])*float32(rescaleFactor)))*255)
		if i >= e.NumPatches*PatchDim && wantPix[i] != 0 {
			t.Fatalf("reference padding value %d is not zero", i)
		}
	}
	tolMax, _ := fullLosslessMax, 0
	if isJPEG(e.File) {
		tolMax = fullJPEGMax
	}
	if maxd > tolMax {
		t.Errorf("raw padded pixel_values: max diff %.0f levels", maxd)
	}
}

func checkDecode(t *testing.T, e imgRef, data []byte) {
	t.Helper()
	dec, err := decodeImage(data)
	if err != nil {
		t.Fatal(err)
	}
	v := dec.view()
	rgb, w, h := decodePNGRGB(t, e.DecodedPNG)
	if v.w != w || v.h != h {
		t.Fatalf("decoded size %dx%d (orientation %d), want %dx%d", v.w, v.h, dec.orient, w, h)
	}
	got := packView(v)
	var maxd int
	var sum float64
	for i := range got {
		d := int(got[i]) - int(rgb[i])
		if d < 0 {
			d = -d
		}
		maxd = max(maxd, d)
		sum += float64(d)
	}
	mean := sum / float64(len(got))
	tolMax, tolMean := 0.0, 0.0
	if isJPEG(e.File) {
		tolMax, tolMean = decodeJPEGMax, decodeJPEGMean
	}
	t.Logf("decoded %dx%d (orientation %d): max diff %d levels, mean %.5f", w, h, dec.orient, maxd, mean)
	if float64(maxd) > tolMax || mean > tolMean {
		t.Errorf("decoded picture differs from PIL: max %d (limit %.0f), mean %.5f (limit %.5f)", maxd, tolMax, mean, tolMean)
	}
}
