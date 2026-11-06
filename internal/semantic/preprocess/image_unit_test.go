package preprocess

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"testing"
)

// grid builds a w x h picture whose pixel (x, y) is (x, y, 7).
func grid(w, h int) decodedImage {
	pix := make([]uint8, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			pix[(y*w+x)*3], pix[(y*w+x)*3+1], pix[(y*w+x)*3+2] = uint8(x), uint8(y), 7
		}
	}
	return decodedImage{w: w, h: h, pix: pix, orient: 1}
}

// TestOrientationViews checks the eight EXIF orientations against the PIL
// transpose operations they stand for, written out for a 3x2 picture.
func TestOrientationViews(t *testing.T) {
	// Source (x, y):   (0,0) (1,0) (2,0)
	//                  (0,1) (1,1) (2,1)
	type xy [2]int
	want := map[int][][]xy{ // rows of oriented picture, each entry the source pixel
		1: {{{0, 0}, {1, 0}, {2, 0}}, {{0, 1}, {1, 1}, {2, 1}}},   // as stored
		2: {{{2, 0}, {1, 0}, {0, 0}}, {{2, 1}, {1, 1}, {0, 1}}},   // FLIP_LEFT_RIGHT
		3: {{{2, 1}, {1, 1}, {0, 1}}, {{2, 0}, {1, 0}, {0, 0}}},   // ROTATE_180
		4: {{{0, 1}, {1, 1}, {2, 1}}, {{0, 0}, {1, 0}, {2, 0}}},   // FLIP_TOP_BOTTOM
		5: {{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{2, 0}, {2, 1}}}, // TRANSPOSE
		6: {{{0, 1}, {0, 0}}, {{1, 1}, {1, 0}}, {{2, 1}, {2, 0}}}, // ROTATE_270 (90 clockwise)
		7: {{{2, 1}, {2, 0}}, {{1, 1}, {1, 0}}, {{0, 1}, {0, 0}}}, // TRANSVERSE
		8: {{{2, 0}, {2, 1}}, {{1, 0}, {1, 1}}, {{0, 0}, {0, 1}}}, // ROTATE_90 (counter-clockwise)
	}
	for o, rows := range want {
		d := grid(3, 2)
		d.orient = o
		v := d.view()
		if v.h != len(rows) || v.w != len(rows[0]) {
			t.Fatalf("orientation %d: view %dx%d, want %dx%d", o, v.w, v.h, len(rows[0]), len(rows))
		}
		for y, row := range rows {
			for x, src := range row {
				p := v.base + y*v.sy + x*v.sx
				got := [2]int{int(v.pix[p]), int(v.pix[p+1])}
				if got != src {
					t.Errorf("orientation %d: pixel (%d,%d) = source %v, want %v", o, x, y, got, src)
				}
			}
		}
	}
}

func TestPatchifyLayout(t *testing.T) {
	// 96 x 48 picture: two patch rows... pixel value encodes (x, y, c).
	w, h := 96, 48
	rgb := make([]uint8, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rgb[(y*w+x)*3] = uint8(x)
			rgb[(y*w+x)*3+1] = uint8(y)
			rgb[(y*w+x)*3+2] = uint8(x ^ y)
		}
	}
	in := patchify(rgb, w, h)
	if in.NumPatches != 6*3 || in.SoftTokens != 2 {
		t.Fatalf("patches=%d soft=%d", in.NumPatches, in.SoftTokens)
	}
	// Patch (px=4, py=2) is index 2*6+4; its element (ky=5, kx=9, c=1) is
	// source pixel (4*16+9, 2*16+5) channel 1.
	idx := (2*6+4)*PatchDim + (5*16+9)*3 + 1
	if got, want := in.Pixels[idx], float32(2*16+5)*float32(rescaleFactor); got != want {
		t.Fatalf("pixel layout: got %v want %v", got, want)
	}
	if in.Positions[(2*6+4)*2] != 4 || in.Positions[(2*6+4)*2+1] != 2 {
		t.Fatalf("positions (x,y) wrong: %v", in.Positions[(2*6+4)*2:(2*6+4)*2+2])
	}
}

func TestPaddedLayout(t *testing.T) {
	in := patchify(grid(48, 48).pix, 48, 48)
	p := Padded(in)
	if len(p.Pixels) != MaxPatches*PatchDim || len(p.Positions) != MaxPatches*2 {
		t.Fatalf("padded sizes %d %d", len(p.Pixels), len(p.Positions))
	}
	if p.NumPatches != 9 || p.SoftTokens != 1 {
		t.Fatalf("counts changed: %d %d", p.NumPatches, p.SoftTokens)
	}
	if p.Positions[8*2] != 2 || p.Positions[9*2] != -1 || p.Positions[MaxPatches*2-1] != -1 {
		t.Fatalf("padding positions wrong")
	}
	if p.Pixels[9*PatchDim] != 0 {
		t.Fatalf("padding pixels not zero")
	}
}

func TestResizeConstantImageStaysConstant(t *testing.T) {
	d := decodedImage{w: 37, h: 23, pix: bytes.Repeat([]byte{200, 100, 5}, 37*23), orient: 1}
	out := resizeBicubic(d.view(), 96, 144)
	for i := 0; i < len(out); i += 3 {
		if out[i] != 200 || out[i+1] != 100 || out[i+2] != 5 {
			t.Fatalf("pixel %d = %v", i/3, out[i:i+3])
		}
	}
}

func TestImageErrors(t *testing.T) {
	if _, err := Image(bytes.NewReader([]byte("definitely not a picture"))); !errors.Is(err, ErrUnsupported) {
		t.Errorf("garbage: %v", err)
	}
	if _, err := Image(bytes.NewReader(nil)); !errors.Is(err, ErrUnsupported) {
		t.Errorf("empty: %v", err)
	}
	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for i := range img.Pix {
		img.Pix[i] = 0x80
	}
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	if _, err := Image(bytes.NewReader(data[:len(data)/2])); err == nil || errors.Is(err, ErrTooLarge) {
		t.Errorf("truncated png: %v", err)
	}
	old := MaxPixels
	MaxPixels = 64*64 - 1
	_, err := Image(bytes.NewReader(data))
	MaxPixels = old
	if !errors.Is(err, ErrTooLarge) {
		t.Errorf("MaxPixels not enforced before decoding: %v", err)
	}
	oldBytes := MaxImageBytes
	MaxImageBytes = 100
	_, err = Image(bytes.NewReader(data))
	MaxImageBytes = oldBytes
	if !errors.Is(err, ErrTooLarge) {
		t.Errorf("MaxImageBytes not enforced: %v", err)
	}
	// A plain solid picture still goes through: 64x64 -> 3x...
	in, err := Image(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if in.NumPatches%9 != 0 || in.SoftTokens != in.NumPatches/9 || in.NumPatches > MaxPatches {
		t.Errorf("counts: %d patches, %d soft tokens", in.NumPatches, in.SoftTokens)
	}
	_ = color.RGBA{}
}

// PIL opens a 16-bit grey PNG as mode "I;16" and convert("RGB") clips the
// sample to 255 (verified with Pillow 13: [0, 255, 256, 40000] -> [0, 255,
// 255, 255]); the reference behaviour is reproduced.
func TestGray16FollowsPIL(t *testing.T) {
	img := image.NewGray16(image.Rect(0, 0, 4, 1))
	for i, v := range []uint16{0, 255, 256, 40000} {
		img.SetGray16(i, 0, color.Gray16{Y: v})
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	d, err := decodeImage(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{0, 255, 255, 255}
	for i, w := range want {
		if d.pix[i*3] != w || d.pix[i*3+1] != w || d.pix[i*3+2] != w {
			t.Errorf("sample %d = %v, want %d", i, d.pix[i*3:i*3+3], w)
		}
	}
}

// Alpha is dropped, not composited: a fully transparent pixel keeps its colour.
func TestAlphaIsDropped(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	img.SetNRGBA(0, 0, color.NRGBA{R: 10, G: 20, B: 30, A: 0})
	img.SetNRGBA(1, 0, color.NRGBA{R: 200, G: 100, B: 50, A: 128})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	d, err := decodeImage(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{10, 20, 30, 200, 100, 50}
	if !bytes.Equal(d.pix, want) {
		t.Errorf("got %v, want %v", d.pix, want)
	}
}

func TestExifOrientationParsing(t *testing.T) {
	// Little- and big-endian TIFF blocks with one orientation entry.
	le := []byte{'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0}
	be := []byte{'M', 'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0}
	if got := exifOrientation(le); got != 6 {
		t.Errorf("little endian: %d", got)
	}
	if got := exifOrientation(append([]byte("Exif\x00\x00"), be...)); got != 8 {
		t.Errorf("big endian with Exif prefix: %d", got)
	}
	for _, bad := range [][]byte{nil, {1}, []byte("II"), {'I', 'I', 42, 0, 255, 255, 255, 255}, {'I', 'I', 42, 0, 8, 0, 0, 0, 200, 0}} {
		if got := exifOrientation(bad); got != 1 {
			t.Errorf("%v -> %d", bad, got)
		}
	}
	// Out-of-range value.
	le[18] = 9
	if got := exifOrientation(le); got != 1 {
		t.Errorf("orientation 9 -> %d", got)
	}
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 20000; i++ {
		b := make([]byte, rng.Intn(64))
		rng.Read(b)
		if len(b) > 8 && i%2 == 0 {
			copy(b, le[:8])
		}
		exifOrientation(b)
		jpegOrientation(append([]byte{0xFF, 0xD8, 0xFF, 0xE1}, b...))
		pngOrientation(append([]byte("\x89PNG\r\n\x1a\n"), b...))
	}
}
