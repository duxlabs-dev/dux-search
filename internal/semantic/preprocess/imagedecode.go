package preprocess

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"

	"golang.org/x/image/bmp"
	"golang.org/x/image/vp8"
	"golang.org/x/image/vp8l"
)

// decodedImage is a picture converted to packed RGB8 plus its EXIF
// orientation (1 = as stored).
type decodedImage struct {
	w, h   int
	pix    []uint8
	orient int
}

// view returns the picture with the EXIF orientation applied, as a strided
// window on the stored pixels (no copy).
func (d decodedImage) view() pixView {
	w, h := d.w, d.h
	row := 3 * w
	switch d.orient {
	case 2: // mirror horizontally
		return pixView{pix: d.pix, base: 3 * (w - 1), sx: -3, sy: row, w: w, h: h}
	case 3: // rotate 180
		return pixView{pix: d.pix, base: (h-1)*row + 3*(w-1), sx: -3, sy: -row, w: w, h: h}
	case 4: // mirror vertically
		return pixView{pix: d.pix, base: (h - 1) * row, sx: 3, sy: -row, w: w, h: h}
	case 5: // transpose
		return pixView{pix: d.pix, sx: row, sy: 3, w: h, h: w}
	case 6: // rotate 90 clockwise
		return pixView{pix: d.pix, base: (h - 1) * row, sx: -row, sy: 3, w: h, h: w}
	case 7: // transverse
		return pixView{pix: d.pix, base: (h-1)*row + 3*(w-1), sx: -row, sy: -3, w: h, h: w}
	case 8: // rotate 90 counter-clockwise
		return pixView{pix: d.pix, base: 3 * (w - 1), sx: row, sy: -3, w: h, h: w}
	}
	return pixView{pix: d.pix, sx: 3, sy: row, w: w, h: h}
}

// checkSize rejects pictures that are empty, exceed MaxPixels, or whose
// decoding would need more than MaxDecodeBytes: bytesPerPixel is what the
// format's decoder allocates per pixel (3 more are added for the RGB buffer).
func checkSize(w, h int, bytesPerPixel float64) error {
	if w <= 0 || h <= 0 {
		return fmt.Errorf("preprocess: invalid image size %dx%d", w, h)
	}
	px := int64(w) * int64(h)
	if px > int64(MaxPixels) {
		return fmt.Errorf("%w: %dx%d pixels (limit %d)", ErrTooLarge, w, h, MaxPixels)
	}
	if need := float64(px) * (bytesPerPixel + 3); need > float64(MaxDecodeBytes) {
		return fmt.Errorf("%w: %dx%d pixels would need %.0f MB to decode (limit %d MB)", ErrTooLarge, w, h, need/(1<<20), MaxDecodeBytes>>20)
	}
	return nil
}

// colorModelBytes is the per-pixel size of the picture type a standard
// decoder produces for the colour model.
func colorModelBytes(m color.Model) float64 {
	switch m {
	case color.GrayModel, color.AlphaModel:
		return 1
	case color.Gray16Model:
		return 2
	case color.RGBA64Model, color.NRGBA64Model:
		return 8
	case color.YCbCrModel:
		return 1.5
	case color.CMYKModel:
		return 4
	}
	return 4 // RGBA, NRGBA, paletted images carrying 4-byte colours
}

// decodeImage sniffs the format from the magic bytes and decodes to RGB.
func decodeImage(data []byte) (d decodedImage, err error) {
	defer guard("image", &err)
	switch {
	case bytes.HasPrefix(data, []byte{0xFF, 0xD8}):
		return decodeJPEG(data)
	case bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")):
		return decodePNG(data)
	case bytes.HasPrefix(data, []byte("GIF8")):
		return decodeGIF(data)
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		return decodeWebP(data)
	case bytes.HasPrefix(data, []byte("BM")):
		return decodeBMP(data)
	}
	return decodedImage{}, fmt.Errorf("%w: unrecognised image format", ErrUnsupported)
}

func decodeJPEG(data []byte) (decodedImage, error) {
	cfg, err := jpeg.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("jpeg", err)
	}
	if err := checkSize(cfg.Width, cfg.Height, colorModelBytes(cfg.ColorModel)); err != nil {
		return decodedImage{}, err
	}
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("jpeg", err)
	}
	d := decodedImage{w: cfg.Width, h: cfg.Height, orient: jpegOrientation(data)}
	switch m := img.(type) {
	case *image.YCbCr:
		d.pix = ycbcrToRGB(m)
	default:
		d.pix = imageToRGB(img)
	}
	return d, nil
}

func decodePNG(data []byte) (decodedImage, error) {
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("png", err)
	}
	if err := checkSize(cfg.Width, cfg.Height, colorModelBytes(cfg.ColorModel)); err != nil {
		return decodedImage{}, err
	}
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("png", err)
	}
	return decodedImage{w: cfg.Width, h: cfg.Height, pix: imageToRGB(img), orient: pngOrientation(data)}, nil
}

func decodeBMP(data []byte) (decodedImage, error) {
	cfg, err := bmp.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("bmp", err)
	}
	if err := checkSize(cfg.Width, cfg.Height, 4); err != nil {
		return decodedImage{}, err
	}
	img, err := bmp.Decode(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("bmp", err)
	}
	return decodedImage{w: cfg.Width, h: cfg.Height, pix: imageToRGB(img), orient: 1}, nil
}

// wrapDecodeErr maps the standard decoders' "unsupported feature" errors to
// ErrUnsupported and keeps every other failure a plain decode error.
func wrapDecodeErr(format string, err error) error {
	switch err.(type) {
	case jpeg.UnsupportedError, png.UnsupportedError:
		return fmt.Errorf("%w: %s: %v", ErrUnsupported, format, err)
	}
	if format == "bmp" && errors.Is(err, bmp.ErrUnsupported) {
		return fmt.Errorf("%w: bmp: %v", ErrUnsupported, err)
	}
	return fmt.Errorf("preprocess: decode %s: %w", format, err)
}

// imageToRGB converts any decoded picture to packed RGB8 the way PIL's
// convert("RGB") does: alpha is dropped (not composited), 16-bit samples keep
// their high byte, palettes are looked up unpremultiplied, CMYK uses PIL's
// formula.
func imageToRGB(img image.Image) []uint8 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h*3)
	switch m := img.(type) {
	case *image.Gray:
		for y := 0; y < h; y++ {
			row := m.Pix[y*m.Stride : y*m.Stride+w]
			for x, v := range row {
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = v, v, v
			}
		}
	case *image.Gray16:
		// PIL opens 16-bit grey as mode "I;16" and convert("RGB") clips the
		// sample to 255 instead of scaling it, so anything brighter than 255 of
		// 65535 turns white. The reference does this, so it is reproduced.
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				p := y*m.Stride + x*2
				v := uint8(255)
				if m.Pix[p] == 0 && m.Pix[p+1] < 255 {
					v = m.Pix[p+1]
				}
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = v, v, v
			}
		}
	case *image.RGBA:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				p := y*m.Stride + x*4
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = m.Pix[p], m.Pix[p+1], m.Pix[p+2]
			}
		}
	case *image.NRGBA:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				p := y*m.Stride + x*4
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = m.Pix[p], m.Pix[p+1], m.Pix[p+2]
			}
		}
	case *image.NRGBA64:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				p := y*m.Stride + x*8
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = m.Pix[p], m.Pix[p+2], m.Pix[p+4]
			}
		}
	case *image.RGBA64:
		// Opaque or premultiplied 16-bit RGB: PNG decoding yields it only for
		// opaque data, where premultiplied equals straight.
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				p := y*m.Stride + x*8
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = m.Pix[p], m.Pix[p+2], m.Pix[p+4]
			}
		}
	case *image.Paletted:
		pal := make([][3]uint8, len(m.Palette))
		for i, c := range m.Palette {
			pal[i] = rawRGB(c)
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				idx := int(m.Pix[y*m.Stride+x])
				o := (y*w + x) * 3
				if idx < len(pal) {
					out[o], out[o+1], out[o+2] = pal[idx][0], pal[idx][1], pal[idx][2]
				}
			}
		}
	case *image.CMYK:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				p := y*m.Stride + x*4
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = cmykToRGB(m.Pix[p], m.Pix[p+1], m.Pix[p+2], m.Pix[p+3])
			}
		}
	default:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := rawRGB(img.At(b.Min.X+x, b.Min.Y+y))
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = c[0], c[1], c[2]
			}
		}
	}
	return out
}

// rawRGB returns the stored (unpremultiplied) red, green and blue of a colour.
func rawRGB(c color.Color) [3]uint8 {
	switch v := c.(type) {
	case color.NRGBA:
		return [3]uint8{v.R, v.G, v.B}
	case color.RGBA:
		return [3]uint8{v.R, v.G, v.B}
	case color.Gray:
		return [3]uint8{v.Y, v.Y, v.Y}
	case color.NRGBA64:
		return [3]uint8{uint8(v.R >> 8), uint8(v.G >> 8), uint8(v.B >> 8)}
	}
	r, g, b, _ := c.RGBA()
	return [3]uint8{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
}

func muldiv255(a, b uint32) uint32 {
	t := a*b + 128
	return ((t >> 8) + t) >> 8
}

// cmykToRGB is PIL's cmyk2rgb (Convert.c): R = (255-K) - C*(255-K)/255.
func cmykToRGB(c, m, y, k uint8) (uint8, uint8, uint8) {
	nk := uint32(255 - k)
	f := func(v uint8) uint8 {
		r := int32(nk) - int32(muldiv255(uint32(v), nk))
		return clampU8(r)
	}
	return f(c), f(m), f(y)
}

// decodeGIF decodes the first frame the way Pillow shows it: the frame is
// placed on a logical-screen sized canvas, and palette colours are taken from
// the file (Go's decoder zeroes the transparent entry; PIL keeps its colour).
func decodeGIF(data []byte) (decodedImage, error) {
	cfg, err := gif.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("gif", err)
	}
	// DecodeAll keeps every frame; bound the whole animation by its first frame size.
	if err := checkSize(cfg.Width, cfg.Height, 2); err != nil {
		return decodedImage{}, err
	}
	g, err := gif.DecodeAll(bytes.NewReader(data))
	if err != nil {
		return decodedImage{}, wrapDecodeErr("gif", err)
	}
	if len(g.Image) == 0 {
		return decodedImage{}, errors.New("preprocess: gif without frames")
	}
	first := g.Image[0]
	pal, err := gifFirstFramePalette(data)
	if err != nil || len(pal) == 0 {
		pal = make([][3]uint8, len(first.Palette))
		for i, c := range first.Palette {
			pal[i] = rawRGB(c)
		}
	}
	w, h := cfg.Width, cfg.Height
	out := make([]uint8, w*h*3)
	fb := first.Bounds()
	if fb.Min.X != 0 || fb.Min.Y != 0 || fb.Dx() != w || fb.Dy() != h {
		// Uncovered canvas shows the background colour index.
		bg := int(g.BackgroundIndex)
		if bg < len(pal) {
			for i := 0; i < w*h; i++ {
				out[i*3], out[i*3+1], out[i*3+2] = pal[bg][0], pal[bg][1], pal[bg][2]
			}
		}
	}
	for y := fb.Min.Y; y < fb.Max.Y; y++ {
		if y < 0 || y >= h {
			continue
		}
		for x := fb.Min.X; x < fb.Max.X; x++ {
			if x < 0 || x >= w {
				continue
			}
			idx := int(first.Pix[(y-fb.Min.Y)*first.Stride+(x-fb.Min.X)])
			if idx < len(pal) {
				o := (y*w + x) * 3
				out[o], out[o+1], out[o+2] = pal[idx][0], pal[idx][1], pal[idx][2]
			}
		}
	}
	return decodedImage{w: w, h: h, pix: out, orient: 1}, nil
}

// gifFirstFramePalette reads the colour table that applies to the first
// image: its local table when present, else the global one.
func gifFirstFramePalette(data []byte) ([][3]uint8, error) {
	if len(data) < 13 {
		return nil, errors.New("short gif")
	}
	flags := data[10]
	pos := 13
	readTable := func(n int) ([][3]uint8, error) {
		if pos+3*n > len(data) {
			return nil, errors.New("short gif palette")
		}
		t := make([][3]uint8, n)
		for i := range t {
			t[i] = [3]uint8{data[pos+3*i], data[pos+3*i+1], data[pos+3*i+2]}
		}
		pos += 3 * n
		return t, nil
	}
	var global [][3]uint8
	if flags&0x80 != 0 {
		var err error
		if global, err = readTable(1 << (int(flags&7) + 1)); err != nil {
			return nil, err
		}
	}
	for pos < len(data) {
		switch data[pos] {
		case 0x21: // extension: skip sub-blocks
			pos += 2
			for pos < len(data) && data[pos] != 0 {
				pos += int(data[pos]) + 1
			}
			pos++
		case 0x2C: // image descriptor
			if pos+10 > len(data) {
				return nil, errors.New("short gif descriptor")
			}
			lf := data[pos+9]
			pos += 10
			if lf&0x80 != 0 {
				return readTable(1 << (int(lf&7) + 1))
			}
			return global, nil
		default:
			return global, nil
		}
	}
	return global, nil
}

// webpChunk walks a RIFF/WEBP container.
type webpChunk struct {
	id      string
	payload []byte
}

func webpChunks(data []byte) []webpChunk {
	var out []webpChunk
	pos := 12
	for pos+8 <= len(data) {
		id := string(data[pos : pos+4])
		n := int(binary.LittleEndian.Uint32(data[pos+4:]))
		pos += 8
		if n < 0 || pos+n > len(data) {
			n = len(data) - pos
		}
		out = append(out, webpChunk{id, data[pos : pos+n]})
		pos += n + n&1
	}
	return out
}

func le24(b []byte) int { return int(b[0]) | int(b[1])<<8 | int(b[2])<<16 }

// decodeWebP decodes a still picture or the first frame of an animation. Lossy
// frames go through libwebp's fancy upsampler and colour conversion; lossless
// frames carry RGB directly; alpha is dropped like convert("RGB") does.
func decodeWebP(data []byte) (decodedImage, error) {
	chunks := webpChunks(data)
	var canvasW, canvasH int
	var frame []webpChunk // bitstream chunks of the frame to decode
	var fx, fy int
	orient := 1
	for _, c := range chunks {
		switch c.id {
		case "VP8X":
			if len(c.payload) >= 10 {
				canvasW, canvasH = le24(c.payload[4:])+1, le24(c.payload[7:])+1
			}
		case "EXIF":
			orient = exifOrientation(c.payload)
		case "ANMF":
			if frame == nil && len(c.payload) >= 16 {
				fx, fy = 2*le24(c.payload[0:]), 2*le24(c.payload[3:])
				frame = webpChunks(append([]byte("RIFFxxxxWEBP"), c.payload[16:]...))
			}
		case "VP8 ", "VP8L":
			if frame == nil {
				frame = append(frame, c)
			}
		}
	}
	var bits *webpChunk
	for i := range frame {
		if frame[i].id == "VP8 " || frame[i].id == "VP8L" {
			bits = &frame[i]
			break
		}
	}
	if bits == nil {
		return decodedImage{}, fmt.Errorf("%w: webp without a decodable frame", ErrUnsupported)
	}
	var rgb []uint8
	var w, h int
	if bits.id == "VP8L" {
		cfg, err := vp8l.DecodeConfig(bytes.NewReader(bits.payload))
		if err != nil {
			return decodedImage{}, wrapDecodeErr("webp", err)
		}
		if err := checkSize(cfg.Width, cfg.Height, 4); err != nil {
			return decodedImage{}, err
		}
		img, err := vp8l.Decode(bytes.NewReader(bits.payload))
		if err != nil {
			return decodedImage{}, wrapDecodeErr("webp", err)
		}
		w, h = cfg.Width, cfg.Height
		rgb = imageToRGB(img)
	} else {
		dec := vp8.NewDecoder()
		dec.Init(bytes.NewReader(bits.payload), len(bits.payload))
		fh, err := dec.DecodeFrameHeader()
		if err != nil {
			return decodedImage{}, wrapDecodeErr("webp", err)
		}
		if err := checkSize(fh.Width, fh.Height, 1.5); err != nil {
			return decodedImage{}, err
		}
		img, err := dec.DecodeFrame()
		if err != nil {
			return decodedImage{}, wrapDecodeErr("webp", err)
		}
		w, h = fh.Width, fh.Height
		rgb = webpYCbCrToRGB(img)
	}
	if canvasW == 0 || (canvasW == w && canvasH == h && fx == 0 && fy == 0) {
		return decodedImage{w: w, h: h, pix: rgb, orient: orient}, nil
	}
	// Animation frame smaller than the canvas: transparent black around it.
	if err := checkSize(canvasW, canvasH, 0); err != nil {
		return decodedImage{}, err
	}
	out := make([]uint8, canvasW*canvasH*3)
	for y := 0; y < h && fy+y < canvasH; y++ {
		for x := 0; x < w && fx+x < canvasW; x++ {
			copy(out[((fy+y)*canvasW+fx+x)*3:], rgb[(y*w+x)*3:(y*w+x)*3+3])
		}
	}
	return decodedImage{w: canvasW, h: canvasH, pix: out, orient: orient}, nil
}
