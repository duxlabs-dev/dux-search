package preprocess

import (
	"bytes"
	"encoding/binary"
)

// EXIF orientation (tag 0x0112) lookup for the containers PIL reads it from:
// JPEG APP1, PNG eXIf and WebP EXIF. 1 means "as stored" and is also returned
// when the tag is missing or malformed.

func jpegOrientation(data []byte) int {
	pos := 2
	for pos+4 <= len(data) {
		if data[pos] != 0xFF {
			return 1
		}
		marker := data[pos+1]
		if marker == 0xFF { // fill byte
			pos++
			continue
		}
		if marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7) {
			pos += 2
			continue
		}
		if marker == 0xDA || marker == 0xD9 { // image data starts
			return 1
		}
		n := int(binary.BigEndian.Uint16(data[pos+2:]))
		if n < 2 || pos+2+n > len(data) {
			return 1
		}
		if marker == 0xE1 {
			seg := data[pos+4 : pos+2+n]
			if bytes.HasPrefix(seg, []byte("Exif\x00\x00")) {
				if o := exifOrientation(seg[6:]); o != 1 {
					return o
				}
			}
		}
		pos += 2 + n
	}
	return 1
}

func pngOrientation(data []byte) int {
	pos := 8
	for pos+12 <= len(data) {
		n := int(binary.BigEndian.Uint32(data[pos:]))
		typ := string(data[pos+4 : pos+8])
		if n < 0 || pos+12+n > len(data) {
			return 1
		}
		if typ == "eXIf" {
			return exifOrientation(data[pos+8 : pos+8+n])
		}
		pos += 12 + n
	}
	return 1
}

// exifOrientation parses a TIFF block (optionally prefixed by "Exif\0\0") and
// returns the orientation of IFD0.
func exifOrientation(b []byte) int {
	b = bytes.TrimPrefix(b, []byte("Exif\x00\x00"))
	if len(b) < 8 {
		return 1
	}
	var bo binary.ByteOrder
	switch string(b[:2]) {
	case "II":
		bo = binary.LittleEndian
	case "MM":
		bo = binary.BigEndian
	default:
		return 1
	}
	if bo.Uint16(b[2:]) != 42 {
		return 1
	}
	off := int(bo.Uint32(b[4:]))
	if off < 8 || off+2 > len(b) {
		return 1
	}
	n := int(bo.Uint16(b[off:]))
	for i := 0; i < n; i++ {
		e := off + 2 + i*12
		if e+12 > len(b) {
			return 1
		}
		if bo.Uint16(b[e:]) != 0x0112 {
			continue
		}
		typ, count := bo.Uint16(b[e+2:]), bo.Uint32(b[e+4:])
		if typ != 3 || count < 1 { // SHORT
			return 1
		}
		if v := int(bo.Uint16(b[e+8:])); v >= 2 && v <= 8 {
			return v
		}
		return 1
	}
	return 1
}
