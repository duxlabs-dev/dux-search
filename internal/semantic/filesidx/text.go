package filesidx

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	xunicode "golang.org/x/text/encoding/unicode"
)

var errBinary = errors.New("binary content")

// readText reads at most max bytes of a text file and returns it as UTF-8
// together with the sha256 of the bytes read. It understands UTF-8, UTF-16
// (byte order mark) and falls back to a legacy single-byte code page for
// files that are not valid UTF-8; binary files give errBinary.
func readText(path string, max int64, legacy encoding.Encoding) (text, hash string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, max))
	if err != nil {
		return "", "", err
	}
	sum := sha256.Sum256(raw)
	text, err = decodeText(raw, int64(len(raw)) == max, legacy)
	return text, hex.EncodeToString(sum[:]), err
}

// decodeText converts raw bytes to UTF-8; cut says the data was truncated at
// the size limit, so a rune split at the end is not an error.
func decodeText(raw []byte, cut bool, legacy encoding.Encoding) (string, error) {
	switch {
	case bytes.HasPrefix(raw, []byte{0xEF, 0xBB, 0xBF}):
		raw = raw[3:]
	case bytes.HasPrefix(raw, []byte{0xFF, 0xFE}), bytes.HasPrefix(raw, []byte{0xFE, 0xFF}):
		dec := xunicode.UTF16(xunicode.LittleEndian, xunicode.UseBOM).NewDecoder()
		if raw[0] == 0xFE {
			dec = xunicode.UTF16(xunicode.BigEndian, xunicode.UseBOM).NewDecoder()
		}
		if cut && len(raw)%2 == 1 {
			raw = raw[:len(raw)-1]
		}
		out, err := dec.Bytes(raw)
		if err != nil {
			return "", errBinary
		}
		return cleanText(string(out))
	}
	if bytes.IndexByte(raw[:min(len(raw), 8192)], 0) >= 0 {
		return "", errBinary
	}
	if cut {
		raw = trimPartialRune(raw)
	}
	if utf8.Valid(raw) {
		return cleanText(string(raw))
	}
	if legacy == nil {
		legacy = charmap.Windows1252
	}
	out, err := legacy.NewDecoder().Bytes(raw)
	if err != nil {
		return "", errBinary
	}
	return cleanText(string(out))
}

// trimPartialRune drops an incomplete multi-byte rune at the end.
func trimPartialRune(b []byte) []byte {
	for i := 1; i <= utf8.UTFMax-1 && i <= len(b); i++ {
		if r, size := utf8.DecodeLastRune(b); r == utf8.RuneError && size <= 1 {
			if !utf8.RuneStart(b[len(b)-i]) {
				continue
			}
			return b[:len(b)-i]
		}
		break
	}
	return b
}

// cleanText rejects text that is mostly control characters.
func cleanText(s string) (string, error) {
	total, ctrl := 0, 0
	for _, r := range s {
		total++
		if r == utf8.RuneError || (unicode.IsControl(r) && r != '\n' && r != '\r' && r != '\t' && r != '\f') {
			ctrl++
		}
	}
	if total > 0 && ctrl*50 > total { // more than 2%
		return "", errBinary
	}
	return strings.ToValidUTF8(s, ""), nil
}
