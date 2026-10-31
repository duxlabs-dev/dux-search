package filesidx

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/unicode"
)

const turkish = "Ey Türk gençliği! Birinci vazifen, Türk istiklâlini, Türk cumhuriyetini, ilelebet muhafaza ve müdafaa etmektir."

func TestDecodeTextFormats(t *testing.T) {
	utf16le, _ := unicode.UTF16(unicode.LittleEndian, unicode.UseBOM).NewEncoder().String(turkish)
	utf16be, _ := unicode.UTF16(unicode.BigEndian, unicode.UseBOM).NewEncoder().String(turkish)
	cp1254, _ := charmap.Windows1254.NewEncoder().String(turkish)
	cases := map[string]struct {
		raw    []byte
		legacy bool
	}{
		"utf8":         {[]byte(turkish), false},
		"utf8 bom":     {append([]byte{0xEF, 0xBB, 0xBF}, turkish...), false},
		"utf16 le bom": {[]byte(utf16le), false},
		"utf16 be bom": {[]byte(utf16be), false},
		"windows-1254": {[]byte(cp1254), true},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			legacy := charmap.Windows1254
			got, err := decodeText(c.raw, false, legacy)
			if err != nil || got != turkish {
				t.Fatalf("got %q, %v", got, err)
			}
		})
	}
	// Without the Turkish code page the same bytes decode to something else
	// but never fail: the locale decides.
	got, err := decodeText([]byte(cp1254), false, charmap.Windows1252)
	if err != nil || got == turkish {
		t.Fatalf("code page choice has no effect: %q %v", got, err)
	}
}

func TestDecodeTextRejectsBinary(t *testing.T) {
	bin := bytes.Repeat([]byte{0x00, 0x01, 0x02, 0xFF, 0xFE, 0x10}, 500)
	if _, err := decodeText(bin, false, nil); !errors.Is(err, errBinary) {
		t.Fatalf("binary: %v", err)
	}
	if _, err := decodeText([]byte("text\x00with a NUL byte"), false, nil); !errors.Is(err, errBinary) {
		t.Fatalf("NUL: %v", err)
	}
}

func TestDecodeTextCutInsideRune(t *testing.T) {
	full := strings.Repeat("ğüşiöç", 100)
	for cut := 1; cut < 12; cut++ {
		raw := []byte(full)[:len(full)-cut]
		got, err := decodeText(raw, true, charmap.Windows1254)
		if err != nil || !strings.HasPrefix(full, got) || len(got) < len(full)-cut-4 {
			t.Fatalf("cut %d: %q %v", cut, got[max(0, len(got)-10):], err)
		}
	}
}

func TestReadTextLimitsAndHashes(t *testing.T) {
	p := filepath.Join(t.TempDir(), "big.txt")
	if err := os.WriteFile(p, []byte(strings.Repeat("büyük dosya satırı\n", 10_000)), 0o644); err != nil {
		t.Fatal(err)
	}
	text, hash, err := readText(p, 1000, charmap.Windows1254)
	if err != nil || len(text) > 1000 || len(hash) != 64 || !strings.HasPrefix(text, "büyük") {
		t.Fatalf("%d bytes, hash %q, %v", len(text), hash, err)
	}
	_, hash2, _ := readText(p, 1000, nil)
	if hash2 != hash {
		t.Fatal("hash is not stable")
	}
	if _, _, err := readText(filepath.Join(t.TempDir(), "missing"), 10, nil); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing file: %v", err)
	}
}

func TestLegacyEncodingFromLocale(t *testing.T) {
	for locale, want := range map[string]string{
		"tr_TR.UTF-8": "Windows 1254", "pl_PL": "Windows 1250", "ru_RU.UTF-8": "Windows 1251",
		"en_US.UTF-8": "Windows 1252", "": "Windows 1252", "de_DE@euro": "Windows 1252",
	} {
		if got := legacyEncodingFor(locale).(*charmap.Charmap).String(); !strings.Contains(strings.ReplaceAll(got, "-", " "), want) {
			t.Errorf("%q -> %s, want %s", locale, got, want)
		}
	}
}
