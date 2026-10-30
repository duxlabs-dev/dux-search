package preprocess

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// wavFile builds a RIFF/WAVE file from a format description and raw data.
func wavFile(format, channels, rate, bits int, data []byte, extensible bool, extra ...[]byte) []byte {
	var fmtc bytes.Buffer
	tag := format
	if extensible {
		tag = wavExtensible
	}
	binary.Write(&fmtc, binary.LittleEndian, uint16(tag))
	binary.Write(&fmtc, binary.LittleEndian, uint16(channels))
	binary.Write(&fmtc, binary.LittleEndian, uint32(rate))
	binary.Write(&fmtc, binary.LittleEndian, uint32(rate*channels*bits/8))
	binary.Write(&fmtc, binary.LittleEndian, uint16(channels*bits/8))
	binary.Write(&fmtc, binary.LittleEndian, uint16(bits))
	if extensible {
		binary.Write(&fmtc, binary.LittleEndian, uint16(22))
		binary.Write(&fmtc, binary.LittleEndian, uint16(bits))
		binary.Write(&fmtc, binary.LittleEndian, uint32(3))
		binary.Write(&fmtc, binary.LittleEndian, uint16(format))
		fmtc.Write(make([]byte, 14))
	}
	var b bytes.Buffer
	chunk := func(id string, payload []byte) {
		b.WriteString(id)
		binary.Write(&b, binary.LittleEndian, uint32(len(payload)))
		b.Write(payload)
		if len(payload)&1 == 1 {
			b.WriteByte(0)
		}
	}
	for _, e := range extra {
		chunk("LIST", e)
	}
	chunk("fmt ", fmtc.Bytes())
	chunk("data", data)
	out := append([]byte("RIFF\x00\x00\x00\x00WAVE"), b.Bytes()...)
	binary.LittleEndian.PutUint32(out[4:], uint32(len(out)-8))
	return out
}

func TestWAVFormats(t *testing.T) {
	le16 := func(vs ...int16) []byte {
		var b bytes.Buffer
		for _, v := range vs {
			binary.Write(&b, binary.LittleEndian, v)
		}
		return b.Bytes()
	}
	cases := []struct {
		name string
		file []byte
		rate int
		want []float32
	}{
		{"u8 mono", wavFile(1, 1, 16000, 8, []byte{128, 255, 0}, false), 16000, []float32{0, 127.0 / 128, -1}},
		{"s16 stereo mean", wavFile(1, 2, 16000, 16, le16(16384, -16384, 32767, 0), false), 16000, []float32{0, 32767.0 / 65536}},
		{"s24 mono", wavFile(1, 1, 8000, 24, []byte{0, 0, 0x40, 0xFF, 0xFF, 0xFF}, false), 8000, []float32{0.5, -1.0 / 8388608}},
		{"s32 mono", wavFile(1, 1, 8000, 32, []byte{0, 0, 0, 0x80, 0, 0, 0, 0x40}, false), 8000, []float32{-1, 0.5}},
		{"f32 mono", wavFile(3, 1, 8000, 32, binary.LittleEndian.AppendUint32(nil, math.Float32bits(0.25)), false), 8000, []float32{0.25}},
		{"f64 stereo", wavFile(3, 2, 8000, 64, append(binary.LittleEndian.AppendUint64(nil, math.Float64bits(0.5)), binary.LittleEndian.AppendUint64(nil, math.Float64bits(-0.25))...), false), 8000, []float32{0.125}},
		{"extensible s16 with LIST chunk", wavFile(1, 1, 22050, 16, le16(1000), true, []byte("abc")), 22050, []float32{1000.0 / 32768}},
		{"a-law", wavFile(wavALaw, 1, 8000, 8, []byte{0xD5, 0x55}, false), 8000, []float32{8.0 / 32768, -8.0 / 32768}},
		{"mu-law", wavFile(wavMuLaw, 1, 8000, 8, []byte{0xFF, 0x7F}, false), 8000, []float32{0, 0}},
	}
	for _, c := range cases {
		got, rate, err := decodeAudio(bytes.NewReader(c.file))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if rate != c.rate || len(got) != len(c.want) {
			t.Errorf("%s: rate %d len %d, want %d %d", c.name, rate, len(got), c.rate, len(c.want))
			continue
		}
		for i := range got {
			if math.Abs(float64(got[i]-c.want[i])) > 1e-7 {
				t.Errorf("%s: sample %d = %v, want %v", c.name, i, got[i], c.want[i])
			}
		}
	}
}

func TestWAVTruncatedAndStreamed(t *testing.T) {
	data := make([]byte, 400)
	for i := range data {
		data[i] = byte(i)
	}
	f := wavFile(1, 1, 16000, 16, data, false)
	// Truncated in the middle of the data chunk: keep what is there.
	got, _, err := decodeAudio(bytes.NewReader(f[:len(f)-101]))
	if err != nil || len(got) != 149 {
		t.Errorf("truncated: %d samples, %v", len(got), err)
	}
	// Data length 0xFFFFFFFF (streamed): read to end of file.
	g := append([]byte(nil), f...)
	binary.LittleEndian.PutUint32(g[len(g)-len(data)-4:], 0xFFFFFFFF)
	got, _, err = decodeAudio(bytes.NewReader(g))
	if err != nil || len(got) != 200 {
		t.Errorf("streamed: %d samples, %v", len(got), err)
	}
	if _, _, err := decodeAudio(bytes.NewReader(f[:20])); err == nil {
		t.Errorf("header only: expected error")
	}
	if _, _, err := decodeAudio(bytes.NewReader(wavFile(1, 1, 16000, 16, nil, false))); !errors.Is(err, ErrEmpty) {
		t.Errorf("no samples: %v", err)
	}
	if _, _, err := decodeAudio(bytes.NewReader(wavFile(1, 1, 16000, 12, data, false))); !errors.Is(err, ErrUnsupported) {
		t.Errorf("12-bit: %v", err)
	}
}

func TestAudioUnsupported(t *testing.T) {
	for name, b := range map[string][]byte{
		"empty":  nil,
		"text":   []byte("this is not audio at all"),
		"opus":   []byte("OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"),
		"matrox": []byte("FORM\x00\x00\x00\x00AIFF"),
	} {
		if _, _, err := decodeAudio(bytes.NewReader(b)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, _, err := decodeAudio(bytes.NewReader([]byte("FORM\x00\x00\x00\x00AIFF"))); !errors.Is(err, ErrUnsupported) {
		t.Errorf("aiff: %v", err)
	}
}

func TestAudioLimitStopsDecoding(t *testing.T) {
	// 40 s of 16 kHz silence: only MaxSamples are collected.
	data := make([]byte, 2*16000*40)
	got, _, err := decodeAudio(bytes.NewReader(wavFile(1, 1, 16000, 16, data, false)))
	if err != nil || len(got) != MaxSamples {
		t.Fatalf("got %d samples (%v), want %d", len(got), err, MaxSamples)
	}
}

func TestAudioSamplesErrors(t *testing.T) {
	if _, err := AudioSamples(nil, 16000); !errors.Is(err, ErrEmpty) {
		t.Errorf("empty: %v", err)
	}
	if _, err := AudioSamples(make([]float32, 100), 16000); !errors.Is(err, ErrEmpty) {
		t.Errorf("too short: %v", err)
	}
	for _, rate := range []int{0, -5, 999, 768001, 4_000_000_000} {
		if _, err := AudioSamples(make([]float32, 1600), rate); !errors.Is(err, ErrUnsupported) {
			t.Errorf("rate %d: %v", rate, err)
		}
	}
}

func TestAudioSoftTokens(t *testing.T) {
	// 1120 real frames -> 280 tokens (the model limit).
	m := make([]bool, 1120)
	for i := range m {
		m[i] = true
	}
	if n := AudioSoftTokens(m); n != AudioSeqLength {
		t.Errorf("1120 frames -> %d tokens, want %d", n, AudioSeqLength)
	}
	if AudioSoftTokens(nil) != 0 || AudioSoftTokens([]bool{true}) != 1 {
		t.Errorf("edge cases")
	}
}
