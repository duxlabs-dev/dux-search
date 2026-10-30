package preprocess

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/preprocess/internal/mp3dec"
	"github.com/jfreymuth/oggvorbis"
	"github.com/mewkiz/flac"
)

// Audio decodes the file at path (wav, flac, mp3, ogg/vorbis; the container is
// recognised from the content, not the extension), mixes it down to mono by
// averaging the channels, resamples to 16 kHz and builds the audio tower input
// from the first MaxSeconds. Other formats return ErrUnsupported.
//
// Decoding stops as soon as enough samples for MaxSeconds are available, so a
// long recording costs no more than a short one.
func Audio(path string) (semantic.AudioInput, error) {
	f, err := os.Open(path)
	if err != nil {
		return semantic.AudioInput{}, err
	}
	defer f.Close()
	pcm, rate, err := decodeAudio(f)
	if err != nil {
		return semantic.AudioInput{}, err
	}
	return AudioSamples(pcm, rate)
}

// need returns how many input frames at rate are needed for MaxSamples output
// samples.
func need(rate int) int {
	if rate == SampleRate || !validRate(rate) {
		return MaxSamples // an invalid rate is rejected by AudioSamples
	}
	return newResamplerPlan(rate).inputNeeded(MaxSamples)
}

// newResamplerPlan computes the resampler geometry without building its
// tables.
func newResamplerPlan(rate int) *resampler {
	nyq := float64(min(rate, SampleRate)) / 2
	trans := (resampleStop - resamplePass) * nyq
	seconds := (resampleStopbandDB - 7.95) / (14.36 * trans)
	return &resampler{
		inRate:  rate,
		outRate: SampleRate,
		ratio:   float64(rate) / SampleRate,
		half:    int(math.Ceil(seconds / 2 * float64(rate))),
	}
}

// monoSink collects interleaved frames as their channel mean until limit.
type monoSink struct {
	channels int
	limit    int
	out      []float32
}

func (s *monoSink) full() bool { return len(s.out) >= s.limit }

// addFrame appends one frame; interleaved holds channels values.
func (s *monoSink) addFrame(interleaved []float32) {
	if s.channels == 1 {
		s.out = append(s.out, interleaved[0])
		return
	}
	var sum float32
	for _, v := range interleaved[:s.channels] {
		sum += v
	}
	s.out = append(s.out, sum/float32(s.channels))
}

func decodeAudio(r io.Reader) (pcm []float32, rate int, err error) {
	defer guard("audio", &err)
	br := bufio.NewReaderSize(r, 64<<10)
	head, _ := br.Peek(12)
	switch {
	case len(head) >= 12 && string(head[:4]) == "RIFF" && string(head[8:12]) == "WAVE":
		return decodeWAV(br)
	case len(head) >= 4 && string(head[:4]) == "fLaC":
		return decodeFLAC(br)
	case len(head) >= 4 && string(head[:4]) == "OggS":
		return decodeOgg(br)
	case len(head) >= 3 && (string(head[:3]) == "ID3" || (head[0] == 0xFF && head[1]&0xE0 == 0xE0)):
		return decodeMP3(br)
	}
	return nil, 0, fmt.Errorf("%w: unrecognised audio format", ErrUnsupported)
}

// ---- WAV ----

const (
	wavPCM        = 1
	wavFloat      = 3
	wavALaw       = 6
	wavMuLaw      = 7
	wavExtensible = 0xFFFE
)

func decodeWAV(br *bufio.Reader) ([]float32, int, error) {
	var hdr [12]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, 0, err
	}
	var (
		format, channels, bits, align int
		rate                          int
		haveFmt                       bool
	)
	for {
		var ch [8]byte
		if _, err := io.ReadFull(br, ch[:]); err != nil {
			return nil, 0, fmt.Errorf("preprocess: wav without data chunk: %w", err)
		}
		size := int64(binary.LittleEndian.Uint32(ch[4:]))
		switch string(ch[:4]) {
		case "fmt ":
			if size < 16 {
				return nil, 0, errors.New("preprocess: wav fmt chunk too short")
			}
			buf := make([]byte, size+size&1)
			if _, err := io.ReadFull(br, buf); err != nil {
				return nil, 0, err
			}
			format = int(binary.LittleEndian.Uint16(buf))
			channels = int(binary.LittleEndian.Uint16(buf[2:]))
			rate = int(binary.LittleEndian.Uint32(buf[4:]))
			align = int(binary.LittleEndian.Uint16(buf[12:]))
			bits = int(binary.LittleEndian.Uint16(buf[14:]))
			if format == wavExtensible && size >= 26 {
				format = int(binary.LittleEndian.Uint16(buf[24:])) // sub-format GUID, first two bytes
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, 0, errors.New("preprocess: wav data before fmt")
			}
			return readWAVData(br, size, format, channels, bits, align, rate)
		default:
			if _, err := io.CopyN(io.Discard, br, size+size&1); err != nil {
				return nil, 0, fmt.Errorf("preprocess: wav without data chunk: %w", err)
			}
		}
	}
}

func readWAVData(br *bufio.Reader, size int64, format, channels, bits, align, rate int) ([]float32, int, error) {
	if channels < 1 || rate <= 0 {
		return nil, 0, fmt.Errorf("preprocess: invalid wav header (%d channels, %d Hz)", channels, rate)
	}
	var dec func(b []byte) float32
	bytesPer := bits / 8
	switch {
	case format == wavPCM && bits == 8:
		dec = func(b []byte) float32 { return (float32(b[0]) - 128) / 128 }
	case format == wavPCM && bits == 16:
		dec = func(b []byte) float32 { return float32(int16(binary.LittleEndian.Uint16(b))) / 32768 }
	case format == wavPCM && bits == 24:
		dec = func(b []byte) float32 {
			v := int32(b[0])<<8 | int32(b[1])<<16 | int32(b[2])<<24
			return float32(v>>8) / 8388608
		}
	case format == wavPCM && bits == 32:
		dec = func(b []byte) float32 { return float32(int32(binary.LittleEndian.Uint32(b))) / 2147483648 }
	case format == wavFloat && bits == 32:
		dec = func(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }
	case format == wavFloat && bits == 64:
		dec = func(b []byte) float32 { return float32(math.Float64frombits(binary.LittleEndian.Uint64(b))) }
	case format == wavALaw && bits == 8:
		dec = func(b []byte) float32 { return float32(alawToPCM(b[0])) / 32768 }
	case format == wavMuLaw && bits == 8:
		dec = func(b []byte) float32 { return float32(mulawToPCM(b[0])) / 32768 }
	default:
		return nil, 0, fmt.Errorf("%w: wav format %d with %d bits", ErrUnsupported, format, bits)
	}
	frameBytes := bytesPer * channels
	if align != 0 && align != frameBytes {
		return nil, 0, fmt.Errorf("%w: wav block align %d", ErrUnsupported, align)
	}
	sink := &monoSink{channels: channels, limit: need(rate)}
	// A zero or oversized data length (streamed or truncated files) means
	// "until end of file".
	remaining := size
	if size == 0 || size == 0xFFFFFFFF {
		remaining = math.MaxInt64
	}
	buf := make([]byte, frameBytes*4096)
	vals := make([]float32, channels)
	for remaining > 0 && !sink.full() {
		want := int64(len(buf))
		if want > remaining {
			want = remaining
		}
		n, err := io.ReadFull(br, buf[:want])
		n -= n % frameBytes
		for off := 0; off < n && !sink.full(); off += frameBytes {
			for c := 0; c < channels; c++ {
				vals[c] = dec(buf[off+c*bytesPer:])
			}
			sink.addFrame(vals)
		}
		remaining -= int64(n)
		if err != nil {
			break // io.EOF / io.ErrUnexpectedEOF: truncated file, keep what we have
		}
	}
	if len(sink.out) == 0 {
		return nil, 0, ErrEmpty
	}
	return sink.out, rate, nil
}

func alawToPCM(a byte) int16 {
	a ^= 0x55
	t := int16(a&0x0F) << 4
	seg := (a & 0x70) >> 4
	switch seg {
	case 0:
		t += 8
	case 1:
		t += 0x108
	default:
		t += 0x108
		t <<= seg - 1
	}
	if a&0x80 != 0 {
		return t
	}
	return -t
}

func mulawToPCM(u byte) int16 {
	u = ^u
	t := (int16(u&0x0F) << 3) + 0x84
	t <<= (u & 0x70) >> 4
	if u&0x80 != 0 {
		return 0x84 - t
	}
	return t - 0x84
}

// ---- FLAC ----

func decodeFLAC(br *bufio.Reader) ([]float32, int, error) {
	stream, err := flac.New(br)
	if err != nil {
		return nil, 0, fmt.Errorf("preprocess: decode flac: %w", err)
	}
	info := stream.Info
	channels, bps, rate := int(info.NChannels), int(info.BitsPerSample), int(info.SampleRate)
	if channels < 1 || rate <= 0 || bps < 4 || bps > 32 {
		return nil, 0, fmt.Errorf("%w: flac with %d channels, %d bits", ErrUnsupported, channels, bps)
	}
	scale := float32(1) / float32(int64(1)<<(bps-1))
	sink := &monoSink{channels: channels, limit: need(rate)}
	vals := make([]float32, channels)
	for !sink.full() {
		fr, err := stream.ParseNext()
		if err == io.EOF {
			break
		}
		if err != nil {
			if len(sink.out) > 0 {
				break
			}
			return nil, 0, fmt.Errorf("preprocess: decode flac: %w", err)
		}
		n := int(fr.BlockSize)
		for i := 0; i < n && !sink.full(); i++ {
			for c := 0; c < channels; c++ {
				vals[c] = float32(fr.Subframes[c].Samples[i]) * scale
			}
			sink.addFrame(vals)
		}
	}
	if len(sink.out) == 0 {
		return nil, 0, ErrEmpty
	}
	return sink.out, rate, nil
}

// ---- Ogg Vorbis ----

func decodeOgg(br *bufio.Reader) ([]float32, int, error) {
	// A reader without Seek keeps the decoder streaming (it would otherwise
	// scan the whole file for the length).
	rd, err := oggvorbis.NewReader(struct{ io.Reader }{br})
	if err != nil {
		// Ogg also carries Opus, Speex and FLAC.
		return nil, 0, fmt.Errorf("%w: ogg stream is not vorbis: %v", ErrUnsupported, err)
	}
	channels, rate := rd.Channels(), rd.SampleRate()
	if channels < 1 || rate <= 0 {
		return nil, 0, fmt.Errorf("preprocess: invalid ogg stream (%d channels, %d Hz)", channels, rate)
	}
	sink := &monoSink{channels: channels, limit: need(rate)}
	buf := make([]float32, channels*4096)
	for !sink.full() {
		n, err := rd.Read(buf)
		for off := 0; off+channels <= n && !sink.full(); off += channels {
			sink.addFrame(buf[off : off+channels])
		}
		if err != nil {
			if err != io.EOF && len(sink.out) == 0 {
				return nil, 0, fmt.Errorf("preprocess: decode ogg: %w", err)
			}
			break
		}
	}
	if len(sink.out) == 0 {
		return nil, 0, ErrEmpty
	}
	return sink.out, rate, nil
}

// ---- MP3 ----

// decodeMP3 decodes with mp3dec (a float-output fork of go-mp3, MPEG Layer III) and applies
// the gapless rules of mpg123, which libsndfile (hence the Python reference)
// uses: a leading Xing/Info frame is not audio and is dropped; when it carries
// a frame count, encoder delay plus the 529-sample decoder delay are trimmed
// from the start and the encoder padding from the end (LAME extension values,
// zero when absent). Streams without such a frame are decoded untrimmed.
func decodeMP3(br *bufio.Reader) ([]float32, int, error) {
	if err := skipID3(br); err != nil {
		return nil, 0, fmt.Errorf("preprocess: decode mp3: %w", err)
	}
	tag := mp3Gapless(br)
	dec, err := mp3dec.NewDecoder(br)
	if err != nil {
		return nil, 0, fmt.Errorf("preprocess: decode mp3: %w", err)
	}
	rate := dec.SampleRate()
	// The decoder returns the tag frame like any other: it is the first samples.
	drop := tag.tagFrameSamples + tag.beginSkip()
	sink := &monoSink{channels: 1, limit: need(rate) + drop}
	buf := make([]float32, 2*4096)
	var pair [1]float32
	complete := false
	for !sink.full() {
		n, err := dec.Read(buf)
		for off := 0; off+1 < n && !sink.full(); off += 2 {
			pair[0] = (buf[off] + buf[off+1]) / 2
			sink.addFrame(pair[:])
		}
		if err != nil {
			if err != io.EOF && len(sink.out) == 0 {
				return nil, 0, fmt.Errorf("preprocess: decode mp3: %w", err)
			}
			complete = true
			break
		}
	}
	out := sink.out
	if complete {
		if end := tag.endSample(); end > 0 && end < len(out) {
			out = out[:end]
		}
	}
	if drop > len(out) {
		drop = len(out)
	}
	out = out[drop:]
	if len(out) == 0 {
		return nil, 0, ErrEmpty
	}
	return out, rate, nil
}

// skipID3 discards any ID3v2 tags at the start of the stream.
func skipID3(br *bufio.Reader) error {
	for {
		h, err := br.Peek(10)
		if err != nil || string(h[:3]) != "ID3" {
			return nil
		}
		size := int(h[6]&0x7F)<<21 | int(h[7]&0x7F)<<14 | int(h[8]&0x7F)<<7 | int(h[9]&0x7F)
		n := 10 + size
		if h[5]&0x10 != 0 { // footer present
			n += 10
		}
		if _, err := br.Discard(n); err != nil {
			return err
		}
	}
}

// mp3Tag describes the Xing/Info frame at the start of the stream.
type mp3Tag struct {
	tagFrameSamples int // samples the tag frame decodes to (0: no tag frame)
	frames          int // audio frame count, 0 when unknown
	spf             int // samples per frame
	delay, padding  int // LAME encoder delay and padding
}

// decoderDelay is the delay of a conforming MPEG Layer III decoder.
const mp3DecoderDelay = 529

func (t mp3Tag) beginSkip() int {
	if t.frames == 0 {
		return 0
	}
	return t.delay + mp3DecoderDelay
}

// endSample is the length of the audio in decoder output samples, tag frame
// included: past it only encoder padding follows. 0 when there is no gapless
// information.
func (t mp3Tag) endSample() int {
	if t.frames == 0 {
		return 0
	}
	return t.tagFrameSamples + t.frames*t.spf - t.padding + mp3DecoderDelay
}

// mp3Gapless parses the first frame, without consuming it, for an Xing/Info
// header and its LAME extension.
func mp3Gapless(br *bufio.Reader) mp3Tag {
	head, _ := br.Peek(4096)
	if len(head) < 4 || head[0] != 0xFF || head[1]&0xE0 != 0xE0 {
		return mp3Tag{}
	}
	version := (head[1] >> 3) & 3 // 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
	layer := (head[1] >> 1) & 3   // 1 = Layer III
	if layer != 1 {
		return mp3Tag{}
	}
	spf := 1152
	if version != 3 {
		spf = 576
	}
	mono := head[3]>>6 == 3
	off := 4
	if head[1]&1 == 0 { // CRC present
		off += 2
	}
	switch {
	case version == 3 && !mono:
		off += 32
	case version == 3 && mono, version != 3 && !mono:
		off += 17
	default:
		off += 9
	}
	if off+8 > len(head) {
		return mp3Tag{}
	}
	if t := string(head[off : off+4]); t != "Xing" && t != "Info" {
		if t == "VBRI" || string(head[4+32:4+36]) == "VBRI" {
			return mp3Tag{tagFrameSamples: spf, spf: spf}
		}
		return mp3Tag{}
	}
	tag := mp3Tag{tagFrameSamples: spf, spf: spf}
	flags := binary.BigEndian.Uint32(head[off+4:])
	p := off + 8
	if flags&1 != 0 {
		if p+4 > len(head) {
			return tag
		}
		tag.frames = int(binary.BigEndian.Uint32(head[p:]))
		p += 4
	}
	if flags&2 != 0 {
		p += 4
	}
	if flags&4 != 0 {
		p += 100
	}
	if flags&8 != 0 {
		p += 4
	}
	// LAME extension: 9 bytes encoder string, then at +21 the 12-bit encoder
	// delay and 12-bit padding.
	if p+24 <= len(head) {
		switch string(head[p : p+4]) {
		case "LAME", "Lavc", "Lavf", "L3.9":
			d := head[p+21 : p+24]
			tag.delay = int(d[0])<<4 | int(d[1])>>4
			tag.padding = int(d[1]&0x0F)<<8 | int(d[2])
		}
	}
	return tag
}
