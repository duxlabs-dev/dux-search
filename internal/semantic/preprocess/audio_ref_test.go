package preprocess

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

type audioRef struct {
	File        string `json:"file"`
	SrcRate     int    `json:"src_rate"`
	SrcChannels int    `json:"src_channels"`
	Samples16k  int    `json:"samples_16k"`
	Frames      int    `json:"frames"`
	ValidFrames int    `json:"valid_frames"`
	SoftTokens  int    `json:"soft_tokens"`
	MelFile     string `json:"mel_file"`
	MaskFile    string `json:"mask_file"`
}

type audioRefs struct {
	Versions   map[string]string `json:"versions"`
	MaxSamples int               `json:"max_samples"`
	Clips      []audioRef        `json:"clips"`
}

// audioTolerance returns the accepted log-mel difference to transformers for a
// clip: max absolute, and mean absolute over the real frames, in natural-log
// units of the mel energy. The classes are, from the tightest:
//
//   - 16 kHz wav/flac: no resampling and a bit-exact decoder, so this tests the
//     feature extractor alone (measured: identical to 4 decimals).
//   - other wav/flac: adds the resampler. libsoxr (soxr_hq) is not available in
//     pure Go; the Kaiser polyphase resampler in resample.go follows its
//     specification and the difference is at most 0.063 (mean 0.0005).
//   - ogg/vorbis: the decoder (jfreymuth/oggvorbis) matches libvorbis to float
//     rounding except for one sample in Help_Sortable... (0.016 off), which
//     shows as the 0.25 maximum there.
//   - mp3: decoded by the float-output fork of go-mp3 in internal/mp3dec,
//     aligned to libsndfile's gapless output; the difference is the
//     resampler's plus decoder float rounding (max 0.033, mean 0.0003).
func audioTolerance(e audioRef) (maxAbs, meanAbs float64) {
	lossless := strings.HasSuffix(e.File, ".wav") || strings.HasSuffix(e.File, ".flac")
	switch {
	case lossless && e.SrcRate == SampleRate:
		return 1e-3, 1e-5
	case lossless:
		return 0.1, 0.002
	case strings.HasSuffix(e.File, ".ogg"):
		return 0.3, 0.002
	}
	return 0.1, 0.002 // mp3
}

func TestAudioReference(t *testing.T) {
	var ref audioRefs
	readJSON(t, "audio.json", &ref)
	if ref.MaxSamples != MaxSamples {
		t.Fatalf("reference was generated with max_length %d, MaxSamples is %d", ref.MaxSamples, MaxSamples)
	}
	if len(ref.Clips) < 6 {
		t.Fatalf("only %d reference clips", len(ref.Clips))
	}
	for _, e := range ref.Clips {
		e := e
		t.Run(e.File, func(t *testing.T) {
			got, err := Audio(filepath.Join("testdata", "audio", e.File))
			if err != nil {
				t.Fatal(err)
			}
			wantMel := f32s(readGz(t, e.MelFile))
			wantMask := readGz(t, e.MaskFile)
			if got.Frames != e.Frames || len(got.Mel) != e.Frames*NumMel || len(got.Mask) != e.Frames {
				t.Fatalf("frames: got %d (mel %d, mask %d), want %d", got.Frames, len(got.Mel), len(got.Mask), e.Frames)
			}
			if len(wantMel) != e.Frames*NumMel || len(wantMask) != e.Frames {
				t.Fatalf("bad reference sizes")
			}
			valid := 0
			var maskDiff []int
			for i, m := range got.Mask {
				if m {
					valid++
				}
				if m != (wantMask[i] != 0) {
					maskDiff = append(maskDiff, i)
				}
			}
			if valid != e.ValidFrames {
				t.Errorf("valid frames: got %d want %d", valid, e.ValidFrames)
			}
			if tok := AudioSoftTokens(got.Mask); tok != e.SoftTokens {
				t.Errorf("soft tokens: got %d want %d", tok, e.SoftTokens)
			}
			// The mask depends on the sample count only; a clip whose decoded
			// length differs from the reference by a sample can flip the last
			// frame, which is reported but only fails beyond that.
			if len(maskDiff) > 0 {
				t.Errorf("mask differs at frames %v", maskDiff)
			}
			var maxd, sum float64
			var n int
			for f := 0; f < e.Frames; f++ {
				if !got.Mask[f] {
					for j := 0; j < NumMel; j++ {
						if got.Mel[f*NumMel+j] != 0 {
							t.Fatalf("masked frame %d is not zero", f)
						}
					}
					continue
				}
				for j := 0; j < NumMel; j++ {
					d := math.Abs(float64(got.Mel[f*NumMel+j]) - float64(wantMel[f*NumMel+j]))
					maxd = math.Max(maxd, d)
					sum += d
					n++
				}
			}
			mean := sum / float64(n)
			tolMax, tolMean := audioTolerance(e)
			t.Logf("%d Hz x%d -> %d frames (%d tokens): max |dlog-mel| %.4f, mean %.6f", e.SrcRate, e.SrcChannels, e.Frames, e.SoftTokens, maxd, mean)
			if maxd > tolMax || mean > tolMean {
				t.Errorf("log-mel differs from transformers: max %.4f (limit %.4f), mean %.6f (limit %.6f)", maxd, tolMax, mean, tolMean)
			}
		})
	}
}
