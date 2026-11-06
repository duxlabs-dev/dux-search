package preprocess

import (
	"fmt"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// Audio length limit. processor_config.json sets audio_seq_length = 280, the
// most audio soft tokens one clip may produce (Gemma4Processor caps the
// placeholder count there), at audio_ms_per_token = 40 ms per token: 11.2 s.
// The audio tower emits one token per 4 mel frames (two stride-2
// convolutions), so only the first MaxSeconds of a longer clip are used; a
// longer mel would yield more tokens than the model's context takes.
const (
	AudioSeqLength  = 280
	AudioMsPerToken = 40
	// MaxSamples is the longest waveform used, at SampleRate.
	MaxSamples = AudioSeqLength * AudioMsPerToken * SampleRate / 1000
	// MaxSeconds is MaxSamples in seconds (11.2).
	MaxSeconds = float64(MaxSamples) / SampleRate
)

// Sample rates outside this range are rejected: they are corrupt headers or
// instruments, and the resampler's filter length grows with the rate.
const (
	minSampleRate = 1000
	maxSampleRate = 768000
)

func validRate(r int) bool { return r >= minSampleRate && r <= maxSampleRate }

// AudioSamples builds the audio tower input from mono samples in [-1, 1] at
// sampleRate. Other rates are resampled to 16 kHz; only the first MaxSeconds
// are used.
func AudioSamples(pcm []float32, sampleRate int) (semantic.AudioInput, error) {
	if !validRate(sampleRate) {
		return semantic.AudioInput{}, fmt.Errorf("%w: sample rate %d Hz", ErrUnsupported, sampleRate)
	}
	if len(pcm) == 0 {
		return semantic.AudioInput{}, ErrEmpty
	}
	if sampleRate != SampleRate {
		pcm = resamplerFor(sampleRate).resample(pcm, MaxSamples)
	}
	if len(pcm) > MaxSamples {
		pcm = pcm[:MaxSamples]
	}
	return featuresFromPCM16k(pcm)
}

func featuresFromPCM16k(pcm []float32) (semantic.AudioInput, error) {
	mel, mask, frames := logMel(pcm)
	if AudioSoftTokens(mask) == 0 {
		return semantic.AudioInput{}, fmt.Errorf("%w: clip shorter than one audio token (%d samples)", ErrEmpty, len(pcm))
	}
	return semantic.AudioInput{Mel: mel, Mask: mask, Frames: frames}, nil
}

// AudioSoftTokens is the number of audio soft tokens the tower produces for
// mask, mirroring Gemma4Processor.replace_audio_token: the mask is passed
// through the two stride-2, kernel-3, pad-1 convolutions of the subsampler and
// the surviving true entries are counted.
func AudioSoftTokens(mask []bool) int {
	t := len(mask)
	if t == 0 {
		return 0
	}
	cur := mask
	for i := 0; i < 2; i++ {
		tOut := (t+2-3)/2 + 1
		next := make([]bool, 0, tOut)
		for j := 0; j < len(cur) && len(next) < tOut; j += 2 {
			next = append(next, cur[j])
		}
		cur = next
		t = len(cur)
	}
	n := 0
	for _, v := range cur {
		if v {
			n++
		}
	}
	return n
}
