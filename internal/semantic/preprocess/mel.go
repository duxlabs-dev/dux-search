package preprocess

import (
	"math"
	"math/cmplx"
	"sync"
)

// Gemma4AudioFeatureExtractor constants (processor_config.json,
// feature_extractor).
const (
	SampleRate  = 16000
	FrameLength = 320 // 20 ms
	HopLength   = 160 // 10 ms
	FFTLength   = 512
	NumMel      = 128
	MelFloor    = 1e-3
	minFreq     = 0.0
	maxFreq     = 8000.0
	padMultiple = 128 // pad_to_multiple_of in Gemma4AudioFeatureExtractor.__call__
)

var (
	melOnce    sync.Once
	melFilters [FFTLength/2 + 1][NumMel]float64 // [bin][mel], htk scale, no norm
	hannWindow [FrameLength]float32
	fftTwiddle [FFTLength / 2]complex128
	melLo      [NumMel]int // first non-zero bin of each filter
	melHi      [NumMel]int // one past the last non-zero bin
)

func hzToMel(f float64) float64 { return 2595.0 * math.Log10(1.0+f/700.0) }
func melToHz(m float64) float64 { return 700.0 * (math.Pow(10, m/2595.0) - 1.0) }

// numpyLinspace mirrors np.linspace(start, stop, num) (endpoint included).
func numpyLinspace(start, stop float64, num int) []float64 {
	out := make([]float64, num)
	step := (stop - start) / float64(num-1)
	for i := range out {
		out[i] = start + float64(i)*step
	}
	out[num-1] = stop
	return out
}

func initMel() {
	melOnce.Do(func() {
		// transformers.audio_utils.mel_filter_bank(norm=None, mel_scale="htk",
		// triangularize_in_mel_space=False).
		const bins = FFTLength/2 + 1
		melFreqs := numpyLinspace(hzToMel(minFreq), hzToMel(maxFreq), NumMel+2)
		filterFreqs := make([]float64, len(melFreqs))
		for i, m := range melFreqs {
			filterFreqs[i] = melToHz(m)
		}
		fftFreqs := numpyLinspace(0, float64(SampleRate/2), bins)
		for j := 0; j < NumMel; j++ {
			d0 := filterFreqs[j+1] - filterFreqs[j]
			d1 := filterFreqs[j+2] - filterFreqs[j+1]
			melLo[j], melHi[j] = bins, 0
			for i := 0; i < bins; i++ {
				down := -(filterFreqs[j] - fftFreqs[i]) / d0
				up := (filterFreqs[j+2] - fftFreqs[i]) / d1
				v := math.Max(0, math.Min(down, up))
				melFilters[i][j] = v
				if v != 0 {
					melLo[j] = min(melLo[j], i)
					melHi[j] = max(melHi[j], i+1)
				}
			}
		}
		// Periodic Hann window: np.hanning(N+1)[:-1] cast to float32.
		for n := 0; n < FrameLength; n++ {
			hannWindow[n] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(FrameLength)))
		}
		for k := range fftTwiddle {
			fftTwiddle[k] = cmplx.Exp(complex(0, -2*math.Pi*float64(k)/FFTLength))
		}
	})
}

// fft512 is an in-place radix-2 decimation-in-time FFT of FFTLength points.
func fft512(a *[FFTLength]complex128) {
	n := FFTLength
	// Bit reversal.
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		halfSize := size >> 1
		step := n / size
		for start := 0; start < n; start += size {
			for k := 0; k < halfSize; k++ {
				w := fftTwiddle[k*step]
				u := a[start+k]
				v := a[start+k+halfSize] * w
				a[start+k] = u + v
				a[start+k+halfSize] = u - v
			}
		}
	}
}

// logMel computes the log-mel spectrogram of pcm (mono, 16 kHz) exactly like
// Gemma4AudioFeatureExtractor._extract_spectrogram: the signal is zero-padded
// to a multiple of padMultiple samples, FrameLength/2 zeros are prepended
// (semicausal padding), each 320-sample frame is Hann-windowed, zero-padded to
// 512 points, transformed, reduced to the magnitude, projected on the mel
// bank and log(mel + MelFloor) is taken. Frames that touch padding are masked
// to zero.
func logMel(pcm []float32) (mel []float32, mask []bool, frames int) {
	initMel()
	padded := (len(pcm) + padMultiple - 1) / padMultiple * padMultiple
	left := FrameLength / 2
	total := padded + left
	frames = (total-(FrameLength+1))/HopLength + 1
	if frames <= 0 {
		return nil, nil, 0
	}
	mel = make([]float32, frames*NumMel)
	mask = make([]bool, frames)
	var buf [FFTLength]complex128
	var mag [FFTLength/2 + 1]float64
	for f := 0; f < frames; f++ {
		// A frame is real only if the sample at its (unused) 321st position is.
		end := f*HopLength + FrameLength
		if n := end - left; n >= 0 && n < len(pcm) {
			mask[f] = true
		} else {
			continue
		}
		for i := range buf {
			buf[i] = 0
		}
		for j := 0; j < FrameLength; j++ {
			n := f*HopLength + j - left
			if n >= 0 && n < len(pcm) {
				buf[j] = complex(float64(pcm[n]*hannWindow[j]), 0)
			}
		}
		fft512(&buf)
		for k := range mag {
			mag[k] = cmplx.Abs(buf[k])
		}
		dst := mel[f*NumMel : (f+1)*NumMel]
		for j := 0; j < NumMel; j++ {
			var s float64
			for k := melLo[j]; k < melHi[j]; k++ {
				s += mag[k] * melFilters[k][j]
			}
			dst[j] = float32(math.Log(s + MelFloor))
		}
	}
	return mel, mask, frames
}
