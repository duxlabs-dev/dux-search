package preprocess

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

func TestFFTAgainstDFT(t *testing.T) {
	initMel()
	rng := rand.New(rand.NewSource(1))
	var a [FFTLength]complex128
	in := make([]complex128, FFTLength)
	for i := range in {
		in[i] = complex(rng.NormFloat64(), rng.NormFloat64())
		a[i] = in[i]
	}
	fft512(&a)
	for _, k := range []int{0, 1, 2, 17, 255, 256, 257, 400, 511} {
		var want complex128
		for n, v := range in {
			want += v * cmplx.Exp(complex(0, -2*math.Pi*float64(k*n%FFTLength)/FFTLength))
		}
		if cmplx.Abs(a[k]-want) > 1e-9 {
			t.Errorf("bin %d: %v want %v", k, a[k], want)
		}
	}
}

func TestMelFilterBankShape(t *testing.T) {
	initMel()
	// Every filter is a triangle: non-negative, at most 1, peak 1 only if a bin
	// sits on the centre. The documented all-zero top filter (a 8 kHz Nyquist
	// edge falling between FFT bins) must be the only empty one.
	empty := 0
	for j := 0; j < NumMel; j++ {
		var peak float64
		for k := 0; k <= FFTLength/2; k++ {
			v := melFilters[k][j]
			if v < 0 || v > 1 {
				t.Fatalf("filter %d bin %d = %v", j, k, v)
			}
			peak = math.Max(peak, v)
		}
		if peak == 0 {
			empty++
		}
	}
	t.Logf("%d empty filters", empty)
	if empty > 1 {
		t.Errorf("%d empty mel filters", empty)
	}
}

func TestLogMelFrames(t *testing.T) {
	// 9990 samples are padded to 10112 (a multiple of 128): (10112+160-321)/160+1
	// = 63 frames. Frame f is real while f*160+160 < 9990, so the last one
	// (f = 62: 10080) only touches padding and is masked to zero.
	pcm := sine(SampleRate, 440, 9990)
	mel, mask, frames := logMel(pcm)
	if frames != 63 || len(mel) != 63*NumMel || len(mask) != 63 {
		t.Fatalf("frames=%d len(mel)=%d len(mask)=%d", frames, len(mel), len(mask))
	}
	valid := 0
	for _, m := range mask {
		if m {
			valid++
		}
	}
	if valid != 62 || mask[62] {
		t.Errorf("valid frames %d (last masked: %v), want 62", valid, !mask[62])
	}
	for _, v := range mel[62*NumMel:] {
		if v != 0 {
			t.Fatalf("masked frame not zero")
		}
	}
	// A tone frame has its energy in the filter around 440 Hz.
	row := mel[10*NumMel : 11*NumMel]
	best := 0
	for j, v := range row {
		if v > row[best] {
			best = j
		}
	}
	if centre := melToHz(hzToMel(0) + (hzToMel(maxFreq)-hzToMel(0))*float64(best+1)/float64(NumMel+1)); math.Abs(centre-440) > 80 {
		t.Errorf("tone peaks in filter %d (centre %.0f Hz)", best, centre)
	}
}
