package preprocess

import (
	"math"
	"testing"
)

func sine(rate int, hz float64, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(0.5 * math.Sin(2*math.Pi*hz*float64(i)/float64(rate)))
	}
	return x
}

func rms(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(len(x)))
}

// TestResamplerResponse checks the filter against the soxr_hq specification
// it stands in for: flat to 91.3 % of the lower Nyquist frequency, strongly
// attenuated from the Nyquist frequency up, unity DC gain, no delay.
func TestResamplerResponse(t *testing.T) {
	for _, in := range []int{8000, 11025, 22050, 24000, 32000, 44100, 48000, 96000} {
		r := newResampler(in, SampleRate)
		nyq := float64(min(in, SampleRate)) / 2
		trim := func(y []float32) []float32 { return y[len(y)/8 : len(y)-len(y)/8] }
		ref := rms(sine(in, 1000, in))
		for _, c := range []struct {
			hz       float64
			minRatio float64
			maxRatio float64
		}{
			{0.1 * nyq, 0.999, 1.001},
			{0.5 * nyq, 0.999, 1.001},
			{0.9 * nyq, 0.995, 1.005},
			{1.02 * nyq, 0, 1e-3},
			{1.3 * nyq, 0, 1e-5},
		} {
			if c.hz >= float64(in)/2 {
				continue // above the input Nyquist: not representable
			}
			x := sine(in, c.hz, 2*in)
			y := trim(r.resample(x, 0))
			ratio := rms(y) / rms(trim(x))
			_ = ref
			if ratio < c.minRatio || ratio > c.maxRatio {
				t.Errorf("%d Hz -> 16 kHz, tone %.0f Hz: gain %.5f outside [%g, %g]", in, c.hz, ratio, c.minRatio, c.maxRatio)
			}
		}
		// Linear phase and no delay: the output is aligned with the input.
		x := sine(in, 0.1*nyq, in)
		y := r.resample(x, 0)
		var worst float64
		for m := len(y) / 4; m < 3*len(y)/4; m++ {
			want := 0.5 * math.Sin(2*math.Pi*0.1*nyq*float64(m)/SampleRate)
			worst = math.Max(worst, math.Abs(float64(y[m])-want))
		}
		if worst > 2e-4 {
			t.Errorf("%d Hz: output misaligned with input, worst sample error %.2e", in, worst)
		}
	}
}

func TestResamplerLength(t *testing.T) {
	r := newResampler(44100, SampleRate)
	for _, n := range []int{1, 100, 44100, 44101} {
		want := int(math.Ceil(float64(n) * 16000 / 44100))
		if got := len(r.resample(make([]float32, n), 0)); got != want {
			t.Errorf("%d samples -> %d, want %d", n, got, want)
		}
	}
	if got := len(r.resample(make([]float32, 100000), 1000)); got != 1000 {
		t.Errorf("maxOut ignored: %d", got)
	}
}

// Odd rates need more than phaseTableMax phases and use the interpolated table.
func TestResamplerInterpolatedPhases(t *testing.T) {
	r := newResampler(44111, SampleRate)
	if r.interp == nil {
		t.Fatalf("expected the interpolated phase table for 44111 Hz")
	}
	x := sine(44111, 3000, 44111)
	y := r.resample(x, 0)
	var worst float64
	for m := len(y) / 4; m < 3*len(y)/4; m++ {
		worst = math.Max(worst, math.Abs(float64(y[m])-0.5*math.Sin(2*math.Pi*3000*float64(m)/SampleRate)))
	}
	if worst > 5e-4 {
		t.Errorf("worst error %.2e", worst)
	}
}
