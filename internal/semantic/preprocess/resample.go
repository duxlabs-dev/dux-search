package preprocess

import (
	"math"
	"sync"
)

// Band-limited resampler to 16 kHz.
//
// The Python reference loads audio with librosa, whose default is soxr's
// "soxr_hq" resampler (libsoxr, linear phase, 20-bit precision, passband to
// 91.3 % and stopband from 100 % of the lower Nyquist frequency). libsoxr is C,
// so this is a Kaiser-windowed-sinc polyphase interpolator designed to the same
// specification: the cut-off sits in the middle of the 0.913..1.0 transition
// band and the window is sized for the same stopband depth. The result is not
// bit-identical to soxr; testdata/README.md records the measured difference.

const (
	resampleStopbandDB = 130.0 // Kaiser attenuation target (soxr_hq: 20 bit)
	resamplePass       = 0.913 // soxr_hq passband end, fraction of lower Nyquist
	resampleStop       = 1.0   // soxr_hq stopband start
	phaseTableMax      = 4096  // largest exact polyphase table; beyond it phases are interpolated
	interpPhases       = 2048
)

type resampler struct {
	inRate, outRate int
	half            int     // taps on each side of the centre, in input samples
	ratio           float64 // inRate / outRate: input samples per output sample
	fcNorm          float64 // cut-off, cycles per input sample
	exact           [][]float64
	phases          int
	interp          [][]float64
}

func bessel0(x float64) float64 {
	sum, term := 1.0, 1.0
	q := x * x / 4
	for k := 1; k < 200; k++ {
		term *= q / float64(k*k)
		sum += term
		if term < sum*1e-17 {
			break
		}
	}
	return sum
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func newResampler(inRate, outRate int) *resampler {
	nyq := float64(min(inRate, outRate)) / 2
	pass, stop := resamplePass*nyq, resampleStop*nyq
	fc := (pass + stop) / 2
	trans := stop - pass
	// Kaiser window length for the transition width, in seconds.
	seconds := (resampleStopbandDB - 7.95) / (14.36 * trans)
	r := &resampler{
		inRate:  inRate,
		outRate: outRate,
		ratio:   float64(inRate) / float64(outRate),
		fcNorm:  fc / float64(inRate),
		half:    int(math.Ceil(seconds / 2 * float64(inRate))),
	}
	g := gcd(inRate, outRate)
	r.phases = outRate / g
	if r.phases <= phaseTableMax {
		r.exact = make([][]float64, r.phases)
		step := inRate / g
		for p := range r.exact {
			// Output sample m = p (mod phases) sits at input position m*in/out;
			// its fractional part is ((p*step) mod phases) / phases.
			frac := float64((p*step)%r.phases) / float64(r.phases)
			r.exact[p] = r.taps(frac)
		}
	} else {
		r.interp = make([][]float64, interpPhases+1)
		for p := range r.interp {
			r.interp[p] = r.taps(float64(p) / interpPhases)
		}
	}
	return r
}

// taps returns the 2*half filter weights for an output sample whose position
// is floor(t) + frac: weight k belongs to input sample floor(t) - half + 1 + k.
func (r *resampler) taps(frac float64) []float64 {
	n := 2 * r.half
	w := make([]float64, n)
	beta := 0.1102 * (resampleStopbandDB - 8.7)
	i0b := bessel0(beta)
	var sum float64
	for k := 0; k < n; k++ {
		u := float64(k-r.half+1) - frac // input sample minus output position
		x := 2 * r.fcNorm * u
		var s float64
		if x == 0 {
			s = 1
		} else {
			s = math.Sin(math.Pi*x) / (math.Pi * x)
		}
		a := u / float64(r.half)
		var win float64
		if a > -1 && a < 1 {
			win = bessel0(beta*math.Sqrt(1-a*a)) / i0b
		}
		w[k] = s * win
		sum += w[k]
	}
	for k := range w {
		w[k] /= sum
	}
	return w
}

// inputNeeded is the number of input samples required to produce n outputs.
func (r *resampler) inputNeeded(n int) int {
	return int(math.Ceil(float64(n)*r.ratio)) + r.half + 1
}

// outputLen is the length of the full-signal result: ceil(in * out / inRate).
func (r *resampler) outputLen(n int) int {
	return int((int64(n)*int64(r.outRate) + int64(r.inRate) - 1) / int64(r.inRate))
}

// resample converts x (mono, inRate) to outRate. Samples outside x are zero.
// At most maxOut samples are produced (0 = all).
func (r *resampler) resample(x []float32, maxOut int) []float32 {
	n := r.outputLen(len(x))
	if maxOut > 0 && n > maxOut {
		n = maxOut
	}
	out := make([]float32, n)
	m64 := int64(r.inRate)
	for m := 0; m < n; m++ {
		// Position in input samples as an exact rational: m*inRate/outRate.
		num := int64(m) * m64
		base := int(num / int64(r.outRate))
		var taps []float64
		if r.exact != nil {
			taps = r.exact[m%r.phases]
		} else {
			frac := float64(num%int64(r.outRate)) / float64(r.outRate) * interpPhases
			p := int(frac)
			f := frac - float64(p)
			a, b := r.interp[p], r.interp[p+1]
			taps = make([]float64, len(a))
			for k := range a {
				taps[k] = a[k]*(1-f) + b[k]*f
			}
		}
		start := base - r.half + 1
		var acc float64
		lo, hi := 0, len(taps)
		if start < 0 {
			lo = -start
		}
		if start+hi > len(x) {
			hi = len(x) - start
		}
		for k := lo; k < hi; k++ {
			acc += taps[k] * float64(x[start+k])
		}
		out[m] = float32(acc)
	}
	return out
}

var (
	resamplersMu sync.Mutex
	resamplers   = map[int]*resampler{}
)

// maxCachedResamplers bounds the table cache: an odd rate costs a table of a
// few MB, and files carry arbitrary rates.
const maxCachedResamplers = 6

// resamplerFor returns the resampler for an input rate; its tables are shared
// between calls (they are immutable once built).
func resamplerFor(rate int) *resampler {
	resamplersMu.Lock()
	defer resamplersMu.Unlock()
	if r, ok := resamplers[rate]; ok {
		return r
	}
	if len(resamplers) >= maxCachedResamplers {
		clear(resamplers)
	}
	r := newResampler(rate, SampleRate)
	resamplers[rate] = r
	return r
}
