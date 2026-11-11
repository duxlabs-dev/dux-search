package vecstore

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// floatCentered is the exact float64 reference of SearchCentered.
func floatCentered(vecs [][]float32, q []float32) (scores []float64, mean, std float64) {
	dim := len(q)
	cent := make([]float64, dim)
	for _, v := range vecs {
		for i, x := range v {
			cent[i] += float64(x)
		}
	}
	for i := range cent {
		cent[i] /= float64(len(vecs))
	}
	cos := func(a []float32) float64 {
		var ab, aa, bb float64
		for i := range a {
			x, y := float64(a[i])-cent[i], float64(q[i])-cent[i]
			ab, aa, bb = ab+x*y, aa+x*x, bb+y*y
		}
		return ab / math.Sqrt(aa*bb)
	}
	var sum, sumsq float64
	for _, v := range vecs {
		s := cos(v)
		scores = append(scores, s)
		sum += s
		sumsq += s * s
	}
	n := float64(len(vecs))
	mean = sum / n
	return scores, mean, math.Sqrt(sumsq/n - mean*mean)
}

// On real model vectors the int8 centered scan must agree with the exact
// computation, including after replacing and deleting rows (the centroid is
// kept incrementally).
func TestSearchCenteredMatchesFloat(t *testing.T) {
	for _, dim := range []int{256, 512, 768} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			ids, vecs, queries := realCorpus(dim)
			s, _ := openTemp(t, Options{})
			require.NoError(t, s.EnsureCollection("c", dim, testModel))
			require.NoError(t, s.Upsert("c", toVectors(ids, vecs)))
			// replace the first row with the second's vector, delete the last
			require.NoError(t, s.Upsert("c", toVectors(ids[:1], vecs[1:2])))
			require.NoError(t, s.Delete("c", ids[len(ids)-1:]))
			vecs = append([][]float32{vecs[1]}, vecs[1:len(vecs)-1]...)
			ids = ids[:len(ids)-1]

			var maxErr, statErr float64
			top1 := 0
			for _, q := range queries {
				want, mean, std := floatCentered(vecs, q)
				got, st, err := s.SearchCentered("c", q, len(ids))
				require.NoError(t, err)
				require.Len(t, got, len(ids))
				assert.Equal(t, len(ids), st.N)
				byID := map[string]float64{}
				best, bestID := -2.0, ""
				for i, id := range ids {
					byID[id] = want[i]
					if want[i] > best {
						best, bestID = want[i], id
					}
				}
				for _, h := range got {
					maxErr = math.Max(maxErr, math.Abs(float64(h.Score)-byID[h.ID]))
				}
				statErr = math.Max(statErr, math.Max(math.Abs(st.Mean-mean), math.Abs(st.Std-std)))
				if byID[got[0].ID] == best || got[0].ID == bestID {
					top1++
				}
			}
			t.Logf("dim %d: max |score error| %.5f, max stat error %.5f, top-1 %d/%d", dim, maxErr, statErr, top1, len(queries))
			assert.Less(t, maxErr, 0.02)
			assert.Less(t, statErr, 0.005)
			assert.GreaterOrEqual(t, float64(top1), 0.95*float64(len(queries)))
		})
	}
}

func TestSearchCenteredParallelMatchesSerial(t *testing.T) {
	const n, dim = 6000, 256
	data := perturbed(n, dim, 2)
	ids := make([]string, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("v%05d", i)
	}
	serial, _ := openTemp(t, Options{ParallelThreshold: 1 << 30})
	parallel, _ := openTemp(t, Options{ParallelThreshold: 100, Workers: 7})
	for _, s := range []*Store{serial, parallel} {
		require.NoError(t, s.EnsureCollection("c", dim, testModel))
		require.NoError(t, s.Upsert("c", toVectors(ids, data)))
	}
	_, _, queries := realCorpus(dim)
	for _, q := range queries {
		a, sa, err := serial.SearchCentered("c", q, 25)
		require.NoError(t, err)
		b, sb, err := parallel.SearchCentered("c", q, 25)
		require.NoError(t, err)
		requireHitsEqual(t, a, b)
		assert.InDelta(t, sa.Mean, sb.Mean, 1e-9)
		assert.InDelta(t, sa.Std, sb.Std, 1e-9)
	}
}

func TestSearchCenteredSurvivesRestart(t *testing.T) {
	ids, vecs, queries := realCorpus(256)
	s, path := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids, vecs)))
	before, st1, err := s.SearchCentered("c", queries[0], 10)
	require.NoError(t, err)
	require.NoError(t, s.Close())
	s2, err := Open(path, Options{})
	require.NoError(t, err)
	t.Cleanup(func() { _ = s2.Close() })
	after, st2, err := s2.SearchCentered("c", queries[0], 10)
	require.NoError(t, err)
	requireHitsEqual(t, before, after)
	assert.InDelta(t, st1.Mean, st2.Mean, 1e-9)
}
