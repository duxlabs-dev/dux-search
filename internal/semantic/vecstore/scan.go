package vecstore

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// maxDim keeps the int32 accumulator of the dot product exact:
// 127*127*maxDim < 2^31.
const maxDim = 100_000

// maxKeyLen is bbolt's key size limit; ids and collection names live in keys.
const maxKeyLen = 32768

var errZeroVector = errors.New("zero vector")

// quantize maps v to symmetric int8: q = round(v/scale), scale = max|v|/127.
// inv is 1/||q||, so dot(qa,qb)*inv_a*inv_b is the exact cosine of the two
// quantized vectors, independent of either scale.
func quantize(v []float32) (q []int8, scale, inv float32, err error) {
	var maxAbs float32
	for _, x := range v {
		if x != x || math.IsInf(float64(x), 0) {
			return nil, 0, 0, errors.New("non-finite value")
		}
		if a := abs32(x); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		return nil, 0, 0, errZeroVector
	}
	scale = maxAbs / 127
	q = make([]int8, len(v))
	var sumsq int64
	for i, x := range v {
		r := int(math.Round(float64(x / scale)))
		if r > 127 {
			r = 127
		} else if r < -127 {
			r = -127
		}
		q[i] = int8(r)
		sumsq += int64(r * r)
	}
	return q, scale, float32(1 / math.Sqrt(float64(sumsq))), nil
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// dotInt8 is the inner loop of every search. Four independent accumulators keep
// the multiply-adds pipelined; the slice re-slicing lets the compiler drop the
// bounds checks.
func dotInt8(a, b []int8) int32 {
	n := len(a)
	b = b[:n]
	var s0, s1, s2, s3 int32
	i := 0
	for ; i+4 <= n; i += 4 {
		x, y := a[i:i+4:i+4], b[i:i+4:i+4]
		s0 += int32(x[0]) * int32(y[0])
		s1 += int32(x[1]) * int32(y[1])
		s2 += int32(x[2]) * int32(y[2])
		s3 += int32(x[3]) * int32(y[3])
	}
	for ; i < n; i++ {
		s0 += int32(a[i]) * int32(b[i])
	}
	return s0 + s1 + s2 + s3
}

type scored struct {
	row   int
	score float32
}

// topK keeps the k best rows with a binary min-heap (worst on top).
type topK struct {
	k int
	h []scored
}

func (t *topK) offer(row int, score float32) {
	if len(t.h) < t.k {
		t.h = append(t.h, scored{row, score})
		t.up(len(t.h) - 1)
		return
	}
	if score <= t.h[0].score {
		return
	}
	t.h[0] = scored{row, score}
	t.down(0)
}

func (t *topK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if t.h[p].score <= t.h[i].score {
			return
		}
		t.h[p], t.h[i] = t.h[i], t.h[p]
		i = p
	}
}

func (t *topK) down(i int) {
	n := len(t.h)
	for {
		l, r, m := 2*i+1, 2*i+2, i
		if l < n && t.h[l].score < t.h[m].score {
			m = l
		}
		if r < n && t.h[r].score < t.h[m].score {
			m = r
		}
		if m == i {
			return
		}
		t.h[i], t.h[m] = t.h[m], t.h[i]
		i = m
	}
}

// scanRange scores rows [lo,hi) against the quantized query.
func (c *collection) scanRange(q []int8, qinv float32, k, lo, hi int) []scored {
	t := topK{k: k, h: make([]scored, 0, k)}
	dim := c.dim
	for r := lo; r < hi; r++ {
		d := dotInt8(q, c.data[r*dim:(r+1)*dim])
		t.offer(r, float32(d)*c.inv[r]*qinv)
	}
	return t.h
}

// Search returns the k most similar vectors by cosine similarity, best first.
// The query is quantized like the stored vectors, so scores carry the same small
// quantization error on both sides. An unknown collection yields no hits.
func (s *Store) Search(name string, query []float32, k int) ([]semantic.Hit, error) {
	if k <= 0 {
		return nil, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	c := s.cols[name]
	if c == nil {
		return nil, nil
	}
	if len(query) != c.dim {
		return nil, fmt.Errorf("%w: query has %d values, collection %q has %d", ErrDimMismatch, len(query), name, c.dim)
	}
	q, _, qinv, err := quantize(query)
	if err != nil {
		return nil, fmt.Errorf("%w: query: %v", ErrInvalidVector, err)
	}
	n := c.len()
	if n == 0 {
		return nil, nil
	}
	if k > n {
		k = n
	}

	var best []scored
	workers := s.opt.Workers
	if n < s.opt.ParallelThreshold || workers < 2 {
		best = c.scanRange(q, qinv, k, 0, n)
	} else {
		if workers > n {
			workers = n
		}
		parts := make([][]scored, workers)
		var wg sync.WaitGroup
		step := (n + workers - 1) / workers
		for w := 0; w < workers; w++ {
			lo, hi := w*step, min((w+1)*step, n)
			if lo >= hi {
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				parts[w] = c.scanRange(q, qinv, k, lo, hi)
			}()
		}
		wg.Wait()
		merged := topK{k: k, h: make([]scored, 0, k)}
		for _, p := range parts {
			for _, e := range p {
				merged.offer(e.row, e.score)
			}
		}
		best = merged.h
	}

	// Ties break on id so results do not depend on row order or worker count.
	sort.Slice(best, func(i, j int) bool {
		if best[i].score != best[j].score {
			return best[i].score > best[j].score
		}
		return c.ids[best[i].row] < c.ids[best[j].row]
	})
	hits := make([]semantic.Hit, len(best))
	for i, e := range best {
		hits[i] = semantic.Hit{ID: c.ids[e.row], Score: e.score}
	}
	return hits, nil
}
