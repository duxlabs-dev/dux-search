package vecstore

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// ScoreStats describes the scores of every row of a collection for one query.
type ScoreStats struct {
	N    int
	Mean float64
	Std  float64
}

// Z is how many standard deviations score lies above the collection mean.
func (st ScoreStats) Z(score float32) float64 {
	if st.Std == 0 {
		return 0
	}
	return (float64(score) - st.Mean) / st.Std
}

// SearchCentered ranks rows by the cosine of query and row after both are moved
// by the collection centroid, and returns the score distribution over all rows.
//
// Embeddings of one modality share a large common direction, so raw cosines sit
// in a narrow band and a few rows near that direction (hubs) score high for
// every query. Removing the centroid is the standard remedy for hubness
// (Suzuki et al., "Centering similarity measures to reduce hubs", EMNLP 2013);
// measured on real screenshots it moved the expected image from rank 47 to 3
// and from 116 to 13 for queries the raw cosine missed. The statistics let
// callers compare collections of different modalities on one scale.
func (s *Store) SearchCentered(name string, query []float32, k int) ([]semantic.Hit, ScoreStats, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.checkOpen(); err != nil {
		return nil, ScoreStats{}, err
	}
	c := s.cols[name]
	if c == nil || c.len() == 0 || k <= 0 {
		return nil, ScoreStats{}, nil
	}
	if len(query) != c.dim {
		return nil, ScoreStats{}, fmt.Errorf("%w: query has %d values, collection %q has %d", ErrDimMismatch, len(query), name, c.dim)
	}
	n := c.len()
	k = min(k, n)

	cent := make([]float32, c.dim)
	var cc float64
	for i, v := range c.sum {
		cent[i] = float32(v / float64(n))
		cc += float64(cent[i]) * float64(cent[i])
	}
	qc := make([]float32, c.dim)
	var qcc, qcn float64
	for i := range qc {
		qc[i] = query[i] - cent[i]
		qcc += float64(cent[i]) * float64(qc[i])
		qcn += float64(qc[i]) * float64(qc[i])
	}
	qcn = math.Sqrt(qcn)
	c8, cScale, _, errC := quantize(cent)
	q8, qScale, _, errQ := quantize(qc)
	if errQ != nil || qcn == 0 {
		return nil, ScoreStats{}, fmt.Errorf("%w: query equals the collection centroid", ErrInvalidVector)
	}
	if errC != nil && !errors.Is(errC, errZeroVector) {
		return nil, ScoreStats{}, errC
	}

	scan := func(lo, hi int) ([]scored, float64, float64) {
		t := topK{k: k, h: make([]scored, 0, k)}
		var sum, sumsq float64
		for r := lo; r < hi; r++ {
			row := c.data[r*c.dim : (r+1)*c.dim]
			sc := float64(c.scale[r])
			xn2 := sc / float64(c.inv[r])
			xn2 *= xn2
			var xc float64
			if c8 != nil {
				xc = sc * float64(cScale) * float64(dotInt8(row, c8))
			}
			den := xn2 - 2*xc + cc
			score := 0.0
			if den > 0 {
				num := sc*float64(qScale)*float64(dotInt8(row, q8)) - qcc
				score = num / (math.Sqrt(den) * qcn)
			}
			sum += score
			sumsq += score * score
			t.offer(r, float32(score))
		}
		return t.h, sum, sumsq
	}

	var best []scored
	var sum, sumsq float64
	workers := s.opt.Workers
	if n < s.opt.ParallelThreshold || workers < 2 {
		best, sum, sumsq = scan(0, n)
	} else {
		workers = min(workers, n)
		type part struct {
			h          []scored
			sum, sumsq float64
		}
		parts := make([]part, workers)
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
				h, a, b := scan(lo, hi)
				parts[w] = part{h, a, b}
			}()
		}
		wg.Wait()
		merged := topK{k: k, h: make([]scored, 0, k)}
		for _, p := range parts {
			sum += p.sum
			sumsq += p.sumsq
			for _, e := range p.h {
				merged.offer(e.row, e.score)
			}
		}
		best = merged.h
	}

	mean := sum / float64(n)
	st := ScoreStats{N: n, Mean: mean, Std: math.Sqrt(max(0, sumsq/float64(n)-mean*mean))}
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
	return hits, st, nil
}
