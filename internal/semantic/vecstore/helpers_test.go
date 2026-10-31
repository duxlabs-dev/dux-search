package vecstore

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"sort"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service/fixture"
	"github.com/stretchr/testify/require"
)

func openTemp(t testing.TB, opt Options) (*Store, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "semantic.db")
	s, err := Open(path, opt)
	require.NoError(t, err)
	t.Cleanup(func() { s.Close() })
	return s, path
}

// floatTop is the reference: exact cosine of float32 vectors, brute force.
func floatTop(ids []string, vecs [][]float32, q []float32, k int) []semantic.Hit {
	hits := make([]semantic.Hit, len(ids))
	nq := norm(q)
	for i := range ids {
		hits[i] = semantic.Hit{ID: ids[i], Score: float32(dot(vecs[i], q) / (norm(vecs[i]) * nq))}
	}
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].Score != hits[b].Score {
			return hits[a].Score > hits[b].Score
		}
		return hits[a].ID < hits[b].ID
	})
	if k < len(hits) {
		hits = hits[:k]
	}
	return hits
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func norm(a []float32) float64 { return math.Sqrt(dot(a, a)) }

// realCorpus returns the model vectors of every fixture document truncated to dim,
// with unique ids.
func realCorpus(dim int) (ids []string, vecs [][]float32, queries [][]float32) {
	d := fixture.MustLoad()
	for i, x := range d.Docs {
		ids = append(ids, docID(i, x))
		vecs = append(vecs, semantic.Truncate(x.Vec, dim))
	}
	for _, q := range d.Queries {
		queries = append(queries, semantic.Truncate(q.Vec, dim))
	}
	return
}

func docID(i int, x fixture.Doc) string {
	if x.Kind == "action" {
		return "action:" + x.ID
	}
	return "file:" + x.Pkg + "#" + string(rune('0'+x.Chunk))
}

func toVectors(ids []string, vecs [][]float32) []semantic.Vector {
	out := make([]semantic.Vector, len(ids))
	for i := range ids {
		out[i] = semantic.Vector{ID: ids[i], Hash: "h-" + ids[i], Vec: vecs[i]}
	}
	return out
}

// perturbed derives n vectors of the given dim from real model vectors by adding
// gaussian noise and renormalizing. It only provides volume for scan-equivalence
// tests and timing; accuracy claims are made on real vectors alone.
func perturbed(n, dim int, seed uint64) [][]float32 {
	base := fixture.MustLoad().Docs
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	out := make([][]float32, n)
	for i := range out {
		b := semantic.Truncate(base[r.IntN(len(base))].Vec, dim)
		v := make([]float32, dim)
		for j := range v {
			v[j] = b[j] + float32(r.NormFloat64())*0.02
		}
		out[i] = semantic.Truncate(v, dim)
	}
	return out
}

func requireHitsEqual(t *testing.T, want, got []semantic.Hit) {
	t.Helper()
	require.Equal(t, len(want), len(got))
	for i := range want {
		require.Equal(t, want[i].ID, got[i].ID, "rank %d", i)
		require.InDelta(t, want[i].Score, got[i].Score, 1e-6, "rank %d", i)
	}
}
