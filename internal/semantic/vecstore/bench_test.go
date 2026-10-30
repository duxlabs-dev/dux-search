package vecstore

import (
	"fmt"
	"runtime"
	"testing"
)

// memoryStore builds a collection straight in memory: the benchmark measures the
// scan, not bbolt writes of a million records.
func memoryStore(b *testing.B, n, dim int) (*Store, []float32) {
	b.Helper()
	s := &Store{opt: Options{ParallelThreshold: DefaultParallelThreshold, Workers: runtime.GOMAXPROCS(0)}, cols: map[string]*collection{}}
	c := newCollection("c", dim, testModel)
	const chunk = 50_000
	for done := 0; done < n; done += chunk {
		for i, v := range perturbed(min(chunk, n-done), dim, uint64(done)+7) {
			q, sc, inv, err := quantize(v)
			if err != nil {
				b.Fatal(err)
			}
			c.set(fmt.Sprintf("f%d", done+i), "", q, sc, inv)
		}
	}
	s.cols["c"] = c
	_, _, queries := realCorpus(dim)
	return s, queries[0]
}

func BenchmarkSearch(b *testing.B) {
	for _, n := range []int{5_000, 100_000, 1_000_000} {
		b.Run(fmt.Sprintf("n=%d/dim=256", n), func(b *testing.B) {
			s, q := memoryStore(b, n, 256)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := s.Search("c", q, 20); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	b.Run("n=1000000/dim=256/serial", func(b *testing.B) {
		s, q := memoryStore(b, 1_000_000, 256)
		s.opt.Workers = 1
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := s.Search("c", q, 20); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("n=100000/dim=768", func(b *testing.B) {
		s, q := memoryStore(b, 100_000, 768)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := s.Search("c", q, 20); err != nil {
				b.Fatal(err)
			}
		}
	})
}
