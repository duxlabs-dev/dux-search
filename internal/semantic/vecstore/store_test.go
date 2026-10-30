package vecstore

import (
	"fmt"
	"math"
	"os"
	"sync"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	bolt "go.etcd.io/bbolt"
)

const testModel = "embeddinggemma-2-int8"

// The quantized scan must rank like exact float32 cosine on real model vectors,
// at every Matryoshka dimension the service uses.
func TestQuantizedRankingMatchesFloat(t *testing.T) {
	for _, dim := range []int{256, 512, 768} {
		t.Run(fmt.Sprintf("dim%d", dim), func(t *testing.T) {
			ids, vecs, queries := realCorpus(dim)
			s, _ := openTemp(t, Options{})
			require.NoError(t, s.EnsureCollection("c", dim, testModel))
			require.NoError(t, s.Upsert("c", toVectors(ids, vecs)))

			const k = 10
			var overlap, top1 float64
			var maxErr float64
			for _, q := range queries {
				want := floatTop(ids, vecs, q, k)
				got, err := s.Search("c", q, k)
				require.NoError(t, err)
				require.Len(t, got, k)
				wantSet := map[string]bool{}
				for _, h := range want {
					wantSet[h.ID] = true
				}
				for _, h := range got {
					if wantSet[h.ID] {
						overlap++
					}
				}
				if got[0].ID == want[0].ID {
					top1++
				}
				// the score itself must stay close to the float cosine
				for _, h := range got {
					for i, id := range ids {
						if id == h.ID {
							ref := dot(vecs[i], q) / (norm(vecs[i]) * norm(q))
							maxErr = math.Max(maxErr, math.Abs(float64(h.Score)-ref))
						}
					}
				}
			}
			n := float64(len(queries))
			t.Logf("dim %d: top-%d overlap %.4f, top-1 agreement %.4f, max |score error| %.5f (%d queries x %d docs)",
				dim, k, overlap/(n*k), top1/n, maxErr, len(queries), len(ids))
			assert.GreaterOrEqual(t, overlap/(n*k), 0.97)
			assert.GreaterOrEqual(t, top1/n, 0.95)
			assert.Less(t, maxErr, 0.01)
		})
	}
}

func TestSearchSemantics(t *testing.T) {
	ids, vecs, queries := realCorpus(256)
	s, _ := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids, vecs)))

	hits, err := s.Search("c", queries[0], 0)
	require.NoError(t, err)
	assert.Empty(t, hits)

	hits, err = s.Search("c", queries[0], 100000)
	require.NoError(t, err)
	assert.Len(t, hits, len(ids), "k larger than the collection returns everything")
	for i := 1; i < len(hits); i++ {
		assert.GreaterOrEqual(t, hits[i-1].Score, hits[i].Score)
	}

	hits, err = s.Search("missing", queries[0], 5)
	require.NoError(t, err)
	assert.Empty(t, hits, "unknown collection has no hits")

	_, err = s.Search("c", queries[0][:10], 5)
	assert.ErrorIs(t, err, ErrDimMismatch)
	_, err = s.Search("c", make([]float32, 256), 5)
	assert.ErrorIs(t, err, ErrInvalidVector)

	// a stored vector is its own nearest neighbour with cosine ~1
	hits, err = s.Search("c", vecs[3], 1)
	require.NoError(t, err)
	assert.Equal(t, ids[3], hits[0].ID)
	assert.InDelta(t, 1, hits[0].Score, 0.005)
}

func TestUpsertValidationIsAtomic(t *testing.T) {
	ids, vecs, _ := realCorpus(256)
	s, _ := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))

	bad := map[string]semantic.Vector{
		"wrong dim": {ID: "x", Vec: vecs[0][:100]},
		"zero":      {ID: "x", Vec: make([]float32, 256)},
		"nan":       {ID: "x", Vec: append([]float32{float32(math.NaN())}, vecs[0][1:]...)},
		"inf":       {ID: "x", Vec: append([]float32{float32(math.Inf(1))}, vecs[0][1:]...)},
		"no id":     {ID: "", Vec: vecs[0]},
	}
	for name, v := range bad {
		err := s.Upsert("c", []semantic.Vector{{ID: ids[0], Vec: vecs[0]}, v})
		require.Error(t, err, name)
		n, _ := s.Count("c")
		assert.Zero(t, n, "%s: nothing of the batch may be stored", name)
	}
	assert.ErrorIs(t, s.Upsert("nope", []semantic.Vector{{ID: "a", Vec: vecs[0]}}), ErrNoCollection)
	assert.NoError(t, s.Upsert("c", nil))
}

func TestUpsertReplacesAndDeleteKeepsRowsDense(t *testing.T) {
	ids, vecs, queries := realCorpus(256)
	s, _ := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids, vecs)))

	// replace one entry with another document's vector and hash
	require.NoError(t, s.Upsert("c", []semantic.Vector{{ID: ids[0], Hash: "new", Vec: vecs[1]}}))
	n, _ := s.Count("c")
	assert.Equal(t, len(ids), n)
	h, _ := s.Hashes("c")
	assert.Equal(t, "new", h[ids[0]])
	hits, _ := s.Search("c", vecs[1], 2)
	assert.ElementsMatch(t, []string{ids[0], ids[1]}, []string{hits[0].ID, hits[1].ID})

	// delete a spread of ids incl. first, last and an unknown one
	gone := []string{ids[0], ids[len(ids)-1], ids[40], ids[41], "unknown"}
	require.NoError(t, s.Delete("c", gone))
	require.NoError(t, s.Delete("missing", gone))
	require.NoError(t, s.Delete("c", nil))
	var keepIDs []string
	var keepVecs [][]float32
	goneSet := map[string]bool{}
	for _, g := range gone {
		goneSet[g] = true
	}
	for i, id := range ids {
		if !goneSet[id] {
			keepIDs = append(keepIDs, id)
			keepVecs = append(keepVecs, vecs[i])
		}
	}
	n, _ = s.Count("c")
	assert.Equal(t, len(keepIDs), n)
	for _, q := range queries[:10] {
		want := floatTop(keepIDs, keepVecs, q, 5)
		got, err := s.Search("c", q, 5)
		require.NoError(t, err)
		for _, g := range got {
			assert.False(t, goneSet[g.ID], "deleted id %s resurfaced", g.ID)
		}
		assert.Equal(t, want[0].ID, got[0].ID)
	}
}

func TestPersistenceAcrossRestart(t *testing.T) {
	ids, vecs, queries := realCorpus(256)
	s, path := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.EnsureCollection("other", 512, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids, vecs)))
	require.NoError(t, s.Delete("c", ids[:5]))
	before, err := s.Search("c", queries[2], 10)
	require.NoError(t, err)
	hashesBefore, _ := s.Hashes("c")
	require.NoError(t, s.Close())

	r, err := Open(path, Options{})
	require.NoError(t, err)
	defer r.Close()
	assert.Equal(t, LoadReport{}, r.Report(), "a healthy file loads without drops")
	after, err := r.Search("c", queries[2], 10)
	require.NoError(t, err)
	requireHitsEqual(t, before, after)
	hashesAfter, _ := r.Hashes("c")
	assert.Equal(t, hashesBefore, hashesAfter)
	assert.Equal(t, []CollectionInfo{
		{Name: "c", Dim: 256, Model: testModel, Schema: SchemaVersion, Count: len(ids) - 5},
		{Name: "other", Dim: 512, Model: testModel, Schema: SchemaVersion},
	}, r.Collections())
	v, ok := r.Vector("c", ids[10])
	require.True(t, ok)
	assert.InDelta(t, 1, dot(v, vecs[10]), 0.01, "dequantized vector points the same way")
}

func TestEnsureCollectionInvalidatesOnShapeChange(t *testing.T) {
	ids, vecs, _ := realCorpus(256)
	s, path := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids[:20], vecs[:20])))

	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	n, _ := s.Count("c")
	assert.Equal(t, 20, n, "same shape keeps the vectors")

	// model change
	require.NoError(t, s.EnsureCollection("c", 256, "model-2"))
	n, _ = s.Count("c")
	assert.Zero(t, n, "model change drops the vectors so everything is embedded again")
	h, _ := s.Hashes("c")
	assert.Empty(t, h)
	// dimension change
	require.NoError(t, s.EnsureCollection("c", 512, "model-2"))
	info, _ := s.Info("c")
	assert.Equal(t, 512, info.Dim)
	require.Error(t, s.Upsert("c", toVectors(ids[:1], vecs[:1])), "old-dim vectors are rejected")

	// and the invalidation is durable
	require.NoError(t, s.Close())
	r, err := Open(path, Options{})
	require.NoError(t, err)
	defer r.Close()
	info, _ = r.Info("c")
	assert.Equal(t, CollectionInfo{Name: "c", Dim: 512, Model: "model-2", Schema: SchemaVersion}, info)
}

func TestDropCollection(t *testing.T) {
	ids, vecs, _ := realCorpus(256)
	s, path := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids[:5], vecs[:5])))
	require.NoError(t, s.DropCollection("c"))
	require.NoError(t, s.DropCollection("c"))
	_, ok := s.Info("c")
	assert.False(t, ok)
	require.NoError(t, s.Close())
	r, err := Open(path, Options{})
	require.NoError(t, err)
	defer r.Close()
	assert.Empty(t, r.Collections())
	assert.Empty(t, r.Report().DroppedCollections, "no orphan bucket is left behind")
}

// Damaged records and collections are dropped on load, never fatal.
func TestLoadSurvivesDamagedRecords(t *testing.T) {
	ids, vecs, queries := realCorpus(256)
	s, path := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.EnsureCollection("badmeta", 256, testModel))
	require.NoError(t, s.EnsureCollection("oldschema", 256, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids[:30], vecs[:30])))
	require.NoError(t, s.Upsert("badmeta", toVectors(ids[:3], vecs[:3])))
	require.NoError(t, s.Close())

	db, err := bolt.Open(path, 0o600, nil)
	require.NoError(t, err)
	require.NoError(t, db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(vecBucket).Bucket([]byte("c"))
		good := b.Get([]byte(ids[0]))
		require.NotNil(t, good)
		require.NoError(t, b.Put([]byte("short"), []byte{1}))                           // truncated
		require.NoError(t, b.Put([]byte("long"), append(append([]byte{}, good...), 0))) // wrong size
		zero := append([]byte{}, good...)
		for i := len(zero) - 256; i < len(zero); i++ {
			zero[i] = 0
		}
		require.NoError(t, b.Put([]byte("zero"), zero)) // all-zero vector
		bad := append([]byte{}, good...)
		bad[len(bad)-1] = 0x80 // -128 is never produced
		require.NoError(t, b.Put([]byte("neg128"), bad))
		m := tx.Bucket(metaBucket)
		require.NoError(t, m.Put([]byte(metaPrefix+"badmeta"), []byte("{not json")))
		require.NoError(t, m.Put([]byte(metaPrefix+"oldschema"), []byte(`{"dim":256,"model":"m","schema":0}`)))
		_, err := tx.Bucket(vecBucket).CreateBucket([]byte("orphan"))
		return err
	}))
	require.NoError(t, db.Close())

	r, err := Open(path, Options{})
	require.NoError(t, err)
	defer r.Close()
	rep := r.Report()
	assert.Equal(t, 4, rep.DroppedRecords)
	assert.ElementsMatch(t, []string{"badmeta", "oldschema", "orphan"}, rep.DroppedCollections)
	n, _ := r.Count("c")
	assert.Equal(t, 30, n, "the undamaged records are all there")
	hits, err := r.Search("c", queries[0], 5)
	require.NoError(t, err)
	assert.Len(t, hits, 5)
	assert.Len(t, r.Collections(), 1)
	require.NoError(t, r.Close())

	again, err := Open(path, Options{})
	require.NoError(t, err)
	defer again.Close()
	assert.Equal(t, LoadReport{}, again.Report(), "the damage was repaired on disk, not just in memory")
}

func TestOpenCorruptFile(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/semantic.db"
	require.NoError(t, os.WriteFile(path, []byte("this is not a bbolt database, it is long enough to be mistaken for one......."+string(make([]byte, 8192))), 0o600))

	_, err := Open(path, Options{})
	require.Error(t, err, "without RecoverCorrupt a damaged file is an error, never silently replaced")

	s, err := Open(path, Options{RecoverCorrupt: true})
	require.NoError(t, err)
	defer s.Close()
	assert.True(t, s.Report().Recovered)
	_, statErr := os.Stat(path + ".corrupt")
	assert.NoError(t, statErr, "the damaged file is kept for inspection")
	require.NoError(t, s.EnsureCollection("c", 8, "m"))
}

func TestNewerFormatIsNotOverwritten(t *testing.T) {
	s, path := openTemp(t, Options{})
	require.NoError(t, s.Close())
	db, err := bolt.Open(path, 0o600, nil)
	require.NoError(t, err)
	require.NoError(t, db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(metaBucket).Put(formatKey, []byte("99"))
	}))
	require.NoError(t, db.Close())
	_, err = Open(path, Options{RecoverCorrupt: true})
	require.ErrorIs(t, err, errNewerFormat, "a database from a newer dsearch must not be 'recovered' into an empty one")
	_, statErr := os.Stat(path + ".corrupt")
	assert.True(t, os.IsNotExist(statErr))
}

func TestSecondProcessGetsLockError(t *testing.T) {
	_, path := openTemp(t, Options{})
	_, err := Open(path, Options{RecoverCorrupt: true})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "locked")
	_, statErr := os.Stat(path + ".corrupt")
	assert.True(t, os.IsNotExist(statErr), "a lock timeout is not corruption")
}

func TestParallelScanMatchesSerial(t *testing.T) {
	const n, dim = 6000, 256
	data := perturbed(n, dim, 1)
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
		a, err := serial.Search("c", q, 25)
		require.NoError(t, err)
		b, err := parallel.Search("c", q, 25)
		require.NoError(t, err)
		requireHitsEqual(t, a, b)
	}
}

func TestConcurrentUseIsRaceFree(t *testing.T) {
	ids, vecs, queries := realCorpus(256)
	s, _ := openTemp(t, Options{ParallelThreshold: 50, Workers: 4})
	require.NoError(t, s.EnsureCollection("c", 256, testModel))
	require.NoError(t, s.Upsert("c", toVectors(ids, vecs)))

	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(3)
		go func() {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				if _, err := s.Search("c", queries[(w+i)%len(queries)], 8); err != nil {
					t.Error(err)
				}
			}
		}()
		go func() {
			defer wg.Done()
			for i := 0; i < 15; i++ {
				j := (w*37 + i*11) % len(ids)
				if err := s.Upsert("c", []semantic.Vector{{ID: ids[j], Hash: fmt.Sprint(i), Vec: vecs[j]}}); err != nil {
					t.Error(err)
				}
				if err := s.Delete("c", []string{ids[(j+1)%len(ids)]}); err != nil {
					t.Error(err)
				}
				_, _ = s.Hashes("c")
				_, _ = s.Count("c")
			}
		}()
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				name := fmt.Sprintf("tmp%d", w)
				_ = s.EnsureCollection(name, 256, testModel)
				_ = s.Upsert(name, toVectors(ids[:3], vecs[:3]))
				_ = s.DropCollection(name)
			}
		}()
	}
	wg.Wait()
	// the in-memory block and the id index still agree
	s.mu.RLock()
	defer s.mu.RUnlock()
	c := s.cols["c"]
	require.Equal(t, len(c.ids), len(c.index))
	require.Equal(t, len(c.ids)*c.dim, len(c.data))
	for i, id := range c.ids {
		require.Equal(t, i, c.index[id])
	}
}

func TestItemsRoundTrip(t *testing.T) {
	s, path := openTemp(t, Options{})
	items := []semantic.Item{
		{ID: "b", Title: "Bee", Text: "buzz", Keywords: []string{"insect"}, Payload: map[string]any{"category": "x"}},
		{ID: "a", Title: "Ant", Text: "walk"},
	}
	require.NoError(t, s.PutItems("shell.test", items))
	got, err := s.Items("shell.test")
	require.NoError(t, err)
	require.Len(t, got, 2)
	assert.Equal(t, "a", got[0].ID, "ordered by id")
	assert.Equal(t, items[0].Payload, got[1].Payload)

	require.NoError(t, s.PutItems("shell.test", items[1:]))
	got, _ = s.Items("shell.test")
	require.Len(t, got, 1, "PutItems replaces the whole set")

	require.NoError(t, s.Close())
	r, err := Open(path, Options{})
	require.NoError(t, err)
	defer r.Close()
	names, err := r.ItemCollections()
	require.NoError(t, err)
	assert.Equal(t, []string{"shell.test"}, names)
	require.NoError(t, r.DeleteItems("shell.test"))
	require.NoError(t, r.DeleteItems("shell.test"))
	got, _ = r.Items("shell.test")
	assert.Empty(t, got)
	assert.Error(t, r.PutItems("c", []semantic.Item{{ID: ""}}))
}

func TestClosedStore(t *testing.T) {
	s, _ := openTemp(t, Options{})
	require.NoError(t, s.EnsureCollection("c", 8, "m"))
	require.NoError(t, s.Close())
	_, err := s.Search("c", make([]float32, 8), 1)
	assert.ErrorIs(t, err, ErrClosed)
	assert.ErrorIs(t, s.EnsureCollection("c", 8, "m"), ErrClosed)
}

func TestDotInt8(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 5, 255, 256, 768} {
		a, b := make([]int8, n), make([]int8, n)
		var want int32
		for i := range a {
			a[i], b[i] = int8(i%255-127), int8((i*7)%255-127)
			want += int32(a[i]) * int32(b[i])
		}
		assert.Equal(t, want, dotInt8(a, b), "n=%d", n)
	}
	// extremes cannot overflow the int32 accumulator at the largest dimension
	a, b := make([]int8, maxDim), make([]int8, maxDim)
	for i := range a {
		a[i], b[i] = 127, 127
	}
	assert.Equal(t, int32(127*127*maxDim), dotInt8(a, b))
}
