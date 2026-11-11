package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	bleve "github.com/blevesearch/bleve/v2"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// corpus is a real directory of real files (the package documentation of the Go
// standard library, one .txt per package), indexed by the real lexical indexer,
// with the real model vectors of their chunks in files.text, under the id scheme
// of the file indexer: "<path>#<chunk>".
type corpus struct {
	*env
	root  string
	idx   *indexer.Indexer
	paths map[string]string // package -> file path
}

func newCorpus(t *testing.T) *corpus {
	t.Helper()
	e := newEnv(t)
	root := t.TempDir()
	cfg := config.Default()
	cfg.IndexPath = filepath.Join(root, ".idx", "index")
	cfg.IndexPaths = []config.IndexPath{{Path: root, MaxDepth: 10, ExcludeDirs: []string{}}}
	cfg.BuildMaps()
	idx, err := indexer.New(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { idx.Close() })

	c := &corpus{env: e, root: root, idx: idx, paths: map[string]string{}}
	chunks := map[string][]string{}
	var pkgs []string
	for _, d := range e.data.FileDocs() {
		if _, ok := chunks[d.Pkg]; !ok {
			pkgs = append(pkgs, d.Pkg)
		}
		chunks[d.Pkg] = append(chunks[d.Pkg], d.Text)
	}
	for _, p := range pkgs {
		path := filepath.Join(root, strings.ReplaceAll(p, "/", "_")+".txt")
		require.NoError(t, os.WriteFile(path, []byte(strings.Join(chunks[p], "\n\n")), 0o644))
		require.NoError(t, idx.Index(path))
		c.paths[p] = path
	}
	require.NoError(t, e.st.EnsureCollection(CollFilesText, DimFilesText, "files-model"))
	var vecs []semantic.Vector
	for _, d := range e.data.FileDocs() {
		vecs = append(vecs, semantic.Vector{ID: fmt.Sprintf("%s#%d", c.paths[d.Pkg], d.Chunk), Hash: "h", Vec: semantic.Truncate(d.Vec, DimFilesText)})
	}
	require.NoError(t, e.st.Upsert(CollFilesText, vecs))
	return c
}

func (c *corpus) hybrid(t *testing.T, opts indexer.SearchOptions, req HybridRequest) *HybridResult {
	t.Helper()
	if opts.Limit == 0 {
		opts.Limit = 10
	}
	res, err := c.svc.HybridSearch(context.Background(), c.idx, &opts, req)
	require.NoError(t, err)
	return res
}

func ids(hits []ResultHit) []string {
	var out []string
	for _, h := range hits {
		out = append(out, h.ID)
	}
	return out
}

// The labelled queries of the fixture are the recall side of fileMinZ: on these
// 37 queries the expected file reaches the line in 33, the 4 it misses are ones
// where the model itself ranks it 3rd to 7th (z 2.4-2.8). The precision side,
// queries without an answer, is in the comment on fileMinZ.
func TestFileCandidatesContainTheExpectedFile(t *testing.T) {
	c := newCorpus(t)
	qs := c.data.QueriesOf("file")
	found, top1 := 0, 0
	for _, q := range qs {
		resp, err := c.svc.Search(context.Background(), SearchRequest{Query: q.Text, Files: true})
		require.NoError(t, err)
		require.True(t, resp.Available)
		require.LessOrEqual(t, len(resp.Files), 1)
		if len(resp.Files) == 0 {
			continue // nothing stood out
		}
		require.Equal(t, CollFilesText, resp.Files[0].Collection)
		hits := resp.Files[0].Hits
		want := c.paths[q.Expect]
		for i, h := range hits {
			if h.Path == want {
				found++
				if i == 0 {
					top1++
				}
			}
		}
		assert.LessOrEqual(t, len(hits), filePool)
		assert.True(t, sort.SliceIsSorted(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score }))
		for _, h := range hits {
			assert.GreaterOrEqual(t, h.Z, fileMinZ, "only files that stand out from the collection")
			assert.NotContains(t, h.Path, "#", "chunk ids are folded to paths")
		}
	}
	t.Logf("files: expected package first in %d/%d queries, among the candidates in %d/%d", top1, len(qs), found, len(qs))
	assert.GreaterOrEqual(t, found, 33)
	assert.GreaterOrEqual(t, top1, 29)
}

func TestSemanticOnlyHitLooksLikeAnyLexicalHit(t *testing.T) {
	c := newCorpus(t)
	// Turkish query, English documents: bleve finds nothing, the embeddings do.
	res := c.hybrid(t, indexer.SearchOptions{Query: "düzenli ifadelerle metin eşleştirme", Type: "file"}, HybridRequest{})
	require.True(t, res.Semantic.Available, res.Semantic.Reason)
	require.True(t, res.Semantic.Fused)
	require.NotEmpty(t, res.Hits)
	assert.Equal(t, c.paths["regexp"], res.Hits[0].ID)
	require.NotNil(t, res.Hits[0].SemanticScore)
	assert.Greater(t, *res.Hits[0].SemanticScore, float32(0), "centred cosine of a file that stands out")
	assert.GreaterOrEqual(t, res.Semantic.Added, 1)
	assert.Equal(t, uint64(res.Semantic.Added), res.SearchResult.Total, "semantic-only hits count in the total")

	// the same file found lexically (by its name) has the same JSON shape
	lex := c.hybrid(t, indexer.SearchOptions{Query: "regexp", Type: "file"}, HybridRequest{})
	var lexHit *ResultHit
	for i := range lex.Hits {
		if lex.Hits[i].ID == c.paths["regexp"] {
			lexHit = &lex.Hits[i]
		}
	}
	require.NotNil(t, lexHit, "found by file name")
	keys := func(h ResultHit) []string {
		raw, err := json.Marshal(h)
		require.NoError(t, err)
		var m map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(raw, &m))
		delete(m, "semantic_score")
		var k []string
		for key := range m {
			k = append(k, key)
		}
		sort.Strings(k)
		return k
	}
	assert.Equal(t, keys(*lexHit), keys(res.Hits[0]))
	raw, _ := json.Marshal(res.Hits[0])
	assert.Contains(t, string(raw), `"locations":{"doc_type":{"file"`, "the launcher reads hit.locations.doc_type")
	assert.NotContains(t, string(raw), `"path":`, "nothing of the helper path query leaks into locations or fragments")
}

func TestHitsWithoutSemanticAreByteCompatible(t *testing.T) {
	c := newCorpus(t)
	opts := indexer.SearchOptions{Query: "regexp", Type: "file", Limit: 10}
	want, err := c.idx.SearchWithOptions(&opts)
	require.NoError(t, err)

	c.emb.SetAvailable(false, "no model")
	res := c.hybrid(t, opts, HybridRequest{})
	assert.False(t, res.Semantic.Available)
	assert.False(t, res.Semantic.Fused)
	var got []string
	for _, h := range res.Hits {
		got = append(got, h.ID)
		assert.Nil(t, h.SemanticScore)
	}
	var wantIDs []string
	for _, h := range want.Hits {
		wantIDs = append(wantIDs, h.ID)
	}
	assert.Equal(t, wantIDs, got, "model missing: exactly the lexical hits, same order")
	for i, h := range res.Hits {
		assert.Equal(t, want.Hits[i].Score, h.Score, "scores are the lexical ones when nothing was fused")
	}
}

func TestFusionRanksFilesFoundByBothFirst(t *testing.T) {
	c := newCorpus(t)
	q := "regular expression matching"
	lex, err := c.idx.SearchWithOptions(&indexer.SearchOptions{Query: q, Type: "file", Limit: 10})
	require.NoError(t, err)
	lexRank := 0
	for i, h := range lex.Hits {
		if h.ID == c.paths["regexp"] {
			lexRank = i + 1
		}
	}
	require.NotZero(t, lexRank, "the lexical search finds the regexp doc too")

	res := c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file"}, HybridRequest{})
	require.NotEmpty(t, res.Hits)
	assert.Equal(t, c.paths["regexp"], res.Hits[0].ID)
	assert.InDelta(t, 1/float64(rrfK+lexRank)+1/float64(rrfK+1), res.Hits[0].Score, 1e-12, "RRF of its lexical and semantic rank")
	assert.Equal(t, res.Hits[0].Score, res.SearchResult.MaxScore)
	for i := 1; i < len(res.Hits); i++ {
		assert.GreaterOrEqual(t, res.Hits[i-1].Score, res.Hits[i].Score, "descending by fused score")
	}
}

func TestLimitAndPool(t *testing.T) {
	c := newCorpus(t)
	for _, limit := range []int{1, 3, 7} {
		res := c.hybrid(t, indexer.SearchOptions{Query: "date and time formatting", Type: "file", Limit: limit}, HybridRequest{})
		assert.Len(t, res.Hits, limit)
	}
}

func TestFiltersApplyToSemanticOnlyHits(t *testing.T) {
	c := newCorpus(t)
	q := "düzenli ifadelerle metin eşleştirme"
	want := c.paths["regexp"]
	contains := func(res *HybridResult) bool {
		for _, h := range res.Hits {
			if h.ID == want {
				return true
			}
		}
		return false
	}
	assert.True(t, contains(c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file", Extension: ".txt"}, HybridRequest{})))
	assert.False(t, contains(c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file", Extension: ".go"}, HybridRequest{})), "extension filter")
	assert.False(t, contains(c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file", MinSize: 1 << 30}, HybridRequest{})), "size filter")
	assert.True(t, contains(c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file", Folder: c.root}, HybridRequest{})), "folder filter, inside")
	other := filepath.Join(c.root, "elsewhere")
	require.NoError(t, os.Mkdir(other, 0o755))
	assert.False(t, contains(c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file", Folder: other}, HybridRequest{})), "folder filter, outside")
	// a stale vector of a file that is gone from the index is dropped silently
	require.NoError(t, c.idx.Delete(want))
	assert.False(t, contains(c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file"}, HybridRequest{})))
}

func TestTypeAllAndDirUntouched(t *testing.T) {
	c := newCorpus(t)
	q := "düzenli ifadelerle metin eşleştirme"
	res := c.hybrid(t, indexer.SearchOptions{Query: q, Type: "all"}, HybridRequest{})
	require.NotEmpty(t, res.Hits)
	assert.Equal(t, c.paths["regexp"], res.Hits[0].ID, "type=all fuses the file side")

	before := c.emb.QueriesEmbedded()
	res = c.hybrid(t, indexer.SearchOptions{Query: q, Type: "dir"}, HybridRequest{})
	assert.Empty(t, res.Hits, "type=dir never returns semantic files")
	assert.Equal(t, before, c.emb.QueriesEmbedded(), "and does not embed for them")
}

func TestExplicitSortKeepsOrderAndAnnotates(t *testing.T) {
	c := newCorpus(t)
	q := "regular expression matching"
	plain, err := c.idx.SearchWithOptions(&indexer.SearchOptions{Query: q, Type: "file", Limit: 10, SortBy: "filename"})
	require.NoError(t, err)
	res := c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file", SortBy: "filename"}, HybridRequest{})
	assert.False(t, res.Semantic.Fused)
	var want []string
	for _, h := range plain.Hits {
		want = append(want, h.ID)
	}
	assert.Equal(t, want, ids(res.Hits), "the caller's sort order is not touched")
	annotated := 0
	for _, h := range res.Hits {
		if h.SemanticScore != nil {
			annotated++
		}
	}
	assert.Positive(t, annotated)
}

func TestFilenameOnlySearchIsNotFused(t *testing.T) {
	c := newCorpus(t)
	before := c.emb.QueriesEmbedded()
	res := c.hybrid(t, indexer.SearchOptions{Query: "regexp", Field: "filename", Type: "file"}, HybridRequest{})
	assert.False(t, res.Semantic.Fused)
	assert.Equal(t, before, c.emb.QueriesEmbedded(), "a filename-only search has no semantic side")
}

func TestMatchAllDoesNotEmbed(t *testing.T) {
	c := newCorpus(t)
	before := c.emb.QueriesEmbedded()
	res := c.hybrid(t, indexer.SearchOptions{Query: "*", Type: "file"}, HybridRequest{})
	assert.True(t, res.Semantic.Available)
	assert.Equal(t, before, c.emb.QueriesEmbedded())
	assert.NotEmpty(t, res.Hits)
}

func TestHybridCollectionsAndFilesTogether(t *testing.T) {
	c := newCorpus(t)
	_, err := c.svc.SetCollection(context.Background(), "shell.actions", c.actionItems())
	require.NoError(t, err)
	before := c.emb.QueriesEmbedded()
	res := c.hybrid(t, indexer.SearchOptions{Query: "karanlık", Type: "file"}, HybridRequest{Collections: []string{"shell.actions"}, CollectionLimit: 2})
	assert.Equal(t, before+1, c.emb.QueriesEmbedded(), "the query is embedded once for collections and files")
	require.Len(t, res.Collections, 1)
	assert.Equal(t, "dark", res.Collections[0].Hits[0].ID)
	assert.Len(t, res.Collections[0].Hits, 2)

	jsonKeys := func(v any) map[string]bool {
		raw, err := json.Marshal(v)
		require.NoError(t, err)
		var m map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(raw, &m))
		keys := map[string]bool{}
		for k := range m {
			keys[k] = true
		}
		return keys
	}
	plain, err := c.idx.SearchWithOptions(&indexer.SearchOptions{Query: "karanlık", Type: "file", Limit: 5})
	require.NoError(t, err)
	got := jsonKeys(res)
	for k := range jsonKeys(&indexer.SearchResult{SearchResult: plain}) {
		assert.True(t, got[k], "the existing response field %q must stay", k)
	}
	assert.True(t, got["collections"])
	assert.True(t, got["semantic"])
}

type failingLexical struct{ err error }

func (f failingLexical) SearchWithOptions(*indexer.SearchOptions) (*bleve.SearchResult, error) {
	return nil, f.err
}
func (f failingLexical) SearchAll(*indexer.SearchOptions) (*indexer.SearchResult, error) {
	return nil, f.err
}

func TestLexicalFailureKeepsCollectionResults(t *testing.T) {
	e := newEnv(t)
	_, err := e.svc.SetCollection(context.Background(), "shell.actions", e.actionItems())
	require.NoError(t, err)
	lex := failingLexical{errors.New("index not ready")}

	res, err := e.svc.HybridSearch(context.Background(), lex, &indexer.SearchOptions{Query: "karanlık", Limit: 5}, HybridRequest{Collections: []string{"shell.actions"}})
	require.NoError(t, err)
	assert.Equal(t, "index not ready", res.LexicalError)
	assert.Empty(t, res.Hits)
	assert.Equal(t, "dark", res.Collections[0].Hits[0].ID)
	raw, _ := json.Marshal(res)
	assert.Contains(t, string(raw), `"hits":[]`)

	_, err = e.svc.HybridSearch(context.Background(), lex, &indexer.SearchOptions{Query: "karanlık", Limit: 5}, HybridRequest{})
	assert.EqualError(t, err, "index not ready", "without collections the lexical error is the error")
}

func TestHybridWhenDisabled(t *testing.T) {
	svc, err := New(Options{})
	require.NoError(t, err)
	var calls atomic.Int32
	lex := countingLexical{n: &calls}
	res, err := svc.HybridSearch(context.Background(), lex, &indexer.SearchOptions{Query: "x", Limit: 4}, HybridRequest{Collections: []string{"a"}})
	require.NoError(t, err)
	assert.False(t, res.Semantic.Available)
	assert.Equal(t, ErrDisabled.Error(), res.Semantic.Reason)
	assert.EqualValues(t, 1, calls.Load(), "exactly one lexical search, no pool, no fetches")
}

type countingLexical struct{ n *atomic.Int32 }

func (c countingLexical) SearchWithOptions(o *indexer.SearchOptions) (*bleve.SearchResult, error) {
	c.n.Add(1)
	if o.Limit != 4 {
		return nil, fmt.Errorf("limit changed to %d", o.Limit)
	}
	return &bleve.SearchResult{}, nil
}
func (c countingLexical) SearchAll(*indexer.SearchOptions) (*indexer.SearchResult, error) {
	return nil, errors.New("unexpected")
}

func TestSlowModelDegradesToLexicalResults(t *testing.T) {
	c := newCorpus(t)
	c.close()
	c.open(func(o *Options) { o.SearchTimeout = 30 * time.Millisecond })
	c.emb.SetDelay(func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() })

	opts := indexer.SearchOptions{Query: "regexp", Type: "file", Limit: 5}
	want, err := c.idx.SearchWithOptions(&opts)
	require.NoError(t, err)
	start := time.Now()
	res := c.hybrid(t, opts, HybridRequest{})
	assert.Less(t, time.Since(start), 2*time.Second)
	assert.False(t, res.Semantic.Available)
	assert.Contains(t, res.Semantic.Reason, "timed out")
	assert.Len(t, res.Hits, len(want.Hits), "the lexical answer is still delivered")

	// the caller giving up is an error, not a degradation
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = c.svc.HybridSearch(ctx, c.idx, &opts, HybridRequest{})
	assert.ErrorIs(t, err, context.Canceled)
}

func TestPoolNeverShrinksTheCallersLimitAndDirectoriesStayWithinIt(t *testing.T) {
	c := newCorpus(t)
	for _, d := range []string{"regexp_a", "regexp_b", "regexp_c"} {
		require.NoError(t, os.Mkdir(filepath.Join(c.root, d), 0o755))
	}
	require.NoError(t, c.idx.ReindexAll())

	res := c.hybrid(t, indexer.SearchOptions{Query: "regexp", Type: "all", Limit: 1}, HybridRequest{})
	assert.Len(t, res.Hits, 1)
	assert.Len(t, res.SearchResult.DirectoryHits, 1, "directories are not pooled")

	// --limit 0 of the CLI asks for 10000: the lexical search must be asked for
	// at least that many, not for the pool cap
	rec := &recordingLexical{Lexical: c.idx}
	_, err := c.svc.HybridSearch(context.Background(), rec, &indexer.SearchOptions{Query: "date and time formatting", Type: "file", Limit: 10000}, HybridRequest{})
	require.NoError(t, err)
	assert.Equal(t, 10000, rec.firstLimit)
	rec = &recordingLexical{Lexical: c.idx}
	_, err = c.svc.HybridSearch(context.Background(), rec, &indexer.SearchOptions{Query: "date and time formatting", Type: "file", Limit: 3}, HybridRequest{})
	require.NoError(t, err)
	assert.Equal(t, minPool, rec.firstLimit, "a small limit gets the default pool")
}

type recordingLexical struct {
	Lexical
	firstLimit int
}

func (r *recordingLexical) SearchWithOptions(o *indexer.SearchOptions) (*bleve.SearchResult, error) {
	if r.firstLimit == 0 {
		r.firstLimit = o.Limit
	}
	return r.Lexical.SearchWithOptions(o)
}

// Image and audio vectors have no chunk suffix, and the lists of all file
// collections become one ranking by z before the fusion. The 512-d vectors here
// are real model vectors of the fixture documents standing in for image vectors:
// this test is about how lists combine, not about image quality (which needs the
// image tower).
func TestFileCollectionsMergeIntoOneRanking(t *testing.T) {
	c := newCorpus(t)
	require.NoError(t, c.st.EnsureCollection(CollFilesImage, DimFilesImage, "files-model"))
	hashPath := "/shots/screen#1.png" // a '#' in a name is not a chunk suffix for images
	var vecs []semantic.Vector
	for _, d := range c.data.FileDocs() {
		v := semantic.Truncate(d.Vec, DimFilesImage)
		if d.Pkg == "math/big" && d.Chunk == 0 {
			vecs = append(vecs,
				semantic.Vector{ID: c.paths["math/big"], Hash: "h", Vec: v},
				semantic.Vector{ID: hashPath, Hash: "h", Vec: v})
			continue
		}
		if d.Chunk == 0 && d.Pkg != "math/big" { // the rest of the collection, so z has a distribution
			vecs = append(vecs, semantic.Vector{ID: fmt.Sprintf("/shots/%s.png", strings.ReplaceAll(d.Pkg, "/", "_")), Hash: "h", Vec: v})
		}
	}
	require.GreaterOrEqual(t, len(vecs), fileMinRows)
	require.NoError(t, c.st.Upsert(CollFilesImage, vecs))

	q := "çok büyük tam sayılarla hesaplama" // no lexical hit
	resp, err := c.svc.Search(context.Background(), SearchRequest{Query: q, Files: true})
	require.NoError(t, err)
	got := map[string][]string{}
	for _, l := range resp.Files {
		for _, h := range l.Hits {
			got[l.Collection] = append(got[l.Collection], h.Path)
		}
	}
	assert.Contains(t, got[CollFilesText], c.paths["math/big"])
	assert.Contains(t, got[CollFilesImage], hashPath, "the image id is the path, '#1' included")
	assert.Contains(t, got[CollFilesImage], c.paths["math/big"])

	ranking := semanticFileRanking(resp.Files)
	seen := map[string]bool{}
	for i, h := range ranking {
		assert.False(t, seen[h.Path], "a file in several lists appears once")
		seen[h.Path] = true
		if i > 0 {
			assert.GreaterOrEqual(t, ranking[i-1].Z, h.Z, "one ranking by z across modalities")
		}
	}
	rank := 0
	for i, h := range ranking {
		if h.Path == c.paths["math/big"] {
			rank = i + 1
		}
	}
	require.NotZero(t, rank)

	res := c.hybrid(t, indexer.SearchOptions{Query: q, Type: "file"}, HybridRequest{})
	require.NotEmpty(t, res.Hits)
	top := res.Hits[0]
	assert.Equal(t, c.paths["math/big"], top.ID)
	assert.InDelta(t, 1/float64(rrfK+rank), top.Score, 1e-12, "no lexical hit: one semantic term, whatever number of lists it is in")
	// a file that exists only as an image vector but is not in the index is dropped
	for _, h := range res.Hits {
		assert.NotEqual(t, hashPath, h.ID)
	}
}

// A collection too small for its score distribution to mean anything adds no
// candidates: with a handful of rows the best one always looks like an outlier.
func TestSmallFileCollectionAddsNothing(t *testing.T) {
	c := newCorpus(t)
	require.NoError(t, c.st.EnsureCollection(CollFilesImage, DimFilesImage, "files-model"))
	var vecs []semantic.Vector
	for _, d := range c.data.FileDocs() {
		if d.Chunk == 0 && len(vecs) < fileMinRows-1 {
			vecs = append(vecs, semantic.Vector{ID: "/shots/" + strings.ReplaceAll(d.Pkg, "/", "_") + ".png", Hash: "h", Vec: semantic.Truncate(d.Vec, DimFilesImage)})
		}
	}
	require.NoError(t, c.st.Upsert(CollFilesImage, vecs))
	resp, err := c.svc.Search(context.Background(), SearchRequest{Query: "regular expression matching", Files: true})
	require.NoError(t, err)
	for _, l := range resp.Files {
		if l.Collection == CollFilesImage {
			assert.Empty(t, l.Hits)
		}
	}
}
