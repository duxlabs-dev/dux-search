package service

import (
	"context"
	"errors"
	"sort"
	"strings"
	"sync"
	"time"

	bleve "github.com/blevesearch/bleve/v2"
	"github.com/blevesearch/bleve/v2/search"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
)

// rrfK is the constant of Reciprocal Rank Fusion, score = sum 1/(k+rank) over the
// ranked lists a document appears in. 60 is the value of the original paper
// (Cormack, Clarke, Buettcher, SIGIR 2009) and the de-facto default. It is the
// right shape here because the lists have incomparable scores (bleve text scores
// are unbounded, cosines sit in 0.6-0.8) and RRF only looks at ranks, so nothing
// has to be tuned or normalized per corpus.
const rrfK = 60

const (
	fetchWorkers = 8
	minPool      = 20
	maxPool      = 200
)

// Lexical is the part of the lexical indexer the hybrid search needs.
type Lexical interface {
	SearchWithOptions(opts *indexer.SearchOptions) (*bleve.SearchResult, error)
	SearchAll(opts *indexer.SearchOptions) (*indexer.SearchResult, error)
}

// HybridRequest carries the semantic parameters of a search call.
type HybridRequest struct {
	Collections     []string
	CollectionLimit int
}

// ResultHit is a file hit in the response: exactly what the lexical search
// returns, plus the embedding similarity when there is one.
type ResultHit struct {
	*search.DocumentMatch
	SemanticScore *float32 `json:"semantic_score,omitempty"`
}

// SemanticInfo tells the client what the semantic half of a search did.
type SemanticInfo struct {
	Available bool    `json:"available"`
	Reason    string  `json:"reason,omitempty"`
	Fused     bool    `json:"fused"`   // file hits were merged with embedding results
	Added     int     `json:"added"`   // file hits found by embeddings only
	TookMs    float64 `json:"took_ms"` // time of the embedding side
}

// HybridResult is the search response when semantic is requested. It is the
// lexical result (same fields as without semantic) with hits that may carry
// semantic_score, plus the collection results.
type HybridResult struct {
	*indexer.SearchResult
	Hits         []ResultHit      `json:"hits"` // shadows the lexical Hits
	Collections  []CollectionHits `json:"collections,omitempty"`
	Semantic     *SemanticInfo    `json:"semantic,omitempty"`
	LexicalError string           `json:"lexical_error,omitempty"`
}

// Plain wraps a lexical-only result in the response type, for transports that
// have one response type for both kinds of search. It encodes like the plain result.
func Plain(res *indexer.SearchResult) *HybridResult {
	out := &HybridResult{SearchResult: res, Hits: []ResultHit{}}
	if res != nil && res.SearchResult != nil {
		for _, h := range res.Hits {
			out.Hits = append(out.Hits, ResultHit{DocumentMatch: h})
		}
	}
	return out
}

// HybridSearch runs the lexical search and the embedding search side by side and
// merges the file hits with RRF. The embedding of the query (slow when the model
// is cold) overlaps with the lexical search instead of following it.
func (s *Service) HybridSearch(ctx context.Context, lex Lexical, opts *indexer.SearchOptions, req HybridRequest) (*HybridResult, error) {
	for _, name := range req.Collections {
		if err := validCollectionName(name); err != nil {
			return nil, err // a malformed request, not a degradation
		}
	}
	info := &SemanticInfo{}
	out := &HybridResult{Semantic: info, Hits: []ResultHit{}}

	query := strings.TrimSpace(opts.Query)
	matchAll := query == "" || query == "*"
	// Files are searched by embedding unless the request is about directories,
	// one field only, or has no text to embed.
	wantFiles := s.opt.Enabled && !matchAll && opts.Field == "" && opts.Type != "dir"
	fuse := wantFiles && scoreOrder(opts.SortBy)
	wantSemantic := s.opt.Enabled && (len(req.Collections) > 0 || wantFiles)

	type semOut struct {
		resp *SearchResponse
		err  error
		took time.Duration
	}
	semCh := make(chan semOut, 1)
	if wantSemantic {
		go func() {
			start := time.Now()
			sctx, cancel := context.WithTimeout(ctx, s.opt.SearchTimeout)
			defer cancel()
			resp, err := s.Search(sctx, SearchRequest{
				Query:           query,
				Collections:     req.Collections,
				CollectionLimit: req.CollectionLimit,
				Files:           wantFiles,
			})
			semCh <- semOut{resp, err, time.Since(start)}
		}()
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	lexOpts := *opts
	lexOpts.Limit = limit
	if fuse {
		// A deeper lexical list gives the fusion more to work with; asking for
		// more than the caller's own limit stays untouched.
		lexOpts.Limit = max(limit, min(max(2*limit, minPool), maxPool))
	}
	var lexRes *indexer.SearchResult
	var lexErr error
	if opts.Type == "all" {
		lexRes, lexErr = lex.SearchAll(&lexOpts)
	} else {
		var r *bleve.SearchResult
		if r, lexErr = lex.SearchWithOptions(&lexOpts); lexErr == nil {
			lexRes = &indexer.SearchResult{SearchResult: r}
		}
	}

	var sem *SearchResponse
	var semErr error
	if wantSemantic {
		o := <-semCh
		sem, semErr = o.resp, o.err
		info.TookMs = float64(o.took.Microseconds()) / 1000
	}
	switch {
	case !s.opt.Enabled:
		info.Reason = s.disabledErr().Error()
	case !wantSemantic:
		info.Available = true // nothing to embed for this request
	case semErr != nil:
		if ctx.Err() != nil {
			return nil, ctx.Err() // the caller gave up
		}
		info.Reason = semErr.Error()
		if errors.Is(semErr, context.DeadlineExceeded) {
			info.Reason = "semantic search timed out after " + s.opt.SearchTimeout.String()
		}
	default:
		info.Available, info.Reason = sem.Available, sem.Reason
		out.Collections = sem.Collections
	}

	if lexErr != nil {
		if out.Collections == nil {
			return nil, lexErr
		}
		out.LexicalError = lexErr.Error()
		return out, nil
	}
	out.SearchResult = lexRes
	if len(lexRes.DirectoryHits) > limit { // the pool is for file fusion only
		lexRes.DirectoryHits = lexRes.DirectoryHits[:limit]
	}

	var lists []FileList
	if sem != nil && sem.Available {
		lists = sem.Files
	}
	if !fuse {
		// Explicit sort or filename-only search: the order is the caller's choice.
		// Lexical hits only get their semantic score annotated.
		annotated := annotate(lexRes.Hits, lists)
		if len(annotated) > limit {
			annotated = annotated[:limit]
		}
		out.Hits = annotated
		return out, nil
	}
	info.Fused = len(lists) > 0
	if len(lists) == 0 {
		hits := annotate(lexRes.Hits, nil)
		if len(hits) > limit {
			hits = hits[:limit]
		}
		out.Hits = hits
		return out, nil
	}
	out.Hits, info.Added = s.fuse(lex, &lexOpts, lexRes, lists, limit)
	return out, nil
}

// scoreOrder says whether the caller wants relevance order (the default).
func scoreOrder(sortBy string) bool { return sortBy == "" || sortBy == "score" }

func annotate(hits search.DocumentMatchCollection, lists []FileList) []ResultHit {
	sem := bestScores(lists)
	out := make([]ResultHit, 0, len(hits))
	for _, h := range hits {
		rh := ResultHit{DocumentMatch: h}
		if sc, ok := sem[h.ID]; ok {
			rh.SemanticScore = &sc
		}
		out = append(out, rh)
	}
	return out
}

func bestScores(lists []FileList) map[string]float32 {
	best := map[string]float32{}
	for _, l := range lists {
		for _, h := range l.Hits {
			if cur, ok := best[h.Path]; !ok || h.Score > cur {
				best[h.Path] = h.Score
			}
		}
	}
	return best
}

type fused struct {
	path    string
	score   float64
	lexRank int // 0 = not a lexical hit
	doc     *search.DocumentMatch
}

// semanticFileRanking merges the file lists of all modalities into one ranking.
// Their z scores are on one scale, their cosines and ranks are not: each list
// has a first place even when nothing in it is relevant, so giving every list
// its own RRF term let the best screenshot of an unrelated query outrank real
// matches. A file in several lists keeps its best z.
func semanticFileRanking(lists []FileList) []FileHit {
	best := map[string]FileHit{}
	for _, l := range lists {
		for _, h := range l.Hits {
			if cur, ok := best[h.Path]; !ok || h.Z > cur.Z {
				best[h.Path] = h
			}
		}
	}
	out := make([]FileHit, 0, len(best))
	for _, h := range best {
		out = append(out, h)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Z != out[j].Z {
			return out[i].Z > out[j].Z
		}
		return out[i].Path < out[j].Path
	})
	return out
}

// fuse merges the lexical hits and the semantic file ranking with RRF and returns
// the first limit hits. Files found by embeddings only are fetched from the
// lexical index with the caller's filters, so they have the same shape as any
// other hit and cannot violate folder/extension/size/... filters.
func (s *Service) fuse(lex Lexical, opts *indexer.SearchOptions, lexRes *indexer.SearchResult, lists []FileList, limit int) ([]ResultHit, int) {
	byPath := map[string]*fused{}
	var order []*fused
	get := func(path string) *fused {
		f := byPath[path]
		if f == nil {
			f = &fused{path: path}
			byPath[path] = f
			order = append(order, f)
		}
		return f
	}
	for i, h := range lexRes.Hits {
		f := get(h.ID)
		f.doc, f.lexRank = h, i+1
		f.score += 1 / float64(rrfK+i+1)
	}
	for i, h := range semanticFileRanking(lists) {
		get(h.Path).score += 1 / float64(rrfK+i+1)
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if a.score != b.score {
			return a.score > b.score
		}
		if (a.lexRank != 0) != (b.lexRank != 0) {
			return a.lexRank != 0 // on equal score the lexical match wins
		}
		if a.lexRank != b.lexRank {
			return a.lexRank < b.lexRank
		}
		return a.path < b.path
	})

	sem := bestScores(lists)
	var out []ResultHit
	added := 0
	for next := 0; len(out) < limit && next < len(order); {
		window := order[next:min(next+(limit-len(out)), len(order))]
		next += len(window)
		var missing []*fused
		for _, f := range window {
			if f.doc == nil {
				missing = append(missing, f)
			}
		}
		fetchDocs(lex, opts, missing)
		for _, f := range window {
			if f.doc == nil {
				continue // not in the index, or filtered out
			}
			if f.lexRank == 0 {
				added++
			}
			f.doc.Score = f.score
			rh := ResultHit{DocumentMatch: f.doc}
			if sc, ok := sem[f.path]; ok {
				rh.SemanticScore = &sc
			}
			out = append(out, rh)
		}
	}
	if out == nil {
		out = []ResultHit{}
	}
	if r := lexRes.SearchResult; r != nil {
		r.Total += uint64(added)
		r.MaxScore = 0
		if len(out) > 0 {
			r.MaxScore = out[0].Score
		}
	}
	return out, added
}

// fetchDocs loads the documents of files that only the embeddings found, running
// the lexical search restricted to the exact path with the caller's filters.
func fetchDocs(lex Lexical, opts *indexer.SearchOptions, items []*fused) {
	if len(items) == 0 {
		return
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, fetchWorkers)
	for _, f := range items {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			f.doc = fetchDoc(lex, opts, f.path)
		}()
	}
	wg.Wait()
}

func fetchDoc(lex Lexical, opts *indexer.SearchOptions, path string) *search.DocumentMatch {
	o := *opts
	o.Query, o.Field, o.Fuzzy = path, "path", false
	o.Limit, o.Facets, o.SortBy, o.SortDesc = 3, nil, "", false
	if o.Type == "all" {
		o.Type = "file"
	}
	res, err := lex.SearchWithOptions(&o)
	if err != nil {
		return nil
	}
	for _, h := range res.Hits {
		if h.ID != path { // the path field is case-folded, so a differently cased sibling can match too
			continue
		}
		// What matched is the path itself, which says nothing about the user's
		// query: drop its location and highlight so the hit looks like one the
		// lexical search returned (the type filter's doc_type entries stay).
		delete(h.Locations, "path")
		delete(h.Fragments, "path")
		if len(h.Fragments) == 0 {
			h.Fragments = nil
		}
		return h
	}
	return nil
}
