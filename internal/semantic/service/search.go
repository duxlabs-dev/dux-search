package service

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// Candidate selection for file search. Raw embeddinggemma-2 cosines are
// compressed (related and unrelated files both land between 0.6 and 0.8) and
// their level differs per modality (audio clips sit higher than screenshots),
// so neither an absolute floor nor a band below the best hit separates matches
// from noise, and ranks of different modalities are not comparable either.
// Files are therefore scored with the centroid-centred cosine
// (vecstore.SearchCentered) and judged by how far a file stands out from the
// rest of its own collection for this query: z = (score - mean) / std over all
// rows. Measured on real files (162 screenshots, voice clips and documents,
// 19 queries with a known answer and 4 with none): true matches reach z 3.0-4.4,
// the best hit of a query without an answer stays below 2.7 in 11 of 12
// collection/query pairs. fileMinZ keeps a file only above that line; the one
// exception left is a hub screenshot at z 3.2.
//
// With fewer than fileMinRows vectors the distribution says nothing, so such a
// collection adds no candidates. The pool bounds the work of the lexical fetch.
const (
	fileMinZ    = 3.0
	fileMinRows = 20
	filePool    = 30
)

// CollectionHits is the ranked result of one collection.
type CollectionHits struct {
	Collection string         `json:"collection"`
	Hits       []semantic.Hit `json:"hits"`
}

// FileHit is a file found by embedding similarity: the best centred cosine among
// the vectors of the path (one per text chunk, or the image/audio vector) and how
// far it stands above its collection for this query.
type FileHit struct {
	Path  string
	Score float32
	Z     float64
}

// FileList is the ranked files of one files.* collection.
type FileList struct {
	Collection string
	Hits       []FileHit
}

// SearchRequest asks for semantic results.
type SearchRequest struct {
	Query string
	// Collections are caller-owned collections to search, best collection_limit each.
	Collections     []string
	CollectionLimit int
	// Files also searches the files.* collections.
	Files bool
}

// SearchResponse is what Search found. Available false (with Reason) means the
// model could not embed the query; callers fall back to lexical results.
type SearchResponse struct {
	Available   bool
	Reason      string
	Collections []CollectionHits
	Files       []FileList
}

// Search embeds the query once and searches the requested collections and the
// file collections with it. It does not fail when the model is missing: the
// response says so and the caller keeps its lexical results.
func (s *Service) Search(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	if !s.opt.Enabled {
		return &SearchResponse{Reason: s.disabledErr().Error()}, nil
	}
	query := strings.TrimSpace(req.Query)
	if query == "" {
		return nil, errors.New("query is required")
	}
	limit := req.CollectionLimit
	switch {
	case limit <= 0:
		limit = defaultCollectionHits
	case limit > maxCollectionHits:
		limit = maxCollectionHits
	}
	for _, name := range req.Collections {
		if err := validCollectionName(name); err != nil {
			return nil, err
		}
	}

	vecs, err := s.opt.Embedder.EmbedQueries(ctx, []string{query})
	if err == nil && len(vecs) != 1 {
		err = fmt.Errorf("embedder returned %d vectors for one query", len(vecs))
	}
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		s.noteEmbedError(err)
		return &SearchResponse{Reason: reasonOf(err)}, nil
	}
	qv := vecs[0]

	resp := &SearchResponse{Available: true}
	for _, name := range req.Collections {
		hits, err := s.searchCollection(name, qv, limit)
		if err != nil {
			return nil, err
		}
		resp.Collections = append(resp.Collections, CollectionHits{Collection: name, Hits: hits})
	}
	if req.Files {
		for _, name := range []string{CollFilesText, CollFilesImage, CollFilesAudio} {
			list, err := s.searchFiles(name, qv)
			if err != nil {
				return nil, err
			}
			if len(list.Hits) > 0 {
				resp.Files = append(resp.Files, list)
			}
		}
	}
	return resp, nil
}

func reasonOf(err error) string {
	if errors.Is(err, semantic.ErrModelUnavailable) {
		return strings.TrimPrefix(err.Error(), semantic.ErrModelUnavailable.Error()+": ")
	}
	return "embedding failed: " + err.Error()
}

func (s *Service) searchCollection(name string, qv []float32, k int) ([]semantic.Hit, error) {
	info, ok := s.opt.Store.Info(name)
	if !ok {
		return []semantic.Hit{}, nil
	}
	q, err := fit(qv, info.Dim)
	if err != nil {
		return nil, err
	}
	hits, err := s.opt.Store.Search(name, q, k)
	if err != nil {
		return nil, err
	}
	if hits == nil {
		hits = []semantic.Hit{}
	}
	return hits, nil
}

// searchFiles returns the candidate paths of one file collection: best centred
// cosine per path, at least fileMinZ above the collection, best first.
func (s *Service) searchFiles(name string, qv []float32) (FileList, error) {
	list := FileList{Collection: name}
	info, ok := s.opt.Store.Info(name)
	if !ok || info.Count < fileMinRows {
		return list, nil
	}
	q, err := fit(qv, info.Dim)
	if err != nil {
		return list, err
	}
	// Several chunks of one file compete for the pool, so look deeper than the pool.
	hits, st, err := s.opt.Store.SearchCentered(name, q, filePool*4)
	if err != nil {
		return list, err
	}
	best := map[string]float32{}
	order := []string{}
	for _, h := range hits {
		if st.Z(h.Score) < fileMinZ {
			break // hits are sorted, the rest is below the line too
		}
		p := h.ID
		if name == CollFilesText {
			p = PathOf(h.ID) // only text chunks carry the #n suffix
		}
		if cur, seen := best[p]; !seen {
			best[p] = h.Score
			order = append(order, p)
		} else if h.Score > cur {
			best[p] = h.Score
		}
	}
	sort.SliceStable(order, func(i, j int) bool { return best[order[i]] > best[order[j]] })
	for _, p := range order[:min(len(order), filePool)] {
		list.Hits = append(list.Hits, FileHit{Path: p, Score: best[p], Z: st.Z(best[p])})
	}
	return list, nil
}

// PathOf maps a vector id of a files.* collection to its file path: text chunks
// are "path#n" (images and audio use the path as is, do not call this for them).
// A '#' that is part of the file name is told apart by the numeric suffix.
func PathOf(id string) string {
	i := strings.LastIndexByte(id, '#')
	if i <= 0 || i == len(id)-1 {
		return id
	}
	if _, err := strconv.Atoi(id[i+1:]); err != nil {
		return id
	}
	return id[:i]
}
