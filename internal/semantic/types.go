// Package semantic holds the types shared by the embedding runtime, the
// vector store and the file indexer.
package semantic

import (
	"context"
	"errors"
	"math"
)

const NativeDim = 768

type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
)

// ErrModelUnavailable means the model or the runtime library is not installed
// or semantic search is disabled; callers fall back to lexical search.
var ErrModelUnavailable = errors.New("semantic model unavailable")

type TextDoc struct {
	Title string // "none" when empty
	Text  string
}

// ImageInput is the vision tower input built by preprocess.Image.
type ImageInput struct {
	Pixels     []float32 // NumPatches*768, patch = 16x16x3, rescaled to [0,1]
	Positions  []int64   // NumPatches*2 (x, y) patch grid positions
	NumPatches int
	SoftTokens int // image_features rows the vision tower returns
}

// AudioInput is the audio tower input built by preprocess.Audio.
type AudioInput struct {
	Mel    []float32 // Frames*128 log-mel
	Mask   []bool    // Frames, true = real frame
	Frames int
}

// Embedder returns L2-normalized NativeDim vectors. Prefixes from
// config_sentence_transformers.json are applied inside: queries get
// "task: search result | query: ", documents "title: {Title} | text: ".
type Embedder interface {
	EmbedQueries(ctx context.Context, texts []string) ([][]float32, error)
	EmbedDocuments(ctx context.Context, docs []TextDoc) ([][]float32, error)
	EmbedImages(ctx context.Context, imgs []ImageInput) ([][]float32, error)
	EmbedAudio(ctx context.Context, clips []AudioInput) ([][]float32, error)
	// Warm loads the given towers ahead of use and resets the idle timer.
	Warm(ctx context.Context, mods ...Modality) error
	Status() EmbedderStatus
	Close() error
}

type EmbedderStatus struct {
	Available bool       `json:"available"`
	Reason    string     `json:"reason,omitempty"`
	Running   bool       `json:"running"`
	Loaded    []Modality `json:"loaded"`
}

// Truncate keeps the leading dim values and re-normalizes (Matryoshka).
func Truncate(v []float32, dim int) []float32 {
	if dim <= 0 || dim > len(v) {
		dim = len(v)
	}
	out := make([]float32, dim)
	var sum float64
	for i := range out {
		out[i] = v[i]
		sum += float64(v[i]) * float64(v[i])
	}
	if sum == 0 {
		return out
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range out {
		out[i] *= inv
	}
	return out
}

type Item struct {
	ID       string         `json:"id"`
	Title    string         `json:"title"`
	Text     string         `json:"text"`
	Keywords []string       `json:"keywords,omitempty"`
	Payload  map[string]any `json:"payload,omitempty"`
}

type Hit struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"` // cosine similarity
}

type Vector struct {
	ID   string
	Hash string // content hash; unchanged hash is not re-embedded
	Vec  []float32
}

// VectorStore keeps int8-quantized vectors per collection in memory, bbolt
// only for persistence.
type VectorStore interface {
	EnsureCollection(name string, dim int, model string) error
	Hashes(collection string) (map[string]string, error)
	Upsert(collection string, vecs []Vector) error
	Delete(collection string, ids []string) error
	Search(collection string, query []float32, k int) ([]Hit, error)
	Count(collection string) (int, error)
}
