// Package fixture serves vectors that the real embeddinggemma-2 int8 text tower
// produced for a fixed set of queries and documents (see gen_vectors.py). Tests
// use it wherever they need an semantic.Embedder: it answers with the true model
// output for the known texts and fails loudly for anything else, so no test ever
// runs on invented vectors.
package fixture

import (
	"context"
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

//go:embed data/fixture.json
var manifestJSON []byte

//go:embed data/vectors.f32
var vectorsRaw []byte

// Query is one search phrase with the item it must find.
type Query struct {
	Kind   string    `json:"kind"` // "action" or "file"
	Text   string    `json:"text"`
	Expect string    `json:"expect"` // action id, or Go package (file kind)
	Vec    []float32 `json:"-"`
}

// Doc is one document (an action description or a chunk of package documentation).
type Doc struct {
	Kind  string    `json:"kind"` // "action" or "file"
	ID    string    `json:"id"`   // action id
	Pkg   string    `json:"pkg"`  // Go package path, file kind
	Chunk int       `json:"chunk"`
	Title string    `json:"title"`
	Text  string    `json:"text"`
	Vec   []float32 `json:"-"`
}

// Data is the whole fixture. Vectors are NativeDim long and L2-normalized.
type Data struct {
	Model   string  `json:"model"`
	Dim     int     `json:"dim"`
	Queries []Query `json:"queries"`
	Docs    []Doc   `json:"docs"`
}

var (
	once   sync.Once
	loaded *Data
	loadEr error
)

// Load decodes the embedded fixture once.
func Load() (*Data, error) {
	once.Do(func() { loaded, loadEr = decode() })
	return loaded, loadEr
}

// MustLoad is Load for tests.
func MustLoad() *Data {
	d, err := Load()
	if err != nil {
		panic(err)
	}
	return d
}

func decode() (*Data, error) {
	var d Data
	if err := json.Unmarshal(manifestJSON, &d); err != nil {
		return nil, err
	}
	rows := len(d.Queries) + len(d.Docs)
	if d.Dim != semantic.NativeDim || len(vectorsRaw) != rows*d.Dim*4 {
		return nil, fmt.Errorf("fixture: %d bytes of vectors do not match %d rows of dim %d", len(vectorsRaw), rows, d.Dim)
	}
	row := func(i int) []float32 {
		v := make([]float32, d.Dim)
		for j := range v {
			v[j] = math.Float32frombits(binary.LittleEndian.Uint32(vectorsRaw[(i*d.Dim+j)*4:]))
		}
		return v
	}
	for i := range d.Queries {
		d.Queries[i].Vec = row(i)
	}
	for i := range d.Docs {
		d.Docs[i].Vec = row(len(d.Queries) + i)
	}
	return &d, nil
}

// ActionDocs returns the action-kind documents.
func (d *Data) ActionDocs() []Doc { return d.docs("action") }

// FileDocs returns the file-kind documents.
func (d *Data) FileDocs() []Doc { return d.docs("file") }

func (d *Data) docs(kind string) []Doc {
	var out []Doc
	for _, x := range d.Docs {
		if x.Kind == kind {
			out = append(out, x)
		}
	}
	return out
}

// QueriesOf returns the queries of one kind.
func (d *Data) QueriesOf(kind string) []Query {
	var out []Query
	for _, q := range d.Queries {
		if q.Kind == kind {
			out = append(out, q)
		}
	}
	return out
}

// Embedder is a semantic.Embedder backed by the fixture.
type Embedder struct {
	mu        sync.Mutex
	queries   map[string][]float32
	docs      map[string][]float32
	available bool
	reason    string
	docCalls  int
	docsDone  int
	queryDone int
	warmed    []semantic.Modality
	delay     func(ctx context.Context) error
}

// NewEmbedder returns an available embedder over the fixture.
func NewEmbedder() *Embedder {
	d := MustLoad()
	e := &Embedder{queries: map[string][]float32{}, docs: map[string][]float32{}, available: true}
	for _, q := range d.Queries {
		e.queries[q.Text] = q.Vec
	}
	for _, x := range d.Docs {
		e.docs[docKey(semantic.TextDoc{Title: x.Title, Text: x.Text})] = x.Vec
	}
	return e
}

func docKey(d semantic.TextDoc) string { return d.Title + "\x00" + d.Text }

// SetAvailable simulates the model or runtime library appearing or disappearing.
func (e *Embedder) SetAvailable(ok bool, reason string) {
	e.mu.Lock()
	e.available, e.reason = ok, reason
	e.mu.Unlock()
}

// SetDelay makes every call wait on fn before answering (nil = no wait).
func (e *Embedder) SetDelay(fn func(ctx context.Context) error) {
	e.mu.Lock()
	e.delay = fn
	e.mu.Unlock()
}

// DocsEmbedded counts the documents embedded so far (re-embedding shows up here).
func (e *Embedder) DocsEmbedded() int { e.mu.Lock(); defer e.mu.Unlock(); return e.docsDone }

// QueriesEmbedded counts the queries embedded so far.
func (e *Embedder) QueriesEmbedded() int { e.mu.Lock(); defer e.mu.Unlock(); return e.queryDone }

// Warmed returns the modalities passed to Warm.
func (e *Embedder) Warmed() []semantic.Modality {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]semantic.Modality(nil), e.warmed...)
}

func (e *Embedder) gate(ctx context.Context) error {
	e.mu.Lock()
	ok, reason, delay := e.available, e.reason, e.delay
	e.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w: %s", semantic.ErrModelUnavailable, reason)
	}
	if delay != nil {
		return delay(ctx)
	}
	return ctx.Err()
}

func (e *Embedder) EmbedQueries(ctx context.Context, texts []string) ([][]float32, error) {
	if err := e.gate(ctx); err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, ok := e.queries[t]
		if !ok {
			return nil, fmt.Errorf("fixture has no model vector for query %q", t)
		}
		out[i] = append([]float32(nil), v...)
	}
	e.mu.Lock()
	e.queryDone += len(texts)
	e.mu.Unlock()
	return out, nil
}

func (e *Embedder) EmbedDocuments(ctx context.Context, docs []semantic.TextDoc) ([][]float32, error) {
	if err := e.gate(ctx); err != nil {
		return nil, err
	}
	out := make([][]float32, len(docs))
	for i, d := range docs {
		v, ok := e.docs[docKey(d)]
		if !ok {
			return nil, fmt.Errorf("fixture has no model vector for document %q", d.Title)
		}
		out[i] = append([]float32(nil), v...)
	}
	e.mu.Lock()
	e.docsDone += len(docs)
	e.docCalls++
	e.mu.Unlock()
	return out, nil
}

func (e *Embedder) EmbedImages(context.Context, []semantic.ImageInput) ([][]float32, error) {
	return nil, fmt.Errorf("%w: fixture has no image tower", semantic.ErrModelUnavailable)
}

func (e *Embedder) EmbedAudio(context.Context, []semantic.AudioInput) ([][]float32, error) {
	return nil, fmt.Errorf("%w: fixture has no audio tower", semantic.ErrModelUnavailable)
}

func (e *Embedder) Warm(ctx context.Context, mods ...semantic.Modality) error {
	if err := e.gate(ctx); err != nil {
		return err
	}
	e.mu.Lock()
	e.warmed = append(e.warmed, mods...)
	e.mu.Unlock()
	return nil
}

func (e *Embedder) Status() semantic.EmbedderStatus {
	e.mu.Lock()
	defer e.mu.Unlock()
	st := semantic.EmbedderStatus{Available: e.available, Reason: e.reason, Loaded: []semantic.Modality{}}
	if e.available {
		st.Running = true
		st.Loaded = []semantic.Modality{semantic.ModalityText}
	}
	return st
}

func (e *Embedder) Close() error { return nil }

var _ semantic.Embedder = (*Embedder)(nil)
