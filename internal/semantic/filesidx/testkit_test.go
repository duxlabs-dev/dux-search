package filesidx

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// The tests run the real indexer on real files. Embeddings come from
// testdata/vectors.json, which testdata/gen_vectors.py produced with the real
// EmbeddingGemma 2 int8 ONNX towers (and the reference Python preprocessing)
// for exactly the texts the chunker emits and the media files under
// testdata/media. Nothing is invented: when the indexer asks for an input the
// table does not know, the helper records it in $DUX_FIXTURE_MISSING and the
// test fails, telling to regenerate the table.

type vectorTable struct {
	Text    map[string]string `json:"text"`    // sha256(title \0 text) -> base64 float32
	Tokens  map[string]int    `json:"tokens"`  // same keys: real token count of the chunk text
	Query   map[string]string `json:"query"`   // query -> base64 float32
	Image   map[string]string `json:"image"`   // sha256(file bytes) -> base64 float32
	Audio   map[string]string `json:"audio"`   // sha256(file bytes) -> base64 float32
	Comment string            `json:"comment"` // provenance
}

var (
	tableOnce sync.Once
	table     vectorTable
	tableErr  error
)

func loadTable() (*vectorTable, error) {
	tableOnce.Do(func() {
		raw, err := os.ReadFile(filepath.Join("testdata", "vectors.json"))
		if err != nil {
			tableErr = err
			return
		}
		tableErr = json.Unmarshal(raw, &table)
	})
	return &table, tableErr
}

func decodeVec(s string) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil || len(raw) != 4*semantic.NativeDim {
		return nil, fmt.Errorf("bad vector (%d bytes): %v", len(raw), err)
	}
	v := make([]float32, semantic.NativeDim)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return v, nil
}

func textKey(title, text string) string {
	h := sha256.Sum256([]byte(title + "\x00" + text))
	return hex.EncodeToString(h[:])
}

func bytesKey(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

var missingMu sync.Mutex

func recordMissing(kind string, entry any) {
	path := os.Getenv("DUX_FIXTURE_MISSING")
	if path == "" {
		return
	}
	missingMu.Lock()
	defer missingMu.Unlock()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	line, _ := json.Marshal(map[string]any{"kind": kind, "entry": entry})
	f.Write(append(line, '\n'))
}

// fixtureEmbedder implements semantic.Embedder over the recorded vectors.
type fixtureEmbedder struct {
	mu        sync.Mutex
	available bool
	docCalls  int
	docs      []semantic.TextDoc // every document embedded, in order
	imgCalls  int
	audCalls  int
	failNext  int   // the next n calls fail with err
	failErr   error // error returned while failNext > 0
	gate      func(ctx context.Context) error
}

func newFixtureEmbedder() *fixtureEmbedder { return &fixtureEmbedder{available: true} }

func (e *fixtureEmbedder) setAvailable(v bool) { e.mu.Lock(); e.available = v; e.mu.Unlock() }

func (e *fixtureEmbedder) enter(ctx context.Context) error {
	e.mu.Lock()
	avail, gate := e.available, e.gate
	var err error
	if e.failNext > 0 {
		e.failNext--
		err = e.failErr
	}
	e.mu.Unlock()
	switch {
	case !avail:
		return semantic.ErrModelUnavailable
	case err != nil:
		return err
	case gate != nil:
		return gate(ctx)
	}
	return nil
}

func lookup(m map[string]string, key string) ([]float32, error) {
	s, ok := m[key]
	if !ok {
		return nil, errMissingVector
	}
	return decodeVec(s)
}

var errMissingVector = errors.New("no recorded vector for this input: regenerate testdata/vectors.json with testdata/gen_vectors.py (see its header)")

func (e *fixtureEmbedder) EmbedQueries(ctx context.Context, texts []string) ([][]float32, error) {
	if err := e.enter(ctx); err != nil {
		return nil, err
	}
	t, err := loadTable()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i, q := range texts {
		v, err := lookup(t.Query, q)
		if err != nil {
			recordMissing("query", q)
			return nil, fmt.Errorf("query %q: %w", q, err)
		}
		out[i] = v
	}
	return out, nil
}

func (e *fixtureEmbedder) EmbedDocuments(ctx context.Context, docs []semantic.TextDoc) ([][]float32, error) {
	if err := e.enter(ctx); err != nil {
		return nil, err
	}
	t, err := loadTable()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(docs))
	var missing error
	for i, d := range docs {
		v, err := lookup(t.Text, textKey(d.Title, d.Text))
		if err != nil {
			recordMissing("text", map[string]string{"title": d.Title, "text": d.Text})
			missing = fmt.Errorf("document %q chunk %.40q: %w", d.Title, d.Text, err)
			continue
		}
		out[i] = v
	}
	if missing != nil {
		return nil, missing
	}
	e.mu.Lock()
	e.docCalls++
	e.docs = append(e.docs, docs...)
	e.mu.Unlock()
	return out, nil
}

// The recorded vectors were computed from the source *files*, so the test
// preprocessor (which stands in for preprocess.Image/Audio, owned by another
// package) hands the embedder a small token that names the file by its sha256:
// four int64 words in ImageInput.Positions, eight float32 bit patterns in
// AudioInput.Mel.

func imageToken(sum [32]byte) semantic.ImageInput {
	w := make([]int64, 4)
	for i := range w {
		w[i] = int64(binary.BigEndian.Uint64(sum[8*i:]))
	}
	return semantic.ImageInput{Positions: w}
}

func imageTokenKey(in semantic.ImageInput) string {
	b := make([]byte, 0, 32)
	for _, w := range in.Positions {
		b = binary.BigEndian.AppendUint64(b, uint64(w))
	}
	return hex.EncodeToString(b)
}

func audioToken(sum [32]byte) semantic.AudioInput {
	m := make([]float32, 8)
	for i := range m {
		m[i] = math.Float32frombits(binary.BigEndian.Uint32(sum[4*i:]))
	}
	return semantic.AudioInput{Mel: m}
}

func audioTokenKey(in semantic.AudioInput) string {
	b := make([]byte, 0, 32)
	for _, f := range in.Mel {
		b = binary.BigEndian.AppendUint32(b, math.Float32bits(f))
	}
	return hex.EncodeToString(b)
}

func (e *fixtureEmbedder) EmbedImages(ctx context.Context, imgs []semantic.ImageInput) ([][]float32, error) {
	if err := e.enter(ctx); err != nil {
		return nil, err
	}
	t, err := loadTable()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(imgs))
	for i, in := range imgs {
		v, err := lookup(t.Image, imageTokenKey(in))
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	e.mu.Lock()
	e.imgCalls += len(imgs)
	e.mu.Unlock()
	return out, nil
}

func (e *fixtureEmbedder) EmbedAudio(ctx context.Context, clips []semantic.AudioInput) ([][]float32, error) {
	if err := e.enter(ctx); err != nil {
		return nil, err
	}
	t, err := loadTable()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(clips))
	for i, in := range clips {
		v, err := lookup(t.Audio, audioTokenKey(in))
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	e.mu.Lock()
	e.audCalls += len(clips)
	e.mu.Unlock()
	return out, nil
}

func (e *fixtureEmbedder) Warm(context.Context, ...semantic.Modality) error { return nil }
func (e *fixtureEmbedder) Close() error                                     { return nil }
func (e *fixtureEmbedder) Status() semantic.EmbedderStatus {
	e.mu.Lock()
	defer e.mu.Unlock()
	st := semantic.EmbedderStatus{Available: e.available, Loaded: []semantic.Modality{semantic.ModalityText}}
	if !e.available {
		st.Reason = "model not installed"
	}
	return st
}

func (e *fixtureEmbedder) calls() (docs, imgs, auds int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.docs), e.imgCalls, e.audCalls
}

// fixturePre is the Preprocessor of the tests. A file the real preprocess
// package would reject (corrupt or unsupported) is rejected here the same way:
// it must start with a known image/audio signature.
type fixturePre struct{ rejected atomic.Int32 }

var errNotMedia = errors.New("unsupported or corrupt media file")

func (p *fixturePre) Image(r io.Reader) (semantic.ImageInput, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return semantic.ImageInput{}, err
	}
	if !(bytes.HasPrefix(b, []byte{0xFF, 0xD8, 0xFF}) || bytes.HasPrefix(b, []byte("\x89PNG")) || bytes.HasPrefix(b, []byte("GIF8")) || bytes.HasPrefix(b, []byte("BM")) || (len(b) > 12 && string(b[8:12]) == "WEBP")) {
		p.rejected.Add(1)
		return semantic.ImageInput{}, errNotMedia
	}
	return imageToken(sha256.Sum256(b)), nil
}

func (p *fixturePre) Audio(path string) (semantic.AudioInput, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return semantic.AudioInput{}, err
	}
	if !(bytes.HasPrefix(b, []byte("OggS")) || bytes.HasPrefix(b, []byte("fLaC")) || bytes.HasPrefix(b, []byte("RIFF")) || bytes.HasPrefix(b, []byte("ID3")) || bytes.HasPrefix(b, []byte{0xFF, 0xFB})) {
		p.rejected.Add(1)
		return semantic.AudioInput{}, errNotMedia
	}
	return audioToken(sha256.Sum256(b)), nil
}

// memStore is an in-memory semantic.VectorStore with exact cosine search.
// EnsureCollection with a different model name empties the collection, which
// is what the real store has to do as well.
type memStore struct {
	mu    sync.Mutex
	cols  map[string]*memCol
	calls map[string]int
}

type memCol struct {
	dim   int
	model string
	vecs  map[string]semantic.Vector
}

func newMemStore() *memStore { return &memStore{cols: map[string]*memCol{}, calls: map[string]int{}} }

func (s *memStore) EnsureCollection(name string, dim int, model string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.cols[name]
	if !ok || c.dim != dim || c.model != model {
		s.cols[name] = &memCol{dim: dim, model: model, vecs: map[string]semantic.Vector{}}
	}
	return nil
}

func (s *memStore) col(name string) (*memCol, error) {
	c, ok := s.cols[name]
	if !ok {
		return nil, fmt.Errorf("unknown collection %q", name)
	}
	return c, nil
}

func (s *memStore) Hashes(name string) (map[string]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, err := s.col(name)
	if err != nil {
		return nil, err
	}
	out := make(map[string]string, len(c.vecs))
	for id, v := range c.vecs {
		out[id] = v.Hash
	}
	return out, nil
}

func (s *memStore) Upsert(name string, vecs []semantic.Vector) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, err := s.col(name)
	if err != nil {
		return err
	}
	s.calls["upsert"]++
	for _, v := range vecs {
		if len(v.Vec) != c.dim {
			return fmt.Errorf("%s: vector of %d dimensions, collection has %d", name, len(v.Vec), c.dim)
		}
		c.vecs[v.ID] = v
	}
	return nil
}

func (s *memStore) Delete(name string, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, err := s.col(name)
	if err != nil {
		return err
	}
	s.calls["delete"]++
	for _, id := range ids {
		delete(c.vecs, id)
	}
	return nil
}

func (s *memStore) Count(name string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, err := s.col(name)
	if err != nil {
		return 0, err
	}
	return len(c.vecs), nil
}

func (s *memStore) Search(name string, q []float32, k int) ([]semantic.Hit, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, err := s.col(name)
	if err != nil {
		return nil, err
	}
	if len(q) != c.dim {
		return nil, fmt.Errorf("%s: query of %d dimensions, collection has %d", name, len(q), c.dim)
	}
	hits := make([]semantic.Hit, 0, len(c.vecs))
	for id, v := range c.vecs {
		var dot float32
		for i := range q {
			dot += q[i] * v.Vec[i]
		}
		hits = append(hits, semantic.Hit{ID: id, Score: dot})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits, nil
}

func (s *memStore) ids(name string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	if c, ok := s.cols[name]; ok {
		for id := range c.vecs {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// fixedPower and fixedActivity are the machine state a test wants: a desktop
// test machine cannot be unplugged, so these stand for the sysfs/D-Bus answers.
type fixedPower struct{ on atomic.Bool }

func (p *fixedPower) OnBattery(context.Context) (bool, error) { return p.on.Load(), nil }

type fixedActivity struct{ active atomic.Bool }

func (a *fixedActivity) UserActive(context.Context) (bool, bool, error) {
	return a.active.Load(), true, nil
}

// rig is a running indexer over a temp directory.
type rig struct {
	t      *testing.T
	root   string // files live here
	cfg    Config
	emb    *fixtureEmbedder
	store  *memStore
	pre    *fixturePre
	power  *fixedPower
	active *fixedActivity
	x      *Indexer
	cancel context.CancelFunc
}

func newRig(t *testing.T, mod func(*Config)) *rig {
	t.Helper()
	r := &rig{t: t, root: t.TempDir(), emb: newFixtureEmbedder(), store: newMemStore(), pre: &fixturePre{}, power: &fixedPower{}, active: &fixedActivity{}}
	r.cfg = DefaultConfig(filepath.Join(t.TempDir(), "state", "files.db"))
	r.cfg.Preprocessor = r.pre
	r.cfg.Power, r.cfg.Activity = r.power, r.active
	r.cfg.PolicyInterval = 20 * time.Millisecond
	r.cfg.ModelRetry = 50 * time.Millisecond
	r.cfg.LegacyEncoding = legacyEncodingFor("tr_TR.UTF-8")
	if mod != nil {
		mod(&r.cfg)
	}
	return r
}

// start launches (or relaunches, after stop) the indexer on the same state.
func (r *rig) start() {
	r.t.Helper()
	r.x = New(r.cfg, r.emb, r.store)
	ctx, cancel := context.WithCancel(context.Background())
	r.cancel = cancel
	if err := r.x.Start(ctx); err != nil {
		cancel()
		r.t.Fatal(err)
	}
	r.t.Cleanup(r.stop)
}

func (r *rig) stop() {
	if r.cancel != nil {
		r.cancel()
		r.x.Wait()
		r.cancel = nil
	}
}

// copyFixture puts a file from testdata below the rig root.
func (r *rig) copyFixture(from, to string) string {
	r.t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", from))
	if err != nil {
		r.t.Fatal(err)
	}
	return r.write(to, b)
}

func (r *rig) write(rel string, b []byte) string {
	r.t.Helper()
	p := filepath.Join(r.root, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		r.t.Fatal(err)
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		r.t.Fatal(err)
	}
	return p
}

// settle waits until the queue is empty and the worker is idle.
func (r *rig) settle() Progress {
	r.t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		p := r.x.Progress()
		if p.Pending == 0 {
			// Pending drops when results are committed, which happens after
			// the last vectors were stored.
			return p
		}
		time.Sleep(10 * time.Millisecond)
	}
	r.t.Fatalf("indexer did not settle: %+v", r.x.Progress())
	return Progress{}
}

func (r *rig) waitFor(what string, cond func() bool) {
	r.t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	r.t.Fatalf("timed out waiting for %s (progress %+v)", what, r.x.Progress())
}

// search returns the paths of the hits of a real query vector, best first,
// with chunk suffixes removed and duplicates collapsed.
func (r *rig) search(collection, query string, dim int) []string {
	r.t.Helper()
	vecs, err := r.emb.EmbedQueries(context.Background(), []string{query})
	if err != nil {
		r.t.Fatal(err)
	}
	hits, err := r.store.Search(collection, semantic.Truncate(vecs[0], dim), 50)
	if err != nil {
		r.t.Fatal(err)
	}
	var out []string
	seen := map[string]bool{}
	for _, h := range hits {
		p, _, _ := strings.Cut(h.ID, "#")
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	return out
}

const hour = time.Hour

var mtimeBase = time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

func itoa(i int) string { return fmt.Sprint(i) }
