package engine

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/embedtest"
)

const testdata = "../testdata"

func newEngine(t *testing.T, towers ...string) *Engine {
	t.Helper()
	dir := embedtest.ModelDir(t, towers...)
	e, err := New(Config{ModelDir: dir, ORTLibrary: embedtest.ORTLibrary(), Threads: 4})
	if err != nil {
		if errors.Is(err, ErrUnavailable) {
			t.Skipf("runtime not available: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(e.Close)
	return e
}

type textRef struct {
	Items []struct {
		Kind string    `json:"kind"`
		Text string    `json:"text"`
		Vec  []float32 `json:"vec"`
	} `json:"items"`
}

func loadTextRef(t *testing.T) textRef {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(testdata, "text_reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r textRef
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// TestTextMatchesPython embeds the 20 actions and 15 queries of
// degerlendirme/eylem_eslesme.py in one padded batch and checks every vector
// against Python's (cosine >= 0.999) and the retrieval result (15/15 at 768d).
func TestTextMatchesPython(t *testing.T) {
	e := newEngine(t)
	ref := loadTextRef(t)
	texts := make([]string, len(ref.Items))
	for i, it := range ref.Items {
		texts[i] = it.Text
	}
	got, err := e.EmbedTexts(texts)
	if err != nil {
		t.Fatal(err)
	}
	minCos := 1.0
	for i, v := range got {
		if len(v) != semantic.NativeDim {
			t.Fatalf("vector %d has %d values", i, len(v))
		}
		c := embedtest.Cosine(v, ref.Items[i].Vec)
		minCos = math.Min(minCos, c)
		if c < 0.999 {
			t.Errorf("%q: cosine to Python %.5f", texts[i], c)
		}
		var n float64
		for _, x := range v {
			n += float64(x) * float64(x)
		}
		if math.Abs(n-1) > 1e-5 {
			t.Errorf("vector %d not unit length: %v", i, n)
		}
	}
	t.Logf("minimum cosine to Python over %d vectors: %.6f", len(got), minCos)

	var cfg struct {
		Actions []struct{ Key string }  `json:"actions"`
		Queries []struct{ Want string } `json:"queries"`
	}
	raw, _ := os.ReadFile(filepath.Join(testdata, "actions.json"))
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	na := len(cfg.Actions)
	hits := 0
	for qi, q := range cfg.Queries {
		qv := got[na+qi]
		best, bestScore := -1, -2.0
		for ai := 0; ai < na; ai++ {
			if s := embedtest.Cosine(qv, got[ai]); s > bestScore {
				best, bestScore = ai, s
			}
		}
		if cfg.Actions[best].Key == q.Want {
			hits++
		} else {
			t.Errorf("query %q: best action %s (%.3f), want %s", ref.Items[na+qi].Text, cfg.Actions[best].Key, bestScore, q.Want)
		}
	}
	if hits != len(cfg.Queries) {
		t.Fatalf("first-rank hits %d/%d", hits, len(cfg.Queries))
	}
	t.Logf("first-rank hits %d/%d at 768d", hits, len(cfg.Queries))
}

// Padding and batching must not change a vector: one text alone, the same
// text padded inside a batch with a much longer one.
func TestBatchingInvariance(t *testing.T) {
	e := newEngine(t)
	short := "task: search result | query: gözlerim yoruldu"
	long := "title: none | text: "
	for i := 0; i < 300; i++ {
		long += "Night Light warms the screen colors. "
	}
	alone, err := e.EmbedTexts([]string{short})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := e.EmbedTexts([]string{long, short, short, long})
	if err != nil {
		t.Fatal(err)
	}
	if c := embedtest.Cosine(alone[0], batch[1]); c < 0.9999 {
		t.Errorf("padded batch changes the vector: cosine %.6f", c)
	}
	if c := embedtest.Cosine(batch[0], batch[3]); c < 0.99999 {
		t.Errorf("identical texts differ: cosine %.6f", c)
	}
	// Result order follows input order even though runs are length-sorted.
	if c := embedtest.Cosine(batch[1], batch[2]); c < 0.99999 {
		t.Errorf("identical short texts differ: %.6f", c)
	}
	if c := embedtest.Cosine(batch[0], batch[1]); c > 0.9 {
		t.Errorf("long and short texts should differ, cosine %.4f", c)
	}
}

func TestLongTextIsTruncatedToContext(t *testing.T) {
	if testing.Short() {
		t.Skip("a full 8K-token run takes about half a minute")
	}
	e := newEngine(t)
	long := "task: search result | query: "
	for i := 0; i < 4000; i++ {
		long += "gözlerim yoruldu ve ekran çok parlak. "
	}
	v, err := e.EmbedTexts([]string{long})
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 1 || len(v[0]) != semantic.NativeDim {
		t.Fatalf("got %d vectors", len(v))
	}
}

func TestEmptyBatchAndEmptyText(t *testing.T) {
	e := newEngine(t)
	if v, err := e.EmbedTexts(nil); err != nil || len(v) != 0 {
		t.Fatalf("empty batch: %v %v", v, err)
	}
	v, err := e.EmbedTexts([]string{""})
	if err != nil || len(v) != 1 || len(v[0]) != semantic.NativeDim {
		t.Fatalf("empty text: %v %v", v, err)
	}
}

func TestLoadUnload(t *testing.T) {
	e := newEngine(t)
	if len(e.Loaded()) != 0 {
		t.Fatalf("loaded before use: %v", e.Loaded())
	}
	if _, err := e.EmbedTexts([]string{"a"}); err != nil {
		t.Fatal(err)
	}
	if l := e.Loaded(); len(l) != 1 || l[0] != semantic.ModalityText {
		t.Fatalf("loaded %v", l)
	}
	e.Unload(semantic.ModalityText)
	if len(e.Loaded()) != 0 {
		t.Fatalf("still loaded: %v", e.Loaded())
	}
	// Using it again reloads transparently.
	if _, err := e.EmbedTexts([]string{"a"}); err != nil {
		t.Fatal(err)
	}
}

func TestModelMissing(t *testing.T) {
	_, err := New(Config{ModelDir: filepath.Join(t.TempDir(), "none")})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("want ErrUnavailable, got %v", err)
	}
	// config.json present, tower files absent.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	e, err := New(Config{ModelDir: dir, ORTLibrary: embedtest.ORTLibrary()})
	if err != nil {
		if errors.Is(err, ErrUnavailable) {
			t.Skip("runtime not installed")
		}
		t.Fatal(err)
	}
	defer e.Close()
	if _, err := e.EmbedTexts([]string{"x"}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("want ErrUnavailable, got %v", err)
	}
}

// TestImageMatchesPython feeds the tensors transformers' processor builds for
// a real image (patches with the padding dropped) to the vision tower and the
// text tower and compares with Python's embedding.
func TestImageMatchesPython(t *testing.T) {
	e := newEngine(t, "vision_encoder")
	ref := embedtest.LoadRef(t, filepath.Join(testdata, "image_reference.dref"))
	in := semantic.ImageInput{
		Pixels: ref.Float32("pixels"), Positions: ref.Int64("positions"),
		NumPatches: ref.Int("patches"), SoftTokens: ref.Int("soft_tokens"),
	}
	got, err := e.EmbedImages([]semantic.ImageInput{in})
	if err != nil {
		t.Fatal(err)
	}
	c := embedtest.Cosine(got[0], ref.Float32("embedding"))
	t.Logf("image: %d patches, %d soft tokens, cosine to Python %.6f", in.NumPatches, in.SoftTokens, c)
	if c < 0.999 {
		t.Fatalf("cosine %.5f < 0.999", c)
	}
	// A batch of two identical images gives the same vector twice.
	two, err := e.EmbedImages([]semantic.ImageInput{in, in})
	if err != nil {
		t.Fatal(err)
	}
	if embedtest.Cosine(two[0], got[0]) < 0.9999 || embedtest.Cosine(two[1], got[0]) < 0.9999 {
		t.Fatal("batched image embeddings differ from the single run")
	}
}

func TestImageRejectsBadInput(t *testing.T) {
	e := newEngine(t, "vision_encoder")
	ref := embedtest.LoadRef(t, filepath.Join(testdata, "image_reference.dref"))
	good := semantic.ImageInput{
		Pixels: ref.Float32("pixels"), Positions: ref.Int64("positions"),
		NumPatches: ref.Int("patches"), SoftTokens: ref.Int("soft_tokens"),
	}
	bad := good
	bad.SoftTokens++ // disagrees with what the tower returns
	if _, err := e.EmbedImages([]semantic.ImageInput{bad}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("wrong soft token count: %v", err)
	}
	bad = good
	bad.Pixels = bad.Pixels[:len(bad.Pixels)-1]
	if _, err := e.EmbedImages([]semantic.ImageInput{bad}); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("short pixels: %v", err)
	}
}

// TestAudioMatchesPython does the same for a real speech clip.
func TestAudioMatchesPython(t *testing.T) {
	e := newEngine(t, "audio_encoder")
	ref := embedtest.LoadRef(t, filepath.Join(testdata, "audio_reference.dref"))
	in := semantic.AudioInput{Mel: ref.Float32("mel"), Mask: ref.Bool("mask"), Frames: ref.Int("frames")}
	got, err := e.EmbedAudio([]semantic.AudioInput{in})
	if err != nil {
		t.Fatal(err)
	}
	c := embedtest.Cosine(got[0], ref.Float32("embedding"))
	t.Logf("audio: %d frames, %d tokens, cosine to Python %.6f", in.Frames, ref.Int("tokens"), c)
	if c < 0.999 {
		t.Fatalf("cosine %.5f < 0.999", c)
	}
}

// Images, audio and text share one vector space: the spoken "walk sign is on"
// announcement must sit closer to a matching text than to an unrelated one.
func TestAudioNearItsDescription(t *testing.T) {
	e := newEngine(t, "audio_encoder")
	ref := embedtest.LoadRef(t, filepath.Join(testdata, "audio_reference.dref"))
	in := semantic.AudioInput{Mel: ref.Float32("mel"), Mask: ref.Bool("mask"), Frames: ref.Int("frames")}
	a, err := e.EmbedAudio([]semantic.AudioInput{in})
	if err != nil {
		t.Fatal(err)
	}
	txt, err := e.EmbedTexts([]string{
		"task: search result | query: a voice announcing that the walk sign is on to cross the street",
		"task: search result | query: recipe for chocolate cake with vanilla frosting",
	})
	if err != nil {
		t.Fatal(err)
	}
	near, far := embedtest.Cosine(a[0], txt[0]), embedtest.Cosine(a[0], txt[1])
	t.Logf("audio vs matching text %.3f, vs unrelated %.3f", near, far)
	if near <= far {
		t.Fatalf("audio is not closer to its description (%.3f <= %.3f)", near, far)
	}
}

// Several media items in one padded batch keep their own features: the order
// the tower rows are concatenated in must match the order of the sequences.
func TestMixedMediaBatchKeepsOrder(t *testing.T) {
	e := newEngine(t, "vision_encoder", "audio_encoder")

	iref := embedtest.LoadRef(t, filepath.Join(testdata, "image_reference.dref"))
	a := semantic.ImageInput{Pixels: iref.Float32("pixels"), Positions: iref.Int64("positions"), NumPatches: iref.Int("patches"), SoftTokens: iref.Int("soft_tokens")}
	b := a // same pixels on a transposed grid: a different, still valid input
	b.Positions = make([]int64, len(a.Positions))
	for i := 0; i+1 < len(b.Positions); i += 2 {
		b.Positions[i], b.Positions[i+1] = a.Positions[i+1], a.Positions[i]
	}
	ia, err := e.EmbedImages([]semantic.ImageInput{a})
	if err != nil {
		t.Fatal(err)
	}
	ib, err := e.EmbedImages([]semantic.ImageInput{b})
	if err != nil {
		t.Skipf("transposed grid is not a valid input for the tower here: %v", err)
	}
	if embedtest.Cosine(ia[0], ib[0]) > 0.9999 {
		t.Skip("transposing the grid did not change the embedding; cannot observe ordering")
	}
	both, err := e.EmbedImages([]semantic.ImageInput{a, b, a})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range [][]float32{ia[0], ib[0], ia[0]} {
		if c := embedtest.Cosine(both[i], want); c < 0.9999 {
			t.Errorf("image %d in a batch differs from its single run: cosine %.6f", i, c)
		}
	}

	aref := embedtest.LoadRef(t, filepath.Join(testdata, "audio_reference.dref"))
	full := semantic.AudioInput{Mel: aref.Float32("mel"), Mask: aref.Bool("mask"), Frames: aref.Int("frames")}
	n := 90 // a shorter clip: the first n frames
	short := semantic.AudioInput{Mel: full.Mel[:n*128], Mask: full.Mask[:n], Frames: n}
	sf, err := e.EmbedAudio([]semantic.AudioInput{full})
	if err != nil {
		t.Fatal(err)
	}
	ss, err := e.EmbedAudio([]semantic.AudioInput{short})
	if err != nil {
		t.Fatal(err)
	}
	both2, err := e.EmbedAudio([]semantic.AudioInput{short, full, short})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range [][]float32{ss[0], sf[0], ss[0]} {
		if c := embedtest.Cosine(both2[i], want); c < 0.9999 {
			t.Errorf("clip %d in a batch differs from its single run: cosine %.6f", i, c)
		}
	}
	if embedtest.Cosine(sf[0], ss[0]) > 0.9999 {
		t.Error("a truncated clip should embed differently from the full clip")
	}
}
