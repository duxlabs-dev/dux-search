package filesidx

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// corpus copies the real fixture files into the rig and returns name -> path.
func (r *rig) corpus() map[string]string {
	r.t.Helper()
	p := map[string]string{}
	for _, f := range []string{"gencligehitabe.txt", "hitabe_windows1254.txt", "hitabe_utf16.txt", "pride_and_prejudice.txt", "sort.go", "builder.go"} {
		p[f] = r.copyFixture(filepath.Join("files", f), filepath.Join("docs", f))
	}
	for _, f := range []string{"earth.jpg", "bootprint.jpg", "cat.jpg", "error_screenshot.png"} {
		p[f] = r.copyFixture(filepath.Join("media", f), filepath.Join("pictures", f))
	}
	for _, f := range []string{"small_step.ogg", "chopin_nocturne_8s.flac"} {
		p[f] = r.copyFixture(filepath.Join("media", f), filepath.Join("sound", f))
	}
	return p
}

func (r *rig) enqueueAll(paths map[string]string) {
	for _, p := range paths {
		r.x.Enqueue(p, OpIndex)
	}
}

func TestEndToEnd(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	// Things that must be skipped, not embedded or crash the indexer.
	binary := r.write("docs/blob.txt", append([]byte("MZ\x90\x00"), make([]byte, 4096)...))
	empty := r.write("docs/empty.md", nil)
	fakePNG := r.write("pictures/broken.png", []byte("this is not a png at all"))
	unknown := r.write("docs/archive.zip", []byte("PK\x03\x04 not indexed by extension"))
	files["blob"], files["empty"], files["broken"], files["archive"] = binary, empty, fakePNG, unknown

	r.start()
	r.enqueueAll(files)
	prog := r.settle()

	if prog.Pending != 0 || prog.PausedReason != "" {
		t.Fatalf("progress %+v", prog)
	}
	if prog.Failed != 2 { // the NUL-filled text file and the broken image
		t.Errorf("failed = %d, want 2 (%+v)", prog.Failed, prog)
	}

	// Collections and ids.
	textIDs := r.store.ids(CollectionText)
	for _, id := range textIDs {
		base, n, ok := strings.Cut(id, "#")
		if !ok || n == "" || !strings.HasPrefix(base, r.root) {
			t.Fatalf("text id %q is not path#n", id)
		}
	}
	if len(textIDs) < 10 {
		t.Errorf("only %d text vectors", len(textIDs))
	}
	if got := r.store.ids(CollectionImage); len(got) != 4 {
		t.Errorf("image ids = %v", got)
	}
	if got := r.store.ids(CollectionAudio); len(got) != 2 {
		t.Errorf("audio ids = %v", got)
	}
	for _, id := range append(r.store.ids(CollectionImage), r.store.ids(CollectionAudio)...) {
		if strings.Contains(id, "#") {
			t.Errorf("media id %q must be the bare path", id)
		}
	}
	for _, skipped := range []string{binary, empty, fakePNG, unknown} {
		for _, col := range []string{CollectionText, CollectionImage, CollectionAudio} {
			for _, id := range r.store.ids(col) {
				if strings.HasPrefix(id, skipped) {
					t.Errorf("%s was embedded into %s", skipped, col)
				}
			}
		}
	}
	// Vector shapes: matryoshka cut and normalized.
	for col, dim := range map[string]int{CollectionText: DimText, CollectionImage: DimImage, CollectionAudio: DimAudio} {
		for _, v := range r.store.cols[col].vecs {
			if len(v.Vec) != dim {
				t.Fatalf("%s vector has %d dimensions, want %d", col, len(v.Vec), dim)
			}
			var n float64
			for _, x := range v.Vec {
				n += float64(x) * float64(x)
			}
			if n < 0.999 || n > 1.001 {
				t.Fatalf("%s vector norm² = %v", col, n)
			}
			if v.Hash == "" {
				t.Fatalf("%s vector without content hash", col)
			}
		}
	}

	// Real retrieval through the real model's vectors.
	for _, q := range loadQueries(t) {
		dim := map[string]int{CollectionText: DimText, CollectionImage: DimImage, CollectionAudio: DimAudio}[q.Collection]
		got := r.search(q.Collection, q.Query, dim)
		ok := false
		for _, w := range q.Want {
			ok = ok || (len(got) > 0 && got[0] == files[w])
		}
		if !ok {
			t.Errorf("%s %q: best match %v, want one of %v (ranking: %v)", q.Collection, q.Query, shortList(got[:min(3, len(got))], r.root), q.Want, shortList(got, r.root))
		}
	}
}

func shortList(paths []string, root string) []string {
	out := make([]string, len(paths))
	for i, p := range paths {
		out[i] = strings.TrimPrefix(p, root+"/")
	}
	return out
}

func TestEncodingsProduceTheSameText(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	r.start()
	for _, n := range []string{"gencligehitabe.txt", "hitabe_windows1254.txt", "hitabe_utf16.txt"} {
		r.x.Enqueue(files[n], OpIndex)
	}
	r.settle()
	r.emb.mu.Lock()
	defer r.emb.mu.Unlock()
	byTitle := map[string]string{}
	for _, d := range r.emb.docs {
		byTitle[d.Title] += d.Text + "\x00"
	}
	a, b, c := byTitle["gencligehitabe.txt"], byTitle["hitabe_windows1254.txt"], byTitle["hitabe_utf16.txt"]
	if a == "" || a != b || a != c {
		t.Fatalf("the three encodings did not decode to the same chunks:\n%q\n%q\n%q", a, b, c)
	}
}

func TestIncrementalEditsEmbedOnlyChangedChunks(t *testing.T) {
	r := newRig(t, nil)
	r.start()
	note := filepath.Join(r.root, "novel.txt")
	gen := 0
	put := func(variant string) {
		b, err := os.ReadFile(filepath.Join("testdata", "variants", "novel.txt", variant))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(note, b, 0o644); err != nil {
			t.Fatal(err)
		}
		gen++ // a distinct mtime even on coarse clocks
		tm := mtimeBase.Add(time.Duration(gen) * time.Minute)
		os.Chtimes(note, tm, tm)
		r.x.Enqueue(note, OpIndex)
		r.settle()
	}
	chunksOf := func() []string {
		var ids []string
		for _, id := range r.store.ids(CollectionText) {
			if strings.HasPrefix(id, note+"#") {
				ids = append(ids, strings.TrimPrefix(id, note+"#"))
			}
		}
		return ids
	}

	put("v1")
	n1 := len(chunksOf())
	docs1, _, _ := r.emb.calls()
	if n1 < 2 || docs1 != n1 {
		t.Fatalf("v1: %d vectors, %d documents embedded", n1, docs1)
	}

	// Appending a chapter: the leading chunks are unchanged and are not embedded again.
	put("v2")
	n2 := len(chunksOf())
	docs2, _, _ := r.emb.calls()
	if n2 <= n1 {
		t.Fatalf("v2 should have more chunks than v1: %d vs %d", n2, n1)
	}
	if redo := docs2 - docs1; redo >= n2 {
		t.Errorf("v2 embedded %d documents for %d chunks: unchanged chunks were embedded again", redo, n2)
	} else {
		t.Logf("v1 %d chunks; v2 %d chunks, %d embedded again", n1, n2, redo)
	}

	// Shrinking to a different text: stale trailing chunks are removed.
	put("v3")
	n3 := len(chunksOf())
	if n3 >= n2 {
		t.Fatalf("v3 has %d chunks, v2 had %d", n3, n2)
	}
	for i := 0; i < n3; i++ {
		if !slices.Contains(chunksOf(), itoa(i)) {
			t.Fatalf("chunk ids are not dense: %v", chunksOf())
		}
	}

	// Touching without editing re-embeds nothing.
	docs3, _, _ := r.emb.calls()
	future := mtimeBase.Add(48 * hour)
	os.Chtimes(note, future, future)
	r.x.Enqueue(note, OpIndex)
	r.settle()
	if docs4, _, _ := r.emb.calls(); docs4 != docs3 {
		t.Errorf("a touched, unchanged file was embedded again (%d documents)", docs4-docs3)
	}
	if rec, ok := mustRecord(t, r.x.st, note); !ok || rec.MTime != future.UnixNano() {
		t.Errorf("record not refreshed after touch: %+v", rec)
	}

	// Deleting removes every chunk.
	os.Remove(note)
	r.x.Enqueue(note, OpDelete)
	r.settle()
	if left := chunksOf(); len(left) != 0 {
		t.Errorf("chunks left after delete: %v", left)
	}
	if _, ok := mustRecord(t, r.x.st, note); ok {
		t.Error("record left after delete")
	}
}

func TestDeleteOfImageAndAudio(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	r.settle()
	for _, n := range []string{"earth.jpg", "small_step.ogg"} {
		os.Remove(files[n])
		r.x.Enqueue(files[n], OpDelete)
	}
	r.settle()
	if got := r.store.ids(CollectionImage); len(got) != 3 || slices.Contains(got, files["earth.jpg"]) {
		t.Errorf("images after delete: %v", shortList(got, r.root))
	}
	if got := r.store.ids(CollectionAudio); len(got) != 1 {
		t.Errorf("audio after delete: %v", shortList(got, r.root))
	}
	// A file that vanished before its index event was processed.
	ghost := filepath.Join(r.root, "ghost.txt")
	r.x.Enqueue(ghost, OpIndex)
	r.settle()
	if p := r.x.Progress(); p.Failed != 0 {
		t.Errorf("a vanished file counted as failure: %+v", p)
	}
	_ = semantic.ModalityText
}
