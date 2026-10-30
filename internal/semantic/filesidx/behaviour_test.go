package filesidx

import (
	"context"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

func listFiles(paths map[string]string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, p := range paths {
			if !yield(p) {
				return
			}
		}
	}
}

func allIDs(s *memStore) []string {
	var ids []string
	for _, c := range []string{CollectionText, CollectionImage, CollectionAudio} {
		ids = append(ids, s.ids(c)...)
	}
	return ids
}

func TestQueueSurvivesRestart(t *testing.T) {
	// Reference: an uninterrupted run over the same files.
	ref := newRig(t, nil)
	refFiles := ref.corpus()
	ref.start()
	ref.enqueueAll(refFiles)
	ref.settle()
	wantCount := len(allIDs(ref.store))

	r := newRig(t, nil)
	files := r.corpus()
	var calls atomic.Int32
	r.emb.gate = func(ctx context.Context) error {
		if calls.Add(1) > 3 { // the process "dies" after three embedder calls
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}
	r.start()
	r.enqueueAll(files)
	r.waitFor("three embedder calls", func() bool { return calls.Load() > 3 })
	r.stop() // events still in memory are written on the way out

	// A new process: same state file and vector store.
	r.emb.gate = nil
	r.start()
	if p := r.x.Progress(); p.Pending == 0 {
		t.Fatalf("nothing left in the queue after the restart: %+v", p)
	}
	r.settle()
	if got := len(allIDs(r.store)); got != wantCount {
		t.Errorf("after the restart %d vectors, an uninterrupted run has %d", got, wantCount)
	}
	if p := r.x.Progress(); p.Indexed != len(files) {
		t.Errorf("records: %d for %d files", p.Indexed, len(files))
	}
}

func TestNothingLostWhenStoppedRightAway(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	r.start()
	r.enqueueAll(files) // still buffered in memory
	r.stop()
	r.start()
	r.settle()
	if n := len(r.store.ids(CollectionImage)); n != 4 {
		t.Errorf("%d images indexed after stop/start", n)
	}
	if n := len(r.store.ids(CollectionAudio)); n != 2 {
		t.Errorf("%d audio files indexed after stop/start", n)
	}
}

func TestPausesOnBatteryAndResumes(t *testing.T) {
	r := newRig(t, nil)
	r.power.on.Store(true)
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	r.waitFor("battery pause", func() bool { return r.x.Progress().PausedReason == PauseBattery })
	time.Sleep(150 * time.Millisecond)
	if docs, imgs, auds := r.emb.calls(); docs+imgs+auds != 0 {
		t.Fatalf("embedded %d/%d/%d inputs on battery", docs, imgs, auds)
	}
	if p := r.x.Progress(); p.Done != 0 || p.Pending != len(files) {
		t.Fatalf("progress on battery: %+v", p)
	}
	r.power.on.Store(false)
	p := r.settle()
	if p.PausedReason != "" || p.Done < int64(len(files)) {
		t.Fatalf("after plugging in: %+v", p)
	}
}

func TestBatteryPolicyCanBeTurnedOff(t *testing.T) {
	r := newRig(t, func(c *Config) { c.PauseOnBattery = false })
	r.power.on.Store(true)
	r.start()
	r.x.Enqueue(r.copyFixture("files/sort.go", "sort.go"), OpIndex)
	r.settle()
	if n := len(r.store.ids(CollectionText)); n == 0 {
		t.Fatal("pause_on_battery=false still paused")
	}
}

func TestPausesWhileUserIsActive(t *testing.T) {
	r := newRig(t, nil)
	r.active.active.Store(true)
	r.start()
	r.x.Enqueue(r.copyFixture("files/sort.go", "sort.go"), OpIndex)
	r.waitFor("user pause", func() bool { return r.x.Progress().PausedReason == PauseUserActive })
	time.Sleep(100 * time.Millisecond)
	if docs, _, _ := r.emb.calls(); docs != 0 {
		t.Fatalf("embedded %d documents while the user was active", docs)
	}
	r.active.active.Store(false)
	r.settle()
	if len(r.store.ids(CollectionText)) == 0 {
		t.Fatal("nothing indexed once the user went idle")
	}
}

func TestWaitsForModel(t *testing.T) {
	r := newRig(t, nil)
	r.emb.setAvailable(false)
	r.start()
	p := r.copyFixture("files/builder.go", "builder.go")
	r.x.Enqueue(p, OpIndex)
	r.waitFor("model pause", func() bool { return r.x.Progress().PausedReason == PauseModelUnavailable })
	if docs, _, _ := r.emb.calls(); docs != 0 {
		t.Fatal("embedder called while unavailable")
	}
	if pr := r.x.Progress(); pr.Pending != 1 || pr.Failed != 0 {
		t.Fatalf("progress without model: %+v", pr)
	}
	r.emb.setAvailable(true) // "download finished"
	r.settle()
	if len(r.store.ids(CollectionText)) == 0 {
		t.Fatal("not indexed after the model arrived")
	}
}

func TestModelVanishesMidWork(t *testing.T) {
	r := newRig(t, nil)
	var n atomic.Int32
	r.emb.gate = func(context.Context) error {
		if n.Add(1) == 1 {
			return semantic.ErrModelUnavailable // reported by the call, Status still says available
		}
		return nil
	}
	r.start()
	r.x.Enqueue(r.copyFixture("files/builder.go", "builder.go"), OpIndex)
	r.settle()
	if len(r.store.ids(CollectionText)) == 0 || r.x.Progress().Failed != 0 {
		t.Fatalf("file lost after ErrModelUnavailable: %+v", r.x.Progress())
	}
}

func TestReconcileCatchesUp(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	note := r.copyFixture("variants/novel.txt/v1", "novel.txt")
	files["novel.txt"] = note
	r.start()
	r.enqueueAll(files)
	r.settle()
	docsBefore, _, _ := r.emb.calls()

	// While the indexer was not told: one file edited, one deleted, one added.
	b, _ := os.ReadFile(filepath.Join("testdata", "variants", "novel.txt", "v2"))
	os.WriteFile(note, b, 0o644)
	later := mtimeBase.Add(10 * hour)
	os.Chtimes(note, later, later)
	os.Remove(files["cat.jpg"])
	delete(files, "cat.jpg")
	files["extra.go"] = r.copyFixture("files/builder.go", "extra/builder.go")

	if err := r.x.Reconcile(context.Background(), listFiles(files)); err != nil {
		t.Fatal(err)
	}
	r.settle()

	if ids := r.store.ids(CollectionImage); len(ids) != 3 || slices.ContainsFunc(ids, func(s string) bool { return strings.HasSuffix(s, "cat.jpg") }) {
		t.Errorf("deleted image still indexed: %v", shortList(ids, r.root))
	}
	if !slices.ContainsFunc(r.store.ids(CollectionText), func(s string) bool { return strings.HasPrefix(s, files["extra.go"]+"#") }) {
		t.Error("new file not indexed")
	}
	novelChunks := 0
	for _, id := range r.store.ids(CollectionText) {
		if strings.HasPrefix(id, note+"#") {
			novelChunks++
		}
	}
	if want := len(chunksOfFixture(t, "variants/novel.txt/v2")); novelChunks != want {
		t.Errorf("edited file has %d chunks, v2 has %d", novelChunks, want)
	}
	// Only the edited, new and removed files were touched: the other documents
	// were not embedded again.
	docsAfter, imgs, _ := r.emb.calls()
	if redo := docsAfter - docsBefore; redo > 4+len(chunksOfFixture(t, "files/builder.go")) {
		t.Errorf("reconcile embedded %d documents; unchanged files must be skipped", redo)
	}
	if imgs != 4 {
		t.Errorf("images embedded %d times, once each expected", imgs)
	}
}

func chunksOfFixture(t *testing.T, rel string) []string {
	t.Helper()
	text, _, err := readText(filepath.Join("testdata", rel), 2<<20, nil)
	if err != nil {
		t.Fatal(err)
	}
	c, _ := splitText(text, 64)
	return c
}

func TestStoreRepairOnStart(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	r.settle()
	r.stop()
	want := allIDs(r.store)

	// 1. The vector store lost some vectors and gained strays.
	lost := r.store.ids(CollectionImage)[:2]
	r.store.mu.Lock()
	for _, id := range lost {
		delete(r.store.cols[CollectionImage].vecs, id)
	}
	r.store.mu.Unlock()
	stray := semantic.Vector{ID: "/deleted/long/ago.txt#0", Hash: "x", Vec: make([]float32, DimText)}
	stray2 := semantic.Vector{ID: "/deleted/photo.jpg", Hash: "y", Vec: make([]float32, DimImage)}
	r.store.Upsert(CollectionText, []semantic.Vector{stray})
	r.store.Upsert(CollectionImage, []semantic.Vector{stray2})
	docsBefore, imgsBefore, _ := r.emb.calls()

	r.start()
	r.settle()
	if got := allIDs(r.store); !slices.Equal(got, want) {
		t.Errorf("store after repair differs:\n got %v\nwant %v", shortList(got, r.root), shortList(want, r.root))
	}
	docs, imgs, _ := r.emb.calls()
	if docs != docsBefore {
		t.Errorf("repair re-embedded %d text chunks that were still stored", docs-docsBefore)
	}
	if imgs-imgsBefore != 2 {
		t.Errorf("repair re-embedded %d images, want exactly the 2 that were lost", imgs-imgsBefore)
	}
	r.stop()

	// 2. A brand new, empty vector store (for example the database was removed).
	r.store = newMemStore()
	r.start()
	r.settle()
	if got := allIDs(r.store); !slices.Equal(got, want) {
		t.Errorf("rebuilt store differs: %d vs %d vectors", len(got), len(want))
	}
}

func TestModelChangeReembedsEverything(t *testing.T) {
	r := newRig(t, func(c *Config) { c.ModelID = "model-a" })
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	r.settle()
	r.stop()
	want := allIDs(r.store)
	docs1, imgs1, auds1 := r.emb.calls()

	r.cfg.ModelID = "model-b"
	r.start()
	r.settle()
	docs2, imgs2, auds2 := r.emb.calls()
	if docs2 != 2*docs1 || imgs2 != 2*imgs1 || auds2 != 2*auds1 {
		t.Errorf("embedded %d/%d/%d inputs after the model change, expected the same amount again (%d/%d/%d)", docs2-docs1, imgs2-imgs1, auds2-auds1, docs1, imgs1, auds1)
	}
	if got := allIDs(r.store); !slices.Equal(got, want) {
		t.Errorf("vectors after the model change: %d, want %d", len(got), len(want))
	}
}

func TestDisablingAKindRemovesItsVectors(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	r.settle()
	r.stop()

	r.cfg.IndexImages = false
	r.start()
	r.settle()
	if n := len(r.store.ids(CollectionImage)); n != 0 {
		t.Errorf("%d image vectors left after turning image indexing off", n)
	}
	if len(r.store.ids(CollectionText)) == 0 || len(r.store.ids(CollectionAudio)) != 2 {
		t.Error("other kinds were touched")
	}
	r.stop()

	// Turning it back on needs a Reconcile: dsearch sends no events for unchanged files.
	r.cfg.IndexImages = true
	r.start()
	if err := r.x.Reconcile(context.Background(), listFiles(files)); err != nil {
		t.Fatal(err)
	}
	r.settle()
	if n := len(r.store.ids(CollectionImage)); n != 4 {
		t.Errorf("%d image vectors after turning it on again", n)
	}
}

func TestNoPreprocessorMeansNoMedia(t *testing.T) {
	r := newRig(t, func(c *Config) { c.Preprocessor = nil })
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	r.settle()
	if len(r.store.ids(CollectionImage))+len(r.store.ids(CollectionAudio)) != 0 {
		t.Error("media indexed without a preprocessor")
	}
	if len(r.store.ids(CollectionText)) == 0 {
		t.Error("text not indexed")
	}
}

func TestLimitsAndExclusions(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.MaxChunksPerFile = 2
		c.MaxImageBytes = 70_000 // cat.jpg fits, the others do not
		c.Allow = func(p string) bool { return !strings.Contains(p, "/private/") }
	})
	files := r.corpus()
	secret := r.copyFixture("files/builder.go", "private/builder.go")
	files["secret"] = secret
	r.start()
	r.enqueueAll(files)
	r.settle()

	perFile := map[string]int{}
	for _, id := range r.store.ids(CollectionText) {
		p, _, _ := strings.Cut(id, "#")
		perFile[filepath.Base(p)]++
	}
	if perFile["sort.go"] != 2 {
		t.Errorf("sort.go has %d chunks with max_chunks_per_file=2", perFile["sort.go"])
	}
	if perFile["pride_and_prejudice.txt"] != 2 {
		t.Errorf("novel has %d chunks", perFile["pride_and_prejudice.txt"])
	}
	if perFile["builder.go"] != 0 && slices.ContainsFunc(r.store.ids(CollectionText), func(s string) bool { return strings.HasPrefix(s, secret) }) {
		t.Error("excluded file was indexed")
	}
	rec, _ := mustRecord(t, r.x.st, files["sort.go"])
	if !rec.Trunc {
		t.Error("record does not say the file was cut")
	}
	if ids := r.store.ids(CollectionImage); len(ids) != 2 || !strings.HasSuffix(ids[0], "cat.jpg") || !strings.HasSuffix(ids[1], "error_screenshot.png") {
		t.Errorf("images within the size limit: %v", shortList(ids, r.root))
	}
	if rec, _ := mustRecord(t, r.x.st, files["earth.jpg"]); rec.Err == "" || !strings.Contains(rec.Err, "larger") {
		t.Errorf("oversized image record: %+v", rec)
	}
}

func TestExclusionAddedLaterRemovesVectors(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	r.settle()
	r.stop()

	r.cfg.Allow = func(p string) bool { return !strings.HasSuffix(p, ".jpg") && !strings.Contains(p, "pride") }
	r.start()
	if err := r.x.Reconcile(context.Background(), listFiles(files)); err != nil {
		t.Fatal(err)
	}
	r.settle()
	if ids := r.store.ids(CollectionImage); len(ids) != 1 { // only the png remains
		t.Errorf("images after excluding jpg: %v", shortList(ids, r.root))
	}
	for _, id := range r.store.ids(CollectionText) {
		if strings.Contains(id, "pride") {
			t.Fatalf("excluded document still has vector %s", id)
		}
	}
}

func TestTransientEmbedderErrorsAreRetried(t *testing.T) {
	old := retryBackoff
	retryBackoff = []time.Duration{10 * time.Millisecond}
	defer func() { retryBackoff = old }()

	r := newRig(t, nil)
	r.emb.failNext, r.emb.failErr = 2, errors.New("helper crashed")
	r.start()
	r.x.Enqueue(r.copyFixture("files/builder.go", "builder.go"), OpIndex)
	p := r.settle()
	if p.Failed != 0 || len(r.store.ids(CollectionText)) == 0 {
		t.Fatalf("not recovered from transient errors: %+v", p)
	}
}

func TestPersistentEmbedderErrorGivesUpOnce(t *testing.T) {
	old := retryBackoff
	retryBackoff = []time.Duration{5 * time.Millisecond}
	defer func() { retryBackoff = old }()

	r := newRig(t, nil)
	r.emb.failNext, r.emb.failErr = 1000, errors.New("out of memory")
	r.start()
	p := r.copyFixture("files/builder.go", "builder.go")
	r.x.Enqueue(p, OpIndex)
	pr := r.settle()
	if pr.Failed != 1 || pr.Pending != 0 {
		t.Fatalf("progress %+v", pr)
	}
	rec, _ := mustRecord(t, r.x.st, p)
	if !strings.Contains(rec.Err, "out of memory") {
		t.Fatalf("record %+v", rec)
	}
	// The failure is remembered: a restart plus Reconcile does not try again...
	r.stop()
	r.emb.mu.Lock()
	r.emb.failNext = 0
	r.emb.mu.Unlock()
	r.start()
	r.x.Reconcile(context.Background(), listFiles(map[string]string{"x": p}))
	r.settle()
	if len(r.store.ids(CollectionText)) != 0 {
		t.Fatal("a failed file was retried without a change")
	}
	// ...until the file changes.
	later := mtimeBase.Add(5 * hour)
	os.Chtimes(p, later, later)
	r.x.Enqueue(p, OpIndex)
	r.settle()
	if len(r.store.ids(CollectionText)) == 0 {
		t.Fatal("a changed file was not retried")
	}
}

func TestEnqueueBurstIsFastAndDeduplicated(t *testing.T) {
	r := newRig(t, nil)
	r.start()
	start := time.Now()
	const n = 30_000
	for i := 0; i < n; i++ {
		r.x.Enqueue(filepath.Join(r.root, "burst", strconv.Itoa(i)+".txt"), OpIndex)
	}
	for i := 0; i < n; i++ { // the same paths again: still one entry each
		r.x.Enqueue(filepath.Join(r.root, "burst", strconv.Itoa(i)+".txt"), OpDelete)
	}
	r.x.Enqueue("relative/path.txt", OpIndex) // ignored
	r.x.Enqueue(filepath.Join(r.root, "x.txt"), Op(99))
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("%d Enqueue calls took %s: the indexer thread would stall", 2*n, d)
	}
	t.Logf("%d Enqueue calls in %s", 2*n, time.Since(start))
	if p := r.x.Progress(); p.Pending < 1 || p.Pending > 2*n {
		t.Errorf("pending %d for %d distinct paths", p.Pending, n)
	}
	r.settle()
	// A path whose first event was already processed when the second arrived
	// is processed twice; the rest collapse into one entry.
	if p := r.x.Progress(); p.Pending != 0 || p.Done < n || p.Done > 2*n {
		t.Errorf("after the burst: %+v, want between %d and %d paths processed", p, n, 2*n)
	}
}

func TestStartTwiceAndEnqueueAfterClose(t *testing.T) {
	r := newRig(t, nil)
	r.start()
	if err := r.x.Start(context.Background()); err == nil {
		t.Error("second Start must fail")
	}
	x := r.x
	r.stop()
	x.Enqueue(filepath.Join(r.root, "late.txt"), OpIndex) // must not panic or block
	if err := x.Close(); err != nil {
		t.Error(err)
	}
}

func TestEventsBeforeStartAreKept(t *testing.T) {
	r := newRig(t, nil)
	r.x = New(r.cfg, r.emb, r.store)
	p := r.copyFixture("files/builder.go", "builder.go")
	r.x.Enqueue(p, OpIndex)
	ctx, cancel := context.WithCancel(context.Background())
	r.cancel = cancel
	if err := r.x.Start(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(r.stop)
	r.settle()
	if len(r.store.ids(CollectionText)) == 0 {
		t.Fatal("event queued before Start was lost")
	}
}

func TestStartFailsOnUnusableState(t *testing.T) {
	r := newRig(t, nil)
	dir := t.TempDir()
	blocker := filepath.Join(dir, "file")
	os.WriteFile(blocker, nil, 0o644)
	r.cfg.StatePath = filepath.Join(blocker, "state.db") // a directory cannot be created below a file
	x := New(r.cfg, r.emb, r.store)
	if err := x.Start(context.Background()); err == nil {
		t.Fatal("Start succeeded without a usable state file")
	}
}

func TestDirectoriesAndOddEventsAreIgnored(t *testing.T) {
	r := newRig(t, nil)
	r.start()
	dir := filepath.Join(r.root, "somedir.txt") // a directory with a text extension
	os.MkdirAll(dir, 0o755)
	r.x.Enqueue(dir, OpIndex)
	r.x.Enqueue(filepath.Join(r.root, "never-existed.png"), OpDelete)
	p := r.settle()
	if p.Failed != 0 || len(allIDs(r.store)) != 0 {
		t.Fatalf("%+v %v", p, allIDs(r.store))
	}
}

func TestProgressNumbers(t *testing.T) {
	r := newRig(t, nil)
	files := r.corpus()
	r.start()
	r.enqueueAll(files)
	p := r.settle()
	if p.Done != int64(len(files)) || p.Indexed != len(files) || p.Pending != 0 {
		t.Fatalf("%+v for %d files", p, len(files))
	}
}

func TestEnqueueDoesNotBlockForeverWhenStateIsBroken(t *testing.T) {
	oldMax, oldWait := maxBuffered, enqueueWait
	maxBuffered, enqueueWait = 10, 300*time.Millisecond
	defer func() { maxBuffered, enqueueWait = oldMax, oldWait }()

	r := newRig(t, nil)
	r.start()
	r.x.st.db.Close() // the state file becomes unusable under the running indexer
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 50; i++ {
			r.x.Enqueue(filepath.Join(r.root, strconv.Itoa(i)+".txt"), OpIndex)
		}
	}()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("Enqueue blocked the caller although the state cannot be written")
	}
}

func mustRecord(t *testing.T, st *stateDB, path string) (record, bool) {
	t.Helper()
	rec, ok, err := st.record(path)
	if err != nil {
		t.Fatal(err)
	}
	return rec, ok
}
