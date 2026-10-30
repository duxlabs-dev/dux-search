package service

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service/fixture"
	"github.com/duxlabs-dev/dux-search/internal/semantic/vecstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testModelID = "embeddinggemma-2-test"

type env struct {
	t    *testing.T
	dbp  string
	emb  *fixture.Embedder
	st   *vecstore.Store
	svc  *Service
	data *fixture.Data
}

func newEnv(t *testing.T, opts ...func(*Options)) *env {
	t.Helper()
	e := &env{t: t, dbp: filepath.Join(t.TempDir(), "semantic.db"), emb: fixture.NewEmbedder(), data: fixture.MustLoad()}
	e.open(opts...)
	t.Cleanup(e.close)
	return e
}

func (e *env) options(opts ...func(*Options)) Options {
	o := Options{Enabled: true, ModelID: testModelID, Embedder: e.emb, Store: e.st, RetryInterval: 20 * time.Millisecond}
	for _, f := range opts {
		f(&o)
	}
	return o
}

// open (re)opens store and service over the same database file.
func (e *env) open(opts ...func(*Options)) {
	var err error
	e.st, err = vecstore.Open(e.dbp, vecstore.Options{})
	require.NoError(e.t, err)
	e.svc, err = New(e.options(opts...))
	require.NoError(e.t, err)
	require.NoError(e.t, e.svc.Start(context.Background()))
}

func (e *env) close() {
	if e.svc != nil {
		e.svc.Close()
		e.svc = nil
	}
	if e.st != nil {
		e.st.Close()
		e.st = nil
	}
}

func (e *env) actionItems() []semantic.Item {
	var items []semantic.Item
	for _, d := range e.data.ActionDocs() {
		items = append(items, semantic.Item{ID: d.ID, Title: d.Title, Text: d.Text, Payload: map[string]any{"category": "system"}})
	}
	return items
}

func (e *env) eventually(cond func() bool, msg string) {
	e.t.Helper()
	require.Eventually(e.t, cond, 5*time.Second, 10*time.Millisecond, msg)
}

func TestSetCollectionAndSearchActions(t *testing.T) {
	e := newEnv(t)
	res, err := e.svc.SetCollection(context.Background(), "shell.actions", e.actionItems())
	require.NoError(t, err)
	assert.Equal(t, SetResult{Collection: "shell.actions", Total: 20, Embedded: 20}, res)

	ok := 0
	qs := e.data.QueriesOf("action")
	for _, q := range qs {
		resp, err := e.svc.Search(context.Background(), SearchRequest{Query: q.Text, Collections: []string{"shell.actions"}, CollectionLimit: 3})
		require.NoError(t, err)
		require.True(t, resp.Available)
		require.Len(t, resp.Collections, 1)
		hits := resp.Collections[0].Hits
		require.Len(t, hits, 3)
		if hits[0].ID == q.Expect {
			ok++
		}
	}
	t.Logf("action queries: expected action first in %d/%d", ok, len(qs))
	assert.GreaterOrEqual(t, ok, 14, "the real-model eylem_eslesme baseline is 15/15 at 768-d")
}

func TestSetCollectionIsIncremental(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	items := e.actionItems()
	_, err := e.svc.SetCollection(ctx, "shell.actions", items)
	require.NoError(t, err)
	before := e.emb.DocsEmbedded()

	res, err := e.svc.SetCollection(ctx, "shell.actions", items)
	require.NoError(t, err)
	assert.Equal(t, SetResult{Collection: "shell.actions", Total: 20}, res, "same content, nothing to embed")
	assert.Equal(t, before, e.emb.DocsEmbedded())

	// payload is not embedded, so changing it keeps the vector
	items[0].Payload = map[string]any{"category": "other"}
	res, _ = e.svc.SetCollection(ctx, "shell.actions", items)
	assert.Zero(t, res.Embedded)

	// changing the text of one item and dropping another
	items[1].Text += " Also known as the second one."
	items = items[:len(items)-1]
	removedID := e.actionItems()[19].ID
	res, err = e.svc.SetCollection(ctx, "shell.actions", items)
	require.NoError(t, err)
	assert.Equal(t, SetResult{Collection: "shell.actions", Total: 19, Embedded: 0, Removed: 1, Pending: 1}, res,
		"the edited item has no model vector in the fixture, so it stays pending instead of failing the call")
	h, _ := e.st.Hashes("shell.actions")
	assert.NotContains(t, h, removedID)
	assert.Len(t, h, 19, "the edited item keeps its previous vector until the new one is embedded")
}

func TestKeywordsAreEmbeddedAndHashed(t *testing.T) {
	it := semantic.Item{ID: "a", Title: "T", Text: "body", Keywords: []string{"x", "y"}}
	assert.Equal(t, semantic.TextDoc{Title: "T", Text: "body Keywords: x, y"}, docFor(it))
	assert.Equal(t, semantic.TextDoc{Title: "T", Text: "body"}, docFor(semantic.Item{Title: "T", Text: "body"}))
	assert.Equal(t, semantic.TextDoc{Title: "", Text: "Keywords: x"}, docFor(semantic.Item{Keywords: []string{"x"}}))

	base := itemHash(it, "m", 768)
	assert.Equal(t, base, itemHash(it, "m", 768))
	for name, changed := range map[string]string{
		"title":    itemHash(semantic.Item{ID: "a", Title: "T2", Text: "body", Keywords: it.Keywords}, "m", 768),
		"text":     itemHash(semantic.Item{ID: "a", Title: "T", Text: "body2", Keywords: it.Keywords}, "m", 768),
		"keywords": itemHash(semantic.Item{ID: "a", Title: "T", Text: "body", Keywords: []string{"x"}}, "m", 768),
		"model":    itemHash(it, "m2", 768),
		"dim":      itemHash(it, "m", 256),
	} {
		assert.NotEqual(t, base, changed, name)
	}
	// boundaries cannot be shifted between fields
	assert.NotEqual(t, itemHash(semantic.Item{Title: "ab", Text: "c"}, "m", 1), itemHash(semantic.Item{Title: "a", Text: "bc"}, "m", 1))
	assert.Equal(t, itemHash(semantic.Item{Title: "x"}, "m", 1), itemHash(semantic.Item{ID: "other-id", Title: "x"}, "m", 1), "the id is not part of the content hash")
}

// With no model the items are kept; when the model appears the worker embeds
// them without the client sending them again.
func TestItemsWaitForTheModel(t *testing.T) {
	e := newEnv(t)
	e.emb.SetAvailable(false, "model not installed")
	ctx := context.Background()

	res, err := e.svc.SetCollection(ctx, "shell.actions", e.actionItems())
	require.NoError(t, err)
	assert.Equal(t, SetResult{Collection: "shell.actions", Total: 20, Pending: 20}, res)
	items, _ := e.st.Items("shell.actions")
	assert.Len(t, items, 20, "stored although nothing could be embedded")
	assert.Equal(t, 20, e.svc.Status().Pending["shell.actions"])

	resp, err := e.svc.Search(ctx, SearchRequest{Query: "karanlık", Collections: []string{"shell.actions"}})
	require.NoError(t, err)
	assert.False(t, resp.Available)
	assert.Contains(t, resp.Reason, "model not installed")

	e.emb.SetAvailable(true, "")
	e.eventually(func() bool { n, _ := e.st.Count("shell.actions"); return n == 20 }, "worker embeds the stored items")
	e.eventually(func() bool { return len(e.svc.Status().Pending) == 0 }, "pending clears")

	resp, err = e.svc.Search(ctx, SearchRequest{Query: "karanlık", Collections: []string{"shell.actions"}, CollectionLimit: 1})
	require.NoError(t, err)
	require.True(t, resp.Available)
	assert.Equal(t, "dark", resp.Collections[0].Hits[0].ID)
}

func TestRestartRepairsFromStoredItems(t *testing.T) {
	e := newEnv(t)
	e.emb.SetAvailable(false, "no model yet")
	_, err := e.svc.SetCollection(context.Background(), "shell.actions", e.actionItems())
	require.NoError(t, err)
	e.close()

	// the daemon restarts with the model in place
	e.emb.SetAvailable(true, "")
	e.open()
	assert.Equal(t, 20, e.svc.Status().Pending["shell.actions"], "pending is recomputed from the stored items")
	e.eventually(func() bool { n, _ := e.st.Count("shell.actions"); return n == 20 }, "embedded after restart")
	embedded := e.emb.DocsEmbedded()
	e.close()

	// and a restart with everything embedded does no work at all
	e.open()
	assert.Empty(t, e.svc.Status().Pending)
	time.Sleep(60 * time.Millisecond)
	assert.Equal(t, embedded, e.emb.DocsEmbedded())
	assert.Equal(t, 20, e.svc.Status().Collections["shell.actions"])
}

func TestModelChangeEmbedsEverythingAgain(t *testing.T) {
	e := newEnv(t)
	_, err := e.svc.SetCollection(context.Background(), "shell.actions", e.actionItems())
	require.NoError(t, err)
	before := e.emb.DocsEmbedded()
	e.close()

	e.open(func(o *Options) { o.ModelID = "embeddinggemma-3-test" })
	assert.Equal(t, 20, e.svc.Status().Pending["shell.actions"], "the new model has no vectors yet")
	e.eventually(func() bool { n, _ := e.st.Count("shell.actions"); return n == 20 }, "re-embedded")
	assert.Equal(t, before+20, e.emb.DocsEmbedded())
	info, _ := e.st.Info("shell.actions")
	assert.Equal(t, "embeddinggemma-3-test", info.Model)
}

func TestSetCollectionBudgetMovesTheRestToTheBackground(t *testing.T) {
	e := newEnv(t, func(o *Options) { o.SetBudget = time.Millisecond })
	e.emb.SetDelay(func(ctx context.Context) error {
		select {
		case <-time.After(15 * time.Millisecond):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	// 4 batches of 16 would be needed; the budget allows one
	var items []semantic.Item
	for _, d := range e.data.FileDocs()[:64] {
		items = append(items, semantic.Item{ID: fmt.Sprintf("%s#%d", d.Pkg, d.Chunk), Title: d.Title, Text: d.Text})
	}
	res, err := e.svc.SetCollection(context.Background(), "shell.docs", items)
	require.NoError(t, err)
	assert.Equal(t, 64, res.Total)
	assert.Equal(t, embedBatch, res.Embedded)
	assert.Equal(t, 64-embedBatch, res.Pending)
	e.eventually(func() bool { n, _ := e.st.Count("shell.docs"); return n == 64 }, "worker finishes the rest")
}

func TestSetCollectionValidation(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	ok := []semantic.Item{{ID: "a", Title: e.data.ActionDocs()[0].Title, Text: e.data.ActionDocs()[0].Text}}
	for name, c := range map[string]struct {
		coll  string
		items []semantic.Item
	}{
		"empty name":       {"", ok},
		"bad characters":   {"shell actions", ok},
		"slash":            {"a/b", ok},
		"reserved":         {"files.text", ok},
		"reserved any":     {"files.other", ok},
		"missing id":       {"c", []semantic.Item{{Title: "x"}}},
		"duplicate id":     {"c", []semantic.Item{{ID: "a", Title: "x"}, {ID: "a", Title: "y"}}},
		"nothing to embed": {"c", []semantic.Item{{ID: "a"}}},
	} {
		_, err := e.svc.SetCollection(ctx, c.coll, c.items)
		assert.Error(t, err, name)
	}
	n, _ := e.st.ItemCollections()
	assert.Empty(t, n, "a rejected set stores nothing")

	// an empty set is valid and clears the collection
	_, err := e.svc.SetCollection(ctx, "c", ok)
	require.NoError(t, err)
	res, err := e.svc.SetCollection(ctx, "c", nil)
	require.NoError(t, err)
	assert.Equal(t, SetResult{Collection: "c", Removed: 1}, res)
}

func TestDeleteCollection(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	_, err := e.svc.SetCollection(ctx, "shell.actions", e.actionItems())
	require.NoError(t, err)
	res, err := e.svc.DeleteCollection("shell.actions")
	require.NoError(t, err)
	assert.Equal(t, DeleteResult{Collection: "shell.actions", Removed: 20}, res)
	_, ok := e.st.Info("shell.actions")
	assert.False(t, ok)
	items, _ := e.st.Items("shell.actions")
	assert.Empty(t, items)
	res, err = e.svc.DeleteCollection("shell.actions")
	require.NoError(t, err)
	assert.Zero(t, res.Removed, "deleting twice is fine")
	_, err = e.svc.DeleteCollection("files.text")
	assert.Error(t, err)

	resp, err := e.svc.Search(ctx, SearchRequest{Query: "karanlık", Collections: []string{"shell.actions"}})
	require.NoError(t, err)
	assert.Equal(t, []CollectionHits{{Collection: "shell.actions", Hits: []semantic.Hit{}}}, resp.Collections, "unknown collection is an empty result, not an error")
}

func TestDisabledService(t *testing.T) {
	svc, err := New(Options{Enabled: false})
	require.NoError(t, err)
	require.NoError(t, svc.Start(context.Background()))
	defer svc.Close()
	ctx := context.Background()

	_, err = svc.SetCollection(ctx, "c", []semantic.Item{{ID: "a", Title: "x"}})
	assert.ErrorIs(t, err, ErrDisabled)
	_, err = svc.DeleteCollection("c")
	assert.ErrorIs(t, err, ErrDisabled)
	assert.ErrorIs(t, svc.Warm(ctx), ErrDisabled)
	resp, err := svc.Search(ctx, SearchRequest{Query: "q"})
	require.NoError(t, err)
	assert.False(t, resp.Available)
	assert.Equal(t, ErrDisabled.Error(), resp.Reason)
	st := svc.Status()
	assert.False(t, st.Enabled)
	assert.False(t, st.Embedder.Available)
	assert.Equal(t, ErrDisabled.Error(), st.Embedder.Reason)
	assert.Empty(t, st.Collections)
	svc.FileChanged("/x", false) // no-op, must not panic
	_, err = New(Options{Enabled: true})
	assert.Error(t, err, "enabling without embedder and store is a wiring bug")
}

func TestWarm(t *testing.T) {
	e := newEnv(t)
	require.NoError(t, e.svc.Warm(context.Background()))
	require.NoError(t, e.svc.Warm(context.Background(), semantic.ModalityImage, semantic.ModalityText))
	assert.Equal(t, []semantic.Modality{"text", "image", "text"}, e.emb.Warmed())
	assert.Error(t, e.svc.Warm(context.Background(), "video"))
	e.emb.SetAvailable(false, "no runtime")
	assert.ErrorIs(t, e.svc.Warm(context.Background()), semantic.ErrModelUnavailable)
}

func TestParseModalities(t *testing.T) {
	got, err := ParseModalities([]string{"image", "text", "image"})
	require.NoError(t, err)
	assert.Equal(t, []semantic.Modality{"image", "text"}, got)
	_, err = ParseModalities([]string{"text", "pdf"})
	assert.Error(t, err)
}

func TestSearchRequestValidation(t *testing.T) {
	e := newEnv(t)
	_, err := e.svc.Search(context.Background(), SearchRequest{Query: "  "})
	assert.Error(t, err)
	_, err = e.svc.Search(context.Background(), SearchRequest{Query: "karanlık", Collections: []string{"bad name"}})
	assert.Error(t, err)
	// the fixture knows no vector for this text: the failure is reported, not hidden
	resp, err := e.svc.Search(context.Background(), SearchRequest{Query: "text outside the fixture"})
	require.NoError(t, err)
	assert.False(t, resp.Available)
	assert.Contains(t, resp.Reason, "embedding failed")
}

func TestSearchHonoursCancellation(t *testing.T) {
	e := newEnv(t)
	e.emb.SetDelay(func(ctx context.Context) error { <-ctx.Done(); return ctx.Err() })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := e.svc.Search(ctx, SearchRequest{Query: "karanlık"})
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestConcurrentSetAndSearch(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	items := e.actionItems()
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				set := items[:10+(w+i)%10]
				if _, err := e.svc.SetCollection(ctx, "shell.actions", set); err != nil {
					t.Error(err)
				}
			}
		}()
		go func() {
			defer wg.Done()
			qs := e.data.QueriesOf("action")
			for i := 0; i < 20; i++ {
				if _, err := e.svc.Search(ctx, SearchRequest{Query: qs[(w+i)%len(qs)].Text, Collections: []string{"shell.actions"}}); err != nil {
					t.Error(err)
				}
				_ = e.svc.Status()
			}
		}()
	}
	wg.Wait()
}

type fakeFiles struct {
	mu     sync.Mutex
	events []string
	prog   IndexProgress
}

func (f *fakeFiles) Enqueue(path string, deleted bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.events = append(f.events, fmt.Sprintf("%s:%v", path, deleted))
}
func (f *fakeFiles) Progress() IndexProgress { return f.prog }

type fakeModels struct {
	status   any
	release  chan struct{}
	failWith error
	removed  []semantic.Modality
	got      []semantic.Modality
}

func (m *fakeModels) Status() any { return m.status }
func (m *fakeModels) Download(ctx context.Context, towers []semantic.Modality, progress func(ModelProgress)) error {
	m.got = towers
	for _, tw := range towers {
		progress(ModelProgress{Tower: tw, Bytes: 5, Total: 10, State: "downloading"})
	}
	select {
	case <-m.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	if m.failWith != nil {
		return m.failWith
	}
	for _, tw := range towers {
		progress(ModelProgress{Tower: tw, Bytes: 10, Total: 10, State: "done"})
	}
	return nil
}
func (m *fakeModels) Remove(towers []semantic.Modality) error { m.removed = towers; return nil }

func TestStatusFileEventsAndModelLifecycle(t *testing.T) {
	files := &fakeFiles{prog: IndexProgress{Pending: 7, Done: 3, PausedReason: "on battery"}}
	models := &fakeModels{status: map[string]any{"installed": []string{"text"}}, release: make(chan struct{})}
	e := newEnv(t, func(o *Options) { o.Files, o.Models = files, models })

	e.svc.FileChanged("/a/b.txt", false)
	e.svc.FileChanged("/a/c.txt", true)
	assert.Equal(t, []string{"/a/b.txt:false", "/a/c.txt:true"}, files.events)

	st := e.svc.Status()
	assert.True(t, st.Enabled)
	assert.True(t, st.Embedder.Available)
	assert.Equal(t, files.prog, st.Indexing)
	assert.Equal(t, models.status, st.Model)

	var evMu sync.Mutex
	var events []ModelProgress
	e.svc.OnModelProgress(func(p ModelProgress) { evMu.Lock(); events = append(events, p); evMu.Unlock() })

	require.NoError(t, e.svc.ModelDownload([]semantic.Modality{semantic.ModalityText, semantic.ModalityImage}))
	assert.ErrorIs(t, e.svc.ModelDownload([]semantic.Modality{semantic.ModalityText}), ErrDownloadRunning)
	assert.ErrorIs(t, e.svc.ModelRemove([]semantic.Modality{semantic.ModalityText}), ErrDownloadRunning, "no removal under a running download")
	e.eventually(func() bool { return e.svc.Status().Download.Towers[semantic.ModalityImage].State == "downloading" }, "progress visible in status")
	assert.True(t, e.svc.Status().Download.Active)

	close(models.release)
	e.eventually(func() bool { return !e.svc.Status().Download.Active }, "download ends")
	evMu.Lock()
	last := events[len(events)-1]
	evMu.Unlock()
	assert.Equal(t, ModelProgress{State: StateFinished}, last)
	assert.Equal(t, "done", e.svc.Status().Download.Towers[semantic.ModalityText].State)
	assert.Empty(t, e.svc.Status().Download.Error)

	require.NoError(t, e.svc.ModelRemove([]semantic.Modality{semantic.ModalityImage}))
	assert.Equal(t, []semantic.Modality{semantic.ModalityImage}, models.removed)
	assert.Error(t, e.svc.ModelDownload(nil))
	assert.Error(t, e.svc.ModelRemove(nil))
}

func TestModelDownloadFailureIsReported(t *testing.T) {
	models := &fakeModels{release: make(chan struct{}), failWith: errors.New("disk full")}
	e := newEnv(t, func(o *Options) { o.Models = models })
	var mu sync.Mutex
	var events []ModelProgress
	e.svc.OnModelProgress(func(p ModelProgress) { mu.Lock(); events = append(events, p); mu.Unlock() })
	close(models.release)
	require.NoError(t, e.svc.ModelDownload([]semantic.Modality{semantic.ModalityText}))
	e.eventually(func() bool { return e.svc.Status().Download.Error != "" }, "error recorded")
	assert.Equal(t, "disk full", e.svc.Status().Download.Error)
	mu.Lock()
	assert.Equal(t, ModelProgress{State: StateFailed, Error: "disk full"}, events[len(events)-1])
	mu.Unlock()
	require.NoError(t, e.svc.ModelDownload([]semantic.Modality{semantic.ModalityText}), "a new attempt is allowed after a failure")
}

func TestModelManagementWithoutManager(t *testing.T) {
	e := newEnv(t)
	_, err := e.svc.ModelStatus()
	assert.ErrorIs(t, err, ErrNoModelManager)
	assert.ErrorIs(t, e.svc.ModelDownload([]semantic.Modality{"text"}), ErrNoModelManager)
	assert.ErrorIs(t, e.svc.ModelRemove([]semantic.Modality{"text"}), ErrNoModelManager)
	assert.Nil(t, e.svc.Status().Model)
}

func TestPathOf(t *testing.T) {
	for id, want := range map[string]string{
		"/a/b.txt#0":   "/a/b.txt",
		"/a/b.txt#12":  "/a/b.txt",
		"/a/b.txt":     "/a/b.txt",
		"/a/b#c.txt":   "/a/b#c.txt",
		"/a/b#c.txt#3": "/a/b#c.txt",
		"/a/b#":        "/a/b#",
		"#1":           "#1",
	} {
		assert.Equal(t, want, PathOf(id), id)
	}
}

func TestDisabledReasonIsShown(t *testing.T) {
	svc, err := New(Options{DisabledReason: "vector database: locked by another process"})
	require.NoError(t, err)
	_, err = svc.SetCollection(context.Background(), "c", []semantic.Item{{ID: "a", Title: "x"}})
	assert.ErrorIs(t, err, ErrDisabled)
	assert.Contains(t, err.Error(), "locked by another process")
	assert.Contains(t, svc.Status().Embedder.Reason, "locked by another process")
}

// A daemon whose model runtime cannot start keeps collections and falls back to
// lexical search; the reason reaches the client.
func TestUnavailableEmbedderKeepsTheServiceUsable(t *testing.T) {
	st, err := vecstore.Open(filepath.Join(t.TempDir(), "semantic.db"), vecstore.Options{})
	require.NoError(t, err)
	defer st.Close()
	svc, err := New(Options{Enabled: true, ModelID: testModelID, Embedder: UnavailableEmbedder("libonnxruntime.so not found"), Store: st, RetryInterval: 10 * time.Millisecond})
	require.NoError(t, err)
	require.NoError(t, svc.Start(context.Background()))
	defer svc.Close()

	data := fixture.MustLoad()
	var items []semantic.Item
	for _, d := range data.ActionDocs() {
		items = append(items, semantic.Item{ID: d.ID, Title: d.Title, Text: d.Text})
	}
	res, err := svc.SetCollection(context.Background(), "shell.actions", items)
	require.NoError(t, err)
	assert.Equal(t, 20, res.Pending)
	resp, err := svc.Search(context.Background(), SearchRequest{Query: "karanlık", Collections: []string{"shell.actions"}})
	require.NoError(t, err)
	assert.False(t, resp.Available)
	assert.Equal(t, "libonnxruntime.so not found", resp.Reason)
	stt := svc.Status()
	assert.True(t, stt.Enabled)
	assert.False(t, stt.Embedder.Available)
	assert.Equal(t, "libonnxruntime.so not found", stt.Embedder.Reason)
	assert.Error(t, svc.Warm(context.Background()))
	time.Sleep(50 * time.Millisecond) // the worker retries without spinning or failing
	assert.Equal(t, 20, svc.Status().Pending["shell.actions"])
}

func TestProgressEventsAreThrottled(t *testing.T) {
	svc, err := New(Options{})
	require.NoError(t, err)
	var mu sync.Mutex
	var got []ModelProgress
	svc.OnModelProgress(func(p ModelProgress) { mu.Lock(); got = append(got, p); mu.Unlock() })
	svc.dl = downloadState{towers: map[semantic.Modality]ModelProgress{}}

	for i := 1; i <= 1000; i++ { // a downloader reports every chunk
		svc.reportProgress(ModelProgress{Tower: "text", Bytes: int64(i), Total: 1000, State: "downloading"})
	}
	svc.reportProgress(ModelProgress{Tower: "text", Bytes: 1000, Total: 1000, State: "verified"})
	mu.Lock()
	defer mu.Unlock()
	assert.Less(t, len(got), 10, "chunk-level reports are collapsed")
	assert.Equal(t, "downloading", got[0].State, "the first report goes out at once")
	assert.Equal(t, "verified", got[len(got)-1].State, "a state change is never dropped")
	assert.EqualValues(t, 1000, svc.Status().Download.Towers["text"].Bytes, "status keeps the latest numbers")
}
