package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/AvengeMedia/dankgo/ipc"
	"github.com/duxlabs-dev/dux-search/internal/client"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service/servicetest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests drive the real thing end to end: the unix socket server, the
// client package the CLI uses, the router, the semantic service with a real
// vector store, and a real lexical index over real files. Only the embedder is
// the fixture (true model vectors for known texts), because the model runtime
// belongs to another package.

type stubWatcher struct{}

func (stubWatcher) Start() error    { return nil }
func (stubWatcher) Stop() error     { return nil }
func (stubWatcher) IsRunning() bool { return false }

func serve(t *testing.T, st *servicetest.Stack) *ipc.Server {
	t.Helper()
	runtime, err := os.MkdirTemp("", "dsrt")
	require.NoError(t, err)
	t.Cleanup(func() { os.RemoveAll(runtime) })
	t.Setenv("XDG_RUNTIME_DIR", runtime)

	router := NewRouter(st.Indexer, stubWatcher{}, st.Config, WithSemantic(st.Service))
	srv := NewUnix(router)
	require.NoError(t, srv.Listen())
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); _ = srv.Serve(ctx) }()
	t.Cleanup(func() { cancel(); <-done })
	return srv
}

type fakeModels struct {
	mu      sync.Mutex
	release chan struct{}
	fail    error
	removed []semantic.Modality
}

func (m *fakeModels) Status() any {
	return map[string]any{"towers": map[string]any{"text": map[string]any{"installed": true}}}
}

func (m *fakeModels) Download(ctx context.Context, towers []semantic.Modality, progress func(service.ModelProgress)) error {
	for _, tw := range towers {
		progress(service.ModelProgress{Tower: tw, Bytes: 10, Total: 100, State: "downloading"})
	}
	select {
	case <-m.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	if m.fail != nil {
		return m.fail
	}
	for _, tw := range towers {
		progress(service.ModelProgress{Tower: tw, Bytes: 100, Total: 100, State: "verified"})
	}
	return nil
}

func (m *fakeModels) Remove(towers []semantic.Modality) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.removed = towers
	return nil
}

func TestHandshakeAnnouncesCapabilities(t *testing.T) {
	st := servicetest.NewStack(t, servicetest.Options{})
	srv := serve(t, st)
	conn, err := (&net_{}).dial(srv.SocketPath())
	require.NoError(t, err)
	defer conn.Close()
	line, err := bufio.NewReader(conn).ReadBytes('\n')
	require.NoError(t, err)
	var caps ipc.Capabilities
	require.NoError(t, json.Unmarshal(line, &caps))
	assert.Equal(t, APIVersion, caps.APIVersion)
	assert.Equal(t, []string{"semantic", "collections"}, caps.Capabilities)
}

func TestSearchWithoutSemanticIsTheOldResponse(t *testing.T) {
	st := servicetest.NewStack(t, servicetest.Options{})
	serve(t, st)

	res, err := client.SearchWithOptions(&client.SearchOptions{Query: "regexp", Limit: 10, Type: "file"})
	require.NoError(t, err)
	want, err := st.Indexer.SearchWithOptions(&searchOpts{Query: "regexp", Limit: 10, Type: "file"})
	require.NoError(t, err)
	require.Len(t, res.Hits, len(want.Hits))
	for i, h := range res.Hits {
		assert.Equal(t, want.Hits[i].ID, h.ID)
		assert.Equal(t, want.Hits[i].Score, h.Score)
		assert.Nil(t, h.SemanticScore, "semantic_score only appears when semantic was asked for")
	}
	assert.Nil(t, res.Semantic)
	assert.Nil(t, res.Collections)
	assert.Equal(t, want.Total, res.Total)

	// and no embedding was done for it
	assert.Zero(t, st.Embedder.QueriesEmbedded())
}

func TestCollectionsAndSemanticSearchOverTheSocket(t *testing.T) {
	st := servicetest.NewStack(t, servicetest.Options{})
	serve(t, st)

	set, err := client.CollectionsSet("shell.actions", st.ActionItems())
	require.NoError(t, err)
	assert.Equal(t, service.SetResult{Collection: "shell.actions", Total: 20, Embedded: 20}, *set)
	again, err := client.CollectionsSet("shell.actions", st.ActionItems())
	require.NoError(t, err)
	assert.Equal(t, service.SetResult{Collection: "shell.actions", Total: 20}, *again, "an unchanged set is free")

	res, err := client.SearchWithOptions(&client.SearchOptions{
		Query: "gözlerim yoruldu", Limit: 5, Type: "file",
		Semantic: true, Collections: []string{"shell.actions"}, CollectionLimit: 3,
	})
	require.NoError(t, err)
	require.NotNil(t, res.Semantic)
	assert.True(t, res.Semantic.Available, res.Semantic.Reason)
	require.Len(t, res.Collections, 1)
	assert.Equal(t, "shell.actions", res.Collections[0].Collection)
	require.Len(t, res.Collections[0].Hits, 3)
	assert.Equal(t, "nightlight", res.Collections[0].Hits[0].ID)
	assert.Greater(t, res.Collections[0].Hits[0].Score, res.Collections[0].Hits[2].Score)

	// default collection_limit is 8
	res, err = client.SearchWithOptions(&client.SearchOptions{Query: "karanlık", Limit: 5, Semantic: true, Collections: []string{"shell.actions"}})
	require.NoError(t, err)
	assert.Len(t, res.Collections[0].Hits, 8)

	del, err := client.CollectionsDelete("shell.actions")
	require.NoError(t, err)
	assert.Equal(t, 20, del.Removed)
	res, err = client.SearchWithOptions(&client.SearchOptions{Query: "karanlık", Limit: 5, Semantic: true, Collections: []string{"shell.actions"}})
	require.NoError(t, err)
	assert.Empty(t, res.Collections[0].Hits)
}

func TestSemanticFileSearchOverTheSocket(t *testing.T) {
	st := servicetest.NewStack(t, servicetest.Options{})
	serve(t, st)

	res, err := client.SearchWithOptions(&client.SearchOptions{Query: "düzenli ifadelerle metin eşleştirme", Limit: 5, Type: "file", Semantic: true})
	require.NoError(t, err)
	require.NotEmpty(t, res.Hits)
	assert.Equal(t, st.Files["regexp"], res.Hits[0].ID, "found by meaning; there is no Turkish word in the file")
	require.NotNil(t, res.Hits[0].SemanticScore)
	assert.True(t, res.Semantic.Fused)
	assert.Equal(t, uint64(res.Semantic.Added), res.Total)

	// the raw response keeps the shape the launcher reads
	raw := rawSearch(t, map[string]any{"query": "düzenli ifadelerle metin eşleştirme", "limit": 3, "type": "file", "semantic": true})
	var m struct {
		Hits []struct {
			ID            string                    `json:"id"`
			Score         float64                   `json:"score"`
			SemanticScore float64                   `json:"semantic_score"`
			Locations     map[string]map[string]any `json:"locations"`
		} `json:"hits"`
	}
	require.NoError(t, json.Unmarshal(raw, &m))
	require.NotEmpty(t, m.Hits)
	assert.Equal(t, st.Files["regexp"], m.Hits[0].ID)
	assert.Contains(t, m.Hits[0].Locations["doc_type"], "file")
	assert.Positive(t, m.Hits[0].SemanticScore)
}

func TestSemanticParametersAreValidated(t *testing.T) {
	st := servicetest.NewStack(t, servicetest.Options{})
	serve(t, st)

	_, err := client.SearchWithOptions(&client.SearchOptions{Query: "x", Limit: 3, Collections: []string{"shell.actions"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "collections requires semantic=true")

	_, err = client.SearchWithOptions(&client.SearchOptions{Query: "x", Limit: 3, Semantic: true, Collections: []string{"bad name"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid collection name")

	for name, params := range map[string]map[string]any{
		"no collection":   {"items": []any{}},
		"no items":        {"collection": "c"},
		"items not array": {"collection": "c", "items": "x"},
		"item not object": {"collection": "c", "items": []any{"x"}},
		"id wrong type":   {"collection": "c", "items": []any{map[string]any{"id": 5}}},
		"files reserved":  {"collection": "files.text", "items": []any{}},
		"empty id":        {"collection": "c", "items": []any{map[string]any{"title": "x"}}},
	} {
		_, err := rawCall("collections.set", params)
		assert.Error(t, err, name)
	}
	_, err = rawCall("collections.delete", map[string]any{})
	assert.Error(t, err)
	_, err = rawCall("model.download", map[string]any{"towers": []string{"video"}})
	assert.Error(t, err)
	_, err = rawCall("model.download", map[string]any{})
	assert.Error(t, err)
	_, err = rawCall("semantic.warm", map[string]any{"modalities": []string{"video"}})
	assert.Error(t, err)

	// extra item fields are tolerated
	_, err = rawCall("collections.set", map[string]any{"collection": "c", "items": []any{
		map[string]any{"id": "a", "title": "Dark theme", "text": st.Data.ActionDocs()[0].Text, "icon": "dark_mode", "keywords": []string{"night"}},
	}})
	assert.NoError(t, err)
}

func TestStatusAndWarm(t *testing.T) {
	files := &fakeIndexing{p: service.IndexProgress{Pending: 12, Done: 30, PausedReason: "battery"}}
	st := servicetest.NewStack(t, servicetest.Options{Models: &fakeModels{}, Indexing: files})
	serve(t, st)
	_, err := client.CollectionsSet("shell.actions", st.ActionItems())
	require.NoError(t, err)

	s, err := client.SemanticStatus()
	require.NoError(t, err)
	assert.True(t, s.Enabled)
	assert.True(t, s.Embedder.Available)
	assert.Equal(t, 20, s.Collections["shell.actions"])
	assert.Equal(t, 253, s.Collections["files.text"])
	assert.Equal(t, service.IndexProgress{Pending: 12, Done: 30, PausedReason: "battery"}, s.Indexing)
	assert.NotNil(t, s.Model)

	raw, err := rawCall("semantic.status", nil)
	require.NoError(t, err)
	var keys map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &keys))
	for _, k := range []string{"enabled", "embedder", "model", "collections", "indexing"} {
		assert.Contains(t, keys, k, "semantic.status carries %q", k)
	}

	_, err = client.SemanticWarm(nil)
	require.NoError(t, err)
	_, err = client.SemanticWarm([]string{"text", "image"})
	require.NoError(t, err)
	assert.Equal(t, []semantic.Modality{"text", "image", "text"}, st.Embedder.Warmed(), "default is text; the list is sorted")

	st.Embedder.SetAvailable(false, "libonnxruntime.so not found")
	_, err = client.SemanticWarm(nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "libonnxruntime.so not found")
	s, _ = client.SemanticStatus()
	assert.False(t, s.Embedder.Available)
	assert.Equal(t, "libonnxruntime.so not found", s.Embedder.Reason)
}

type fakeIndexing struct{ p service.IndexProgress }

func (f *fakeIndexing) Enqueue(string, bool)            {}
func (f *fakeIndexing) Progress() service.IndexProgress { return f.p }

func TestModelDownloadProgressEvents(t *testing.T) {
	models := &fakeModels{release: make(chan struct{})}
	st := servicetest.NewStack(t, servicetest.Options{Models: models})
	serve(t, st)

	ms, err := client.ModelStatus()
	require.NoError(t, err)
	assert.Contains(t, string(ms.Model), `"installed":true`)

	var mu sync.Mutex
	var events []service.ModelProgress
	done := make(chan error, 1)
	go func() {
		done <- client.ModelDownload([]string{"text", "image"}, func(p service.ModelProgress) {
			mu.Lock()
			events = append(events, p)
			mu.Unlock()
		})
	}()
	require.Eventually(t, func() bool {
		s, err := client.SemanticStatus()
		return err == nil && s.Download.Active && s.Download.Towers["image"].State == "downloading"
	}, 5*time.Second, 10*time.Millisecond, "progress shows up in status")

	// a second download is refused while one runs
	err = client.ModelDownload([]string{"audio"}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already running")
	assert.Error(t, client.ModelRemove([]string{"text"}), "removal during a download is refused")

	close(models.release)
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("download never reported completion")
	}
	mu.Lock()
	defer mu.Unlock()
	require.NotEmpty(t, events)
	assert.Equal(t, service.ModelProgress{State: service.StateFinished}, events[len(events)-1])
	towers := map[semantic.Modality]bool{}
	for _, e := range events {
		towers[e.Tower] = true
	}
	assert.True(t, towers["text"] && towers["image"], "both towers reported")

	require.NoError(t, client.ModelRemove([]string{"image"}))
	assert.Equal(t, []semantic.Modality{"image"}, models.removed)
}

func TestModelDownloadFailureReachesTheClient(t *testing.T) {
	models := &fakeModels{release: make(chan struct{}), fail: errors.New("sha256 mismatch for model_quantized.onnx_data")}
	close(models.release)
	st := servicetest.NewStack(t, servicetest.Options{Models: models})
	serve(t, st)
	err := client.ModelDownload([]string{"text"}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sha256 mismatch")
	s, _ := client.SemanticStatus()
	assert.Contains(t, s.Download.Error, "sha256 mismatch")
}

// Semantic search off: the daemon answers every old request as before, refuses
// collection writes clearly, and still manages model files.
func TestDisabledService(t *testing.T) {
	models := &fakeModels{release: make(chan struct{})}
	close(models.release)
	st := servicetest.NewStack(t, servicetest.Options{Disabled: true, Models: models})
	serve(t, st)

	res, err := client.SearchWithOptions(&client.SearchOptions{Query: "regexp", Limit: 5, Type: "file", Semantic: true, Collections: []string{"shell.actions"}})
	require.NoError(t, err)
	require.NotNil(t, res.Semantic)
	assert.False(t, res.Semantic.Available)
	assert.Equal(t, "semantic search is disabled", res.Semantic.Reason)
	assert.NotEmpty(t, res.Hits, "the lexical result is intact")
	plain, err := client.SearchWithOptions(&client.SearchOptions{Query: "regexp", Limit: 5, Type: "file"})
	require.NoError(t, err)
	require.Len(t, plain.Hits, len(res.Hits))
	for i := range plain.Hits {
		assert.Equal(t, plain.Hits[i].ID, res.Hits[i].ID)
		assert.Equal(t, plain.Hits[i].Score, res.Hits[i].Score, "same scores: nothing was fused")
	}

	_, err = client.CollectionsSet("shell.actions", st.ActionItems())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "disabled")
	_, err = client.SemanticWarm(nil)
	require.Error(t, err)

	s, err := client.SemanticStatus()
	require.NoError(t, err)
	assert.False(t, s.Enabled)
	assert.False(t, s.Embedder.Available)
	assert.Equal(t, "semantic search is disabled", s.Embedder.Reason)
	assert.Empty(t, s.Collections)

	// models can be fetched before the feature is turned on
	require.NoError(t, client.ModelDownload([]string{"text"}, nil))
	_, statErr := os.Stat(st.Config.SemanticDBPath())
	assert.True(t, os.IsNotExist(statErr), "a disabled daemon never creates the vector database")
}

func TestHTTPAPI(t *testing.T) {
	models := &fakeModels{release: make(chan struct{})}
	st := servicetest.NewStack(t, servicetest.Options{Models: models})
	h := newHTTPHandler(st.Indexer, stubWatcher{}, WithSemantic(st.Service))

	call := func(method, path, body string) (int, []byte) {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code, rec.Body.Bytes()
	}

	items, _ := json.Marshal(map[string]any{"items": st.ActionItems()})
	code, body := call(http.MethodPut, "/collections/shell.actions", string(items))
	require.Equal(t, http.StatusOK, code, string(body))
	var set service.SetResult
	require.NoError(t, json.Unmarshal(body, &set))
	assert.Equal(t, service.SetResult{Collection: "shell.actions", Total: 20, Embedded: 20}, set)

	code, body = call(http.MethodGet, "/search?q="+urlq("düzenli ifadelerle metin eşleştirme")+"&type=file&semantic=true&collections=shell.actions&collection_limit=2", "")
	require.Equal(t, http.StatusOK, code, string(body))
	var res service.HybridResult
	require.NoError(t, json.Unmarshal(body, &struct {
		Hits        *[]service.ResultHit      `json:"hits"`
		Collections *[]service.CollectionHits `json:"collections"`
	}{&res.Hits, &res.Collections}))
	require.NotEmpty(t, res.Hits)
	assert.Equal(t, st.Files["regexp"], res.Hits[0].ID)
	require.NotNil(t, res.Hits[0].SemanticScore)
	require.Len(t, res.Collections, 1)
	assert.Len(t, res.Collections[0].Hits, 2)

	// without semantic the response has no extras and collections is rejected
	code, body = call(http.MethodGet, "/search?q=regexp&type=file", "")
	require.Equal(t, http.StatusOK, code)
	assert.NotContains(t, string(body), `"semantic"`)
	assert.NotContains(t, string(body), `"collections"`)
	code, _ = call(http.MethodGet, "/search?q=regexp&collections=shell.actions", "")
	assert.Equal(t, http.StatusBadRequest, code)

	code, body = call(http.MethodGet, "/semantic/status", "")
	require.Equal(t, http.StatusOK, code)
	assert.Contains(t, string(body), `"shell.actions":20`)
	code, _ = call(http.MethodPost, "/semantic/warm?modalities=text", "")
	assert.Equal(t, http.StatusOK, code)
	code, _ = call(http.MethodPost, "/semantic/warm?modalities=video", "")
	assert.Equal(t, http.StatusBadRequest, code)
	code, body = call(http.MethodGet, "/model/status", "")
	require.Equal(t, http.StatusOK, code)
	assert.Contains(t, string(body), `"installed":true`)

	code, _ = call(http.MethodPost, "/model/download", `{"towers":["text"]}`)
	assert.Equal(t, http.StatusOK, code)
	code, _ = call(http.MethodPost, "/model/download", `{"towers":["text"]}`)
	assert.Equal(t, http.StatusConflict, code)
	code, _ = call(http.MethodPost, "/model/download", `{"towers":["video"]}`)
	assert.Equal(t, http.StatusBadRequest, code)
	close(models.release)
	require.Eventually(t, func() bool { return !st.Service.Status().Download.Active }, 5*time.Second, 10*time.Millisecond)
	code, _ = call(http.MethodPost, "/model/remove", `{"towers":["image"]}`)
	assert.Equal(t, http.StatusOK, code)

	code, _ = call(http.MethodDelete, "/collections/shell.actions", "")
	assert.Equal(t, http.StatusOK, code)
	code, _ = call(http.MethodPut, "/collections/files.text", `{"items":[]}`)
	assert.Equal(t, http.StatusBadRequest, code, "file collections belong to the indexer")

	code, body = call(http.MethodGet, "/openapi.json", "")
	require.Equal(t, http.StatusOK, code)
	for _, p := range []string{"/search", "/semantic/status", "/semantic/warm", "/collections/{name}", "/model/status", "/model/download", "/model/remove"} {
		assert.Contains(t, string(body), `"`+p+`"`, "documented in the OpenAPI schema")
	}
}

func TestHTTPDisabledAndWithoutModelManager(t *testing.T) {
	st := servicetest.NewStack(t, servicetest.Options{Disabled: true})
	h := newHTTPHandler(st.Indexer, stubWatcher{}, WithSemantic(st.Service))
	do := func(method, path, body string) int {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code
	}
	assert.Equal(t, http.StatusConflict, do(http.MethodPut, "/collections/c", `{"items":[]}`))
	assert.Equal(t, http.StatusNotImplemented, do(http.MethodGet, "/model/status", ""))
	assert.Equal(t, http.StatusNotImplemented, do(http.MethodPost, "/model/download", `{"towers":["text"]}`))
	assert.Equal(t, http.StatusOK, do(http.MethodGet, "/semantic/status", ""))
	assert.Equal(t, http.StatusOK, do(http.MethodGet, "/search?q=regexp&semantic=true", ""))

	// a handler built without any semantic option behaves the same
	bare := newHTTPHandler(st.Indexer, stubWatcher{})
	rec := httptest.NewRecorder()
	bare.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/semantic/status", nil))
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), `"enabled":false`)
}

func TestHTTPCollectionsParameterForms(t *testing.T) {
	st := servicetest.NewStack(t, servicetest.Options{})
	h := newHTTPHandler(st.Indexer, stubWatcher{}, WithSemantic(st.Service))
	for _, name := range []string{"shell.actions", "shell.apps"} {
		_, err := st.Service.SetCollection(context.Background(), name, st.ActionItems())
		require.NoError(t, err)
	}
	for name, qs := range map[string]string{
		"repeated": "collections=shell.actions&collections=shell.apps",
		"comma":    "collections=shell.actions,shell.apps",
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/search?q="+urlq("karanlık")+"&type=file&semantic=true&"+qs, nil))
		require.Equal(t, http.StatusOK, rec.Code, name)
		var body struct {
			Collections []service.CollectionHits `json:"collections"`
		}
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		require.Len(t, body.Collections, 2, name)
		assert.Equal(t, "shell.actions", body.Collections[0].Collection)
		assert.Equal(t, "shell.apps", body.Collections[1].Collection)
	}
}
