package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

const hub = "https://huggingface.co"

// Real Hugging Face files stand in for the towers: the mechanics (Range,
// sha256, atomic rename) are the same for a 300 MB file and a 5 MB one, and
// the large files are exercised by TestDownloadFullModel on demand.
var (
	fConfig    = pick("config.json")
	fChat      = pick("chat_template.jinja")
	fProc      = pick("processor_config.json")
	fPre       = pick("preprocessor_config.json")
	fTokenizer = pick("tokenizer.model")
)

func pick(path string) File {
	for _, f := range commonFiles {
		if f.Path == path {
			return f
		}
	}
	panic(path)
}

func smallFiles(t semantic.Modality) []File {
	switch t {
	case semantic.ModalityText:
		return []File{fConfig, fChat, fTokenizer}
	case semantic.ModalityImage:
		return []File{fProc}
	case semantic.ModalityAudio:
		return []File{fPre}
	}
	return nil
}

var (
	netOnce sync.Once
	netErr  error
)

func needNet(t *testing.T) {
	t.Helper()
	netOnce.Do(func() {
		c := http.Client{Timeout: 15 * time.Second}
		resp, err := c.Get(hub + "/api/models/" + onnxRepo)
		if err != nil {
			netErr = err
			return
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			netErr = fmt.Errorf("HTTP %d", resp.StatusCode)
		}
	})
	if netErr != nil {
		t.Skipf("huggingface.co is not reachable (%v); this test needs the real hub", netErr)
	}
}

func testDownloader() *downloader {
	d := newDownloader()
	d.files = smallFiles
	d.retries = 2
	d.backoff = func(int) time.Duration { return 10 * time.Millisecond }
	d.margin = func(int64) int64 { return 1 << 20 }
	return d
}

// countingTransport records the Range header of every content response.
type countingTransport struct {
	rt     http.RoundTripper
	mu     sync.Mutex
	ranges []string
	bytes  atomic.Int64
}

func (c *countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	resp, err := c.rt.RoundTrip(r)
	// Redirect hops carry no payload; only the content responses count.
	if err == nil && (resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusPartialContent) {
		c.mu.Lock()
		c.ranges = append(c.ranges, r.Header.Get("Range"))
		c.mu.Unlock()
		resp.Body = &countBody{ReadCloser: resp.Body, n: &c.bytes}
	}
	return resp, err
}

type countBody struct {
	io.ReadCloser
	n *atomic.Int64
}

func (b *countBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.n.Add(int64(n))
	return n, err
}

func (c *countingTransport) requests() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.ranges...)
}

func counting(d *downloader) *countingTransport {
	ct := &countingTransport{rt: d.client.Transport}
	d.client = &http.Client{Transport: ct}
	return ct
}

var realBytes sync.Map // path -> []byte, fetched once from the hub

func realFile(t *testing.T, f File) []byte {
	t.Helper()
	needNet(t)
	if v, ok := realBytes.Load(f.Path); ok {
		return v.([]byte)
	}
	resp, err := http.Get(f.URL(hub))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("fetching %s: %v (HTTP %d)", f.Path, err, resp.StatusCode)
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != f.SHA256 || int64(len(b)) != f.Size {
		t.Fatalf("hub content of %s differs from the pinned manifest", f.Path)
	}
	realBytes.Store(f.Path, b)
	return b
}

// statusSmall is Status for the stand-in files.
func statusSmall(dir string) ModelStatus { return statusOf(dir, smallFiles) }

func TestManifestMatchesHub(t *testing.T) {
	needNet(t)
	trees := map[string]map[string]struct {
		Size int64 `json:"size"`
		LFS  *struct {
			Oid string `json:"oid"`
		} `json:"lfs"`
	}{}
	get := func(repo, rev string) map[string]struct {
		Size int64 `json:"size"`
		LFS  *struct {
			Oid string `json:"oid"`
		} `json:"lfs"`
	} {
		key := repo + "@" + rev
		if m, ok := trees[key]; ok {
			return m
		}
		resp, err := http.Get(fmt.Sprintf("%s/api/models/%s/tree/%s?recursive=1", hub, repo, rev))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var entries []struct {
			Type string `json:"type"`
			Path string `json:"path"`
			Size int64  `json:"size"`
			LFS  *struct {
				Oid string `json:"oid"`
			} `json:"lfs"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
			t.Fatal(err)
		}
		m := map[string]struct {
			Size int64 `json:"size"`
			LFS  *struct {
				Oid string `json:"oid"`
			} `json:"lfs"`
		}{}
		for _, e := range entries {
			if e.Type == "file" {
				m[e.Path] = struct {
					Size int64 `json:"size"`
					LFS  *struct {
						Oid string `json:"oid"`
					} `json:"lfs"`
				}{e.Size, e.LFS}
			}
		}
		trees[key] = m
		return m
	}
	for _, tower := range allTowers {
		for _, f := range Files(tower) {
			e, ok := get(f.Repo, f.Revision)[f.Path]
			if !ok {
				t.Errorf("%s/%s not in the pinned commit", f.Repo, f.Path)
				continue
			}
			if e.Size != f.Size {
				t.Errorf("%s: size %d, hub says %d", f.Path, f.Size, e.Size)
			}
			switch {
			case e.LFS != nil:
				if e.LFS.Oid != f.SHA256 {
					t.Errorf("%s: sha256 %s, hub LFS oid says %s", f.Path, f.SHA256, e.LFS.Oid)
				}
			default:
				realFile(t, f) // checks the content sha256 itself
			}
		}
	}
	if onnxRevision != "daa72c51243991dfcaf9f9137d2c573d8f7790c0" {
		t.Errorf("onnx revision changed to %s", onnxRevision)
	}
}

func TestDownloadStatusAndIdempotence(t *testing.T) {
	needNet(t)
	dir := filepath.Join(t.TempDir(), "models", "gemma")
	d := testDownloader()
	ct := counting(d)

	if st := statusSmall(dir); st.Ready || len(st.Installed()) != 0 {
		t.Fatalf("empty dir reports installed towers: %+v", st.Installed())
	}
	var last Progress
	var seen int
	err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText, semantic.ModalityAudio}, func(p Progress) {
		seen++
		last = p
		if p.Bytes > p.Total {
			t.Errorf("progress beyond total: %+v", p)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if seen == 0 || last.State != StateDone || last.Tower != semantic.ModalityAudio || last.Bytes != last.Total {
		t.Fatalf("last progress = %+v after %d reports", last, seen)
	}
	st := statusSmall(dir)
	if !st.Ready || fmt.Sprint(st.Installed()) != "[text audio]" {
		t.Fatalf("installed = %v", st.Installed())
	}
	img, _ := st.Tower(semantic.ModalityImage)
	if img.Installed || img.Files[0].State != FileMissing {
		t.Fatalf("image tower must be absent: %+v", img)
	}
	for _, f := range smallFiles(semantic.ModalityText) {
		b, err := os.ReadFile(filepath.Join(dir, f.Path))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(b)
		if hex.EncodeToString(sum[:]) != f.SHA256 {
			t.Errorf("%s on disk does not match the pinned sha256", f.Path)
		}
	}
	if ents, _ := filepath.Glob(filepath.Join(dir, "*.part")); len(ents) != 0 {
		t.Errorf("leftover part files: %v", ents)
	}

	// A second run with everything verified needs no network at all.
	before := len(ct.requests())
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText, semantic.ModalityAudio}, nil); err != nil {
		t.Fatal(err)
	}
	if n := len(ct.requests()) - before; n != 0 {
		t.Fatalf("second download made %d requests", n)
	}
}

func TestTextTowerIsMandatory(t *testing.T) {
	needNet(t)
	dir := t.TempDir()
	d := testDownloader()
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityImage}, nil); err != nil {
		t.Fatal(err)
	}
	if got := statusSmall(dir).Installed(); fmt.Sprint(got) != "[text image]" {
		t.Fatalf("installed = %v, want text added to image", got)
	}
}

func TestResumeUsesRange(t *testing.T) {
	needNet(t)
	full := realFile(t, fTokenizer)
	dir := t.TempDir()
	cut := int64(1_500_000)
	part := filepath.Join(dir, fTokenizer.Path+partSuffix)
	if err := os.WriteFile(part, full[:cut], 0o644); err != nil {
		t.Fatal(err)
	}
	d := testDownloader()
	d.files = func(semantic.Modality) []File { return []File{fTokenizer} }
	ct := counting(d)

	if st := statusOf(dir, d.files); !st.Towers[0].Partial || st.Towers[0].Bytes != cut {
		t.Fatalf("partial download not reported: %+v", st.Towers[0])
	}
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil); err != nil {
		t.Fatal(err)
	}
	reqs := ct.requests()
	if len(reqs) == 0 || reqs[0] != fmt.Sprintf("bytes=%d-", cut) {
		t.Fatalf("expected a resumed Range request, got %v", reqs)
	}
	if got := ct.bytes.Load(); got != fTokenizer.Size-cut {
		t.Fatalf("transferred %d bytes, want %d", got, fTokenizer.Size-cut)
	}
	if !statusOf(dir, d.files).Ready {
		t.Fatal("not installed after resume")
	}
}

func TestCorruptPartStartsOver(t *testing.T) {
	needNet(t)
	full := realFile(t, fTokenizer)
	dir := t.TempDir()
	bad := append([]byte(nil), full[:900_000]...)
	bad[1234] ^= 0xff
	if err := os.WriteFile(filepath.Join(dir, fTokenizer.Path+partSuffix), bad, 0o644); err != nil {
		t.Fatal(err)
	}
	d := testDownloader()
	d.files = func(semantic.Modality) []File { return []File{fTokenizer} }
	ct := counting(d)
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil); err != nil {
		t.Fatal(err)
	}
	reqs := ct.requests()
	if len(reqs) != 2 || reqs[0] == "" || reqs[1] != "" {
		t.Fatalf("want a resumed request followed by a clean restart, got %v", reqs)
	}
	got, _ := os.ReadFile(filepath.Join(dir, fTokenizer.Path))
	if string(got) != string(full) {
		t.Fatal("file content differs after restart")
	}
}

func TestAdoptsExistingFileWithoutNetwork(t *testing.T) {
	needNet(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, fConfig.Path), realFile(t, fConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	if st := statusOf(dir, func(semantic.Modality) []File { return []File{fConfig} }); st.Towers[0].Files[0].State != FileUnverified {
		t.Fatalf("copied file must be unverified until hashed, got %s", st.Towers[0].Files[0].State)
	}
	d := testDownloader()
	d.files = func(semantic.Modality) []File { return []File{fConfig} }
	d.interval = 0 // every report, so none can hide behind the throttle
	ct := counting(d)
	var states []string
	progress := func(p Progress) { states = append(states, p.State) }
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, progress); err != nil {
		t.Fatal(err)
	}
	if n := len(ct.requests()); n != 0 {
		t.Fatalf("adopting a correct file downloaded it again (%d requests)", n)
	}
	if !statusOf(dir, d.files).Ready {
		t.Fatal("adopted file not verified")
	}
	if slices.Contains(states, StateDownloading) {
		t.Fatalf("a file that was only hashed is reported as downloading: %v", states)
	}
}

func TestReplacedFileIsNoLongerVerified(t *testing.T) {
	needNet(t)
	dir := t.TempDir()
	d := testDownloader()
	d.files = func(semantic.Modality) []File { return []File{fConfig} }
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, fConfig.Path)
	raw, _ := os.ReadFile(p)
	raw[10] ^= 1 // same size, different content, later mtime
	future := time.Now().Add(time.Hour)
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	os.Chtimes(p, future, future)
	if statusOf(dir, d.files).Ready {
		t.Fatal("modified file still reported as verified")
	}
	// Download notices, hashes, finds the mismatch and fetches a good copy.
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(p)
	if string(got) != string(realFile(t, fConfig)) {
		t.Fatal("modified file was not repaired")
	}
}

func TestInsufficientSpace(t *testing.T) {
	dir := t.TempDir()
	d := testDownloader()
	d.freeFn = func(string) (uint64, error) { return 1 << 20, nil }
	d.margin = func(int64) int64 { return 64 << 20 }
	err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil)
	var se *SpaceError
	if !errors.Is(err, ErrInsufficientSpace) || !errors.As(err, &se) || se.Free != 1<<20 || se.Need <= 64<<20 {
		t.Fatalf("err = %v", err)
	}
	if ents, _ := os.ReadDir(dir); len(ents) > 1 { // only the lock file may exist
		t.Fatalf("files were created despite missing space: %v", ents)
	}
}

func TestCancelKeepsPartAndResumes(t *testing.T) {
	needNet(t)
	dir := t.TempDir()
	d := testDownloader()
	d.files = func(semantic.Modality) []File { return []File{fTokenizer} }
	d.interval = 0
	ctx, cancel := context.WithCancel(context.Background())
	err := d.download(ctx, dir, []semantic.Modality{semantic.ModalityText}, func(p Progress) {
		if p.Bytes > 600_000 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	info, serr := os.Stat(filepath.Join(dir, fTokenizer.Path+partSuffix))
	if serr != nil || info.Size() == 0 || info.Size() >= fTokenizer.Size {
		t.Fatalf("expected a resumable part file, got %v %v", info, serr)
	}
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil); err != nil {
		t.Fatal(err)
	}
	if !statusOf(dir, d.files).Ready {
		t.Fatal("not installed after resume")
	}
}

func TestNetworkFailureIsMeaningful(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close() // nothing listens here any more
	d := testDownloader()
	d.baseURL = "http://" + addr
	d.files = func(semantic.Modality) []File { return []File{fConfig} }
	err = d.download(context.Background(), t.TempDir(), []semantic.Modality{semantic.ModalityText}, nil)
	var ne *NetworkError
	if !errors.Is(err, ErrNetwork) || !errors.As(err, &ne) || ne.File != fConfig.Path || ne.Attempts != 3 {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(err.Error(), "config.json") {
		t.Fatalf("error does not name the file: %v", err)
	}
}

// proxy serves the real bytes of a file with the failure modes under test.
func proxy(t *testing.T, f File, mode string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	full := realFile(t, f)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		switch mode {
		case "notfound":
			http.NotFound(w, r)
			return
		case "unavailable":
			if n <= 2 {
				http.Error(w, "try later", http.StatusServiceUnavailable)
				return
			}
		case "ignorerange":
			r.Header.Del("Range")
		case "wrongsize":
			w.Header().Set("Content-Length", fmt.Sprint(len(full)+5))
			w.Write(full)
			return
		case "tampered":
			b := append([]byte(nil), full...)
			b[100] ^= 1
			w.Write(b)
			return
		}
		http.ServeContent(w, r, "", time.Time{}, strings.NewReader(string(full)))
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func serveVia(srv *httptest.Server, f File) *downloader {
	d := testDownloader()
	d.baseURL = srv.URL
	d.files = func(semantic.Modality) []File { return []File{f} }
	return d
}

func TestHTTPFailureModes(t *testing.T) {
	needNet(t)
	cases := []struct {
		mode string
		ok   bool
		want func(error) bool
	}{
		{"notfound", false, func(err error) bool { var he *HTTPStatusError; return errors.As(err, &he) && he.Status == 404 }},
		{"unavailable", true, nil},
		{"ignorerange", true, nil},
		{"wrongsize", false, func(err error) bool { return errors.Is(err, ErrNetwork) }},
		{"tampered", false, func(err error) bool { return errors.Is(err, ErrChecksum) }},
	}
	for _, c := range cases {
		t.Run(c.mode, func(t *testing.T) {
			srv, hits := proxy(t, fConfig, c.mode)
			d := serveVia(srv, fConfig)
			dir := t.TempDir()
			if c.mode == "ignorerange" { // force a resume attempt against a server that ignores Range
				full := realFile(t, fConfig)
				os.WriteFile(filepath.Join(dir, fConfig.Path+partSuffix), full[:1000], 0o644)
			}
			err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil)
			if c.ok {
				if err != nil {
					t.Fatal(err)
				}
				if !statusOf(dir, d.files).Ready {
					t.Fatal("not installed")
				}
				if c.mode == "unavailable" && hits.Load() != 3 {
					t.Fatalf("expected 2 retries after 503, got %d requests", hits.Load())
				}
				return
			}
			if err == nil || !c.want(err) {
				t.Fatalf("err = %v", err)
			}
			if c.mode == "notfound" && hits.Load() != 1 {
				t.Fatalf("404 must not be retried, got %d requests", hits.Load())
			}
			if _, serr := os.Stat(filepath.Join(dir, fConfig.Path)); serr == nil {
				t.Fatal("a bad download reached its final name")
			}
			if _, serr := os.Stat(filepath.Join(dir, fConfig.Path+partSuffix)); c.mode == "tampered" && serr == nil {
				t.Fatal("tampered part file kept")
			}
		})
	}
}

func TestStalledConnectionRetriesWithRange(t *testing.T) {
	needNet(t)
	full := realFile(t, fTokenizer)
	var hits atomic.Int32
	var ranges []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := hits.Add(1)
		mu.Lock()
		ranges = append(ranges, r.Header.Get("Range"))
		mu.Unlock()
		if n == 1 { // send a prefix, then go silent
			w.Header().Set("Content-Length", fmt.Sprint(len(full)))
			w.Write(full[:700_000])
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			return
		}
		http.ServeContent(w, r, "", time.Time{}, strings.NewReader(string(full)))
	}))
	t.Cleanup(srv.Close)
	d := serveVia(srv, fTokenizer)
	d.stall = 300 * time.Millisecond
	dir := t.TempDir()
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[0] != "" || ranges[1] != "bytes=700000-" {
		t.Fatalf("requests = %q", ranges)
	}
	if !statusOf(dir, d.files).Ready {
		t.Fatal("not installed")
	}
}

func TestRemoveAndBusy(t *testing.T) {
	needNet(t)
	dir := filepath.Join(t.TempDir(), "gemma")
	d := testDownloader()
	if err := d.download(context.Background(), dir, []semantic.Modality{semantic.ModalityText, semantic.ModalityImage}, nil); err != nil {
		t.Fatal(err)
	}
	foreign := filepath.Join(dir, "notes.txt")
	os.WriteFile(foreign, []byte("mine"), 0o644)

	unlock, err := lockDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := removeFiles(dir, []semantic.Modality{semantic.ModalityImage}, smallFiles); !errors.Is(err, ErrBusy) {
		t.Fatalf("remove while locked: %v", err)
	}
	unlock()

	if err := removeFiles(dir, []semantic.Modality{semantic.ModalityImage}, smallFiles); err != nil {
		t.Fatal(err)
	}
	if got := statusSmall(dir).Installed(); fmt.Sprint(got) != "[text]" {
		t.Fatalf("after removing image: %v", got)
	}
	if err := removeFiles(dir, []semantic.Modality{semantic.ModalityText}, smallFiles); err != nil {
		t.Fatal(err)
	}
	if st := statusSmall(dir); st.Ready || len(st.Installed()) != 0 {
		t.Fatalf("after removing text: %v", st.Installed())
	}
	if _, err := os.Stat(foreign); err != nil {
		t.Fatal("foreign file was removed")
	}
	os.Remove(foreign)
	if err := removeFiles(dir, []semantic.Modality{semantic.ModalityText}, smallFiles); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("empty model dir should be gone: %v", err)
	}
	if err := Remove(filepath.Join(t.TempDir(), "never"), []semantic.Modality{semantic.ModalityText}); err != nil {
		t.Fatalf("removing a missing model: %v", err)
	}
}

func TestUnknownTower(t *testing.T) {
	if err := Download(context.Background(), t.TempDir(), []semantic.Modality{"video"}, nil); !errors.Is(err, ErrUnknownTower) {
		t.Fatalf("Download: %v", err)
	}
	if err := Remove(t.TempDir(), []semantic.Modality{"video"}); !errors.Is(err, ErrUnknownTower) {
		t.Fatalf("Remove: %v", err)
	}
}

func TestConcurrentDownloadIsRejected(t *testing.T) {
	dir := t.TempDir()
	unlock, err := lockDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	if err := testDownloader().download(context.Background(), dir, []semantic.Modality{semantic.ModalityText}, nil); !errors.Is(err, ErrBusy) {
		t.Fatalf("err = %v", err)
	}
}

func TestStatusOnPinnedManifest(t *testing.T) {
	st := Status(filepath.Join(t.TempDir(), "absent"))
	if st.Ready || len(st.Towers) != 3 || st.ModelID != ModelID {
		t.Fatalf("%+v", st)
	}
	var total int64
	for _, ts := range st.Towers {
		total += ts.Total
		if ts.Total != TowerSize(ts.Tower) || len(ts.Files) != len(Files(ts.Tower)) {
			t.Fatalf("%s: %+v", ts.Tower, ts)
		}
	}
	// text 314 MB + tokenizer/config 37 MB, image 195 MB, audio 340 MB (MB = 10^6)
	if total < 835<<20 || total > 860<<20 {
		t.Fatalf("manifest total %d MiB", total>>20)
	}
}

// TestDownloadFullModel installs the three real towers. It moves ~900 MB, so
// it only runs when DUX_MODEL_FULL_DOWNLOAD_DIR names an empty scratch dir.
func TestDownloadFullModel(t *testing.T) {
	dir := os.Getenv("DUX_MODEL_FULL_DOWNLOAD_DIR")
	if dir == "" {
		t.Skip("set DUX_MODEL_FULL_DOWNLOAD_DIR to run the ~900 MB download")
	}
	start := time.Now()
	var lastLog time.Time
	err := Download(context.Background(), dir, Towers(), func(p Progress) {
		if time.Since(lastLog) > 15*time.Second || p.State != StateDownloading {
			lastLog = time.Now()
			t.Logf("%s %s %d/%d MiB %s", p.Tower, p.File, p.Bytes>>20, p.Total>>20, p.State)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("downloaded in %s", time.Since(start).Round(time.Second))
	st := Status(dir)
	if fmt.Sprint(st.Installed()) != "[text image audio]" {
		t.Fatalf("installed = %v", st.Installed())
	}
	for _, ts := range st.Towers {
		for _, f := range ts.Files {
			if f.State != FileVerified {
				t.Errorf("%s: %s", f.Path, f.State)
			}
		}
	}
	// A fresh Status after "restart" must not hash anything: time it.
	t0 := time.Now()
	Status(dir)
	if d := time.Since(t0); d > 100*time.Millisecond {
		t.Errorf("Status took %s, it must only stat files", d)
	}
}

// TestRemoveFullModel removes a model installed by TestDownloadFullModel (same
// environment variable) tower by tower and checks that nothing is left.
func TestRemoveFullModel(t *testing.T) {
	dir := os.Getenv("DUX_MODEL_FULL_DOWNLOAD_DIR")
	if dir == "" {
		t.Skip("set DUX_MODEL_FULL_DOWNLOAD_DIR to a directory holding the full model")
	}
	if got := fmt.Sprint(Status(dir).Installed()); got != "[text image audio]" {
		t.Skipf("no full model in %s (%s)", dir, got)
	}
	if err := Remove(dir, []semantic.Modality{semantic.ModalityAudio}); err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(Status(dir).Installed()); got != "[text image]" {
		t.Fatalf("after removing audio: %s", got)
	}
	if err := Remove(dir, []semantic.Modality{semantic.ModalityText}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		left, _ := os.ReadDir(dir)
		t.Fatalf("model directory still exists: %v %v", err, left)
	}
}
