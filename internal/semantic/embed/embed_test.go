package embed

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/ort"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/embedtest"
)

// requireRuntime skips the test when no ONNX Runtime library can be loaded.
func requireRuntime(t testing.TB) {
	t.Helper()
	rt, err := ort.Load(embedtest.ORTLibrary())
	if err != nil {
		t.Skipf("onnxruntime not available: %v", err)
	}
	rt.Close()
}

const parentEnv = "DSEARCH_EMBED_TEST_PARENT"

func TestMain(m *testing.M) {
	if os.Getenv(parentEnv) != "" {
		parentMain()
		return
	}
	code := m.Run()
	embedtest.CleanupHelper()
	os.Exit(code)
}

// parentMain is the body of the child process TestHelperDiesWithParent kills:
// it embeds one query (so the helper is running), prints the helper's pid and
// waits to be killed.
func parentMain() {
	e, err := New(Config{ModelDir: os.Getenv("MODEL_DIR"), HelperPath: os.Getenv("HELPER"), ORTLibrary: embedtest.ORTLibrary()})
	if err != nil {
		fmt.Println("error", err)
		os.Exit(1)
	}
	if _, err := e.EmbedQueries(context.Background(), []string{"hello"}); err != nil {
		fmt.Println("error", err)
		os.Exit(1)
	}
	fmt.Println(e.(*client).h.cmd.Process.Pid)
	select {}
}

func newClient(t *testing.T, mut func(*Config)) *client {
	t.Helper()
	dir := embedtest.ModelDir(t)
	requireRuntime(t)
	cfg := Config{ModelDir: dir, HelperPath: embedtest.HelperBinary(t), ORTLibrary: embedtest.ORTLibrary(), IdleUnload: time.Minute}
	if mut != nil {
		mut(&cfg)
	}
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	return e.(*client)
}

func helperPID(c *client) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.h == nil {
		return 0
	}
	return c.h.cmd.Process.Pid
}

func processExists(pid int) bool {
	// Signal 0 succeeds for zombies too; read the state from /proc instead.
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return false
	}
	// "pid (comm) S ..." — Z means the child was killed but not yet reaped.
	for i := len(b) - 1; i > 0; i-- {
		if b[i] == ')' {
			return i+2 < len(b) && b[i+2] != 'Z'
		}
	}
	return true
}

func waitFor(t *testing.T, what string, d time.Duration, cond func() bool) {
	t.Helper()
	end := time.Now().Add(d)
	for time.Now().Before(end) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

type actionsFile struct {
	Actions []struct{ Key, Text string }  `json:"actions"`
	Queries []struct{ Text, Want string } `json:"queries"`
}

func loadActions(t *testing.T) actionsFile {
	t.Helper()
	raw, err := os.ReadFile("testdata/actions.json")
	if err != nil {
		t.Fatal(err)
	}
	var a actionsFile
	if err := json.Unmarshal(raw, &a); err != nil {
		t.Fatal(err)
	}
	return a
}

func loadTextRef(t *testing.T) [][]float32 {
	t.Helper()
	raw, err := os.ReadFile("testdata/text_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var r struct {
		Items []struct {
			Vec []float32 `json:"vec"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	out := make([][]float32, len(r.Items))
	for i := range r.Items {
		out[i] = r.Items[i].Vec
	}
	return out
}

// TestActionsEndToEnd is the acceptance run of the text path: the real helper
// process, real model, the 20 actions and 15 Turkish queries of
// degerlendirme/eylem_eslesme.py at 768d: 15/15 first-rank hits and every
// vector within cosine 0.999 of Python's.
func TestActionsEndToEnd(t *testing.T) {
	c := newClient(t, nil)
	a, ref := loadActions(t), loadTextRef(t)
	ctx := context.Background()

	if st := c.Status(); !st.Available || st.Running {
		t.Fatalf("before use: %+v", st)
	}
	docs := make([]semantic.TextDoc, len(a.Actions))
	for i, x := range a.Actions {
		docs[i] = semantic.TextDoc{Title: x.Key, Text: x.Text}
	}
	qs := make([]string, len(a.Queries))
	for i, q := range a.Queries {
		qs[i] = q.Text
	}
	dv, err := c.EmbedDocuments(ctx, docs)
	if err != nil {
		t.Fatal(err)
	}
	qv, err := c.EmbedQueries(ctx, qs)
	if err != nil {
		t.Fatal(err)
	}
	if st := c.Status(); !st.Available || !st.Running || len(st.Loaded) != 1 || st.Loaded[0] != semantic.ModalityText {
		t.Fatalf("after use: %+v", st)
	}
	minCos := 1.0
	check := func(v, want []float32, what string) {
		cs := embedtest.Cosine(v, want)
		if cs < minCos {
			minCos = cs
		}
		if cs < 0.999 {
			t.Errorf("%s: cosine to Python %.5f", what, cs)
		}
	}
	for i := range dv {
		check(dv[i], ref[i], "doc "+a.Actions[i].Key)
	}
	hits := 0
	for i := range qv {
		check(qv[i], ref[len(dv)+i], "query "+qs[i])
		best, bs := 0, -2.0
		for j := range dv {
			if s := embedtest.Cosine(qv[i], dv[j]); s > bs {
				best, bs = j, s
			}
		}
		if a.Actions[best].Key == a.Queries[i].Want {
			hits++
		} else {
			t.Errorf("query %q: got %s, want %s", qs[i], a.Actions[best].Key, a.Queries[i].Want)
		}
	}
	t.Logf("first-rank hits %d/%d; minimum cosine to Python %.6f", hits, len(qs), minCos)
	if hits != len(qs) {
		t.Fatalf("hits %d/%d", hits, len(qs))
	}
}

func TestDocumentWithoutTitleUsesNone(t *testing.T) {
	c := newClient(t, nil)
	ctx := context.Background()
	a, err := c.EmbedDocuments(ctx, []semantic.TextDoc{{Text: "Night Light warms the screen."}})
	if err != nil {
		t.Fatal(err)
	}
	b, err := c.EmbedDocuments(ctx, []semantic.TextDoc{{Title: "none", Text: "Night Light warms the screen."}})
	if err != nil {
		t.Fatal(err)
	}
	if cs := embedtest.Cosine(a[0], b[0]); cs < 0.99999 {
		t.Fatalf("empty title must mean \"none\": cosine %.6f", cs)
	}
}

func TestEmptyInputs(t *testing.T) {
	c := newClient(t, nil)
	ctx := context.Background()
	if v, err := c.EmbedQueries(ctx, nil); err != nil || len(v) != 0 {
		t.Fatalf("%v %v", v, err)
	}
	if v, err := c.EmbedImages(ctx, nil); err != nil || len(v) != 0 {
		t.Fatalf("%v %v", v, err)
	}
	if v, err := c.EmbedAudio(ctx, nil); err != nil || len(v) != 0 {
		t.Fatalf("%v %v", v, err)
	}
	if helperPID(c) != 0 {
		t.Fatal("empty calls must not start the helper")
	}
}

func TestConcurrentCalls(t *testing.T) {
	c := newClient(t, nil)
	ctx := context.Background()
	want, err := c.EmbedQueries(ctx, []string{"karanlık", "fener", "sesi kıs", "pil tasarrufu"})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				j := (g + i) % len(want)
				q := []string{"karanlık", "fener", "sesi kıs", "pil tasarrufu"}[j]
				v, err := c.EmbedQueries(ctx, []string{q})
				if err != nil {
					errs <- err
					return
				}
				if embedtest.Cosine(v[0], want[j]) < 0.9999 {
					errs <- fmt.Errorf("goroutine %d: wrong vector for %q", g, q)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestIdleUnloadAndRestart(t *testing.T) {
	c := newClient(t, func(cfg *Config) { cfg.IdleUnload = 400 * time.Millisecond })
	ctx := context.Background()
	if _, err := c.EmbedQueries(ctx, []string{"karanlık"}); err != nil {
		t.Fatal(err)
	}
	pid := helperPID(c)
	if pid == 0 || !processExists(pid) {
		t.Fatalf("helper not running (pid %d)", pid)
	}
	waitFor(t, "idle unload", 5*time.Second, func() bool { return !c.Status().Running })
	waitFor(t, "helper process to be gone", 5*time.Second, func() bool { return !processExists(pid) })
	if st := c.Status(); len(st.Loaded) != 0 || !st.Available {
		t.Fatalf("after unload: %+v", st)
	}
	// The next call starts a new helper.
	if _, err := c.EmbedQueries(ctx, []string{"fener"}); err != nil {
		t.Fatal(err)
	}
	if p2 := helperPID(c); p2 == 0 || p2 == pid {
		t.Fatalf("expected a fresh helper, pid %d -> %d", pid, p2)
	}
}

func TestBusyHelperIsNotUnloaded(t *testing.T) {
	c := newClient(t, func(cfg *Config) { cfg.IdleUnload = 300 * time.Millisecond })
	// A batch that takes longer than the idle time must finish normally.
	texts := make([]string, 200)
	for i := range texts {
		texts[i] = fmt.Sprintf("Night Light number %d warms the screen colors and reduces eye strain during long evenings of reading", i)
	}
	start := time.Now()
	v, err := c.EmbedQueries(context.Background(), texts)
	if err != nil || len(v) != len(texts) {
		t.Fatalf("%v (%d vectors)", err, len(v))
	}
	if time.Since(start) < 400*time.Millisecond {
		t.Skip("machine too fast for this scenario")
	}
}

func TestCrashRecovery(t *testing.T) {
	c := newClient(t, nil)
	ctx := context.Background()
	if _, err := c.EmbedQueries(ctx, []string{"karanlık"}); err != nil {
		t.Fatal(err)
	}
	pid := helperPID(c)
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "client to notice the exit", 5*time.Second, func() bool { return !c.Status().Running })
	v, err := c.EmbedQueries(ctx, []string{"karanlık"})
	if err != nil || len(v) != 1 {
		t.Fatalf("after crash: %v", err)
	}
	if p2 := helperPID(c); p2 == pid {
		t.Fatal("same pid after a kill")
	}
}

// A kill in the middle of a call is invisible to the caller: the request is
// retried once on a fresh helper.
func TestCrashDuringCallIsRetried(t *testing.T) {
	c := newClient(t, nil)
	ctx := context.Background()
	if err := c.Warm(ctx); err != nil {
		t.Fatal(err)
	}
	pid := helperPID(c)
	texts := make([]string, 120)
	for i := range texts {
		texts[i] = fmt.Sprintf("document %d about night light, dark theme and wireless headphones for long reading sessions", i)
	}
	go func() {
		time.Sleep(150 * time.Millisecond)
		syscall.Kill(pid, syscall.SIGKILL)
	}()
	v, err := c.EmbedQueries(ctx, texts)
	if err != nil || len(v) != len(texts) {
		t.Fatalf("call that lost its helper: %v (%d vectors)", err, len(v))
	}
}

func TestCrashLoopFailsFast(t *testing.T) {
	// A helper that dies on every start must not be restarted in a hot loop.
	dir := t.TempDir()
	bad := filepath.Join(dir, "dsearch-embed")
	if err := os.WriteFile(bad, []byte("#!/bin/sh\necho 'boom: cannot map weights' >&2\nexit 3\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	model := embedtest.ModelDir(t)
	e, err := New(Config{ModelDir: model, HelperPath: bad})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	_, err = e.EmbedQueries(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected an error")
	}
	if errors.Is(err, semantic.ErrModelUnavailable) {
		t.Fatalf("a crashing helper is not 'model unavailable': %v", err)
	}
	t.Logf("error: %v", err)
	begin := time.Now()
	if _, err := e.EmbedQueries(context.Background(), []string{"x"}); err == nil || time.Since(begin) > 500*time.Millisecond {
		t.Fatalf("second call should fail fast, took %v (%v)", time.Since(begin), err)
	}
}

func TestHelperDiesWithParent(t *testing.T) {
	model := embedtest.ModelDir(t)
	requireRuntime(t)
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), parentEnv+"=1", "MODEL_DIR="+model, "HELPER="+embedtest.HelperBinary(t))
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Process.Kill()
	line, err := bufio.NewReader(out).ReadString('\n')
	if err != nil {
		t.Fatalf("parent said nothing: %v", err)
	}
	pid, err := strconv.Atoi(line[:len(line)-1])
	if err != nil {
		t.Fatalf("parent said %q", line)
	}
	if !processExists(pid) {
		t.Fatal("helper not running")
	}
	cmd.Process.Kill() // SIGKILL: no chance to clean up
	cmd.Wait()
	waitFor(t, "helper to follow its parent", 5*time.Second, func() bool { return !processExists(pid) })
}

func TestUnavailable(t *testing.T) {
	ctx := context.Background()
	requireRuntime(t) // the cases that need a working runtime build on it
	helper := embedtest.HelperBinary(t)

	t.Run("no model", func(t *testing.T) {
		e, _ := New(Config{ModelDir: filepath.Join(t.TempDir(), "nope"), HelperPath: helper})
		defer e.Close()
		st := e.Status()
		if st.Available || st.Reason == "" {
			t.Fatalf("status %+v", st)
		}
		_, err := e.EmbedQueries(ctx, []string{"x"})
		if !errors.Is(err, semantic.ErrModelUnavailable) {
			t.Fatalf("want ErrModelUnavailable, got %v", err)
		}
		if err := e.Warm(ctx); !errors.Is(err, semantic.ErrModelUnavailable) {
			t.Fatalf("warm: %v", err)
		}
		t.Logf("reason: %s", st.Reason)
	})
	t.Run("no helper", func(t *testing.T) {
		model := embedtest.ModelDir(t)
		e, _ := New(Config{ModelDir: model, HelperPath: filepath.Join(t.TempDir(), "missing")})
		defer e.Close()
		if st := e.Status(); st.Available || st.Reason == "" {
			t.Fatalf("status %+v", st)
		}
		if _, err := e.EmbedQueries(ctx, []string{"x"}); !errors.Is(err, semantic.ErrModelUnavailable) {
			t.Fatalf("got %v", err)
		}
	})
	t.Run("no runtime library", func(t *testing.T) {
		model := embedtest.ModelDir(t)
		e, _ := New(Config{ModelDir: model, HelperPath: helper, ORTLibrary: filepath.Join(t.TempDir(), "libonnxruntime.so")})
		defer e.Close()
		st := e.Status()
		if st.Available || st.Reason == "" {
			t.Fatalf("status %+v", st)
		}
		if _, err := e.EmbedQueries(ctx, []string{"x"}); !errors.Is(err, semantic.ErrModelUnavailable) {
			t.Fatalf("got %v", err)
		}
	})
	t.Run("runtime library that is not onnxruntime", func(t *testing.T) {
		model := embedtest.ModelDir(t)
		lib := "/usr/lib/libc.so.6"
		if _, err := os.Stat(lib); err != nil {
			t.Skip("no libc.so.6")
		}
		e, _ := New(Config{ModelDir: model, HelperPath: helper, ORTLibrary: lib})
		defer e.Close()
		_, err := e.EmbedQueries(ctx, []string{"x"})
		if !errors.Is(err, semantic.ErrModelUnavailable) {
			t.Fatalf("got %v", err)
		}
		// Discovered by the helper, so Status reports it afterwards.
		if st := e.Status(); st.Available || st.Reason == "" {
			t.Fatalf("status after failure: %+v", st)
		}
		t.Logf("error: %v", err)
	})
	t.Run("model appears later", func(t *testing.T) {
		src := embedtest.ModelDir(t)
		dir := t.TempDir()
		e, _ := New(Config{ModelDir: dir, HelperPath: helper, ORTLibrary: embedtest.ORTLibrary()})
		defer e.Close()
		if _, err := e.EmbedQueries(ctx, []string{"x"}); !errors.Is(err, semantic.ErrModelUnavailable) {
			t.Fatalf("got %v", err)
		}
		// Install by symlinking the real model files.
		for _, f := range []string{"tokenizer.model", "config.json", "onnx"} {
			if err := os.Symlink(filepath.Join(src, f), filepath.Join(dir, f)); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := e.EmbedQueries(ctx, []string{"x"}); err != nil {
			t.Fatalf("after installing the model: %v", err)
		}
	})
}

func TestCancelledContext(t *testing.T) {
	c := newClient(t, nil)
	if _, err := c.EmbedQueries(context.Background(), []string{"warm up"}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.EmbedQueries(ctx, []string{"x"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	// Cancelled mid-call: the caller returns at once and the client stays usable.
	pid := helperPID(c)
	ctx, cancel = context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	texts := make([]string, 300)
	for i := range texts {
		texts[i] = fmt.Sprintf("a longer document number %d about wireless headphones, bluetooth pairing and battery life in the evening", i)
	}
	begin := time.Now()
	_, err := c.EmbedQueries(ctx, texts)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	if d := time.Since(begin); d > time.Second {
		t.Fatalf("cancelled call took %v", d)
	}
	if v, err := c.EmbedQueries(context.Background(), []string{"fener"}); err != nil || len(v) != 1 {
		t.Fatalf("after cancel: %v", err)
	}
	if helperPID(c) != pid {
		t.Fatal("a caller's timeout must not restart the helper (it would lose the loaded model)")
	}
}

// Requests a caller gave up on while they were still queued are skipped, so
// a launcher that cancels keystroke after keystroke does not pile up work.
func TestAbandonedQueuedWorkIsSkipped(t *testing.T) {
	c := newClient(t, nil)
	bulk := make([]string, 120)
	for i := range bulk {
		bulk[i] = fmt.Sprintf("a longer document number %d about wireless headphones, bluetooth pairing and battery life in the evening", i)
	}
	if _, err := c.EmbedQueries(context.Background(), []string{"warm up"}); err != nil {
		t.Fatal(err)
	}
	begin := time.Now()
	if _, err := c.EmbedQueries(context.Background(), bulk); err != nil {
		t.Fatal(err)
	}
	one := time.Since(begin)
	if one < 300*time.Millisecond {
		t.Skipf("a batch takes only %v here; too fast to observe queueing", one)
	}

	first := make(chan error, 1)
	start := time.Now()
	go func() { _, err := c.EmbedQueries(context.Background(), bulk); first <- err }()
	time.Sleep(one / 10)
	for i := 0; i < 3; i++ { // abandoned while queued behind the first batch
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		if _, err := c.EmbedQueries(ctx, bulk); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("abandoned call: %v", err)
		}
		cancel()
	}
	if _, err := c.EmbedQueries(context.Background(), []string{"son"}); err != nil {
		t.Fatal(err)
	}
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	total := time.Since(start)
	t.Logf("one batch %v; first batch + 3 abandoned + 1 short query finished in %v", one, total)
	if total > one+one/2 {
		t.Fatalf("abandoned requests were executed: %v for work worth about %v", total, one)
	}
}

func TestClose(t *testing.T) {
	c := newClient(t, nil)
	if _, err := c.EmbedQueries(context.Background(), []string{"x"}); err != nil {
		t.Fatal(err)
	}
	pid := helperPID(c)
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if processExists(pid) {
		t.Fatal("helper survived Close")
	}
	if _, err := c.EmbedQueries(context.Background(), []string{"x"}); err == nil {
		t.Fatal("calls after Close must fail")
	}
	if err := c.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestImageAndAudioThroughHelper(t *testing.T) {
	dir := embedtest.ModelDir(t, "vision_encoder", "audio_encoder")
	_ = dir
	c := newClient(t, func(cfg *Config) { cfg.MediaIdle = 600 * time.Millisecond })
	ctx := context.Background()

	iref := embedtest.LoadRef(t, "testdata/image_reference.dref")
	img := semantic.ImageInput{Pixels: iref.Float32("pixels"), Positions: iref.Int64("positions"), NumPatches: iref.Int("patches"), SoftTokens: iref.Int("soft_tokens")}
	iv, err := c.EmbedImages(ctx, []semantic.ImageInput{img})
	if err != nil {
		t.Fatal(err)
	}
	if cs := embedtest.Cosine(iv[0], iref.Float32("embedding")); cs < 0.999 {
		t.Fatalf("image cosine %.5f", cs)
	} else {
		t.Logf("image cosine to Python %.6f", cs)
	}
	if st := c.Status(); !hasModality(st.Loaded, semantic.ModalityImage) || !hasModality(st.Loaded, semantic.ModalityText) {
		t.Fatalf("loaded after image: %+v", st)
	}

	aref := embedtest.LoadRef(t, "testdata/audio_reference.dref")
	clip := semantic.AudioInput{Mel: aref.Float32("mel"), Mask: aref.Bool("mask"), Frames: aref.Int("frames")}
	av, err := c.EmbedAudio(ctx, []semantic.AudioInput{clip})
	if err != nil {
		t.Fatal(err)
	}
	if cs := embedtest.Cosine(av[0], aref.Float32("embedding")); cs < 0.999 {
		t.Fatalf("audio cosine %.5f", cs)
	} else {
		t.Logf("audio cosine to Python %.6f", cs)
	}

	// Media towers go away on their own while the text tower stays.
	waitFor(t, "media towers to unload", 6*time.Second, func() bool {
		st := c.Status()
		return !hasModality(st.Loaded, semantic.ModalityImage) && !hasModality(st.Loaded, semantic.ModalityAudio)
	})
	st := c.Status()
	if !st.Running || !hasModality(st.Loaded, semantic.ModalityText) {
		t.Fatalf("text tower should stay loaded: %+v", st)
	}
	// And come back on demand.
	if _, err := c.EmbedImages(ctx, []semantic.ImageInput{img}); err != nil {
		t.Fatalf("after media unload: %v", err)
	}
}

func TestBadMediaInput(t *testing.T) {
	c := newClient(t, nil)
	ctx := context.Background()
	if _, err := c.EmbedImages(ctx, []semantic.ImageInput{{NumPatches: 2, Pixels: make([]float32, 5)}}); err == nil {
		t.Fatal("malformed image accepted")
	}
	iref := embedtest.LoadRef(t, "testdata/image_reference.dref")
	bad := semantic.ImageInput{Pixels: iref.Float32("pixels"), Positions: iref.Int64("positions"), NumPatches: iref.Int("patches"), SoftTokens: iref.Int("soft_tokens") + 3}
	_, err := c.EmbedImages(ctx, []semantic.ImageInput{bad})
	if err == nil || errors.Is(err, semantic.ErrModelUnavailable) {
		t.Fatalf("wrong soft tokens: %v", err)
	}
	// A rejected request leaves the helper serving.
	if _, err := c.EmbedQueries(ctx, []string{"x"}); err != nil {
		t.Fatal(err)
	}
}

func hasModality(l []semantic.Modality, m semantic.Modality) bool {
	for _, x := range l {
		if x == m {
			return true
		}
	}
	return false
}
