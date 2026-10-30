// Package embed implements semantic.Embedder on top of the dsearch-embed
// helper process. The helper holds the ONNX Runtime sessions; this client
// starts it on first use, keeps one instance, restarts it after a crash and
// shuts it down when idle so the memory goes back to the desktop.
package embed

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/wire"
)

// Config selects model, runtime and helper.
type Config struct {
	ModelDir   string        // contains onnx/, tokenizer.model, config files
	ORTLibrary string        // path to libonnxruntime.so; "" = search standard paths
	HelperPath string        // dsearch-embed binary; "" = next to os.Executable()
	Threads    int           // 0 = min(4, NumCPU)
	IdleUnload time.Duration // helper exits after this idle time; 0 = 10 minutes, negative = never
	// MediaIdle is how long the image and audio towers stay loaded after their
	// last use; they are large and only needed while files are being indexed.
	// 0 = one minute (or IdleUnload when that is shorter), negative = until the
	// helper exits.
	MediaIdle time.Duration
}

const (
	defaultIdle  = 10 * time.Minute
	defaultMedia = time.Minute
	helloTimeout = 20 * time.Second
	// requestTimeout bounds a call that carries no deadline of its own, so a
	// wedged helper cannot block callers forever.
	requestTimeout = 10 * time.Minute
	// failureTTL is how long an "unavailable" or crash-loop verdict is cached
	// before the helper is started again.
	unavailableTTL = 10 * time.Second
	crashTTL       = 5 * time.Second
	stopGrace      = 3 * time.Second
	helperName     = "dsearch-embed"
)

// Query and document prefixes of the model (config_sentence_transformers.json,
// prompts "query" and "document").
const (
	queryPrefix = "task: search result | query: "
	docPrefix   = "title: "
	docNoTitle  = "none"
)

// New returns an Embedder. It never starts the helper and does not fail when
// the model or runtime is missing: calls then return ErrModelUnavailable and
// Status says why, so the model can be installed later without restarting.
func New(cfg Config) (semantic.Embedder, error) {
	if cfg.ModelDir == "" {
		return nil, errors.New("embed: ModelDir is required")
	}
	if cfg.Threads < 0 {
		return nil, errors.New("embed: negative Threads")
	}
	if cfg.Threads == 0 {
		cfg.Threads = min(4, runtime.NumCPU())
	}
	if cfg.IdleUnload == 0 {
		cfg.IdleUnload = defaultIdle
	}
	if cfg.MediaIdle == 0 {
		cfg.MediaIdle = defaultMedia
		if cfg.IdleUnload > 0 && cfg.IdleUnload < cfg.MediaIdle {
			cfg.MediaIdle = cfg.IdleUnload
		}
	}
	return &client{cfg: cfg, lastUse: map[semantic.Modality]time.Time{}, mediaTmr: map[semantic.Modality]*time.Timer{}}, nil
}

type client struct {
	cfg Config

	mu        sync.Mutex
	h         *helper
	starting  chan struct{} // non-nil while a helper is being started
	loaded    []semantic.Modality
	inflight  int
	idle      *time.Timer
	lastUse   map[semantic.Modality]time.Time
	mediaTmr  map[semantic.Modality]*time.Timer
	failUntil time.Time
	failErr   error
	closed    bool
}

// ---- availability ---------------------------------------------------------

// towerFiles are the files each tower needs, relative to ModelDir.
var towerFiles = map[semantic.Modality][]string{
	semantic.ModalityText:  {"onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data", "tokenizer.model", "config.json"},
	semantic.ModalityImage: {"onnx/vision_encoder_quantized.onnx", "onnx/vision_encoder_quantized.onnx_data"},
	semantic.ModalityAudio: {"onnx/audio_encoder_quantized.onnx", "onnx/audio_encoder_quantized.onnx_data"},
}

// precheck looks for what can be seen from this process without loading
// anything: helper binary, model files, an explicit runtime path.
func (c *client) precheck(mods ...semantic.Modality) error {
	if _, err := c.helperPath(); err != nil {
		return err
	}
	if c.cfg.ORTLibrary != "" {
		if _, err := os.Stat(c.cfg.ORTLibrary); err != nil {
			return fmt.Errorf("onnxruntime library %s: %w", c.cfg.ORTLibrary, errNotThere(err))
		}
	}
	for _, m := range append([]semantic.Modality{semantic.ModalityText}, mods...) {
		for _, f := range towerFiles[m] {
			if _, err := os.Stat(filepath.Join(c.cfg.ModelDir, f)); err != nil {
				return fmt.Errorf("%s model file %s is missing", m, f)
			}
		}
	}
	return nil
}

func errNotThere(err error) error {
	if errors.Is(err, os.ErrNotExist) {
		return errors.New("not found")
	}
	return err
}

func unavailable(err error) error {
	if err == nil || errors.Is(err, semantic.ErrModelUnavailable) {
		return err
	}
	return fmt.Errorf("%w: %v", semantic.ErrModelUnavailable, err)
}

// helperPath locates dsearch-embed.
func (c *client) helperPath() (string, error) {
	if c.cfg.HelperPath != "" {
		if _, err := os.Stat(c.cfg.HelperPath); err != nil {
			return "", fmt.Errorf("embedding helper %s: %w", c.cfg.HelperPath, errNotThere(err))
		}
		return c.cfg.HelperPath, nil
	}
	var cands []string
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		cands = append(cands, filepath.Join(dir, helperName), filepath.Join(dir, "..", "libexec", "dsearch", helperName))
	}
	cands = append(cands, "/usr/libexec/dsearch/"+helperName, "/usr/lib/dsearch/"+helperName)
	for _, p := range cands {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
	}
	if p, err := exec.LookPath(helperName); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("embedding helper %s not found next to %s, in libexec or on PATH", helperName, exeOrUnknown())
}

func exeOrUnknown() string {
	if exe, err := os.Executable(); err == nil {
		return exe
	}
	return "the executable"
}

// Status reports availability without starting anything.
func (c *client) Status() semantic.EmbedderStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	st := semantic.EmbedderStatus{Available: true, Loaded: append([]semantic.Modality{}, c.loaded...)}
	if c.h != nil && c.h.alive() {
		st.Running = true
	} else {
		st.Loaded = []semantic.Modality{}
	}
	if err := c.precheck(); err != nil {
		st.Available, st.Reason = false, err.Error()
		return st
	}
	if time.Now().Before(c.failUntil) && c.failErr != nil && errors.Is(c.failErr, semantic.ErrModelUnavailable) {
		st.Available, st.Reason = false, c.failErr.Error()
	}
	return st
}

// ---- helper lifecycle -----------------------------------------------------

func (c *client) args() []string {
	a := []string{"--model-dir", c.cfg.ModelDir, "--threads", strconv.Itoa(c.cfg.Threads)}
	if c.cfg.ORTLibrary != "" {
		a = append(a, "--ort-lib", c.cfg.ORTLibrary)
	}
	return a
}

// acquire returns a live helper, starting one if needed, and counts the call
// as in flight so the idle timer leaves it alone.
func (c *client) acquire(ctx context.Context, mods []semantic.Modality) (*helper, error) {
	for {
		c.mu.Lock()
		if c.closed {
			c.mu.Unlock()
			return nil, errors.New("embed: closed")
		}
		if c.h != nil && c.h.alive() {
			h := c.h
			c.touchLocked()
			c.mu.Unlock()
			return h, nil
		}
		if time.Now().Before(c.failUntil) && c.failErr != nil {
			err := c.failErr
			c.mu.Unlock()
			return nil, err
		}
		if c.starting != nil {
			wait := c.starting
			c.mu.Unlock()
			select {
			case <-wait:
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		if err := c.precheck(mods...); err != nil {
			c.mu.Unlock()
			return nil, unavailable(err)
		}
		c.starting = make(chan struct{})
		c.mu.Unlock()

		h, err := c.start()

		c.mu.Lock()
		close(c.starting)
		c.starting = nil
		if err != nil {
			if errors.Is(err, semantic.ErrModelUnavailable) {
				c.failErr, c.failUntil = err, time.Now().Add(unavailableTTL)
			} else {
				c.failErr, c.failUntil = err, time.Now().Add(crashTTL)
			}
			c.mu.Unlock()
			return nil, err
		}
		if c.closed {
			c.mu.Unlock()
			h.stop(stopGrace)
			return nil, errors.New("embed: closed")
		}
		c.h, c.failErr = h, nil
		c.touchLocked()
		c.mu.Unlock()
		return h, nil
	}
}

// start launches the helper and maps its hello to an error.
func (c *client) start() (*helper, error) {
	path, err := c.helperPath()
	if err != nil {
		return nil, unavailable(err)
	}
	h, resp, err := startHelper(path, c.args(), c.onExit)
	if err != nil {
		var gone *errGone
		if errors.As(err, &gone) {
			return nil, fmt.Errorf("embedding helper failed to start: %w", err)
		}
		return nil, err
	}
	if !resp.OK {
		h.stop(stopGrace)
		return nil, c.mapError(resp)
	}
	if resp.Proto != wire.Protocol {
		h.stop(stopGrace)
		return nil, unavailable(fmt.Errorf("helper %s speaks protocol %d, this build needs %d", path, resp.Proto, wire.Protocol))
	}
	return h, nil
}

// onExit runs when a helper process has ended for any reason.
func (c *client) onExit(h *helper) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.h == h {
		c.h = nil
		c.loaded = nil
	}
}

// touchLocked marks use: counts a call in flight and pushes the idle timer.
func (c *client) touchLocked() {
	c.inflight++
	if c.idle != nil {
		c.idle.Stop()
	}
}

// release ends an in-flight call and re-arms the idle timer.
func (c *client) release(used semantic.Modality) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.inflight--
	now := time.Now()
	if used == semantic.ModalityImage || used == semantic.ModalityAudio {
		c.lastUse[used] = now
		c.armMediaLocked(used, now)
	}
	c.armIdleLocked()
}

func (c *client) armIdleLocked() {
	if c.cfg.IdleUnload < 0 || c.inflight > 0 || c.h == nil {
		return
	}
	h := c.h
	if c.idle != nil {
		c.idle.Stop()
	}
	c.idle = time.AfterFunc(c.cfg.IdleUnload, func() { c.idleExpired(h) })
}

func (c *client) idleExpired(h *helper) {
	c.mu.Lock()
	if c.h != h || c.inflight > 0 || c.closed {
		c.mu.Unlock()
		return
	}
	c.h, c.loaded = nil, nil
	c.mu.Unlock()
	h.stop(stopGrace)
}

// armMediaLocked schedules unloading of an image or audio tower.
func (c *client) armMediaLocked(m semantic.Modality, stamp time.Time) {
	if c.cfg.MediaIdle < 0 || c.h == nil {
		return
	}
	h := c.h
	if t := c.mediaTmr[m]; t != nil {
		t.Stop()
	}
	c.mediaTmr[m] = time.AfterFunc(c.cfg.MediaIdle, func() { c.mediaExpired(h, m, stamp) })
}

func (c *client) mediaExpired(h *helper, m semantic.Modality, stamp time.Time) {
	c.mu.Lock()
	if c.h != h || c.closed || !c.lastUse[m].Equal(stamp) {
		c.mu.Unlock() // helper replaced, or the tower was used again since
		return
	}
	if c.inflight > 0 {
		c.armMediaLocked(m, stamp) // busy: look again later
		c.mu.Unlock()
		return
	}
	c.inflight++ // keep the helper alive while we talk to it
	c.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	resp, _, err := h.roundTrip(ctx, wire.Request{Op: wire.OpUnload, Towers: []string{string(m)}}, nil)
	cancel()
	c.mu.Lock()
	if err == nil && resp.OK && c.h == h {
		c.loaded = toModalities(resp.Loaded)
	}
	c.inflight--
	c.armIdleLocked()
	c.mu.Unlock()
}

// Close stops the helper. The Embedder is unusable afterwards.
func (c *client) Close() error {
	c.mu.Lock()
	c.closed = true
	h := c.h
	c.h, c.loaded = nil, nil
	if c.idle != nil {
		c.idle.Stop()
	}
	c.mu.Unlock()
	if h != nil {
		h.stop(stopGrace)
	}
	return nil
}

// ---- calls ----------------------------------------------------------------

// mapError turns a helper error reply into a Go error.
func (c *client) mapError(r wire.Response) error {
	switch r.Code {
	case wire.CodeUnavailable:
		return fmt.Errorf("%w: %s", semantic.ErrModelUnavailable, r.Error)
	case wire.CodeBadRequest:
		return fmt.Errorf("embed: invalid input: %s", r.Error)
	}
	return fmt.Errorf("embed: %s", r.Error)
}

// call runs one request, retrying once on a fresh helper when the helper
// died under it. used is the tower whose idle clock the call resets.
func (c *client) call(ctx context.Context, req wire.Request, blob []byte, used semantic.Modality, mods ...semantic.Modality) (wire.Response, []byte, error) {
	// Our own deadline for callers that set none; only this one is taken as
	// a sign of a wedged helper. A caller's own timeout just abandons the
	// request (the helper skips it if still queued).
	ownTimeout := false
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, requestTimeout)
		defer cancel()
		ownTimeout = true
	}
	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		if err := ctx.Err(); err != nil {
			return wire.Response{}, nil, err
		}
		h, err := c.acquire(ctx, mods)
		if err != nil {
			return wire.Response{}, nil, err
		}
		resp, rblob, err := h.roundTrip(ctx, req, blob)
		c.release(used)
		var gone *errGone
		switch {
		case err == nil && resp.OK:
			c.mu.Lock()
			if c.h == h && resp.Loaded != nil {
				c.loaded = toModalities(resp.Loaded)
			}
			c.mu.Unlock()
			return resp, rblob, nil
		case err == nil:
			return wire.Response{}, nil, c.mapError(resp)
		case errors.As(err, &gone):
			lastErr = err
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return wire.Response{}, nil, err
			}
			continue
		case ownTimeout && errors.Is(err, context.DeadlineExceeded):
			// No answer in minutes: the helper is wedged; do not leave it holding the queue.
			h.kill()
			return wire.Response{}, nil, err
		default:
			return wire.Response{}, nil, err
		}
	}
	// Two helpers in a row died on this request: do not hammer.
	c.mu.Lock()
	c.failErr, c.failUntil = lastErr, time.Now().Add(crashTTL)
	c.mu.Unlock()
	return wire.Response{}, nil, lastErr
}

func toModalities(names []string) []semantic.Modality {
	out := make([]semantic.Modality, 0, len(names))
	for _, n := range names {
		out = append(out, semantic.Modality(n))
	}
	return out
}

// decode splits a vector reply into rows.
func decode(resp wire.Response, blob []byte, want int) ([][]float32, error) {
	if resp.N != want || resp.Dim != semantic.NativeDim {
		return nil, fmt.Errorf("embed: helper returned %d vectors of %d values, expected %d of %d", resp.N, resp.Dim, want, semantic.NativeDim)
	}
	r := wire.NewReader(blob)
	out := make([][]float32, want)
	for i := range out {
		out[i] = r.Float32(resp.Dim)
	}
	if err := r.Done(); err != nil {
		return nil, err
	}
	return out, nil
}

func (c *client) embedText(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	const chunk = 1024 // keeps one request well under the helper's per-request limit
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += chunk {
		part := texts[start:min(start+chunk, len(texts))]
		resp, blob, err := c.call(ctx, wire.Request{Op: wire.OpText, Texts: part}, nil, semantic.ModalityText)
		if err != nil {
			return nil, err
		}
		vecs, err := decode(resp, blob, len(part))
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (c *client) EmbedQueries(ctx context.Context, texts []string) ([][]float32, error) {
	prefixed := make([]string, len(texts))
	for i, t := range texts {
		prefixed[i] = queryPrefix + t
	}
	return c.embedText(ctx, prefixed)
}

func (c *client) EmbedDocuments(ctx context.Context, docs []semantic.TextDoc) ([][]float32, error) {
	prefixed := make([]string, len(docs))
	for i, d := range docs {
		title := d.Title
		if title == "" {
			title = docNoTitle
		}
		prefixed[i] = docPrefix + title + " | text: " + d.Text
	}
	return c.embedText(ctx, prefixed)
}

// Batches of media are split so one request stays a few tens of megabytes.
const (
	imageChunk = 8
	audioChunk = 8
)

func (c *client) EmbedImages(ctx context.Context, imgs []semantic.ImageInput) ([][]float32, error) {
	out := make([][]float32, 0, len(imgs))
	for start := 0; start < len(imgs); start += imageChunk {
		part := imgs[start:min(start+imageChunk, len(imgs))]
		req := wire.Request{Op: wire.OpImage, Images: make([]wire.ImageMeta, len(part))}
		var blob []byte
		for i, im := range part {
			if im.NumPatches <= 0 || len(im.Pixels) != im.NumPatches*768 || len(im.Positions) != im.NumPatches*2 || im.SoftTokens <= 0 {
				return nil, fmt.Errorf("embed: image %d is malformed (%d patches, %d pixels, %d positions, %d soft tokens)",
					start+i, im.NumPatches, len(im.Pixels), len(im.Positions), im.SoftTokens)
			}
			req.Images[i] = wire.ImageMeta{Patches: im.NumPatches, SoftTokens: im.SoftTokens}
			blob = wire.AppendFloat32(blob, im.Pixels)
			blob = wire.AppendInt64(blob, im.Positions)
		}
		resp, rb, err := c.call(ctx, req, blob, semantic.ModalityImage, semantic.ModalityImage)
		if err != nil {
			return nil, err
		}
		vecs, err := decode(resp, rb, len(part))
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (c *client) EmbedAudio(ctx context.Context, clips []semantic.AudioInput) ([][]float32, error) {
	out := make([][]float32, 0, len(clips))
	for start := 0; start < len(clips); start += audioChunk {
		part := clips[start:min(start+audioChunk, len(clips))]
		req := wire.Request{Op: wire.OpAudio, Clips: make([]wire.AudioMeta, len(part))}
		var blob []byte
		for i, cl := range part {
			if cl.Frames <= 0 || len(cl.Mel) != cl.Frames*128 || len(cl.Mask) != cl.Frames {
				return nil, fmt.Errorf("embed: clip %d is malformed (%d frames, %d mel values, %d mask entries)", start+i, cl.Frames, len(cl.Mel), len(cl.Mask))
			}
			req.Clips[i] = wire.AudioMeta{Frames: cl.Frames}
			blob = wire.AppendFloat32(blob, cl.Mel)
			blob = wire.AppendBool(blob, cl.Mask)
		}
		resp, rb, err := c.call(ctx, req, blob, semantic.ModalityAudio, semantic.ModalityAudio)
		if err != nil {
			return nil, err
		}
		vecs, err := decode(resp, rb, len(part))
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

// Warm loads the towers (the text tower when none are named) and resets the
// idle timers.
func (c *client) Warm(ctx context.Context, mods ...semantic.Modality) error {
	if len(mods) == 0 {
		mods = []semantic.Modality{semantic.ModalityText}
	}
	names := make([]string, len(mods))
	used := semantic.ModalityText
	for i, m := range mods {
		names[i] = string(m)
		if m == semantic.ModalityImage || m == semantic.ModalityAudio {
			used = m
		}
	}
	_, _, err := c.call(ctx, wire.Request{Op: wire.OpLoad, Towers: names}, nil, used, mods...)
	return err
}
