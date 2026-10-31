// Package engine runs the EmbeddingGemma 2 towers through ONNX Runtime. It is
// the part of the embedder that lives inside the dsearch-embed helper process.
package engine

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	"github.com/duxlabs-dev/dux-search/internal/ort"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/tokenize"
)

// Limits of the model: its context window and the largest padded batch one
// text run is allowed to hold (keeps peak memory bounded for long documents).
const (
	MaxTokens      = 8192
	maxBatch       = 16
	maxBatchTokens = 2 * MaxTokens
	featureDim     = 512 // width of image_features / audio_features
	patchDim       = 768 // 16*16*3 values per image patch
	melBins        = 128
)

// ErrUnavailable marks missing or unusable model files and runtime library.
var ErrUnavailable = errors.New("model unavailable")

// ErrInvalidInput marks an input the model cannot take (wrong sizes, too long).
var ErrInvalidInput = errors.New("invalid input")

// kindError carries only the detail in its message and matches a sentinel
// with errors.Is, so messages are not prefixed with the sentinel's text.
type kindError struct {
	kind error
	msg  string
}

func (e *kindError) Error() string        { return e.msg }
func (e *kindError) Is(target error) bool { return target == e.kind }

func unavailable(format string, a ...any) error {
	return &kindError{ErrUnavailable, fmt.Sprintf(format, a...)}
}

// Invalid builds an error that matches ErrInvalidInput.
func Invalid(format string, a ...any) error {
	return &kindError{ErrInvalidInput, fmt.Sprintf(format, a...)}
}

// Devices the towers run on.
const (
	DeviceAuto = "auto" // the GPU when the runtime and the driver can use it, else the CPU
	DeviceCPU  = "cpu"
	DeviceCUDA = "cuda" // an NVIDIA GPU through the CUDA execution provider; no fallback
)

// Config selects the model and runtime.
type Config struct {
	ModelDir   string // contains onnx/, tokenizer.model, config.json
	ORTLibrary string // "" = standard locations
	Threads    int    // intra-op threads; 0 = min(4, NumCPU)
	Device     string // DeviceAuto, DeviceCPU or DeviceCUDA; "" = DeviceAuto
	// GPUMemLimit caps the video memory of each tower's arena in bytes; 0 =
	// no limit. A run that needs more fails (and is repeated on the CPU with
	// DeviceAuto).
	GPUMemLimit uint64
}

// tower file names (int8 export).
var towerFile = map[semantic.Modality]string{
	semantic.ModalityText:  "model_quantized.onnx",
	semantic.ModalityImage: "vision_encoder_quantized.onnx",
	semantic.ModalityAudio: "audio_encoder_quantized.onnx",
}

// Engine owns the runtime, the tokenizer and the loaded tower sessions. All
// methods are safe for concurrent use; runs are serialized.
type Engine struct {
	cfg Config
	rt  *ort.Runtime

	mu       sync.Mutex
	tk       *tokenize.Tokenizer
	tokens   specialTokens
	dim      int
	sessions map[semantic.Modality]*ort.Session
	onGPU    map[semantic.Modality]bool
	gpuErr   error // why DeviceAuto fell back to the CPU; the GPU is not tried again
	// cpuSpare holds CPU sessions of GPU towers, opened when a run failed on
	// the GPU (DeviceAuto only).
	cpuSpare map[semantic.Modality]*ort.Session
}

// specialTokens are the ids that wrap image and audio placeholders, read from
// config.json (see processor expansion in transformers' EmbeddingGemma2Processor:
// boi + image*N + eoi, boa + audio*N + eoa).
type specialTokens struct {
	boi, image, eoi, boa, audio, eoa int32
}

// New loads the ONNX Runtime library. Model files are read on first use.
func New(cfg Config) (*Engine, error) {
	if cfg.Threads <= 0 {
		cfg.Threads = min(4, runtime.NumCPU())
	}
	switch cfg.Device {
	case "":
		cfg.Device = DeviceAuto
	case DeviceAuto, DeviceCPU, DeviceCUDA:
	default:
		return nil, Invalid("unknown device %q (want %s, %s or %s)", cfg.Device, DeviceAuto, DeviceCPU, DeviceCUDA)
	}
	if _, err := os.Stat(filepath.Join(cfg.ModelDir, "config.json")); err != nil {
		return nil, unavailable("model directory %s: %v", cfg.ModelDir, err)
	}
	rt, err := ort.Load(cfg.ORTLibrary)
	if err != nil {
		return nil, unavailable("%v", err)
	}
	return &Engine{cfg: cfg, rt: rt, sessions: map[semantic.Modality]*ort.Session{}, onGPU: map[semantic.Modality]bool{}, cpuSpare: map[semantic.Modality]*ort.Session{}}, nil
}

// RuntimeVersion returns the ONNX Runtime version in use.
func (e *Engine) RuntimeVersion() string { return e.rt.Version() }

// Device names where the loaded towers run: DeviceCUDA when any of them is on
// the GPU, DeviceCPU otherwise, "" when none is loaded.
func (e *Engine) Device() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.sessions) == 0 {
		return ""
	}
	for m := range e.sessions {
		if e.onGPU[m] {
			return DeviceCUDA
		}
	}
	return DeviceCPU
}

// wantGPU says whether the next tower is put on the GPU.
func (e *Engine) wantGPU() bool {
	switch e.cfg.Device {
	case DeviceCUDA:
		return true
	case DeviceCPU:
		return false
	}
	return e.gpuErr == nil && e.rt.HasProvider(ort.ProviderCUDA)
}

// Close releases all sessions and the runtime environment.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	for m, s := range e.sessions {
		s.Close()
		delete(e.sessions, m)
	}
	for m, s := range e.cpuSpare {
		s.Close()
		delete(e.cpuSpare, m)
	}
	e.rt.Close()
}

// Loaded lists the towers currently in memory.
func (e *Engine) Loaded() []semantic.Modality {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.loadedLocked()
}

func (e *Engine) loadedLocked() []semantic.Modality {
	var out []semantic.Modality
	for _, m := range []semantic.Modality{semantic.ModalityText, semantic.ModalityImage, semantic.ModalityAudio} {
		if e.sessions[m] != nil {
			out = append(out, m)
		}
	}
	return out
}

// Load brings towers into memory. Image and audio towers need the text tower
// to produce embeddings, but are loaded independently.
func (e *Engine) Load(mods ...semantic.Modality) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, m := range mods {
		if err := e.loadLocked(m); err != nil {
			return err
		}
	}
	return nil
}

// Unload drops towers and returns their memory to the system as far as the
// allocator allows.
func (e *Engine) Unload(mods ...semantic.Modality) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, m := range mods {
		if s := e.sessions[m]; s != nil {
			s.Close()
			delete(e.sessions, m)
			delete(e.onGPU, m)
		}
		if s := e.cpuSpare[m]; s != nil {
			s.Close()
			delete(e.cpuSpare, m)
		}
	}
}

var wantInputs = map[semantic.Modality][]string{
	semantic.ModalityText:  {"input_ids", "attention_mask", "image_features", "video_features", "audio_features"},
	semantic.ModalityImage: {"pixel_values", "pixel_position_ids"},
	semantic.ModalityAudio: {"input_features", "input_features_mask"},
}

var wantOutput = map[semantic.Modality]string{
	semantic.ModalityText:  "sentence_embedding",
	semantic.ModalityImage: "image_features",
	semantic.ModalityAudio: "audio_features",
}

func (e *Engine) loadLocked(m semantic.Modality) error {
	if e.sessions[m] != nil {
		return nil
	}
	file, ok := towerFile[m]
	if !ok {
		return Invalid("unknown modality %q", m)
	}
	if err := e.initTextSideLocked(); err != nil {
		return err
	}
	gpu := e.wantGPU()
	s, err := e.openLocked(m, file, gpu)
	if err != nil && gpu && e.cfg.Device == DeviceAuto {
		// A runtime built with CUDA on a machine whose GPU or driver it
		// cannot use: run on the CPU instead.
		e.gpuErr = err
		fmt.Fprintf(os.Stderr, "dsearch-embed: GPU not usable, running on the CPU: %v\n", err)
		gpu = false
		s, err = e.openLocked(m, file, false)
	}
	if err != nil {
		return err
	}
	e.sessions[m] = s
	e.onGPU[m] = gpu
	return nil
}

// openLocked creates a session of one tower on the GPU or the CPU.
func (e *Engine) openLocked(m semantic.Modality, file string, gpu bool) (*ort.Session, error) {
	path := filepath.Join(e.cfg.ModelDir, "onnx", file)
	if _, err := os.Stat(path); err != nil {
		return nil, unavailable("%s tower: %v", m, err)
	}
	opts := ort.Options{
		IntraOpThreads: e.cfg.Threads,
		// Intermediate buffers are freed after each run: the helper is idle
		// most of the time and its resident size matters more than the last
		// few percent of speed.
		DisableCPUMemArena: true,
	}
	if gpu {
		opts.CUDA = &ort.CUDAOptions{MemLimit: e.cfg.GPUMemLimit}
	}
	s, err := e.rt.NewSession(path, opts)
	if err != nil {
		return nil, unavailable("%s tower: %v", m, err)
	}
	if got, want := s.InputNames(), wantInputs[m]; !equalStrings(got, want) {
		s.Close()
		return nil, unavailable("%s tower has inputs %v, expected %v", m, got, want)
	}
	if out := s.OutputNames(); !contains(out, wantOutput[m]) {
		s.Close()
		return nil, unavailable("%s tower has outputs %v, missing %s", m, out, wantOutput[m])
	}
	return s, nil
}

// runLocked runs a loaded tower. With DeviceAuto a run that fails on the GPU,
// for example because another program holds its memory, is repeated on the
// CPU, so indexing slows down instead of failing.
func (e *Engine) runLocked(m semantic.Modality, inputs []ort.Input, outputs []string) ([]*ort.Tensor, error) {
	res, err := e.sessions[m].Run(inputs, outputs)
	if err == nil || !e.onGPU[m] || e.cfg.Device != DeviceAuto {
		return res, err
	}
	cpu := e.cpuSpare[m]
	if cpu == nil {
		var cerr error
		if cpu, cerr = e.openLocked(m, towerFile[m], false); cerr != nil {
			return nil, errors.Join(err, cerr)
		}
		e.cpuSpare[m] = cpu
	}
	fmt.Fprintf(os.Stderr, "dsearch-embed: %s run failed on the GPU, repeated on the CPU: %v\n", m, err)
	return cpu.Run(inputs, outputs)
}

// initTextSideLocked reads the tokenizer and config.json once.
func (e *Engine) initTextSideLocked() error {
	if e.tk != nil {
		return nil
	}
	raw, err := os.ReadFile(filepath.Join(e.cfg.ModelDir, "config.json"))
	if err != nil {
		return unavailable("%v", err)
	}
	var c struct {
		BOI   int32 `json:"boi_token_id"`
		Image int32 `json:"image_token_id"`
		EOI   int32 `json:"eoi_token_id"`
		BOA   int32 `json:"boa_token_id"`
		Audio int32 `json:"audio_token_id"`
		EOA   int32 `json:"eoa_token_index"`
		Text  struct {
			Dim int `json:"embedding_dim"`
			BOS int `json:"bos_token_id"`
			EOS int `json:"eos_token_id"`
			Pad int `json:"pad_token_id"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return unavailable("config.json: %v", err)
	}
	if c.Text.Dim != semantic.NativeDim {
		return unavailable("config.json embedding_dim %d, expected %d", c.Text.Dim, semantic.NativeDim)
	}
	if c.Text.BOS != tokenize.BOSID || c.Text.EOS != tokenize.EOSID || c.Text.Pad != tokenize.PadID {
		return unavailable("config.json bos/eos/pad %d/%d/%d differ from the tokenizer's", c.Text.BOS, c.Text.EOS, c.Text.Pad)
	}
	st := specialTokens{boi: c.BOI, image: c.Image, eoi: c.EOI, boa: c.BOA, audio: c.Audio, eoa: c.EOA}
	tk, err := tokenize.Load(filepath.Join(e.cfg.ModelDir, "tokenizer.model"))
	if err != nil {
		return unavailable("%v", err)
	}
	// The ids in config.json must be the tokenizer's literal markers, so the
	// two files cannot drift apart unnoticed.
	for _, p := range []struct {
		text string
		id   int32
	}{{"<|image>", st.boi}, {"<|image|>", st.image}, {"<image|>", st.eoi}, {"<|audio>", st.boa}, {"<|audio|>", st.audio}, {"<audio|>", st.eoa}} {
		if got := tk.EncodeRaw(p.text); len(got) != 1 || got[0] != p.id {
			return unavailable("config.json id %d for %s does not match the tokenizer (%v)", p.id, p.text, got)
		}
	}
	e.tk, e.tokens, e.dim = tk, st, c.Text.Dim
	return nil
}

// EmbedTexts embeds already prefixed texts and returns one L2-normalized
// vector per text, in order.
func (e *Engine) EmbedTexts(texts []string) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.loadLocked(semantic.ModalityText); err != nil {
		return nil, err
	}
	seqs := make([][]int32, len(texts))
	for i, t := range texts {
		ids := e.tk.Encode(t)
		if len(ids) > MaxTokens {
			// Keep the closing <eos>: the model was trained with it.
			ids = append(ids[:MaxTokens-1], tokenize.EOSID)
		}
		seqs[i] = ids
	}
	return e.runTextBatches(seqs, nil, nil)
}

// EmbedImages embeds images through the vision tower.
func (e *Engine) EmbedImages(imgs []semantic.ImageInput) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.loadLocked(semantic.ModalityText); err != nil {
		return nil, err
	}
	if err := e.loadLocked(semantic.ModalityImage); err != nil {
		return nil, err
	}
	seqs := make([][]int32, len(imgs))
	feats := make([][]float32, len(imgs))
	for i, im := range imgs {
		f, err := e.runVision(im)
		if err != nil {
			return nil, fmt.Errorf("image %d: %w", i, err)
		}
		feats[i] = f
		seqs[i] = e.mediaSequence(e.tokens.boi, e.tokens.image, e.tokens.eoi, im.SoftTokens)
		if len(seqs[i]) > MaxTokens {
			return nil, Invalid("image %d needs %d tokens, the context holds %d", i, len(seqs[i]), MaxTokens)
		}
	}
	return e.runTextBatches(seqs, feats, nil)
}

// EmbedAudio embeds audio clips through the audio tower.
func (e *Engine) EmbedAudio(clips []semantic.AudioInput) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.loadLocked(semantic.ModalityText); err != nil {
		return nil, err
	}
	if err := e.loadLocked(semantic.ModalityAudio); err != nil {
		return nil, err
	}
	seqs := make([][]int32, len(clips))
	feats := make([][]float32, len(clips))
	for i, c := range clips {
		f, rows, err := e.runAudio(c)
		if err != nil {
			return nil, fmt.Errorf("clip %d: %w", i, err)
		}
		feats[i] = f
		seqs[i] = e.mediaSequence(e.tokens.boa, e.tokens.audio, e.tokens.eoa, rows)
		if len(seqs[i]) > MaxTokens {
			return nil, Invalid("clip %d needs %d tokens, the context holds %d", i, len(seqs[i]), MaxTokens)
		}
	}
	return e.runTextBatches(seqs, nil, feats)
}

// mediaSequence is <bos> open placeholder*n close <eos>, the tokenization of
// the processor's expanded "<|image|>" / "<|audio|>" prompt.
func (e *Engine) mediaSequence(open, mid, closeTok int32, n int) []int32 {
	ids := make([]int32, 0, n+4)
	ids = append(ids, tokenize.BOSID, open)
	for i := 0; i < n; i++ {
		ids = append(ids, mid)
	}
	return append(ids, closeTok, tokenize.EOSID)
}

func (e *Engine) runVision(im semantic.ImageInput) ([]float32, error) {
	p := im.NumPatches
	if p <= 0 || len(im.Pixels) != p*patchDim || len(im.Positions) != p*2 {
		return nil, Invalid("image has %d patches, %d pixel values and %d positions", p, len(im.Pixels), len(im.Positions))
	}
	if im.SoftTokens <= 0 {
		return nil, Invalid("image has %d soft tokens", im.SoftTokens)
	}
	px, err := e.rt.NewFloat32([]int64{1, int64(p), patchDim}, im.Pixels)
	if err != nil {
		return nil, err
	}
	defer px.Close()
	pos, err := e.rt.NewInt64([]int64{1, int64(p), 2}, im.Positions)
	if err != nil {
		return nil, err
	}
	defer pos.Close()
	out, err := e.runLocked(semantic.ModalityImage,
		[]ort.Input{{Name: "pixel_values", Tensor: px}, {Name: "pixel_position_ids", Tensor: pos}},
		[]string{"image_features"})
	if err != nil {
		return nil, err
	}
	defer out[0].Close()
	f, shape, err := out[0].Float32()
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[1] != featureDim || int(shape[0]) != im.SoftTokens {
		return nil, Invalid("vision tower returned shape %v for %d soft tokens", shape, im.SoftTokens)
	}
	return f, nil
}

func (e *Engine) runAudio(c semantic.AudioInput) ([]float32, int, error) {
	f := c.Frames
	if f <= 0 || len(c.Mel) != f*melBins || len(c.Mask) != f {
		return nil, 0, Invalid("clip has %d frames, %d mel values and %d mask entries", f, len(c.Mel), len(c.Mask))
	}
	mel, err := e.rt.NewFloat32([]int64{1, int64(f), melBins}, c.Mel)
	if err != nil {
		return nil, 0, err
	}
	defer mel.Close()
	mask, err := e.rt.NewBool([]int64{1, int64(f)}, c.Mask)
	if err != nil {
		return nil, 0, err
	}
	defer mask.Close()
	out, err := e.runLocked(semantic.ModalityAudio,
		[]ort.Input{{Name: "input_features", Tensor: mel}, {Name: "input_features_mask", Tensor: mask}},
		[]string{"audio_features"})
	if err != nil {
		return nil, 0, err
	}
	defer out[0].Close()
	feat, shape, err := out[0].Float32()
	if err != nil {
		return nil, 0, err
	}
	if len(shape) != 2 || shape[1] != featureDim || shape[0] == 0 {
		return nil, 0, Invalid("audio tower returned shape %v", shape)
	}
	return feat, int(shape[0]), nil
}

// runTextBatches runs the text tower over sequences (image and audio feature
// rows, when given, belong to the sequence with the same index) in padded
// batches ordered by length, and returns the vectors in input order.
func (e *Engine) runTextBatches(seqs [][]int32, image, audio [][]float32) ([][]float32, error) {
	order := make([]int, len(seqs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(seqs[order[a]]) > len(seqs[order[b]]) })
	out := make([][]float32, len(seqs))
	for start := 0; start < len(order); {
		maxLen := len(seqs[order[start]])
		end := start + 1
		for end < len(order) && end-start < maxBatch && (end-start+1)*maxLen <= maxBatchTokens {
			end++
		}
		batch := order[start:end]
		vecs, err := e.runTextBatch(batch, seqs, image, audio, maxLen)
		if err != nil {
			return nil, err
		}
		for i, idx := range batch {
			out[idx] = vecs[i]
		}
		start = end
	}
	return out, nil
}

func (e *Engine) runTextBatch(batch []int, seqs [][]int32, image, audio [][]float32, maxLen int) ([][]float32, error) {
	b := len(batch)
	ids := make([]int64, b*maxLen) // zero is the pad id
	mask := make([]int64, b*maxLen)
	var imgRows, audRows []float32
	for row, idx := range batch {
		for j, id := range seqs[idx] {
			ids[row*maxLen+j] = int64(id)
			mask[row*maxLen+j] = 1
		}
		if image != nil {
			imgRows = append(imgRows, image[idx]...)
		}
		if audio != nil {
			audRows = append(audRows, audio[idx]...)
		}
	}
	shape := []int64{int64(b), int64(maxLen)}
	idT, err := e.rt.NewInt64(shape, ids)
	if err != nil {
		return nil, err
	}
	defer idT.Close()
	maskT, err := e.rt.NewInt64(shape, mask)
	if err != nil {
		return nil, err
	}
	defer maskT.Close()
	imgT, err := e.rt.NewFloat32([]int64{int64(len(imgRows) / featureDim), featureDim}, imgRows)
	if err != nil {
		return nil, err
	}
	defer imgT.Close()
	vidT, err := e.rt.NewFloat32([]int64{0, featureDim}, nil)
	if err != nil {
		return nil, err
	}
	defer vidT.Close()
	audT, err := e.rt.NewFloat32([]int64{int64(len(audRows) / featureDim), featureDim}, audRows)
	if err != nil {
		return nil, err
	}
	defer audT.Close()
	res, err := e.runLocked(semantic.ModalityText, []ort.Input{
		{Name: "input_ids", Tensor: idT}, {Name: "attention_mask", Tensor: maskT},
		{Name: "image_features", Tensor: imgT}, {Name: "video_features", Tensor: vidT}, {Name: "audio_features", Tensor: audT},
	}, []string{"sentence_embedding"})
	if err != nil {
		return nil, err
	}
	defer res[0].Close()
	flat, rs, err := res[0].Float32()
	if err != nil {
		return nil, err
	}
	if len(rs) != 2 || int(rs[0]) != b || int(rs[1]) != e.dim {
		return nil, fmt.Errorf("text tower returned shape %v for batch %d", rs, b)
	}
	vecs := make([][]float32, b)
	for i := range vecs {
		v := flat[i*e.dim : (i+1)*e.dim : (i+1)*e.dim]
		normalize(v)
		vecs[i] = v
	}
	return vecs, nil
}

// normalize scales v to unit length in place (the graph already normalizes;
// this removes the int8 rounding residue so callers can rely on |v| = 1).
func normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func contains(s []string, x string) bool {
	for _, v := range s {
		if v == x {
			return true
		}
	}
	return false
}
