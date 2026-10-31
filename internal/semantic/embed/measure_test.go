package embed

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/embedtest"
)

// procMB reads a /proc/<pid>/status field (VmRSS, VmHWM) in MB.
func procMB(pid int, field string) float64 {
	f, err := os.Open("/proc/" + strconv.Itoa(pid) + "/status")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if rest, ok := strings.CutPrefix(sc.Text(), field+":"); ok {
			kb, _ := strconv.ParseFloat(strings.Fields(rest)[0], 64)
			return kb / 1024
		}
	}
	return 0
}

func stats(d []time.Duration) string {
	sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
	return fmt.Sprintf("median %.1f ms, p95 %.1f ms, max %.1f ms (n=%d)",
		ms(d[len(d)/2]), ms(d[len(d)*95/100]), ms(d[len(d)-1]), len(d))
}

// trimWait is a little longer than the helper's quiet period before it returns memory.
const trimWait = 2500 * time.Millisecond

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// TestMeasure prints the latency and memory figures of the helper. It is a
// report, not an assertion: set DSEARCH_MEASURE=1 (and optionally
// DSEARCH_MEASURE_REFS to a directory holding a 280-token image reference and
// a long audio reference made by testdata/gen_reference.py).
func TestMeasure(t *testing.T) {
	if os.Getenv("DSEARCH_MEASURE") == "" {
		t.Skip("DSEARCH_MEASURE not set")
	}
	embedtest.ModelDir(t, "vision_encoder", "audio_encoder")
	refs := os.Getenv("DSEARCH_MEASURE_REFS")
	imgPath, audPath := "testdata/image_reference.dref", "testdata/audio_reference.dref"
	if refs != "" {
		imgPath, audPath = refs+"/image_reference.dref", refs+"/audio_reference.dref"
	}
	img := embedtest.LoadRef(t, imgPath)
	aud := embedtest.LoadRef(t, audPath)
	in := semantic.ImageInput{Pixels: img.Float32("pixels"), Positions: img.Int64("positions"), NumPatches: img.Int("patches"), SoftTokens: img.Int("soft_tokens")}
	clip := semantic.AudioInput{Mel: aud.Float32("mel"), Mask: aud.Bool("mask"), Frames: aud.Int("frames")}

	ctx := context.Background()
	c := newClient(t, func(cfg *Config) { cfg.IdleUnload = 15 * time.Second; cfg.MediaIdle = 8 * time.Second })
	self := os.Getpid()
	t.Logf("parent (test process) RSS before: %.0f MB", procMB(self, "VmRSS"))

	// Process start and handshake alone.
	var spawn []time.Duration
	for i := 0; i < 5; i++ {
		begin := time.Now()
		h, _, err := startHelper(c.cfg.HelperPath, c.args(), nil)
		if err != nil {
			t.Fatal(err)
		}
		spawn = append(spawn, time.Since(begin))
		h.stop(stopGrace)
	}
	t.Logf("helper spawn + hello (no model): %s", stats(spawn))

	begin := time.Now()
	if _, err := c.EmbedQueries(ctx, []string{"karanlık"}); err != nil {
		t.Fatal(err)
	}
	t.Logf("cold start (spawn + load text tower + first query): %.0f ms", ms(time.Since(begin)))
	pid := helperPID(c)
	t.Logf("helper RSS, text tower loaded, after first query: %.0f MB (peak %.0f MB)", procMB(pid, "VmRSS"), procMB(pid, "VmHWM"))

	words := []string{"karanlık", "gözlerim yoruldu", "kablosuz kulaklık", "fener", "sesi kıs", "parlaklığı yüzde kırk yap",
		"bildirimleri sustur", "pil tasarrufu", "ekranı kaydet", "on beş dakika zamanlayıcı kur", "oda ışığını aç", "internet kapat"}
	var single []time.Duration
	for i := 0; i < 100; i++ {
		q := fmt.Sprintf("%s %d", words[i%len(words)], i)
		begin := time.Now()
		if _, err := c.EmbedQueries(ctx, []string{q}); err != nil {
			t.Fatal(err)
		}
		single = append(single, time.Since(begin))
	}
	t.Logf("single query, warm: %s", stats(single))
	batch := make([]string, 16)
	var b16 []time.Duration
	for i := 0; i < 30; i++ {
		for j := range batch {
			batch[j] = fmt.Sprintf("%s %d", words[(i+j)%len(words)], i*16+j)
		}
		begin := time.Now()
		if _, err := c.EmbedQueries(ctx, batch); err != nil {
			t.Fatal(err)
		}
		b16 = append(b16, time.Since(begin))
	}
	t.Logf("batch of 16 queries, warm: %s", stats(b16))

	docs := make([]semantic.TextDoc, 16)
	for i := range docs {
		docs[i] = semantic.TextDoc{Title: "notes", Text: strings.Repeat(fmt.Sprintf("Section %d: Night Light warms the screen colors and reduces eye strain while reading at night. ", i), 14)}
	}
	var d16 []time.Duration
	for i := 0; i < 8; i++ {
		begin := time.Now()
		if _, err := c.EmbedDocuments(ctx, docs); err != nil {
			t.Fatal(err)
		}
		d16 = append(d16, time.Since(begin))
	}
	t.Logf("batch of 16 documents (~280 tokens each), warm: %s", stats(d16))
	t.Logf("helper RSS right after text work: %.0f MB (peak %.0f MB)", procMB(pid, "VmRSS"), procMB(pid, "VmHWM"))
	time.Sleep(trimWait)
	t.Logf("helper RSS 2.5 s after the last request (memory returned): %.0f MB", procMB(pid, "VmRSS"))

	begin = time.Now()
	if _, err := c.EmbedImages(ctx, []semantic.ImageInput{in}); err != nil {
		t.Fatal(err)
	}
	t.Logf("first image (%d soft tokens, loads the vision tower): %.0f ms", in.SoftTokens, ms(time.Since(begin)))
	time.Sleep(trimWait)
	t.Logf("helper RSS, text + vision loaded, settled: %.0f MB (peak %.0f MB)", procMB(pid, "VmRSS"), procMB(pid, "VmHWM"))
	var im []time.Duration
	for i := 0; i < 8; i++ {
		begin := time.Now()
		if _, err := c.EmbedImages(ctx, []semantic.ImageInput{in}); err != nil {
			t.Fatal(err)
		}
		im = append(im, time.Since(begin))
	}
	t.Logf("image, warm: %s", stats(im))

	begin = time.Now()
	if _, err := c.EmbedAudio(ctx, []semantic.AudioInput{clip}); err != nil {
		t.Fatal(err)
	}
	t.Logf("first audio clip (%d frames = %.1f s, loads the audio tower): %.0f ms", clip.Frames, float64(clip.Frames)*0.01, ms(time.Since(begin)))
	time.Sleep(trimWait)
	t.Logf("helper RSS, text + vision + audio loaded, settled: %.0f MB (peak %.0f MB)", procMB(pid, "VmRSS"), procMB(pid, "VmHWM"))
	var au []time.Duration
	for i := 0; i < 5; i++ {
		begin := time.Now()
		if _, err := c.EmbedAudio(ctx, []semantic.AudioInput{clip}); err != nil {
			t.Fatal(err)
		}
		au = append(au, time.Since(begin))
	}
	t.Logf("audio, warm: %s", stats(au))

	waitFor(t, "media towers to unload", 20*time.Second, func() bool {
		st := c.Status()
		return !hasModality(st.Loaded, semantic.ModalityImage) && !hasModality(st.Loaded, semantic.ModalityAudio)
	})
	time.Sleep(trimWait)
	t.Logf("helper RSS after the image and audio towers unloaded (text kept): %.0f MB", procMB(pid, "VmRSS"))

	waitFor(t, "idle exit", 20*time.Second, func() bool { return !processExists(pid) })
	t.Logf("helper after idle unload: process gone (RSS 0 MB); parent RSS %.0f MB", procMB(self, "VmRSS"))
}
