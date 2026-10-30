package main

import (
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/engine"
	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/wire"
)

// Sanity bounds on request sizes; the model's own limits are enforced by the
// engine, these keep a malformed request from allocating gigabytes.
const (
	maxTexts     = 4096
	maxPatches   = 20000 // 2*max soft tokens (1120) budget with headroom
	maxSoft      = 2048
	maxMelFrames = 4 * engine.MaxTokens
	patchDim     = 768
	melBins      = 128
)

func parseTowers(names []string) ([]semantic.Modality, error) {
	if len(names) == 0 {
		return nil, engine.Invalid("no towers given")
	}
	out := make([]semantic.Modality, len(names))
	for i, n := range names {
		switch m := semantic.Modality(n); m {
		case semantic.ModalityText, semantic.ModalityImage, semantic.ModalityAudio:
			out[i] = m
		default:
			return nil, engine.Invalid("unknown tower %q", n)
		}
	}
	return out, nil
}

func towerNames(e *engine.Engine) []string {
	var out []string
	for _, m := range e.Loaded() {
		out = append(out, string(m))
	}
	return out
}

func embed(eng *engine.Engine, j job) ([][]float32, error) {
	switch j.req.Op {
	case wire.OpText:
		if len(j.req.Texts) > maxTexts {
			return nil, engine.Invalid("%d texts, at most %d per request", len(j.req.Texts), maxTexts)
		}
		return eng.EmbedTexts(j.req.Texts)
	case wire.OpImage:
		imgs, err := decodeImages(j)
		if err != nil {
			return nil, err
		}
		return eng.EmbedImages(imgs)
	default:
		clips, err := decodeClips(j)
		if err != nil {
			return nil, err
		}
		return eng.EmbedAudio(clips)
	}
}

func decodeImages(j job) ([]semantic.ImageInput, error) {
	r := wire.NewReader(j.blob)
	imgs := make([]semantic.ImageInput, len(j.req.Images))
	for i, m := range j.req.Images {
		if m.Patches <= 0 || m.Patches > maxPatches || m.SoftTokens <= 0 || m.SoftTokens > maxSoft {
			return nil, engine.Invalid("image %d has %d patches and %d soft tokens", i, m.Patches, m.SoftTokens)
		}
		imgs[i] = semantic.ImageInput{
			Pixels:     r.Float32(m.Patches * patchDim),
			Positions:  r.Int64(m.Patches * 2),
			NumPatches: m.Patches,
			SoftTokens: m.SoftTokens,
		}
	}
	if err := r.Done(); err != nil {
		return nil, engine.Invalid("%v", err)
	}
	return imgs, nil
}

func decodeClips(j job) ([]semantic.AudioInput, error) {
	r := wire.NewReader(j.blob)
	clips := make([]semantic.AudioInput, len(j.req.Clips))
	for i, m := range j.req.Clips {
		if m.Frames <= 0 || m.Frames > maxMelFrames {
			return nil, engine.Invalid("clip %d has %d frames", i, m.Frames)
		}
		clips[i] = semantic.AudioInput{
			Mel:    r.Float32(m.Frames * melBins),
			Mask:   r.Bool(m.Frames),
			Frames: m.Frames,
		}
	}
	if err := r.Done(); err != nil {
		return nil, engine.Invalid("%v", err)
	}
	return clips, nil
}
