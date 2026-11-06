// Package preprocess turns image and audio files into the tensors the
// EmbeddingGemma 2 vision and audio towers consume.
//
// The numerics follow the Hugging Face transformers reference
// (Gemma4ImageProcessor, Gemma4AudioFeatureExtractor, processor_config.json of
// onnx-community/embeddinggemma-2-ONNX) step by step; the package is pure Go
// (no cgo). The acceptance tests compare the output with transformers on real
// files, see testdata/gen_reference.py and testdata/README.md for the
// reference versions, the measured differences and the tolerances.
//
// Images: jpeg, png, gif (first frame), webp (still and first animation
// frame) and bmp. Audio: wav, flac, mp3 (internal/mp3dec) and ogg/vorbis; every other container
// yields ErrUnsupported.
package preprocess
