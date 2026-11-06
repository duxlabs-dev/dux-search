package model

import (
	"fmt"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

// ModelID identifies the pinned model build. The vector store keys its
// collections by it, so vectors from another build are never mixed in.
const ModelID = "embeddinggemma-2-onnx-int8@daa72c5"

// Pinned sources. Every revision is an immutable commit, so a resumed
// download can never splice two different builds together.
const (
	onnxRepo          = "onnx-community/embeddinggemma-2-ONNX"
	onnxRevision      = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
	tokenizerRepo     = "google/embeddinggemma-2"
	tokenizerRevision = "f9972ac22e696a105c482202646979e33e4b5a1e"
)

// File is one pinned file of the model.
type File struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	Path     string `json:"path"` // slash separated, also the path below the model dir
	Size     int64  `json:"size"`
	SHA256   string `json:"sha256"`
}

// URL is the immutable download address below base.
func (f File) URL(base string) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", base, f.Repo, f.Revision, f.Path)
}

func onnx(path string, size int64, sum string) File {
	return File{Repo: onnxRepo, Revision: onnxRevision, Path: path, Size: size, SHA256: sum}
}

// The sha256 values come from the Hugging Face tree API (LFS oid) for LFS
// files and from the content of the pinned commit for the small plain files.
// TestManifestMatchesHub re-derives every row from the hub; regenerate the
// table from that test's output when the pinned revisions are bumped.
var (
	commonFiles = []File{
		onnx("tokenizer.json", 32170510, "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
		onnx("tokenizer_config.json", 1599, "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874"),
		onnx("config.json", 5031, "8d011bfe08b5e345bbe0b81e5c6fd02c381920b345b986047bc2a33ce7b90d1d"),
		onnx("config_sentence_transformers.json", 1565, "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5"),
		onnx("processor_config.json", 1788, "168f6a08522f3ce5dea596d94d003af2fd691742d4f41fe1f9d8cce76bfbf69c"),
		onnx("preprocessor_config.json", 560, "9344893f8d0573a46ebb2ca54c03f56d044cea194c8dcfb1f4d652240ac21daa"),
		onnx("chat_template.jinja", 1016, "4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e"),
		{
			Repo: tokenizerRepo, Revision: tokenizerRevision, Path: "tokenizer.model",
			Size: 4689013, SHA256: "e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03",
		},
	}

	towerFiles = map[semantic.Modality][]File{
		semantic.ModalityText: {
			onnx("onnx/model_quantized.onnx", 495165, "d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953"),
			onnx("onnx/model_quantized.onnx_data", 313724928, "278a7ff1248c3618e4bd11a607fc54f7bdc7778854230f3956d3f86bd9db4f3b"),
		},
		semantic.ModalityImage: {
			onnx("onnx/vision_encoder_quantized.onnx", 162495, "bb0de2df53a2448a32dc7908a187c168c8afd514d4d6f674f7f46024875fa4e3"),
			onnx("onnx/vision_encoder_quantized.onnx_data", 195228672, "3dabd69c0a36e9a8771ad82030dde74daa5a0e02b7047a5d3f3382b1137bab89"),
		},
		semantic.ModalityAudio: {
			onnx("onnx/audio_encoder_quantized.onnx", 249330, "04a9a9094ba76fb169e4c69be45a4b621580188654d6be59eab860eadcd42d3c"),
			onnx("onnx/audio_encoder_quantized.onnx_data", 340058624, "aa6361d898e1f1d6303f8dd2b5fabf4fd3629e15fd709e9cb06ac5cb9416a030"),
		},
	}

	allTowers = []semantic.Modality{semantic.ModalityText, semantic.ModalityImage, semantic.ModalityAudio}
)

// Files lists the pinned files a tower needs. The text tower includes the
// tokenizer and config files shared by every tower, because the image and the
// audio towers feed the text model and are useless without it.
func Files(tower semantic.Modality) []File {
	fs := append([]File(nil), towerFiles[tower]...)
	if tower == semantic.ModalityText {
		fs = append(append([]File(nil), commonFiles...), fs...)
	}
	return fs
}

// Towers lists every tower, text first.
func Towers() []semantic.Modality { return append([]semantic.Modality(nil), allTowers...) }

// TowerSize is the total size of a tower's files in bytes.
func TowerSize(tower semantic.Modality) int64 {
	var n int64
	for _, f := range Files(tower) {
		n += f.Size
	}
	return n
}
