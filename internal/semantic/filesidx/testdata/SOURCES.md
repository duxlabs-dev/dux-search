# Test fixtures and their sources

Everything here is real data under an open licence or public domain, used as
test input. `vectors.json` holds embeddings that the real EmbeddingGemma 2 int8
ONNX model computed for these inputs (see `gen_vectors.py`).

| File | Source | Licence |
|---|---|---|
| `files/gencligehitabe.txt`, `hitabe_windows1254.txt`, `hitabe_utf16.txt` | Gençliğe Hitabe, Mustafa Kemal Atatürk, 1927 (Nutuk); the two other files are the same text re-encoded (Windows-1254, UTF-16 LE with BOM) | public domain |
| `files/pride_and_prejudice.txt`, `variants/novel.txt/v1..v3` | Jane Austen, Pride and Prejudice (1813), chapters 1-3, Project Gutenberg eBook #1342 (https://www.gutenberg.org/ebooks/1342), CRLF line endings kept | public domain |
| `files/sort.go`, `files/builder.go` | Go standard library (`sort/sort.go`, `strings/builder.go`), Go 1.27.3 | BSD-3-Clause, Copyright The Go Authors (header in each file) |
| `media/earth.jpg` | "The Earth seen from Apollo 17", NASA (Wikimedia Commons, 500 px thumbnail) | public domain |
| `media/bootprint.jpg` | "Apollo 11 bootprint", NASA (Wikimedia Commons, 500 px thumbnail) | public domain |
| `media/cat.jpg` | "Tabby cat with blue eyes", Pixabay (Wikimedia Commons, 500 px thumbnail) | CC0 |
| `media/error_screenshot.png` | rendered from the real stderr of `python3 -c "import nonexistent_module"` and `cat /etc/shadow` on the author's machine | CC0 |
| `media/small_step.ogg` | "Armstrong Small Step", NASA (Wikimedia Commons), Vorbis, 11.025 kHz | public domain |
| `media/chopin_nocturne_8s.flac` | Chopin, Nocturne No. 2 in E-flat major, Op. 9 No. 2, performed by Frank Levy (Wikimedia Commons), first 8 s, 16 kHz mono | public domain |
