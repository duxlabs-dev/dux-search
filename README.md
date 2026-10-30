<div align="center">
  <a href="https://duxlabs.dev">
    <img src="assets/danklogo.svg" alt="dsearch" width="200">
  </a>

  # dsearch

  ### Fast, configurable filesystem search with fuzzy matching

[![Documentation](https://img.shields.io/badge/docs-duxlabs.dev-9ccbfb?style=for-the-badge&labelColor=101418)](https://duxlabs.dev/docs/dsearch)
[![GitHub stars](https://img.shields.io/github/stars/duxlabs-dev/dux-search?style=for-the-badge&labelColor=101418&color=ffd700)](https://github.com/duxlabs-dev/dux-search/stargazers)
[![GitHub release](https://img.shields.io/github/v/release/duxlabs-dev/dux-search?style=for-the-badge&labelColor=101418&color=9ccbfb)](https://github.com/duxlabs-dev/dux-search/releases)
[![AUR version (git)](https://img.shields.io/aur/version/dux-search-git?style=for-the-badge&labelColor=101418&color=9ccbfb&label=AUR%20(git))](https://aur.archlinux.org/packages/dux-search-git)
[![GitHub License](https://img.shields.io/github/license/duxlabs-dev/dux-search?style=for-the-badge&labelColor=101418&color=b9c8da)](https://github.com/duxlabs-dev/dux-search/blob/master/LICENSE)

</div>

---

A configurable filesystem search service powered by [bleve](https://github.com/blevesearch/bleve) for Unix-based operating systems.

**[Full documentation →](https://duxlabs.dev/docs/dsearch)**

---

## Features

- **Indexing** - Indexes all files in specified paths
- **Configuration** - Configure exclusions, depth, etc.
- **Fuzzy search** - Typo-tolerant search queries
- **Text content extraction** - Optionally extracts and indexes content from text-based files
- **EXIF metadata** - Extracts and indexes EXIF data from image files for advanced search
- **Virtual folders** - Search within specific directories
- **Concurrent indexing** - Multi-worker parallel indexing for fast performance
- **Real-time updates** - File watcher for incremental index updates
- **Semantic search** (optional, off by default) - Find files, actions and settings by meaning, in any language, fully on your machine

## Installation

Requires [Go](https://go.dev) 1.26+

```bash
make && sudo make install && make install-service
```


### NixOS

The repository contains a flake that provides a Home Manager module. To use it, you simply need to add
it as a flake input in your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # ...

    dsearch = {
      url = "github:duxlabs-dev/dux-search";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
}
```

and import/enable it in your Home Manager configuration:

```nix
{ inputs, ... }:  {
  imports = [ inputs.dsearch.homeModules.default ];

  programs.dsearch = {
    enable = true;

    # Put your config here or omit this for dsearch to generate the default config at runtime
    config = {
      # ...
    };
  };
}
```

## Usage

Best to run as a systemd user unit:

```bash
systemctl --user enable --now dsearch.service
```

Then search with various options:

```bash
# Basic search
dsearch search "golang"

# Show all results
dsearch search "*.md" --limit 0

# JSON output for scripting
dsearch search "config" --json
dsearch search "README" --limit 0 --json | jq '.hits[].id'

# Advanced options
dsearch search "function" --field filename --ext .go
dsearch search "typo" --fuzzy
dsearch search "*" --sort mtime --desc
# Scores content higher than filenames
dsearch search --field body mountain

# Virtual folder search
dsearch search "*" --folder /home/user/Pictures

# EXIF metadata search and filtering
dsearch search "*" --ext .jpg --exif-make Canon
dsearch search "*" --exif-model "Canon EOS 5D"
dsearch search "*" --folder /home/user/Photos --exif-make Nikon

# Sort photos by date taken (newest first)
dsearch search "*" --ext .jpg --sort exif_date --desc

# Find photos by ISO range (low-light shots)
dsearch search "*" --exif-min-iso 1600 --exif-max-iso 6400

# Find portrait shots (focal length range)
dsearch search "*" --exif-min-focal-len 50 --exif-max-focal-len 135

# Find wide aperture shots (shallow depth of field)
dsearch search "*" --exif-min-aperture 1.4 --exif-max-aperture 2.8

# Photos taken in date range
dsearch search "*" --exif-date-after "2024-06-01T00:00:00Z" --exif-date-before "2024-08-31T23:59:59Z"

# GPS location filtering (bounding box)
dsearch search "*" --exif-lat-min 40.0 --exif-lat-max 41.0 --exif-lon-min -74.0 --exif-lon-max -73.0
```

For all options:
```bash
dsearch search --help
```

The REST API is off by default. Start the service with `dsearch serve --http` or set `http_enabled = true` in the config, then visit [http://localhost:43654/docs](http://localhost:43654/docs) for the API documentation. The API has no authentication, so every local account can reach it once enabled.

## Semantic search

Semantic search finds things by what they mean instead of by the words they contain: "tired eyes" finds Night Light, "düzenli ifadeler" finds a document about regular expressions, and a text query can find screenshots, photos and recordings. It uses the [embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) model (Apache-2.0) running locally; text, images and audio are mapped into one vector space, so the same query works across all of them.

It is **off by default**. With `enabled = false` dsearch behaves exactly as it did before: no helper process, no vector database, no network access, and search results are unchanged.

### Requirements

- A system **ONNX Runtime 1.22 or newer** (`libonnxruntime.so`): `onnxruntime` on Arch and Fedora, `pkgs.onnxruntime` on Nix. dsearch loads it at run time; nothing is linked, so release binaries stay CGO-free. Set `onnxruntime_lib` in the config if it lives outside the standard library paths.
- The model files, downloaded once (`dsearch model download`): text tower about 300 MB, image tower about 190 MB, audio tower about 340 MB. Only the text tower is needed to search; the others are for indexing images and audio.
- About 0.5 GB of memory while the model is loaded. The model runs in a separate helper process (`dsearch-embed`, installed next to `dsearch`), is loaded on first use and exits after `idle_unload_minutes` without use (10 by default); if it crashes, dsearch keeps running and starts it again on the next query.

### Setup

```bash
dsearch model download            # text tower; --all adds image and audio
```

then in `~/.config/dsearch/config.toml`:

```toml
[semantic]
enabled = true
```

and restart the service (`systemctl --user restart dsearch`). `dsearch semantic status` shows the embedder, the model, the collections and the background indexing progress. Until the model is installed, everything keeps working with lexical results only and `semantic status` says what is missing.

See [config.example.toml](./config.example.toml) for all `[semantic]` options.

### Privacy

Everything runs on your machine. The only network access is `dsearch model download`, which fetches the model from Hugging Face at pinned revisions and checks every file against a pinned SHA-256 sum. Queries, file names and file contents are never sent anywhere. The vectors are stored in `semantic.db` next to the index (`~/.cache/dsearch/` by default). Background embedding of your files runs at low priority and pauses while you are active and, by default, on battery power; turn it off per kind with `index_text`, `index_images` and `index_audio`. `dsearch model remove` deletes the model files again.

### Searching

```bash
# Merge meaning-based results into the file hits (Reciprocal Rank Fusion)
dsearch search "regular expressions" --semantic

# Also search caller-owned collections (see below)
dsearch search "tired eyes" --semantic --collections shell.actions --collection-limit 5
```

File hits found by meaning carry the same fields as any other hit, plus `semantic_score` (cosine similarity after subtracting the mean vector of the file's collection, which keeps files that resemble everything from matching every query); hits are filtered by the usual options (`--ext`, `--folder`, `--min-size`, ...) either way. A file counts as found by meaning only when its score stands at least 3 standard deviations above the other files of its kind (text, image or audio) for that query; collections with fewer than 20 files add none. The meaning matches of all kinds form one ranking by that distance, and it is merged with the lexical results by [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) (`1/(60+rank)` per list), which needs no score normalization between BM25-style and cosine scores; when fusion ran, `score` is the fused score and the hits are ordered by it. Explicit sort orders (`--sort mtime`, ...) are never rearranged. With `--json` the response gains `collections` and a `semantic` object (`available`, `reason`, `fused`, `added`, `took_ms`); if the model is missing or too slow the lexical results are returned and `semantic.reason` says why.

### Collections

A collection is a set of items (`id`, `title`, `text`, `keywords`) that a client such as a launcher keeps in dsearch to search by meaning, for example its actions, settings or applications. A client replaces the whole set; items whose content did not change are not embedded again, and when the model is not installed yet the items are stored and embedded as soon as it is.

```bash
dsearch collections set shell.actions --json actions.json
dsearch collections delete shell.actions
dsearch semantic warm            # load the model ahead of use
```

Over the unix socket (JSON lines, `$XDG_RUNTIME_DIR/dsearch-<pid>.sock`; `dsearch version --json` prints the path and `capabilities`):

```jsonc
// -> collections.set
{"id":1,"method":"collections.set","params":{"collection":"shell.actions","items":[
  {"id":"nightlight","title":"Night Light","text":"Warm screen colors that reduce blue light; tired eyes, reading at night","keywords":["eye strain"]}]}}
// <-
{"id":1,"result":{"collection":"shell.actions","total":1,"embedded":1,"removed":0,"pending":0}}

// -> search with semantic results
{"id":2,"method":"search","params":{"query":"gözlerim yoruldu","semantic":true,"collections":["shell.actions"],"collection_limit":3,"type":"file"}}
// <- the usual search result, with semantic_score on file hits, plus
{"collections":[{"collection":"shell.actions","hits":[{"id":"nightlight","score":0.66}]}],
 "semantic":{"available":true,"fused":true,"added":0,"took_ms":38.2}}
```

Other methods: `collections.delete {collection}`, `semantic.status`, `semantic.warm {modalities?}`, `model.status`, `model.download {towers}`, `model.remove {towers}`. Progress of a download is published as `model.progress` events (`{"tower","bytes","total","state"}`, ending with `state: "finished"` or `"failed"`) to connections that sent `{"method":"subscribe","params":{"topics":["model.progress"]}}`. The HTTP API (when enabled) has the same operations: `GET /search?semantic=true&collections=...`, `PUT|DELETE /collections/{name}`, `GET /semantic/status`, `POST /semantic/warm`, `GET /model/status`, `POST /model/download`, `POST /model/remove`; like the rest of that API they are unauthenticated.

## Configuration

See [config.example.toml](./config.example.toml) for configuration options.

On startup, a configuration file will be created at `~/.config/dsearch/config.toml`