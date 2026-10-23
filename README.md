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

Or visit [http://localhost:43654/docs](http://localhost:43654/docs) for rest API documentation.

## Configuration

See [config.example.toml](./config.example.toml) for configuration options.

On startup, a configuration file will be created at `~/.config/dsearch/config.toml`