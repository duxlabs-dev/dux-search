package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// SemanticConfig is the [semantic] table: embedding based search, off by default.
type SemanticConfig struct {
	// Enabled turns semantic search on. Off, dsearch behaves exactly as before:
	// no helper process, no vector database, no model access.
	Enabled bool `toml:"enabled"`
	// ModelDir holds the model files; default $XDG_DATA_HOME/dsearch/models/embeddinggemma-2.
	ModelDir string `toml:"model_dir"`
	// OnnxRuntimeLib is the path of libonnxruntime.so; empty searches the standard locations.
	OnnxRuntimeLib string `toml:"onnxruntime_lib"`
	// Threads for model inference; 0 = min(4, CPU count).
	Threads int `toml:"threads"`
	// IdleUnloadMinutes is how long the model helper stays in memory without use.
	IdleUnloadMinutes int `toml:"idle_unload_minutes"`
	// IndexText, IndexImages and IndexAudio choose which file kinds are embedded
	// in the background.
	IndexText   bool `toml:"index_text"`
	IndexImages bool `toml:"index_images"`
	IndexAudio  bool `toml:"index_audio"`
	// MaxChunksPerFile caps the text chunks embedded per file.
	MaxChunksPerFile int `toml:"max_chunks_per_file"`
	// PauseOnBattery pauses background embedding while on battery power.
	PauseOnBattery bool `toml:"pause_on_battery"`
}

// Limits of the numeric settings.
const (
	MaxSemanticThreads  = 256
	MaxIdleUnloadMin    = 24 * 60
	MaxChunksPerFileCap = 1000
)

// DefaultSemantic returns the defaults: disabled, everything else ready to use.
func DefaultSemantic() SemanticConfig {
	return SemanticConfig{
		Enabled:           false,
		ModelDir:          defaultModelDir(),
		IdleUnloadMinutes: 10,
		IndexText:         true,
		IndexImages:       true,
		IndexAudio:        true,
		MaxChunksPerFile:  8,
		PauseOnBattery:    true,
	}
}

// IdleUnload returns the idle timeout as a duration.
func (s SemanticConfig) IdleUnload() time.Duration {
	return time.Duration(s.IdleUnloadMinutes) * time.Minute
}

// Validate rejects values that cannot work. It runs on load, so a typo is a
// startup error with the key name instead of a silent misbehavior.
func (s SemanticConfig) Validate() error {
	var errs []error
	if s.Threads < 0 || s.Threads > MaxSemanticThreads {
		errs = append(errs, fmt.Errorf("semantic.threads = %d: must be 0 (automatic) to %d", s.Threads, MaxSemanticThreads))
	}
	if s.IdleUnloadMinutes < 1 || s.IdleUnloadMinutes > MaxIdleUnloadMin {
		errs = append(errs, fmt.Errorf("semantic.idle_unload_minutes = %d: must be 1 to %d", s.IdleUnloadMinutes, MaxIdleUnloadMin))
	}
	if s.MaxChunksPerFile < 1 || s.MaxChunksPerFile > MaxChunksPerFileCap {
		errs = append(errs, fmt.Errorf("semantic.max_chunks_per_file = %d: must be 1 to %d", s.MaxChunksPerFile, MaxChunksPerFileCap))
	}
	if s.ModelDir == "" {
		errs = append(errs, errors.New("semantic.model_dir must not be empty"))
	} else if !filepath.IsAbs(s.ModelDir) {
		errs = append(errs, fmt.Errorf("semantic.model_dir = %q: must be an absolute path (~ and $VARS are expanded)", s.ModelDir))
	}
	if s.OnnxRuntimeLib != "" && !filepath.IsAbs(s.OnnxRuntimeLib) {
		errs = append(errs, fmt.Errorf("semantic.onnxruntime_lib = %q: must be an absolute path or empty", s.OnnxRuntimeLib))
	}
	return errors.Join(errs...)
}

func dataBase() string {
	if runtime.GOOS == "windows" {
		if base := os.Getenv("LOCALAPPDATA"); base != "" {
			return base
		}
		return filepath.Join(os.Getenv("USERPROFILE"), "AppData", "Local")
	}
	if base := os.Getenv("XDG_DATA_HOME"); base != "" {
		return base
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share")
}

func defaultModelDir() string {
	return filepath.Join(dataBase(), appDirName, "models", "embeddinggemma-2")
}
