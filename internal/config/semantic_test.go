package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.toml")
	require.NoError(t, os.WriteFile(path, []byte(body), 0o644))
	return path
}

func TestSemanticDefaults(t *testing.T) {
	data := t.TempDir()
	t.Setenv("XDG_DATA_HOME", data)

	cfg := Default()
	assert.False(t, cfg.Semantic.Enabled, "semantic search is opt-in")
	assert.Equal(t, filepath.Join(data, "dsearch", "models", "embeddinggemma-2"), cfg.Semantic.ModelDir)
	assert.Empty(t, cfg.Semantic.OnnxRuntimeLib)
	assert.Zero(t, cfg.Semantic.Threads)
	assert.Equal(t, "auto", cfg.Semantic.Device)
	assert.Equal(t, 10, cfg.Semantic.IdleUnloadMinutes)
	assert.Equal(t, 10*time.Minute, cfg.Semantic.IdleUnload())
	assert.True(t, cfg.Semantic.IndexText && cfg.Semantic.IndexImages && cfg.Semantic.IndexAudio)
	assert.Equal(t, 8, cfg.Semantic.MaxChunksPerFile)
	assert.True(t, cfg.Semantic.PauseOnBattery)
	assert.NoError(t, cfg.Semantic.Validate())
}

func TestSemanticModelDirFallsBackToHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("XDG_DATA_HOME", "")
	t.Setenv("HOME", home)
	assert.Equal(t, filepath.Join(home, ".local", "share", "dsearch", "models", "embeddinggemma-2"), Default().Semantic.ModelDir)
}

// A config written before the feature existed must load unchanged, with the
// feature off.
func TestLegacyConfigWithoutSemanticTable(t *testing.T) {
	path := writeConfig(t, "index_path = \"/tmp/x/index\"\nmax_file_bytes = 1000\n")
	cfg, err := Load(path)
	require.NoError(t, err)
	assert.False(t, cfg.Semantic.Enabled)
	assert.Equal(t, DefaultSemantic(), cfg.Semantic)
	assert.Equal(t, int64(1000), cfg.MaxFileBytes)
}

func TestSemanticTableOverridesOnlyWhatItSets(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MODELS_ROOT", "/srv/models")
	path := writeConfig(t, `
[semantic]
enabled = true
model_dir = "~/gemma"
onnxruntime_lib = "$MODELS_ROOT/lib/libonnxruntime.so"
threads = 2
device = "cuda"
idle_unload_minutes = 30
index_images = false
max_chunks_per_file = 3
pause_on_battery = false
`)
	cfg, err := Load(path)
	require.NoError(t, err)
	s := cfg.Semantic
	assert.True(t, s.Enabled)
	assert.Equal(t, filepath.Join(home, "gemma"), s.ModelDir, "~ is expanded")
	assert.Equal(t, "/srv/models/lib/libonnxruntime.so", s.OnnxRuntimeLib, "$VARS are expanded")
	assert.Equal(t, 2, s.Threads)
	assert.Equal(t, "cuda", s.Device)
	assert.Equal(t, 30*time.Minute, s.IdleUnload())
	assert.True(t, s.IndexText, "unset keys keep their default")
	assert.False(t, s.IndexImages)
	assert.True(t, s.IndexAudio)
	assert.Equal(t, 3, s.MaxChunksPerFile)
	assert.False(t, s.PauseOnBattery)
}

func TestSemanticValidation(t *testing.T) {
	for name, tc := range map[string]struct {
		body string
		want string
	}{
		"negative threads":   {"threads = -1", "semantic.threads"},
		"huge threads":       {"threads = 100000", "semantic.threads"},
		"zero idle":          {"idle_unload_minutes = 0", "semantic.idle_unload_minutes"},
		"idle too long":      {"idle_unload_minutes = 99999", "semantic.idle_unload_minutes"},
		"zero chunks":        {"max_chunks_per_file = 0", "semantic.max_chunks_per_file"},
		"too many chunks":    {"max_chunks_per_file = 100000", "semantic.max_chunks_per_file"},
		"empty model dir":    {`model_dir = ""`, "semantic.model_dir"},
		"relative model dir": {`model_dir = "models/gemma"`, "semantic.model_dir"},
		"relative ort":       {`onnxruntime_lib = "libonnxruntime.so"`, "semantic.onnxruntime_lib"},
		"unknown device":     {`device = "gpu"`, "semantic.device"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Load(writeConfig(t, "[semantic]\n"+tc.body+"\n"))
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.want)
		})
	}
	// every problem is reported, not just the first
	_, err := Load(writeConfig(t, "[semantic]\nthreads = -1\nmax_chunks_per_file = 0\n"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "semantic.threads")
	assert.Contains(t, err.Error(), "semantic.max_chunks_per_file")
}

func TestSemanticTypeErrorIsAnError(t *testing.T) {
	_, err := Load(writeConfig(t, "[semantic]\nenabled = \"yes\"\n"))
	assert.Error(t, err)
}

func TestSemanticDBPathSitsNextToTheIndex(t *testing.T) {
	cfg := &Config{IndexPath: "/var/cache/dsearch/index"}
	cfg.BuildMaps()
	assert.Equal(t, "/var/cache/dsearch/semantic.db", cfg.SemanticDBPath())
	assert.Equal(t, "/var/cache/dsearch/meta.db", cfg.metastorePath)
	empty := &Config{}
	empty.BuildMaps()
	assert.Empty(t, empty.SemanticDBPath())
}

// The vector database and the model files are dsearch's own data and must not
// be indexed as user files.
func TestOwnSemanticDataIsNotIndexed(t *testing.T) {
	root := t.TempDir()
	cfg := Default()
	cfg.IndexPath = filepath.Join(root, "cache", "index")
	cfg.Semantic.ModelDir = filepath.Join(root, "models", "gemma")
	cfg.IndexPaths = []IndexPath{{Path: root, ExcludeDirs: []string{}}}
	cfg.BuildMaps()

	assert.True(t, cfg.ShouldIndexFile(filepath.Join(root, "notes.txt")))
	assert.False(t, cfg.ShouldIndexFile(filepath.Join(root, "cache", "semantic.db")))
	assert.False(t, cfg.ShouldIndexFile(filepath.Join(root, "cache", "semantic.db.corrupt")))
	assert.False(t, cfg.ShouldIndexFile(filepath.Join(root, "cache", "meta.db")))
	assert.False(t, cfg.ShouldIndexFile(filepath.Join(root, "models", "gemma", "onnx", "model.onnx_data")))
	assert.True(t, cfg.ShouldIndexFile(filepath.Join(root, "models", "gemma-notes.txt")), "a sibling that merely shares the prefix is a user file")
}

func TestSavedDefaultConfigRoundTrips(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	path := filepath.Join(t.TempDir(), "config.toml")
	require.NoError(t, Default().Save(path))
	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(raw), "[semantic]")
	assert.Contains(t, string(raw), "enabled = false")
	cfg, err := Load(path)
	require.NoError(t, err)
	assert.Equal(t, DefaultSemantic(), cfg.Semantic)
}

// The documented example must stay loadable.
func TestExampleConfigLoads(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	cfg, err := Load(filepath.Join("..", "..", "config.example.toml"))
	require.NoError(t, err)
	assert.False(t, cfg.Semantic.Enabled)
	assert.Equal(t, DefaultSemantic(), cfg.Semantic, "the example documents the defaults and changes none")
}
