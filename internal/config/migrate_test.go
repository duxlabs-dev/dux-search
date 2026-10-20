package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMigrateLegacyDirs(t *testing.T) {
	home := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, "config"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(home, "cache"))

	legacyConfig := filepath.Join(home, "config", "danksearch", "config.toml")
	require.NoError(t, os.MkdirAll(filepath.Dir(legacyConfig), 0o755))
	require.NoError(t, os.WriteFile(legacyConfig, []byte("index_path = \"x\""), 0o644))

	dotfiles := filepath.Join(home, "dotfiles", "index")
	require.NoError(t, os.MkdirAll(dotfiles, 0o755))
	require.NoError(t, os.MkdirAll(filepath.Join(home, "cache"), 0o755))
	require.NoError(t, os.Symlink(filepath.Dir(dotfiles), filepath.Join(home, "cache", "danksearch")))

	for range 2 {
		require.NoError(t, MigrateLegacyDirs())
	}

	data, err := os.ReadFile(GetDefaultConfigPath())
	require.NoError(t, err)
	assert.Equal(t, "index_path = \"x\"", string(data))
	_, err = os.ReadFile(legacyConfig)
	assert.NoError(t, err, "legacy config path keeps resolving through the symlink")
	info, err := os.Stat(filepath.Dir(getDefaultIndexPath()))
	require.NoError(t, err)
	assert.True(t, info.IsDir())
}
