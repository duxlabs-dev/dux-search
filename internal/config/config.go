package config

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/BurntSushi/toml"
)

type IndexPath struct {
	Path                    string   `toml:"path"`
	MaxDepth                int      `toml:"max_depth"`
	ExcludeHidden           bool     `toml:"exclude_hidden"`
	ExcludeDirs             []string `toml:"exclude_dirs"`
	MergeDefaultExcludeDirs bool     `toml:"merge_default_exclude_dirs,omitempty"`
	ExtractExif             *bool    `toml:"extract_exif,omitempty"`
	ExtractXattrTags        bool     `toml:"extract_xattr_tags"`
	Watch                   *bool    `toml:"watch,omitempty"`

	excludeDirsMap   map[string]bool
	excludeDirsRegex []*regexp.Regexp
}

type Config struct {
	IndexPath      string      `toml:"index_path"`
	ListenAddr     string      `toml:"listen_addr"`
	MaxFileBytes   int64       `toml:"max_file_bytes"`
	WorkerCount    int         `toml:"worker_count"`
	IndexPaths     []IndexPath `toml:"index_paths"`
	TextExts       []string    `toml:"text_extensions"`
	IndexAllFiles  bool        `toml:"index_all_files"`
	IndexXattrTags bool        `toml:"index_xattr_tags"`

	RootDir       string   `toml:"root_dir,omitempty"`
	MaxDepth      int      `toml:"max_depth,omitempty"`
	ExcludeHidden bool     `toml:"exclude_hidden,omitempty"`
	ExcludeDirs   []string `toml:"exclude_dirs,omitempty"`

	textExtsMap   map[string]bool
	metastorePath string
}

func DefaultExcludeDirs() []string {
	return []string{
		".git", ".hg", ".svn",
		"node_modules", "bower_components", ".npm", ".yarn",
		"site-packages", "__pycache__", ".venv", "venv", ".tox", ".pytest_cache", ".eggs",
		"dist", "build", "out", "bin", "obj",
		"target", "vendor",
		".gradle", ".m2", "bundle",
		".cache", ".parcel-cache", ".next", ".nuxt", ".serverless",
		"Library", ".Trash-1000",
		".postgresql", ".postgres", ".mysql", ".mongodb", ".redis", "pgdata", "pg_data",
		"go", ".cargo", ".pyenv", ".rbenv", ".nvm", ".rustup", ".composer", ".gem",
		".idea", ".vscode",
	}
}

func Default() *Config {
	home, _ := os.UserHomeDir()

	workerCount := runtime.NumCPU() / 2
	if workerCount < 1 {
		workerCount = 1
	}

	cfg := &Config{
		IndexPath:      getDefaultIndexPath(),
		ListenAddr:     "127.0.0.1:43654",
		MaxFileBytes:   2 * 1024 * 1024,
		WorkerCount:    workerCount,
		IndexAllFiles:  true,
		IndexXattrTags: true,
		IndexPaths: []IndexPath{
			{
				Path:          home,
				MaxDepth:      6,
				ExcludeHidden: true,
				ExcludeDirs:   DefaultExcludeDirs(),
			},
		},
		TextExts: []string{
			".txt", ".md", ".go", ".py", ".js", ".ts",
			".jsx", ".tsx", ".json", ".yaml", ".yml",
			".toml", ".html", ".css", ".rs", ".c",
			".cpp", ".h", ".java", ".rb", ".php", ".sh",
		},
	}

	cfg.BuildMaps()
	return cfg
}

func Load(path string) (*Config, error) {
	cfg := Default()

	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := cfg.Save(path); err != nil {
			log.Warnf("failed to create default config at %s: %v", path, err)
		} else {
			log.Infof("created default config at %s", path)
		}
		return cfg, nil
	}

	md, err := toml.DecodeFile(path, cfg)
	if err != nil {
		return nil, err
	}
	for _, key := range md.Undecoded() {
		log.Warnf("config: unknown key %q in %s is ignored", key.String(), path)
	}

	if cfg.RootDir != "" && len(cfg.IndexPaths) == 0 {
		cfg.IndexPaths = []IndexPath{
			{
				Path:          cfg.RootDir,
				MaxDepth:      cfg.MaxDepth,
				ExcludeHidden: cfg.ExcludeHidden,
				ExcludeDirs:   cfg.ExcludeDirs,
			},
		}
	}

	cfg.ExpandPaths()
	cfg.BuildMaps()
	return cfg, nil
}

func (c *Config) Save(path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	f.WriteString("# Dux Search Configuration\n")
	f.WriteString("# See https://github.com/duxlabs-dev/dux-search for documentation\n\n")

	return toml.NewEncoder(f).Encode(c)
}

func expandPath(path string) string {
	switch {
	case path == "~":
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return home
	case strings.HasPrefix(path, "~/"):
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[2:])
	default:
		return os.ExpandEnv(path)
	}
}

func (c *Config) ExpandPaths() {
	c.IndexPath = expandPath(c.IndexPath)
	c.RootDir = expandPath(c.RootDir)
	for i := range c.IndexPaths {
		c.IndexPaths[i].Path = expandPath(c.IndexPaths[i].Path)
	}
}

func (c *Config) BuildMaps() {
	c.metastorePath = ""
	if c.IndexPath != "" {
		c.metastorePath = filepath.Join(filepath.Dir(c.IndexPath), "meta.db")
	}

	for i := range c.IndexPaths {
		dirs := c.IndexPaths[i].resolvedExcludeDirs()
		c.IndexPaths[i].excludeDirsMap = make(map[string]bool, len(dirs))
		c.IndexPaths[i].excludeDirsRegex = nil
		for _, dir := range dirs {
			switch {
			case len(dir) >= 3 && dir[0] == '/' && dir[len(dir)-1] == '/':
				pattern := dir[1 : len(dir)-1]
				re, err := regexp.Compile(pattern)
				if err != nil {
					log.Warnf("invalid regex in exclude_dirs %q: %v", dir, err)
					continue
				}
				c.IndexPaths[i].excludeDirsRegex = append(c.IndexPaths[i].excludeDirsRegex, re)
			default:
				c.IndexPaths[i].excludeDirsMap[dir] = true
			}
		}
	}

	c.textExtsMap = make(map[string]bool, len(c.TextExts))
	for _, ext := range c.TextExts {
		c.textExtsMap[strings.ToLower(ext)] = true
	}
}

// ResolvedExcludeDirs is the exclusion list the walker applies, defaults
// merged in when the path asks for them.
func (ip *IndexPath) ResolvedExcludeDirs() []string {
	return ip.resolvedExcludeDirs()
}

func (ip *IndexPath) resolvedExcludeDirs() []string {
	if !ip.MergeDefaultExcludeDirs {
		return ip.ExcludeDirs
	}

	defaults := DefaultExcludeDirs()
	merged := make([]string, 0, len(defaults)+len(ip.ExcludeDirs))
	seen := make(map[string]struct{}, len(defaults)+len(ip.ExcludeDirs))
	for _, dir := range defaults {
		if _, dup := seen[dir]; dup {
			continue
		}
		seen[dir] = struct{}{}
		merged = append(merged, dir)
	}
	for _, dir := range ip.ExcludeDirs {
		if _, dup := seen[dir]; dup {
			continue
		}
		seen[dir] = struct{}{}
		merged = append(merged, dir)
	}
	return merged
}

const (
	appDirName       = "dsearch"
	legacyAppDirName = "danksearch"
)

func cacheBase() string {
	if runtime.GOOS == "windows" {
		if base := os.Getenv("LOCALAPPDATA"); base != "" {
			return base
		}
		return filepath.Join(os.Getenv("USERPROFILE"), "AppData", "Local")
	}
	if base := os.Getenv("XDG_CACHE_HOME"); base != "" {
		return base
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache")
}

func configBase() string {
	if runtime.GOOS == "windows" {
		if base := os.Getenv("APPDATA"); base != "" {
			return base
		}
		return filepath.Join(os.Getenv("USERPROFILE"), "AppData", "Roaming")
	}
	if base := os.Getenv("XDG_CONFIG_HOME"); base != "" {
		return base
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config")
}

func getDefaultIndexPath() string {
	return filepath.Join(cacheBase(), appDirName, "index")
}

func GetDefaultConfigPath() string {
	return filepath.Join(configBase(), appDirName, "config.toml")
}

// MigrateLegacyDirs moves the danksearch config and cache dirs to dsearch and
// leaves a symlink behind, since configs store absolute index paths.
func MigrateLegacyDirs() error {
	for _, base := range []string{configBase(), cacheBase()} {
		if err := migrateLegacyDir(filepath.Join(base, legacyAppDirName), filepath.Join(base, appDirName)); err != nil {
			return err
		}
	}
	return nil
}

func migrateLegacyDir(oldDir, newDir string) error {
	oldInfo, err := os.Lstat(oldDir)
	if err != nil {
		return nil
	}
	if _, err := os.Lstat(newDir); err == nil {
		return nil
	}
	if oldInfo.Mode()&os.ModeSymlink != 0 {
		return os.Symlink(legacyAppDirName, newDir)
	}
	if !oldInfo.IsDir() {
		return nil
	}
	if err := os.Rename(oldDir, newDir); err != nil {
		return os.Symlink(legacyAppDirName, newDir)
	}
	return os.Symlink(appDirName, oldDir)
}

func (c *Config) findIndexPath(path string) *IndexPath {
	var best *IndexPath
	bestLen := 0
	for i := range c.IndexPaths {
		root := c.IndexPaths[i].Path
		if !isWithin(path, root) {
			continue
		}
		if len(root) <= bestLen {
			continue
		}
		best = &c.IndexPaths[i]
		bestLen = len(root)
	}
	return best
}

func isWithin(path, root string) bool {
	if path == root {
		return true
	}
	if !strings.HasSuffix(root, string(filepath.Separator)) {
		root += string(filepath.Separator)
	}
	return strings.HasPrefix(path, root)
}

func (c *Config) FindIndexPath(path string) *IndexPath {
	return c.findIndexPath(path)
}

func (c *Config) ExclusionReason(path string) string {
	idxPath := c.findIndexPath(path)
	if idxPath == nil {
		return "not under any configured index path"
	}
	if c.isOwnIndexData(path) {
		return "dsearch index data"
	}
	if idxPath.ExcludeHidden && containsHiddenComponent(path, idxPath.Path) {
		return "hidden"
	}
	if comp := excludedComponent(path, idxPath.Path, idxPath.excludeDirsMap, idxPath.excludeDirsRegex); comp != "" {
		return "exclude_dirs: " + comp
	}
	return ""
}

func (c *Config) isOwnIndexData(path string) bool {
	if c.metastorePath == "" {
		return false
	}
	return isWithin(path, c.IndexPath) || strings.HasPrefix(path, c.metastorePath)
}

func (c *Config) ShouldIndexFile(path string) bool {
	idxPath := c.findIndexPath(path)
	if idxPath == nil || c.isOwnIndexData(path) {
		return false
	}

	if idxPath.ExcludeHidden && containsHiddenComponent(path, idxPath.Path) {
		return false
	}

	if containsExcludedComponent(path, idxPath.Path, idxPath.excludeDirsMap, idxPath.excludeDirsRegex) {
		return false
	}

	if c.IndexAllFiles {
		return true
	}
	return c.IsTextFile(path)
}

func (c *Config) ShouldIndexDir(path string) bool {
	idxPath := c.findIndexPath(path)
	if idxPath == nil || c.isOwnIndexData(path) {
		return false
	}

	if idxPath.ExcludeHidden && containsHiddenComponent(path, idxPath.Path) {
		return false
	}

	return !containsExcludedComponent(path, idxPath.Path, idxPath.excludeDirsMap, idxPath.excludeDirsRegex)
}

func containsHiddenComponent(path, rootDir string) bool {
	rel, err := filepath.Rel(rootDir, path)
	if err != nil {
		return false
	}

	if rel == "." {
		return false
	}

	for _, comp := range strings.Split(rel, string(filepath.Separator)) {
		if len(comp) > 0 && comp[0] == '.' {
			return true
		}
	}
	return false
}

func containsExcludedComponent(path, rootDir string, excludeMap map[string]bool, regexes []*regexp.Regexp) bool {
	rel, err := filepath.Rel(rootDir, path)
	if err != nil {
		return false
	}

	if rel == "." {
		return false
	}

	for _, comp := range strings.Split(rel, string(filepath.Separator)) {
		if isExcludedComponent(comp, excludeMap, regexes) {
			return true
		}
	}
	return false
}

func isExcludedComponent(comp string, excludeMap map[string]bool, regexes []*regexp.Regexp) bool {
	if excludeMap[comp] {
		return true
	}
	for _, re := range regexes {
		if re.MatchString(comp) {
			return true
		}
	}
	return false
}

func excludedComponent(path, rootDir string, excludeMap map[string]bool, regexes []*regexp.Regexp) string {
	rel, err := filepath.Rel(rootDir, path)
	if err != nil || rel == "." {
		return ""
	}

	for _, comp := range strings.Split(rel, string(filepath.Separator)) {
		if isExcludedComponent(comp, excludeMap, regexes) {
			return comp
		}
	}
	return ""
}

func (c *Config) GetDepth(path string) int {
	idxPath := c.findIndexPath(path)
	if idxPath == nil {
		return 0
	}

	rel, err := filepath.Rel(idxPath.Path, path)
	if err != nil {
		return 0
	}

	if rel == "." {
		return 0
	}
	return strings.Count(rel, string(filepath.Separator)) + 1
}

func (c *Config) GetMaxDepth(path string) int {
	idxPath := c.findIndexPath(path)
	if idxPath == nil {
		return 0
	}
	return idxPath.MaxDepth
}

func (c *Config) IsTextFile(path string) bool {
	return c.textExtsMap[strings.ToLower(filepath.Ext(path))]
}

func (ip *IndexPath) ShouldWatch() bool {
	return ip.Watch == nil || *ip.Watch
}

func (ip *IndexPath) ShouldExtractExif() bool {
	return ip.ExtractExif == nil || *ip.ExtractExif
}

func (c *Config) ExtractExif(path string) bool {
	idxPath := c.findIndexPath(path)
	return idxPath != nil && idxPath.ShouldExtractExif()
}

type IndexStats struct {
	TotalFiles    int       `json:"total_files"`
	TotalBytes    int64     `json:"total_bytes"`
	LastIndexTime time.Time `json:"last_index_time"`
	IndexDuration string    `json:"index_duration"`

	Phase                 string    `json:"phase,omitempty"`
	PhaseStartedAt        time.Time `json:"phase_started_at,omitzero"`
	FilesProcessed        int64     `json:"files_processed,omitempty"`
	BytesProcessed        int64     `json:"bytes_processed,omitempty"`
	SchemaVersion         int       `json:"schema_version,omitempty"`
	ExpectedSchemaVersion int       `json:"expected_schema_version,omitempty"`
}
