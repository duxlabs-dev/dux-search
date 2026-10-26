package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/AvengeMedia/dankgo/app"
	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/client"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	"github.com/duxlabs-dev/dux-search/internal/metastore"
	"github.com/duxlabs-dev/dux-search/internal/server"
	"github.com/duxlabs-dev/dux-search/internal/watcher"
	"github.com/spf13/cobra"
)

var (
	Version   string = "dev"
	buildTime string = "unknown"
	commit    string = "unknown"

	configFile    string
	rootDir       string
	indexPath     string
	listenAddr    string
	maxFileBytes  int64
	workerCount   int
	maxDepth      int
	excludeHidden bool
	noWatch       bool
	enableHTTP    bool
	socketOnly    bool

	searchLimit    int
	searchField    string
	searchExt      string
	searchFuzzy    bool
	searchSort     string
	searchSortDesc bool
	searchMinSize  int64
	searchMaxSize  int64
	searchJSON     bool

	filesLimit            int
	searchFolder          string
	searchExifMake        string
	searchExifModel       string
	searchExifDateAfter   string
	searchExifDateBefore  string
	searchExifMinISO      int
	searchExifMaxISO      int
	searchExifMinAp       float64
	searchExifMaxAp       float64
	searchExifMinFocalLen float64
	searchExifMaxFocalLen float64
	searchExifLatMin      float64
	searchExifLatMax      float64
	searchExifLonMin      float64
	searchExifLonMax      float64
	searchXattrTags       string
	searchType            string
)

var rootCmd = &cobra.Command{
	Use:           "dsearch",
	Short:         "Desktop filesystem search service",
	Long:          "A filesystem search service using Bleve to index and search files",
	SilenceUsage:  true,
	SilenceErrors: true,
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the search service",
	RunE:  runServe,
}

var searchCmd = &cobra.Command{
	Use:   "search [query]",
	Short: "Search the index",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runSearch,
}

var indexCmd = &cobra.Command{
	Use:   "index",
	Short: "Manage the search index",
}

var indexGenerateCmd = &cobra.Command{
	Use:   "generate",
	Short: "Trigger a full reindex (async)",
	RunE:  runIndexGenerate,
}

var indexSyncCmd = &cobra.Command{
	Use:   "sync",
	Short: "Incremental sync - add new/updated, remove deleted files",
	RunE:  runIndexSync,
}

var indexStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show index statistics",
	RunE:  runIndexStatus,
}

var indexDirsCmd = &cobra.Command{
	Use:   "dirs",
	Short: "List configured index paths",
	RunE:  runIndexDirs,
}

var indexCheckCmd = &cobra.Command{
	Use:   "check <path>",
	Short: "Check whether a path is indexed",
	Args:  cobra.ExactArgs(1),
	RunE:  runIndexCheck,
}

var indexFilesCmd = &cobra.Command{
	Use:   "files [prefix]",
	Short: "List files in the metastore",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runIndexFiles,
}

var watchCmd = &cobra.Command{
	Use:   "watch",
	Short: "Manage file watcher",
}

var watchStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Check watcher status",
	RunE:  runWatchStatus,
}

var watchStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start file watcher",
	RunE:  runWatchStart,
}

var watchStopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop file watcher",
	RunE:  runWatchStop,
}

var versionJSON bool

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Show version information",
	Run: func(cmd *cobra.Command, args []string) {
		if versionJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(map[string]any{
				"version":      Version,
				"build_time":   buildTime,
				"commit":       commit,
				"index_schema": indexer.SchemaVersion,
			})
			return
		}
		log.Infof("dsearch version %s", Version)
		log.Infof("  Build time: %s", buildTime)
		log.Infof("  Commit: %s", commit)
		log.Infof("  Index schema: %d", indexer.SchemaVersion)
	},
}

func init() {
	rootCmd.PersistentFlags().StringVarP(&configFile, "config", "c", "", "config file path (default: ~/.config/dsearch/config.toml)")

	serveCmd.Flags().StringVar(&rootDir, "root", "", "directory to index")
	serveCmd.Flags().StringVar(&indexPath, "index", "", "index storage path")
	serveCmd.Flags().StringVar(&listenAddr, "listen", "", "HTTP listen address (implies --http)")
	serveCmd.Flags().Int64Var(&maxFileBytes, "max-bytes", 0, "max file size to index")
	serveCmd.Flags().IntVar(&workerCount, "workers", 0, "number of indexing workers")
	serveCmd.Flags().IntVar(&maxDepth, "max-depth", -1, "maximum directory depth to traverse (0 = unlimited)")
	serveCmd.Flags().BoolVar(&excludeHidden, "exclude-hidden", false, "exclude hidden files and directories")
	serveCmd.Flags().BoolVar(&noWatch, "no-watch", false, "disable automatic file watching")
	serveCmd.Flags().BoolVar(&enableHTTP, "http", false, "also serve the unauthenticated HTTP API on the listen address")
	serveCmd.Flags().BoolVar(&socketOnly, "socket", false, "unix socket only, overrides http_enabled from config")

	searchCmd.Flags().IntVarP(&searchLimit, "limit", "n", 10, "maximum number of results (0 for all)")
	searchCmd.Flags().StringVarP(&searchField, "field", "f", "", "search specific field (filename, body, title)")
	searchCmd.Flags().StringVarP(&searchExt, "ext", "e", "", "filter by file extension (e.g., .jpg)")
	searchCmd.Flags().BoolVar(&searchFuzzy, "fuzzy", false, "enable fuzzy matching for typos")
	searchCmd.Flags().StringVar(&searchSort, "sort", "score", "sort by: score, mtime, size, filename, exif_date, iso, focal_length, aperture")
	searchCmd.Flags().BoolVar(&searchSortDesc, "desc", true, "sort descending")
	searchCmd.Flags().Int64Var(&searchMinSize, "min-size", 0, "minimum file size in bytes")
	searchCmd.Flags().Int64Var(&searchMaxSize, "max-size", 0, "maximum file size in bytes")
	searchCmd.Flags().BoolVar(&searchJSON, "json", false, "output results in JSON format")
	searchCmd.Flags().StringVar(&searchFolder, "folder", "", "filter by folder path (e.g., /home/user/Pictures)")
	searchCmd.Flags().StringVar(&searchExifMake, "exif-make", "", "filter by camera make (e.g., Canon)")
	searchCmd.Flags().StringVar(&searchExifModel, "exif-model", "", "filter by camera model (e.g., Canon EOS 5D)")
	searchCmd.Flags().StringVar(&searchExifDateAfter, "exif-date-after", "", "photos taken after date (RFC3339 or EXIF format)")
	searchCmd.Flags().StringVar(&searchExifDateBefore, "exif-date-before", "", "photos taken before date (RFC3339 or EXIF format)")
	searchCmd.Flags().IntVar(&searchExifMinISO, "exif-min-iso", 0, "minimum ISO value")
	searchCmd.Flags().IntVar(&searchExifMaxISO, "exif-max-iso", 0, "maximum ISO value")
	searchCmd.Flags().Float64Var(&searchExifMinAp, "exif-min-aperture", 0, "minimum aperture (f-number)")
	searchCmd.Flags().Float64Var(&searchExifMaxAp, "exif-max-aperture", 0, "maximum aperture (f-number)")
	searchCmd.Flags().Float64Var(&searchExifMinFocalLen, "exif-min-focal-len", 0, "minimum focal length in mm")
	searchCmd.Flags().Float64Var(&searchExifMaxFocalLen, "exif-max-focal-len", 0, "maximum focal length in mm")
	searchCmd.Flags().Float64Var(&searchExifLatMin, "exif-lat-min", 0, "minimum GPS latitude")
	searchCmd.Flags().Float64Var(&searchExifLatMax, "exif-lat-max", 0, "maximum GPS latitude")
	searchCmd.Flags().Float64Var(&searchExifLonMin, "exif-lon-min", 0, "minimum GPS longitude")
	searchCmd.Flags().Float64Var(&searchExifLonMax, "exif-lon-max", 0, "maximum GPS longitude")
	searchCmd.Flags().StringVar(&searchXattrTags, "xattr-tags", "", "tags in user.xdg.tags xattr")
	searchCmd.Flags().StringVarP(&searchType, "type", "t", "", "filter by type (file, dir, all)")

	versionCmd.Flags().BoolVar(&versionJSON, "json", false, "output version in JSON format")

	indexFilesCmd.Flags().IntVar(&filesLimit, "limit", 100, "maximum number of files to list")

	indexCmd.AddCommand(indexGenerateCmd)
	indexCmd.AddCommand(indexSyncCmd)
	indexCmd.AddCommand(indexStatusCmd)
	indexCmd.AddCommand(indexDirsCmd)
	indexCmd.AddCommand(indexCheckCmd)
	indexCmd.AddCommand(indexFilesCmd)

	watchCmd.AddCommand(watchStatusCmd)
	watchCmd.AddCommand(watchStartCmd)
	watchCmd.AddCommand(watchStopCmd)

	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(searchCmd)
	rootCmd.AddCommand(indexCmd)
	rootCmd.AddCommand(watchCmd)
	rootCmd.AddCommand(versionCmd)
}

func buildConfig() *config.Config {
	cfgPath := configFile
	if cfgPath == "" {
		cfgPath = config.GetDefaultConfigPath()
	}

	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	if indexPath != "" {
		cfg.IndexPath = indexPath
	}
	if listenAddr != "" {
		cfg.ListenAddr = listenAddr
	}
	if maxFileBytes > 0 {
		cfg.MaxFileBytes = maxFileBytes
	}
	if workerCount > 0 {
		cfg.WorkerCount = workerCount
	}
	cfg.BuildMaps()
	return cfg
}

func applyIndexPathOverrides(cmd *cobra.Command, cfg *config.Config) {
	excludeHiddenSet := cmd.Flags().Changed("exclude-hidden")

	if rootDir != "" {
		cfg.IndexPaths = []config.IndexPath{{
			Path:        rootDir,
			ExcludeDirs: config.DefaultExcludeDirs(),
		}}
	}

	for idx := range cfg.IndexPaths {
		if maxDepth >= 0 {
			cfg.IndexPaths[idx].MaxDepth = maxDepth
		}
		if excludeHiddenSet {
			cfg.IndexPaths[idx].ExcludeHidden = excludeHidden
		}
	}

	cfg.ExpandPaths()
	cfg.BuildMaps()
}

func runServe(cmd *cobra.Command, args []string) error {
	cfg := buildConfig()
	applyIndexPathOverrides(cmd, cfg)

	idx, err := indexer.New(cfg)
	if err != nil {
		return err
	}
	defer closeIndexOrGiveUp(idx, indexCloseTimeout)

	count, err := idx.GetDocCount()
	if err != nil {
		return err
	}

	if count == 0 || idx.NeedsReindex() {
		reason := "empty index"
		switch rebuild, _ := idx.RebuildReason(); {
		case rebuild != "":
			reason = "previous run reported: " + rebuild
		case count > 0:
			reason = "schema upgrade"
		}
		log.Infof("index requires rebuild (%s), reindexing...", reason)
		go func() {
			if err := idx.ReindexAll(); err != nil {
				log.Errorf("reindex failed: %v", err)
			} else {
				log.Infof("reindex complete")
			}
		}()
	}

	w := watcher.New(idx, cfg)

	if !noWatch {
		if err := w.Start(); err != nil {
			log.Errorf("failed to start watcher: %v", err)
			log.Infof("continuing without file watching")
		}
	}

	if enableHTTP && socketOnly {
		return fmt.Errorf("cannot specify both --http and --socket flags")
	}
	if listenAddr != "" && socketOnly {
		return fmt.Errorf("cannot specify both --listen and --socket flags")
	}

	runners := []app.Runner{app.RunnerFunc(func(ctx context.Context) error {
		select {
		case <-ctx.Done():
			return nil
		case err := <-idx.Fatal():
			return err
		}
	})}

	if httpRequested(cfg) {
		runners = append(runners, server.NewHTTP(cfg.ListenAddr, idx, w))
	}

	unixServer := server.NewUnix(server.NewRouter(idx, w, cfg))
	if err := unixServer.Listen(); err != nil {
		return err
	}
	runners = append(runners, unixServer)

	serveErr := app.Serve(context.Background(), runners...)

	if w.IsRunning() {
		if err := w.Stop(); err != nil {
			log.Errorf("failed to stop watcher: %v", err)
		}
	}

	return serveErr
}

func httpRequested(cfg *config.Config) bool {
	if socketOnly {
		return false
	}
	return cfg.HTTPEnabled || enableHTTP || listenAddr != ""
}

const indexCloseTimeout = 15 * time.Second

func closeIndexOrGiveUp(idx *indexer.Indexer, timeout time.Duration) {
	done := make(chan error, 1)
	go func() { done <- idx.Close() }()

	select {
	case err := <-done:
		if err != nil {
			log.Errorf("failed to close index: %v", err)
		}
	case <-time.After(timeout):
		log.Errorf("index close did not finish within %s, exiting anyway", timeout)
		_ = pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
	}
}

func runSearch(cmd *cobra.Command, args []string) error {
	query := "*"
	if len(args) > 0 {
		query = args[0]
	}

	limit := searchLimit
	if limit == 0 {
		limit = 10000
	}

	clientOpts := &client.SearchOptions{
		Query:           query,
		Limit:           limit,
		Field:           searchField,
		Extension:       searchExt,
		Fuzzy:           searchFuzzy,
		SortBy:          searchSort,
		SortDesc:        searchSortDesc,
		MinSize:         searchMinSize,
		MaxSize:         searchMaxSize,
		Folder:          searchFolder,
		ExifMake:        searchExifMake,
		ExifModel:       searchExifModel,
		ExifDateAfter:   searchExifDateAfter,
		ExifDateBefore:  searchExifDateBefore,
		ExifMinISO:      searchExifMinISO,
		ExifMaxISO:      searchExifMaxISO,
		ExifMinAperture: searchExifMinAp,
		ExifMaxAperture: searchExifMaxAp,
		ExifMinFocalLen: searchExifMinFocalLen,
		ExifMaxFocalLen: searchExifMaxFocalLen,
		ExifLatMin:      searchExifLatMin,
		ExifLatMax:      searchExifLatMax,
		ExifLonMin:      searchExifLonMin,
		ExifLonMax:      searchExifLonMax,
		XattrTags:       searchXattrTags,
		Type:            searchType,
	}

	result, err := client.SearchWithOptions(clientOpts)
	switch {
	case err == nil:
		if searchJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(result)
		}

		log.Infof("found %d results in %s", result.Total, result.Took)
		for i, hit := range result.Hits {
			log.Infof("%d. %s (score: %.4f)", i+1, hit.ID, hit.Score)
		}
		if len(result.DirectoryHits) > 0 {
			log.Infof("\ndirectories:")
			for i, hit := range result.DirectoryHits {
				log.Infof("%d. %s (score: %.4f)", i+1, hit.ID, hit.Score)
			}
		}
		return nil
	case !errors.Is(err, client.ErrServiceNotRunning):
		return err
	}

	idx, err := openLocalIndexer(buildConfig(), 3*time.Second)
	if err != nil {
		return err
	}
	defer idx.Close()

	count, err := idx.GetDocCount()
	if err != nil {
		return err
	}

	if count == 0 {
		return fmt.Errorf("index is empty - run 'dsearch index generate' or 'dsearch serve' first")
	}

	indexerOpts := &indexer.SearchOptions{
		Query:           query,
		Limit:           limit,
		Field:           searchField,
		Extension:       searchExt,
		Fuzzy:           searchFuzzy,
		SortBy:          searchSort,
		SortDesc:        searchSortDesc,
		MinSize:         searchMinSize,
		MaxSize:         searchMaxSize,
		Folder:          searchFolder,
		ExifMake:        searchExifMake,
		ExifModel:       searchExifModel,
		ExifDateAfter:   searchExifDateAfter,
		ExifDateBefore:  searchExifDateBefore,
		ExifMinISO:      searchExifMinISO,
		ExifMaxISO:      searchExifMaxISO,
		ExifMinAperture: searchExifMinAp,
		ExifMaxAperture: searchExifMaxAp,
		ExifMinFocalLen: searchExifMinFocalLen,
		ExifMaxFocalLen: searchExifMaxFocalLen,
		ExifLatMin:      searchExifLatMin,
		ExifLatMax:      searchExifLatMax,
		ExifLonMin:      searchExifLonMin,
		ExifLonMax:      searchExifLonMax,
		XattrTags:       searchXattrTags,
		Type:            searchType,
	}

	var idxResult *indexer.SearchResult
	if searchType == "all" {
		idxResult, err = idx.SearchAll(indexerOpts)
		if err != nil {
			return err
		}
	} else {
		bleveResult, err := idx.SearchWithOptions(indexerOpts)
		if err != nil {
			return err
		}
		idxResult = &indexer.SearchResult{SearchResult: bleveResult}
	}

	if searchJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(idxResult)
	}

	log.Infof("found %d results in %s", idxResult.Total, idxResult.Took)
	for i, hit := range idxResult.Hits {
		log.Infof("%d. %s (score: %.4f)", i+1, hit.ID, hit.Score)
	}
	if len(idxResult.DirectoryHits) > 0 {
		log.Infof("\ndirectories:")
		for i, hit := range idxResult.DirectoryHits {
			log.Infof("%d. %s (score: %.4f)", i+1, hit.ID, hit.Score)
		}
	}

	return nil
}

func runIndexGenerate(cmd *cobra.Command, args []string) error {
	status, err := client.Reindex()
	switch {
	case err == nil:
		log.Infof("%s", status)
		return nil
	case !errors.Is(err, client.ErrServiceNotRunning):
		return err
	}

	idx, err := openLocalIndexer(buildConfig(), 3*time.Second)
	if err != nil {
		return err
	}
	defer idx.Close()

	log.Infof("starting full reindex...")
	return idx.ReindexAll()
}

func runIndexSync(cmd *cobra.Command, args []string) error {
	status, err := client.Sync()
	switch {
	case err == nil:
		log.Infof("%s", status)
		return nil
	case !errors.Is(err, client.ErrServiceNotRunning):
		return err
	}

	idx, err := openLocalIndexer(buildConfig(), 3*time.Second)
	if err != nil {
		return err
	}
	defer idx.Close()

	log.Infof("starting incremental sync...")
	return idx.SyncIncremental()
}

func runIndexStatus(cmd *cobra.Command, args []string) error {
	stats, source, err := loadIndexStats()
	if err != nil {
		return err
	}
	printIndexStatus(stats, source)
	return nil
}

func loadIndexStats() (*config.IndexStats, string, error) {
	stats, err := client.Stats()
	switch {
	case err == nil:
		return stats, "server", nil
	case !errors.Is(err, client.ErrServiceNotRunning):
		return nil, "", err
	}

	idx, err := openLocalIndexer(buildConfig(), 3*time.Second)
	if err != nil {
		return nil, "", err
	}
	defer idx.Close()
	return idx.Stats(), "local", nil
}

func openLocalIndexer(cfg *config.Config, timeout time.Duration) (*indexer.Indexer, error) {
	type result struct {
		idx *indexer.Indexer
		err error
	}
	ch := make(chan result, 1)
	go func() {
		idx, err := indexer.New(cfg)
		ch <- result{idx, err}
	}()

	select {
	case r := <-ch:
		if r.err != nil {
			return nil, fmt.Errorf("server not running and cannot open index: %v", r.err)
		}
		return r.idx, nil
	case <-time.After(timeout):
		return nil, fmt.Errorf("server not running and index is locked by another process (likely the daemon starting up) — try again in a moment")
	}
}

func printIndexStatus(stats *config.IndexStats, source string) {
	log.Infof("Index Statistics (%s):", source)

	schema := fmt.Sprintf("v%d", stats.ExpectedSchemaVersion)
	switch {
	case stats.SchemaVersion == 0:
		schema = fmt.Sprintf("v%d (uninitialized)", stats.ExpectedSchemaVersion)
	case stats.SchemaVersion != stats.ExpectedSchemaVersion:
		schema = fmt.Sprintf("v%d → v%d (migrating)", stats.SchemaVersion, stats.ExpectedSchemaVersion)
	}
	log.Infof("  Schema:      %s", schema)

	switch stats.Phase {
	case "", indexer.PhaseIdle:
		log.Infof("  Status:      idle")
	default:
		elapsed := time.Since(stats.PhaseStartedAt).Round(time.Second)
		log.Infof("  Status:      %s (%s elapsed)", stats.Phase, elapsed)
		log.Infof("  Progress:    %d files, %s processed",
			stats.FilesProcessed, formatBytes(stats.BytesProcessed))
	}

	log.Infof("  Total files: %d", stats.TotalFiles)
	log.Infof("  Total size:  %s", formatBytes(stats.TotalBytes))

	switch {
	case stats.LastIndexTime.IsZero():
		log.Infof("  Last index:  never")
	default:
		ago := time.Since(stats.LastIndexTime).Round(time.Second)
		log.Infof("  Last index:  %s (%s ago)",
			stats.LastIndexTime.Local().Format("2006-01-02 15:04:05"), ago)
	}

	if stats.IndexDuration != "" {
		log.Infof("  Duration:    %s", stats.IndexDuration)
	}
}

func formatBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

func runIndexDirs(cmd *cobra.Command, args []string) error {
	cfg := buildConfig()
	for _, ip := range cfg.IndexPaths {
		depth := "unlimited"
		if ip.MaxDepth > 0 {
			depth = fmt.Sprintf("%d", ip.MaxDepth)
		}
		hidden := "included"
		if ip.ExcludeHidden {
			hidden = "excluded"
		}
		watch := "on"
		if !ip.ShouldWatch() {
			watch = "off"
		}
		exif := "off"
		if ip.ShouldExtractExif() {
			exif = "on"
		}
		fmt.Printf("%s (depth: %s, hidden: %s, watch: %s, exif: %s)\n",
			ip.Path, depth, hidden, watch, exif)
		if len(ip.ExcludeDirs) > 0 {
			fmt.Printf("  exclude: %s\n", strings.Join(ip.ExcludeDirs, ", "))
		}
	}
	return nil
}

func runIndexCheck(cmd *cobra.Command, args []string) error {
	path, err := filepath.Abs(args[0])
	if err != nil {
		return err
	}

	cfg := buildConfig()
	idxPath := cfg.FindIndexPath(path)

	fmt.Printf("%s\n", path)

	if idxPath == nil {
		fmt.Printf("  index path: none\n")
		fmt.Printf("  config:     excluded (not under any configured index path)\n")
		return nil
	}
	fmt.Printf("  index path: %s\n", idxPath.Path)

	reason := cfg.ExclusionReason(path)
	if reason != "" {
		fmt.Printf("  config:     excluded (%s)\n", reason)
		return nil
	}
	fmt.Printf("  config:     included\n")

	info, statErr := os.Stat(path)
	if statErr == nil && info.IsDir() {
		depth := cfg.GetDepth(path)
		maxDepth := cfg.GetMaxDepth(path)
		if maxDepth > 0 {
			fmt.Printf("  depth:      %d (max: %d)\n", depth, maxDepth)
		} else {
			fmt.Printf("  depth:      %d (unlimited)\n", depth)
		}
		return nil
	}

	store, err := metastore.New(cfg.IndexPath)
	if err != nil {
		fmt.Printf("  indexed:    unknown (cannot open metastore: %v)\n", err)
		return nil
	}
	defer store.Close()

	meta, found, err := store.Get(path)
	if err != nil {
		fmt.Printf("  indexed:    unknown (metastore error: %v)\n", err)
		return nil
	}
	if found {
		fmt.Printf("  indexed:    yes (%s, %s)\n",
			meta.ModTime.Format("2006-01-02 15:04:05"), formatSize(meta.Size))
	} else {
		fmt.Printf("  indexed:    no\n")
	}
	return nil
}

func runIndexFiles(cmd *cobra.Command, args []string) error {
	prefix := ""
	if len(args) > 0 {
		prefix = args[0]
	}

	result, err := client.Files(prefix, filesLimit)
	switch {
	case err == nil:
		for _, f := range result.Files {
			t, _ := time.Parse(time.RFC3339, f.ModTime)
			fmt.Printf("%s (%s, %s)\n", f.Path, t.Format("2006-01-02"), formatSize(f.Size))
		}
		if result.Total > filesLimit {
			fmt.Printf("(showing %d of %d files)\n", len(result.Files), result.Total)
		}
		return nil
	case !errors.Is(err, client.ErrServiceNotRunning):
		return err
	}

	idx, err := openLocalIndexer(buildConfig(), 3*time.Second)
	if err != nil {
		return err
	}
	defer idx.Close()

	files, total, err := idx.ListFiles(prefix, filesLimit)
	if err != nil {
		return err
	}

	for _, f := range files {
		fmt.Printf("%s (%s, %s)\n", f.Path, f.ModTime.Format("2006-01-02"), formatSize(f.Size))
	}
	if total > filesLimit {
		fmt.Printf("(showing %d of %d files)\n", len(files), total)
	}
	return nil
}

func formatSize(bytes int64) string {
	switch {
	case bytes >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(1<<30))
	case bytes >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(1<<20))
	case bytes >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(1<<10))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func runWatchStatus(cmd *cobra.Command, args []string) error {
	status, err := client.WatchStatus()
	if err != nil {
		return err
	}

	log.Infof("Watcher status: %s", status)
	return nil
}

func runWatchStart(cmd *cobra.Command, args []string) error {
	status, err := client.WatchStart()
	if err != nil {
		return err
	}

	log.Infof("%s", status)
	return nil
}

func runWatchStop(cmd *cobra.Command, args []string) error {
	status, err := client.WatchStop()
	if err != nil {
		return err
	}

	log.Infof("%s", status)
	return nil
}

func main() {
	log.SetEnvPrefix("DANKSEARCH")
	if os.Getenv("DANKSEARCH_DEBUG") != "" {
		log.SetLevel("debug")
	}
	if err := config.MigrateLegacyDirs(); err != nil {
		log.Warnf("dsearch dir migration: %v", err)
	}
	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("%v", err)
	}
}
