package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/client"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
	"github.com/spf13/cobra"
)

var (
	searchSemantic        bool
	searchCollections     []string
	searchCollectionLimit int

	collectionsJSON string
	statusJSON      bool
	warmTowers      []string
	modelAll        bool
)

var collectionsCmd = &cobra.Command{
	Use:   "collections",
	Short: "Manage semantic collections (needs the running service)",
	Long: `Collections hold caller-owned items (id, title, text, keywords) that are
searched by meaning. The launcher keeps its actions, settings and apps in them.`,
}

var collectionsSetCmd = &cobra.Command{
	Use:   "set <name>",
	Short: "Replace a collection with the items read from a JSON file or stdin",
	Long: `Replaces the whole collection. The JSON is an array of items, or an object with an
"items" array:

  [{"id":"dark","title":"Dark theme","text":"Switch between dark and light mode","keywords":["night"]}]

Items whose content did not change are not embedded again. Without a model the
items are stored and embedded as soon as it is installed.`,
	Example: `  dsearch collections set shell.actions --json actions.json
  cat actions.json | dsearch collections set shell.actions --json -`,
	Args: cobra.ExactArgs(1),
	RunE: runCollectionsSet,
}

var collectionsDeleteCmd = &cobra.Command{
	Use:   "delete <name>",
	Short: "Delete a collection",
	Args:  cobra.ExactArgs(1),
	RunE:  runCollectionsDelete,
}

var semanticCmd = &cobra.Command{
	Use:   "semantic",
	Short: "Semantic search service control",
}

var semanticStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show embedder, model, collections and background indexing",
	Args:  cobra.NoArgs,
	RunE:  runSemanticStatus,
}

var semanticWarmCmd = &cobra.Command{
	Use:   "warm",
	Short: "Load the model into memory ahead of use",
	Args:  cobra.NoArgs,
	RunE:  runSemanticWarm,
}

var modelCmd = &cobra.Command{
	Use:   "model",
	Short: "Manage the embedding model files",
	Long: `The model has three towers: text (needed for any semantic search), image and
audio. Files are downloaded from Hugging Face at pinned revisions and verified
against pinned SHA-256 sums.`,
}

var modelDownloadCmd = &cobra.Command{
	Use:       "download [text|image|audio]...",
	Short:     "Download model towers (default: text)",
	ValidArgs: []string{"text", "image", "audio"},
	RunE:      runModelDownload,
}

var modelRemoveCmd = &cobra.Command{
	Use:       "remove [text|image|audio]...",
	Short:     "Delete model towers (default: text)",
	ValidArgs: []string{"text", "image", "audio"},
	RunE:      runModelRemove,
}

var modelStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show which towers are installed",
	Args:  cobra.NoArgs,
	RunE:  runModelStatus,
}

func init() {
	searchCmd.Flags().BoolVar(&searchSemantic, "semantic", false, "also search by meaning and merge file results (needs the running service with semantic search enabled)")
	searchCmd.Flags().StringSliceVar(&searchCollections, "collections", nil, "collections to search by meaning, comma separated (implies --semantic)")
	searchCmd.Flags().IntVar(&searchCollectionLimit, "collection-limit", 0, "hits per collection (default 8)")

	collectionsSetCmd.Flags().StringVar(&collectionsJSON, "json", "-", "JSON file with the items, or - for stdin")
	collectionsCmd.AddCommand(collectionsSetCmd, collectionsDeleteCmd)

	semanticStatusCmd.Flags().BoolVar(&statusJSON, "json", false, "output JSON")
	semanticWarmCmd.Flags().StringSliceVar(&warmTowers, "tower", nil, "towers to load: text, image, audio (default text)")
	semanticCmd.AddCommand(semanticStatusCmd, semanticWarmCmd)

	modelStatusCmd.Flags().BoolVar(&statusJSON, "json", false, "output JSON")
	modelDownloadCmd.Flags().BoolVar(&modelAll, "all", false, "download every tower")
	modelRemoveCmd.Flags().BoolVar(&modelAll, "all", false, "remove every tower")
	modelCmd.AddCommand(modelDownloadCmd, modelRemoveCmd, modelStatusCmd)

	rootCmd.AddCommand(collectionsCmd, semanticCmd, modelCmd)
}

func serviceRequired(err error) error {
	if errors.Is(err, client.ErrServiceNotRunning) {
		return errors.New("the dsearch service is not running (start it with `dsearch serve`)")
	}
	return err
}

func readItems(path string) ([]semantic.Item, error) {
	var r io.Reader = os.Stdin
	if path != "-" {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		r = f
	}
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return parseItems(raw)
}

// parseItems accepts an array of items or {"items": [...]}.
func parseItems(raw []byte) ([]semantic.Item, error) {
	var wrapped struct {
		Items []semantic.Item `json:"items"`
	}
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return nil, errors.New("no JSON given: pass --json <file> or pipe it to --json -")
	}
	if strings.HasPrefix(trimmed, "{") {
		if err := json.Unmarshal(raw, &wrapped); err != nil {
			return nil, fmt.Errorf("invalid items JSON: %w", err)
		}
		if wrapped.Items == nil {
			return nil, errors.New(`invalid items JSON: object without an "items" array`)
		}
		return wrapped.Items, nil
	}
	var items []semantic.Item
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("invalid items JSON: %w", err)
	}
	return items, nil
}

func runCollectionsSet(cmd *cobra.Command, args []string) error {
	items, err := readItems(collectionsJSON)
	if err != nil {
		return err
	}
	res, err := client.CollectionsSet(args[0], items)
	if err != nil {
		return serviceRequired(err)
	}
	fmt.Printf("%s: %d items, %d embedded now, %d removed, %d waiting for the model\n",
		res.Collection, res.Total, res.Embedded, res.Removed, res.Pending)
	return nil
}

func runCollectionsDelete(cmd *cobra.Command, args []string) error {
	res, err := client.CollectionsDelete(args[0])
	if err != nil {
		return serviceRequired(err)
	}
	fmt.Printf("%s: deleted (%d vectors)\n", res.Collection, res.Removed)
	return nil
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func runSemanticStatus(cmd *cobra.Command, args []string) error {
	st, err := client.SemanticStatus()
	if err != nil {
		return serviceRequired(err)
	}
	if statusJSON {
		return printJSON(st)
	}
	if !st.Enabled {
		fmt.Println("Semantic search: disabled")
		if st.Embedder.Reason != service.ErrDisabled.Error() { // off for a reason beyond the config
			fmt.Printf("  Reason:      %s\n", st.Embedder.Reason)
		} else {
			fmt.Println("  Enable it with `enabled = true` under [semantic] in the config and restart dsearch.")
		}
	} else {
		fmt.Println("Semantic search: enabled")
		switch {
		case !st.Embedder.Available:
			fmt.Printf("  Embedder:    unavailable (%s)\n", st.Embedder.Reason)
		case st.Embedder.Running:
			fmt.Printf("  Embedder:    running, loaded: %s\n", joinModalities(st.Embedder.Loaded))
		default:
			fmt.Println("  Embedder:    ready (model not loaded)")
		}
		names := make([]string, 0, len(st.Collections))
		for n := range st.Collections {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, n := range names {
			line := fmt.Sprintf("  Collection:  %-16s %d vectors", n, st.Collections[n])
			if p := st.Pending[n]; p > 0 {
				line += fmt.Sprintf(", %d waiting to be embedded", p)
			}
			fmt.Println(line)
		}
		idx := fmt.Sprintf("%d pending, %d done", st.Indexing.Pending, st.Indexing.Done)
		if st.Indexing.PausedReason != "" {
			idx += " (paused: " + st.Indexing.PausedReason + ")"
		}
		fmt.Println("  File index: ", idx)
	}
	printDownload(st.Download)
	return nil
}

func joinModalities(m []semantic.Modality) string {
	if len(m) == 0 {
		return "none"
	}
	parts := make([]string, len(m))
	for i, x := range m {
		parts[i] = string(x)
	}
	return strings.Join(parts, ", ")
}

func printDownload(d service.DownloadState) {
	if !d.Active && len(d.Towers) == 0 && d.Error == "" {
		return
	}
	state := "last download"
	if d.Active {
		state = "downloading"
	}
	fmt.Printf("  Model:       %s\n", state)
	towers := make([]string, 0, len(d.Towers))
	for t := range d.Towers {
		towers = append(towers, string(t))
	}
	sort.Strings(towers)
	for _, t := range towers {
		p := d.Towers[semantic.Modality(t)]
		fmt.Printf("    %-6s %s\n", t, describeProgress(p))
	}
	if d.Error != "" {
		fmt.Printf("    error: %s\n", d.Error)
	}
}

func describeProgress(p service.ModelProgress) string {
	if p.Total > 0 {
		return fmt.Sprintf("%s  %s / %s (%d%%)", p.State, formatBytes(p.Bytes), formatBytes(p.Total), p.Bytes*100/p.Total)
	}
	return p.State
}

func runSemanticWarm(cmd *cobra.Command, args []string) error {
	towers, err := towerNames(warmTowers, false)
	if err != nil {
		return err
	}
	start := time.Now()
	st, err := client.SemanticWarm(towers)
	if err != nil {
		return serviceRequired(err)
	}
	fmt.Printf("warm in %s, loaded: %s\n", time.Since(start).Round(time.Millisecond), joinModalities(st.Embedder.Loaded))
	return nil
}

// towerNames validates tower arguments; with none given the default is text, or
// every tower with --all.
func towerNames(args []string, all bool) ([]string, error) {
	if all {
		return []string{"text", "image", "audio"}, nil
	}
	if len(args) == 0 {
		return []string{"text"}, nil
	}
	mods, err := service.ParseModalities(args)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(mods))
	for i, m := range mods {
		out[i] = string(m)
	}
	return out, nil
}

func runModelDownload(cmd *cobra.Command, args []string) error {
	towers, err := towerNames(args, modelAll)
	if err != nil {
		return err
	}
	cfg := buildConfig()
	progress := newProgressPrinter(os.Stderr)
	defer progress.finish()

	err = client.ModelDownload(towers, progress.update)
	switch {
	case err == nil:
		progress.finish()
		fmt.Printf("model towers ready: %s\n", strings.Join(towers, ", "))
		return nil
	case !errors.Is(err, client.ErrServiceNotRunning):
		return err
	}

	// No daemon: download straight into the model directory.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	mods, _ := service.ParseModalities(towers)
	if err := newModelManager(cfg).Download(ctx, mods, progress.update); err != nil {
		return err
	}
	progress.finish()
	fmt.Printf("model towers ready in %s: %s\n", cfg.Semantic.ModelDir, strings.Join(towers, ", "))
	return nil
}

func runModelRemove(cmd *cobra.Command, args []string) error {
	towers, err := towerNames(args, modelAll)
	if err != nil {
		return err
	}
	err = client.ModelRemove(towers)
	switch {
	case err == nil:
	case errors.Is(err, client.ErrServiceNotRunning):
		mods, _ := service.ParseModalities(towers)
		if err := newModelManager(buildConfig()).Remove(mods); err != nil {
			return err
		}
	default:
		return err
	}
	fmt.Printf("removed: %s\n", strings.Join(towers, ", "))
	return nil
}

func runModelStatus(cmd *cobra.Command, args []string) error {
	var model any
	var download service.DownloadState
	source := "service"
	res, err := client.ModelStatus()
	switch {
	case err == nil:
		var m any
		if len(res.Model) > 0 {
			_ = json.Unmarshal(res.Model, &m)
		}
		model, download = m, res.Download
	case errors.Is(err, client.ErrServiceNotRunning):
		source = "files"
		model = newModelManager(buildConfig()).Status()
	default:
		return err
	}
	if statusJSON {
		return printJSON(map[string]any{"model": model, "download": download, "source": source})
	}
	cfg := buildConfig()
	fmt.Printf("Model directory: %s (read from %s)\n", cfg.Semantic.ModelDir, source)
	if err := printJSONIndented(model); err != nil {
		return err
	}
	printDownload(download)
	return nil
}

func printJSONIndented(v any) error {
	raw, err := json.MarshalIndent(v, "  ", "  ")
	if err != nil {
		return err
	}
	fmt.Printf("  %s\n", raw)
	return nil
}

// progressPrinter renders model.progress events: a live line on a terminal, one
// line per state change otherwise.
type progressPrinter struct {
	w     io.Writer
	tty   bool
	last  map[semantic.Modality]string
	dirty bool
}

func newProgressPrinter(w *os.File) *progressPrinter {
	fi, err := w.Stat()
	return &progressPrinter{w: w, tty: err == nil && fi.Mode()&os.ModeCharDevice != 0, last: map[semantic.Modality]string{}}
}

func (p *progressPrinter) update(ev service.ModelProgress) {
	if ev.Tower == "" { // end-of-download marker
		return
	}
	line := fmt.Sprintf("%-6s %s", ev.Tower, describeProgress(ev))
	if p.tty {
		fmt.Fprintf(p.w, "\r\033[K%s", line)
		p.dirty = true
		return
	}
	if p.last[ev.Tower] != ev.State {
		p.last[ev.Tower] = ev.State
		fmt.Fprintln(p.w, line)
	}
}

func (p *progressPrinter) finish() {
	if p.dirty {
		fmt.Fprintln(p.w)
		p.dirty = false
	}
}

// printSemanticResult prints the semantic parts of a search response.
func printSemanticResult(res *client.SearchResult) {
	if res.Semantic != nil && !res.Semantic.Available {
		fmt.Fprintf(os.Stderr, "semantic search unavailable: %s (lexical results only)\n", res.Semantic.Reason)
	}
	if res.LexicalError != "" {
		fmt.Fprintf(os.Stderr, "lexical search failed: %s\n", res.LexicalError)
	}
	for _, c := range res.Collections {
		fmt.Printf("\ncollection %s:\n", c.Collection)
		for i, h := range c.Hits {
			fmt.Printf("%d. %s (similarity: %.4f)\n", i+1, h.ID, h.Score)
		}
		if len(c.Hits) == 0 {
			fmt.Println("(no hits)")
		}
	}
}
