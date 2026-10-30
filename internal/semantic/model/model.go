// Package model installs, inspects and removes the pinned EmbeddingGemma 2
// ONNX model on disk. Status never hashes the large files: the full sha256
// is computed once while a file is downloaded (or adopted) and recorded in a
// small marker file next to the model; later runs compare the file's size and
// modification time with that record.
package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"golang.org/x/sys/unix"
)

const (
	markerName = ".dux-model.json"
	lockName   = ".dux-model.lock"
	partSuffix = ".part"
)

var (
	// ErrBusy means another Download or Remove holds the model directory.
	ErrBusy = errors.New("model directory is busy: another download or removal is running")
	// ErrUnknownTower rejects a tower name that is not text, image or audio.
	ErrUnknownTower = errors.New("unknown model tower")
)

// FileStatus describes one pinned file on disk.
type FileStatus struct {
	Path     string `json:"path"`
	Size     int64  `json:"size"`     // bytes on disk (a partial download counts)
	Expected int64  `json:"expected"` // bytes in the pinned manifest
	State    string `json:"state"`
}

// File states reported by Status.
const (
	FileMissing    = "missing"    // nothing on disk
	FilePartial    = "partial"    // interrupted download, resumable
	FileUnverified = "unverified" // right size, no matching verification record
	FileVerified   = "verified"   // right size and sha256 verified at install time
	FileCorrupt    = "corrupt"    // present with the wrong size
)

// TowerStatus is the state of one tower.
type TowerStatus struct {
	Tower     semantic.Modality `json:"tower"`
	Installed bool              `json:"installed"` // every file present and verified
	Bytes     int64             `json:"bytes"`     // on disk, partial downloads included
	Total     int64             `json:"total"`     // size of the pinned files
	Partial   bool              `json:"partial"`   // an interrupted download can be resumed
	Files     []FileStatus      `json:"files"`
}

// ModelStatus is the result of Status.
type ModelStatus struct {
	Dir      string        `json:"dir"`
	ModelID  string        `json:"model_id"`
	Revision string        `json:"revision"`
	Ready    bool          `json:"ready"` // the mandatory text tower is installed
	Towers   []TowerStatus `json:"towers"`
}

// Tower returns the status of one tower.
func (s ModelStatus) Tower(t semantic.Modality) (TowerStatus, bool) {
	for _, ts := range s.Towers {
		if ts.Tower == t {
			return ts, true
		}
	}
	return TowerStatus{}, false
}

// Installed lists the towers that are complete and verified.
func (s ModelStatus) Installed() []semantic.Modality {
	var out []semantic.Modality
	for _, ts := range s.Towers {
		if ts.Installed {
			out = append(out, ts.Tower)
		}
	}
	return out
}

type markerEntry struct {
	Size    int64  `json:"size"`
	MTimeNS int64  `json:"mtime_ns"`
	SHA256  string `json:"sha256"`
}

type marker struct {
	Version int                    `json:"version"`
	Files   map[string]markerEntry `json:"files"`
}

func readMarker(dir string) marker {
	m := marker{Version: 1, Files: map[string]markerEntry{}}
	raw, err := os.ReadFile(filepath.Join(dir, markerName))
	if err != nil {
		return m
	}
	var got marker
	if json.Unmarshal(raw, &got) != nil || got.Version != 1 || got.Files == nil {
		return m
	}
	return got
}

func writeMarker(dir string, m marker) error {
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(filepath.Join(dir, markerName), raw, 0o644)
}

// writeFileAtomic writes through a temporary name, fsyncs and renames.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	fail := func(err error) error {
		tmp.Close()
		os.Remove(name)
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		return fail(err)
	}
	if err := tmp.Chmod(perm); err != nil {
		return fail(err)
	}
	if err := tmp.Sync(); err != nil {
		return fail(err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, path); err != nil {
		os.Remove(name)
		return err
	}
	return syncDir(filepath.Dir(path))
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

func localPath(dir string, f File) string { return filepath.Join(dir, filepath.FromSlash(f.Path)) }

// verified reports whether the marker vouches for the file on disk.
func (m marker) verified(f File, info fs.FileInfo) bool {
	e, ok := m.Files[f.Path]
	return ok && e.SHA256 == f.SHA256 && e.Size == f.Size && info.Size() == f.Size && e.MTimeNS == info.ModTime().UnixNano()
}

func fileStatus(dir string, f File, m marker) FileStatus {
	st := FileStatus{Path: f.Path, Expected: f.Size, State: FileMissing}
	p := localPath(dir, f)
	if info, err := os.Stat(p); err == nil && info.Mode().IsRegular() {
		st.Size = info.Size()
		switch {
		case info.Size() != f.Size:
			st.State = FileCorrupt
		case m.verified(f, info):
			st.State = FileVerified
		default:
			st.State = FileUnverified
		}
		return st
	}
	if info, err := os.Stat(p + partSuffix); err == nil && info.Mode().IsRegular() && info.Size() <= f.Size {
		st.Size = info.Size()
		st.State = FilePartial
	}
	return st
}

// Status reports which towers are complete and verified. It only stats files
// and reads the verification marker; it never hashes a model file.
func Status(dir string) ModelStatus {
	return statusOf(dir, nil)
}

func statusOf(dir string, files func(semantic.Modality) []File) ModelStatus {
	if files == nil {
		files = Files
	}
	m := readMarker(dir)
	out := ModelStatus{Dir: dir, ModelID: ModelID, Revision: onnxRevision}
	for _, t := range allTowers {
		ts := TowerStatus{Tower: t, Installed: true}
		for _, f := range files(t) {
			fs := fileStatus(dir, f, m)
			ts.Files = append(ts.Files, fs)
			ts.Total += f.Size
			ts.Bytes += fs.Size
			ts.Partial = ts.Partial || fs.State == FilePartial
			ts.Installed = ts.Installed && fs.State == FileVerified
		}
		out.Towers = append(out.Towers, ts)
		if t == semantic.ModalityText {
			out.Ready = ts.Installed
		}
	}
	return out
}

// normalizeTowers validates, de-duplicates and orders towers.
func normalizeTowers(towers []semantic.Modality) ([]semantic.Modality, error) {
	for _, t := range towers {
		if !slices.Contains(allTowers, t) {
			return nil, fmt.Errorf("%w %q", ErrUnknownTower, string(t))
		}
	}
	var out []semantic.Modality
	for _, t := range allTowers {
		if slices.Contains(towers, t) {
			out = append(out, t)
		}
	}
	return out, nil
}

// lockDir takes the exclusive advisory lock of a model directory.
func lockDir(dir string) (unlock func(), err error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(dir, lockName), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		f.Close()
		if errors.Is(err, unix.EWOULDBLOCK) {
			return nil, ErrBusy
		}
		return nil, err
	}
	// Closing the descriptor releases the lock even if the explicit unlock failed.
	return func() { _ = unix.Flock(int(f.Fd()), unix.LOCK_UN); f.Close() }, nil
}

// Remove deletes the files of the given towers, partial downloads included.
// Removing the text tower removes the whole model: the image and the audio
// towers feed the text model and cannot work without it, and the shared
// tokenizer and config files go with it. Files that are not part of the model
// are left alone, and a directory is removed only when it ends up empty.
func Remove(dir string, towers []semantic.Modality) error {
	return removeFiles(dir, towers, Files)
}

func removeFiles(dir string, towers []semantic.Modality, files func(semantic.Modality) []File) error {
	if dir == "" {
		return errors.New("model directory is empty")
	}
	towers, err := normalizeTowers(towers)
	if err != nil {
		return err
	}
	if len(towers) == 0 {
		return nil
	}
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	unlock, err := lockDir(dir)
	if err != nil {
		return err
	}
	defer unlock()

	if slices.Contains(towers, semantic.ModalityText) {
		towers = allTowers
	}
	m := readMarker(dir)
	var firstErr error
	for _, t := range towers {
		for _, f := range files(t) {
			p := localPath(dir, f)
			for _, name := range []string{p, p + partSuffix} {
				if err := os.Remove(name); err != nil && !errors.Is(err, fs.ErrNotExist) && firstErr == nil {
					firstErr = err
				}
			}
			delete(m.Files, f.Path)
		}
	}
	prune(dir, files)
	if len(m.Files) == 0 {
		os.Remove(filepath.Join(dir, markerName))
	} else if err := writeMarker(dir, m); err != nil && firstErr == nil {
		firstErr = err
	}
	if len(readDirNames(dir)) <= 1 { // only the lock file is left
		os.Remove(filepath.Join(dir, lockName))
	}
	os.Remove(dir) // fails unless empty, which is intended
	return firstErr
}

// prune removes the model's subdirectories once they are empty.
func prune(dir string, files func(semantic.Modality) []File) {
	seen := map[string]bool{}
	for _, t := range allTowers {
		for _, f := range files(t) {
			d := filepath.Dir(f.Path)
			for d != "." && d != "/" && !seen[d] {
				seen[d] = true
				d = filepath.Dir(d)
			}
		}
	}
	dirs := make([]string, 0, len(seen))
	for d := range seen {
		dirs = append(dirs, d)
	}
	slices.SortFunc(dirs, func(a, b string) int { return strings.Count(b, "/") - strings.Count(a, "/") })
	for _, d := range dirs {
		os.Remove(filepath.Join(dir, filepath.FromSlash(d)))
	}
}

func readDirNames(dir string) []string {
	ents, _ := os.ReadDir(dir)
	var out []string
	for _, e := range ents {
		out = append(out, e.Name())
	}
	return out
}
