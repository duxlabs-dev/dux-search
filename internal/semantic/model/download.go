package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"golang.org/x/sys/unix"
)

// Progress states.
const (
	StateDownloading = "downloading"
	StateVerifying   = "verifying"
	StateDone        = "done"
	StateError       = "error"
)

// Progress is reported per tower. Bytes and Total cover the tower's files; the
// text tower includes the shared tokenizer and config files.
type Progress struct {
	Tower semantic.Modality `json:"tower"`
	File  string            `json:"file,omitempty"`
	Bytes int64             `json:"bytes"`
	Total int64             `json:"total"`
	State string            `json:"state"`
}

var (
	// ErrNetwork wraps every failure to reach or read from the download host.
	ErrNetwork = errors.New("network error")
	// ErrChecksum means a complete download does not match the pinned sha256.
	ErrChecksum = errors.New("sha256 mismatch")
	// ErrInsufficientSpace means the disk cannot hold the download plus margin.
	ErrInsufficientSpace = errors.New("not enough free disk space")
)

// SpaceError carries the numbers behind ErrInsufficientSpace.
type SpaceError struct {
	Dir  string
	Need int64 // bytes still to download plus the safety margin
	Free int64
}

func (e *SpaceError) Error() string {
	return fmt.Sprintf("%v: %s needs %d MiB, %d MiB free", ErrInsufficientSpace, e.Dir, e.Need>>20, e.Free>>20)
}
func (e *SpaceError) Is(target error) bool { return target == ErrInsufficientSpace }

// NetworkError says which file failed and after how many attempts.
type NetworkError struct {
	File     string
	Attempts int
	Err      error
}

func (e *NetworkError) Error() string {
	return fmt.Sprintf("download of %s failed after %d attempt(s): %v", e.File, e.Attempts, e.Err)
}
func (e *NetworkError) Unwrap() []error { return []error{ErrNetwork, e.Err} }

// HTTPStatusError is a permanent server answer such as 404.
type HTTPStatusError struct {
	URL    string
	Status int
}

func (e *HTTPStatusError) Error() string { return fmt.Sprintf("%s: HTTP %d", e.URL, e.Status) }

var errStalled = errors.New("no data received for too long")

type downloader struct {
	client   *http.Client
	baseURL  string
	retries  int // attempts per file after the first
	backoff  func(attempt int) time.Duration
	stall    time.Duration
	freeFn   func(dir string) (uint64, error)
	margin   func(need int64) int64
	interval time.Duration // minimum time between progress callbacks
	files    func(semantic.Modality) []File
}

func newDownloader() *downloader {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.DisableCompression = true // Range and Content-Length must refer to the stored bytes
	tr.ResponseHeaderTimeout = 30 * time.Second
	return &downloader{
		client:  &http.Client{Transport: tr},
		baseURL: "https://huggingface.co",
		retries: 6,
		backoff: func(attempt int) time.Duration {
			d := min(time.Second<<attempt, 30*time.Second)
			return d/2 + time.Duration(rand.Int64N(int64(d/2)+1))
		},
		stall:    30 * time.Second,
		freeFn:   freeSpace,
		margin:   func(need int64) int64 { return 64<<20 + need/20 },
		interval: 100 * time.Millisecond,
		files:    Files,
	}
}

func freeSpace(dir string) (uint64, error) {
	for {
		var st unix.Statfs_t
		err := unix.Statfs(dir, &st)
		if err == nil {
			return uint64(st.Bavail) * uint64(st.Bsize), nil // field types differ per OS
		}
		parent := filepath.Dir(dir)
		if !errors.Is(err, unix.ENOENT) || parent == dir {
			return 0, err
		}
		dir = parent
	}
}

// Download installs the given towers into dir. The text tower is mandatory
// and is added when missing; image and audio are optional. Downloads resume
// from a ".part" file with a Range request, every file is verified against its
// pinned sha256 and moved into place atomically after an fsync, so dir only
// ever holds complete, verified files. progress may be nil; it is called from
// the downloading goroutine.
func Download(ctx context.Context, dir string, towers []semantic.Modality, progress func(Progress)) error {
	return newDownloader().download(ctx, dir, towers, progress)
}

func (d *downloader) download(ctx context.Context, dir string, towers []semantic.Modality, progress func(Progress)) error {
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
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	unlock, err := lockDir(dir)
	if err != nil {
		return err
	}
	defer unlock()

	m := readMarker(dir)
	if st := statusOf(dir, d.files); !st.Ready && !containsTower(towers, semantic.ModalityText) {
		towers = append([]semantic.Modality{semantic.ModalityText}, towers...)
	}

	// Disk space: what is still missing, plus a margin.
	var need int64
	for _, t := range towers {
		for _, f := range d.files(t) {
			need += remaining(dir, f, m)
		}
	}
	if need > 0 {
		margin := d.margin(need)
		free, err := d.freeFn(dir)
		if err != nil {
			return fmt.Errorf("cannot determine free disk space of %s: %w", dir, err)
		}
		if int64(min(free, 1<<62)) < need+margin {
			return &SpaceError{Dir: dir, Need: need + margin, Free: int64(min(free, 1<<62))}
		}
	}

	for _, t := range towers {
		tp := &towerProgress{tower: t, report: progress, interval: d.interval}
		files := d.files(t)
		for _, f := range files {
			tp.total += f.Size
		}
		for _, f := range files {
			tp.file = f.Path
			err := d.fetch(ctx, dir, f, &m, tp)
			if err != nil {
				tp.emit(StateError, true)
				return err
			}
			tp.base += f.Size
			tp.cur = 0
		}
		tp.file = ""
		tp.emit(StateDone, true)
	}
	return nil
}

func containsTower(ts []semantic.Modality, t semantic.Modality) bool {
	for _, x := range ts {
		if x == t {
			return true
		}
	}
	return false
}

// remaining estimates the bytes that still have to be downloaded for f.
func remaining(dir string, f File, m marker) int64 {
	p := localPath(dir, f)
	if info, err := os.Stat(p); err == nil && info.Size() == f.Size {
		return 0 // present: at worst it is hashed, never downloaded again
	}
	if info, err := os.Stat(p + partSuffix); err == nil && info.Size() <= f.Size {
		return f.Size - info.Size()
	}
	return f.Size
}

type towerProgress struct {
	tower    semantic.Modality
	file     string
	total    int64
	base     int64 // bytes of the finished files
	cur      int64 // bytes of the current file
	report   func(Progress)
	interval time.Duration
	last     time.Time
}

func (p *towerProgress) set(cur int64) {
	p.cur = cur
	p.emit(StateDownloading, false)
}

// adopted counts a file that was already on disk; it was verified, not downloaded.
func (p *towerProgress) adopted(size int64) {
	p.cur = size
	p.emit(StateVerifying, true)
}

func (p *towerProgress) emit(state string, force bool) {
	if p.report == nil {
		return
	}
	now := time.Now()
	if !force && now.Sub(p.last) < p.interval {
		return
	}
	p.last = now
	p.report(Progress{Tower: p.tower, File: p.file, Bytes: p.base + p.cur, Total: p.total, State: state})
}

// fetch makes f present and verified in dir.
func (d *downloader) fetch(ctx context.Context, dir string, f File, m *marker, tp *towerProgress) error {
	dest := localPath(dir, f)
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}

	// Already on disk: trust a matching marker, otherwise hash once and adopt.
	if info, err := os.Stat(dest); err == nil {
		if info.Size() == f.Size {
			if m.verified(f, info) {
				tp.adopted(f.Size)
				return nil
			}
			tp.emit(StateVerifying, true)
			sum, err := hashFile(ctx, dest)
			if err != nil {
				return err
			}
			if sum == f.SHA256 {
				tp.adopted(f.Size)
				return d.record(dir, f, dest, m)
			}
		}
		if err := os.Remove(dest); err != nil {
			return err
		}
	}

	part := dest + partSuffix
	scratchTried := false
	var lastErr error
	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			if attempt > d.retries {
				return &NetworkError{File: f.Path, Attempts: attempt, Err: lastErr}
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(d.backoff(attempt - 1)):
			}
		}
		fromZero, err := d.attempt(ctx, f, part, tp)
		switch {
		case err == nil:
			// attempt verified the sha256 and fsynced the part file
		case errors.Is(err, ErrChecksum):
			os.Remove(part)
			if fromZero || scratchTried {
				return fmt.Errorf("%s: %w (the server delivered different content than pinned)", f.Path, err)
			}
			// a resumed prefix was corrupt: start over once
			scratchTried = true
			tp.cur = 0
			attempt--
			continue
		case ctx.Err() != nil:
			return ctx.Err()
		case isPermanent(err):
			return err
		default:
			lastErr = err
			continue
		}
		if err := os.Rename(part, dest); err != nil {
			return err
		}
		if err := syncDir(filepath.Dir(dest)); err != nil {
			return err
		}
		return d.record(dir, f, dest, m)
	}
}

func isPermanent(err error) bool {
	var he *HTTPStatusError
	if errors.As(err, &he) {
		return he.Status != http.StatusRequestTimeout && he.Status != http.StatusTooManyRequests && he.Status < 500
	}
	var pe *fs.PathError
	return errors.As(err, &pe) && !isNetErr(err)
}

func isNetErr(err error) bool {
	var ne net.Error
	return errors.As(err, &ne)
}

func (d *downloader) record(dir string, f File, dest string, m *marker) error {
	info, err := os.Stat(dest)
	if err != nil {
		return err
	}
	m.Files[f.Path] = markerEntry{Size: f.Size, MTimeNS: info.ModTime().UnixNano(), SHA256: f.SHA256}
	return writeMarker(dir, *m)
}

// attempt downloads (the rest of) f into part and verifies the sha256. fromZero
// tells whether the part file was started in this attempt.
func (d *downloader) attempt(ctx context.Context, f File, part string, tp *towerProgress) (fromZero bool, err error) {
	out, err := os.OpenFile(part, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return false, err
	}
	defer out.Close()
	info, err := out.Stat()
	if err != nil {
		return false, err
	}
	h := sha256.New()
	have := info.Size()
	if have > f.Size {
		have = 0
		if err := out.Truncate(0); err != nil {
			return false, err
		}
	}
	if have > 0 {
		if _, err := io.Copy(h, io.LimitReader(out, have)); err != nil { // out is at offset 0
			return false, err
		}
	}
	fromZero = have == 0
	tp.set(have)

	if have < f.Size {
		got, err := d.stream(ctx, f, out, h, have, tp)
		if err != nil {
			return fromZero, err
		}
		have = got
	}
	if have != f.Size {
		return fromZero, fmt.Errorf("%w: received %d of %d bytes", io.ErrUnexpectedEOF, have, f.Size)
	}
	tp.emit(StateVerifying, true)
	if err := out.Sync(); err != nil {
		return fromZero, err
	}
	if sum := hex.EncodeToString(h.Sum(nil)); sum != f.SHA256 {
		return fromZero, fmt.Errorf("%w: %s", ErrChecksum, f.Path)
	}
	return fromZero, nil
}

// stream requests f from offset and appends the body to out, hashing as it goes.
// It returns the new file size.
func (d *downloader) stream(ctx context.Context, f File, out *os.File, h hash.Hash, offset int64, tp *towerProgress) (int64, error) {
	rctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	url := f.URL(d.baseURL)
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, url, nil)
	if err != nil {
		return offset, err
	}
	req.Header.Set("User-Agent", "dsearch-model/1")
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	watchdog := time.AfterFunc(d.stall, func() { cancel(errStalled) })
	defer watchdog.Stop()

	resp, err := d.client.Do(req)
	if err != nil {
		return offset, wrapCause(rctx, err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		if resp.ContentLength >= 0 && resp.ContentLength != f.Size {
			return offset, fmt.Errorf("%s: server announces %d bytes, expected %d", url, resp.ContentLength, f.Size)
		}
		if offset > 0 { // the server ignored the Range header: start over
			if err := restart(out, h); err != nil {
				return 0, err
			}
			offset = 0
			tp.set(0)
		}
	case http.StatusPartialContent:
		start, total, ok := parseContentRange(resp.Header.Get("Content-Range"))
		if !ok || start != offset || (total >= 0 && total != f.Size) {
			return offset, fmt.Errorf("%s: unexpected Content-Range %q for offset %d", url, resp.Header.Get("Content-Range"), offset)
		}
	case http.StatusRequestedRangeNotSatisfiable:
		// the part file is not a valid prefix of the pinned file
		if err := restart(out, h); err != nil {
			return 0, err
		}
		tp.set(0)
		return 0, fmt.Errorf("%w: range not satisfiable at offset %d", io.ErrUnexpectedEOF, offset)
	default:
		return offset, &HTTPStatusError{URL: url, Status: resp.StatusCode}
	}

	if _, err := out.Seek(offset, io.SeekStart); err != nil {
		return offset, err
	}
	buf := make([]byte, 256<<10)
	pos := offset
	for {
		watchdog.Reset(d.stall)
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if pos+int64(n) > f.Size {
				return pos, fmt.Errorf("%s: more data than the pinned size %d", url, f.Size)
			}
			if _, err := out.Write(buf[:n]); err != nil {
				return pos, err
			}
			h.Write(buf[:n])
			pos += int64(n)
			tp.set(pos)
		}
		if rerr == io.EOF {
			return pos, nil
		}
		if rerr != nil {
			return pos, wrapCause(rctx, rerr)
		}
	}
}

func restart(out *os.File, h hash.Hash) error {
	h.Reset()
	if err := out.Truncate(0); err != nil {
		return err
	}
	_, err := out.Seek(0, io.SeekStart)
	return err
}

// wrapCause prefers the cancellation cause (stall) over the generic context error.
func wrapCause(ctx context.Context, err error) error {
	if c := context.Cause(ctx); c != nil && errors.Is(c, errStalled) {
		return c
	}
	return err
}

// parseContentRange reads "bytes START-END/TOTAL"; total is -1 for "*".
func parseContentRange(v string) (start, total int64, ok bool) {
	rest, found := strings.CutPrefix(v, "bytes ")
	if !found {
		return 0, 0, false
	}
	rng, tot, found := strings.Cut(rest, "/")
	if !found {
		return 0, 0, false
	}
	s, _, found := strings.Cut(rng, "-")
	if !found {
		return 0, 0, false
	}
	start, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, 0, false
	}
	total = -1
	if tot != "*" {
		if total, err = strconv.ParseInt(tot, 10, 64); err != nil {
			return 0, 0, false
		}
	}
	return start, total, true
}

func hashFile(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	buf := make([]byte, 1<<20)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, err := f.Read(buf)
		h.Write(buf[:n])
		if err == io.EOF {
			return hex.EncodeToString(h.Sum(nil)), nil
		}
		if err != nil {
			return "", err
		}
	}
}
