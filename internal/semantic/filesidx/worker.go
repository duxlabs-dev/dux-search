package filesidx

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
)

const (
	maxTries        = 5
	chunkHashDigits = 12
	commitEvery     = 64
)

// retryBackoff is the wait before the n-th retry of a transient failure.
var retryBackoff = []time.Duration{30 * time.Second, 2 * time.Minute, 10 * time.Minute, time.Hour, 6 * time.Hour}

// worker is the only goroutine that reads files and calls the embedder. It
// runs on a locked, lowest priority thread.
func (x *Indexer) worker(ctx context.Context) {
	if !lowerThreadPriority() {
		log.Debugf("semantic files: could not lower the indexing thread priority")
	}
	st := x.st
	skip := map[uint64]uint64{}
	var outs []outcome
	lastCommit := time.Now()
	flush := func() {
		if len(outs) == 0 {
			return
		}
		removed, delta, err := st.commit(outs)
		if err != nil {
			log.Errorf("semantic files: saving %d results: %v", len(outs), err)
		} else {
			x.queued.Add(-int64(removed))
			x.indexed.Add(int64(delta))
		}
		outs, lastCommit = nil, time.Now()
		clear(skip)
	}
	defer flush()

	for ctx.Err() == nil {
		if x.pauseReason() != "" {
			flush()
			x.wait(ctx, time.Time{})
			continue
		}
		if !x.emb.Status().Available {
			flush()
			x.modelRetryAt.Store(time.Now().Add(x.cfg.ModelRetry).UnixNano())
			continue
		}
		it, ok, wakeAt, err := st.next(time.Now(), skip)
		if err != nil {
			flush()
			log.Errorf("semantic files: reading the queue: %v", err)
			x.wait(ctx, time.Time{})
			continue
		}
		if !ok {
			if len(outs) > 0 {
				flush() // retries committed now may fall due soon: look again before sleeping
				continue
			}
			x.wait(ctx, wakeAt)
			continue
		}
		out := x.process(ctx, st, it)
		if out.abort {
			flush()
			continue
		}
		skip[it.seq] = it.item.Ver
		outs = append(outs, out)
		if !out.retry {
			x.done.Add(1)
			if out.rec != nil && out.rec.Err != "" {
				x.failed.Add(1)
			}
		}
		if len(outs) >= commitEvery || time.Since(lastCommit) > time.Second {
			flush()
		}
	}
}

// wait sleeps until there is something to do: new events, a changed policy,
// the time a retry falls due, or the model retry time.
func (x *Indexer) wait(ctx context.Context, due time.Time) {
	d := time.Minute
	if due.IsZero() {
		if at := x.modelRetryAt.Load(); at > 0 {
			if left := time.Until(time.Unix(0, at)); left > 0 {
				d = min(d, left)
			}
		}
	} else {
		d = min(d, max(time.Until(due), time.Millisecond))
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-x.wake:
	case <-t.C:
	}
}

// interrupted reports that the work on the current file must stop.
func (x *Indexer) interrupted(ctx context.Context) bool {
	return ctx.Err() != nil || x.pauseReason() != ""
}

// job is one queue item being processed.
type job struct {
	x     *Indexer
	q     pendingQueued
	path  string
	info  os.FileInfo
	kind  string
	old   record
	hasOl bool
}

func (x *Indexer) process(ctx context.Context, st *stateDB, q pendingQueued) outcome {
	j := &job{x: x, q: q, path: q.item.Path}
	var err error
	if j.old, j.hasOl, err = st.record(j.path); err != nil {
		return j.retry(err)
	}

	if q.item.Op == OpDelete {
		return j.remove(ctx)
	}
	info, err := os.Stat(j.path)
	switch {
	case errors.Is(err, fs.ErrNotExist), errors.Is(err, syscall.ENOTDIR):
		return j.remove(ctx)
	case err != nil:
		return j.failed(err) // unreadable (permissions); retried when the file changes
	case !info.Mode().IsRegular():
		return j.remove(ctx)
	}
	j.info = info
	j.kind = x.kindOf(j.path)
	if j.kind == "" || !x.kinds[j.kind] || !x.cfg.Allow(j.path) {
		return j.remove(ctx)
	}
	if j.hasOl && j.old.current(info, j.kind) {
		return outcome{q: q, keep: true}
	}
	if j.kind == kindText {
		return j.indexText(ctx)
	}
	return j.indexMedia(ctx)
}

// remove deletes the file's vectors and forgets its record.
func (j *job) remove(ctx context.Context) outcome {
	if !j.hasOl {
		return outcome{q: j.q} // never embedded: nothing to remove
	}
	if err := j.dropVectors(j.old, 0); err != nil {
		return j.retry(err)
	}
	return outcome{q: j.q}
}

// dropVectors deletes the stored vectors of rec beyond the first keep chunks.
func (j *job) dropVectors(rec record, keep int) error {
	name, _ := collectionOf(rec.Kind)
	if name == "" {
		return nil
	}
	ids := recordIDs(j.path, rec)
	if rec.Kind == kindText {
		if keep >= len(ids) {
			return nil
		}
		ids = ids[keep:]
	}
	for chunk := range slices.Chunk(ids, deleteChunk) {
		if err := j.x.store.Delete(name, chunk); err != nil {
			return fmt.Errorf("delete vectors: %w", err)
		}
	}
	return nil
}

// failed records a permanent reason the file cannot be indexed and removes
// vectors of an earlier version of it.
func (j *job) failed(reason error) outcome {
	rec := record{Size: -1, Err: reason.Error(), Ver: pipelineVersion}
	if j.info != nil {
		rec.Size, rec.MTime = j.info.Size(), j.info.ModTime().UnixNano()
	}
	if j.hasOl {
		if err := j.dropVectors(j.old, 0); err != nil {
			return j.retry(err)
		}
	}
	log.Debugf("semantic files: not indexing %s: %v", j.path, reason)
	return outcome{q: j.q, rec: &rec}
}

// retry schedules another attempt after a transient failure, or gives up.
func (j *job) retry(err error) outcome {
	if j.q.item.Tries+1 >= maxTries {
		log.Warnf("semantic files: giving up on %s: %v", j.path, err)
		return j.failed(fmt.Errorf("embedding failed: %w", err))
	}
	wait := retryBackoff[min(j.q.item.Tries, len(retryBackoff)-1)]
	log.Warnf("semantic files: %s: %v (retry in %s)", j.path, err, wait)
	return outcome{q: j.q, keep: true, retry: true, retryAt: time.Now().Add(wait)}
}

// embedErr maps an embedder error to an outcome.
func (j *job) embedErr(ctx context.Context, err error) outcome {
	switch {
	case ctx.Err() != nil:
		return outcome{abort: true}
	case errors.Is(err, semantic.ErrModelUnavailable):
		j.x.modelRetryAt.Store(time.Now().Add(j.x.cfg.ModelRetry).UnixNano())
		return outcome{abort: true}
	}
	return j.retry(err)
}

func (j *job) newRecord(kind, hash string) record {
	return record{Kind: kind, Size: j.info.Size(), MTime: j.info.ModTime().UnixNano(), Hash: hash, Ver: pipelineVersion}
}

func (j *job) indexText(ctx context.Context) outcome {
	x := j.x
	text, hash, err := readText(j.path, x.cfg.MaxTextBytes, x.cfg.LegacyEncoding)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return j.remove(ctx)
	case errors.Is(err, errBinary):
		return j.failed(errBinary)
	case err != nil:
		return j.failed(err)
	}
	same := j.hasOl && j.old.Kind == kindText && j.old.Ver == pipelineVersion
	if same && j.old.Hash == hash { // touched, not edited
		rec := j.old
		rec.Size, rec.MTime = j.info.Size(), j.info.ModTime().UnixNano()
		return outcome{q: j.q, rec: &rec}
	}

	chunks, cut := splitText(text, x.cfg.MaxChunksPerFile)
	cut = cut || j.info.Size() > x.cfg.MaxTextBytes
	title := filepath.Base(j.path)
	hashes := make([]string, len(chunks))
	for i, c := range chunks {
		sum := sha256.Sum256([]byte(title + "\x00" + c))
		hashes[i] = hex.EncodeToString(sum[:])[:chunkHashDigits]
	}
	var oldHashes []string
	if same {
		for i := 0; i+chunkHashDigits <= len(j.old.ChunkHash); i += chunkHashDigits {
			oldHashes = append(oldHashes, j.old.ChunkHash[i:i+chunkHashDigits])
		}
	}
	var todo []int
	for i := range chunks {
		if i >= len(oldHashes) || oldHashes[i] != hashes[i] {
			todo = append(todo, i)
		}
	}

	for batch := range slices.Chunk(todo, x.cfg.TextBatch) {
		if x.interrupted(ctx) {
			return outcome{abort: true}
		}
		docs := make([]semantic.TextDoc, len(batch))
		for k, i := range batch {
			docs[k] = semantic.TextDoc{Title: title, Text: chunks[i]}
		}
		vecs, err := x.emb.EmbedDocuments(ctx, docs)
		if err != nil {
			return j.embedErr(ctx, err)
		}
		if len(vecs) != len(docs) {
			return j.retry(fmt.Errorf("embedder returned %d vectors for %d chunks", len(vecs), len(docs)))
		}
		out := make([]semantic.Vector, len(batch))
		for k, i := range batch {
			v, err := fitVector(vecs[k], DimText)
			if err != nil {
				return j.retry(err)
			}
			out[k] = semantic.Vector{ID: chunkID(j.path, i), Hash: hashes[i], Vec: v}
		}
		if err := x.store.Upsert(CollectionText, out); err != nil {
			return j.retry(fmt.Errorf("store vectors: %w", err))
		}
	}

	if j.hasOl { // chunks that no longer exist
		stale := j.old
		if stale.Kind != kindText {
			if err := j.dropVectors(stale, 0); err != nil {
				return j.retry(err)
			}
		} else if err := j.dropVectors(stale, len(chunks)); err != nil {
			return j.retry(err)
		}
	}
	rec := j.newRecord(kindText, hash)
	rec.Chunks, rec.ChunkHash, rec.Trunc = len(chunks), strings.Join(hashes, ""), cut
	return outcome{q: j.q, rec: &rec}
}

func (j *job) indexMedia(ctx context.Context) outcome {
	x := j.x
	limit := x.cfg.MaxImageBytes
	if j.kind == kindAudio {
		limit = x.cfg.MaxAudioBytes
	}
	if j.info.Size() > limit {
		return j.failed(fmt.Errorf("file larger than %d MiB", limit>>20))
	}
	hash, err := hashFile(ctx, j.path)
	switch {
	case ctx.Err() != nil:
		return outcome{abort: true}
	case errors.Is(err, fs.ErrNotExist):
		return j.remove(ctx)
	case err != nil:
		return j.failed(err)
	}
	if j.hasOl && j.old.Kind == j.kind && j.old.Ver == pipelineVersion && j.old.Hash == hash {
		rec := j.old
		rec.Size, rec.MTime = j.info.Size(), j.info.ModTime().UnixNano()
		return outcome{q: j.q, rec: &rec}
	}
	if x.interrupted(ctx) {
		return outcome{abort: true}
	}

	var vec []float32
	var dim int
	var name string
	switch j.kind {
	case kindImage:
		name, dim = CollectionImage, DimImage
		f, err := os.Open(j.path)
		if err != nil {
			return j.failed(err)
		}
		in, perr := x.cfg.Preprocessor.Image(f)
		f.Close()
		if perr != nil {
			return j.failed(fmt.Errorf("image: %w", perr))
		}
		vecs, err := x.emb.EmbedImages(ctx, []semantic.ImageInput{in})
		if err != nil {
			return j.embedErr(ctx, err)
		}
		if len(vecs) != 1 {
			return j.retry(fmt.Errorf("embedder returned %d vectors for one image", len(vecs)))
		}
		vec = vecs[0]
	default:
		name, dim = CollectionAudio, DimAudio
		in, perr := x.cfg.Preprocessor.Audio(j.path)
		if perr != nil {
			return j.failed(fmt.Errorf("audio: %w", perr))
		}
		vecs, err := x.emb.EmbedAudio(ctx, []semantic.AudioInput{in})
		if err != nil {
			return j.embedErr(ctx, err)
		}
		if len(vecs) != 1 {
			return j.retry(fmt.Errorf("embedder returned %d vectors for one clip", len(vecs)))
		}
		vec = vecs[0]
	}
	v, err := fitVector(vec, dim)
	if err != nil {
		return j.retry(err)
	}
	if err := x.store.Upsert(name, []semantic.Vector{{ID: j.path, Hash: hash, Vec: v}}); err != nil {
		return j.retry(fmt.Errorf("store vector: %w", err))
	}
	if j.hasOl && j.old.Kind != j.kind { // the extension decides the kind; cannot change, kept for safety
		if err := j.dropVectors(j.old, 0); err != nil {
			return j.retry(err)
		}
	}
	rec := j.newRecord(j.kind, hash)
	rec.Chunks = 1
	return outcome{q: j.q, rec: &rec}
}

// fitVector checks an embedder result and applies the matryoshka cut.
func fitVector(v []float32, dim int) ([]float32, error) {
	if len(v) != semantic.NativeDim {
		return nil, fmt.Errorf("embedder returned %d dimensions, expected %d", len(v), semantic.NativeDim)
	}
	return semantic.Truncate(v, dim), nil
}

// hashFile is the sha256 of a whole file, read in a way that honours ctx.
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
