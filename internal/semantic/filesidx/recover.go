package filesidx

import (
	"fmt"
	"slices"

	"github.com/AvengeMedia/dankgo/log"
)

const deleteChunk = 500

func collectionOf(kind string) (name string, dim int) {
	switch kind {
	case kindText:
		return CollectionText, DimText
	case kindImage:
		return CollectionImage, DimImage
	case kindAudio:
		return CollectionAudio, DimAudio
	}
	return "", 0
}

func recordIDs(path string, rec record) []string {
	switch rec.Kind {
	case kindText:
		ids := make([]string, rec.Chunks)
		for i := range ids {
			ids[i] = chunkID(path, i)
		}
		return ids
	case kindImage, kindAudio:
		return []string{path}
	}
	return nil
}

// prepare brings the state and the vector store into agreement before the
// workers start: a changed model or pipeline re-embeds everything, vectors
// that the state does not know are removed, and files whose vectors went
// missing from the store are queued again.
func (x *Indexer) prepare(st *stateDB) error {
	prev, err := st.getMeta("model")
	if err != nil {
		return err
	}
	if prev != "" && prev != x.cfg.ModelID {
		log.Infof("semantic files: model changed (%s -> %s), re-embedding all files", prev, x.cfg.ModelID)
		if err := x.requeueAll(st); err != nil {
			return err
		}
	}
	if err := st.setMeta("model", x.cfg.ModelID); err != nil {
		return err
	}
	if err := st.setMeta("schema", fmt.Sprint(stateSchema)); err != nil {
		return err
	}
	// A disabled kind keeps its collection while records of it exist, so the
	// vectors can still be removed.
	stored := map[string]bool{}
	if err := st.forEachRecord(func(_ string, rec record) error { stored[rec.Kind] = true; return nil }); err != nil {
		return err
	}
	for _, kind := range []string{kindText, kindImage, kindAudio} {
		if !x.kinds[kind] && !stored[kind] {
			continue
		}
		name, dim := collectionOf(kind)
		if err := x.store.EnsureCollection(name, dim, x.cfg.ModelID); err != nil {
			return fmt.Errorf("prepare collection %s: %w", name, err)
		}
	}
	return x.verify(st)
}

// requeueAll queues every known file for indexing and forgets its record.
func (x *Indexer) requeueAll(st *stateDB) error {
	var paths []string
	if err := st.forEachRecord(func(p string, _ record) error { paths = append(paths, p); return nil }); err != nil {
		return err
	}
	ops := make(map[string]Op, len(paths))
	drop := make(map[string]*record, len(paths))
	for _, p := range paths {
		ops[p] = OpIndex
		drop[p] = nil
	}
	if _, err := st.enqueueBatch(ops); err != nil {
		return err
	}
	return st.putRecords(drop)
}

func (x *Indexer) verify(st *stateDB) error {
	expectedCount := map[string]int{}
	var stale []string // records the current configuration no longer wants, or from an older pipeline
	err := st.forEachRecord(func(p string, rec record) error {
		if rec.Kind != "" && !x.kinds[rec.Kind] || rec.Ver != pipelineVersion {
			stale = append(stale, p)
		}
		if rec.Kind != "" && x.kinds[rec.Kind] {
			name, _ := collectionOf(rec.Kind)
			expectedCount[name] += len(recordIDs(p, rec))
		}
		return nil
	})
	if err != nil {
		return err
	}
	requeue := map[string]Op{}
	for _, p := range stale {
		requeue[p] = OpIndex
	}

	for _, kind := range []string{kindText, kindImage, kindAudio} {
		if !x.kinds[kind] {
			continue
		}
		name, _ := collectionOf(kind)
		have, err := x.store.Count(name)
		if err != nil {
			return fmt.Errorf("count %s: %w", name, err)
		}
		if have == expectedCount[name] {
			continue
		}
		log.Warnf("semantic files: %s holds %d vectors but the state expects %d, repairing", name, have, expectedCount[name])
		hashes, err := x.store.Hashes(name)
		if err != nil {
			return fmt.Errorf("list %s: %w", name, err)
		}
		expected := map[string]string{} // id -> owning path
		if err := st.forEachRecord(func(p string, rec record) error {
			if rec.Kind == kind {
				for _, id := range recordIDs(p, rec) {
					expected[id] = p
				}
			}
			return nil
		}); err != nil {
			return err
		}
		var orphans []string
		for id := range hashes {
			if _, ok := expected[id]; !ok {
				orphans = append(orphans, id)
			}
		}
		for chunk := range slices.Chunk(orphans, deleteChunk) {
			if err := x.store.Delete(name, chunk); err != nil {
				return fmt.Errorf("delete orphans of %s: %w", name, err)
			}
		}
		invalid := map[string]*record{}
		for id, p := range expected {
			if _, ok := hashes[id]; ok {
				continue
			}
			if _, done := invalid[p]; done {
				continue
			}
			rec, _, err := st.record(p)
			if err != nil {
				return err
			}
			rec.Ver = 0 // never "current": the next pass re-embeds the file
			invalid[p] = &rec
			requeue[p] = OpIndex
		}
		if err := st.putRecords(invalid); err != nil {
			return err
		}
	}
	_, err = st.enqueueBatch(requeue)
	return err
}
