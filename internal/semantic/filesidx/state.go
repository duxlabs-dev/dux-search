package filesidx

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	bolt "go.etcd.io/bbolt"
)

// Op is the kind of change an indexer event reports.
type Op int

const (
	// OpIndex adds a file or re-embeds a changed one.
	OpIndex Op = iota + 1
	// OpDelete removes every vector of a file.
	OpDelete
)

func (o Op) String() string {
	switch o {
	case OpIndex:
		return "index"
	case OpDelete:
		return "delete"
	}
	return fmt.Sprintf("op(%d)", int(o))
}

const (
	stateSchema     = 1
	pipelineVersion = 1 // bump when chunking or preprocessing changes: every file is re-embedded
)

var (
	bMeta  = []byte("meta")
	bFiles = []byte("files") // path -> record
	bQueue = []byte("queue") // seq -> queueItem
	bIdx   = []byte("qidx")  // path -> seq
)

// Kinds of indexed content.
const (
	kindText  = "text"
	kindImage = "image"
	kindAudio = "audio"
)

// record is what the indexer knows about one file it has processed.
type record struct {
	Kind      string `json:"k,omitempty"`  // kindText, kindImage, kindAudio; "" when nothing is stored
	Size      int64  `json:"s"`            // file size when processed
	MTime     int64  `json:"m"`            // modification time (unix nanoseconds) when processed
	Hash      string `json:"h,omitempty"`  // sha256 of the content that was embedded
	Chunks    int    `json:"c,omitempty"`  // vectors stored: text path#0..n-1, image/audio one under path
	ChunkHash string `json:"ch,omitempty"` // text: 12 hex digits per chunk
	Trunc     bool   `json:"t,omitempty"`  // text was cut at the chunk or size limit
	Err       string `json:"e,omitempty"`  // permanent reason the file is not indexed
	Ver       int    `json:"v"`
}

type queueItem struct {
	Path  string `json:"p"`
	Op    Op     `json:"o"`
	Ver   uint64 `json:"n"`           // bumps when the same path is queued again while in flight
	Tries int    `json:"t,omitempty"` // transient failures so far
	Due   int64  `json:"d,omitempty"` // unix nanoseconds before which the item is not retried
}

type pendingQueued struct {
	seq  uint64
	item queueItem
}

type stateDB struct{ db *bolt.DB }

func openState(path, modelID string) (*stateDB, error) {
	db, err := bolt.Open(path, 0o600, &bolt.Options{Timeout: 2 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("open index state %s: %w", path, err)
	}
	err = db.Update(func(tx *bolt.Tx) error {
		for _, b := range [][]byte{bMeta, bFiles, bQueue, bIdx} {
			if _, err := tx.CreateBucketIfNotExists(b); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		db.Close()
		return nil, err
	}
	return &stateDB{db: db}, nil
}

func (s *stateDB) close() error { return s.db.Close() }

func (s *stateDB) getMeta(key string) (string, error) {
	var v string
	err := s.db.View(func(tx *bolt.Tx) error {
		v = string(tx.Bucket(bMeta).Get([]byte(key)))
		return nil
	})
	return v, err
}

func (s *stateDB) setMeta(key, value string) error {
	return s.db.Update(func(tx *bolt.Tx) error { return tx.Bucket(bMeta).Put([]byte(key), []byte(value)) })
}

func (s *stateDB) record(path string) (record, bool, error) {
	var rec record
	var ok bool
	err := s.db.View(func(tx *bolt.Tx) error {
		if raw := tx.Bucket(bFiles).Get([]byte(path)); raw != nil {
			ok = json.Unmarshal(raw, &rec) == nil
		}
		return nil
	})
	return rec, ok, err
}

// forEachRecord calls fn for every record inside one read transaction.
func (s *stateDB) forEachRecord(fn func(path string, rec record) error) error {
	return s.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket(bFiles).ForEach(func(k, v []byte) error {
			var rec record
			if err := json.Unmarshal(v, &rec); err != nil {
				return nil
			}
			return fn(string(k), rec)
		})
	})
}

func seqKey(seq uint64) []byte {
	k := make([]byte, 8)
	binary.BigEndian.PutUint64(k, seq)
	return k
}

// enqueueBatch queues many events in one transaction. A path that is already
// queued keeps its position and takes the newer operation. A deletion of a
// path that is neither queued nor recorded has nothing to remove and is
// dropped.
func (s *stateDB) enqueueBatch(ops map[string]Op) (added int, err error) {
	if len(ops) == 0 {
		return 0, nil
	}
	err = s.db.Update(func(tx *bolt.Tx) error {
		added = 0
		q, idx, files := tx.Bucket(bQueue), tx.Bucket(bIdx), tx.Bucket(bFiles)
		for path, op := range ops {
			var seq uint64
			item := queueItem{Path: path, Op: op}
			cur := idx.Get([]byte(path))
			if cur == nil && op == OpDelete && files.Get([]byte(path)) == nil {
				continue
			}
			if cur != nil {
				seq = binary.BigEndian.Uint64(cur)
				var old queueItem
				if raw := q.Get(seqKey(seq)); raw != nil && json.Unmarshal(raw, &old) == nil {
					item.Ver = old.Ver + 1
				}
			} else {
				var err error
				if seq, err = q.NextSequence(); err != nil {
					return err
				}
				if err := idx.Put([]byte(path), seqKey(seq)); err != nil {
					return err
				}
				added++
			}
			raw, _ := json.Marshal(item)
			if err := q.Put(seqKey(seq), raw); err != nil {
				return err
			}
		}
		return nil
	})
	return added, err
}

// next returns the oldest queue item that is due. When nothing is due but
// retries are waiting, wake is the time the earliest one becomes due.
// Items in skip (seq -> Ver) were processed but not committed yet.
func (s *stateDB) next(now time.Time, skip map[uint64]uint64) (it pendingQueued, found bool, wake time.Time, err error) {
	err = s.db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket(bQueue).Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			var item queueItem
			if json.Unmarshal(v, &item) != nil {
				continue
			}
			if v, done := skip[binary.BigEndian.Uint64(k)]; done && v == item.Ver {
				continue
			}
			if item.Due > now.UnixNano() {
				if due := time.Unix(0, item.Due); wake.IsZero() || due.Before(wake) {
					wake = due
				}
				continue
			}
			it, found = pendingQueued{seq: binary.BigEndian.Uint64(k), item: item}, true
			return nil
		}
		return nil
	})
	return it, found, wake, err
}

// pendingCount is the number of queued paths.
func (s *stateDB) pendingCount() (int, error) {
	n := 0
	err := s.db.View(func(tx *bolt.Tx) error { n = tx.Bucket(bQueue).Stats().KeyN; return nil })
	return n, err
}

func (s *stateDB) fileCount() (int, error) {
	n := 0
	err := s.db.View(func(tx *bolt.Tx) error { n = tx.Bucket(bFiles).Stats().KeyN; return nil })
	return n, err
}

// outcome is the result of processing one queue item.
type outcome struct {
	q       pendingQueued
	rec     *record // new record; nil removes the path's record
	keep    bool    // true: leave the path's record untouched
	abort   bool    // paused or shutting down: nothing was done, nothing to commit
	retry   bool    // transient failure: keep the item queued
	retryAt time.Time
}

// commit applies finished items: it stores their records and removes them
// from the queue unless the path was queued again while it was processed.
func (s *stateDB) commit(outs []outcome) (removed, recordDelta int, err error) {
	if len(outs) == 0 {
		return 0, 0, nil
	}
	err = s.db.Update(func(tx *bolt.Tx) error {
		removed, recordDelta = 0, 0
		q, idx, files := tx.Bucket(bQueue), tx.Bucket(bIdx), tx.Bucket(bFiles)
		for _, o := range outs {
			path := []byte(o.q.item.Path)
			switch {
			case o.keep:
			case o.rec == nil:
				if files.Get(path) != nil {
					recordDelta--
				}
				if err := files.Delete(path); err != nil {
					return err
				}
			default:
				if files.Get(path) == nil {
					recordDelta++
				}
				raw, _ := json.Marshal(o.rec)
				if err := files.Put(path, raw); err != nil {
					return err
				}
			}
			key := seqKey(o.q.seq)
			raw := q.Get(key)
			if raw == nil {
				continue
			}
			var cur queueItem
			if json.Unmarshal(raw, &cur) != nil || cur.Ver != o.q.item.Ver {
				continue // queued again meanwhile: process the newer event
			}
			if o.retry {
				cur.Tries++
				cur.Due = o.retryAt.UnixNano()
				raw, _ := json.Marshal(cur)
				if err := q.Put(key, raw); err != nil {
					return err
				}
				continue
			}
			if err := q.Delete(key); err != nil {
				return err
			}
			if err := idx.Delete(path); err != nil {
				return err
			}
			removed++
		}
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	return removed, recordDelta, nil
}

// putRecords writes records without touching the queue (used by recovery).
func (s *stateDB) putRecords(recs map[string]*record) error {
	return s.db.Update(func(tx *bolt.Tx) error {
		files := tx.Bucket(bFiles)
		for p, rec := range recs {
			if rec == nil {
				if err := files.Delete([]byte(p)); err != nil {
					return err
				}
				continue
			}
			raw, _ := json.Marshal(rec)
			if err := files.Put([]byte(p), raw); err != nil {
				return err
			}
		}
		return nil
	})
}

var errClosed = errors.New("file indexer is not running")
