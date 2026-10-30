package vecstore

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/AvengeMedia/dankgo/log"
	bolt "go.etcd.io/bbolt"
	berrors "go.etcd.io/bbolt/errors"
)

// Layout of the database:
//
//	meta   "format"  -> SchemaVersion the file was created with
//	meta   "c/<name>" -> JSON collMeta
//	vec    <name>/<id> -> record
//	items  <name>/<id> -> JSON semantic.Item (the desired state, see items.go)
//
// A record is [u16 hashLen][hash][f32 scale][dim x int8], little endian.
var (
	metaBucket  = []byte("meta")
	vecBucket   = []byte("vec")
	itemsBucket = []byte("items")
	formatKey   = []byte("format")
)

const metaPrefix = "c/"

type collMeta struct {
	Dim    int    `json:"dim"`
	Model  string `json:"model"`
	Schema int    `json:"schema"`
}

type record struct {
	id    string
	hash  string
	q     []int8
	scale float32
	inv   float32
}

func encodeRecord(r record) []byte {
	buf := make([]byte, 2+len(r.hash)+4+len(r.q))
	binary.LittleEndian.PutUint16(buf, uint16(len(r.hash)))
	n := 2 + copy(buf[2:], r.hash)
	binary.LittleEndian.PutUint32(buf[n:], math.Float32bits(r.scale))
	n += 4
	for i, v := range r.q {
		buf[n+i] = byte(v)
	}
	return buf
}

// decodeRecord validates a stored record for a collection of the given dim.
func decodeRecord(id string, b []byte, dim int) (record, error) {
	if len(b) < 2 {
		return record{}, errors.New("truncated record")
	}
	hl := int(binary.LittleEndian.Uint16(b))
	if len(b) != 2+hl+4+dim {
		return record{}, fmt.Errorf("record is %d bytes, want %d", len(b), 2+hl+4+dim)
	}
	scale := math.Float32frombits(binary.LittleEndian.Uint32(b[2+hl:]))
	if scale != scale || scale <= 0 || math.IsInf(float64(scale), 0) {
		return record{}, errors.New("invalid scale")
	}
	q := make([]int8, dim)
	var sumsq int64
	for i := range q {
		v := int8(b[2+hl+4+i])
		if v == -128 {
			return record{}, errors.New("out of range value")
		}
		q[i] = v
		sumsq += int64(v) * int64(v)
	}
	if sumsq == 0 {
		return record{}, errors.New("zero vector")
	}
	return record{id: id, hash: string(b[2 : 2+hl]), q: q, scale: scale, inv: float32(1 / math.Sqrt(float64(sumsq)))}, nil
}

// bbolt reports damaged pages by panicking or faulting; every database touch
// converts that into an error (same approach as internal/metastore).
func recoverDBPanic(err *error) {
	if r := recover(); r != nil {
		*err = fmt.Errorf("vecstore db panic: %v", r)
	}
}

func armFaultPanics() func() {
	prev := debug.SetPanicOnFault(true)
	return func() { debug.SetPanicOnFault(prev) }
}

func (s *Store) dbUpdate(fn func(tx *bolt.Tx) error) (err error) {
	defer recoverDBPanic(&err)
	defer armFaultPanics()()
	return s.db.Update(fn)
}

func (s *Store) dbView(fn func(tx *bolt.Tx) error) (err error) {
	defer recoverDBPanic(&err)
	defer armFaultPanics()()
	return s.db.View(fn)
}

func isCorruption(err error) bool {
	if err == nil || errors.Is(err, berrors.ErrTimeout) || errors.Is(err, errNewerFormat) {
		return false
	}
	return true
}

var errNewerFormat = errors.New("vecstore: database was written by a newer dsearch")

func moveAside(path string) (string, error) {
	backup := path + ".corrupt"
	_ = os.Remove(backup)
	if err := os.Rename(path, backup); err != nil {
		return "", err
	}
	return backup, nil
}

func (s *Store) openAndLoad(path string) (err error) {
	defer recoverDBPanic(&err)
	defer armFaultPanics()()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	db, err := bolt.Open(path, 0o600, &bolt.Options{Timeout: time.Second})
	if err != nil {
		if errors.Is(err, berrors.ErrTimeout) {
			return fmt.Errorf("semantic store locked by another process (is the daemon running?): %w", err)
		}
		return err
	}
	s.db = db
	defer func() {
		if err != nil {
			db.Close()
			s.db = nil
		}
	}()

	err = db.Update(func(tx *bolt.Tx) error {
		meta, err := tx.CreateBucketIfNotExists(metaBucket)
		if err != nil {
			return err
		}
		for _, name := range [][]byte{vecBucket, itemsBucket} {
			if _, err := tx.CreateBucketIfNotExists(name); err != nil {
				return err
			}
		}
		if cur := meta.Get(formatKey); cur != nil {
			if v, convErr := parseFormat(cur); convErr != nil {
				return convErr
			} else if v > SchemaVersion {
				return fmt.Errorf("%w (format %d, this build reads %d)", errNewerFormat, v, SchemaVersion)
			}
		}
		return meta.Put(formatKey, []byte(fmt.Sprint(SchemaVersion)))
	})
	if err != nil {
		return err
	}
	return s.load()
}

func parseFormat(b []byte) (int, error) {
	var v int
	if _, err := fmt.Sscan(string(b), &v); err != nil {
		return 0, fmt.Errorf("unreadable format marker %q", b)
	}
	return v, nil
}

// load reads every collection into memory. Unusable records or collections are
// dropped (and removed from disk) rather than failing the whole database: the
// caller embeds them again from the items it still has.
func (s *Store) load() error {
	type dropped struct {
		name string
		ids  []string
		all  bool
	}
	var drops []dropped

	err := s.dbView(func(tx *bolt.Tx) error {
		meta, vec := tx.Bucket(metaBucket), tx.Bucket(vecBucket)
		known := map[string]bool{}
		err := meta.ForEach(func(k, v []byte) error {
			key := string(k)
			if !strings.HasPrefix(key, metaPrefix) {
				return nil
			}
			name := strings.TrimPrefix(key, metaPrefix)
			known[name] = true
			var m collMeta
			if json.Unmarshal(v, &m) != nil || m.Dim <= 0 || m.Dim > maxDim || m.Schema != SchemaVersion {
				drops = append(drops, dropped{name: name, all: true})
				return nil
			}
			c := newCollection(name, m.Dim, m.Model)
			b := vec.Bucket([]byte(name))
			d := dropped{name: name}
			if b != nil {
				cerr := b.ForEach(func(k, v []byte) error {
					r, err := decodeRecord(string(k), v, m.Dim)
					if err != nil {
						d.ids = append(d.ids, string(k))
						return nil
					}
					c.set(r.id, r.hash, r.q, r.scale, r.inv)
					return nil
				})
				if cerr != nil {
					return cerr
				}
			}
			if len(d.ids) > 0 {
				drops = append(drops, d)
			}
			s.cols[name] = c
			return nil
		})
		if err != nil {
			return err
		}
		// vector buckets that no collection claims
		return vec.ForEachBucket(func(k []byte) error {
			if !known[string(k)] {
				drops = append(drops, dropped{name: string(k), all: true})
			}
			return nil
		})
	})
	if err != nil {
		return err
	}
	if len(drops) == 0 {
		return nil
	}

	for _, d := range drops {
		if d.all {
			s.report.DroppedCollections = append(s.report.DroppedCollections, d.name)
			delete(s.cols, d.name)
		} else {
			s.report.DroppedRecords += len(d.ids)
		}
	}
	log.Warnf("semantic store: dropped %d collections and %d damaged records, they will be embedded again", len(s.report.DroppedCollections), s.report.DroppedRecords)
	return s.dbUpdate(func(tx *bolt.Tx) error {
		for _, d := range drops {
			if d.all {
				if err := dropCollection(tx, d.name); err != nil {
					return err
				}
				continue
			}
			if err := deleteRecords(tx, d.name, d.ids); err != nil {
				return err
			}
		}
		return nil
	})
}

func vecSub(tx *bolt.Tx, name string) (*bolt.Bucket, error) {
	return tx.Bucket(vecBucket).CreateBucketIfNotExists([]byte(name))
}

// resetCollection replaces the collection with an empty one of the given shape.
func resetCollection(tx *bolt.Tx, name string, dim int, model string) error {
	vec := tx.Bucket(vecBucket)
	if vec.Bucket([]byte(name)) != nil {
		if err := vec.DeleteBucket([]byte(name)); err != nil {
			return err
		}
	}
	if _, err := vec.CreateBucket([]byte(name)); err != nil {
		return err
	}
	raw, err := json.Marshal(collMeta{Dim: dim, Model: model, Schema: SchemaVersion})
	if err != nil {
		return err
	}
	return tx.Bucket(metaBucket).Put([]byte(metaPrefix+name), raw)
}

func dropCollection(tx *bolt.Tx, name string) error {
	vec := tx.Bucket(vecBucket)
	if vec.Bucket([]byte(name)) != nil {
		if err := vec.DeleteBucket([]byte(name)); err != nil {
			return err
		}
	}
	return tx.Bucket(metaBucket).Delete([]byte(metaPrefix + name))
}

func putRecords(tx *bolt.Tx, name string, recs []record) error {
	b, err := vecSub(tx, name)
	if err != nil {
		return err
	}
	for i := range recs {
		if err := b.Put([]byte(recs[i].id), encodeRecord(recs[i])); err != nil {
			return err
		}
	}
	return nil
}

func deleteRecords(tx *bolt.Tx, name string, ids []string) error {
	b := tx.Bucket(vecBucket).Bucket([]byte(name))
	if b == nil {
		return nil
	}
	for _, id := range ids {
		if err := b.Delete([]byte(id)); err != nil {
			return err
		}
	}
	return nil
}
