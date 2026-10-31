// Package vecstore is the semantic.VectorStore of dsearch: every collection is
// held in memory as one contiguous []int8 block (symmetric per-vector
// quantization) and scanned with an int32 dot product; bbolt only makes the
// data survive restarts and is never read while searching.
package vecstore

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	bolt "go.etcd.io/bbolt"
)

// SchemaVersion is the on-disk record format. A collection written with another
// version is dropped on load and embedded again.
const SchemaVersion = 1

// DefaultParallelThreshold is the collection size from which a scan is split
// across goroutines; below it the scheduling cost outweighs the gain.
const DefaultParallelThreshold = 200_000

var (
	// ErrNoCollection is returned by Upsert when the collection was never ensured.
	ErrNoCollection = errors.New("vecstore: collection does not exist")
	// ErrDimMismatch is returned when a vector does not match the collection dimension.
	ErrDimMismatch = errors.New("vecstore: vector dimension does not match collection")
	// ErrInvalidVector is returned for empty, zero, NaN or infinite vectors.
	ErrInvalidVector = errors.New("vecstore: invalid vector")
	// ErrClosed is returned after Close.
	ErrClosed = errors.New("vecstore: store is closed")
)

// Options tune a Store. The zero value is valid.
type Options struct {
	// ParallelThreshold is the row count from which a search uses several
	// goroutines (default DefaultParallelThreshold).
	ParallelThreshold int
	// Workers is the number of scan goroutines (default GOMAXPROCS).
	Workers int
	// RecoverCorrupt moves an unreadable database aside (<path>.corrupt) and
	// starts an empty one instead of failing. Everything in it is derived data
	// that gets embedded again, so this is the right behavior for a daemon.
	RecoverCorrupt bool
}

// CollectionInfo describes one collection.
type CollectionInfo struct {
	Name   string `json:"name"`
	Dim    int    `json:"dim"`
	Model  string `json:"model"`
	Schema int    `json:"schema"`
	Count  int    `json:"count"`
}

// LoadReport says what Open had to throw away. All zero on a healthy file.
type LoadReport struct {
	Recovered           bool     // the file was unreadable and replaced
	DroppedCollections  []string // schema/meta unusable, will be embedded again
	DroppedRecords      int      // single records that failed validation
	RecoveredFromBackup string   // path of the moved-aside corrupt file
}

type collection struct {
	name   string
	dim    int
	model  string
	schema int

	ids    []string
	index  map[string]int
	hashes []string
	data   []int8    // len(ids)*dim
	scale  []float32 // quantization scale, kept so the float vector can be rebuilt
	inv    []float32 // 1/||q||, makes the dot product an exact cosine of the quantized vectors
	sum    []float64 // sum of the dequantized rows, the centroid times len(ids)
}

func newCollection(name string, dim int, model string) *collection {
	return &collection{name: name, dim: dim, model: model, schema: SchemaVersion, index: map[string]int{}}
}

func (c *collection) len() int { return len(c.ids) }

func (c *collection) row(i int) []int8 { return c.data[i*c.dim : (i+1)*c.dim] }

func (c *collection) addToSum(q []int8, scale float32, sign float64) {
	if c.sum == nil {
		c.sum = make([]float64, c.dim)
	}
	f := sign * float64(scale)
	for i, x := range q {
		c.sum[i] += f * float64(x)
	}
}

func (c *collection) set(id, hash string, q []int8, scale, inv float32) {
	c.addToSum(q, scale, 1)
	if i, ok := c.index[id]; ok {
		c.addToSum(c.row(i), c.scale[i], -1)
		copy(c.row(i), q)
		c.hashes[i], c.scale[i], c.inv[i] = hash, scale, inv
		return
	}
	c.index[id] = len(c.ids)
	c.ids = append(c.ids, id)
	c.hashes = append(c.hashes, hash)
	c.data = append(c.data, q...)
	c.scale = append(c.scale, scale)
	c.inv = append(c.inv, inv)
}

// remove deletes by moving the last row into the gap, keeping the block dense.
func (c *collection) remove(id string) bool {
	i, ok := c.index[id]
	if !ok {
		return false
	}
	c.addToSum(c.row(i), c.scale[i], -1)
	last := len(c.ids) - 1
	if i != last {
		copy(c.row(i), c.row(last))
		c.ids[i], c.hashes[i], c.scale[i], c.inv[i] = c.ids[last], c.hashes[last], c.scale[last], c.inv[last]
		c.index[c.ids[i]] = i
	}
	delete(c.index, id)
	c.ids[last], c.hashes[last] = "", ""
	c.ids, c.hashes = c.ids[:last], c.hashes[:last]
	c.data = c.data[:last*c.dim]
	c.scale, c.inv = c.scale[:last], c.inv[:last]
	return true
}

// Store implements semantic.VectorStore.
type Store struct {
	db  *bolt.DB
	opt Options

	wmu sync.Mutex // serializes writers so disk and memory change in the same order
	mu  sync.RWMutex
	// cols is read under mu; collection contents change only under wmu+mu.
	cols   map[string]*collection
	closed bool
	report LoadReport
}

var _ semantic.VectorStore = (*Store)(nil)

// Open opens or creates the database at path and loads every collection into memory.
func Open(path string, opt Options) (*Store, error) {
	if opt.ParallelThreshold <= 0 {
		opt.ParallelThreshold = DefaultParallelThreshold
	}
	if opt.Workers <= 0 {
		opt.Workers = runtime.GOMAXPROCS(0)
	}
	s := &Store{opt: opt, cols: map[string]*collection{}}

	err := s.openAndLoad(path)
	if err == nil {
		return s, nil
	}
	if !opt.RecoverCorrupt || !isCorruption(err) {
		return nil, err
	}
	backup, mvErr := moveAside(path)
	if mvErr != nil {
		return nil, fmt.Errorf("%w (and the damaged file could not be moved aside: %v)", err, mvErr)
	}
	s = &Store{opt: opt, cols: map[string]*collection{}}
	if err := s.openAndLoad(path); err != nil {
		return nil, err
	}
	s.report.Recovered = true
	s.report.RecoveredFromBackup = backup
	return s, nil
}

// Report returns what Open discarded.
func (s *Store) Report() LoadReport { return s.report }

// Close flushes and closes the database.
func (s *Store) Close() error {
	s.wmu.Lock()
	defer s.wmu.Unlock()
	s.mu.Lock()
	s.closed = true
	s.cols = map[string]*collection{}
	s.mu.Unlock()
	return s.db.Close()
}

func (s *Store) checkOpen() error {
	if s.closed {
		return ErrClosed
	}
	return nil
}

// EnsureCollection creates the collection, or keeps it when dim, model and schema
// still match. When they changed, its vectors are dropped so every item is
// embedded again (Hashes then reports nothing).
func (s *Store) EnsureCollection(name string, dim int, model string) error {
	if name == "" {
		return errors.New("vecstore: empty collection name")
	}
	if dim <= 0 || dim > maxDim {
		return fmt.Errorf("vecstore: invalid dimension %d", dim)
	}
	s.wmu.Lock()
	defer s.wmu.Unlock()

	s.mu.RLock()
	if err := s.checkOpen(); err != nil {
		s.mu.RUnlock()
		return err
	}
	cur := s.cols[name]
	s.mu.RUnlock()
	if cur != nil && cur.dim == dim && cur.model == model && cur.schema == SchemaVersion {
		return nil
	}

	if err := s.dbUpdate(func(tx *bolt.Tx) error { return resetCollection(tx, name, dim, model) }); err != nil {
		return err
	}
	s.mu.Lock()
	s.cols[name] = newCollection(name, dim, model)
	s.mu.Unlock()
	return nil
}

// DropCollection removes the collection and its vectors. Missing is not an error.
func (s *Store) DropCollection(name string) error {
	s.wmu.Lock()
	defer s.wmu.Unlock()
	if err := s.dbUpdate(func(tx *bolt.Tx) error { return dropCollection(tx, name) }); err != nil {
		return err
	}
	s.mu.Lock()
	delete(s.cols, name)
	s.mu.Unlock()
	return nil
}

// Hashes returns id -> content hash of everything stored. Unknown collection: empty.
func (s *Store) Hashes(name string) (map[string]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.checkOpen(); err != nil {
		return nil, err
	}
	c := s.cols[name]
	if c == nil {
		return map[string]string{}, nil
	}
	out := make(map[string]string, c.len())
	for i, id := range c.ids {
		out[id] = c.hashes[i]
	}
	return out, nil
}

// Upsert stores the vectors, replacing entries with the same id. It is all or
// nothing: one invalid vector rejects the batch before anything is written.
func (s *Store) Upsert(name string, vecs []semantic.Vector) error {
	if len(vecs) == 0 {
		return nil
	}
	s.mu.RLock()
	if err := s.checkOpen(); err != nil {
		s.mu.RUnlock()
		return err
	}
	cur := s.cols[name]
	s.mu.RUnlock()
	if cur == nil {
		return fmt.Errorf("%w: %q", ErrNoCollection, name)
	}
	dim := cur.dim // immutable for the life of a collection object

	recs := make([]record, len(vecs))
	for i, v := range vecs {
		if v.ID == "" || len(v.ID) > maxKeyLen {
			return fmt.Errorf("%w: empty or oversized id", ErrInvalidVector)
		}
		if len(v.Hash) > math.MaxUint16 {
			return fmt.Errorf("%w: hash of %q too long", ErrInvalidVector, v.ID)
		}
		if len(v.Vec) != dim {
			return fmt.Errorf("%w: %q has %d values, collection %q has %d", ErrDimMismatch, v.ID, len(v.Vec), name, dim)
		}
		q, scale, inv, err := quantize(v.Vec)
		if err != nil {
			return fmt.Errorf("%w: %q: %v", ErrInvalidVector, v.ID, err)
		}
		recs[i] = record{id: v.ID, hash: v.Hash, q: q, scale: scale, inv: inv}
	}

	s.wmu.Lock()
	defer s.wmu.Unlock()
	s.mu.RLock()
	c := s.cols[name]
	s.mu.RUnlock()
	if c == nil || c != cur {
		// dropped or reset while this batch was being quantized
		return fmt.Errorf("%w: %q changed during upsert", ErrNoCollection, name)
	}
	if err := s.dbUpdate(func(tx *bolt.Tx) error { return putRecords(tx, name, recs) }); err != nil {
		return err
	}
	s.mu.Lock()
	for i := range recs {
		c.set(recs[i].id, recs[i].hash, recs[i].q, recs[i].scale, recs[i].inv)
	}
	s.mu.Unlock()
	return nil
}

// Delete removes the ids. Unknown ids and collections are ignored.
func (s *Store) Delete(name string, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	s.wmu.Lock()
	defer s.wmu.Unlock()
	s.mu.RLock()
	if err := s.checkOpen(); err != nil {
		s.mu.RUnlock()
		return err
	}
	c := s.cols[name]
	s.mu.RUnlock()
	if c == nil {
		return nil
	}
	if err := s.dbUpdate(func(tx *bolt.Tx) error { return deleteRecords(tx, name, ids) }); err != nil {
		return err
	}
	s.mu.Lock()
	for _, id := range ids {
		c.remove(id)
	}
	s.mu.Unlock()
	return nil
}

// Count returns the number of vectors. Unknown collection: 0.
func (s *Store) Count(name string) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := s.checkOpen(); err != nil {
		return 0, err
	}
	if c := s.cols[name]; c != nil {
		return c.len(), nil
	}
	return 0, nil
}

// Info returns the collection metadata.
func (s *Store) Info(name string) (CollectionInfo, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	c := s.cols[name]
	if c == nil {
		return CollectionInfo{}, false
	}
	return c.info(), true
}

func (c *collection) info() CollectionInfo {
	return CollectionInfo{Name: c.name, Dim: c.dim, Model: c.model, Schema: c.schema, Count: c.len()}
}

// Collections lists every collection ordered by name.
func (s *Store) Collections() []CollectionInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]CollectionInfo, 0, len(s.cols))
	for _, c := range s.cols {
		out = append(out, c.info())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Vector rebuilds the stored vector of id as float32 (the quantized values times
// their scale, so within quantization error of what was upserted).
func (s *Store) Vector(name, id string) ([]float32, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	c := s.cols[name]
	if c == nil {
		return nil, false
	}
	i, ok := c.index[id]
	if !ok {
		return nil, false
	}
	out := make([]float32, c.dim)
	for j, q := range c.row(i) {
		out[j] = float32(q) * c.scale[i]
	}
	return out, true
}
