package vecstore

import (
	"encoding/json"
	"fmt"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	bolt "go.etcd.io/bbolt"
)

// The item bucket keeps the *desired* content of caller-owned collections
// (collections.set): the vectors are derived from it, so a model that arrives
// later, a model change or a damaged vector file can be repaired by embedding
// the items again without asking the client to resend them.

// PutItems replaces the stored items of the collection.
func (s *Store) PutItems(name string, items []semantic.Item) error {
	if name == "" {
		return fmt.Errorf("vecstore: empty collection name")
	}
	encoded := make([][2][]byte, len(items))
	for i, it := range items {
		if it.ID == "" || len(it.ID) > maxKeyLen {
			return fmt.Errorf("vecstore: item %d has an empty or oversized id", i)
		}
		raw, err := json.Marshal(it)
		if err != nil {
			return err
		}
		encoded[i] = [2][]byte{[]byte(it.ID), raw}
	}
	s.wmu.Lock()
	defer s.wmu.Unlock()
	if err := s.checkOpenLocked(); err != nil {
		return err
	}
	return s.dbUpdate(func(tx *bolt.Tx) error {
		root := tx.Bucket(itemsBucket)
		if root.Bucket([]byte(name)) != nil {
			if err := root.DeleteBucket([]byte(name)); err != nil {
				return err
			}
		}
		b, err := root.CreateBucket([]byte(name))
		if err != nil {
			return err
		}
		for _, kv := range encoded {
			if err := b.Put(kv[0], kv[1]); err != nil {
				return err
			}
		}
		return nil
	})
}

// Items returns the stored items ordered by id. Unreadable entries are skipped.
func (s *Store) Items(name string) ([]semantic.Item, error) {
	var out []semantic.Item
	err := s.dbView(func(tx *bolt.Tx) error {
		b := tx.Bucket(itemsBucket).Bucket([]byte(name))
		if b == nil {
			return nil
		}
		return b.ForEach(func(k, v []byte) error {
			var it semantic.Item
			if err := json.Unmarshal(v, &it); err != nil || it.ID != string(k) {
				log.Warnf("semantic store: skipping damaged item %q of %q", k, name)
				return nil
			}
			out = append(out, it)
			return nil
		})
	})
	return out, err
}

// DeleteItems removes every stored item of the collection.
func (s *Store) DeleteItems(name string) error {
	s.wmu.Lock()
	defer s.wmu.Unlock()
	if err := s.checkOpenLocked(); err != nil {
		return err
	}
	return s.dbUpdate(func(tx *bolt.Tx) error {
		root := tx.Bucket(itemsBucket)
		if root.Bucket([]byte(name)) == nil {
			return nil
		}
		return root.DeleteBucket([]byte(name))
	})
}

// ItemCollections lists the collections that have stored items.
func (s *Store) ItemCollections() ([]string, error) {
	var out []string
	err := s.dbView(func(tx *bolt.Tx) error {
		return tx.Bucket(itemsBucket).ForEachBucket(func(k []byte) error {
			out = append(out, string(k))
			return nil
		})
	})
	return out, err
}

func (s *Store) checkOpenLocked() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.checkOpen()
}
