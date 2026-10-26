package indexer

import (
	"time"

	"github.com/AvengeMedia/dankgo/log"
	"github.com/duxlabs-dev/dux-search/internal/metastore"
)

const (
	defaultBatchSize     = 500
	defaultBatchBytes    = 16 << 20
	defaultBatchInterval = 2 * time.Second
	batchQueueDepth      = 64
)

type batchJob struct {
	path string
	doc  *Document
	meta metastore.FileMeta
}

type batcher struct {
	idx      *Indexer
	in       chan batchJob
	done     chan struct{}
	size     int
	maxBytes int
	interval time.Duration
}

func newBatcher(idx *Indexer, size, maxBytes int, interval time.Duration) *batcher {
	b := &batcher{
		idx:      idx,
		in:       make(chan batchJob, batchQueueDepth),
		done:     make(chan struct{}),
		size:     size,
		maxBytes: maxBytes,
		interval: interval,
	}
	go b.run()
	return b
}

func (b *batcher) submit(job batchJob) {
	b.in <- job
}

func (b *batcher) close() {
	close(b.in)
	<-b.done
}

func (b *batcher) run() {
	defer close(b.done)

	idx := b.idx.currentIndex()
	batch := idx.NewBatch()
	pending := make([]batchJob, 0, b.size)
	pendingBytes := 0

	ticker := time.NewTicker(b.interval)
	defer ticker.Stop()

	flush := func() {
		if len(pending) == 0 {
			return
		}
		switch err := idx.Batch(batch); {
		case err != nil:
			log.Warnf("batch submit failed for %d docs: %v", len(pending), err)
		default:
			entries := make(map[string]metastore.FileMeta, len(pending))
			for _, job := range pending {
				entries[job.path] = job.meta
			}
			if err := b.idx.meta.PutBatch(entries); err != nil {
				log.Warnf("failed to update metastore for %d docs: %v", len(entries), err)
			}
			b.idx.indexComplete.Store(true)
		}
		batch = idx.NewBatch()
		pending = pending[:0]
		pendingBytes = 0
	}

	for {
		select {
		case job, ok := <-b.in:
			if !ok {
				flush()
				return
			}
			if err := batch.Index(job.path, job.doc); err != nil {
				log.Warnf("failed to stage %s in batch: %v", job.path, err)
				continue
			}
			pending = append(pending, job)
			pendingBytes += len(job.doc.Body)
			if len(pending) >= b.size || pendingBytes >= b.maxBytes {
				flush()
			}
		case <-ticker.C:
			flush()
		}
	}
}
