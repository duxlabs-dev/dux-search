package preprocess

import (
	"errors"
	"fmt"
)

var (
	// ErrUnsupported means the file format (or a variant of it) is not
	// handled; callers skip the file for semantic indexing.
	ErrUnsupported = errors.New("preprocess: unsupported format")

	// ErrTooLarge means the image exceeds MaxPixels, MaxDecodeBytes or
	// MaxImageBytes and was rejected before any pixel was decoded.
	ErrTooLarge = errors.New("preprocess: image too large")

	// ErrEmpty means the audio contains no samples, or fewer than one audio
	// token's worth.
	ErrEmpty = errors.New("preprocess: no audio samples")
)

// guard turns a panic inside a third-party decoder (corrupt input can trip
// index errors in them) into an ordinary error, so one bad file cannot take
// the long-running indexer down.
func guard(what string, err *error) {
	if r := recover(); r != nil {
		*err = fmt.Errorf("preprocess: %s decoder failed on malformed input: %v", what, r)
	}
}
