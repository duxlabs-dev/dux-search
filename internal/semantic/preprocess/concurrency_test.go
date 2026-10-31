package preprocess

import (
	"bytes"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// Image and Audio are called from the indexer's worker goroutines: concurrent
// use must be safe (shared state is the mel tables and the resampler cache).
func TestConcurrentUse(t *testing.T) {
	img, err := os.ReadFile(filepath.Join("testdata", "images", "Clun_and_the_Clun_valley_1295153.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	rates := []string{"A_anaa.wav", "French_What_is_your_phone_number.wav", "Contact_type_8k.wav", "Hold_recording_11k_u8_stereo.wav", "Walk_Sign_22k.flac"}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 3; i++ {
				if _, err := Image(bytes.NewReader(img)); err != nil {
					t.Error(err)
				}
				if _, err := Audio(filepath.Join("testdata", "audio", rates[(g+i)%len(rates)])); err != nil {
					t.Error(err)
				}
			}
		}(g)
	}
	wg.Wait()
}
