package preprocess

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestMalformedFiles feeds truncated and corrupted copies of the real test
// files to the decoders: whatever the third-party decoders do with them, the
// package must return (a result or an error) and never panic or hang.
func TestMalformedFiles(t *testing.T) {
	images := []string{"Clun_and_the_Clun_valley_1295153.jpg", "OsmAnd_Wikivoyage_Integration_2019.png", "iss634.gif", "iss634.webp", "flower.webp", "hopper.bmp", "pil_sample_cmyk.jpg", "hopper_orientation_6.jpg"}
	audios := []string{"Speech_Prosody_audio_example.wav", "Or-.flac", "Fake_Metal_Bat_stereo_24bit.flac", "Alarm_Clock_Directory.Audio_.mp3", "Lb-jaizen1.mp3", "Walk_Sign_to_Cross_Sixth.ogg", "En-us-phone_number.ogg"}
	rng := rand.New(rand.NewSource(7))
	variants := func(b []byte) [][]byte {
		out := [][]byte{b[:len(b)/4], b[:len(b)/2], b[:len(b)*3/4], b[:len(b)-1], b[:8]}
		for i := 0; i < 8; i++ {
			c := append([]byte(nil), b...)
			// Corrupt a short run somewhere past the first bytes (headers
			// included for every other variant).
			start := 0
			if i%2 == 1 {
				start = 12
			}
			at := start + rng.Intn(len(c)-start-16)
			for j := 0; j < 1+rng.Intn(16); j++ {
				c[at+j] = byte(rng.Intn(256))
			}
			out = append(out, c)
		}
		return out
	}
	deadline := time.Now().Add(60 * time.Second)
	for _, name := range images {
		data, err := os.ReadFile(filepath.Join("testdata", "images", name))
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range variants(data) {
			if time.Now().After(deadline) {
				t.Fatal("malformed-input test too slow")
			}
			_, _ = Image(bytes.NewReader(v)) // must return
			_ = i
		}
	}
	for _, name := range audios {
		data, err := os.ReadFile(filepath.Join("testdata", "audio", name))
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range variants(data) {
			if time.Now().After(deadline) {
				t.Fatal("malformed-input test too slow")
			}
			_, _, _ = decodeAudio(bytes.NewReader(v))
		}
	}
}

func TestGuardConvertsPanics(t *testing.T) {
	err := func() (err error) {
		defer guard("test", &err)
		var a []int
		_ = a[3]
		return nil
	}()
	if err == nil {
		t.Fatal("panic not converted to an error")
	}
}
