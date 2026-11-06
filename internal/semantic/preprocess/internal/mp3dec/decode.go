// Copyright 2017 Hajime Hoshi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package mp3dec is github.com/hajimehoshi/go-mp3 v0.3.4 (Apache-2.0) reduced to
// a streaming decoder that returns float32 samples. See NOTICE.md for the changes.
package mp3dec

import (
	"io"

	"github.com/duxlabs-dev/dux-search/internal/semantic/preprocess/internal/mp3dec/consts"
	"github.com/duxlabs-dev/dux-search/internal/semantic/preprocess/internal/mp3dec/frame"
)

// A Decoder decodes an MP3 stream on the fly.
type Decoder struct {
	source     *source
	sampleRate int
	buf        []float32
	frame      *frame.Frame
}

func (d *Decoder) readFrame() error {
	var err error
	d.frame, _, err = frame.Read(d.source, d.source.pos, d.frame)
	if err != nil {
		if err == io.EOF {
			return io.EOF
		}
		if _, ok := err.(*consts.UnexpectedEOF); ok {
			return io.EOF
		}
		return err
	}
	d.buf = append(d.buf, d.frame.DecodeFloat32()...)
	return nil
}

// Read fills dst with interleaved stereo float32 samples (left, right; a mono
// stream is duplicated) and returns the number of values written. It returns
// io.EOF when the stream ends.
func (d *Decoder) Read(dst []float32) (int, error) {
	for len(d.buf) == 0 {
		if err := d.readFrame(); err != nil {
			return 0, err
		}
	}
	n := copy(dst, d.buf)
	d.buf = d.buf[n:]
	return n, nil
}

// SampleRate returns the sampling frequency of the first frame.
func (d *Decoder) SampleRate() int { return d.sampleRate }

// NewDecoder starts decoding r. Leading ID3v2 tags (and an ID3v1 tag) are skipped.
func NewDecoder(r io.Reader) (*Decoder, error) {
	d := &Decoder{source: &source{reader: r}}
	if err := d.source.skipTags(); err != nil {
		return nil, err
	}
	if err := d.readFrame(); err != nil {
		return nil, err
	}
	freq, err := d.frame.SamplingFrequency()
	if err != nil {
		return nil, err
	}
	d.sampleRate = freq
	return d, nil
}
