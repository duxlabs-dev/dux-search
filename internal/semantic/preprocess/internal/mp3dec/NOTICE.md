# mp3dec

A trimmed fork of [github.com/hajimehoshi/go-mp3](https://github.com/hajimehoshi/go-mp3)
v0.3.4 (Apache License 2.0, see LICENSE and AUTHORS), the pure-Go MPEG Layer III
decoder, made for `preprocess`'s audio path.

Changes against upstream:

* `frame.Frame.DecodeFloat32` replaces `Decode`: the synthesis filterbank output is
  returned as float32 in [-1, 1] instead of being scaled by 32767 and truncated to
  16 bits. The reference pipeline decodes with mpg123 to floats; the 16-bit
  truncation changed the log-mel of quiet passages (the floor is 1e-3) and, through
  the int8 audio tower, whole token rows (cosine down to 0.5 on a 6 s clip).
* `Decoder` keeps only `NewDecoder`, `Read` (float32) and `SampleRate`; seeking,
  length scanning and the `io.Seeker` requirement are gone (the stream is read once,
  front to back, and stopped early).
* `consts.UnexpectedEOF` literals are keyed (go vet).
* Import paths point at this directory; tests and the oto-based example are dropped.
* Lint: the unused main data size computation in `sideinfo.Read` is removed and two
  error strings lose their trailing punctuation (staticcheck ST1005).

The decoding algorithm itself is untouched.
