# Test data and reference generation

The tests in `..` compare this package with the Hugging Face `transformers`
implementation of the EmbeddingGemma 2 processor on **real files**:

* `images/` and `audio/` hold the input files (sizes: images 3.2 MB, audio 2.6 MB).
  `manifest.json` in each folder gives file, source page, licence and notes.
* `ref/` holds what `transformers` produced from them (8.3 MB):
  `images.json`/`audio.json` (summary, versions), `img_NN.png` (the resized picture
  or, for decode-level entries, the decoded picture), `img_NN.pix.gz`/`.pos.gz` (raw
  padded `pixel_values` and `image_position_ids` for three small pictures),
  `aud_NN.mel.gz`/`.mask.gz` (float32 log-mel and mask).
* `processor_config.json` is a copy of the one shipped with
  `onnx-community/embeddinggemma-2-ONNX` (Apache-2.0); `gen_reference.py` reads it.

Regenerate with a venv holding the packages listed below:

    python gen_reference.py            # images and audio (about 10 s on CPU)

## Reference versions

Images: transformers 5.22.1, torch 2.15.0+cpu, torchvision 0.29.2+cpu, pillow 13.0.0, numpy 2.5.4.
Audio: transformers 5.22.1, librosa 1.0.0, soundfile 0.14.0, libsndfile 1.2.2, soxr 1.1.0, numpy 2.5.4.

transformers 5.22.1 is the current release that contains `Gemma4ImageProcessor`
and `Gemma4AudioFeatureExtractor` (checked on PyPI, 2026-10-30; the
`EmbeddingGemma2Processor` named in `processor_config.json` is not in a release
yet, it composes exactly these two classes with the config's parameters).
`Gemma4ImageProcessor` is the torchvision-backend class; the script also runs the
PIL-backend class `Gemma4ImageProcessorPil` and records the difference between
them (`torchvision_vs_pil_*` in `ref/images.json`).

## What is compared, and how well it agrees

Run `go test -v ./internal/semantic/preprocess/ -run Reference` for the numbers
of the current build.

### Images (`TestImageReference`, 33 pictures, 20 through the whole processor)

* `pixel_position_ids`, `num_soft_tokens_per_image` and the patch count: **identical**
  for every picture; the target size follows `get_aspect_ratio_preserving_size`
  (also checked on 30 shapes including 1x10000, 13441x1 and the zero-side edge cases).
* Decoded pictures (`decode` entries, before the resize): png, gif (incl. the first
  frame of animations), bmp, webp lossy and lossless, EXIF orientations 2-8 on webp
  and jpeg: **bit-identical to Pillow** for every non-JPEG format, including the
  libwebp "fancy" chroma upsampler and its YUV conversion, reproduced in `yuv.go`.
* JPEG: libjpeg-turbo's chroma upsampling (h2v1/h2v2/h1v2 "fancy") and YCC
  conversion are reproduced; the remaining difference is Go's `image/jpeg` IDCT
  against libjpeg-turbo's: at most 3 levels (of 255) on about 2 % of the samples,
  mean 0.03 levels. CMYK (Adobe) matches within 2.
* Resize: the algorithm of Pillow's `Image.resize(BICUBIC)` (antialiased, 22-bit
  fixed point) is implemented exactly; for lossless sources the output equals the
  PIL backend of `Gemma4ImageProcessor` (max 0 levels). The reference processor
  uses torchvision, whose uint8 kernel quantises the filter weights differently:
  against it the difference is **at most 2 levels, mean <= 0.007** (about 0.1 % of
  the values differ by one level). Python itself shows the same spread between its
  two backends (`torchvision_vs_pil_max_level` is 1-2).
* Tolerances in the test: lossless sources 2 levels / mean 0.01; JPEG 5 levels /
  mean 0.1 (measured worst case 4 levels, mean 0.04). One level is 0.0039 of
  the [0,1] range.

### Audio (`TestAudioReference`, 16 clips)

* mel frame count, `input_features_mask` and the soft-token count are **identical**
  for every clip (the clips cover 0.5 s to 34 s, mono/stereo, 8 to 48 kHz).
* 16 kHz WAV (no resampling, exact decoding): log-mel identical to 4 decimals,
  i.e. the feature extractor (Hann window, 512-point FFT, HTK mel bank, log with
  floor 1e-3, semicausal padding, mask rule) matches `Gemma4AudioFeatureExtractor`.
* Resampled WAV/FLAC: libsndfile -> librosa -> **soxr_hq** is the reference's
  resampler. libsoxr is C, so `resample.go` implements a Kaiser polyphase filter
  to soxr_hq's specification (passband to 91.3 %, stopband from 100 % of the lower
  Nyquist frequency, 130 dB). Output lengths equal soxr's exactly; waveform SNR
  against soxr is 68-130 dB; the log-mel difference is at most 0.063, mean 0.0005
  (the transition band 7.3-8 kHz is the only place the two filters differ).
* Vorbis: `jfreymuth/oggvorbis` equals libvorbis to float rounding (130+ dB SNR) on
  six of seven streams; on `Help_Sortable...ogg` one sample (index 126030 at 44.1
  kHz) is 0.016 off, which is the 0.25 maximum log-mel difference there (mean 0.0003).
* MP3: pure-Go decoding is `hajimehoshi/go-mp3`, vendored and trimmed as
  `internal/mp3dec` (see its NOTICE.md) because upstream returns 16-bit PCM
  truncated toward zero while mpg123 (libsndfile) returns floats: that
  quantisation noise (3e-5) moved the log of near-silent mel bins (floor 1e-3) so much
  that, through the int8 audio tower, token rows of a 6 s clip reached cosine 0.5
  (mean 0.974); with float output the same clip is at 0.99999 (mean row cosine),
  log-mel max 0.021 / mean 0.0002. Length and alignment match libsndfile exactly
  because the gapless rules of mpg123 (Xing/Info frame dropped, LAME encoder delay +
  529 trimmed at the start, padding at the end) are applied in `decodeMP3`.
* Tolerances in the test: see `audioTolerance`.

### Tower level (one-off check, not part of `go test`)

The Go tensors of five pictures and five clips were run through the int8
`vision_encoder`/`audio_encoder` of the model with ONNX Runtime 1.31.1 (Python) and compared
with the same models fed by `transformers`' tensors (2026-10-30):

* `image_features`: mean row cosine 0.998-0.99999, mean-pooled cosine >= 0.99994 on all five
  (jpeg, png, webp). A few rows fall lower (min 0.51 on the PNG, 0.73 on a JPEG): the same
  happens between `transformers`' own torchvision and PIL backends (min 0.5128 on that PNG),
  i.e. single patch rows of the int8 tower flip on one-level input differences. Feeding the
  unpadded tensors or the 2520-patch padded layout gives identical features (cosine 1.000000,
  max difference 0.0), so the tower input can be unpadded.
* `audio_features`: 16 kHz WAV identical (max difference 0.0); resampled/decoded clips (flac,
  mp3, vorbis) mean row cosine >= 0.9994, min row >= 0.997.

### Not covered by a real file

Documented behaviour that has no real-file test (no suitable open file found):
JPEG 4:4:0/4:1:1 chroma, GIF whose first frame is smaller than the logical
screen, 16-bit grey PNG (unit test of the PIL rule instead), wav with A-law/mu-law/
8-bit/float64 (unit tests on constructed headers).

## Sources and licences

All input files are public domain / CC0, or test fixtures of the Pillow repository
(MIT-CMU licence). Derived audio files (marked) were cut or converted with ffmpeg
from the CC0 originals, which allows it.

### Images

| file | level | notes | licence | source |
|---|---|---|---|---|
| `Policistky_17_10_2012.jpg` | full | camera JPEG, 4:2:2, 2592x1944 | CC0 | https://commons.wikimedia.org/wiki/File:!Policistky_17_10_2012.jpg |
| `Clun_and_the_Clun_valley_1295153.jpg` | full | JPEG 4:2:0, odd height 707 | CC0 | https://commons.wikimedia.org/wiki/File:%22_Clun%22_and_the_Clun_valley_(1295153).jpg |
| `Maison_Nayrac_37_place_du_Marchal_Leclerc_-_09_-_Mirepoix.jpg` | full | tiny JPEG 102x160 (upscaled to the budget) | CC0 | https://commons.wikimedia.org/wiki/File:%22_Maison_Nayrac%22_37_place_du_Mar%C3%A9chal_Leclerc_-_09_-_Mirepoix.jpg |
| `Screenshot_Earwigs_Copyvio_error.jpg` | full | JPEG screenshot | CC0 | https://commons.wikimedia.org/wiki/File:Screenshot_Earwigs_Copyvio_error.jpg |
| `Open-Xchange_Screenshot_Einstellungsmen.png` | full | PNG screenshot, RGBA | CC0 | https://commons.wikimedia.org/wiki/File:Open-Xchange_Screenshot_Einstellungsmen%C3%BC.png |
| `OsmAnd_Wikivoyage_Integration_2019.png` | full | portrait PNG phone screenshot, RGB | CC0 | https://commons.wikimedia.org/wiki/File:OsmAnd_Wikivoyage_Integration_2019.png |
| `CW_59_Example.png` | full | small wide PNG, RGBA | CC0 | https://commons.wikimedia.org/wiki/File:CW_59_Example.png |
| `CodeSandbox_screenshot.webp` | full | 5120x2653 lossy WebP screenshot | CC0 | https://commons.wikimedia.org/wiki/File:CodeSandbox_screenshot.webp |
| `hopper.gif` | decode | GIF palette | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper.gif |
| `hopper.bmp` | decode | BMP 24-bit | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper.bmp |
| `hopper_gray.jpg` | full | grayscale JPEG | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_gray.jpg |
| `pil_sample_cmyk.jpg` | full | CMYK (Adobe) JPEG | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/pil_sample_cmyk.jpg |
| `pil_sample_rgb.jpg` | full | JPEG 4:4:4 Adobe | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/pil_sample_rgb.jpg |
| `flower.jpg` | full | JPEG 4:2:0 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/flower.jpg |
| `flower.webp` | decode | lossy WebP | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/flower.webp |
| `hopper_orientation_2.jpg` | decode | EXIF orientation 2 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_2.jpg |
| `hopper_orientation_3.jpg` | decode | EXIF orientation 3 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_3.jpg |
| `hopper_orientation_4.jpg` | decode | EXIF orientation 4 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_4.jpg |
| `hopper_orientation_5.jpg` | decode | EXIF orientation 5 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_5.jpg |
| `hopper_orientation_6.jpg` | full | EXIF orientation 6 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_6.jpg |
| `hopper_orientation_7.jpg` | decode | EXIF orientation 7 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_7.jpg |
| `hopper_orientation_8.jpg` | decode | EXIF orientation 8 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_8.jpg |
| `hopper_orientation_6.webp` | decode | WebP with EXIF orientation 6 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/hopper_orientation_6.webp |
| `transparent.png` | full | RGBA PNG with transparency | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/transparent.png |
| `transparent.webp` | full | WebP with alpha | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/transparent.webp |
| `transparent.gif` | decode | GIF with a transparent index | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/transparent.gif |
| `pil123p.png` | full | palette PNG with tRNS | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/pil123p.png |
| `l_trns.png` | full | grey PNG with tRNS | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/l_trns.png |
| `tiff_16bit_RGB_target.png` | full | 16-bit RGB PNG | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/tiff_16bit_RGB_target.png |
| `iss634.gif` | decode | animated GIF, first frame | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/iss634.gif |
| `iss634.webp` | decode | animated WebP, first frame | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/iss634.webp |
| `test.colors.gif` | full | GIF 807x606 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/test.colors.gif |
| `orientation_rectangle.jpg` | full | 2x1 JPEG with orientation 8 | MIT-CMU (Pillow test fixtures) | https://github.com/python-pillow/Pillow/blob/main/Tests/images/orientation_rectangle.jpg |

`full` = the whole processor output is compared; `decode` = the decoded RGB picture only.

### Audio

| file | notes | licence | source |
|---|---|---|---|
| `A_anaa.wav` | 44.1 kHz mono 16-bit WAV | CC0 | https://commons.wikimedia.org/wiki/File:A%27anaa.wav |
| `Alarm_Clock_Directory.Audio_.mp3` | 44.1 kHz stereo MP3 with LAME/Lavf gapless header | CC0 | https://commons.wikimedia.org/wiki/File:Alarm_Clock_(Directory.Audio).mp3 |
| `Chocha_perdiz_Scolopax_rusticola_y_Mosquitero_comun_Phylloscopus_collybita_.mp3` | 48 kHz stereo MP3 | CC0 | https://commons.wikimedia.org/wiki/File:Chocha_perdiz_(Scolopax_rusticola)_y_Mosquitero_com%C3%BAn_(Phylloscopus_collybita).mp3 |
| `Citation_sound_effect_-_the_English_word_citation_sung_by_a_soprano_singer.wav` | 44.1 kHz stereo 24-bit WAV | CC0 | https://commons.wikimedia.org/wiki/File:Citation_sound_effect_-_the_English_word_citation_sung_by_a_soprano_singer.wav |
| `Contact_type_8k.wav` | derived from Contact_type.ogg: ffmpeg -t 5 -ar 8000 -ac 1 -c:a pcm_s16le (5 s, 8 kHz telephone-rate WAV) | CC0 (derived) | https://commons.wikimedia.org/wiki/File:Contact_type.ogg |
| `En-us-phone_number.ogg` | 44.1 kHz stereo Vorbis, 0.51 s: shortest clip | CC0 | https://commons.wikimedia.org/wiki/File:En-us-phone_number.ogg |
| `Fake_Metal_Bat.wav` | 48 kHz stereo float32 WAV | CC0 | https://commons.wikimedia.org/wiki/File:Fake_Metal_Bat.wav |
| `Fake_Metal_Bat_stereo_24bit.flac` | derived from Fake_Metal_Bat.wav: ffmpeg -c:a flac -sample_fmt s32 -bits_per_raw_sample 24 (24-bit stereo FLAC with inter-channel decorrelation) | CC0 (derived) | https://commons.wikimedia.org/wiki/File:Fake_Metal_Bat.wav |
| `French_What_is_your_phone_number.wav` | 24 kHz mono 16-bit WAV | CC0 | https://commons.wikimedia.org/wiki/File:French_What_is_your_phone_number.wav |
| `Hold_recording_11k_u8_stereo.wav` | derived from Hold-recording-italy.ogg: ffmpeg -ss 3 -t 5 -ar 11025 -ac 2 -c:a pcm_u8 (8-bit unsigned stereo WAV, 5 s) | CC0 (derived) | https://commons.wikimedia.org/wiki/File:Hold-recording-italy.ogg |
| `Lb-jaizen1.mp3` | 44.1 kHz mono MP3, no gapless header | CC0 | https://commons.wikimedia.org/wiki/File:Lb-jaizen1.mp3 |
| `Or-.flac` | 44.1 kHz mono FLAC | CC0 | https://commons.wikimedia.org/wiki/File:Or-%E0%AC%9A%E0%AC%89%E0%AC%A4%E0%AC%BE%E0%AC%B0%E0%AC%BF.flac |
| `Speech_Prosody_audio_example.wav` | 16 kHz mono 16-bit WAV: no resampling | CC0 | https://commons.wikimedia.org/wiki/File:Speech_Prosody_audio_example.wav |
| `Walk_Sign_22k.flac` | derived from Walk_Sign_to_Cross_Sixth.ogg: ffmpeg -t 5 -ar 22050 -ac 1 -sample_fmt s16 -c:a flac (22.05 kHz FLAC) | Public domain (derived) | https://commons.wikimedia.org/wiki/File:Walk_Sign_to_Cross_Sixth.ogg |
| `Walk_Sign_to_Cross_Sixth.ogg` | 48 kHz mono Vorbis | Public domain | https://commons.wikimedia.org/wiki/File:Walk_Sign_to_Cross_Sixth.ogg |
| `Help_Sortable_tables_Sorting_buttons_in_a_separate_row.ogg` | 44.1 kHz stereo Vorbis, 34 s: longer than the 11.2 s limit; the jfreymuth/oggvorbis decoder deviates from libvorbis on one sample of it (see README) | CC0 | https://commons.wikimedia.org/wiki/File:Help_Sortable_tables_Sorting_buttons_in_a_separate_row.ogg |
