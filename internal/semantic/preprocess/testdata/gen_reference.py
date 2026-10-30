#!/usr/bin/env python3
"""Generates the reference tensors the Go tests compare against.

The Hugging Face `transformers` implementation of the EmbeddingGemma 2
processor (Gemma4ImageProcessor, Gemma4AudioFeatureExtractor, configured by
processor_config.json) is run on the real files under images/ and audio/.
Everything it writes goes to ref/; the Go tests (preprocess_ref_test.go) read
ref/ and compare. See README.md for versions, sources and tolerances.

Usage (venv with torch, torchvision, transformers, pillow, numpy, librosa,
soundfile, soxr):

    python gen_reference.py            # images and audio
    python gen_reference.py images
    python gen_reference.py audio

Image pixel_values are exactly k * float32(1/255) for uint8 k (asserted
below), so the reference is stored as the resized picture (PNG) plus, for a few
pictures, the raw padded tensors; audio mel features are stored as float32.
"""
import gzip
import json
import math
import os
import sys
import warnings

import numpy as np
import PIL
import transformers
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
CFG = json.load(open(os.path.join(HERE, "processor_config.json")))

# Audio is limited to audio_seq_length * audio_ms_per_token (11.2 s), see audio.go.
MAX_AUDIO_SAMPLES = CFG["audio_seq_length"] * CFG["audio_ms_per_token"] * CFG["feature_extractor"]["sampling_rate"] // 1000


# Pictures whose complete padded pixel_values / position ids are stored raw.
RAW_FULL = {"orientation_rectangle.jpg", "test.colors.gif", "CW_59_Example.png"}


def save_gz(name, arr):
    with gzip.GzipFile(os.path.join(REF, name), "wb", compresslevel=9, mtime=0) as f:
        f.write(np.ascontiguousarray(arr).tobytes())


def gen_images():
    import torch
    import torchvision
    from transformers.image_utils import load_image
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor, convert_image_to_patches, get_aspect_ratio_preserving_size
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

    icfg = {k: v for k, v in CFG["image_processor"].items() if k != "image_processor_type"}
    tv = Gemma4ImageProcessor(**icfg)      # torchvision backend: the class named in processor_config.json
    pl = Gemma4ImageProcessorPil(**icfg)   # PIL backend: same class family, PIL resize
    manifest = json.load(open(os.path.join(HERE, "images", "manifest.json")))
    entries = []
    ps = icfg["patch_size"]
    for item in manifest:
        path = os.path.join(HERE, "images", item["file"])
        img = load_image(path)  # exif_transpose + convert("RGB"), what the pipeline does
        if item["level"] == "decode":
            # Decoder / orientation check only: the RGB picture the processor receives.
            idx = len(entries)
            Image.fromarray(np.asarray(img)).save(os.path.join(REF, f"img_{idx:02d}.png"), optimize=True)
            entries.append(dict(file=item["file"], level="decode", src_size=[img.size[0], img.size[1]], decoded_png=f"img_{idx:02d}.png"))
            print(f"{item['file']:60s} {img.size} decoded RGB only")
            continue
        out = tv(images=img, return_tensors="np")
        pv = out["pixel_values"][0]
        pos = out["image_position_ids"][0]
        n_soft = int(out["num_soft_tokens_per_image"][0])
        k = np.rint(pv * 255.0)
        assert np.array_equal(pv, (k.astype(np.float32) * np.float32(icfg["rescale_factor"]))), "pixel_values are not uint8*rescale_factor"
        u8 = k.astype(np.uint8)
        pv_pil = pl(images=img, return_tensors="np")["pixel_values"][0]
        diff_pil = np.abs(pv_pil.astype(np.float64) - pv.astype(np.float64)) * 255.0
        assert out["pixel_values"].shape[1] == icfg["max_soft_tokens"] * icfg["pooling_kernel_size"] ** 2
        valid = int((pos[:, 0] >= 0).sum())
        nw, nh = int(pos[:valid, 0].max()) + 1, int(pos[:valid, 1].max()) + 1
        assert nw * nh == valid
        # Rows are row-major over the patch grid, padding (zeros, position -1) follows.
        assert np.array_equal(pos[:valid], np.stack(np.meshgrid(np.arange(nw), np.arange(nh)), -1).reshape(-1, 2)), "unexpected position layout"
        assert (pos[valid:] == -1).all() and not u8[valid:].any(), "unexpected padding"
        # The resized picture: undo the patch layout, and check the processor's own
        # patchify function reproduces pixel_values from it exactly. The picture is
        # stored as PNG (lossless, far smaller than the raw tensor); the Go test
        # re-patchifies it and compares.
        resized = u8[:valid].reshape(nh, nw, ps, ps, 3).transpose(0, 2, 1, 3, 4).reshape(nh * ps, nw * ps, 3)
        again = convert_image_to_patches(torch.from_numpy(resized.copy()).permute(2, 0, 1), ps).numpy()
        assert np.array_equal(again, u8[:valid]), "patchify round trip failed"
        idx = len(entries)
        Image.fromarray(resized).save(os.path.join(REF, f"img_{idx:02d}.png"), optimize=True)
        e = dict(
            file=item["file"],
            level="full",
            src_size=[img.size[0], img.size[1]],  # width, height after EXIF transpose
            num_patches=valid,
            soft_tokens=n_soft,
            target_size=[nw * ps, nh * ps],  # width, height of the resized picture
            resized_png=f"img_{idx:02d}.png",
            torchvision_vs_pil_max_level=float(diff_pil.max()),
            torchvision_vs_pil_mean_level=float(diff_pil[:valid].mean()),
        )
        if item["file"] in RAW_FULL:  # raw padded tensors for a few small pictures: tests the patch layout itself
            e["pixel_file"], e["pos_file"] = f"img_{idx:02d}.pix.gz", f"img_{idx:02d}.pos.gz"
            save_gz(e["pixel_file"], u8)
            save_gz(e["pos_file"], pos.astype(np.int16))
        entries.append(e)
        print(f"{item['file']:60s} {img.size} patches={valid} soft={n_soft} tv-vs-pil max={e['torchvision_vs_pil_max_level']:.0f} mean={e['torchvision_vs_pil_mean_level']:.4f}")
    # get_aspect_ratio_preserving_size for many shapes, edge cases included
    sizes = []
    shapes = [(1, 1), (1, 7), (1, 100), (1, 10000), (10000, 1), (3, 5), (47, 47), (48, 48), (49, 49), (100, 10000), (10000, 100),
              (480, 640), (1080, 1920), (1920, 1080), (4000, 3000), (3000, 4000), (720, 1280), (17, 3000), (3000, 17), (255, 4096),
              (2, 13440), (13440, 2), (95, 95), (96, 5000), (5000, 96), (640, 480), (500, 500), (1, 13440), (13441, 1), (300, 20)]
    for h, w in shapes:
        try:
            th, tw = get_aspect_ratio_preserving_size(h, w, icfg["patch_size"], icfg["max_soft_tokens"] * icfg["pooling_kernel_size"] ** 2, icfg["pooling_kernel_size"])
            sizes.append(dict(h=h, w=w, th=th, tw=tw))
        except ValueError as ex:
            sizes.append(dict(h=h, w=w, error=str(ex)[:60]))
    versions = dict(transformers=transformers.__version__, torch=torch.__version__, torchvision=torchvision.__version__, pillow=PIL.__version__, numpy=np.__version__)
    json.dump(dict(versions=versions, images=entries, sizes=sizes), open(os.path.join(REF, "images.json"), "w"), indent=1)


def gen_audio():
    import librosa
    import soundfile
    import soxr
    from transformers.audio_utils import load_audio
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

    fcfg = {k: v for k, v in CFG["feature_extractor"].items() if k not in ("feature_extractor_type",)}
    fe = Gemma4AudioFeatureExtractor(**fcfg)
    manifest = json.load(open(os.path.join(HERE, "audio", "manifest.json")))
    entries = []
    for item in manifest:
        path = os.path.join(HERE, "audio", item["file"])
        info = soundfile.info(path)
        # backend="librosa": soundfile/libsndfile decode, mean over channels, soxr_hq resample to 16 kHz.
        wav = load_audio(path, sampling_rate=fcfg["sampling_rate"], backend="librosa")
        assert wav.dtype == np.float32 and wav.ndim == 1
        out = fe([wav], max_length=MAX_AUDIO_SAMPLES)  # a list: the batched path; defaults otherwise: padding=longest, truncation, pad_to_multiple_of=128
        mel = np.asarray(out["input_features"][0], dtype=np.float32)
        mask = np.asarray(out["input_features_mask"][0], dtype=bool)
        t = len(mask)
        m = mask
        for _ in range(2):
            t_out = (t + 2 - 3) // 2 + 1
            m = m[::2][:t_out]
            t = len(m)
        e = dict(
            file=item["file"],
            src_rate=info.samplerate, src_channels=info.channels, src_frames=info.frames, src_subtype=info.subtype, src_format=info.format,
            samples_16k=int(len(wav)),
            frames=int(mel.shape[0]),
            valid_frames=int(mask.sum()),
            soft_tokens=int(m.sum()),
            mel_file=f"aud_{len(entries):02d}.mel.gz",
            mask_file=f"aud_{len(entries):02d}.mask.gz",
        )
        save_gz(e["mel_file"], mel)
        save_gz(e["mask_file"], mask.astype(np.uint8))
        entries.append(e)
        print(f"{item['file']:70s} {info.samplerate:6d}Hz ch{info.channels} {info.subtype:10s} -> {mel.shape} valid={e['valid_frames']} tokens={e['soft_tokens']}")
    versions = dict(transformers=transformers.__version__, librosa=librosa.__version__, soundfile=soundfile.__version__,
                    libsndfile=soundfile.__libsndfile_version__, soxr=soxr.__version__, numpy=np.__version__)
    json.dump(dict(versions=versions, max_samples=MAX_AUDIO_SAMPLES, clips=entries), open(os.path.join(REF, "audio.json"), "w"), indent=1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.makedirs(REF, exist_ok=True)
    what = sys.argv[1:] or ["images", "audio"]
    if "images" in what:
        gen_images()
    if "audio" in what:
        gen_audio()
