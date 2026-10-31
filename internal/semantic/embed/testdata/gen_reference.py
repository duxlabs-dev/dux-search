"""Writes the reference data the embed tests compare against.

  uv venv .venv --python 3.12
  uv pip install --python .venv/bin/python transformers tokenizers onnxruntime numpy pillow \
      --index-strategy unsafe-best-match
  uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch torchvision
  .venv/bin/python gen_reference.py <model dir> <image> <wav> [GOROOT]

(torch and torchvision are only needed because transformers' EmbeddingGemma2Processor
imports its video processor; no tensor math runs through torch.)

Outputs, next to this script (OUT_DIR overrides; MAX_SOFT_TOKENS sets the image budget, default 70):
  text_reference.json   vectors of actions.json (20 actions, 15 queries) from the
                        model's own tokenizer.json + onnxruntime (Python), one text per run
  image_reference.dref  input tensors the transformers processor builds for <image>
                        (budget 70 soft tokens, padding patches dropped) and the
                        embedding Python's onnxruntime computes from them
  audio_reference.dref  same for <wav> (resampled to mono 16 kHz with ffmpeg)

The image and audio files are gzip containers: u32le header length, JSON header
(name, dtype, count of every array in order), then the arrays, little endian.
Inputs come from the transformers processors, so the Go side is checked against
the reference implementation's tensors; the embedding is the reference output of
the same ONNX graphs run by Python. Sources: image = Go's image/testdata/video-001.png
(BSD-3-Clause); audio = ../preprocess/testdata/audio/Walk_Sign_22k.flac (public domain,
see that folder's manifest.json).
"""
import gzip, json, os, struct, subprocess, sys
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

model_dir, image_path, wav_path = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.environ.get("OUT_DIR", here)          # where the references are written
max_soft = int(os.environ.get("MAX_SOFT_TOKENS", "70"))  # image budget; the committed files use 70
DIM = 768
E0 = np.zeros((0, 512), np.float32)

def session(name):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    return ort.InferenceSession(f"{model_dir}/onnx/{name}_quantized.onnx", so, providers=["CPUExecutionProvider"])

def l2(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def embed_ids(text_sess, ids, image=E0, audio=E0):
    ids = np.asarray([ids], np.int64)
    out = text_sess.run(["sentence_embedding"], {
        "input_ids": ids, "attention_mask": np.ones_like(ids),
        "image_features": image, "video_features": E0, "audio_features": audio})[0]
    return l2(out)[0]

def write_dref(path, header, arrays):
    header = dict(header)
    header["arrays"] = [{"name": n, "dtype": str(a.dtype), "count": int(a.size)} for n, a in arrays]
    h = json.dumps(header).encode()
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(struct.pack("<I", len(h))); f.write(h)
        for _, a in arrays:
            f.write(np.ascontiguousarray(a).astype("<" + a.dtype.str[1:]).tobytes())

# ---- text -----------------------------------------------------------------
text_sess = session("model")
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
cfg = json.load(open(f"{here}/actions.json"))
texts = [("document", "title: " + a["key"] + " | text: " + a["text"]) for a in cfg["actions"]]
texts += [("query", "task: search result | query: " + q["text"]) for q in cfg["queries"]]
items = []
for kind, t in texts:
    v = embed_ids(text_sess, tok.encode(t).ids)
    items.append({"kind": kind, "text": t, "vec": [round(float(x), 7) for x in v]})
json.dump({"onnxruntime": ort.__version__, "dim": DIM, "items": items},
          open(f"{out_dir}/text_reference.json", "w"), ensure_ascii=False)
print("text:", len(items), "vectors", file=sys.stderr)

# ---- processors (transformers) ----------------------------------------------
from transformers import (AutoTokenizer, EmbeddingGemma2Processor, Gemma4AudioFeatureExtractor,
                          Gemma4ImageProcessorPil)
hf_tok = AutoTokenizer.from_pretrained(model_dir)
ip = Gemma4ImageProcessorPil.from_pretrained(model_dir)
ip.max_soft_tokens = max_soft   # 70 keeps the committed tensors small; the code path is the same
fe = Gemma4AudioFeatureExtractor.from_pretrained(model_dir)
from transformers import AutoVideoProcessor
vp = AutoVideoProcessor.from_pretrained(model_dir)
proc = EmbeddingGemma2Processor(feature_extractor=fe, image_processor=ip, tokenizer=hf_tok, video_processor=vp,
                                chat_template=open(f"{model_dir}/chat_template.jinja").read())

# ---- image ------------------------------------------------------------------
img = Image.open(image_path).convert("RGB")
out = proc(text=None, images=[[img]], return_tensors="np")
ids = out["input_ids"][0].astype(np.int64)
pv = out["pixel_values"][0].astype(np.float32)             # (max_patches, 768), zero padded
pos = out["image_position_ids"][0].astype(np.int64)        # (max_patches, 2), -1 on padding
valid = pos[:, 0] >= 0
pv_t, pos_t = pv[valid], pos[valid]
vis = session("vision_encoder")
def run_vision(p, q):
    return vis.run(["image_features"], {"pixel_values": p[None], "pixel_position_ids": q[None]})[0]
feat_trim = run_vision(pv_t, pos_t)
feat_pad = run_vision(pv, pos)
print("image: patches", int(valid.sum()), "of", len(valid), "soft tokens", feat_trim.shape[0],
      "padded run rows", feat_pad.shape[0], file=sys.stderr)
soft = int(out["num_soft_tokens_per_image"][0]) if "num_soft_tokens_per_image" in out else feat_trim.shape[0]
assert feat_trim.shape[0] == soft == int((ids == 258880).sum()), (feat_trim.shape, soft)
emb_trim = embed_ids(text_sess, ids, image=feat_trim)
emb_pad = embed_ids(text_sess, ids, image=feat_pad[:soft]) if feat_pad.shape[0] >= soft else emb_trim
print("image: trimmed-vs-padded cosine", float(emb_trim @ emb_pad), file=sys.stderr)
write_dref(f"{out_dir}/image_reference.dref",
           {"patches": int(valid.sum()), "soft_tokens": soft, "embedding_dim": DIM,
            "source": os.path.basename(image_path), "max_soft_tokens": max_soft},
           [("pixels", pv_t), ("positions", pos_t), ("ids", ids), ("embedding", emb_trim.astype(np.float32))])

# ---- audio ------------------------------------------------------------------
pcm = np.frombuffer(subprocess.run(
    ["ffmpeg", "-v", "error", "-i", wav_path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
    capture_output=True, check=True).stdout, dtype=np.float32)
out = proc(text=None, audio=[pcm], return_tensors="np")
ids = out["input_ids"][0].astype(np.int64)
mel = out["input_features"][0].astype(np.float32)           # (frames, 128)
mask = out["input_features_mask"][0].astype(bool)           # (frames,)
aud = session("audio_encoder")
feat = aud.run(["audio_features"], {"input_features": mel[None], "input_features_mask": mask[None]})[0]
assert feat.shape[0] == int((ids == 258881).sum()), (feat.shape, int((ids == 258881).sum()))
emb = embed_ids(text_sess, ids, audio=feat)
print("audio: frames", mel.shape[0], "valid", int(mask.sum()), "tokens", feat.shape[0], file=sys.stderr)
write_dref(f"{out_dir}/audio_reference.dref",
           {"frames": int(mel.shape[0]), "tokens": int(feat.shape[0]), "embedding_dim": DIM,
            "seconds": round(len(pcm) / 16000, 3), "source": os.path.basename(wav_path)},
           [("mel", mel), ("mask", mask.astype(np.uint8)), ("ids", ids), ("embedding", emb.astype(np.float32))])
