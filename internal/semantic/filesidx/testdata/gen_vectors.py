#!/usr/bin/env python3
"""Regenerates testdata/vectors.json: embeddings computed by the real
EmbeddingGemma 2 int8 ONNX model for the inputs the filesidx tests use.

Why a file of vectors: the Go tests need an Embedder, and the model runs in a
separate helper process (internal/semantic/embed). The tests therefore read
vectors that this script produced with the same ONNX files and the reference
Python tokenizer/processors (tokenizers, transformers' EmbeddingGemma2Processor),
so retrieval quality in the tests is the model's own, not an invention.

Inputs
  text    every (title, chunk) the indexer embeds for the files under
          testdata/files and testdata/variants, recorded by the Go test
          TestRecordFixtureInputs (prefix: "title: {title} | text: {chunk}")
  query   testdata/queries.json   (prefix: "task: search result | query: ")
  image   testdata/media/*.{jpg,png}   (processor -> vision tower -> text model)
  audio   testdata/media/*.{ogg,flac}  (16 kHz mono -> processor -> audio tower -> text model)
  tokens  real token count of each text chunk (without the prefix), same keys as text
Keys: text sha256(title NUL chunk); image/audio sha256(file bytes); query text.
Vectors are the full 768-dim, L2-normalized sentence embedding as float32 LE.

Regenerate (after changing the chunker, fixtures or queries):
  cd internal/semantic/filesidx
  DUX_FIXTURE_MISSING=/tmp/inputs.jsonl go test -run TestRecordFixtureInputs .
  python3 testdata/gen_vectors.py MODEL_DIR /tmp/inputs.jsonl     # MODEL_DIR = embeddinggemma-2-onnx
Needs: numpy onnxruntime tokenizers transformers pillow (no PyTorch), and ffmpeg for audio.
Versions used for the committed file: onnxruntime 1.31.1, tokenizers 0.23.3, transformers 5.22.1, numpy 2.5.4, pillow 13.0.0.
"""
import base64, hashlib, json, os, subprocess, sys
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__))
model_dir, inputs = sys.argv[1], sys.argv[2]

def session(name):
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    return ort.InferenceSession(f"{model_dir}/onnx/{name}.onnx", so, providers=["CPUExecutionProvider"])

text_model = session("model_quantized")
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
EMPTY = np.zeros((0, 512), np.float32)

def run_text_model(ids, image_features=EMPTY, audio_features=EMPTY):
    ids = np.asarray(ids, np.int64)[None, :]
    out = text_model.run(["sentence_embedding"], {
        "input_ids": ids, "attention_mask": np.ones_like(ids),
        "image_features": image_features, "video_features": EMPTY, "audio_features": audio_features})[0][0]
    return out / np.linalg.norm(out)

def embed_text(s):
    ids = tok.encode(s).ids[:2048]
    return run_text_model(ids)

def b64(v): return base64.b64encode(np.asarray(v, "<f4").tobytes()).decode()
def key(*parts): return hashlib.sha256("\x00".join(parts).encode()).hexdigest()

table = {"comment": "EmbeddingGemma 2 int8 ONNX (onnx-community/embeddinggemma-2-ONNX daa72c51), full 768 dims, L2-normalized; see gen_vectors.py",
         "text": {}, "tokens": {}, "query": {}, "image": {}, "audio": {}}

seen = set()
for line in open(inputs):
    rec = json.loads(line)
    if rec["kind"] != "text": continue
    t, c = rec["entry"]["title"], rec["entry"]["text"]
    k = key(t, c)
    if k in seen: continue
    seen.add(k)
    table["text"][k] = b64(embed_text(f"title: {t} | text: {c}"))
    table["tokens"][k] = len(tok.encode(c, add_special_tokens=False).ids)  # chunk alone, for TestChunkTokenBudget
print("text", len(table["text"]))

for q in json.load(open(f"{here}/queries.json")):
    table["query"][q["query"]] = b64(embed_text(f"task: search result | query: {q['query']}"))
print("query", len(table["query"]))

from transformers import Gemma4AudioFeatureExtractor, Gemma4ImageProcessorPil
from PIL import Image
pcfg = json.load(open(f"{model_dir}/processor_config.json"))
mcfg = json.load(open(f"{model_dir}/config.json"))
image_proc = Gemma4ImageProcessorPil(**{k: v for k, v in pcfg["image_processor"].items() if k != "image_processor_type"})
audio_proc = Gemma4AudioFeatureExtractor(**{k: v for k, v in pcfg["feature_extractor"].items() if k != "feature_extractor_type"})
BOS, EOS = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
vision, audio = session("vision_encoder_quantized"), session("audio_encoder_quantized")
media = f"{here}/media"
for name in sorted(os.listdir(media)):
    path = f"{media}/{name}"
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    ext = name.rsplit(".", 1)[-1].lower()
    if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
        out = image_proc(images=[Image.open(path).convert("RGB")], return_tensors="np")
        feats = vision.run(["image_features"], {"pixel_values": out["pixel_values"], "pixel_position_ids": out["image_position_ids"]})[0]
        n = int(out["num_soft_tokens_per_image"][0])
        assert feats.shape[0] == n, (name, feats.shape, n)
        # What EmbeddingGemma2Processor builds for text "<|image|>": bos boi image*n eoi eos
        ids = [BOS, mcfg["boi_token_id"]] + [mcfg["image_token_id"]] * n + [mcfg["eoi_token_id"], EOS]
        table["image"][sha] = b64(run_text_model(ids, image_features=feats))
    elif ext in ("ogg", "oga", "flac", "wav", "mp3"):
        pcm = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                                           capture_output=True, check=True).stdout, "<f4")
        out = audio_proc([pcm], sampling_rate=16000, return_tensors="np", return_attention_mask=True)
        mask = out["input_features_mask"].astype(bool)
        feats = audio.run(["audio_features"], {"input_features": out["input_features"].astype(np.float32), "input_features_mask": mask})[0]
        n = feats.shape[0]
        assert n == -(-int(mask.sum()) // 4), (name, n, int(mask.sum()))  # one token per 4 frames of 10 ms = 40 ms
        ids = [BOS, mcfg["boa_token_id"]] + [mcfg["audio_token_id"]] * n + [mcfg["eoa_token_index"], EOS]
        table["audio"][sha] = b64(run_text_model(ids, audio_features=feats))
    print("media", name, ext)

json.dump(table, open(f"{here}/vectors.json", "w"), separators=(",", ":"))
print("wrote", f"{here}/vectors.json", os.path.getsize(f"{here}/vectors.json") // 1024, "KiB")
