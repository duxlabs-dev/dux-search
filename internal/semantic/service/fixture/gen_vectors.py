"""Builds the real-model test fixture used by the vecstore, service and server tests.

Nothing here is synthetic: documents are the package documentation of the Go
standard library (BSD licensed, read from GOROOT) plus the action
descriptions of degerlendirme/eylem_eslesme.py; queries are hand written English and
Turkish phrases; every vector comes from the embeddinggemma-2 int8 ONNX text tower
with the prefixes from config_sentence_transformers.json.

Run (needs onnxruntime, tokenizers, numpy):
    python gen_vectors.py <model_dir> <goroot_src> <out_dir>
Writes <out_dir>/fixture.json and <out_dir>/vectors.f32 (float32 LE, 768 per row:
queries first, then documents, in manifest order).
"""
import json
import os
import re
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: {title} | text: {text}"

ACTIONS = {
    "dark": "Dark theme. Switch the desktop color scheme between dark mode and light mode; night theme, black interface.",
    "nightlight": "Night Light. Warm, yellowish screen colors that reduce blue light and eye strain; tired eyes, reading at night.",
    "dnd": "Do Not Disturb. Silence and hide notifications and alerts.",
    "vol": "Volume. Turn the speaker sound up or down, louder or quieter, mute audio output.",
    "mic": "Microphone mute. Mute or unmute the microphone input for calls.",
    "bright": "Display brightness. Make the screen brighter or dimmer.",
    "kbd": "Keyboard backlight. Turn keyboard illumination on or off.",
    "wifi": "Wi-Fi. Turn the wireless network and internet connection on or off, join a network.",
    "bt": "Bluetooth. Turn Bluetooth on or off, connect wireless headphones, earbuds, speakers and devices.",
    "plane": "Airplane mode. Disable all wireless radios at once.",
    "vpn": "VPN. Connect or disconnect the virtual private network.",
    "perf": "Power profile. Choose performance, balanced or power saver to extend battery life.",
    "lock": "Lock screen. Lock the session so a password is needed.",
    "sleep": "Suspend. Put the computer to sleep.",
    "shot": "Screenshot. Capture an image of a region or the full screen.",
    "rec": "Screen recording. Start or stop recording a video of the screen.",
    "close": "Close window. Close the currently focused application window.",
    "float": "Floating window. Make the focused window float above the tiled layout.",
    "timer": "Timer. Start a countdown timer for a number of minutes.",
    "lamp": "Room light. Turn the smart home room lamp or flashlight on or off.",
}

# (query, expected action id) for the action collection.
ACTION_QUERIES = [
    ("karanlık", "dark"), ("gözlerim yoruldu", "nightlight"), ("kablosuz kulaklık", "bt"),
    ("fener", "lamp"), ("sesi kıs", "vol"), ("parlaklığı %40 yap", "bright"),
    ("bildirimleri sustur", "dnd"), ("pil tasarrufu", "perf"), ("ekranı kaydet", "rec"),
    ("15 dakika zamanlayıcı kur", "timer"), ("oda ışığını aç", "lamp"), ("internet kapat", "wifi"),
    ("bilgisayarı uyut", "sleep"), ("pencereyi kapat", "close"), ("mikrofonu kapat", "mic"),
]

# (query, expected Go package) for the file text collection.
FILE_QUERIES = [
    ("regular expression matching", "regexp"),
    ("düzenli ifadelerle metin eşleştirme", "regexp"),
    ("write an HTTP server and client", "net/http"),
    ("HTTP sunucusu yazmak", "net/http"),
    ("encode and decode JSON", "encoding/json"),
    ("JSON verisini yapıya çevirmek", "encoding/json"),
    ("text templates with actions and pipelines", "text/template"),
    ("şablon dosyalarıyla metin üretmek", "text/template"),
    ("compress data with gzip", "compress/gzip"),
    ("zip arşivi okumak ve yazmak", "archive/zip"),
    ("parse and manipulate URLs", "net/url"),
    ("big integer arithmetic", "math/big"),
    ("çok büyük tam sayılarla hesaplama", "math/big"),
    ("structured logging with levels and attributes", "log/slog"),
    ("yapılandırılmış günlük kaydı", "log/slog"),
    ("formatted I/O with printing verbs", "fmt"),
    ("command line flag parsing", "flag"),
    ("komut satırı bayraklarını okumak", "flag"),
    ("run external commands", "os/exec"),
    ("harici komut çalıştırmak", "os/exec"),
    ("cancellation and deadlines across API boundaries", "context"),
    ("zaman aşımı ve iptal sinyali", "context"),
    ("reading and writing CSV records", "encoding/csv"),
    ("XML encoding", "encoding/xml"),
    ("SQL database access", "database/sql"),
    ("TLS secure connections", "crypto/tls"),
    ("sorting slices and collections", "sort"),
    ("mutual exclusion locks and wait groups", "sync"),
    ("kilitler ve eşzamanlılık ilkelleri", "sync"),
    ("date and time formatting", "time"),
    ("tarih ve saat biçimlendirme", "time"),
    ("write unit tests and benchmarks", "testing"),
    ("birim testi ve karşılaştırma ölçümü yazmak", "testing"),
    ("manipulating file paths", "path/filepath"),
    ("dosya yolu işlemleri", "path/filepath"),
    ("decode images", "image"),
    ("unicode text handling", "unicode"),
]

MAX_CHUNKS_PER_PKG = 3
CHUNK_CHARS = 1100


def package_doc(path):
    """The package comment of a directory: the comment block right before `package x`."""
    best = ""
    for name in sorted(os.listdir(path)):
        if not name.endswith(".go") or name.endswith("_test.go"):
            continue
        src = open(os.path.join(path, name), encoding="utf-8", errors="replace").read()
        m = re.search(r"((?:^//[^\n]*\n)+|/\*.*?\*/\n)package \w+", src, re.S | re.M)
        if not m:
            continue
        block = m.group(1)
        if block.startswith("/*"):
            text = block[2:-3]
        else:
            text = "\n".join(re.sub(r"^// ?", "", ln) for ln in block.splitlines())
        text = text.strip()
        if "Copyright" in text[:80] or text.startswith("Code generated"):
            continue
        if len(text) > len(best):
            best = text
    return best


def chunks(text):
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) > CHUNK_CHARS:
            out.append(cur)
            cur = ""
        cur = (cur + "\n\n" + para).strip()
    if cur:
        out.append(cur)
    return out[:MAX_CHUNKS_PER_PKG]


def collect_docs(goroot_src):
    docs = []
    for root, dirs, files in os.walk(goroot_src):
        dirs[:] = sorted(d for d in dirs if d not in ("internal", "vendor", "testdata", "cmd", "builtin") and not d.startswith("_"))
        rel = os.path.relpath(root, goroot_src)
        if rel == "." or not any(f.endswith(".go") for f in files):
            continue
        text = package_doc(root)
        if len(text) < 60:
            continue
        for i, ch in enumerate(chunks(text)):
            docs.append({"kind": "file", "pkg": rel, "chunk": i, "title": rel, "text": ch})
    return docs


class Embedder:
    def __init__(self, model_dir):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.s = ort.InferenceSession(model_dir + "/onnx/model_quantized.onnx", so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(model_dir + "/tokenizer.json")
        self.empty = np.zeros((0, 512), np.float32)

    def embed(self, texts, batch=8):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            length = max(len(e.ids) for e in enc)
            ids = np.zeros((len(enc), length), np.int64)
            am = np.zeros_like(ids)
            for j, e in enumerate(enc):
                ids[j, :len(e.ids)] = e.ids
                am[j, :len(e.ids)] = 1
            v = self.s.run(["sentence_embedding"], {
                "input_ids": ids, "attention_mask": am,
                "image_features": self.empty, "video_features": self.empty, "audio_features": self.empty,
            })[0]
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


def main():
    model_dir, goroot_src, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    docs = [{"kind": "action", "id": k, "title": k, "text": v} for k, v in ACTIONS.items()]
    docs += collect_docs(goroot_src)
    pkgs = {d["pkg"] for d in docs if d["kind"] == "file"}
    queries = [{"kind": "action", "text": q, "expect": e} for q, e in ACTION_QUERIES]
    for q, p in FILE_QUERIES:
        if p not in pkgs:
            raise SystemExit(f"expected package {p!r} has no documentation chunk in {goroot_src}")
        queries.append({"kind": "file", "text": q, "expect": p})

    emb = Embedder(model_dir)
    qv = emb.embed([QUERY_PREFIX + q["text"] for q in queries])
    dv = emb.embed([DOC_PREFIX.format(title=d["title"] or "none", text=d["text"]) for d in docs])
    assert qv.shape[1] == 768 and dv.shape[1] == 768
    json.dump({
        "model": "embeddinggemma-2 int8 ONNX text tower (model_quantized.onnx)",
        "dim": 768,
        "queries": queries,
        "docs": docs,
    }, open(os.path.join(out_dir, "fixture.json"), "w"), ensure_ascii=False, indent=1)
    np.concatenate([qv, dv]).tofile(os.path.join(out_dir, "vectors.f32"))
    print(f"{len(queries)} queries, {len(docs)} documents ({len(pkgs)} packages)")


if __name__ == "__main__":
    main()
