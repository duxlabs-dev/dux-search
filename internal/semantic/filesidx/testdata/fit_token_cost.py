#!/usr/bin/env python3
"""Checks (and if needed re-fits) the per-character token weights in chunk.go
(tokenCost) against the real tokenizer.
Usage: fit_token_cost.py MODEL_DIR GOROOT      (needs numpy, tokenizers)

First table: real/estimated token ratio per kind of text with the weights that
chunk.go ships (CURRENT below). Ratios near 1.0 mean the chunk budget holds.
Second part: a non-negative least squares fit on the same samples, in
thousandths of a token per character class. The classes are correlated
(newline vs punctuation vs indentation), so a re-fit moves between equivalent
solutions: adjust the cost* constants only when the first table drifts, then
run the tests (TestChunkTokenBudget checks the real chunk sizes).
"""
import glob, os, random, sys, unicodedata
import numpy as np
from tokenizers import Tokenizer

model_dir, goroot = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
names = ["alnum", "punct", "newline", "latinext", "cjk", "space", "tab", "otherletter", "other"]

def cls(ch):
    o = ord(ch)
    if o < 128:
        if ch.isalnum(): return 0
        if ch == " ": return 5
        if ch == "\t": return 6
        if ch == "\n": return 2
        return 1
    n = unicodedata.name(ch, "")
    if any(k in n for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")): return 4
    if ch.isalpha() and o < 0x250: return 3
    if ch.isalpha() or ch.isdigit(): return 7
    return 8

def read(path, n=2000):
    try:
        return open(path, encoding="utf-8").read()[:n]  # skips files in other encodings
    except UnicodeDecodeError:
        return ""

random.seed(1)
samples = []
gofiles = sorted(glob.glob(f"{goroot}/src/**/*.go", recursive=True)); random.shuffle(gofiles)
samples += [("go", read(f)) for f in gofiles[:120]]
samples += [("md", read(f)) for f in sorted(glob.glob(f"{goroot}/**/*.md", recursive=True))[:40]]
samples += [("py", read(f)) for f in sorted(glob.glob("/usr/lib/python3*/*.py"))[:40]]
for f in sorted(glob.glob(f"{here}/files/*")) + sorted(glob.glob(f"{here}/variants/*/*")):
    samples.append((os.path.basename(f), read(f, 6000)))
samples += [("ru", "Это тестовое предложение на русском языке для проверки токенизации. " * k) for k in (5, 20)]
samples += [("cjk", "这是一个用于测试的句子，它很长，包含很多汉字和标点符号。" * k) for k in (5, 20)]
samples += [("emoji", "😀🚀🎉 " * k) for k in (50, 200)]
samples += [("de", "Die Bundesrepublik Deutschland ist ein demokratischer und sozialer Bundesstaat. Alle Staatsgewalt geht vom Volke aus. " * k) for k in (3, 8, 15)]

X, y, g = [], [], []
for kind, text in samples:
    if not text.strip(): continue
    c = np.zeros(len(names))
    for ch in text: c[cls(ch)] += 1
    X.append(c); y.append(len(tok.encode(text, add_special_tokens=False).ids)); g.append(kind)
X, y = np.array(X), np.array(y, float)
CURRENT = np.array([190, 1200, 1300, 1800, 640, 0, 500, 220, 1000]) / 1000  # chunk.go, same order as names
est = X @ CURRENT
print("shipped weights: real/estimated tokens per kind of text")
for kind in sorted(set(g)):
    idx = [i for i, k in enumerate(g) if k == kind]
    r = y[idx] / est[idx]
    print(f"{kind:28s} n={len(idx):3d} mean={r.mean():.2f} min={r.min():.2f} max={r.max():.2f}")
print("\nre-fit on the same samples:")
w = np.full(X.shape[1], 0.3)
lr = 1.0 / np.linalg.norm(X, 2) ** 2
for _ in range(200000):
    w = np.maximum(w - lr * (X.T @ (X @ w - y)), 0)
print({n: int(round(v * 1000)) for n, v in zip(names, w)})
pred = X @ w
for kind in sorted(set(g)):
    idx = [i for i, k in enumerate(g) if k == kind]
    r = y[idx] / pred[idx]
    print(f"{kind:28s} n={len(idx):3d} real/pred mean={r.mean():.2f} min={r.min():.2f} max={r.max():.2f}")
