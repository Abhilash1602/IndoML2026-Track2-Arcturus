"""Harder-SNR monitoring set: data/eval_unseen (104 held-out clean references + unseen FSD50K-eval noise, built by
build_ext_noise.py) with every noise event 4 dB louder -> data/eval_unseen_hard/pairs.json. Used only to select checkpoints.
Usage (repo root): python train/build_unseen_hard.py
"""
import json, os
import numpy as np, soundfile as sf

src, od = "data/eval_unseen/pairs.json", "data/eval_unseen_hard"; os.makedirs(od, exist_ok=True); out = []
for p in json.load(open(src)):
    x, sr = sf.read(p["noisy"], dtype="float32"); c, _ = sf.read(p["clean"], dtype="float32"); n = min(len(x), len(c))
    y = c[:n] + 10 ** (4 / 20) * (x[:n] - c[:n]); cc = c[:n].copy(); pk = np.abs(y).max()
    if pk > 0.99: y, cc = y / pk * 0.99, cc / pk * 0.99
    q = dict(p, noisy=f"{od}/{p['id']}_noisy.wav", clean=f"{od}/{p['id']}_clean.wav")
    sf.write(q["noisy"], y, sr, subtype="FLOAT"); sf.write(q["clean"], cc, sr, subtype="FLOAT"); out.append(q)
json.dump(out, open(f"{od}/pairs.json", "w"), ensure_ascii=False, indent=0); print("hard pairs", len(out))
