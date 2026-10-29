"""Clean-speech bank: the regions between annotated noise events (padded by 0.05 s, at least 2.5 s long) of the first 6,000
Gold (verified-timestamp) clips of the Vaani Noise-Event dataset -> data/bank/clean/c<row>_<k>.wav.
The 3,968 files listed under "clean_bank" in train/ft_data_lists.json come from this step.
Usage (repository root): python train/build_clean_bank.py      (needs `datasets`; set VAANI_HF_CACHE to choose the cache directory)
"""
import io
import os

import librosa
import numpy as np
import soundfile as sf

SR, MAX_GOLD, MIN_CLEAN_S, PAD_S = 16000, 6000, 2.5, 0.05
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"
OUT = "data/bank/clean"


def decode(a):
    """One Hugging Face audio cell (decode=False) -> mono float32 at 16 kHz."""
    if a.get("bytes"):
        w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
    else:
        w, sr = sf.read(a["path"], dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(w, dtype=np.float32)


def event_spans(ex):
    out = []
    for s in ex.get("NoiseSubCategoryTimeStamp") or []:
        try:
            st, en = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if en > st:
            out.append((st, en))
    return sorted(out)


def gaps(spans, dur, pad=PAD_S):
    free, cur = [], 0.0
    for st, en in spans:
        st, en = max(0.0, st - pad), min(dur, en + pad)
        if st > cur:
            free.append((cur, st))
        cur = max(cur, en)
    if cur < dur:
        free.append((cur, dur))
    return free


def main():
    from datasets import Audio, load_dataset
    raw = load_dataset(REPO, split="train", cache_dir=os.environ.get("VAANI_HF_CACHE", "data/hf_cache"))
    raw = raw.cast_column("audio", Audio(decode=False))
    os.makedirs(OUT, exist_ok=True)
    n_gold = n_out = 0
    for i, ex in enumerate(raw):
        if ex["annotationQuality"] != "verified_timestamps":
            continue
        if n_gold >= MAX_GOLD:
            break
        try:
            wav = decode(ex["audio"])
        except Exception:
            continue
        if len(wav) < SR * 0.5:
            continue
        n_gold += 1
        for k, (a, b) in enumerate(gaps(event_spans(ex), len(wav) / SR)):
            seg = wav[int(a * SR):int(b * SR)]
            if b - a < MIN_CLEAN_S or np.abs(seg).max() < 1e-3:
                continue
            sf.write(f"{OUT}/c{i:07d}_{k}.wav", seg, SR); n_out += 1
    print(f"{n_gold} Gold clips -> {n_out} clean spans in {OUT}")


if __name__ == "__main__":
    main()
