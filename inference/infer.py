"""Team Arcturus, IndoML 2026 Track 2 (Noise Event Removal): Phase 2 inference.

    python inference/infer.py <input_dir> <output_dir> [--events events.json] [--config configs/model_config.yaml] [--no-asr]

One general pipeline, applied identically to every clip (natural or synthetic):
  1. every separator in the config enhances the clip in fixed windows (50% Hann overlap-add), RMS-matched to the input;
  2. learned time-frequency fusion networks combine the separator outputs and the noisy input with per-bin softmax weights;
     their RMS-matched outputs are averaged with fixed weights and RMS-matched again;
  3. event-conditioned strength: the fused output inside detected noise events (+-margin, 10 ms ramps) and
     outside_w * fused + (1 - outside_w) * input elsewhere, after level alignment to the input; clips with no detected event
     receive the full fused output, so model output is present on every sample;
  4. 0.99 peak guard, 16-bit PCM WAV; transcription with the mandated ASR (ARTPARK-IISc/SraVaani-1.0).
Event timestamps come from our Track 1 detector run on the test audio (bundled: events/test_events_phase2_v6.json).
There is no noise dictionary, no correlation search and no per-clip decision about whether to enhance.
"""
import argparse, glob, json, os, sys, time

import yaml

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2release.separators import make, chunked, rms_match, SR
from t2release.fusion import Fusion

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repository root: configs/, checkpoints/, events/


def load16(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(1)
    if sr != SR: x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x.astype(np.float32)


def pcm16(y):
    """Round-trip through 16-bit PCM, as the per-model intermediate files of the submitted run were stored."""
    return (np.clip(np.round(y * 32767), -32768, 32767) / 32767).astype(np.float32)


def soft_gate(x, y, events, margin, outside_w):
    n = len(x)
    if not events:
        return y
    g = np.zeros(n, np.float32)
    for e in events:
        a = max(0, int((e["onset"] - margin) * SR)); b = min(n, int((e["offset"] + margin) * SR)); g[a:b] = 1
    k = int(0.01 * SR); g = np.clip(np.convolve(g, np.ones(k) / k, mode="same"), 0, 1)
    o = g < 0.5
    if o.sum() > 0.2 * SR:                                   # level-align the blend to the input outside events
        y = y * float(y[o] @ x[o] / (y[o] @ y[o] + 1e-12))
    g = outside_w + (1 - outside_w) * g
    return g * y + (1 - g) * x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir"); ap.add_argument("output_dir")
    ap.add_argument("--events", default=None, help="{clip_id: [{onset, offset}]} (default: <input_dir>/events.json, else bundled)")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "model_config.yaml"))
    ap.add_argument("--no-asr", action="store_true", help="skip transcription (WAVs only)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev_path = a.events or (os.path.join(a.input_dir, "events.json") if os.path.exists(os.path.join(a.input_dir, "events.json"))
                           else os.path.join(ROOT, cfg["bundled_events"]))
    events = json.load(open(ev_path)) if os.path.exists(ev_path) else {}
    print(f"device {dev}; events from {ev_path} ({len(events)} clips)", flush=True)
    models = [(make(s["type"], os.path.join(ROOT, s["ckpt"]), dev), s["window_s"], s.get("weight", 0.0), s["name"]) for s in cfg["separators"]]
    names = [nm for _, _, _, nm in models]
    fusions = []                     # learned TF fusion(s) over member outputs (+ noisy input); fixed networks, same for every clip
    for fc in (cfg.get("fusions") or ([dict(cfg["fusion"], members=names, weight=1.0)] if cfg.get("fusion") else [])):
        net = Fusion(len(fc["members"]) + 1, fc["ch"], fc["dil"], fc["w0"]).to(dev).eval()
        net.load_state_dict(torch.load(os.path.join(ROOT, fc["ckpt"]), map_location="cpu")["model"])
        fusions.append((net, [names.index(m) for m in fc["members"]], float(fc["weight"])))
    if fusions:
        assert abs(sum(w for _, _, w in fusions) - 1.0) < 1e-6, "fusion weights must sum to 1"
    else:
        assert abs(sum(w for _, _, w, _ in models) - 1.0) < 1e-6, "blend weights must sum to 1"
    files = sorted(p for p in glob.glob(os.path.join(a.input_dir, "*")) if p.lower().endswith((".wav", ".flac")))
    os.makedirs(a.output_dir, exist_ok=True); t0 = time.time()
    for i, p in enumerate(files):
        cid = os.path.splitext(os.path.basename(p))[0]
        x = load16(p)
        outs = [pcm16(rms_match(x, chunked(m, x, win))) for m, win, _, _ in models]
        if fusions:                  # each fusion output RMS-matched + 16-bit (as stored in the submitted run), then weighted average
            y = np.zeros(len(x), np.float32)
            for net, idx, w in fusions:
                with torch.no_grad():
                    f = net(torch.from_numpy(np.stack([outs[i] for i in idx] + [x])).unsqueeze(0).to(dev))[0].cpu().numpy()
                y += w * pcm16(rms_match(x, f))
            y = rms_match(x, y)
        else:
            y = rms_match(x, sum(w * o for o, (_, _, w, _) in zip(outs, models)))
        y = pcm16(y)
        y = soft_gate(x, y, events.get(cid, []), cfg["gate_margin_s"], cfg["outside_w"])
        pk = np.abs(y).max()
        if pk > 0.99: y = y / pk * 0.99
        sf.write(os.path.join(a.output_dir, f"{cid}.wav"), y.astype(np.float32), SR, subtype="PCM_16")
        if (i + 1) % 100 == 0: print(f"{i + 1}/{len(files)} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    if a.no_asr:
        return
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    asr = AutoModel.from_pretrained(snapshot_download("ARTPARK-IISc/SraVaani-1.0"), trust_remote_code=True).to(dev).eval()
    ids = [os.path.splitext(os.path.basename(p))[0] for p in files]; out = {}
    with torch.no_grad():
        for j in range(0, len(ids), 8):
            chunk = ids[j:j + 8]
            hyps = asr.transcribe([os.path.join(a.output_dir, f"{c}.wav") for c in chunk], return_hypotheses=True)
            for c, h in zip(chunk, hyps): out[c] = (h.text if hasattr(h, "text") else str(h)).strip()
    with open(os.path.join(a.output_dir, "transcripts.jsonl"), "w", encoding="utf-8") as f:
        for c in ids: f.write(json.dumps({"clip_id": c, "text": out.get(c, "")}, ensure_ascii=False) + "\n")
    print(f"done: {len(ids)} clips, {sum(not v for v in out.values())} empty transcripts", flush=True)


if __name__ == "__main__":
    main()
