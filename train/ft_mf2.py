"""In-domain fine-tune of MossFormer2_SE_48K (ClearerVoice, Apache-2.0). Same training-data recipe
and held-out evaluation as ft_mgan.py (imports its Mixer/losses). Pipeline mirrors clearvoice's
decode_one_audio_mossformer2_se_48k: 16 kHz -> 48 kHz, x*32768, Kaldi fbank(60)+deltas (180-d) ->
TestNet mask -> mask * STFT(hamming 1920/384) -> iSTFT -> /32768 -> 16 kHz.
Loss: -SI-SDR/10 (16 kHz) + compressed-magnitude MSE (48 kHz).

Usage (repository root): PYTHONPATH=train python train/ft_mf2.py --out checkpoints/<run> [options]   (see train/run_training.sh)
"""
import argparse, json, os, random, time
from types import SimpleNamespace
import numpy as np, torch, torchaudio
from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
from clearvoice.utils.misc import stft, istft, compute_fbank
from ft_mgan import Mixer, load_wav, si_sdr_np, si_sdr_loss, long_groups, evaluate_long, SR, ab_sisdr_loss, event_mask
import ft_mgan
import os as _os_here
_HERE = _os_here.path.dirname(_os_here.path.abspath(__file__))   # train/: data lists and configs

ARGS = SimpleNamespace(mode="inference", win_type="hamming", win_len=1920, win_inc=384, fft_len=1920,
                       num_mels=60, sampling_rate=48000)
MAXV = 32768.0


def load_model(dev):
    m = MossFormer2_SE_48K(ARGS).model
    ck = torch.load("checkpoints/MossFormer2_SE_48K/last_best_checkpoint.pt", map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in ck.get("model", ck).items()}
    missing, _ = m.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    return m.to(dev)


def feats(x48):
    """x48: (T,) scaled audio tensor on CPU -> (1, frames, 180) features (no grad needed)."""
    fb = compute_fbank(x48.unsqueeze(0), ARGS)
    d1 = torchaudio.functional.compute_deltas(fb.T); d2 = torchaudio.functional.compute_deltas(d1)
    return torch.cat([fb, d1.T, d2.T], dim=1).unsqueeze(0)


def enhance_one(m, x16, dev):
    """x16: (T,) float tensor (16 kHz, on dev). Returns 16 kHz output (T,) and 48 kHz masked magnitude."""
    x48 = torchaudio.functional.resample(x16, 16000, 48000) * MAXV
    with torch.no_grad():
        f = feats(x48.detach().cpu()).to(dev)
    mask = m(f)[-1]                                          # (1, frames, F)
    spec = stft(x48, ARGS)                                   # (F, frames, 2)
    T = min(spec.shape[1], mask.shape[1])
    spec, mk = spec[:, :T], mask.permute(2, 1, 0)[:, :T]     # (F, T, 1)
    ms = spec * mk
    out48 = istft(torch.complex(ms[..., 0], ms[..., 1]), ARGS, x48.shape[-1])
    out16 = torchaudio.functional.resample(out48 / MAXV, 48000, 16000)[: x16.shape[-1]]
    if out16.shape[-1] < x16.shape[-1]:
        out16 = torch.nn.functional.pad(out16, (0, x16.shape[-1] - out16.shape[-1]))
    return out16, torch.sqrt(ms[..., 0] ** 2 + ms[..., 1] ** 2 + 1e-8), T


@torch.no_grad()
def evaluate(m, hold, dev):
    m.eval(); s = []
    for d in hold:
        x, c = load_wav(d["noisy"]), load_wav(d["clean"]); n = min(len(x), len(c))
        y, _, _ = enhance_one(m, torch.from_numpy(x[:n]).to(dev), dev)
        s.append(si_sdr_np(c[:n], y.cpu().numpy()))
    m.train(); s = np.array(s)
    return float(s.mean()), float(np.median(s)), float(np.percentile(s, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", default="checkpoints/mf2_ft")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--seg", type=float, default=3.0, help="crop seconds (12 = Phase-2-like long crops)")
    ap.add_argument("--init", default=None, help="warm-start checkpoint (state dict under 'model')")
    ap.add_argument("--all", action="store_true", help="FINAL: train on all 695 pairs (fit + hold); held-out eval becomes in-sample")
    ap.add_argument("--loss", choices=["base", "ab"], default="base", help="ab = artefact-boosted SI-SDR (full + event region)")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--ab-event-w", type=float, default=1.0)
    ap.add_argument("--ext-noise", default=None, help="external category-matched noise root (data/ext_noise)")
    ap.add_argument("--p-ext", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None, help="override the fixed seed (ensemble diversity)")
    ap.add_argument("--unseen", default=None, help="unseen-noise eval set json")
    ap.add_argument("--asr-w", type=float, default=0.0, help="weight of the SraVaani encoder-feature loss (t2asr env only)")
    a = ap.parse_args()
    sd_ = a.seed if a.seed is not None else 1
    random.seed(sd_); np.random.seed(sd_); torch.manual_seed(sd_)
    dev = "cuda"
    lists = json.load(open(_os_here.path.join(_HERE, "ft_data_lists.json")))
    if a.all:
        lists["fit"] = lists["fit"] + lists["hold"]; print("FINAL MODE: training on all", len(lists["fit"]), "pairs (eval is in-sample)", flush=True)
    fit_ids = {d["id"] for d in lists["fit"]}
    snrs = [s for r in json.load(open(_os_here.path.join(_HERE, "synth_recipe_rows.json"))) if r["fn"][:-4] in fit_ids for s in r["snrs"]]
    mixer = Mixer(lists, snrs); mixer.aug = a.aug
    if a.ext_noise: mixer.load_ext(a.ext_noise, a.p_ext)
    unseen = json.load(open(a.unseen)) if a.unseen else None
    mixer.seg = int(a.seg * SR)
    long_mode = a.seg > 6
    if long_mode: mixer.n_events = (2, 3, 4, 4, 5, 6)
    groups = long_groups(lists["hold"]) if long_mode else None
    m = load_model(dev)
    if a.init:
        m.load_state_dict(torch.load(a.init, map_location="cpu")["model"]); print("warm start from", a.init, flush=True)
    asr = None
    if a.asr_w > 0:
        import asr_feat; asr = asr_feat.load_asr(dev); print("SraVaani feature loss on, weight", a.asr_w, flush=True)
    run = lambda x: enhance_one(m, torch.from_numpy(x).to(dev), dev)[0].cpu().numpy()
    def score():  # selection metric: long one-pass held-out SI-SDR in long mode, else short (+unseen set)
        if long_mode:
            m.eval(); r = evaluate_long(run, groups); m.train(); return r + ("",)
        return ft_mgan.evaluate(m, lists["hold"], dev, unseen, run=run)
    if long_mode:
        sm = evaluate(m, lists["hold"], dev)[0]; print(f"start short held-out mean={sm:.2f}", flush=True)
    mean0, med0, p10_0, info0 = score()
    print(f"BASELINE (start) held-out SI-SDR{' LONG' if long_mode else ''}: sel={mean0:.2f} median={med0:.2f} p10={p10_0:.2f} {info0}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr / 20)
    best, t0 = mean0, time.time()
    for step in range(1, a.steps + 1):
        loss = 0.0
        opt.zero_grad(set_to_none=True)
        for _ in range(a.bs):
            xn, cn = mixer.sample()
            x16, c16 = torch.from_numpy(xn).to(dev), torch.from_numpy(cn).to(dev)
            y16, mag, T = enhance_one(m, x16, dev)
            with torch.no_grad():
                c48 = torchaudio.functional.resample(c16, 16000, 48000) * MAXV
                cs = stft(c48, ARGS)[:, :T]
                cmag = torch.sqrt(cs[..., 0] ** 2 + cs[..., 1] ** 2 + 1e-8)
            sc = cmag.mean() + 1e-8
            cm = ((mag / sc) ** 0.3 - (cmag / sc) ** 0.3).pow(2).mean()
            if a.loss == "ab":
                yb, cb, nb = y16.unsqueeze(0), c16.unsqueeze(0), (x16 - c16).unsqueeze(0)
                l = (ab_sisdr_loss(yb, cb, nb, a.alpha) + a.ab_event_w * ab_sisdr_loss(yb, cb, nb, a.alpha, event_mask(nb))) / 10 + 0.5 * cm
            else:
                l = si_sdr_loss(y16.unsqueeze(0), c16.unsqueeze(0)) / 10 + cm
            if asr is not None:
                nb_ = (x16 - c16).unsqueeze(0)
                l = l + a.asr_w * asr_feat.asr_feature_loss(asr, y16.unsqueeze(0), c16.unsqueeze(0), event_mask(nb_))
            (l / a.bs).backward(); loss = loss + l.detach() / a.bs
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} [{(time.time()-t0)/60:.1f}m]", flush=True)
        if step % a.eval_every == 0:
            mean, med, p10, info = score()
            tag = ""
            if mean > best:
                best = mean; torch.save({"model": m.state_dict(), "step": step, "mean": mean}, f"{a.out}/best.pt"); tag = " SAVED"
            print(f"EVAL step {step}: held-out SI-SDR{' LONG' if long_mode else ''} sel={mean:.2f} median={med:.2f} p10={p10:.2f} {info} (baseline {mean0:.2f}){tag}", flush=True)
    torch.save({"model": m.state_dict(), "step": a.steps, "mean": float("nan")}, f"{a.out}/last.pt")
    print(f"DONE best mean {best:.2f} (baseline {mean0:.2f})", flush=True)


if __name__ == "__main__":
    main()
