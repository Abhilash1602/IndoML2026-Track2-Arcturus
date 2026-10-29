"""Fine-tune ClearerVoice MossFormer2_SS_16K (time-domain 2-source separator, Apache-2.0) as a speech / noise-event separator.

A time-domain model (8-sample stride, 24 MossFormer2 layers) whose artefacts differ from those of the STFT-domain separators.
The speech output index is detected on the held-out pairs before training (higher SI-SDR) and then fixed.
Same data/losses as ft_mgan.py: organiser-recipe mixtures (+ external noise, heavier low-SNR tail), AB-SI-SDR (full + event
region) and optionally the SraVaani encoder-feature loss.
Usage (repository root): PYTHONPATH=train python train/ft_mf2ss.py --out checkpoints/<run> [options]   (see train/run_training.sh)
"""
import argparse, json, os, random, time
from types import SimpleNamespace
import numpy as np, torch
import ft_mgan as F
import os as _os_here
_HERE = _os_here.path.dirname(_os_here.path.abspath(__file__))   # train/: data lists and configs

SR = F.SR
SS_ARGS = SimpleNamespace(num_spks=2, encoder_kernel_size=16, encoder_embedding_dim=512, mossformer_sequence_dim=512,
                          num_mossformer_layer=24)
CKPT = "checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt"


def load_model(dev, ckpt=CKPT):
    from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
    m = MossFormer2_SS_16K(SS_ARGS).model
    sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("model", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, missing[:5]
    return m.to(dev)


def separate(m, x, spk):
    """x: (B, T) -> speech estimate (B, T) from output `spk`."""
    outs = m(x)
    return outs[spk][..., :x.shape[-1]]


def run_np(m, spk, dev, win_s=2.0):
    """numpy (T,) -> (T,): fixed 2 s windows with 50% Hann overlap-add (the model's native decode window) for long inputs."""
    @torch.no_grad()
    def one(x):
        return separate(m, torch.from_numpy(x).unsqueeze(0).to(dev), spk)[0].cpu().numpy()
    def run(x):
        w = int(win_s * SR); h = w // 2
        if len(x) <= 2 * w: return one(x)
        out = np.zeros(len(x)); ws = np.zeros(len(x)); win = np.hanning(w) + 1e-3
        for s in list(range(0, len(x) - w, h)) + [len(x) - w]:
            out[s:s + w] += one(x[s:s + w]) * win; ws[s:s + w] += win
        return out / ws
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--init", default=None)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=2.0); ap.add_argument("--asr-w", type=float, default=0.0)
    ap.add_argument("--ext-noise", default="data/ext_noise"); ap.add_argument("--p-ext", type=float, default=0.5)
    ap.add_argument("--snr-shift-p", type=float, default=0.0); ap.add_argument("--snr-shift-max", type=float, default=5.0)
    ap.add_argument("--unseen", default="data/eval_unseen_hard/pairs.json"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="few steps, report memory/speed")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda"
    lists = json.load(open(_os_here.path.join(_HERE, "ft_data_lists.json"))); fit_ids = {d["id"] for d in lists["fit"]}
    snrs = [s for r in json.load(open(_os_here.path.join(_HERE, "synth_recipe_rows.json"))) if r["fn"][:-4] in fit_ids for s in r["snrs"]]
    mixer = F.Mixer(lists, snrs); mixer.aug = True
    if a.ext_noise: mixer.load_ext(a.ext_noise, a.p_ext)
    mixer.snr_shift_p, mixer.snr_shift_max = a.snr_shift_p, a.snr_shift_max
    m = load_model(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu"); m.load_state_dict(sd["model"]); spk = sd["spk"]
    else:   # detect which output carries speech
        m.eval(); sc = []
        for k in (0, 1):
            sc.append(F.run_set(run_np(m, k, dev), lists["hold"][:40])[0])
        spk = int(np.argmax(sc)); print(f"speech output detection on 40 held-out pairs: out0 {sc[0]:.2f} out1 {sc[1]:.2f} -> spk={spk}", flush=True)
    asr = None
    if a.asr_w > 0:
        import asr_feat; asr = asr_feat.load_asr(dev)
    unseen = json.load(open(a.unseen)) if a.unseen else None

    def score():
        return F.evaluate(m, lists["hold"], dev, unseen, run=run_np(m, spk, dev))

    s0, med0, p100, info0 = (0, 0, 0, "smoke") if a.smoke else score()
    print(f"BASELINE (start) sel={s0:.2f} {info0}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr / 20)
    best, t0 = s0, time.time(); m.train()
    for step in range(1, (20 if a.smoke else a.steps) + 1):
        xs, cs = zip(*[mixer.sample() for _ in range(a.bs)])
        x = torch.from_numpy(np.stack(xs)).to(dev); c = torch.from_numpy(np.stack(cs)).to(dev); nz = x - c
        y = separate(m, x, spk)
        loss = (F.ab_sisdr_loss(y, c, nz, a.alpha) + F.ab_sisdr_loss(y, c, nz, a.alpha, F.event_mask(nz))) / 10
        if asr is not None:
            loss = loss + a.asr_w * asr_feat.asr_feature_loss(asr, y, c, F.event_mask(nz))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
        if a.smoke and step == 20:
            print(f"SMOKE 20 steps {(time.time() - t0):.1f}s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, loss {loss.item():.3f}", flush=True); return
        if step % 50 == 0: print(f"step {step} loss {loss.item():.4f} [{(time.time() - t0) / 60:.1f}m]", flush=True)
        if step % a.eval_every == 0:
            mean, med, p10, info = score(); tag = ""
            if mean > best:
                best = mean; torch.save({"model": m.state_dict(), "spk": spk, "step": step, "mean": mean}, f"{a.out}/best.pt"); tag = " SAVED"
            print(f"EVAL step {step}: sel={mean:.2f} {info} (baseline {s0:.2f}){tag}", flush=True)
    torch.save({"model": m.state_dict(), "spk": spk, "step": a.steps}, f"{a.out}/last.pt")
    print(f"DONE best {best:.2f} (baseline {s0:.2f})", flush=True)


if __name__ == "__main__":
    main()
