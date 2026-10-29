"""In-domain fine-tuning of the MossFormerGAN_SE_16K generator (ClearerVoice-Studio, Apache-2.0).

Training mixtures follow the organisers' synthetic recipe measured on their validation pairs: noise only inside 1-4 events of
0.2-1.1 s, per-event SNR drawn from the empirical distribution (median -7.5 dB). Noise events are the exact noise
(noisy - clean) of the 591 training pairs, optionally mixed with external category-matched noise; the 104 held-out pairs are
used only for checkpoint selection. Clean speech is the 591 clean references plus data/bank/clean.
Generator only (no discriminator). Losses: artefact-boosted SI-SDR on the whole crop and on event regions (--loss ab),
compressed-magnitude MSE, optional SraVaani encoder-feature loss (--asr-w), optional partial-noise target (--target-g) and
noise-subtraction form (--residual).

Usage (repository root): PYTHONPATH=train python train/ft_mgan.py --out checkpoints/<run> [options]   (see train/run_training.sh)
"""
import argparse, json, os, random, time
from types import SimpleNamespace
import numpy as np, soundfile as sf, torch
from clearvoice.models.mossformer_gan_se.generator import SyncANet
from clearvoice.utils.misc import stft, istft, power_compress, power_uncompress
import os as _os_here
_HERE = _os_here.path.dirname(_os_here.path.abspath(__file__))   # train/: data lists and configs

SR, SEG = 16000, 3 * 16000
CFG = SimpleNamespace(win_type="hamming", win_len=400, win_inc=100, fft_len=400)


def load_wav(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(1)
    assert sr == SR, (p, sr)
    return x


def load_model(dev):
    m = SyncANet(num_channel=64, num_features=CFG.fft_len // 2 + 1)
    ck = torch.load("checkpoints/MossFormerGAN_SE_16K/last_best_checkpoint.pt", map_location="cpu")
    sd = ck.get("model", ck)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    return m.to(dev)


def grad_checkpoint(m):
    """Recompute activations of the heavy SyncANet sub-modules in backward (fits 12 s crops in memory)."""
    from torch.utils.checkpoint import checkpoint
    mods = [m.dense_encoder, m.mask_decoder, m.complex_decoder] + list(m.blocks)
    for mod in mods:
        f = mod.forward
        mod.forward = (lambda f: lambda *a: checkpoint(f, *a, use_reentrant=False) if torch.is_grad_enabled() else f(*a))(f)


def enhance(m, x):
    """Differentiable version of clearvoice's _decode_one_audio_mossformergan_se_16k. x: (B, T)."""
    T = x.shape[-1]
    nframe = int(np.ceil(T / CFG.win_inc)); pad = nframe * CFG.win_inc - T
    if pad: x = torch.cat([x, x[:, :pad]], dim=-1)
    norm = torch.sqrt(x.shape[-1] / (x.pow(2).sum(-1, keepdim=True) + 1e-8))
    spec = stft(x * norm, CFG, center=True, periodic=True, onesided=True).to(torch.float32)
    spec = power_compress(spec).permute(0, 1, 3, 2)
    re, im = m(spec)
    re, im = re.permute(0, 1, 3, 2), im.permute(0, 1, 3, 2)
    out = istft(power_uncompress(re, im).squeeze(1), CFG, center=True, periodic=True, onesided=True)
    return (out / norm)[:, :T], re, im


def enhance_res(m, x):
    """Noise-SUBTRACTION forward: the decoders predict the NOISE in the compressed complex domain and it is subtracted from the
    input there: Y_c = X_c - (mask_n * |X_c| e^{j phase} + C_n). With zero-initialised heads Y_c = X_c (identity, speech untouched);
    gradients flow because power_uncompress is evaluated at X_c != 0 (a post-hoc y = x - n_hat with zero heads gets no gradient)."""
    T = x.shape[-1]
    nframe = int(np.ceil(T / CFG.win_inc)); pad = nframe * CFG.win_inc - T
    if pad: x = torch.cat([x, x[:, :pad]], dim=-1)
    norm = torch.sqrt(x.shape[-1] / (x.pow(2).sum(-1, keepdim=True) + 1e-8))
    spec = power_compress(stft(x * norm, CFG, center=True, periodic=True, onesided=True).to(torch.float32)).permute(0, 1, 3, 2)
    mag = torch.sqrt(spec[:, 0] ** 2 + spec[:, 1] ** 2 + 1e-12).unsqueeze(1)
    phase = torch.angle(torch.complex(spec[:, 0], spec[:, 1])).unsqueeze(1)
    h = m.dense_encoder(torch.cat([mag, spec], 1))
    for i in range(m.n_layers): h = m.blocks[i](h)
    mask_n = m.mask_decoder(h); cn = m.complex_decoder(h)
    yre = spec[:, 0:1] - (mask_n * mag * torch.cos(phase) + cn[:, 0:1]); yim = spec[:, 1:2] - (mask_n * mag * torch.sin(phase) + cn[:, 1:2])
    out = istft(power_uncompress(yre.permute(0, 1, 3, 2), yim.permute(0, 1, 3, 2)).squeeze(1), CFG, center=True, periodic=True, onesided=True)
    return (out / norm)[:, :T]


def si_sdr_np(ref, est):  # official scorer form (no mean removal)
    n = min(len(ref), len(est)); ref, est = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    s = (est @ ref) / (ref @ ref) * ref; e = est - s
    return float(np.clip(10 * np.log10((s @ s) / (e @ e)), -100, 100))


def si_sdr_loss(est, ref):
    est = est - est.mean(-1, keepdim=True); ref = ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-8) * ref
    e = est - s
    return -(10 * torch.log10(s.pow(2).sum(-1) / (e.pow(2).sum(-1) + 1e-8) + 1e-8)).mean()


def event_mask(n, dil=1600):
    """(B,T) noise -> (B,T) float mask of samples where noise is present, dilated by +-dil samples (0.1 s)."""
    m = (n.abs() > 1e-6).float().unsqueeze(1)
    return torch.nn.functional.max_pool1d(m, 2 * dil + 1, stride=1, padding=dil).squeeze(1)


def ab_sisdr_loss(y, c, n, alpha=2.0, mask=None, eps=1e-8):
    """Artefact-boosted SI-SDR (AB-SDR, arXiv 2404.14860, L=1 bases, no mean removal like the official SI-SDR).
    y/c/n: (B,T) estimate, clean, noise (n = mixture - clean). Orthogonal projection of y on span{c, n}:
    s_t = <y,c>/<c,c> c ; e_noise = P_{c,n} y - s_t ; e_art = y - P_{c,n} y ;
    L = -10 log10(|s_t|^2 / |e_noise + alpha e_art|^2). alpha=1 == -SI-SDR. mask restricts all dot products to a region."""
    if mask is not None:
        y, c, n = y * mask, c * mask, n * mask
    cc, nn, cn = (c * c).sum(-1), (n * n).sum(-1), (c * n).sum(-1)
    yc, yn = (y * c).sum(-1), (y * n).sum(-1)
    det = cc * nn - cn * cn
    ok = det > 1e-6 * cc * nn.clamp_min(eps)            # noise present and not collinear with speech
    a = torch.where(ok, (yc * nn - yn * cn) / det.clamp_min(eps), yc / cc.clamp_min(eps))
    b = torch.where(ok, (yn * cc - yc * cn) / det.clamp_min(eps), torch.zeros_like(yc))
    st = (yc / cc.clamp_min(eps)).unsqueeze(-1) * c
    py = a.unsqueeze(-1) * c + b.unsqueeze(-1) * n
    err = (py - st) + alpha * (y - py)
    return -(10 * torch.log10(st.pow(2).sum(-1) / (err.pow(2).sum(-1) + eps) + eps)).mean()


def sar_in_events(y, c, n, events, sr=SR):
    """Signal-to-artefact ratio (dB) inside ground-truth events: artefact = y - P_{c,n} y restricted to events."""
    g = np.zeros(len(c), bool)
    for st_, en_ in events: g[int(st_ * sr):min(len(c), int(en_ * sr))] = True
    if not g.any(): return None
    y, c, n = y[g].astype(np.float64), c[g].astype(np.float64), n[g].astype(np.float64)
    if c @ c < 1e-8: return None                 # silent clean reference inside the events (59 held-out events)
    A = np.stack([c, n], 1); coef, *_ = np.linalg.lstsq(A, y, rcond=None); py = A @ coef
    st = (y @ c) / (c @ c) * c
    return float(10 * np.log10((st @ st) / (((y - py) ** 2).sum() + 1e-12)))


class Mixer:
    def __init__(self, lists, snrs):
        self.noise, self.clean = [], []
        for d in lists["fit"]:
            x, c = load_wav(d["noisy"]), load_wav(d["clean"]); n = min(len(x), len(c)); nz = x[:n] - c[:n]
            for st, en in d["events"]:
                a, b = int(st * SR), min(n, int(en * SR))
                if b - a > 0.05 * SR: self.noise.append(nz[a:b].copy())
            self.clean.append(c[:n])
        for p in lists["clean_bank"]:
            self.clean.append(load_wav(p))
        self.clean = [c for c in self.clean if len(c) > SR and np.sqrt((c ** 2).mean()) > 1e-3]
        self.snrs = np.asarray(snrs, np.float32)
        print(f"mixer: {len(self.noise)} noise events, {len(self.clean)} clean sources, {len(self.snrs)} SNR samples", flush=True)

    aug = False
    ext, ext_w, p_ext = [], None, 0.0   # external category-matched noise segments (load_ext)
    snr_shift_p, snr_shift_max = 0.0, 0.0  # heavier low-SNR tail: with prob p subtract U(0, max) dB from the event SNR
    seg = SEG                      # crop length in samples (Phase-2-like: 12 s)
    n_events = (1, 2, 3, 4, 4)     # choices per crop

    def _clean(self):
        """A clean crop of self.seg samples; short sources are concatenated (Phase 2 clips are ~12 s)."""
        c = random.choice(self.clean)
        while len(c) < self.seg:
            c = np.concatenate([c, random.choice(self.clean)])
        s = random.randint(0, len(c) - self.seg)
        return c[s:s + self.seg].copy()

    def load_ext(self, root, p_ext):
        """data/ext_noise/<category>/*.wav; categories weighted to the organiser event distribution."""
        import glob
        w_org = {"human_non_speech": 670, "vehicle_traffic": 626, "animal": 548, "baby_child": 231,
                 "phone_signal_alarm": 78, "appliance_machine": 32, "singing_music": 28}
        self.ext, names = [], []
        for cat, w in w_org.items():
            fs = sorted(glob.glob(f"{root}/{cat}/*.wav"))
            if fs: self.ext.append((w, [load_wav(f) for f in fs])); names.append(cat)
        tot = sum(w for w, _ in self.ext); self.ext_w = [w / tot for w, _ in self.ext]; self.p_ext = p_ext
        print("external noise:", {k: len(v) for k, (_, v) in zip(names, self.ext)}, "p_ext", p_ext, flush=True)

    def _noise(self):
        if self.ext and random.random() < self.p_ext:
            pool = self.ext[np.random.choice(len(self.ext), p=self.ext_w)][1]
            nz = random.choice(pool)
        else:
            nz = random.choice(self.noise)
        if self.aug:
            r = random.uniform(0.9, 1.1)  # speed perturbation by resampling
            idx = np.arange(0, len(nz) - 1, r)
            nz = np.interp(idx, np.arange(len(nz)), nz).astype(np.float32)
            if random.random() < 0.5: nz = -nz
            if random.random() < 0.15:  # occasional overlap of two events
                nz2 = random.choice(self.noise)[: len(nz)]
                nz = nz.copy(); nz[: len(nz2)] += nz2 * random.uniform(0.3, 1.0)
        return nz[: int(1.2 * SR)]

    def sample(self):
        if self.seg != SEG:
            c = self._clean()
        else:
            c = random.choice(self.clean)
            if len(c) >= SEG:
                s = random.randint(0, len(c) - SEG); c = c[s:s + SEG].copy()
            else:
                c = np.pad(c, (0, SEG - len(c)))
        if self.aug: c = c * random.uniform(0.5, 1.5)
        x = c.copy()
        for _ in range(random.choice(self.n_events)):
            nz = self._noise()
            a = random.randint(0, max(0, len(c) - len(nz))); b = a + len(nz)
            pc = float((c[a:b] ** 2).mean()); pc = pc if pc > 1e-7 else float((c ** 2).mean())
            pn = float((nz ** 2).mean()) + 1e-12
            snr = float(random.choice(self.snrs))
            if self.snr_shift_p and random.random() < self.snr_shift_p: snr -= random.uniform(0, self.snr_shift_max)
            x[a:b] += nz * np.sqrt(pc / (pn * 10 ** (snr / 10)))
        pk = np.abs(x).max()
        if pk > 0.99: x, c = x / pk * 0.99, c / pk * 0.99
        return x.astype(np.float32), c.astype(np.float32)


def run_set(run, pairs):
    """pairs: dicts with noisy/clean(/events). Returns mean, median, p10 SI-SDR and mean SAR inside events."""
    s, sar = [], []
    for d in pairs:
        x, c = load_wav(d["noisy"]), load_wav(d["clean"]); n = min(len(x), len(c)); x, c = x[:n], c[:n]
        y = run(x); s.append(si_sdr_np(c, y))
        if d.get("events"):
            v = sar_in_events(y, c, x - c, d["events"])
            if v is not None: sar.append(v)
    s = np.array(s)
    return float(s.mean()), float(np.median(s)), float(np.percentile(s, 10)), (float(np.mean(sar)) if sar else float("nan"))


@torch.no_grad()
def evaluate(m, hold, dev, unseen=None, run=None):
    """Seen-noise held-out (+ optional unseen-noise set). Returns (sel, med, p10, info) where sel is the selection metric:
    seen mean SI-SDR, or the average of seen and unseen means when an unseen set is given."""
    m.eval()
    run = run or (lambda x: enhance(m, torch.from_numpy(x).unsqueeze(0).to(dev))[0][0].cpu().numpy())
    sm, smed, sp10, ssar = run_set(run, hold)
    info = f"seen {sm:.2f} (SAR_in {ssar:.2f})"; sel = sm
    if unseen:
        um, umed, up10, usar = run_set(run, unseen)
        info += f" | unseen {um:.2f} (p10 {up10:.2f}, SAR_in {usar:.2f})"; sel = 0.5 * (sm + um)
    m.train()
    return sel, smed, sp10, info


def long_groups(hold):
    """Held-out pairs concatenated in fixed groups of 3 (~15 s, the length of the Phase 2 clips)."""
    pairs = []
    for d in hold:
        x, c = load_wav(d["noisy"]), load_wav(d["clean"]); n = min(len(x), len(c)); pairs.append((x[:n], c[:n]))
    out = []
    for i in range(0, len(pairs), 3):
        g = pairs[i:i + 3]
        out.append((np.concatenate([x for x, _ in g]), [c for _, c in g], np.cumsum([0] + [len(x) for x, _ in g])))
    return out


@torch.no_grad()
def evaluate_long(run, groups):
    """run: np (T,) -> np (T,) one-pass enhancement of a whole long clip; scored per original pair."""
    s = []
    for X, cs, b in groups:
        Y = run(X)
        s += [si_sdr_np(c, Y[b[k]:b[k + 1]]) for k, c in enumerate(cs)]
    s = np.array(s)
    return float(s.mean()), float(np.median(s)), float(np.percentile(s, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--out", default="checkpoints/mgan_ft")
    ap.add_argument("--init", default=None, help="warm-start checkpoint (state dict under 'model')")
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay to lr/20")
    ap.add_argument("--aug", action="store_true", help="noise speed/polarity/overlap + speech gain augmentation")
    ap.add_argument("--seg", type=float, default=3.0, help="crop seconds (12 = Phase 2 clip length)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (long crops)")
    ap.add_argument("--all", action="store_true", help="train on all 695 pairs (fit + held-out); held-out evaluation becomes in-sample")
    ap.add_argument("--loss", choices=["base", "ab"], default="base", help="ab = artefact-boosted SI-SDR (full + event region)")
    ap.add_argument("--alpha", type=float, default=2.0, help="AB-SDR artefact weight (1 = plain SI-SDR)")
    ap.add_argument("--ab-event-w", type=float, default=1.0, help="weight of the event-region AB-SDR term")
    ap.add_argument("--ext-noise", default=None, help="external category-matched noise root (data/ext_noise)")
    ap.add_argument("--p-ext", type=float, default=0.5, help="probability an event uses external noise")
    ap.add_argument("--seed", type=int, default=None, help="override the fixed seed (ensemble diversity)")
    ap.add_argument("--unseen", default=None, help="unseen-noise eval set json (data/eval_unseen/pairs.json)")
    ap.add_argument("--asr-w", type=float, default=0.0, help="weight of the SraVaani encoder-feature loss")
    ap.add_argument("--residual", action="store_true",
                    help="noise-subtraction mode: network output is the NOISE estimate, y = x - n_hat; final decoder layers zero-init "
                         "so the model starts as identity (speech untouched) and errors stay noise-like")
    ap.add_argument("--snr-shift-p", type=float, default=0.0, help="prob of lowering an event SNR by U(0, --snr-shift-max) dB")
    ap.add_argument("--snr-shift-max", type=float, default=5.0)
    ap.add_argument("--target-g", type=float, default=0.0, help="train toward c + g*(x-c): keep attenuated real noise, no speech distortion")
    a = ap.parse_args()
    sd_ = a.seed if a.seed is not None else 0
    random.seed(sd_); np.random.seed(sd_); torch.manual_seed(sd_)
    dev = "cuda"
    lists = json.load(open(_os_here.path.join(_HERE, "ft_data_lists.json")))
    if a.all:
        lists["fit"] = lists["fit"] + lists["hold"]; print("ALL-DATA MODE: training on all", len(lists["fit"]), "pairs (eval is in-sample)", flush=True)
    fit_ids = {d["id"] for d in lists["fit"]}
    snrs = [s for r in json.load(open(_os_here.path.join(_HERE, "synth_recipe_rows.json"))) if r["fn"][:-4] in fit_ids for s in r["snrs"]]
    mixer = Mixer(lists, snrs)
    mixer.aug = a.aug
    if a.ext_noise: mixer.load_ext(a.ext_noise, a.p_ext)
    mixer.snr_shift_p, mixer.snr_shift_max = a.snr_shift_p, a.snr_shift_max
    unseen = json.load(open(a.unseen)) if a.unseen else None
    mixer.seg = int(a.seg * SR)
    if a.seg > 6: mixer.n_events = (2, 3, 4, 4, 5, 6)
    long_mode = a.seg > 6
    groups = long_groups(lists["hold"]) if long_mode else None
    m = load_model(dev)
    if a.init:
        m.load_state_dict(torch.load(a.init, map_location="cpu")["model"]); print("warm start from", a.init, flush=True)
    if a.residual:
        for conv in (m.mask_decoder.final_conv, m.complex_decoder.conv):
            nn_ = torch.nn; nn_.init.zeros_(conv.weight)
            if conv.bias is not None: nn_.init.zeros_(conv.bias)
        print("RESIDUAL (noise-subtraction) mode: decoder heads zero-init -> identity at start", flush=True)
    if a.grad_ckpt: grad_checkpoint(m)
    asr = None
    if a.asr_w > 0:
        import asr_feat; asr = asr_feat.load_asr(dev); print("SraVaani feature loss on, weight", a.asr_w, flush=True)
    def run(x):
        xt = torch.from_numpy(x).unsqueeze(0).to(dev)
        return (enhance_res(m, xt) if a.residual else enhance(m, xt)[0])[0].cpu().numpy()
    def score():  # selection metric: long one-pass held-out SI-SDR in long mode, else short (+unseen set)
        if long_mode:
            m.eval(); r = evaluate_long(run, groups); m.train(); return r + ("",)
        return evaluate(m, lists["hold"], dev, unseen, run=run)
    if long_mode:
        sm = evaluate(m, lists["hold"], dev)[0]; print(f"start short held-out mean={sm:.2f}", flush=True)
    mean0, med0, p10_0, info0 = score()
    print(f"BASELINE (start) held-out SI-SDR{' LONG' if long_mode else ''}: sel={mean0:.2f} median={med0:.2f} p10={p10_0:.2f} {info0}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr / 20) if a.cosine else None
    best, t0 = mean0, time.time()
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            xs, cs = zip(*[mixer.sample() for _ in range(a.bs)])
            x = torch.from_numpy(np.stack(xs)).to(dev); c = torch.from_numpy(np.stack(cs)).to(dev)
            if a.target_g:                  # partial-noise target t = c + g*n (then x - t = (1-g) n: every loss term follows)
                c = c + a.target_g * (x - c)
            if a.residual:
                y = enhance_res(m, x); re = im = None
            else:
                y, re, im = enhance(m, x)
            with torch.no_grad():
                nrm = torch.sqrt(c.shape[-1] / (x.pow(2).sum(-1, keepdim=True) + 1e-8))  # same norm as the input
                tgt = power_compress(stft(c * nrm, CFG, center=True, periodic=True, onesided=True).float()).permute(0, 1, 3, 2)
                tre, tim = tgt[:, 0:1].permute(0, 1, 3, 2), tgt[:, 1:2].permute(0, 1, 3, 2)
            if a.residual:
                nz = x - c
                loss = (ab_sisdr_loss(y, c, nz, a.alpha) + a.ab_event_w * ab_sisdr_loss(y, c, nz, a.alpha, event_mask(nz))) / 10
            elif a.loss == "ab":
                mag = torch.sqrt(re ** 2 + im ** 2 + 1e-8); tmag = torch.sqrt(tre ** 2 + tim ** 2 + 1e-8)
                nz = x - c
                loss = (ab_sisdr_loss(y, c, nz, a.alpha) + a.ab_event_w * ab_sisdr_loss(y, c, nz, a.alpha, event_mask(nz))) / 10 \
                    + 0.5 * (mag - tmag).pow(2).mean()
            else:
                mag = torch.sqrt(re ** 2 + im ** 2 + 1e-8); tmag = torch.sqrt(tre ** 2 + tim ** 2 + 1e-8)
                loss = si_sdr_loss(y, c) / 10 + (mag - tmag).pow(2).mean() + 0.5 * ((re - tre).pow(2).mean() + (im - tim).pow(2).mean())
            if asr is not None:
                loss = loss + a.asr_w * asr_feat.asr_feature_loss(asr, y, c, event_mask(x - c))
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if sched: sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} [{(time.time()-t0)/60:.1f}m]", flush=True)
        if step % a.eval_every == 0:
            mean, med, p10, info = score()
            tag = ""
            if mean > best:
                best = mean; torch.save({"model": m.state_dict(), "step": step, "mean": mean, "residual": a.residual}, f"{a.out}/best.pt"); tag = " SAVED"
            print(f"EVAL step {step}: held-out SI-SDR{' LONG' if long_mode else ''} sel={mean:.2f} median={med:.2f} p10={p10:.2f} {info} (baseline {mean0:.2f}){tag}", flush=True)
    torch.save({"model": m.state_dict(), "step": a.steps, "mean": float("nan"), "residual": a.residual}, f"{a.out}/last.pt")
    print(f"DONE best mean {best:.2f} (baseline {mean0:.2f})", flush=True)


if __name__ == "__main__":
    main()
