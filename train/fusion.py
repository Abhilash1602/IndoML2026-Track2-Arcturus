"""Learned time-frequency fusion of the separator outputs (one fixed network, applied identically to every clip).

Input: STFTs (n_fft 512, hop 128) of K separator outputs and of the noisy input. A dilated-conv network predicts per-bin softmax
weights over the K + 1 sources; output = sum_k w_k(t, f) S_k(t, f) -> iSTFT -> RMS-matched to the input. The final convolution is
zero-initialised and the softmax carries log(w0) biases, so training starts from the fixed blend w0.
Training data: cached separator outputs on recipe mixtures (cache_members.py, cache_extend.py, cache_more.py).
Loss: artefact-boosted SI-SDR (whole crop + event regions) + SraVaani encoder-feature loss.

Usage (repository root): FUSION_CFG=train/configs/fusion_v6.json PYTHONPATH=train python train/fusion.py train --out checkpoints/<run>
"""
import argparse, json, os
import numpy as np, soundfile as sf, torch, torch.nn as nn
import ft_mgan as F

MEMBERS = ["asr1_best", "asr1s7", "asr3c", "asr5", "mf2ft", "ss"]
W0 = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]                         # initial blend over the members + noisy input
CACHE, CH, DIL = "data/fusion/train.npz", 48, (1, 2, 4, 8, 1, 2)    # defaults; the shipped fusions use train/configs/*.json
if os.environ.get("FUSION_CFG"):                                  # members, initial blend, cache file, width, dilations
    _c = json.load(open(os.environ["FUSION_CFG"]))
    MEMBERS, W0 = _c["members"], _c["w0"]; CACHE = _c.get("cache", CACHE)
    CH, DIL = _c.get("ch", CH), tuple(_c.get("dil", DIL))
NFFT, HOP = 512, 128


class Fusion(nn.Module):
    def __init__(self, k=None, ch=None, eps=0.01):
        k = k or len(MEMBERS) + 1; ch = ch or CH
        super().__init__()
        self.inp = nn.Conv2d(k, ch, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=(1, d), dilation=(1, d)), nn.GroupNorm(4, ch), nn.PReLU())
                                     for d in DIL])
        self.out = nn.Conv2d(ch, k, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        w0 = torch.tensor(W0, dtype=torch.float32) + eps; self.register_buffer("logw0", torch.log(w0 / w0.sum()).view(1, -1, 1, 1))
        self.register_buffer("win", torch.hann_window(NFFT))

    def forward(self, srcs):
        """srcs: (B, K+1, T) waveforms (members..., noisy). Returns fused waveform (B, T)."""
        B, K, T = srcs.shape
        S = torch.stft(srcs.reshape(B * K, T), NFFT, HOP, window=self.win, return_complex=True).view(B, K, NFFT // 2 + 1, -1)
        feat = torch.log1p(S.abs() * 10)
        h = self.inp(feat)
        for b in self.blocks: h = h + b(h)
        w = torch.softmax(self.out(h) + self.logw0, dim=1)
        Y = (w * S).sum(1)
        return torch.istft(Y, NFFT, HOP, window=self.win, length=T)


def rms_match(x, y):
    ri = x.pow(2).mean(-1, keepdim=True).sqrt(); ro = y.pow(2).mean(-1, keepdim=True).sqrt()
    return y * ri / ro.clamp_min(1e-8)


def train(a):
    dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    d = np.load(CACHE); X = d["x"].astype(np.float32); C = d["c"].astype(np.float32); Y = d["y"]
    assert list(d["members"]) == MEMBERS, (list(d["members"]), MEMBERS)
    n = len(X); nv = 300; tr = np.arange(n - nv); va = np.arange(n - nv, n)
    net = Fusion().to(dev)
    if a.init: net.load_state_dict(torch.load(a.init, map_location="cpu")["model"]); print("init from", a.init, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    g = a.target_g                    # partial-noise target t = c + g*n: keep a little real noise, never distort speech
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr / 20)
    asr = None
    if a.asr_w > 0:
        import asr_feat; asr = asr_feat.load_asr(dev)

    def batch(idx):
        x = torch.from_numpy(X[idx]).to(dev); c = torch.from_numpy(C[idx]).to(dev)
        srcs = torch.cat([torch.from_numpy(Y[idx].astype(np.float32)).to(dev), x.unsqueeze(1)], 1)
        return x, c, srcs

    @torch.no_grad()
    def val():
        net.eval(); s = []
        for i in range(0, nv, 50):
            x, c, srcs = batch(va[i:i + 50]); y = rms_match(x, net(srcs)); t = c + g * (x - c)
            s += [F.si_sdr_np(t[j].cpu().numpy(), y[j].cpu().numpy()) for j in range(len(x))]     # vs the training target
        net.train(); return float(np.mean(s))

    v0 = val(); best = v0; os.makedirs(a.out, exist_ok=True)
    print(f"BASELINE (fixed blend) val SI-SDR {v0:.2f}", flush=True)
    for step in range(1, a.steps + 1):
        x, c, srcs = batch(np.random.choice(tr, a.bs, replace=False)); nz = x - c
        t, nt = c + g * nz, (1 - g) * nz                              # span{t, nt} = span{c, n}: same artefact projection
        y = net(srcs); em = F.event_mask(nz)
        loss = (F.ab_sisdr_loss(y, t, nt, a.alpha) + F.ab_sisdr_loss(y, t, nt, a.alpha, em)) / 10
        if asr is not None: loss = loss + a.asr_w * asr_feat.asr_feature_loss(asr, y, t if a.asr_target == "t" else c, em)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        if step % 100 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)
        if step % a.eval_every == 0:
            v = val(); tag = ""
            if v > best: best = v; torch.save({"model": net.state_dict(), "step": step, "val": v}, f"{a.out}/best.pt"); tag = " SAVED"
            print(f"EVAL step {step}: val SI-SDR {v:.2f} (fixed blend {v0:.2f}){tag}", flush=True)
    torch.save({"model": net.state_dict(), "step": a.steps}, f"{a.out}/last.pt"); print(f"DONE best {best:.2f} (fixed {v0:.2f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-g", type=float, default=0.0, help="train toward c + g*(x-c) instead of c")
    ap.add_argument("--asr-target", choices=["t", "c"], default="t"); ap.add_argument("--init", default=None)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--alpha", type=float, default=2.0); ap.add_argument("--asr-w", type=float, default=1.0)
    a = ap.parse_args()
    train(a)
