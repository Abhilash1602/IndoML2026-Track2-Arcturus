"""Fine-tuned ClearerVoice-Studio separators (Apache-2.0 model code, `pip install clearvoice --no-deps`).

MossFormerGAN_SE_16K (SyncANet generator) and MossFormer2_SE_48K (mask network), loaded with our in-domain fine-tuned
weights. Inference mirrors clearvoice's own decode functions; every clip is processed in fixed-length windows with 50%
Hann overlap-add (the same window size for every clip; clips shorter than the window are processed in one pass).
"""
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio

SR = 16000
MGAN_CFG = SimpleNamespace(win_type="hamming", win_len=400, win_inc=100, fft_len=400)
MF2_ARGS = SimpleNamespace(mode="inference", win_type="hamming", win_len=1920, win_inc=384, fft_len=1920,
                           num_mels=60, sampling_rate=48000)
MAXV = 32768.0


def _strip(sd):
    sd = sd.get("model", sd)
    return {k.replace("module.", ""): v for k, v in sd.items()}


class MossFormerGAN:
    def __init__(self, ckpt, dev):
        from clearvoice.models.mossformer_gan_se.generator import SyncANet
        from clearvoice.utils.misc import stft, istft, power_compress, power_uncompress
        self._stft, self._istft, self._pc, self._pu = stft, istft, power_compress, power_uncompress
        self.m = SyncANet(num_channel=64, num_features=MGAN_CFG.fft_len // 2 + 1)
        self.m.load_state_dict(_strip(torch.load(ckpt, map_location="cpu")))
        self.m.to(dev).eval(); self.dev = dev

    @torch.no_grad()
    def __call__(self, x):
        """x: (T,) float32 16 kHz -> (T,) enhanced."""
        t = torch.from_numpy(x).unsqueeze(0).to(self.dev); T = t.shape[-1]
        nframe = int(np.ceil(T / MGAN_CFG.win_inc)); pad = nframe * MGAN_CFG.win_inc - T
        if pad: t = torch.cat([t, t[:, :pad]], dim=-1)
        norm = torch.sqrt(t.shape[-1] / (t.pow(2).sum(-1, keepdim=True) + 1e-8))
        spec = self._stft(t * norm, MGAN_CFG, center=True, periodic=True, onesided=True).to(torch.float32)
        spec = self._pc(spec).permute(0, 1, 3, 2)
        re, im = self.m(spec)
        out = self._istft(self._pu(re.permute(0, 1, 3, 2), im.permute(0, 1, 3, 2)).squeeze(1), MGAN_CFG,
                          center=True, periodic=True, onesided=True)
        return (out / norm)[0, :T].cpu().numpy()


class MossFormer2:
    def __init__(self, ckpt, dev):
        from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        from clearvoice.utils.misc import stft, istft, compute_fbank
        self._stft, self._istft, self._fbank = stft, istft, compute_fbank
        self.m = MossFormer2_SE_48K(MF2_ARGS).model
        self.m.load_state_dict(_strip(torch.load(ckpt, map_location="cpu")))
        self.m.to(dev).eval(); self.dev = dev

    @torch.no_grad()
    def __call__(self, x):
        x16 = torch.from_numpy(x).to(self.dev)
        x48 = torchaudio.functional.resample(x16, 16000, 48000) * MAXV
        fb = self._fbank(x48.detach().cpu().unsqueeze(0), MF2_ARGS)
        d1 = torchaudio.functional.compute_deltas(fb.T); d2 = torchaudio.functional.compute_deltas(d1)
        f = torch.cat([fb, d1.T, d2.T], dim=1).unsqueeze(0).to(self.dev)
        mask = self.m(f)[-1]
        spec = self._stft(x48, MF2_ARGS)
        T = min(spec.shape[1], mask.shape[1])
        ms = spec[:, :T] * mask.permute(2, 1, 0)[:, :T]
        out48 = self._istft(torch.complex(ms[..., 0], ms[..., 1]), MF2_ARGS, x48.shape[-1])
        y = torchaudio.functional.resample(out48 / MAXV, 48000, 16000)[: len(x)].cpu().numpy()
        return np.pad(y, (0, max(0, len(x) - len(y))))


def chunked(run, x, win_s):
    """Fixed-window (win_s seconds) 50% overlap-add with a Hann window; identical for every clip."""
    w = int(win_s * SR); h = w // 2
    if len(x) <= w:
        return run(x)
    out = np.zeros(len(x)); wsum = np.zeros(len(x)); win = np.hanning(w) + 1e-3
    for s in list(range(0, len(x) - w, h)) + [len(x) - w]:
        out[s:s + w] += run(x[s:s + w]) * win; wsum[s:s + w] += win
    return (out / wsum).astype(np.float32)


def rms_match(x, y):
    """Output-level normalisation applied to every clip: match input RMS, 0.99 peak guard."""
    y = y[:len(x)]
    if len(y) < len(x): y = np.pad(y, (0, len(x) - len(y)))
    ri, ro = np.sqrt((x ** 2).mean()), np.sqrt((y ** 2).mean())
    if ri > 1e-8 and ro > 1e-8: y = y * (ri / ro)
    pk = np.abs(y).max()
    return (y / pk * 0.99 if pk > 0.99 else y).astype(np.float32)


class MossFormerGANResidual(MossFormerGAN):
    """Noise-subtraction MossFormerGAN: the decoders predict the NOISE in the compressed complex domain, which is subtracted from
    the input there (Y_c = X_c - (mask_n |X_c| e^{j phase} + C_n)); speech is never reconstructed directly."""

    @torch.no_grad()
    def __call__(self, x):
        m = self.m; t = torch.from_numpy(x).unsqueeze(0).to(self.dev); T = t.shape[-1]
        nframe = int(np.ceil(T / MGAN_CFG.win_inc)); pad = nframe * MGAN_CFG.win_inc - T
        if pad: t = torch.cat([t, t[:, :pad]], dim=-1)
        norm = torch.sqrt(t.shape[-1] / (t.pow(2).sum(-1, keepdim=True) + 1e-8))
        spec = self._pc(self._stft(t * norm, MGAN_CFG, center=True, periodic=True, onesided=True).to(torch.float32)).permute(0, 1, 3, 2)
        mag = torch.sqrt(spec[:, 0] ** 2 + spec[:, 1] ** 2 + 1e-12).unsqueeze(1)
        phase = torch.angle(torch.complex(spec[:, 0], spec[:, 1])).unsqueeze(1)
        h = m.dense_encoder(torch.cat([mag, spec], 1))
        for i in range(m.n_layers): h = m.blocks[i](h)
        mask_n = m.mask_decoder(h); cn = m.complex_decoder(h)
        yre = spec[:, 0:1] - (mask_n * mag * torch.cos(phase) + cn[:, 0:1]); yim = spec[:, 1:2] - (mask_n * mag * torch.sin(phase) + cn[:, 1:2])
        out = self._istft(self._pu(yre.permute(0, 1, 3, 2), yim.permute(0, 1, 3, 2)).squeeze(1), MGAN_CFG, center=True, periodic=True, onesided=True)
        return (out / norm)[0, :T].cpu().numpy()


class MossFormer2SS:
    """MossFormer2_SS_16K (time-domain 2-source separator) fine-tuned as speech/noise separator; returns the speech output."""

    def __init__(self, ckpt, dev):
        from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        args = SimpleNamespace(num_spks=2, encoder_kernel_size=16, encoder_embedding_dim=512, mossformer_sequence_dim=512, num_mossformer_layer=24)
        sd = torch.load(ckpt, map_location="cpu"); self.spk = sd.get("spk", 0)
        self.m = MossFormer2_SS_16K(args).model; self.m.load_state_dict(_strip(sd)); self.m.to(dev).eval(); self.dev = dev

    @torch.no_grad()
    def __call__(self, x):
        return self.m(torch.from_numpy(x).unsqueeze(0).to(self.dev))[self.spk][0, :len(x)].cpu().numpy()


def make(kind, ckpt, dev):
    return {"mossformergan": MossFormerGAN, "mossformergan_res": MossFormerGANResidual,
            "mossformer2": MossFormer2, "mossformer2_ss": MossFormer2SS}[kind](ckpt, dev)
