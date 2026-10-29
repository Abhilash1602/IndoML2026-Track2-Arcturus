"""SraVaani-1.0 encoder-feature loss for separator fine-tuning.

The mandated ASR is frozen; gradients flow through its exported TorchScript encoder into the separator. The front end is a
differentiable copy of SraVaaniTDTModel.extract_features. Loss = mean L1 between the encoder outputs of the estimate and of the
clean reference, optionally restricted to encoder frames that overlap noise events (where the speech artefacts occur).
"""
import torch

ASR_REPO = "ARTPARK-IISc/SraVaani-1.0"


def load_asr(dev):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    path = snapshot_download(ASR_REPO)
    asr = AutoModel.from_pretrained(path, trust_remote_code=True).to(dev).eval()
    asr._ensure_loaded()
    for p in getattr(asr.encoder, "parameters", lambda: [])():
        p.requires_grad_(False)
    return asr


def _norm(x, seq_len, c):
    B, _, T = x.shape
    steps = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    valid = steps < seq_len.unsqueeze(1)
    denom = valid.sum(dim=1)
    mean = torch.where(valid.unsqueeze(1), x, torch.zeros_like(x)).sum(2) / denom.unsqueeze(1)
    var = torch.sum(torch.where(valid.unsqueeze(1), x - mean.unsqueeze(2), torch.zeros_like(x)) ** 2, dim=2) / (denom.unsqueeze(1) - 1.0)
    std = torch.sqrt(var + 1e-12).masked_fill(torch.isnan(var), 0.0) + c
    return (x - mean.unsqueeze(2)) / std.unsqueeze(2)


def frontend(asr, wav):
    p = asr._p; dev = wav.device
    x = torch.cat((wav[:, :1], wav[:, 1:] - p["preemph"] * wav[:, :-1]), dim=1)
    spec = torch.stft(x, n_fft=p["n_fft"], hop_length=p["hop_length"], win_length=p["win_length"],
                      window=asr._window.to(dev).float(), center=True, pad_mode="constant", return_complex=True)
    mag = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-12).pow(p["mag_power"])
    mel = torch.log(torch.matmul(asr._fb.to(dev).float(), mag) + p["log_zero_guard_value"])
    seq_len = torch.full((wav.shape[0],), mel.shape[-1], device=dev, dtype=torch.long)
    return _norm(mel, seq_len, p["CONSTANT"]), seq_len


def encode(asr, wav):
    f, L = frontend(asr, wav)
    enc, enc_len = asr.encoder(f.to(asr._io_dtype), L)
    return enc.float()                                       # (B, D, T')


def asr_feature_loss(asr, y, c, mask=None):
    """y, c: (B, T) 16 kHz estimate / clean. mask: (B, T) event mask (1 = event) -> frame weights. Returns scalar."""
    e_y = encode(asr, y)
    with torch.no_grad():
        e_c = encode(asr, c)
    T = min(e_y.shape[-1], e_c.shape[-1]); e_y, e_c = e_y[..., :T], e_c[..., :T]
    d = (e_y - e_c).abs().mean(1)                            # (B, T')
    if mask is None:
        return d.mean()
    w = torch.nn.functional.adaptive_max_pool1d(mask.unsqueeze(1), T).squeeze(1)
    return (d * w).sum() / w.sum().clamp_min(1.0)
