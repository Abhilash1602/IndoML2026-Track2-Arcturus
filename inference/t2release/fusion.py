"""Learned TF fusion network (one fixed network for every clip): per-bin softmax weights over the K member outputs + the noisy input."""
import torch, torch.nn as nn

NFFT, HOP = 512, 128


class Fusion(nn.Module):
    def __init__(self, k, ch, dil, w0, eps=0.01):
        super().__init__()
        self.inp = nn.Conv2d(k, ch, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=(1, d), dilation=(1, d)), nn.GroupNorm(4, ch), nn.PReLU()) for d in dil])
        self.out = nn.Conv2d(ch, k, 1)
        w0 = torch.tensor(w0, dtype=torch.float32) + eps; self.register_buffer("logw0", torch.log(w0 / w0.sum()).view(1, -1, 1, 1))
        self.register_buffer("win", torch.hann_window(NFFT))

    def forward(self, srcs):
        B, K, T = srcs.shape
        S = torch.stft(srcs.reshape(B * K, T), NFFT, HOP, window=self.win, return_complex=True).view(B, K, NFFT // 2 + 1, -1)
        h = self.inp(torch.log1p(S.abs() * 10))
        for b in self.blocks: h = h + b(h)
        w = torch.softmax(self.out(h) + self.logw0, dim=1)
        return torch.istft((w * S).sum(1), NFFT, HOP, window=self.win, length=T)
