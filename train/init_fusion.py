"""Warm-start fusion v6 (v4 members + 2 new) from fusion v4: new input channels and output logits zero -> starts at FU4.
Usage (repository root): FUSION_CFG=train/configs/fusion_v6.json PYTHONPATH=train python train/init_fusion.py <fusion_v4.pt> <out.pt>"""
import sys, torch
import fusion
old = torch.load(sys.argv[1], map_location="cpu")["model"]; net = fusion.Fusion(); new = net.state_dict()
for k, v in old.items():
    if k == "inp.weight": v = torch.cat([v[:, :16], torch.zeros(v.shape[0], 2, *v.shape[2:]), v[:, 16:]], 1)
    elif k in ("out.weight", "out.bias"): v = torch.cat([v[:16], torch.zeros(2, *v.shape[1:]), v[16:]], 0)
    elif k == "logw0": continue
    assert new[k].shape == v.shape, k; new[k] = v
net.load_state_dict(new); torch.save({"model": net.state_dict(), "step": 0}, sys.argv[2]); print("saved", sys.argv[2])
