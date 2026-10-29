"""Extend the fusion cache (data/fusion/train.npz) with more members on the SAME mixtures -> data/fusion/train_v2.npz.
Usage (repository root): PYTHONPATH=train python train/cache_extend.py [spec.json]
"""
import numpy as np, torch
import ft_mgan as F, ft_mf2ss
import json, os, sys
_spec = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
NEW = [tuple(x) for x in _spec["new"]] if _spec else [("asr10", "mgan", "checkpoints/mgan_r4_asr10/last.pt"), ("asr5b", "mgan", "checkpoints/mgan_r4_asr5b/last.pt"),
       ("res2", "res", "checkpoints/mgan_res2/last.pt"), ("ss2", "mf2ss", "checkpoints/mf2ss_ft2/best.pt")]
SRC = _spec["src"] if _spec else "data/fusion/train.npz"; DST = _spec["dst"] if _spec else "data/fusion/train_v2.npz"
import ft_mf2
dev = "cuda"; d = np.load(SRC); X = d["x"].astype(np.float32); Y = d["y"]; members = list(d["members"])
add = np.zeros((len(X), len(NEW), X.shape[1]), np.float16)
for k, (name, kind, ck) in enumerate(NEW):
    sd = torch.load(ck, map_location="cpu") if kind != "tfg" else None
    if kind == "tfg":                 # fine-tuned TF-GridNet (run this script with the t2esp venv)
        import ft_tfg; m = ft_tfg.load_tfg(dev, ck).eval()
        run = lambda x: m(torch.from_numpy(x).unsqueeze(0).to(dev))[0].cpu().numpy()
    elif kind == "mf2":
        m = ft_mf2.load_model(dev); m.load_state_dict(sd["model"]); m.eval()
        run = lambda x: ft_mf2.enhance_one(m, torch.from_numpy(x).to(dev), dev)[0].cpu().numpy()
    elif kind == "mf2ss":
        m = ft_mf2ss.load_model(dev); m.load_state_dict(sd["model"]); m.eval(); run = ft_mf2ss.run_np(m, sd.get("spk", 0), dev)
    else:
        m = F.load_model(dev); m.load_state_dict(sd["model"]); m.eval()
        f = F.enhance_res if kind == "res" else (lambda mm, xx: F.enhance(mm, xx)[0])
        run = lambda x: f(m, torch.from_numpy(x).unsqueeze(0).to(dev))[0].cpu().numpy()
    with torch.no_grad():
        for i in range(len(X)):
            y = run(X[i])[:X.shape[1]]; ri, ro = np.sqrt((X[i] ** 2).mean()), np.sqrt((y ** 2).mean())
            add[i, k] = (y * (ri / ro) if ro > 1e-8 else y).astype(np.float16)
    print("member done", name, flush=True); del m; torch.cuda.empty_cache()
np.savez(DST, x=d["x"], c=d["c"], y=np.concatenate([Y, add], 1), members=np.array(members + [n for n, _, _ in NEW]))
print("saved", members + [n for n, _, _ in NEW], flush=True)
