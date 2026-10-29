"""Fusion cache v5: N NEW organiser-recipe mixtures (different seed) through all 16 v4 members, merged with train_v4.npz -> train_v5.npz.
Usage (repository root): PYTHONPATH=train python train/cache_more.py [N]
"""
import json, random, sys
import numpy as np, torch
import ft_mgan as F, ft_mf2, ft_mf2ss
import os as _os_here
_HERE = _os_here.path.dirname(_os_here.path.abspath(__file__))   # train/: data lists and configs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
spec = json.load(open(_os_here.path.join(_HERE, "configs/members_v4.json"))); members = json.load(open(_os_here.path.join(_HERE, "configs/fusion_v4.json")))["members"]
dev = "cuda"; random.seed(777); np.random.seed(777); torch.manual_seed(777)
lists = json.load(open(_os_here.path.join(_HERE, "ft_data_lists.json"))); fit_ids = {d["id"] for d in lists["fit"]}
snrs = [s for r in json.load(open(_os_here.path.join(_HERE, "synth_recipe_rows.json"))) if r["fn"][:-4] in fit_ids for s in r["snrs"]]
mx = F.Mixer(lists, snrs); mx.aug = True; mx.load_ext("data/ext_noise", 0.5); mx.snr_shift_p, mx.snr_shift_max = 0.3, 5.0
xs, cs = zip(*[mx.sample() for _ in range(N)]); X = np.stack(xs).astype(np.float32); C = np.stack(cs)
Y = np.zeros((N, len(members), X.shape[1]), np.float16)
for k, name in enumerate(members):
    kind, ck = spec[name]; sd = torch.load(ck, map_location="cpu")
    if kind == "mf2":
        m = ft_mf2.load_model(dev); m.load_state_dict(sd["model"]); m.eval(); run = lambda x: ft_mf2.enhance_one(m, torch.from_numpy(x).to(dev), dev)[0].cpu().numpy()
    elif kind == "mf2ss":
        m = ft_mf2ss.load_model(dev); m.load_state_dict(sd["model"]); m.eval(); run = ft_mf2ss.run_np(m, sd.get("spk", 0), dev)
    else:
        m = F.load_model(dev); m.load_state_dict(sd["model"]); m.eval()
        f = F.enhance_res if kind == "res" else (lambda mm, xx: F.enhance(mm, xx)[0])
        run = lambda x: f(m, torch.from_numpy(x).unsqueeze(0).to(dev))[0].cpu().numpy()
    with torch.no_grad():
        for i in range(N):
            y = run(X[i])[:X.shape[1]]; ri, ro = np.sqrt((X[i] ** 2).mean()), np.sqrt((y ** 2).mean())
            Y[i, k] = (y * (ri / ro) if ro > 1e-8 else y).astype(np.float16)
    print("member done", name, flush=True); del m; torch.cuda.empty_cache()
old = np.load("data/fusion/train_v4.npz"); assert list(old["members"]) == members
np.savez("data/fusion/train_v5.npz", x=np.concatenate([old["x"], X.astype(np.float16)]), c=np.concatenate([old["c"], C.astype(np.float16)]),
         y=np.concatenate([old["y"], Y]), members=np.array(members))
print("saved", len(old["x"]) + N, flush=True)
