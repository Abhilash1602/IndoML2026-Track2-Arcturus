"""Fusion training data: N fixed 3 s organiser-recipe mixtures (fit pairs + clean bank, organiser + external noise, mild low-SNR tail)
and the outputs of every frozen ensemble member on them (one pass, same paths as the eval caches in data/ft_eval/<tag>).
Output: data/fusion/train.npz with x (N,T) mixture, c (N,T) clean, y (N,K,T) member outputs (float16), members (K names).
Usage (repository root): PYTHONPATH=train python train/cache_members.py [N]
"""
import json, random, sys
import numpy as np, torch
import ft_mgan as F, ft_mf2, ft_mf2ss

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
MEMBERS = [("asr1_best", "mgan", "checkpoints/mgan_r2_asr1/final_best.pt"),
           ("asr1s7", "mgan", "checkpoints/mgan_r2_asr1_s7/final_best.pt"),
           ("asr3c", "mgan", "checkpoints/mgan_r3_asr3c/last.pt"),
           ("asr5", "mgan", "checkpoints/mgan_r3_asr5/last.pt"),
           ("mf2ft", "mf2", "checkpoints/mf2_ft/final_best.pt"),
           ("ss", "mf2ss", "checkpoints/mf2ss_ft/last.pt")]
dev = "cuda"; random.seed(123); np.random.seed(123); torch.manual_seed(123)
lists = json.load(open(_os_here.path.join(_HERE, "ft_data_lists.json"))); fit_ids = {d["id"] for d in lists["fit"]}
snrs = [s for r in json.load(open(_os_here.path.join(_HERE, "synth_recipe_rows.json"))) if r["fn"][:-4] in fit_ids for s in r["snrs"]]
mx = F.Mixer(lists, snrs); mx.aug = True; mx.load_ext("data/ext_noise", 0.5); mx.snr_shift_p, mx.snr_shift_max = 0.3, 5.0
xs, cs = zip(*[mx.sample() for _ in range(N)]); X = np.stack(xs); C = np.stack(cs)
Y = np.zeros((N, len(MEMBERS), X.shape[1]), np.float16)
for k, (name, kind, ck) in enumerate(MEMBERS):
    if kind == "mgan":
        m = F.load_model(dev); m.load_state_dict(torch.load(ck, map_location="cpu")["model"]); m.eval()
        run = lambda x: F.enhance(m, torch.from_numpy(x).unsqueeze(0).to(dev))[0][0].cpu().numpy()
    elif kind == "mf2":
        m = ft_mf2.load_model(dev); m.load_state_dict(torch.load(ck, map_location="cpu")["model"]); m.eval()
        run = lambda x: ft_mf2.enhance_one(m, torch.from_numpy(x).to(dev), dev)[0].cpu().numpy()
    else:
        sd = torch.load(ck, map_location="cpu"); m = ft_mf2ss.load_model(dev); m.load_state_dict(sd["model"]); m.eval()
        run = ft_mf2ss.run_np(m, sd.get("spk", 0), dev)
    with torch.no_grad():
        for i in range(N):
            y = run(X[i])[:X.shape[1]]
            ri, ro = np.sqrt((X[i] ** 2).mean()), np.sqrt((y ** 2).mean())
            Y[i, k] = (y * (ri / ro) if ro > 1e-8 else y).astype(np.float16)     # RMS-matched, as in the deployed pipeline
    print("member done", name, flush=True)
    del m; torch.cuda.empty_cache()
import os; os.makedirs("data/fusion", exist_ok=True)
import os as _os_here
_HERE = _os_here.path.dirname(_os_here.path.abspath(__file__))   # train/: data lists and configs
np.savez("data/fusion/train.npz", x=X.astype(np.float16), c=C.astype(np.float16), y=Y, members=np.array([m[0] for m in MEMBERS]))
print("saved", X.shape, Y.shape, flush=True)
