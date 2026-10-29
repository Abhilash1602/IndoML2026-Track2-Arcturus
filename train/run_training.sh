#!/bin/bash
# Full Track 2 training chain, one stage at a time:  bash train/run_training.sh <stage>
# Run from the repository root. Stages, in order:
#   pretrained  download the three ClearerVoice-Studio base checkpoints into checkpoints/<name>/
#   data        clean-speech bank (Gold event-free spans), external noise bank, unseen-noise and hard-SNR monitoring sets
#   base        in-domain fine-tunes from the pretrained weights: MossFormerGAN (v1 -> v2) and MossFormer2-SE
#   asr         MossFormerGAN members with the SraVaani encoder-feature loss (rounds 2-4)
#   lowsnr      MossFormerGAN members with a heavier low-SNR tail (round 5)
#   residual    noise-subtraction MossFormerGAN
#   ss          MossFormer2-SS speech/noise separators
#   alldata     all-pairs and 12 s-crop members
#   pnt         partial-noise-target members (target = clean + 0.2 * noise)
#   fusion      member-output caches and the four fusion networks used by the final system
#   export      copy the 22 final files to checkpoints/<name>.pt (names used by configs/model_config.yaml)
# Every stage writes to ./checkpoints/. GPU training is not bit-deterministic: a retrain gives statistically similar models.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=train:.
PY=${PY:-python}
stage=${1:?usage: bash train/run_training.sh <stage>}
CUR="--cosine --aug --ext-noise data/ext_noise --p-ext 0.5 --loss ab --alpha 2"          # recipe mixer + AB-SI-SDR
R2="--steps 4000 --eval-every 1000 $CUR --unseen data/eval_unseen/pairs.json"
R5="--steps 4000 --eval-every 1000 $CUR --unseen data/eval_unseen_hard/pairs.json --snr-shift-p 0.5 --snr-shift-max 5"
V2=checkpoints/mgan_ft_v2/final_best.pt
best() { cp "checkpoints/$1/best.pt" "checkpoints/$1/final_best.pt"; }   # members referenced as final_best.pt

case $stage in
pretrained)
  $PY - <<'PY'
from huggingface_hub import snapshot_download
for name, rev in [("MossFormerGAN_SE_16K", "dcb1b20e1e44d435090072ab9b223c792ea1f624"),
                  ("MossFormer2_SE_48K", "eff8c97925c8bec812af707814b3e5d777fd4503"),
                  ("MossFormer2_SS_16K", "407cb030cd66340918ebb6c8cc63b18f8592cdbe")]:
    snapshot_download(f"alibabasglab/{name}", revision=rev, local_dir=f"checkpoints/{name}")
PY
  ;;
data)
  # Organiser validation set expected in data/validation/{syntheticNoiseAudio,syntheticCleanRefAudio,naturalNoisyAudio}
  $PY train/build_clean_bank.py                                   # data/bank/clean (Gold event-free spans)
  $PY -I train/build_ext_noise.py "${FSD50K:?set FSD50K=<root>}" "${MUSAN:--}" "$(pwd)"   # data/ext_noise, data/eval_unseen
  $PY train/build_unseen_hard.py                                  # data/eval_unseen_hard
  ;;
base)
  $PY train/ft_mgan.py --out checkpoints/mgan_ft --aug --steps 4000
  $PY train/ft_mgan.py --out checkpoints/mgan_ft_v2 --init checkpoints/mgan_ft/best.pt --aug --cosine --steps 8000; best mgan_ft_v2
  $PY train/ft_mf2.py --out checkpoints/mf2_ft --aug --steps 4000; best mf2_ft
  ;;
asr)
  $PY train/ft_mgan.py $R2 --lr 2e-5 --asr-w 1.0 --init $V2 --out checkpoints/mgan_r2_asr1; best mgan_r2_asr1
  $PY train/ft_mgan.py $R2 --lr 2e-5 --asr-w 1.0 --seed 7 --init $V2 --out checkpoints/mgan_r2_asr1_s7; best mgan_r2_asr1_s7
  $PY train/ft_mgan.py $R2 --lr 1e-5 --asr-w 3.0 --init checkpoints/mgan_r2_asr1/final_best.pt --out checkpoints/mgan_r3_asr3c
  $PY train/ft_mgan.py $R2 --lr 2e-5 --asr-w 5.0 --seed 11 --init $V2 --out checkpoints/mgan_r3_asr5
  $PY train/ft_mgan.py $R2 --lr 2e-5 --asr-w 10.0 --seed 21 --init $V2 --out checkpoints/mgan_r4_asr10
  $PY train/ft_mgan.py $R2 --lr 2e-5 --asr-w 5.0 --seed 33 --init $V2 --out checkpoints/mgan_r4_asr5b
  ;;
lowsnr)
  $PY train/ft_mgan.py $R5 --lr 2e-5 --asr-w 3.0 --seed 51 --init $V2 --out checkpoints/mgan_r5a
  $PY train/ft_mgan.py $R5 --lr 1e-5 --asr-w 1.0 --seed 52 --init checkpoints/mgan_r2_asr1/final_best.pt --out checkpoints/mgan_r5b
  ;;
residual)
  # decoders predict the noise in the compressed complex domain; heads zero-initialised (identity at start)
  $PY train/ft_mgan.py $R2 --residual --lr 5e-5 --asr-w 1.0 --init $V2 --out checkpoints/mgan_res2
  ;;
ss)
  $PY train/ft_mf2ss.py --out checkpoints/mf2ss_ft --bs 1 --steps 4000 --eval-every 1000 --lr 1e-5 --asr-w 3.0 \
    --snr-shift-p 0.5 --snr-shift-max 5 --unseen data/eval_unseen_hard/pairs.json
  $PY train/ft_mf2ss.py --out checkpoints/mf2ss_ft2 --init checkpoints/mf2ss_ft/last.pt --bs 2 --steps 3000 --eval-every 1000 \
    --lr 1e-5 --asr-w 3.0 --snr-shift-p 0.5 --snr-shift-max 5 --unseen data/eval_unseen_hard/pairs.json   # best.pt = step 2000
  ;;
alldata)
  $PY train/ft_mgan.py --all --init $V2 --aug --cosine --lr 1e-5 --steps 4000 --out checkpoints/mgan_ft_all
  $PY train/ft_mf2.py --seg 12 --bs 2 --aug --init checkpoints/mf2_ft/final_best.pt --steps 2500 --out checkpoints/mf2_ft_long
  $PY train/ft_mf2.py --all --seg 12 --bs 2 --aug --lr 1e-5 --init checkpoints/mf2_ft_long/best.pt --steps 2000 --out checkpoints/mf2_ft_all
  ;;
pnt)
  $PY train/ft_mgan.py ${R5/--steps 4000/--steps 3000} --target-g 0.2 --lr 1e-5 --asr-w 1.0 --seed 61 \
    --init checkpoints/mgan_r2_asr1/final_best.pt --out checkpoints/mgan_pnt20
  $PY train/ft_mgan.py ${R5/--steps 4000/--steps 3000} --target-g 0.2 --lr 2e-5 --asr-w 1.0 --seed 62 --residual \
    --init checkpoints/mgan_res2/last.pt --out checkpoints/mgan_pntres20
  ;;
fusion)
  $PY train/cache_members.py 3000                                       # data/fusion/train.npz (6 members)
  $PY train/cache_extend.py                                            # + 4 members -> train_v2.npz (10)
  FUSION_CFG=train/configs/fusion_v2.json $PY train/fusion.py train --out checkpoints/fusion2 --steps 4000 --asr-w 1.0   # fu2 = best.pt
  FUSION_CFG=train/configs/fusion_v2.json $PY train/fusion.py train --out checkpoints/fusion3 --steps 5000 --asr-w 3.0   # fu3l = last.pt
  $PY train/cache_extend.py train/configs/cache_v4.json                 # + 6 members -> train_v4.npz (16)
  FUSION_CFG=train/configs/fusion_v4.json $PY train/fusion.py train --out checkpoints/fusion4 --steps 5000 --asr-w 1.0   # fu4 = best.pt
  $PY train/cache_more.py 3000                                          # + 3000 new mixtures -> train_v5.npz
  $PY train/cache_extend.py train/configs/cache_v6.json                 # + 2 partial-noise members -> train_v6.npz (18)
  FUSION_CFG=train/configs/fusion_v6.json $PY train/init_fusion.py checkpoints/fusion4/best.pt checkpoints/fusion6_init.pt
  FUSION_CFG=train/configs/fusion_v6.json $PY train/fusion.py train --out checkpoints/fusion6 --steps 6000 --lr 5e-4 --asr-w 1.0 \
    --init checkpoints/fusion6_init.pt                                       # fu6 = best.pt
  ;;
export)
  $PY - <<'PY'
import json, shutil, yaml
src = json.load(open("train/configs/members_v4.json"))
src.update({"pnt20": ["mgan", "checkpoints/mgan_pnt20/last.pt"], "pntres20": ["res", "checkpoints/mgan_pntres20/last.pt"]})
fus = {"fusion_v6.pt": "checkpoints/fusion6/best.pt", "fusion_v4.pt": "checkpoints/fusion4/best.pt",
       "fusion_v2.pt": "checkpoints/fusion2/best.pt", "fusion_v3l.pt": "checkpoints/fusion3/last.pt"}
cfg = yaml.safe_load(open("configs/model_config.yaml"))
for s in cfg["separators"]: shutil.copy(src[s["name"]][1], s["ckpt"]); print(s["ckpt"])
for f in cfg["fusions"]: shutil.copy(fus[f["ckpt"].split("/")[-1]], f["ckpt"]); print(f["ckpt"])
PY
  ;;
*) echo "unknown stage: $stage"; exit 1 ;;
esac
