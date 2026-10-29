# Team Arcturus: IndoML 2026 Datathon, Track 2 (Noise Event Removal)

Training and inference code for our Track 2 system. For every clip it removes the noise events (breathing, horns, animals, alarms, music, machines, crying babies) from Vaani field recordings while preserving the speech, and transcribes the result with the mandated ASR, ARTPARK-IISc/SraVaani-1.0.

**Final standing: 8th** (leaderboard name: **asymptoticchase**).

| Phase 2 test set (3,178 clips) | Value |
|---|---|
| **Combined score** | **16.6113** |
| SI-SDR, synthetic clips | 13.8237 dB |
| ΔWER, all clips | +2.7876 points |
| ΔWER, synthetic clips | +7.8507 points |
| ΔWER, natural clips | +0.4012 points |

Combined = SI-SDR (synthetic subset) + 100 · ΔWER (full set), where ΔWER = WER(noisy) − WER(enhanced).

- **Technical report** (3 pages, IEEE format): [`docs/Arcturus_track_2_8.pdf`](docs/Arcturus_track_2_8.pdf)
- **Checkpoints** (22 files, 1.28 GB): [`Arcturus_Track2_checkpoints.tar`](https://huggingface.co/asymptoticlearning/IndoML2026-Track2-Arcturus) on Hugging Face, installed and verified by `scripts/download_checkpoints.sh` (see [`checkpoints/README.md`](checkpoints/README.md))

## Method in one paragraph
Eighteen in-domain fine-tuned separators enhance every clip:
- 13 MossFormerGAN-SE-16K, including one that subtracts its noise estimate and two trained toward clean speech plus 0.2 × noise;
- 3 MossFormer2-SE-48K;
- 2 MossFormer2-SS-16K.

They are trained on mixtures that reproduce the organisers' synthetic recipe, with an artefact-weighted SI-SDR loss and a feature loss through the frozen SraVaani encoder. Four learned time-frequency fusion networks combine the separator outputs and the noisy input with per-bin softmax weights, and their outputs are averaged. Noise-event timestamps from our own Track 1 detector set the enhancement strength: the full fused output inside each detected event, and an equal mix of fused output and input elsewhere. Every clip goes through the same pipeline. Details are in the report.

## Repository structure
```
.
├── README.md
├── requirements_train.txt          # training environment
├── requirements_inference.txt      # inference environment
├── CHECKPOINTS.md                  # the 22 checkpoint files with roles, sizes and sha256
├── configs/model_config.yaml       # final inference configuration: separators, windows, fusions, gating
├── inference/
│   ├── infer.py                    # inference entry point (Phase 2 format)
│   └── t2release/                  # separator wrappers (windowed overlap-add) and the fusion network
├── events/test_events_phase2_v6.json  # Track 1 detector timestamps for the Phase 2 test clips
├── train/
│   ├── run_training.sh             # the full training chain, stage by stage
│   ├── config.yaml                 # hyperparameters of the shipped models
│   ├── ft_mgan.py, ft_mf2.py, ft_mf2ss.py   # separator fine-tuning (MossFormerGAN, MossFormer2-SE, MossFormer2-SS)
│   ├── asr_feat.py                 # SraVaani encoder-feature loss
│   ├── fusion.py                   # fusion network and its training
│   ├── cache_members.py, cache_extend.py, cache_more.py, init_fusion.py   # fusion training caches and warm start
│   ├── build_clean_bank.py, build_ext_noise.py, build_unseen_hard.py      # training and selection data
│   ├── configs/                    # fusion member lists and cache specifications
│   └── ft_data_lists.json, synth_recipe_rows.json, ext_noise_LICENSES.csv
├── checkpoints/README.md           # how to fetch and verify the checkpoints
├── scripts/
│   ├── run_inference.sh            # wrapper around inference/infer.py
│   └── download_checkpoints.sh     # download, unpack and verify the checkpoint archive
└── docs/Arcturus_track_2_8.pdf     # technical report
```

## Environment and hardware
| Item | Value |
|---|---|
| Python | 3.12 (development, training and inference) |
| PyTorch / CUDA | torch 2.13.0+cu130, torchaudio 2.11.0, CUDA 13.0; NVIDIA driver 610.43.02 |
| Training GPUs | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) and NVIDIA A100 80GB PCIe (80 GB); one GPU per run |
| Inference | one GPU, 4.7 GB peak GPU memory; about 14 s per 12 s clip, about 12 h for the 3,178 test clips on one RTX PRO 6000. Also runs on CPU (much slower) |
| CPU / RAM | 2 × AMD EPYC 7763 (256 threads), 1 TB RAM |
| OS | Ubuntu 24.04.4 LTS |
| Training time | 2–4 GPU-hours per separator fine-tune (3,000 to 8,000 steps, batch 2); under 1 GPU-hour per fusion network |
| System packages | `libsndfile1` (used by `soundfile`); `git`, `curl`, `tar` |

**Pretrained model dependencies:**

| Model | Version | Source | Role |
|---|---|---|---|
| MossFormerGAN_SE_16K | HF revision `dcb1b20e1e44` | https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K | initialisation of 13 separators |
| MossFormer2_SE_48K | HF revision `eff8c97925c8` | https://huggingface.co/alibabasglab/MossFormer2_SE_48K | initialisation of 3 separators |
| MossFormer2_SS_16K | HF revision `407cb030cd66` | https://huggingface.co/alibabasglab/MossFormer2_SS_16K | initialisation of 2 separators |
| ClearerVoice-Studio model code | `clearvoice` 0.1.2 | https://github.com/modelscope/ClearerVoice-Studio | separator architectures |
| SraVaani-1.0 (mandated ASR, frozen) | HF revision `f5dd5358325a` | https://huggingface.co/ARTPARK-IISc/SraVaani-1.0 | transcription; frozen encoder in the training loss |

The ClearerVoice weights are only needed for training. SraVaani is downloaded from Hugging Face on first use and is not redistributed. Licences are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Setup
```bash
git clone https://github.com/Abhilash1602/IndoML2026-Track2-Arcturus.git && cd IndoML2026-Track2-Arcturus
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements_inference.txt        # or requirements_train.txt for training
pip install clearvoice==0.1.2 --no-deps
bash scripts/download_checkpoints.sh https://huggingface.co/asymptoticlearning/IndoML2026-Track2-Arcturus/resolve/main/Arcturus_Track2_checkpoints.tar
```
`clearvoice` is installed without its dependencies because its pinned requirements conflict with torch 2.13; the packages it needs are in the requirements files.

## Inference
```bash
python inference/infer.py <input_audio_dir> <output_dir>
# or: bash scripts/run_inference.sh <input_audio_dir> <output_dir> [gpu_index]
```
- **Input:** a folder of `.wav` or `.flac` files. Any sample rate is accepted; audio is converted to 16 kHz mono.
- **Event timestamps** (used only for strength control):
  - read from `--events`, or from `<input_dir>/events.json` if present, otherwise from the bundled `events/test_events_phase2_v6.json`, which is our Track 1 detector's output for the Phase 2 test clips;
  - format: `{"<clip_id>": [{"onset": s, "offset": s}, ...]}`.
- **Output:**
  - `<output_dir>/<clip_id>.wav` for every input clip: 16 kHz mono, 16-bit PCM, same file name;
  - `<output_dir>/transcripts.jsonl` with one line per clip: `{"clip_id": "<file name without extension>", "text": "<SraVaani transcript>"}`.

  Zipping these files gives the Phase 2 submission format. `--no-asr` writes only the WAVs.
- **Configuration:** `configs/model_config.yaml` holds the settings of the submitted run.
- **Reproduction check:** rerunning from raw audio reproduced the submitted WAVs to 67–74 dB difference SNR on spot-checked clips, with identical transcripts.

## Training
Hyperparameters are in `train/config.yaml`. Run the chain one stage at a time from the repository root:
```bash
pip install -r requirements_train.txt && pip install clearvoice==0.1.2 --no-deps
bash train/run_training.sh pretrained   # ClearerVoice base weights -> checkpoints/<name>/
FSD50K=<root> MUSAN=<root> bash train/run_training.sh data   # clean-speech bank, external noise bank, selection sets
bash train/run_training.sh base         # in-domain fine-tunes (MossFormerGAN v1 -> v2, MossFormer2-SE)
bash train/run_training.sh asr          # SraVaani-feature-loss MossFormerGAN members
bash train/run_training.sh lowsnr       # members with a heavier low-SNR tail
bash train/run_training.sh residual     # noise-subtraction MossFormerGAN
bash train/run_training.sh ss           # MossFormer2-SS speech/noise separators
bash train/run_training.sh alldata      # all-pairs and 12 s-crop members
bash train/run_training.sh pnt          # partial-noise-target members
bash train/run_training.sh fusion       # separator-output caches and the four fusion networks
bash train/run_training.sh export       # copy the 22 final files to checkpoints/<name>.pt
```
- **Data:**
  - Place the organisers' Track 2 validation set in `data/validation/` (`syntheticNoiseAudio/`, `syntheticCleanRefAudio/`, `naturalNoisyAudio/`). Its 695 synthetic pairs are split 591 / 104 for training and checkpoint selection (`train/ft_data_lists.json`).
  - The Vaani Noise-Event dataset is loaded from Hugging Face (`ARTPARK-IISc/Vaani-Noise-Event-Dataset`) to cut clean-speech spans from its Gold clips. Set `VAANI_HF_CACHE` to choose the cache directory.
  - FSD50K and MUSAN are downloaded from their official pages (links in `THIRD_PARTY.md`).
- **Outputs:** each stage writes to `./checkpoints/`; `export` produces the file names listed in `CHECKPOINTS.md`.
- **Reproducibility:** GPU training is not bit-deterministic. A retrain gives statistically similar models, not byte-identical checkpoints; the submitted results use the checkpoints in `CHECKPOINTS.md`.

## Compliance
- One general pipeline processes every clip; there is no natural/synthetic routing.
- There is no test-time noise dictionary or correlation search, and no rule decides whether a clip is enhanced. Every output sample contains model output.
- No test audio or test labels were used for training.
- External data are FSD50K clips licensed CC0 or CC-BY and MUSAN `rfm` music (CC BY 3.0), listed per clip in `train/ext_noise_LICENSES.csv`.

## Licence
MIT ([`LICENSE`](LICENSE)). Pretrained models and datasets keep their own licences ([`THIRD_PARTY.md`](THIRD_PARTY.md)).

## Team and contact
**Team Arcturus** (Codabench leaderboard name: asymptoticchase)
- Abhilash Jacob Mathew (team leader), HCLTech: abhilashjacob16@gmail.com
- Aman Kumar, KIET GAZIABAD: aman.kumar.cse@kiet.edu
