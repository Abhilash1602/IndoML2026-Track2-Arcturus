# Third-party components

| Component | Version | Source | Licence | Use |
|---|---|---|---|---|
| ClearerVoice-Studio code (`clearvoice` package) | 0.1.2 | https://github.com/modelscope/ClearerVoice-Studio | Apache-2.0 | separator model code |
| MossFormerGAN_SE_16K weights | HF rev `dcb1b20` | https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K | Apache-2.0 | initialisation of 13 separators |
| MossFormer2_SE_48K weights | HF rev `eff8c97` | https://huggingface.co/alibabasglab/MossFormer2_SE_48K | Apache-2.0 | initialisation of 3 separators |
| MossFormer2_SS_16K weights | HF rev `407cb03` | https://huggingface.co/alibabasglab/MossFormer2_SS_16K | Apache-2.0 | initialisation of 2 separators |
| SraVaani-1.0 (mandated ASR) | HF rev `f5dd535` | https://huggingface.co/ARTPARK-IISc/SraVaani-1.0 | its own terms (not redistributed) | transcription; frozen encoder for the training loss |
| FSD50K (CC0 / CC-BY clips only) | Zenodo v1.0 | https://zenodo.org/records/4060432 | CC0 / CC-BY (per clip, `train/ext_noise_LICENSES.csv`) | external training noise |
| MUSAN music, `rfm` subset | OpenSLR 17 | https://www.openslr.org/17/ | CC BY 3.0 | external training noise |
| Vaani Noise-Event dataset | HF | https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset | CC-BY-4.0 | clean speech (Gold event-free spans) |

Our code is released under the MIT licence (`LICENSE`). Our fine-tuned separator checkpoints are derived from the Apache-2.0 ClearerVoice-Studio weights and are distributed under the same Apache-2.0 terms; the fusion networks are our own work (MIT).
