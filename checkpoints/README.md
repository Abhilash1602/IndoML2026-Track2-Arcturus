# Checkpoints

The final system loads the 22 files listed in [`../CHECKPOINTS.md`](../CHECKPOINTS.md) (1.28 GB in total, with sha256 checksums):
18 fine-tuned separators and 4 fusion networks. They are too large for GitHub, so they are distributed as one archive:

- **Download:** [`Arcturus_Track2_checkpoints.tar`](https://huggingface.co/asymptoticlearning/IndoML2026-Track2-Arcturus) on Hugging Face (1.28 GB)
- **Install:** run from the repository root
  ```bash
  bash scripts/download_checkpoints.sh https://huggingface.co/asymptoticlearning/IndoML2026-Track2-Arcturus/resolve/main/Arcturus_Track2_checkpoints.tar
  # or, after downloading the file manually:
  bash scripts/download_checkpoints.sh /path/to/Arcturus_Track2_checkpoints.tar
  ```
  This unpacks `checkpoints/*.pt` here and verifies every checksum.

The mandated ASR `ARTPARK-IISc/SraVaani-1.0` is downloaded automatically from Hugging Face on first use. The ClearerVoice-Studio
base weights are needed only for training (`bash train/run_training.sh pretrained`).
