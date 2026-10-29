# Checkpoints used by the final Track 2 system

Loaded by `python inference/infer.py <input_dir> <output_dir>` (Phase 2: Combined 16.6113, 8th place). They are distributed as one archive
(`Arcturus_Track2_checkpoints.tar`); `bash scripts/download_checkpoints.sh <URL or path>` unpacks them here and verifies every checksum.
The mandated ASR `ARTPARK-IISc/SraVaani-1.0` is downloaded from Hugging Face at first use (not redistributed).

| name | role | file | size (MB) | sha256 |
|---|---|---|---|---|
| asr1_best | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/asr1_best.pt` | 12.9 | `d8489b199dd3782e1c926c67e60762acb34cb6d1453b77beb7516fa0bcaff135` |
| asr1s7 | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/asr1s7.pt` | 12.9 | `d995adb53fa062a033e4ba6a5ab1ecd0435cd40aab826f009d7416de49b56df5` |
| asr3c | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/asr3c.pt` | 12.9 | `e17d24b390ec45fe6f6e027bf45378355a939c1fc9477b53fc2931a30815325a` |
| asr5 | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/asr5.pt` | 12.9 | `2df1428262169e7c187b5980052403c6cab4b571f15dd4364345ef074f969af2` |
| mf2ft | separator: MossFormer2-SE-48K | `checkpoints/mf2ft.pt` | 221.5 | `60f1393c47a3d70c51b5dc2654a181ca85cab86b4017017c6c60feeddfe4b86b` |
| ss | separator: MossFormer2-SS-16K (speech output) | `checkpoints/ss.pt` | 223.4 | `f6694295780cd083506fbb4b582090ab19b3e5e6682886932a88b92609224d0d` |
| asr10 | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/asr10.pt` | 12.9 | `afe8f4eb97196d6bf61d720afaaa52a2e100b7647bd7decbaee6950cc0937f34` |
| asr5b | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/asr5b.pt` | 12.9 | `6c420fe57526644bfc0a030a1c61af2440c457cf0b03a59d68a05f261bc6beda` |
| res2 | separator: MossFormerGAN, noise-subtraction form | `checkpoints/res2.pt` | 12.9 | `0d2d3b0d7b2c2c3de6618760091a0682cb34bb13fec0e43c90f080ebe9ce2a77` |
| ss2 | separator: MossFormer2-SS-16K (speech output) | `checkpoints/ss2.pt` | 223.4 | `9bce09c96b7c8b9f542a515135bf8529f0867a86ef08331765008341a5ddd078` |
| r5a | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/r5a.pt` | 12.9 | `85c1385eb556e67784085171910da2fafe3c697a11fffe06c426c33166342df3` |
| r5b | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/r5b.pt` | 12.9 | `25fbadc02dd452c975994d7542666253690412423de08caee6d0c4107c0b004d` |
| v2 | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/v2.pt` | 12.9 | `5471c37c20d39d5694418d236cb2293767045cd30780e8d155c94a42c40a4a96` |
| mganall | separator: MossFormerGAN-SE-16K, clean target | `checkpoints/mganall.pt` | 12.9 | `65b27b0abd758ee21f5dfc52a0eadcd77637ad787ff2f856e004dd6b35d57539` |
| mf2all | separator: MossFormer2-SE-48K | `checkpoints/mf2all.pt` | 221.5 | `e71fdde6baa17540058672b465cb7cd549944fd8f61578d485658415d5d90c96` |
| mf2long | separator: MossFormer2-SE-48K | `checkpoints/mf2long.pt` | 221.5 | `10b619318a564c8ffb2df180a614da6f972710278a201001c64626f44b01a9ed` |
| pnt20 | separator: MossFormerGAN-SE-16K, clean target (partial-noise target) | `checkpoints/pnt20.pt` | 12.9 | `80e97f5ed77dfadb24d2944ab4704e33b49ebb50e518b46a9f36e232fa0c8e69` |
| pntres20 | separator: MossFormerGAN, noise-subtraction form (partial-noise target) | `checkpoints/pntres20.pt` | 12.9 | `ea4d6b5e9c3ab311d4b3422b3eaa95d01d01c88f7baf5565044aee301bf4c9ff` |
| fu6 | learned TF fusion over 18 separators (weight 0.4) | `checkpoints/fusion_v6.pt` | 1.5 | `3d5545b22a1ce7fe1dc0679c40b5b1c0166a2a50087ca4151234252e254db0ed` |
| fu4 | learned TF fusion over 16 separators (weight 0.2) | `checkpoints/fusion_v4.pt` | 1.5 | `bdfe813c1f83722eccbaedf2531f6e8025fc9caa9fd29ed0bbdf9544f17c8306` |
| fu2 | learned TF fusion over 10 separators (weight 0.2) | `checkpoints/fusion_v2.pt` | 1.5 | `579595713bc0f0ee2ac1837b61db287e274874e4335dffa769b9573c676fa2d9` |
| fu3l | learned TF fusion over 10 separators (weight 0.2) | `checkpoints/fusion_v3l.pt` | 1.5 | `53a5cba5fd8dd48447a5e9be1325f13b0317a955241ef44714519c656c43513f` |

Total: 22 files, 1.28 GB.
