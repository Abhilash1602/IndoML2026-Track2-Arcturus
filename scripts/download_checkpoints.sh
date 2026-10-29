#!/bin/bash
# Usage: bash scripts/download_checkpoints.sh <archive URL or local path to Arcturus_Track2_checkpoints.tar>
# Unpacks the 22 checkpoint files into ./checkpoints/ and verifies them against CHECKPOINTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
src=${1:?usage: bash scripts/download_checkpoints.sh <URL or path>}
if [[ "$src" =~ ^https?:// ]]; then
  tmp=$(mktemp -d); curl -L --fail -o "$tmp/ckpt.tar" "$src"; src="$tmp/ckpt.tar"
fi
tar -xf "$src"            # archive paths are checkpoints/...
python - <<'PY'
import hashlib, re
ok = True
for path, digest in re.findall(r"`(checkpoints/[^`]+)` \| [0-9.]+ \| `([0-9a-f]{64})`", open("CHECKPOINTS.md").read()):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(("ok  " if h == digest else "BAD ") + path); ok &= h == digest
raise SystemExit(0 if ok else "checksum mismatch")
PY
