#!/bin/bash
# Hot-patch: tolerate PreparedExperts without has_csf_blocks (instanttensor 0.1.x
# on the Nov fork). b12x preparation reads the attribute in 5 places; make each
# read fall back to False (no CSF blocks -> skip scale-expansion planning).
set -euo pipefail
TARGET="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/b12x.py"
if [ ! -f "$TARGET" ]; then echo "[fix-b12x-csf-attr] $TARGET missing; skipping"; exit 0; fi
if grep -q 'getattr(prepared, "has_csf_blocks", False)' "$TARGET"; then
  echo "[fix-b12x-csf-attr] already patched; skipping"
else
  sed -i 's/prepared\.has_csf_blocks/getattr(prepared, "has_csf_blocks", False)/g' "$TARGET"
  echo "[fix-b12x-csf-attr] patched has_csf_blocks reads with getattr fallback"
fi
find "$(dirname "$TARGET")/../.." -name "__pycache__" -path "*fused_moe*" -exec rm -rf {} + 2>/dev/null || true
exit 0
