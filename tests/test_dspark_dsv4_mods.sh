#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D2T_MOD="$PROJECT_DIR/mods/fix-dspark-dsv4-d2t/run.sh"
TOPK_MOD="$PROJECT_DIR/mods/add-dsv4-topk256/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# ---------------------------------------------------------------------------
# Fixture 1: vllm dspark module
# ---------------------------------------------------------------------------
VLLM_ROOT="$TMP_DIR/vllm"
D2T_TARGET="$VLLM_ROOT/models/deepseek_v4/nvidia/dspark.py"
mkdir -p "$(dirname "$D2T_TARGET")"

cat > "$D2T_TARGET" <<'EOF'
import torch
try:
    from vllm.model_executor.models.interfaces import HasCrossEncoding
except ImportError:
    HasCrossEncoding = object

class DSparkDeepseekV4ForCausalLM(HasCrossEncoding):
    has_own_lm_head = False

    def __init__(self, config, prefix):
        super().__init__()
        self.config = config
        self.model = None
        self.lm_head = None

    def forward(self, input_ids, positions):
        return None

class DSparkV4MoE(torch.nn.Module):
    has_own_lm_head = False
    pass
EOF

# ---------------------------------------------------------------------------
# Fixture 2: flashinfer sparse-MLA module
# ---------------------------------------------------------------------------
FI_ROOT="$TMP_DIR/flashinfer"
TOPK_TARGET="$FI_ROOT/mla/_sparse_mla_sm120.py"
mkdir -p "$(dirname "$TOPK_TARGET")"

cat > "$TOPK_TARGET" <<'EOF'
import torch

_DECODE_MAX_TOKENS = 64

_DECODE_DSV4_DISPATCH = frozenset({
    (8, 128),
    (8, 512),
    (8, 1024),
    (16, 128),
    (16, 512),
    (16, 1024),
    (32, 128),
    (32, 512),
    (32, 1024),
    (64, 128),
    (64, 512),
    (64, 1024),
    (128, 128),
    (128, 512),
    (128, 1024),
})

class _SparseMLASM120Module(torch.nn.Module):
    def sparse_mla_sm120_paged_attention(self, *args, **kwargs):
        return None

module = _SparseMLASM120Module()

def _paged_attention(num_tokens, num_heads, topk, d_qk, kv_pbs, model_type, extra_topk):
    if (num_heads, topk) in _DECODE_DSV4_DISPATCH and num_tokens <= _DECODE_MAX_TOKENS:
        return "decode"
    return module.sparse_mla_sm120_paged_attention(
        num_tokens, num_heads, topk, d_qk, kv_pbs, model_type, extra_topk
    )
EOF

# ===========================================================================
# fix-dspark-dsv4-d2t
# ===========================================================================

# Case 1: first application
out=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$D2T_MOD")
grep -Fq '[fix-dspark-dsv4-d2t] Added draft_id_to_target_id = None to DSparkDeepseekV4ForCausalLM' <<< "$out"
python3 -m py_compile "$D2T_TARGET"
count=$(grep -c '^[[:space:]]*draft_id_to_target_id = None$' "$D2T_TARGET")
test "$count" -eq 1
anchor_line=$(grep -n 'has_own_lm_head = False' "$D2T_TARGET" | head -1 | cut -d: -f1)
attr_line=$(grep -n '^[[:space:]]*draft_id_to_target_id = None$' "$D2T_TARGET" | head -1 | cut -d: -f1)
test "$attr_line" -gt "$anchor_line"

VLLM_PACKAGE_ROOT="$VLLM_ROOT" python3 - "$D2T_TARGET" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("fixture_dspark", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.DSparkDeepseekV4ForCausalLM.draft_id_to_target_id is None
assert not hasattr(module.DSparkV4MoE, "draft_id_to_target_id")
PY

# Case 2: idempotent re-run
before=$(sha256sum "$D2T_TARGET")
out=$(VLLM_PACKAGE_ROOT="$VLLM_ROOT" bash "$D2T_MOD")
after=$(sha256sum "$D2T_TARGET")
test "$before" = "$after"
grep -Fq 'already present' <<< "$out"

# Case 3: missing vllm install is a safe no-op
out=$(VLLM_PACKAGE_ROOT="$TMP_DIR/empty-vllm" bash "$D2T_MOD")
grep -Fq 'skipping' <<< "$out"

# Case 4: missing anchor must fail loudly and leave the file untouched
NO_ANCHOR="$TMP_DIR/no-anchor/vllm"
NO_ANCHOR_TARGET="$NO_ANCHOR/models/deepseek_v4/nvidia/dspark.py"
mkdir -p "$(dirname "$NO_ANCHOR_TARGET")"
cat > "$NO_ANCHOR_TARGET" <<'EOF'
class DSparkDeepseekV4ForCausalLM:
    def __init__(self, config):
        self.config = config
EOF
no_anchor_before=$(sha256sum "$NO_ANCHOR_TARGET")
if out=$(VLLM_PACKAGE_ROOT="$NO_ANCHOR" bash "$D2T_MOD" 2>&1); then
    echo "[FAIL] mod succeeded without the has_own_lm_head anchor" >&2
    exit 1
fi
grep -Fq 'anchor' <<< "$out"
no_anchor_after=$(sha256sum "$NO_ANCHOR_TARGET")
test "$no_anchor_before" = "$no_anchor_after"

# ===========================================================================
# add-dsv4-topk256
# ===========================================================================

# Case 5: first application
out=$(FLASHINFER_PACKAGE_ROOT="$FI_ROOT" bash "$TOPK_MOD")
grep -Fq '[add-dsv4-topk256] Added 5 (H, 256) entries to _DECODE_DSV4_DISPATCH' <<< "$out"
grep -Fq '[add-dsv4-topk256] Added decode fallthrough guard' <<< "$out"
python3 -m py_compile "$TOPK_TARGET"

for h in 8 16 32 64 128; do
    grep -Fq "($h, 256)," "$TOPK_TARGET"
    l128=$(grep -nF "($h, 128)," "$TOPK_TARGET" | cut -d: -f1)
    l256=$(grep -nF "($h, 256)," "$TOPK_TARGET" | cut -d: -f1)
    l512=$(grep -nF "($h, 512)," "$TOPK_TARGET" | cut -d: -f1)
    test "$l128" -lt "$l256" -a "$l256" -lt "$l512"
done

guard_if_line=$(grep -n 'if num_tokens <= _DECODE_MAX_TOKENS:' "$TOPK_TARGET" | head -1 | cut -d: -f1)
raise_line=$(grep -n 'raise ValueError(' "$TOPK_TARGET" | head -1 | cut -d: -f1)
call_line=$(grep -n 'module.sparse_mla_sm120_paged_attention(' "$TOPK_TARGET" | tail -1 | cut -d: -f1)
test "$guard_if_line" -lt "$call_line" -a "$raise_line" -lt "$call_line"
grep -Fq 'SM120 sparse-MLA has no decode kernel for this shape:' "$TOPK_TARGET"

# Case 6: runtime behavior (requires torch; skip when unavailable)
if python3 -c 'import torch' >/dev/null 2>&1; then
    FLASHINFER_PACKAGE_ROOT="$FI_ROOT" python3 - "$TOPK_TARGET" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("fixture_sparse_mla", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module._paged_attention(
    num_tokens=5, num_heads=32, topk=256, d_qk=512,
    kv_pbs=64, model_type="dsv4", extra_topk=0,
) == "decode"

try:
    module._paged_attention(
        num_tokens=5, num_heads=32, topk=999, d_qk=512,
        kv_pbs=64, model_type="dsv4", extra_topk=0,
    )
except ValueError as e:
    assert "no decode kernel for this shape" in str(e)
else:
    raise AssertionError("expected ValueError for unsupported topk=999")
PY
else
    echo "[SKIP] torch unavailable; skipping runtime behavior sub-checks"
fi

# Case 7: idempotent re-run
before=$(sha256sum "$TOPK_TARGET")
out=$(FLASHINFER_PACKAGE_ROOT="$FI_ROOT" bash "$TOPK_MOD")
after=$(sha256sum "$TOPK_TARGET")
test "$before" = "$after"
grep -Fq 'already present; nothing to do' <<< "$out"

# Case 8: missing flashinfer install is a safe no-op
out=$(FLASHINFER_PACKAGE_ROOT="$TMP_DIR/empty-flashinfer" bash "$TOPK_MOD")
grep -Fq 'skipping' <<< "$out"

# Case 9: dispatch already extended but guard missing -> guard-only patch
GUARD_ONLY="$TMP_DIR/guard-only/flashinfer"
GUARD_ONLY_TARGET="$GUARD_ONLY/mla/_sparse_mla_sm120.py"
mkdir -p "$(dirname "$GUARD_ONLY_TARGET")"
cat > "$GUARD_ONLY_TARGET" <<'EOF'
import torch

_DECODE_MAX_TOKENS = 64

_DECODE_DSV4_DISPATCH = frozenset({
    (8, 128),
    (8, 256),
    (8, 512),
    (8, 1024),
    (16, 128),
    (16, 256),
    (16, 512),
    (16, 1024),
    (32, 128),
    (32, 256),
    (32, 512),
    (32, 1024),
    (64, 128),
    (64, 256),
    (64, 512),
    (64, 1024),
    (128, 128),
    (128, 256),
    (128, 512),
    (128, 1024),
})

class _SparseMLASM120Module(torch.nn.Module):
    def sparse_mla_sm120_paged_attention(self, *args, **kwargs):
        return None

module = _SparseMLASM120Module()

def _paged_attention(num_tokens, num_heads, topk, d_qk, kv_pbs, model_type, extra_topk):
    if (num_heads, topk) in _DECODE_DSV4_DISPATCH and num_tokens <= _DECODE_MAX_TOKENS:
        return "decode"
    return module.sparse_mla_sm120_paged_attention(
        num_tokens, num_heads, topk, d_qk, kv_pbs, model_type, extra_topk
    )
EOF
out=$(FLASHINFER_PACKAGE_ROOT="$GUARD_ONLY" bash "$TOPK_MOD")
grep -Fq '[add-dsv4-topk256] Added decode fallthrough guard' <<< "$out"
if grep -Fq 'Added [0-9]* (H, 256) entries' <<< "$out"; then
    echo "[FAIL] guard-only fixture unexpectedly re-inserted dispatch entries" >&2
    exit 1
fi
python3 -m py_compile "$GUARD_ONLY_TARGET"
before=$(sha256sum "$GUARD_ONLY_TARGET")
out=$(FLASHINFER_PACKAGE_ROOT="$GUARD_ONLY" bash "$TOPK_MOD")
after=$(sha256sum "$GUARD_ONLY_TARGET")
test "$before" = "$after"

echo "[PASS] dspark dsv4 mods (fix-dspark-dsv4-d2t, add-dsv4-topk256) are targeted, valid, and idempotent"
