#!/bin/bash
# Hot-patch: add TOPK=256 decode-DSV4 dispatch entries for DeepSeek-V4 DSpark
# speculative decoding on SM120 (GB10 / DGX Spark, sm_121a).
#
# Why
# ---
# DeepSeek-V4-Flash DSpark draft decode uses (num_heads=32, topk=256, ...) —
# topk=256 is missing from FlashInfer's SM120 DSV4 decode dispatch tables.
# Without the fix, the call falls through to the paged-attention orchestrator
# which hard-asserts num_tokens > 64 and aborts the process at engine warmup.
#
# This mod applies the Python-side changes from FlashInfer PR #3817:
#   1. Extend `_DECODE_DSV4_DISPATCH` frozenset with (H, 256) for H in {8, 16,
#      32, 64, 128}.
#   2. Add a guard in `_paged_attention` that raises a clear `ValueError` when
#      a decode-sized input (num_tokens <= _DECODE_MAX_TOKENS) doesn't match
#      any decode kernel — replacing the silent C++ abort.
#
# The matching C++ template instantiations in
# `csrc/sparse_mla_sm120_decode_dsv4.cu` (DSV4_DISPATCH(H, 256)) require a
# flashinfer wheel rebuild. Until then, this mod ensures the Python dispatch
# accepts topk=256 and that any unsupported shape fails loudly with an
# actionable error message instead of a hard C++ abort.
#
# Reference: https://github.com/flashinfer-ai/flashinfer/pull/3817
# Build-time: --apply-flashinfer-pr 3817 (rebinds flashinfer with the C++
#            template instantiations needed for topk=256 to actually run).
#
# Idempotent: re-running is a no-op when both Python-side changes are present.

set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$MOD_DIR/add_dsv4_topk256.py"