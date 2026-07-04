#!/bin/bash
# Hot-patch: add `draft_id_to_target_id = None` to vLLM's DSparkDeepseekV4ForCausalLM.
#
# Without this attribute, DSparkSpeculator.load_draft_model crashes with
# AttributeError when running DeepSeek-V4-Flash with DSpark speculative decoding.
# vLLM DSparkSpeculator assumes the draft model exposes `model.draft_id_to_target_id`,
# but the DSpark-V4 draft is a full-vocab model (draft ids are target ids) and never
# defined the attribute.
#
# Reference: https://github.com/vllm-project/vllm/pull/47429 (fixes #47418)
#
# Idempotent: re-running is a no-op when the fix is already present.

set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$MOD_DIR/fix_dspark_d2t.py"