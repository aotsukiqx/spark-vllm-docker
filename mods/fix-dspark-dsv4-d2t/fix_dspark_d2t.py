#!/usr/bin/env python3
"""
Apply vLLM PR #47429 hot-patch to add `draft_id_to_target_id = None` to
`DSparkDeepseekV4ForCausalLM`.

Background
----------
DSparkDeepseekV4ForCausalLM is a full-vocab draft model (draft ids are target
ids, no d2t remapping required). However, DSparkSpeculator.load_draft_model
unconditionally accesses `model.draft_id_to_target_id`, causing an
AttributeError when running DeepSeek-V4-Flash with DSpark speculative decoding.

Fix
---
Add `draft_id_to_target_id = None` as a class attribute to
DSparkDeepseekV4ForCausalLM. The speculator's `is not None` check then correctly
evaluates to False, skipping the d2t scatter logic that isn't needed for
full-vocab drafts. This mirrors the pattern used in Qwen3DSparkForCausalLM
(where the same attribute is set to None when draft_vocab == target_vocab).

Reference: https://github.com/vllm-project/vllm/pull/47429
Issue:      https://github.com/vllm-project/vllm/issues/47418

Idempotency
-----------
- No-op if the attribute is already present.
- No-op if the DSpark module is not installed (older vLLM without DSpark).
- Safe to re-run on every container start.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# VLLM_PACKAGE_ROOT overrides the vllm package location for offline tests;
# the default matches the runner image's dist-packages layout.
DSPARK_PATH = Path(
    os.environ.get("VLLM_PACKAGE_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm")
) / "models/deepseek_v4/nvidia/dspark.py"
CLASS_NAME = "DSparkDeepseekV4ForCausalLM"
ATTR_NAME = "draft_id_to_target_id"
ATTR_VALUE = "None"


def main() -> int:
    if not DSPARK_PATH.exists():
        print(
            f"[fix-dspark-dsv4-d2t] DSpark module not found at {DSPARK_PATH}; "
            "skipping (older vLLM without DSpark support)."
        )
        return 0

    text = DSPARK_PATH.read_text()
    lines = text.splitlines(keepends=True)

    # 1. Locate the target class.
    class_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith(f"class {CLASS_NAME}"):
            class_idx = i
            break
    if class_idx is None:
        print(
            f"[fix-dspark-dsv4-d2t] class {CLASS_NAME} not found in {DSPARK_PATH}; "
            "skipping (vLLM version does not ship DSparkDeepseekV4)."
        )
        return 0

    # 2. Already patched? Look for the attribute anywhere inside the class block.
    attr_marker = f"{ATTR_NAME} = {ATTR_VALUE}"
    for line in lines[class_idx:]:
        stripped = line.strip()
        if stripped == attr_marker or stripped.startswith(attr_marker + " "):
            print(
                f"[fix-dspark-dsv4-d2t] {ATTR_NAME} = {ATTR_VALUE} already "
                "present on DSparkDeepseekV4ForCausalLM; skipping."
            )
            return 0

    # 3. Anchor on `has_own_lm_head = False` — that's the line directly
    #    preceding the insertion point in the upstream PR diff.
    anchor = "has_own_lm_head = False"
    anchor_idx: int | None = None
    for i in range(class_idx, len(lines)):
        if lines[i].strip() == anchor:
            anchor_idx = i
            break
    if anchor_idx is None:
        print(
            f"[fix-dspark-dsv4-d2t] anchor {anchor!r} not found after class "
            f"{CLASS_NAME}; cannot patch. The vLLM DSpark module may have been "
            "refactored — please update this mod."
        )
        return 1

    # 4. Insert the two-line fix right after the anchor.
    insert_block = [
        "    # Full-vocab draft: draft ids are target ids, no remapping needed.\n",
        f"    {ATTR_NAME} = {ATTR_VALUE}\n",
    ]
    lines[anchor_idx + 1 : anchor_idx + 1] = insert_block
    DSPARK_PATH.write_text("".join(lines))
    print(
        f"[fix-dspark-dsv4-d2t] Added {ATTR_NAME} = {ATTR_VALUE} to "
        f"{CLASS_NAME} at {DSPARK_PATH}."
    )

    # 5. Clear any cached bytecode so the change takes effect.
    pycache = DSPARK_PATH.parent / "__pycache__"
    if pycache.exists():
        for cache_file in pycache.glob("dspark*.pyc"):
            cache_file.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())