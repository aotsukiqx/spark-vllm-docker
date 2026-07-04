#!/usr/bin/env python3
"""
Apply FlashInfer PR #3817 (Python-side) to add TOPK=256 decode-DSV4 support
for DeepSeek-V4 DSpark speculative decoding on SM120 (GB10 / DGX Spark).

Background
-----------
Serving deepseek-ai/DeepSeek-V4-Flash-DSpark on SM120 (sm_121a) crashes at
engine warmup with:

    tvm.error.InternalError: Check failed: num_tokens > 64 (5 vs. 64)
    : Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2
    or sparse_mla_sm120_decode_dsv4; got num_tokens=5

The DSpark draft model decodes with (num_heads=32, topk=256, d_qk=512,
page_block_size=64) and num_tokens = num_speculative_tokens = 5. topk=256
is absent from both `_DECODE_DSV4_DISPATCH` (Python) and the `DSV4_DISPATCH`
instantiation switch (CUDA), which only cover TOPK in {128, 512, 1024}.

In `_paged_attention`, a num_tokens <= _DECODE_MAX_TOKENS shape that matches
no decode table falls through to `sparse_mla_sm120_paged_attention` (the
prefill orchestrator), which hard-asserts num_tokens > 64 and aborts.

This mod applies two Python-side changes:
  1. Extend `_DECODE_DSV4_DISPATCH` with `(H, 256)` for H in {8, 16, 32, 64,
     128}. Placement matches the upstream PR diff: each `(H, 256)` entry is
     inserted immediately after the existing `(H, 128)` entry so the table
     reads {(H, 128), (H, 256), (H, 512), (H, 1024)} per H.
  2. Add a guard in `_paged_attention` that raises a clear `ValueError` when
     a decode-sized input (num_tokens <= _DECODE_MAX_TOKENS) doesn't match
     any decode kernel — replacing the silent fallthrough to the paged
     orchestrator's C++ abort.

The corresponding C++ change (`DSV4_DISPATCH(H, 256)` template instantiations
in `csrc/sparse_mla_sm120_decode_dsv4.cu`) requires a flashinfer wheel rebuild
— apply it at build time with `--apply-flashinfer-pr 3817`. Until that
happens, this mod adds the Python-side dispatch entries so the call is at
least routed correctly (and any unsupported shape fails loudly with an
actionable error).

Reference: https://github.com/flashinfer-ai/flashinfer/pull/3817

Idempotency
-----------
- No-op if `(8, 256)` (or any other) entry is already in `_DECODE_DSV4_DISPATCH`.
- No-op if the decode fallthrough guard block is already present.
- No-op if flashinfer is not installed.
- Safe to re-run on every container start.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FLASHINFER_SPARSE_MLA = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py"
)
DISPATCH_TABLE_NAME = "_DECODE_DSV4_DISPATCH"
NUM_HEADS = (8, 16, 32, 64, 128)
TOPK_VALUE = 256

GUARD_ANCHOR = "module.sparse_mla_sm120_paged_attention("
GUARD_MARKER = "SM120 sparse-MLA has no decode kernel for this shape:"

GUARD_BLOCK_TEMPLATE = """\
        # Decode shapes (num_tokens <= _DECODE_MAX_TOKENS) must be served by one of
        # the standalone decode kernels above. The paged orchestrator only supports
        # prefill (num_tokens > _DECODE_MAX_TOKENS) and hard-asserts that in C++, so
        # a decode shape reaching here means its (num_heads, topk) is not in the
        # decode instantiation tables. Raise an actionable error instead of letting
        # the kernel abort the process.
        if num_tokens <= _DECODE_MAX_TOKENS:
            raise ValueError(
                "SM120 sparse-MLA has no decode kernel for this shape: "
                f"num_tokens={num_tokens}, num_heads={num_heads}, topk={topk}, "
                f"d_qk={d_qk}, page_block_size={kv_pbs}, model_type={model_type}, "
                f"extra_topk={extra_topk}. Supported decode shapes are enumerated in "
                "_DECODE_DSV4_DISPATCH / _DECODE_DSV3_2_DISPATCH; add the matching "
                "(num_heads, topk) instantiation to support it."
            )

"""


def extend_dispatch_table(text: str) -> tuple[str, int]:
    """Insert (H, 256) entries into _DECODE_DSV4_DISPATCH, returning updated text + count."""
    lines = text.splitlines(keepends=True)

    # Find the dispatch set's opening line.
    open_idx: int | None = None
    table_indent = ""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{DISPATCH_TABLE_NAME} = frozenset("):
            open_idx = i
            table_indent = line[: len(line) - len(stripped)]
            break
    if open_idx is None:
        print(
            f"[add-dsv4-topk256] {DISPATCH_TABLE_NAME} not found in "
            f"{FLASHINFER_SPARSE_MLA}; skipping dispatch extension."
        )
        return text, 0

    # Find the closing brace.
    close_idx: int | None = None
    for j in range(open_idx, len(lines)):
        if lines[j].strip() == "}":
            close_idx = j
            break
    if close_idx is None:
        print(
            f"[add-dsv4-topk256] closing brace of {DISPATCH_TABLE_NAME} not "
            "found; skipping dispatch extension."
        )
        return text, 0

    # Determine the entry indent by looking at the first (NUM_HEADS, TOPK)
    # tuple inside the table body. Entries are nested under
    # frozenset({...}), so the indent varies with formatting.
    entry_indent = ""
    entry_re = re.compile(r"^(\s+)\(\d+,\s*\d+\)\,\s*$")
    for k in range(open_idx + 1, close_idx):
        match = entry_re.match(lines[k])
        if match:
            entry_indent = match.group(1)
            break
    if not entry_indent:
        print(
            f"[add-dsv4-topk256] no (NUM_HEADS, TOPK) entry found inside "
            f"{DISPATCH_TABLE_NAME}; cannot determine entry indent."
        )
        return text, 0

    inserted = 0
    new_lines = list(lines)
    cursor = close_idx  # rebuild close_idx as we insert
    for h in NUM_HEADS:
        anchor = f"{entry_indent}({h}, 128),"
        anchor_idx: int | None = None
        for k in range(open_idx, cursor):
            if new_lines[k].rstrip("\n").rstrip() == anchor:
                anchor_idx = k
                break
        if anchor_idx is None:
            # (H, 128) is the upstream ordering baseline; if missing the table
            # has been refactored and we silently skip rather than mis-locating.
            print(
                f"[add-dsv4-topk256] anchor (H, 128) for H={h} not found in "
                f"{DISPATCH_TABLE_NAME}; skipping H={h}."
            )
            continue
        # Skip if (H, 256) is already present anywhere in the table.
        target = f"{entry_indent}({h}, {TOPK_VALUE}),"
        if any(
            line.rstrip("\n").rstrip() == target
            for line in new_lines[open_idx:cursor]
        ):
            continue
        new_lines.insert(anchor_idx + 1, target + "\n")
        cursor += 1
        inserted += 1

    if inserted:
        text = "".join(new_lines)
    return text, inserted


def add_decode_fallthrough_guard(text: str) -> tuple[str, bool]:
    """Insert the decode-fallthrough ValueError guard before the paged_attention call."""
    if GUARD_MARKER in text:
        return text, False

    lines = text.splitlines(keepends=True)
    anchor_idx: int | None = None
    anchor_indent = ""
    for i, line in enumerate(lines):
        # Skip matches inside multi-line strings or other call sites by checking
        # for the bare call to module.sparse_mla_sm120_paged_attention(...).
        if GUARD_ANCHOR in line and "module." in line:
            anchor_idx = i
            anchor_indent = line[: len(line) - len(line.lstrip())]
            break
    if anchor_idx is None:
        print(
            "[add-dsv4-topk256] Could not locate "
            "`module.sparse_mla_sm120_paged_attention(` call site; skipping guard."
        )
        return text, False

    # Re-indent the guard block to match the call site while preserving the
    # template's relative indentation. We detect the minimum leading whitespace
    # in the template (the "base indent") and then preserve each line's *extra*
    # indent beyond that base — so `if` at 8 spaces (base) and `raise` at 12
    # spaces (+4) keep their relative depth.
    template_lines = GUARD_BLOCK_TEMPLATE.splitlines(keepends=True)
    base_indent: int | None = None
    for raw_line in template_lines:
        if raw_line.strip():
            indent_len = len(raw_line) - len(raw_line.lstrip(" "))
            if base_indent is None or indent_len < base_indent:
                base_indent = indent_len
    if base_indent is None:
        base_indent = 0
    new_leading = " " * len(anchor_indent)
    reindented_lines = []
    for raw_line in template_lines:
        if raw_line.strip():
            line_indent = len(raw_line) - len(raw_line.lstrip(" "))
            extra = line_indent - base_indent
            reindented_lines.append(new_leading + " " * extra + raw_line.lstrip())
        else:
            reindented_lines.append(raw_line)
    block = "".join(reindented_lines)

    lines.insert(anchor_idx, block)
    return "".join(lines), True


def main() -> int:
    if not FLASHINFER_SPARSE_MLA.exists():
        print(
            f"[add-dsv4-topk256] flashinfer not found at {FLASHINFER_SPARSE_MLA}; "
            "skipping (older image without sm120 sparse-MLA)."
        )
        return 0

    text = FLASHINFER_SPARSE_MLA.read_text()

    text, dispatch_inserted = extend_dispatch_table(text)
    text, guard_inserted = add_decode_fallthrough_guard(text)

    if dispatch_inserted or guard_inserted:
        FLASHINFER_SPARSE_MLA.write_text(text)
        if dispatch_inserted:
            print(
                f"[add-dsv4-topk256] Added {dispatch_inserted} (H, {TOPK_VALUE}) "
                f"entries to {DISPATCH_TABLE_NAME}."
            )
        if guard_inserted:
            print(
                "[add-dsv4-topk256] Added decode fallthrough guard "
                "(SM120 sparse-MLA ValueError)."
            )
        # Clear bytecode cache so the change takes effect.
        pycache = FLASHINFER_SPARSE_MLA.parent / "__pycache__"
        if pycache.exists():
            for cache_file in pycache.glob("_sparse_mla_sm120*.pyc"):
                cache_file.unlink()
        print("[add-dsv4-topk256] Patched flashinfer/mla/_sparse_mla_sm120.py.")
    else:
        print(
            "[add-dsv4-topk256] Both dispatch extension and fallthrough guard "
            "already present; nothing to do."
        )

    # Remind the operator that the C++ side still requires a wheel rebuild.
    print(
        "[add-dsv4-topk256] NOTE: matching C++ template instantiations in "
        "csrc/sparse_mla_sm120_decode_dsv4.cu (DSV4_DISPATCH(H, 256)) require a "
        "flashinfer wheel rebuild. Use --apply-flashinfer-pr 3817 at build time "
        "to enable the actual topk=256 decode kernel; this mod only patches the "
        "Python-side dispatch tables and validation."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())