#!/usr/bin/env python3
"""Tokenize paired prompts and align assisted segments to agentic positions."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

DEFAULT_PAIR_PROMPTS = "/root/autodl-tmp/XAI4Agent/experiments/pair/pair_prompts.json"
DEFAULT_MODEL = "/root/autodl-tmp/qwen3-8B"
DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/pair"
LEGACY_PAIR_PROMPTS = "/root/autodl-tmp/xai/exp/pair/pair_prompts.json"

MARKER = "#Function definition and docstring:\n"
FALLBACK_MARKER = "Function definition and docstring:\n"
ASSISTANT_TAG = "<|im_start|>assistant"
CODE_FENCE_PATTERNS = ("```python", "```py", "```")
IM_END = "<|im_end|>"
SYSTEM_START = "<|im_start|>system"
USER_START = "<|im_start|>user"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
TOOLS_OPEN = "<tools>"
TOOLS_CLOSE = "</tools>"
IMPORTANT_OPEN = "<IMPORTANT>"
IMPORTANT_CLOSE = "</IMPORTANT>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tokenized pairs and alignment metadata.")
    parser.add_argument("--pairs", default=DEFAULT_PAIR_PROMPTS)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for pair_tokens.json. If omitted, derive from --pairs or <run-root>/<run-id>/pair_tokens.json.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def find_latest_run_file(run_root: str, filename: str) -> Path | None:
    root = Path(run_root)
    if not root.exists():
        return None
    candidates: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        f = d / filename
        if f.exists():
            candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_output_path(args: argparse.Namespace, pairs_path: Path) -> Path:
    if args.output:
        return Path(args.output)
    if pairs_path.name == "pair_prompts.json":
        return pairs_path.with_name("pair_tokens.json")
    run_id = args.run_id or make_timestamp_id()
    return Path(args.run_root) / run_id / "pair_tokens.json"


def resolve_pairs_path(args: argparse.Namespace) -> Path:
    if args.run_id:
        candidate = Path(args.run_root) / args.run_id / "pair_prompts.json"
        if candidate.exists():
            return candidate
    pair_path = Path(args.pairs)
    if pair_path.exists():
        return pair_path
    latest = find_latest_run_file(args.run_root, "pair_prompts.json")
    if latest is not None:
        return latest
    legacy_pair = Path(LEGACY_PAIR_PROMPTS)
    if legacy_pair.exists():
        return legacy_pair
    raise FileNotFoundError(f"pair prompts not found: {pair_path}")


def split_assisted_prompt(text: str) -> Dict[str, str] | None:
    marker_idx = text.find(MARKER)
    marker = MARKER
    if marker_idx == -1:
        marker_idx = text.find(FALLBACK_MARKER)
        marker = FALLBACK_MARKER
    if marker_idx == -1:
        return None

    im_end_idx = text.find(IM_END, marker_idx)
    if im_end_idx == -1:
        return None

    user_part = text[:marker_idx]
    func_part = text[marker_idx:im_end_idx]
    assistant_idx = text.find(ASSISTANT_TAG, im_end_idx)
    if assistant_idx == -1:
        return None
    canonical_part = text[assistant_idx:]

    # Ensure marker is preserved for function definition segment.
    if not func_part.startswith(marker):
        return None

    return {
        "user_prompt": user_part,
        "function_docstring": func_part,
        "canonical_solution": canonical_part,
    }


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Tokenizer must be fast to return offsets_mapping.")
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc.input_ids, enc.offset_mapping


def char_range_to_token_range(
    offsets: List[Tuple[int, int]], start_char: int, end_char: int
) -> Tuple[int, int]:
    if start_char < 0 or end_char < 0 or end_char <= start_char:
        return -1, -1
    start_token = -1
    end_token = -1
    for i, (s, e) in enumerate(offsets):
        if e <= start_char:
            continue
        start_token = i
        break
    if start_token == -1:
        return -1, -1
    for i, (s, e) in enumerate(offsets):
        if s >= end_char:
            end_token = i
            break
    if end_token == -1:
        end_token = len(offsets)
    return start_token, end_token


def find_marker_range(
    text: str,
    marker: str,
    offsets: List[Tuple[int, int]],
    start_char: int = 0,
) -> Tuple[int, int, int, int]:
    if start_char < 0:
        return -1, -1, -1, -1
    idx = text.find(marker, start_char)
    if idx == -1:
        return -1, -1, -1, -1
    char_start = idx
    char_end = idx + len(marker)
    tok_start, tok_end = char_range_to_token_range(offsets, char_start, char_end)
    return char_start, char_end, tok_start, tok_end


def find_marker_range_any(
    text: str,
    markers: List[str],
    offsets: List[Tuple[int, int]],
    start_char: int = 0,
) -> Tuple[int, int, int, int]:
    for marker in markers:
        cs, ce, ts, te = find_marker_range(text, marker, offsets, start_char)
        if ts != -1:
            return cs, ce, ts, te
    return -1, -1, -1, -1


def find_segment_range(
    text: str,
    segment: str,
    offsets: List[Tuple[int, int]],
    start_char: int = 0,
    end_char: int | None = None,
) -> Tuple[int, int, int, int]:
    if start_char < 0:
        return -1, -1, -1, -1
    if end_char is None:
        end_char = len(text)
    idx = text.find(segment, start_char, end_char)
    if idx == -1:
        return -1, -1, -1, -1
    char_start = idx
    char_end = idx + len(segment)
    tok_start, tok_end = char_range_to_token_range(offsets, char_start, char_end)
    return char_start, char_end, tok_start, tok_end


def find_code_block(text: str) -> Tuple[str, str, str]:
    for fence in CODE_FENCE_PATTERNS:
        start = text.find(fence)
        if start == -1:
            continue
        fence_start = text.find("\n", start)
        if fence_start == -1:
            continue
        end = text.find("```", fence_start + 1)
        if end == -1:
            continue
        prefix = text[:fence_start + 1]
        code = text[fence_start + 1 : end]
        suffix = text[end:]
        return prefix, code, suffix
    return "", text, ""


def build_aligned_ids(
    agentic_len: int,
    pad_token_id: int,
    segments: Dict[str, Dict[str, Any]],
) -> Tuple[List[int], List[int], bool, List[str], List[str]]:
    aligned_ids = [pad_token_id] * agentic_len
    aligned_mask = [0] * agentic_len
    errors: List[str] = []
    warnings: List[str] = []

    for name, info in segments.items():
        start = info.get("start", -1)
        end = info.get("end", -1)
        seg_ids = info.get("token_ids")
        if start is None or start < 0 or end is None or end <= start or not seg_ids:
            errors.append(f"{name}: missing segment")
            continue
        max_len = end - start
        if max_len <= 0 or start >= agentic_len:
            errors.append(f"{name}: out of range")
            continue
        fill_ids = seg_ids[:max_len]
        overlap = any(aligned_mask[start : start + len(fill_ids)])
        if overlap:
            errors.append(f"{name}: overlap")
            continue
        aligned_ids[start : start + len(fill_ids)] = fill_ids
        aligned_mask[start : start + len(fill_ids)] = [1] * len(fill_ids)
        if len(seg_ids) > max_len:
            warnings.append(f"{name}: truncated")

    return aligned_ids, aligned_mask, len(errors) == 0, errors, warnings


def main() -> int:
    args = parse_args()
    pair_path = resolve_pairs_path(args)
    data = json.loads(pair_path.read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id")

    token_output: Dict[str, Any] = {}

    for key, pair in data.items():
        assisted = pair.get("assisted", "")
        agentic = pair.get("agentic", "")
        segments = split_assisted_prompt(assisted)

        agentic_ids, agentic_offsets = tokenize_with_offsets(tokenizer, agentic)
        assisted_ids, _ = tokenize_with_offsets(tokenizer, assisted)

        segment_meta: Dict[str, Dict[str, Any]] = {}
        if segments is None:
            segment_meta["error"] = {"message": "marker or assistant tag not found"}
        else:
            tool_resp_open_char, tool_resp_open_char_end, _, _ = find_marker_range_any(
                agentic, [TOOL_RESPONSE_OPEN + "\n", TOOL_RESPONSE_OPEN], agentic_offsets
            )
            tool_resp_close_char, _, _, _ = find_marker_range_any(
                agentic, [TOOL_RESPONSE_CLOSE], agentic_offsets
            )
            for name, text in segments.items():
                seg_ids = tokenizer(text, add_special_tokens=False).input_ids
                if name == "function_docstring" and tool_resp_open_char != -1:
                    search_start = tool_resp_open_char_end
                    search_end = tool_resp_close_char if tool_resp_close_char != -1 else None
                else:
                    search_start = 0
                    search_end = None
                char_start, char_end, tok_start, tok_end = find_segment_range(
                    agentic, text, agentic_offsets, search_start, search_end
                )
                found = tok_start >= 0
                segment_meta[name] = {
                    "char_start": char_start,
                    "char_end": char_end,
                    "start": tok_start,
                    "end": tok_end,
                    "length": (tok_end - tok_start) if found else 0,
                    "found": found,
                    "token_ids": seg_ids,
                }

        canonical_positions: List[int] = []
        canonical_tokens: List[str] = []
        canonical_fence_start = -1
        canonical_fence_end = -1
        canonical_code_start = -1
        canonical_first_non_ws = -1
        canonical_info = segment_meta.get("canonical_solution")
        if canonical_info and canonical_info.get("found"):
            canonical_text = segments["canonical_solution"]
            prefix, code, _ = find_code_block(canonical_text)
            canonical_char_start = canonical_info.get("char_start", -1)

            fence_idx = canonical_text.find("```python")
            if fence_idx == -1:
                fence_idx = canonical_text.find("```")
            if fence_idx != -1 and canonical_char_start != -1:
                fence_line_end = canonical_text.find("\n", fence_idx)
                if fence_line_end != -1:
                    fence_char_start = canonical_char_start + fence_idx
                    fence_char_end = canonical_char_start + fence_line_end + 1
                    canonical_fence_start, canonical_fence_end = char_range_to_token_range(
                        agentic_offsets, fence_char_start, fence_char_end
                    )

            if canonical_char_start != -1:
                code_char_start = canonical_char_start + len(prefix)
                code_char_end = code_char_start + len(code)
                canonical_code_start, code_end = char_range_to_token_range(
                    agentic_offsets, code_char_start, code_char_end
                )
                if canonical_code_start != -1:
                    for tok_idx in range(canonical_code_start, code_end):
                        tok_id = agentic_ids[tok_idx]
                        tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
                        if tok_text.strip():
                            canonical_positions.append(tok_idx)
                            canonical_tokens.append(tok_text)
                    if canonical_positions:
                        canonical_first_non_ws = canonical_positions[0]

        system_char_start, _, system_start, system_start_end = find_marker_range_any(
            agentic, [SYSTEM_START + "\n", SYSTEM_START], agentic_offsets
        )
        system_char_end, _, system_end_start, system_end_end = find_marker_range(
            agentic, IM_END, agentic_offsets, system_char_start
        )
        system_content_start = system_start_end
        system_content_end = system_end_start

        user_char_start, _, user_start, user_start_end = find_marker_range_any(
            agentic, [USER_START + "\n", USER_START], agentic_offsets, system_char_end
        )
        _, _, user_end_start, user_end_end = find_marker_range(
            agentic, IM_END, agentic_offsets, user_char_start
        )

        _, _, assistant_start, assistant_end = find_marker_range_any(
            agentic, [ASSISTANT_TAG + "\n", ASSISTANT_TAG], agentic_offsets
        )

        _, _, tools_start, tools_end = find_marker_range_any(
            agentic, [TOOLS_OPEN], agentic_offsets
        )
        _, _, tools_close_start, tools_close_end = find_marker_range_any(
            agentic, [TOOLS_CLOSE], agentic_offsets
        )
        _, _, important_start, important_end = find_marker_range_any(
            agentic, [IMPORTANT_OPEN], agentic_offsets
        )
        _, _, important_close_start, important_close_end = find_marker_range_any(
            agentic, [IMPORTANT_CLOSE], agentic_offsets
        )

        _, _, tool_call_example_start, tool_call_example_end = find_marker_range(
            agentic, TOOL_CALL_OPEN, agentic_offsets, system_char_start
        )
        _, _, tool_call_example_close_start, tool_call_example_close_end = find_marker_range(
            agentic, TOOL_CALL_CLOSE, agentic_offsets, system_char_start
        )
        _, _, tool_call_actual_start, tool_call_actual_end = find_marker_range(
            agentic, TOOL_CALL_OPEN, agentic_offsets, system_char_end
        )
        _, _, tool_call_actual_close_start, tool_call_actual_close_end = find_marker_range(
            agentic, TOOL_CALL_CLOSE, agentic_offsets, system_char_end
        )

        _, _, tool_call_start, tool_call_end = find_marker_range_any(
            agentic, [TOOL_CALL_OPEN], agentic_offsets
        )
        _, _, tool_call_close_start, tool_call_close_end = find_marker_range_any(
            agentic, [TOOL_CALL_CLOSE], agentic_offsets
        )
        tool_response_char_start, tool_response_char_end, tool_response_start, tool_response_end = find_marker_range_any(
            agentic, [TOOL_RESPONSE_OPEN + "\n", TOOL_RESPONSE_OPEN], agentic_offsets
        )
        _, _, tool_response_close_start, tool_response_close_end = find_marker_range_any(
            agentic, [TOOL_RESPONSE_CLOSE], agentic_offsets
        )
        tool_response_content_start = tool_response_end if tool_response_end != -1 else -1

        aligned_ids, aligned_mask, aligned_ok, align_errors, align_warnings = build_aligned_ids(
            len(agentic_ids), pad_token_id, segment_meta
        )

        token_output[key] = {
            "agentic_ids": agentic_ids,
            "assisted_ids": assisted_ids,
            "assisted_aligned_ids": aligned_ids,
            "assisted_aligned_attention_mask": aligned_mask,
            "pad_token_id": pad_token_id,
            "agentic_length": len(agentic_ids),
            "assisted_length": len(assisted_ids),
            "segment_positions": {
                name: {
                    "start": info.get("start"),
                    "end": info.get("end"),
                    "length": info.get("length"),
                    "found": info.get("found"),
                }
                for name, info in segment_meta.items()
                if name != "error"
            },
            "tool_positions": {
                "tool_call_first_open": {"start": tool_call_start, "end": tool_call_end},
                "tool_call_first_close": {"start": tool_call_close_start, "end": tool_call_close_end},
                "tool_call_example_open": {
                    "start": tool_call_example_start,
                    "end": tool_call_example_end,
                },
                "tool_call_example_close": {
                    "start": tool_call_example_close_start,
                    "end": tool_call_example_close_end,
                },
                "tool_call_actual_open": {
                    "start": tool_call_actual_start,
                    "end": tool_call_actual_end,
                },
                "tool_call_actual_close": {
                    "start": tool_call_actual_close_start,
                    "end": tool_call_actual_close_end,
                },
                "tool_response_open": {"start": tool_response_start, "end": tool_response_end},
                "tool_response_content_start": tool_response_content_start,
                "tool_response_close": {
                    "start": tool_response_close_start,
                    "end": tool_response_close_end,
                },
                "tools_block_open": {"start": tools_start, "end": tools_end},
                "tools_block_close": {"start": tools_close_start, "end": tools_close_end},
                "important_open": {"start": important_start, "end": important_end},
                "important_close": {"start": important_close_start, "end": important_close_end},
            },
            "message_positions": {
                "system_start": system_start,
                "system_start_end": system_start_end,
                "system_end": system_end_start,
                "system_end_end": system_end_end,
                "system_content_start": system_content_start,
                "system_content_end": system_content_end,
                "user_start": user_start,
                "user_start_end": user_start_end,
                "user_end": user_end_start,
                "user_end_end": user_end_end,
                "assistant_start": assistant_start,
                "assistant_start_end": assistant_end,
            },
            "canonical_positions": {
                "assistant_start": canonical_info["start"] if canonical_info else -1,
                "fence_start": canonical_fence_start,
                "fence_end": canonical_fence_end,
                "code_start": canonical_code_start,
                "first_non_ws": canonical_first_non_ws,
            },
            "canonical_non_ws_positions": canonical_positions,
            "canonical_non_ws_tokens": canonical_tokens,
            "aligned_ok": aligned_ok,
            "aligned_errors": align_errors,
            "aligned_warnings": align_warnings,
        }

    output_path = resolve_output_path(args, pair_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(token_output, ensure_ascii=False, indent=2))
    print(f"[token_build] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
