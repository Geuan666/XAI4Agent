#!/usr/bin/env python3
"""Build agentic/assisted prompts + aligned tokens for Qwen3-30B-A3B.

Outputs:
  - pair_prompts.json
  - pair_tokens.json

Prompts are formatted with tokenizer.apply_chat_template and OpenAI tool schema.
Adds alignment/padding + position metadata (compatible with exp/pair/token_build.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Prefer vLLM tokenizer when HF fast tokenizer is unavailable
try:
    from vllm.tokenizers import get_tokenizer as vllm_get_tokenizer  # type: ignore
except Exception:  # pragma: no cover
    vllm_get_tokenizer = None

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None

DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-30b-a3b"
DEFAULT_PARQUET = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"
DEFAULT_AGENTIC = "/root/autodl-tmp/xai/model/qwen3-30b-a3b/agentic_output.json"
DEFAULT_ASSISTED = "/root/autodl-tmp/xai/exp/assisted/assisted_output.json"
DEFAULT_OUT_DIR = "/root/autodl-tmp/xai/model/qwen3-30b-a3b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file by absolute path. If output says it is partial, call again with offset/limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        },
    }
]

MARKER = "#Function definition and docstring:\n"
FALLBACK_MARKER = "Function definition and docstring:\n"
ASSISTANT_TAG = "<|im_start|>assistant"
IM_END = "<|im_end|>"
SYSTEM_START = "<|im_start|>system"
USER_START = "<|im_start|>user"
TOOLS_OPEN = "<tools>"
TOOLS_CLOSE = "</tools>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
IMPORTANT_OPEN = "<IMPORTANT>"
IMPORTANT_CLOSE = "</IMPORTANT>"
CODE_FENCE_PATTERNS = ("```python", "```py", "```")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompts/tokens for Qwen3-30B-A3B.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--agentic", default=DEFAULT_AGENTIC)
    parser.add_argument("--assisted", default=DEFAULT_ASSISTED)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def get_tokenizer(model_path: str):
    if vllm_get_tokenizer is not None:
        return vllm_get_tokenizer(model_path, trust_remote_code=True, tokenizer_mode="auto")
    if AutoTokenizer is None:
        raise RuntimeError("No tokenizer backend available")
    # Qwen3 fast tokenizer file can be incompatible; use slow tokenizer.
    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=False
    )


def encode(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return encoded.ids if hasattr(encoded, "ids") else encoded
    return tokenizer(text, add_special_tokens=False).input_ids


def decode_token(tokenizer, token_id: int) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode([token_id], skip_special_tokens=False)
    return tokenizer.decode([token_id], skip_special_tokens=False)


def apply_chat_template(tokenizer, messages: List[dict], tools: List[dict] | None) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def normalize_body(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\t", " " * 4)
    lines = cleaned.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0].startswith("def ") and lines[0].lstrip().startswith("def "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return ""

    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    min_indent = min(indents) if indents else 0
    if min_indent >= 4:
        shift = min_indent - 4
        normalized = [line[shift:] if line.strip() else "" for line in lines]
    else:
        shift = 4 - min_indent
        normalized = [
            (" " * shift + line.lstrip(" ")) if line.strip() else "" for line in lines
        ]
    return "\n".join(normalized)


def build_continuation(canonical_solution: str) -> str:
    body = normalize_body(canonical_solution)
    return f"```python\n{body}\n```"


def add_context_marker(text: str) -> str:
    if MARKER in text:
        return text
    if FALLBACK_MARKER in text:
        return text.replace(FALLBACK_MARKER, MARKER, 1)
    return text


def build_messages(message_trace: List[dict]) -> List[dict]:
    # Insert marker into the first tool response content (read_file output).
    formatted: List[dict] = []
    marker_inserted = False
    for msg in trim_final_assistant(message_trace):
        mtype = msg.get("type")
        if mtype == "human":
            formatted.append({"role": "user", "content": msg.get("content", "")})
        elif mtype == "ai":
            tool_calls = convert_tool_calls(msg.get("tool_calls", []))
            payload = {"role": "assistant", "content": msg.get("content", "")}
            if tool_calls:
                payload["tool_calls"] = tool_calls
            formatted.append(payload)
        elif mtype == "tool":
            content = msg.get("content", "")
            if content and not marker_inserted:
                content = MARKER + content
                marker_inserted = True
            formatted.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id"),
                }
            )
    return formatted


def convert_tool_calls(tool_calls: List[dict]) -> List[dict]:
    converted = []
    for call in tool_calls or []:
        name = call.get("name") or call.get("function", {}).get("name")
        args = call.get("args")
        if args is None:
            args = call.get("function", {}).get("arguments", {})
        converted.append(
            {
                "id": call.get("id") or f"tool-{id(call)}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return converted


def trim_final_assistant(messages: List[dict]) -> List[dict]:
    trimmed = list(messages)
    while trimmed:
        last = trimmed[-1]
        if last.get("type") == "ai" and not last.get("tool_calls"):
            trimmed.pop()
            continue
        break
    return trimmed


def split_assisted_prompt(text: str) -> Dict[str, str] | None:
    user_start_idx = text.find(USER_START)
    search_start = user_start_idx if user_start_idx != -1 else 0
    marker_idx = text.find(MARKER, search_start)
    marker = MARKER
    if marker_idx == -1:
        marker_idx = text.find(FALLBACK_MARKER, search_start)
        marker = FALLBACK_MARKER
    if marker_idx == -1:
        return None

    im_end_idx = text.find(IM_END, marker_idx)
    if im_end_idx == -1:
        return None

    user_part = text[search_start:marker_idx]
    func_part = text[marker_idx:im_end_idx]
    assistant_idx = text.find(ASSISTANT_TAG, im_end_idx)
    if assistant_idx == -1:
        canonical_part = text[im_end_idx + len(IM_END) :]
    else:
        canonical_part = text[assistant_idx:]

    if not func_part.startswith(marker):
        return None

    return {
        "user_prompt": user_part,
        "function_docstring": func_part,
        "canonical_solution": canonical_part,
    }


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    ids = encode(tokenizer, text)
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for tid in ids:
        t = decode_token(tokenizer, tid)
        start = cursor
        cursor += len(t)
        offsets.append((start, cursor))
    return ids, offsets


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
    text: str, marker: str, offsets: List[Tuple[int, int]], start_char: int = 0
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
        prefix = text[: fence_start + 1]
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    agentic = load_json(Path(args.agentic))
    assisted = load_json(Path(args.assisted))

    tokenizer = get_tokenizer(args.model_path)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id or eos_token_id")

    prompts_out: Dict[str, Dict[str, str]] = {}
    tokens_out: Dict[str, Dict[str, Any]] = {}

    rows = list(df.itertuples())
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    for row in rows:
        task_id = row.task_id
        number = task_id.split("/")[1]
        key = f"humaneval_{number}"

        assisted_entry = assisted.get(key) or {}
        agentic_entry = agentic.get(key) or {}

        assisted_prompt = assisted_entry.get("prompt", "")
        message_trace = agentic_entry.get("message_trace") or []

        if not assisted_prompt or not message_trace:
            continue

        assisted_messages = [{"role": "user", "content": assisted_prompt}]
        agentic_messages = build_messages(message_trace)

        assisted_prompt_text = apply_chat_template(tokenizer, assisted_messages, tools=None)
        assisted_prompt_text = add_context_marker(assisted_prompt_text)
        agentic_prompt_text = apply_chat_template(tokenizer, agentic_messages, tools=TOOLS)

        continuation = build_continuation(row.canonical_solution)
        assisted_full = assisted_prompt_text + continuation
        agentic_full = agentic_prompt_text + continuation

        prompts_out[key] = {"assisted": assisted_full, "agentic": agentic_full}

        # tokenization
        agentic_ids, agentic_offsets = tokenize_with_offsets(tokenizer, agentic_full)
        assisted_ids, _ = tokenize_with_offsets(tokenizer, assisted_full)

        segments = split_assisted_prompt(assisted_full)
        segment_meta: Dict[str, Dict[str, Any]] = {}
        if segments is None:
            segment_meta["error"] = {"message": "marker or assistant tag not found"}
        else:
            tool_resp_open_char, tool_resp_open_char_end, _, _ = find_marker_range_any(
                agentic_full, [TOOL_RESPONSE_OPEN + "\n", TOOL_RESPONSE_OPEN], agentic_offsets
            )
            tool_resp_close_char, _, _, _ = find_marker_range_any(
                agentic_full, [TOOL_RESPONSE_CLOSE], agentic_offsets
            )
            for name, text in segments.items():
                seg_ids = encode(tokenizer, text)
                if name == "function_docstring" and tool_resp_open_char != -1:
                    search_start = tool_resp_open_char_end
                    search_end = tool_resp_close_char if tool_resp_close_char != -1 else None
                else:
                    search_start = 0
                    search_end = None
                char_start, char_end, tok_start, tok_end = find_segment_range(
                    agentic_full, text, agentic_offsets, search_start, search_end
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

        # canonical positions
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
                        tok_text = decode_token(tokenizer, tok_id)
                        if tok_text.strip():
                            canonical_positions.append(tok_idx)
                            canonical_tokens.append(tok_text)
                    if canonical_positions:
                        canonical_first_non_ws = canonical_positions[0]

        # message positions
        system_char_start, _, system_start, system_start_end = find_marker_range_any(
            agentic_full, [SYSTEM_START + "\n", SYSTEM_START], agentic_offsets
        )
        system_char_end, _, system_end_start, system_end_end = find_marker_range(
            agentic_full, IM_END, agentic_offsets, system_char_start
        )
        system_content_start = system_start_end
        system_content_end = system_end_start

        user_char_start, _, user_start, user_start_end = find_marker_range_any(
            agentic_full, [USER_START + "\n", USER_START], agentic_offsets, system_char_end
        )
        _, _, user_end_start, user_end_end = find_marker_range(
            agentic_full, IM_END, agentic_offsets, user_char_start
        )

        _, _, assistant_start, assistant_end = find_marker_range_any(
            agentic_full, [ASSISTANT_TAG + "\n", ASSISTANT_TAG], agentic_offsets
        )

        # tool positions
        _, _, tools_start, tools_end = find_marker_range_any(
            agentic_full, [TOOLS_OPEN], agentic_offsets
        )
        _, _, tools_close_start, tools_close_end = find_marker_range_any(
            agentic_full, [TOOLS_CLOSE], agentic_offsets
        )
        _, _, important_start, important_end = find_marker_range_any(
            agentic_full, [IMPORTANT_OPEN], agentic_offsets
        )
        _, _, important_close_start, important_close_end = find_marker_range_any(
            agentic_full, [IMPORTANT_CLOSE], agentic_offsets
        )

        _, _, tool_call_example_start, tool_call_example_end = find_marker_range(
            agentic_full, TOOL_CALL_OPEN, agentic_offsets, system_char_start
        )
        _, _, tool_call_example_close_start, tool_call_example_close_end = find_marker_range(
            agentic_full, TOOL_CALL_CLOSE, agentic_offsets, system_char_start
        )
        _, _, tool_call_actual_start, tool_call_actual_end = find_marker_range(
            agentic_full, TOOL_CALL_OPEN, agentic_offsets, system_char_end
        )
        _, _, tool_call_actual_close_start, tool_call_actual_close_end = find_marker_range(
            agentic_full, TOOL_CALL_CLOSE, agentic_offsets, system_char_end
        )

        _, _, tool_call_start, tool_call_end = find_marker_range_any(
            agentic_full, [TOOL_CALL_OPEN], agentic_offsets
        )
        _, _, tool_call_close_start, tool_call_close_end = find_marker_range_any(
            agentic_full, [TOOL_CALL_CLOSE], agentic_offsets
        )
        tool_response_char_start, tool_response_char_end, tool_response_start, tool_response_end = find_marker_range_any(
            agentic_full, [TOOL_RESPONSE_OPEN + "\n", TOOL_RESPONSE_OPEN], agentic_offsets
        )
        _, _, tool_response_close_start, tool_response_close_end = find_marker_range_any(
            agentic_full, [TOOL_RESPONSE_CLOSE], agentic_offsets
        )
        tool_response_content_start = tool_response_end if tool_response_end != -1 else -1

        aligned_ids, aligned_mask, aligned_ok, align_errors, align_warnings = build_aligned_ids(
            len(agentic_ids), pad_token_id, segment_meta
        )

        tokens_out[key] = {
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
                "tool_call_example_open": {"start": tool_call_example_start, "end": tool_call_example_end},
                "tool_call_example_close": {"start": tool_call_example_close_start, "end": tool_call_example_close_end},
                "tool_call_actual_open": {"start": tool_call_actual_start, "end": tool_call_actual_end},
                "tool_call_actual_close": {"start": tool_call_actual_close_start, "end": tool_call_actual_close_end},
                "tool_response_open": {"start": tool_response_start, "end": tool_response_end},
                "tool_response_content_start": tool_response_content_start,
                "tool_response_close": {"start": tool_response_close_start, "end": tool_response_close_end},
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

    (out_dir / "pair_prompts.json").write_text(json.dumps(prompts_out, ensure_ascii=False, indent=2))
    (out_dir / "pair_tokens.json").write_text(json.dumps(tokens_out, ensure_ascii=False, indent=2))

    print(f"Saved prompts: {out_dir / 'pair_prompts.json'}")
    print(f"Saved tokens: {out_dir / 'pair_tokens.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
