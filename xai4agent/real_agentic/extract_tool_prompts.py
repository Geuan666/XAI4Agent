#!/usr/bin/env python3
import argparse
import json
import importlib.util
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from langchain_core.utils.function_calling import convert_to_openai_tool


DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/real"
DEFAULT_MODEL = "/root/autodl-tmp/qwen3-8B"
DEFAULT_AGENT = "/root/autodl-tmp/XAI4Agent/xai4agent/real_agentic/agent.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tool-call prompts from message_trace.")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_root) / (args.run_id or make_timestamp_id())
    input_path = Path(args.input) if args.input else run_dir / "real_output.json"
    output_path = Path(args.output) if args.output else run_dir / "real_tool_prompts.json"
    return input_path, output_path


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    spec.loader.exec_module(module)
    return module


def convert_tool_calls(tool_calls: list[dict]) -> list[dict]:
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
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
        )
    return converted


def build_tools(agent_module) -> list:
    read_tool = agent_module.make_read_file_tool(tool_log=[], base_dir=None)
    write_tool = agent_module.make_write_file_tool(tool_log=[], base_dir=None)
    run_tool = agent_module.make_run_tool(tool_log=[], base_dir=None)
    openai_tools = [
        convert_to_openai_tool(read_tool),
        convert_to_openai_tool(write_tool),
        convert_to_openai_tool(run_tool),
    ]
    return openai_tools


def build_prompt(tokenizer, tools, message_trace: list[dict]) -> str:
    messages = []
    for msg in message_trace:
        mtype = msg.get("type")
        if mtype == "human":
            messages.append({"role": "user", "content": msg.get("content")})
        elif mtype == "ai":
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": convert_tool_calls(msg.get("tool_calls", [])),
                }
            )
        elif mtype == "tool":
            messages.append(
                {
                    "role": "tool",
                    "content": msg.get("content"),
                    "tool_call_id": msg.get("tool_call_id"),
                }
            )
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        return_dict=False,
    )


def main() -> int:
    args = parse_args()
    input_path, output_path = resolve_paths(args)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    data: dict[str, Any] = json.loads(input_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    agent_module = load_module(args.agent, "agent_module")
    output_rows: list[dict[str, Any]] = []

    for sample_id, entry in data.items():
        message_trace = entry.get("message_trace")
        if not isinstance(message_trace, list) or not message_trace:
            continue

        tools = build_tools(agent_module)

        for idx, msg in enumerate(message_trace):
            if msg.get("type") != "ai":
                continue
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                continue
            prompt = build_prompt(tokenizer, tools, message_trace[:idx])
            for j, call in enumerate(tool_calls):
                tool_name = call.get("name") or call.get("function", {}).get("name")
                call_id = call.get("id") or f"{sample_id}_step{idx}_tool{j}"
                output_rows.append(
                    {
                        "id": call_id,
                        "prompt": prompt,
                        "sample_id": sample_id,
                        "tool": tool_name,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    previous = None
    if output_path.exists():
        try:
            previous = json.loads(output_path.read_text())
        except Exception:
            previous = None

    output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2))

    if previous is not None:
        prev_by_id = {row.get("id"): row for row in previous if isinstance(row, dict)}
        mismatched = 0
        for row in output_rows:
            pid = row.get("id")
            prev = prev_by_id.get(pid)
            if not prev or prev.get("prompt") != row.get("prompt"):
                mismatched += 1
        print(f"Saved {len(output_rows)} tool prompts to {output_path}")
        print(f"Compared with previous file: mismatched prompts = {mismatched}")
    else:
        print(f"Saved {len(output_rows)} tool prompts to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
