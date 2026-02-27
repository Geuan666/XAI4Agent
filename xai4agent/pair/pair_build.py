#!/usr/bin/env python3
"""Build paired prompts for assisted vs agentic HumanEval runs."""

import argparse
import json
import importlib.util
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.utils.function_calling import convert_to_openai_tool
from transformers import AutoTokenizer

DEFAULT_PARQUET = "/root/autodl-tmp/XAI4Agent/xai4agent/data/dataset/humaneval/data/test-00000-of-00001.parquet"
DEFAULT_ASSISTED = "/root/autodl-tmp/XAI4Agent/experiments/assisted/assisted_output.json"
DEFAULT_AGENTIC_PRIMARY = "/root/autodl-tmp/XAI4Agent/experiments/agentic/agentic_output1.json"
DEFAULT_AGENTIC_FALLBACK = "/root/autodl-tmp/XAI4Agent/experiments/agentic/agentic_output.json"
DEFAULT_MODEL = "/root/autodl-tmp/qwen3-8B"
DEFAULT_SERVER = "/root/autodl-tmp/XAI4Agent/xai4agent/serving/fastapi/qwen3coder/server1.py"
DEFAULT_AGENT = "/root/autodl-tmp/XAI4Agent/xai4agent/agentic/agent.py"
DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/XAI4Agent/xai4agent/agentic/projects"
DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/pair"
LEGACY_PARQUET = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"
LEGACY_ASSISTED = "/root/autodl-tmp/xai/exp/assisted/assisted_output.json"
LEGACY_AGENTIC_PRIMARY = "/root/autodl-tmp/xai/exp/agentic/agentic_output1.json"
LEGACY_AGENTIC_FALLBACK = "/root/autodl-tmp/xai/exp/agentic/agentic_output.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired prompts for assisted vs agentic.")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--assisted", default=DEFAULT_ASSISTED)
    parser.add_argument("--agentic", default=DEFAULT_AGENTIC_PRIMARY)
    parser.add_argument("--agentic-fallback", default=DEFAULT_AGENTIC_FALLBACK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for pair_prompts.json. If omitted, use <run-root>/<run-id>/pair_prompts.json.",
    )
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    run_id = args.run_id or make_timestamp_id()
    return Path(args.run_root) / run_id / "pair_prompts.json"


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    spec.loader.exec_module(module)
    return module


def build_read_file_tool(server, agent_module, base_dir: str | None = None) -> list:
    read_tool = agent_module.make_read_file_tool(tool_log=[], base_dir=base_dir)
    openai_tool = convert_to_openai_tool(read_tool)
    return [
        server.Tool(
            type="function",
            function=server.FunctionDefinition(**openai_tool["function"]),
        )
    ]


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


def trim_final_assistant(messages: list[dict]) -> list[dict]:
    trimmed = list(messages)
    while trimmed:
        last = trimmed[-1]
        if last.get("type") == "ai" and not last.get("tool_calls"):
            trimmed.pop()
            continue
        break
    return trimmed


def build_agentic_prompt(server, tools, message_trace: list[dict]) -> str:
    trimmed = trim_final_assistant(message_trace)
    formatted = []
    for msg in trimmed:
        mtype = msg.get("type")
        if mtype == "human":
            formatted.append(server.ChatMessage(role="user", content=msg.get("content")))
        elif mtype == "ai":
            formatted.append(
                server.ChatMessage(
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=convert_tool_calls(msg.get("tool_calls", [])),
                )
            )
        elif mtype == "tool":
            formatted.append(
                server.ChatMessage(
                    role="tool",
                    content=msg.get("content"),
                    tool_call_id=msg.get("tool_call_id"),
                )
            )
    return server.build_prompt(formatted, tools)


def build_assisted_prompt(server, user_prompt: str) -> str:
    messages = [server.ChatMessage(role="user", content=user_prompt)]
    return server.build_prompt(messages, tools=None)


def add_context_marker(text: str) -> str:
    marker = "#Function definition and docstring:\n"
    target = "Function definition and docstring:\n"
    if marker in text:
        return text
    if target in text:
        return text.replace(target, marker, 1)
    return text


def insert_tool_response_marker(prompt: str) -> str:
    marker = "#Function definition and docstring:\n"
    if marker in prompt:
        return prompt
    needle = "<tool_response>\n"
    idx = prompt.find(needle)
    if idx == -1:
        return prompt
    insert_at = idx + len(needle)
    return prompt[:insert_at] + marker + prompt[insert_at:]


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def resolve_existing_path(path: str, fallbacks: list[str]) -> Path:
    primary = Path(path)
    if primary.exists():
        return primary
    for candidate in fallbacks:
        p = Path(candidate)
        if p.exists():
            return p
    return primary


def main() -> int:
    args = parse_args()

    parquet_path = resolve_existing_path(args.parquet, [LEGACY_PARQUET])
    assisted_path = resolve_existing_path(args.assisted, [LEGACY_ASSISTED])
    agentic_primary_path = resolve_existing_path(args.agentic, [LEGACY_AGENTIC_PRIMARY])
    agentic_fallback_path = resolve_existing_path(args.agentic_fallback, [LEGACY_AGENTIC_FALLBACK])

    df = pd.read_parquet(parquet_path)
    assisted = load_json(assisted_path)
    agentic_primary = load_json(agentic_primary_path)
    agentic_fallback = load_json(agentic_fallback_path)

    server = load_module(args.server, "qserver")
    agent_module = load_module(args.agent, "agent_module")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    server.tokenizer = tokenizer

    output: dict[str, dict[str, str]] = {}
    project_root = Path(args.project_root)

    for idx, row in df.iterrows():
        task_id = row["task_id"]
        number = task_id.split("/")[1]
        task_key = f"humaneval_{number}"

        assisted_entry = assisted.get(task_key)
        if not isinstance(assisted_entry, dict) or not assisted_entry.get("prompt"):
            raise RuntimeError(f"Missing assisted prompt for {task_key}")
        assisted_user_prompt = assisted_entry["prompt"]
        assisted_prefix = build_assisted_prompt(server, assisted_user_prompt)
        assisted_prefix = add_context_marker(assisted_prefix)

        agentic_entry = agentic_primary.get(task_key)
        if not agentic_entry or not agentic_entry.get("message_trace"):
            agentic_entry = agentic_fallback.get(task_key)
        if not agentic_entry or not agentic_entry.get("message_trace"):
            raise RuntimeError(f"Missing agentic message_trace for {task_key}")
        message_trace = agentic_entry["message_trace"]

        project_dir = project_root / task_key
        tools = build_read_file_tool(server, agent_module, base_dir=str(project_dir))
        agentic_prefix = build_agentic_prompt(server, tools, message_trace)
        agentic_prefix = insert_tool_response_marker(agentic_prefix)

        continuation = build_continuation(row["canonical_solution"])

        output[task_key] = {
            "assisted": assisted_prefix + continuation,
            "agentic": agentic_prefix + continuation,
        }

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[pair_build] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
