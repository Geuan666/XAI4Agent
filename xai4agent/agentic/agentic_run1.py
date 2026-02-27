#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import importlib.util
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent


DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/XAI4Agent/xai4agent/agentic/projects"
DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/agentic"
DEFAULT_HIDDEN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/agentic/hidden_states"
AGENT_MODULE_PATH = "/root/autodl-tmp/XAI4Agent/xai4agent/agentic/agent.py"

USER_TEMPLATE = (
    "You are a code completion assistant.\n"
    "Task: Complete the function body based only on the function definition and docstring.\n"
    "Rules:\n"
    "- Output only the function body (no signature, no docstring).\n"
    "- Preserve correct indentation (4 spaces).\n"
    "- Use only the Python standard library.\n"
    "- Do not add explanations or extra text.\n"
    "- Assume no extra context beyond what is shown.\n"
    "Tool-use requirements (must follow):\n"
    "- Read {project_dir}/main.py using read_file.\n"
    "- Use the file content as context and output only the function body.\n"
    "- Only the read_file tool is available.\n"
    "- Use only this project directory: {project_dir}\n"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic coding run for HumanEval.")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--hidden-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=16)
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def resolve_runtime_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = Path(args.run_root) / (args.run_id or make_timestamp_id())
    output_path = Path(args.output) if args.output else run_dir / "agentic_output1.json"
    log_path = Path(args.log) if args.log else run_dir / "agentic_log1.tsv"
    hidden_dir = Path(args.hidden_dir) if args.hidden_dir else (Path(DEFAULT_HIDDEN_ROOT) / run_dir.name)
    return output_path, log_path, hidden_dir


def list_projects(project_root: Path) -> list[Path]:
    if not project_root.exists():
        return []
    projects = [p for p in project_root.iterdir() if p.is_dir() and p.name.startswith("humaneval_")]
    projects.sort(key=lambda p: int(p.name.split("_")[1]))
    return projects


def load_agent_module():
    agent_path = AGENT_MODULE_PATH
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load agent module")
    spec.loader.exec_module(module)
    return module


def serialize_messages(messages: list) -> list[dict]:
    serialized: list[dict] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            serialized.append(msg.model_dump())
        elif hasattr(msg, "dict"):
            serialized.append(msg.dict())  # type: ignore[attr-defined]
        else:
            serialized.append(
                {
                    "type": type(msg).__name__,
                    "content": getattr(msg, "content", None),
                }
            )
    return serialized


def run_task(project_dir: Path, max_steps: int, hidden_dir: Path, agent_module) -> dict:
    user_prompt = USER_TEMPLATE.format(project_dir=str(project_dir))
    tool_log: list[dict] = []
    tools = [agent_module.make_read_file_tool(tool_log=tool_log, base_dir=str(project_dir))]
    base_llm = agent_module.build_llm()
    hidden_dir.mkdir(parents=True, exist_ok=True)
    hidden_tag_base = f"{project_dir.name}_agentic1"
    hidden_tag = hidden_tag_base
    hidden_path = hidden_dir / f"{hidden_tag}.pt"
    if hidden_path.exists():
        hidden_tag = f"{hidden_tag_base}_{int(time.time())}"
        hidden_path = hidden_dir / f"{hidden_tag}.pt"

    def select_model(state, _runtime):
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", [])
        request_hidden = bool(messages) and isinstance(messages[-1], ToolMessage)
        llm = base_llm.bind_tools(tools)
        if request_hidden:
            llm = llm.bind(
                extra_body={
                    "return_hidden_states": True,
                    "return_attentions": True,
                    "attention_mode": "last_token",
                    "hidden_tag": hidden_tag,
                }
            )
        return llm

    agent = create_react_agent(select_model, tools=tools)
    model = os.getenv("QWEN_MODEL") or os.getenv("MODEL", "qwen3-8b")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "http://127.0.0.1:8000/v1"
    )
    start = time.time()
    state = agent.invoke(
        {"messages": [HumanMessage(content=user_prompt)]},
        config={"recursion_limit": max_steps + 2},
    )
    duration = time.time() - start
    final_content = state["messages"][-1].content or ""
    hidden_saved = hidden_path.exists()

    return {
        "prompt": user_prompt,
        "prompt_chars": len(user_prompt),
        "final_response": final_content,
        "completion_chars": len(final_content),
        "duration_s": duration,
        "tool_calls": tool_log,
        "tool_calls_count": len(tool_log),
        # Persist the full message trace for later analysis.
        "message_trace": serialize_messages(list(state["messages"])),
        "hidden_tag": hidden_tag,
        "hidden_state_path": str(hidden_path),
        "hidden_saved": hidden_saved,
        "attention_mode": "last_token",
        "model": model,
        "base_url": base_url,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    if "BASE_URL" in os.environ and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["BASE_URL"]
    if "MODEL" in os.environ and "QWEN_MODEL" not in os.environ:
        os.environ["QWEN_MODEL"] = os.environ["MODEL"]

    project_root = Path(args.project_root)
    projects = list_projects(project_root)
    if not projects:
        raise SystemExit(f"No projects found under {project_root}")

    output_path, log_path, hidden_dir = resolve_runtime_paths(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_path.write_text("timestamp\tproject\tstatus\tduration_s\tprompt_chars\tcompletion_chars\ttool_calls\n")

    agent_module = load_agent_module()

    results: dict[str, dict] = {}
    if output_path.exists():
        try:
            results = json.loads(output_path.read_text())
        except Exception:
            results = {}
    start_idx = max(args.start, 0)
    end_idx = len(projects) if args.limit is None else min(len(projects), start_idx + args.limit)

    for idx in range(start_idx, end_idx):
        project_dir = projects[idx]
        project_name = project_dir.name

        status = "SUCCESS"
        try:
            result = run_task(project_dir, args.max_steps, hidden_dir, agent_module)
        except Exception as exc:
            status = "ERROR"
            result = {
                "error": str(exc),
                "duration_s": 0,
                "prompt_chars": 0,
                "completion_chars": 0,
                "tool_calls": [],
            }

        results[project_name] = result
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a") as f:
            f.write(
                f"{timestamp}\t{project_name}\t{status}\t{result.get('duration_s', 0):.2f}\t"
                f"{result.get('prompt_chars', 0)}\t{result.get('completion_chars', 0)}\t"
                f"{len(result.get('tool_calls', []))}\n"
            )

        # Write after each task to preserve partial progress on interruption.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        tmp_path.replace(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
