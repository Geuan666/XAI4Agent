#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import importlib.util
from langchain_core.messages import HumanMessage


DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/xai/dataset/mbpp/code/agentic_projects"
DEFAULT_OUTPUT_FILE = "/root/autodl-tmp/xai/dataset/mbpp/code/real_output.json"
DEFAULT_LOG_FILE = "/root/autodl-tmp/xai/dataset/mbpp/code/real_log.tsv"

USER_TEMPLATE = (
    "You are a coding agent.\n"
    "Task: complete the function body in {project_dir}/main.py using only the signature and docstring.\n"
    "Rules:\n"
    "- Use only the Python standard library.\n"
    "- Keep the existing function signature and docstring unchanged.\n"
    "- The function body must be correctly indented with 4 spaces.\n"
    "- Do NOT output code directly in the chat response.\n"
    "- Only modify files inside {project_dir}.\n"
    "Tool-use requirements (must follow):\n"
    "1) Use read_file to read {project_dir}/main.py.\n"
    "2) Build the full updated file content with the completed function body.\n"
    "3) Use write_file to overwrite {project_dir}/main.py with the full updated content.\n"
    "4) Use run to execute {project_dir}/main.py (run only accepts python files).\n"
    "5) If there is a runtime error, fix the code by repeating steps 2-4.\n"
    "Final requirement: you MUST write the final code back to {project_dir}/main.py using write_file before you finish.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic coding run (write/run) for MBPP.")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--log", default=DEFAULT_LOG_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    return parser.parse_args()


def list_projects(project_root: Path) -> list[Path]:
    if not project_root.exists():
        return []
    projects = [p for p in project_root.iterdir() if p.is_dir() and p.name.startswith("humaneval_")]
    projects.sort(key=lambda p: int(p.name.split("_")[1]))
    return projects


def load_agent_module():
    agent_path = "/root/autodl-tmp/xai/exp/real/agent.py"
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


def run_task(project_dir: Path, max_steps: int, agent_module) -> dict:
    user_prompt = USER_TEMPLATE.format(project_dir=str(project_dir))
    tool_log: list[dict] = []
    agent = agent_module.build_agent(tool_log=tool_log, base_dir=str(project_dir))
    model = os.getenv("QWEN_MODEL") or os.getenv("MODEL", "qwen3-coder-30b")
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

    return {
        "prompt": user_prompt,
        "prompt_chars": len(user_prompt),
        "final_response": final_content,
        "completion_chars": len(final_content),
        "duration_s": duration,
        "tool_calls": tool_log,
        "tool_calls_count": len(tool_log),
        "message_trace": serialize_messages(list(state["messages"])),
        "model": model,
        "base_url": base_url,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ.setdefault("MAX_TOKENS", "65536")
    if "BASE_URL" in os.environ and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["BASE_URL"]
    if "MODEL" in os.environ and "QWEN_MODEL" not in os.environ:
        os.environ["QWEN_MODEL"] = os.environ["MODEL"]

    project_root = Path(args.project_root)
    projects = list_projects(project_root)
    if not projects:
        raise SystemExit(f"No projects found under {project_root}")

    output_path = Path(args.output)
    log_path = Path(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            result = run_task(project_dir, args.max_steps, agent_module)
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

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        tmp_path.replace(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
