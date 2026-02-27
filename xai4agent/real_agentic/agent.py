import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...truncated..."


def _qwen_env() -> dict:
    env = os.environ.copy()
    qwen_env_path = os.getenv("QWEN_ENV_PATH", "/root/miniconda3/envs/qwen")
    qwen_bin = os.path.join(qwen_env_path, "bin")
    env["PATH"] = f"{qwen_bin}:{env.get('PATH', '')}"
    env["CONDA_PREFIX"] = qwen_env_path
    return env


def _safe_cwd(base_dir: str | None, requested: str | None) -> str | None:
    if not base_dir:
        return requested
    base_path = Path(base_dir).resolve()
    if not requested:
        return str(base_path)
    try:
        resolved = Path(requested).resolve()
    except Exception:
        return str(base_path)
    if str(resolved).startswith(str(base_path)):
        return str(resolved)
    return str(base_path)


def _safe_path(base_dir: str | None, file_path: str) -> Path | None:
    if not file_path:
        return None
    path = Path(file_path)
    if not path.is_absolute():
        if not base_dir:
            return None
        path = Path(base_dir) / path
    try:
        resolved = path.resolve()
    except Exception:
        return None
    if base_dir:
        base_path = Path(base_dir).resolve()
        if not str(resolved).startswith(str(base_path)):
            return None
    return resolved


def _read_file_content(
    file_path: str,
    *,
    base_dir: str | None,
) -> str:
    resolved = _safe_path(base_dir, file_path)
    if resolved is None:
        return "error: invalid or unauthorized path"
    if not resolved.exists():
        return f"error: file not found: {resolved}"
    if resolved.is_dir():
        return f"error: path is a directory: {resolved}"
    return resolved.read_text()


def _decode_over_escaped(content: str) -> str:
    if not content:
        return content
    if "\n" in content or "\r" in content:
        return content
    updated = content
    if any(token in updated for token in ("\\n", "\\t", "\\r", "\\\\", "\\\"", "\\'")):
        updated = (
            updated.replace("\\r", "\r")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\\"", "\"")
            .replace("\\'", "'")
            .replace("\\\\", "\\")
        )
    if any(token in updated for token in ("/n", "/t", "/r", "/w")):
        updated = (
            updated.replace("/r", "\r")
            .replace("/n", "\n")
            .replace("/t", "\t")
            .replace("/w", " ")
        )
    return updated


def _write_file_content(file_path: str, content: str, *, base_dir: str | None) -> str:
    resolved = _safe_path(base_dir, file_path)
    if resolved is None:
        return "error: invalid or unauthorized path"
    if resolved.exists() and resolved.is_dir():
        return f"error: path is a directory: {resolved}"
    decoded = _decode_over_escaped(content)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(decoded)
    lines = decoded.splitlines()
    return f"ok: wrote {len(lines)} lines to {resolved}"


def _run_python_file(
    file_path: str,
    *,
    base_dir: str | None,
    cwd: str | None,
    timeout_s: int,
) -> str:
    resolved = _safe_path(base_dir, file_path)
    if resolved is None:
        return "error: invalid or unauthorized path"
    if not resolved.exists():
        return f"error: file not found: {resolved}"
    if resolved.suffix != ".py":
        return "error: only .py files can be run"

    safe_cwd = _safe_cwd(base_dir, cwd) if cwd is not None else str(resolved.parent)
    if safe_cwd and not os.path.isdir(safe_cwd):
        return f"error: cwd not found: {safe_cwd}"

    try:
        result = subprocess.run(
            [sys.executable, str(resolved)],
            cwd=safe_cwd,
            env=_qwen_env(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"error: timeout after {timeout_s}s"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    parts = []
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    parts.append(f"exit_code: {result.returncode}")
    return _truncate("\n".join(parts))


@tool
def read_file(file_path: str) -> str:
    """Read a text file by absolute path (full file)."""
    return _read_file_content(file_path, base_dir=None)


@tool
def write_file(file_path: str, content: str) -> str:
    """Write full file text by absolute path (overwrites existing content)."""
    return _write_file_content(file_path, content, base_dir=None)


@tool
def run(file_path: str, cwd: str | None = None, timeout_s: int = 30) -> str:
    """Run a python file only (no shell)."""
    return _run_python_file(file_path, base_dir=None, cwd=cwd, timeout_s=timeout_s)


def make_read_file_tool(
    tool_log: list[dict] | None = None,
    base_dir: str | None = None,
    truncate_limit: int = 4000,
):
    @tool("read_file")
    def read_file_tool(file_path: str) -> str:
        """Read a text file by absolute path (full file)."""
        output = _read_file_content(file_path, base_dir=base_dir)
        if tool_log is not None:
            tool_log.append(
                {
                    "id": "read_file",
                    "file_path": file_path,
                    "output": _truncate(output, limit=truncate_limit),
                }
            )
        return output

    return read_file_tool


def make_write_file_tool(
    tool_log: list[dict] | None = None,
    base_dir: str | None = None,
    truncate_limit: int = 4000,
):
    @tool("write_file")
    def write_file_tool(file_path: str, content: str) -> str:
        """Write full file text by absolute path (overwrites existing content)."""
        output = _write_file_content(file_path, content, base_dir=base_dir)
        if tool_log is not None:
            tool_log.append(
                {
                    "id": "write_file",
                    "file_path": file_path,
                    "content_chars": len(content),
                    "output": _truncate(output, limit=truncate_limit),
                }
            )
        return output

    return write_file_tool


def make_run_tool(
    tool_log: list[dict] | None = None,
    base_dir: str | None = None,
    truncate_limit: int = 4000,
):
    @tool("run")
    def run_tool(
        file_path: str,
        cwd: str | None = None,
        timeout_s: int = 30,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Run a python file only (no shell)."""
        output = _run_python_file(
            file_path, base_dir=base_dir, cwd=cwd, timeout_s=timeout_s
        )
        if tool_log is not None:
            tool_log.append(
                {
                    "id": tool_call_id,
                    "file_path": file_path,
                    "cwd": cwd,
                    "output": _truncate(output, limit=truncate_limit),
                }
            )
        return output

    return run_tool


def build_llm() -> ChatOpenAI:
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    max_tokens = int(os.getenv("MAX_TOKENS", "4096"))

    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if not base_url and dashscope_key:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = dashscope_key
    elif not base_url:
        base_url = "http://127.0.0.1:8000/v1"

    model = os.getenv("QWEN_MODEL", "qwen3-coder-30b")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
    )


def build_agent(tool_log: list[dict] | None = None, base_dir: str | None = None):
    llm = build_llm()
    if tool_log is None and base_dir is None:
        tools = [read_file, write_file, run]
    else:
        tools = [
            make_read_file_tool(tool_log=tool_log, base_dir=base_dir),
            make_write_file_tool(tool_log=tool_log, base_dir=base_dir),
            make_run_tool(tool_log=tool_log, base_dir=base_dir),
        ]
    return create_react_agent(llm, tools=tools)
