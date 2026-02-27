import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Annotated
from langchain_core.messages import HumanMessage
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

def _contains_write_redirection(command: str) -> bool:
    in_single = False
    in_double = False
    escape = False
    for ch in command:
        if escape:
            escape = False
            continue
        if ch == "\\" and not in_single:
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if not in_single and not in_double and ch == ">":
            return True
    return False

def _contains_command_substitution(command: str) -> bool:
    in_single = False
    in_double = False
    in_backticks = False
    escape = False
    i = 0
    while i < len(command):
        ch = command[i]
        nxt = command[i + 1] if i + 1 < len(command) else ""
        if escape:
            escape = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            escape = True
            i += 1
            continue
        if ch == "'" and not in_double and not in_backticks:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single and not in_backticks:
            in_double = not in_double
            i += 1
            continue
        if ch == "`" and not in_single:
            in_backticks = not in_backticks
            i += 1
            continue
        if not in_single and not in_backticks:
            if ch == "$" and nxt == "(":
                return True
            if ch == "<" and nxt == "(" and not in_double:
                return True
            if ch == ">" and nxt == "(" and not in_double:
                return True
        i += 1
    return False

def _split_shell_segments(command: str) -> list[str]:
    segments: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    escape = False
    i = 0
    while i < len(command):
        ch = command[i]
        if escape:
            buf.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            buf.append(ch)
            escape = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue
        if not in_single and not in_double:
            if command.startswith("&&", i) or command.startswith("||", i):
                segment = "".join(buf).strip()
                if segment:
                    segments.append(segment)
                buf = []
                i += 2
                continue
            if ch in (";", "|"):
                segment = "".join(buf).strip()
                if segment:
                    segments.append(segment)
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    segment = "".join(buf).strip()
    if segment:
        segments.append(segment)
    return segments

def _is_read_only_command(command: str) -> tuple[bool, str | None]:
    if not command.strip():
        return False, "error: empty command"
    if _contains_command_substitution(command):
        return False, "error: command substitution is not allowed"
    if _contains_write_redirection(command):
        return False, "error: write redirection is not allowed"

    # Allowlist-only to prevent file writes or state changes.
    read_only_roots = {
        "awk",
        "basename",
        "cat",
        "cd",
        "column",
        "cut",
        "df",
        "dirname",
        "du",
        "echo",
        "env",
        "find",
        "git",
        "grep",
        "head",
        "less",
        "ls",
        "more",
        "printenv",
        "printf",
        "ps",
        "pwd",
        "rg",
        "ripgrep",
        "sed",
        "sort",
        "stat",
        "tail",
        "tree",
        "uniq",
        "wc",
        "which",
        "where",
        "whoami",
    }
    blocked_find_flags = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
    blocked_find_prefixes = ("-fprint", "-fprintf")
    read_only_git_subcommands = {
        "blame",
        "branch",
        "cat-file",
        "diff",
        "grep",
        "log",
        "ls-files",
        "remote",
        "rev-parse",
        "show",
        "status",
        "describe",
    }
    blocked_git_remote_actions = {"add", "remove", "rename", "set-url", "prune", "update"}
    blocked_git_branch_flags = {"-d", "-D", "--delete", "--move", "-m"}

    for segment in _split_shell_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False, "error: failed to parse command"
        if not tokens:
            continue
        idx = 0
        while idx < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[idx]):
            idx += 1
        if idx >= len(tokens):
            continue
        root = tokens[idx].lower()
        args = tokens[idx + 1 :]
        if root not in read_only_roots:
            return False, f"error: command '{root}' is not allowed in read-only mode"
        if root == "sed":
            if any(arg.startswith("-i") or arg == "--in-place" for arg in args):
                return False, "error: sed -i is not allowed in read-only mode"
        if root == "find":
            for arg in args:
                if arg in blocked_find_flags or arg.startswith(blocked_find_prefixes):
                    return False, "error: find write actions are not allowed"
        if root == "git":
            subcommand = None
            for arg in args:
                if arg.startswith("-"):
                    continue
                subcommand = arg.lower()
                break
            if subcommand and subcommand not in read_only_git_subcommands:
                return False, f"error: git subcommand '{subcommand}' is not allowed"
            if subcommand == "remote":
                for arg in args:
                    if arg.lower() in blocked_git_remote_actions:
                        return False, "error: git remote write actions are not allowed"
            if subcommand == "branch":
                for arg in args:
                    if arg in blocked_git_branch_flags:
                        return False, "error: git branch delete/move is not allowed"

    return True, None

def _read_file_content(
    file_path: str,
    *,
    base_dir: str | None,
    offset: int | None,
    limit: int | None,
    max_lines: int = 2000,
) -> str:
    resolved = _safe_path(base_dir, file_path)
    if resolved is None:
        return "error: invalid or unauthorized path"
    if not resolved.exists():
        return f"error: file not found: {resolved}"
    if resolved.is_dir():
        return f"error: path is a directory: {resolved}"

    text = resolved.read_text()
    lines = text.splitlines()
    total = len(lines)
    start = max(offset or 0, 0)
    if limit is None:
        end = min(total, start + max_lines)
    else:
        if limit <= 0:
            return "error: limit must be positive"
        end = min(total, start + limit)

    if start > total:
        return f"error: offset {start} exceeds total lines {total}"

    chunk = lines[start:end]
    content = "\n".join(chunk)
    truncated = end < total
    if truncated:
        # Signal partial reads so the caller can request more via offset/limit.
        return (
            f"Showing lines {start + 1}-{end} of {total} total lines from {resolved}.\n"
            f"Use offset/limit to read more.\n\n---\n{content}"
        )
    return content

def _write_file_content(file_path: str, content: str, *, base_dir: str | None) -> str:
    resolved = _safe_path(base_dir, file_path)
    if resolved is None:
        return "error: invalid or unauthorized path"
    if resolved.exists() and resolved.is_dir():
        return f"error: path is a directory: {resolved}"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    lines = content.splitlines()
    return f"ok: wrote {len(lines)} lines to {resolved}"

@tool
def run_shell(command: str, cwd: str | None = None, timeout_s: int = 30) -> str:
    """Run a read-only shell command for inspection only (no file writes/redirection)."""
    if cwd and not os.path.isdir(cwd):
        return f"error: cwd not found: {cwd}"

    ok, reason = _is_read_only_command(command)
    if not ok:
        return reason or "error: command not allowed"

    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=cwd,
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
def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a text file by absolute path. If output says it is partial, call again with offset/limit."""
    return _read_file_content(file_path, base_dir=None, offset=offset, limit=limit)

@tool
def write_file(file_path: str, content: str) -> str:
    """Write full file text by absolute path (overwrites existing content)."""
    return _write_file_content(file_path, content, base_dir=None)

def make_run_shell_tool(
    tool_log: list[dict] | None = None,
    base_dir: str | None = None,
    truncate_limit: int = 4000,
):
    @tool("run_shell")
    def run_shell_tool(
        command: str,
        cwd: str | None = None,
        timeout_s: int = 30,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str:
        """Run a read-only shell command for inspection only (no file writes/redirection)."""
        safe_cwd = _safe_cwd(base_dir, cwd)
        if safe_cwd and not os.path.isdir(safe_cwd):
            output = f"error: cwd not found: {safe_cwd}"
            if tool_log is not None:
                tool_log.append(
                    {
                        "id": tool_call_id,
                        "command": command,
                        "cwd": safe_cwd,
                        "output": output,
                    }
                )
            return output

        ok, reason = _is_read_only_command(command)
        if not ok:
            output = reason or "error: command not allowed"
            if tool_log is not None:
                tool_log.append(
                    {
                        "id": tool_call_id,
                        "command": command,
                        "cwd": safe_cwd,
                        "output": output,
                    }
                )
            return output

        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=safe_cwd,
                env=_qwen_env(),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            output = f"error: timeout after {timeout_s}s"
            if tool_log is not None:
                tool_log.append(
                    {
                        "id": tool_call_id,
                        "command": command,
                        "cwd": safe_cwd,
                        "output": output,
                    }
                )
            return output

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        parts = []
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        parts.append(f"exit_code: {result.returncode}")
        output = "\n".join(parts)
        if tool_log is not None:
            tool_log.append(
                {
                    "id": tool_call_id,
                    "command": command,
                    "cwd": safe_cwd,
                    "output": _truncate(output, limit=truncate_limit),
                }
            )
        return output

    return run_shell_tool

def make_read_file_tool(
    tool_log: list[dict] | None = None,
    base_dir: str | None = None,
    truncate_limit: int = 4000,
):
    @tool("read_file")
    def read_file_tool(
        file_path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        """Read a text file by absolute path. If output says it is partial, call again with offset/limit."""
        output = _read_file_content(
            file_path, base_dir=base_dir, offset=offset, limit=limit
        )
        if tool_log is not None:
            tool_log.append(
                {
                    "id": "read_file",
                    "file_path": file_path,
                    "offset": offset,
                    "limit": limit,
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
        tools = [read_file]
    else:
        # Restrict tools to file IO for controlled agentic runs.
        tools = [
            make_read_file_tool(tool_log=tool_log, base_dir=base_dir),
        ]
    return create_react_agent(llm, tools=tools)

def main() -> None:
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = input("User: ").strip()
    if not user_text:
        print("No input.")
        return

    agent = build_agent()
    state = agent.invoke({"messages": [HumanMessage(content=user_text)]})
    print(state["messages"][-1].content)

if __name__ == "__main__":
    main()
