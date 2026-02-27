#!/usr/bin/env python3
import ast
import re
import warnings
from collections import Counter
from typing import Iterable, Optional, Sequence


_ASSERT_FUNC_RE = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _to_list(value) -> list:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def extract_function_name_from_tests(tests: Iterable[str]) -> Optional[str]:
    names: list[str] = []
    for test in tests:
        if not isinstance(test, str):
            continue
        match = _ASSERT_FUNC_RE.search(test)
        if match:
            names.append(match.group(1))
    if not names:
        return None
    return Counter(names).most_common(1)[0][0]


def _slice_source_by_lines(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines()
    start = max(start_line - 1, 0)
    end = min(end_line, len(lines))
    return "\n".join(lines[start:end])


def extract_function_from_code(code: str, target_name: Optional[str] = None) -> str:
    if not isinstance(code, str) or not code.strip():
        return ""

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return heuristic_extract_function(code)

    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_nodes:
        return heuristic_extract_function(code)

    chosen = None
    if target_name:
        for node in func_nodes:
            if node.name == target_name:
                chosen = node
                break
    if chosen is None:
        chosen = func_nodes[0]

    decorator_lines = [d.lineno for d in getattr(chosen, "decorator_list", []) if hasattr(d, "lineno")]
    start_line = min([chosen.lineno] + decorator_lines) if decorator_lines else chosen.lineno
    end_line = getattr(chosen, "end_lineno", None)
    if end_line is None:
        return heuristic_extract_function(code)
    return _slice_source_by_lines(code, start_line, end_line)


def heuristic_extract_function(code: str) -> str:
    lines = code.splitlines()
    def_idx = None
    def_indent = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            def_idx = i
            def_indent = len(line) - len(stripped)
            break
    if def_idx is None:
        return ""

    collected = []
    for j in range(def_idx, len(lines)):
        line = lines[j]
        if j > def_idx:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped and indent <= def_indent:
                break
        collected.append(line)
    return "\n".join(collected)


def extract_signature_block(func_code: str) -> str:
    lines = func_code.splitlines()
    def_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            def_idx = i
            break
    if def_idx is None:
        return ""

    sig_lines: list[str] = []
    balance = 0
    ended = False
    for line in lines[def_idx:]:
        sig_lines.append(line.rstrip())
        balance += line.count("(") - line.count(")")
        balance += line.count("[") - line.count("]")
        balance += line.count("{") - line.count("}")
        if balance <= 0 and line.rstrip().endswith(":"):
            ended = True
            break
    if not ended:
        return lines[def_idx].rstrip()
    return "\n".join(sig_lines)


def build_docstring(text: str, example: Optional[str] = None) -> list[str]:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace('"""', '\\"""')
    lines: list[str] = []
    if cleaned:
        lines.extend(cleaned.splitlines())
    if example:
        example_line = example.strip()
        if example_line:
            if lines:
                lines.append("")
            lines.append("Example:")
            lines.append(example_line)
    return lines


def build_prompt(signature_block: str, prompt_text: str, example: Optional[str] = None) -> str:
    doc_lines = build_docstring(prompt_text, example)
    if doc_lines:
        doc_body = "\n    ".join(doc_lines)
        docstring = f'    """\n    {doc_body}\n    """'
    else:
        docstring = '    """\n    """'
    parts = [signature_block.rstrip(), docstring, "    pass"]
    return "\n".join(parts).rstrip() + "\n"


def get_first_test(test_list: Sequence) -> Optional[str]:
    tests = _to_list(test_list)
    if not tests:
        return None
    first = tests[0]
    if isinstance(first, str):
        return first
    return str(first)
