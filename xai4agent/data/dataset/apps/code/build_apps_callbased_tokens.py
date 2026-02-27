#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib.util

import pandas as pd
from transformers import AutoTokenizer

BASE = Path('/root/autodl-tmp/xai/dataset/apps/code')
DATA_DIR = BASE / 'humaneval_like'
MESSAGE_DIR = BASE / 'message_trace'
PAIR_DIR = BASE / 'pair'
PROJECTS_DIR = BASE / 'projects'

for d in (DATA_DIR, MESSAGE_DIR, PAIR_DIR, PROJECTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

APPS_TEST = Path('/root/autodl-tmp/xai/dataset/apps/test.jsonl')
MODEL_PATH = os.getenv('XAI_MODEL_PATH', '/root/autodl-tmp/qwen3-8B')
SERVER_PATH = '/root/autodl-tmp/FastAPI/qwen3coder/server1.py'
TOKEN_BUILD_PATH = '/root/autodl-tmp/xai/exp/pair/token_build.py'

USER_TEMPLATE_ASSISTED = (
    "You are a code completion assistant.\n"
    "Task: Complete the function body based only on the function definition and docstring.\n"
    "Rules:\n"
    "- Output only the function body (no signature, no docstring).\n"
    "- Preserve correct indentation (4 spaces).\n"
    "- Use only the Python standard library.\n"
    "- Do not add explanations or extra text.\n"
    "- Assume no extra context beyond what is shown.\n"
    "Function definition and docstring:\n"
    "{context}"
)

USER_TEMPLATE_AGENTIC = (
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


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f'Failed to load module: {path}')
    spec.loader.exec_module(module)
    return module


def split_args(arg_str: str) -> List[str]:
    args: List[str] = []
    cur: List[str] = []
    depth = 0
    in_str = False
    str_char = ''
    i = 0
    while i < len(arg_str):
        ch = arg_str[i]
        if in_str:
            cur.append(ch)
            if ch == str_char and (i == 0 or arg_str[i - 1] != '\\'):
                in_str = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            str_char = ch
            cur.append(ch)
            i += 1
            continue
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(depth - 1, 0)
        if ch == ',' and depth == 0:
            args.append(''.join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    if cur:
        args.append(''.join(cur).strip())
    return [a for a in args if a]


def extract_def_line(starter_code: str, fn_name: str) -> str | None:
    for line in starter_code.splitlines():
        if line.strip().startswith(f'def {fn_name}'):
            return line.strip()
    for line in starter_code.splitlines():
        if line.lstrip().startswith('def '):
            return line.strip()
    return None


def build_signature(def_line: str | None, fn_name: str) -> str:
    if not def_line:
        return f'def {fn_name}():'
    m = re.search(
        rf'def\s+{re.escape(fn_name)}\s*\((.*)\)\s*(->\s*[^:]+)?\s*:',
        def_line,
    )
    if not m:
        m2 = re.search(r'def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*(->\s*[^:]+)?\s*:', def_line)
        if not m2:
            return f'def {fn_name}():'
        args_str = m2.group(2)
        ret = m2.group(3) or ''
    else:
        args_str = m.group(1)
        ret = m.group(2) or ''
    args = split_args(args_str)
    if args:
        first = args[0].strip()
        if first == 'self' or first.startswith('self:'):
            args = args[1:]
    args_str2 = ', '.join([a for a in args if a])
    return f'def {fn_name}({args_str2}){ret}:'


def collect_preamble(starter_code: str) -> List[str]:
    lines: List[str] = []
    for line in starter_code.splitlines():
        if line.strip().startswith('class Solution'):
            break
        if line.strip().startswith('def '):
            break
        if line.strip() == '':
            if lines and lines[-1] != '':
                lines.append('')
            continue
        if line.strip().startswith('#'):
            lines.append(line.rstrip())
    while lines and lines[-1] == '':
        lines.pop()
    return lines


def infer_typing_imports(signature: str) -> List[str]:
    typing_names = []
    for name in ['List', 'Optional', 'Dict', 'Tuple', 'Set', 'Deque', 'DefaultDict']:
        if re.search(rf'\b{name}\b', signature):
            typing_names.append(name)
    if not typing_names:
        return []
    return [f"from typing import {', '.join(sorted(set(typing_names)))}"]


def build_prompt(starter_code: str, fn_name: str, question: str) -> str:
    def_line = extract_def_line(starter_code, fn_name)
    signature = build_signature(def_line, fn_name)
    preamble = collect_preamble(starter_code)
    imports = infer_typing_imports(signature)

    lines: List[str] = []
    if preamble:
        lines.extend(preamble)
    if imports:
        if lines:
            lines.append('')
        lines.extend(imports)
    if lines:
        lines.append('')
    lines.append(signature)
    lines.append('    """')
    for qline in question.splitlines():
        lines.append('    ' + qline)
    lines.append('    """')
    lines.append('    pass')
    return '\n'.join(lines).rstrip() + '\n'


def extract_canonical_body(solution: str, fn_name: str) -> str:
    lines = solution.replace('\t', '    ').splitlines()
    def_idx = None
    def_indent = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith(f'def {fn_name}'):
            def_idx = i
            def_indent = len(line) - len(line.lstrip(' '))
            break
    if def_idx is None:
        return ''
    body_lines: List[str] = []
    for j in range(def_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            body_lines.append('')
            continue
        indent = len(line) - len(line.lstrip(' '))
        if indent <= def_indent:
            break
        if len(line) >= def_indent + 4:
            body_lines.append(line[def_indent + 4:])
        else:
            body_lines.append(line.lstrip())
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    if not body_lines:
        return ''
    normalized = [('    ' + l) if l.strip() else '' for l in body_lines]
    return '\n'.join(normalized)


def is_call_based(obj: dict) -> Tuple[bool, dict | None]:
    io_raw = obj.get('input_output') or ''
    if not io_raw:
        return False, None
    try:
        io_obj = json.loads(io_raw)
    except Exception:
        return False, None
    if isinstance(io_obj, dict) and io_obj.get('fn_name'):
        return True, io_obj
    return False, io_obj


def build_read_file_tool(server, base_dir: str | None = None) -> list:
    schema = {
        'name': 'read_file',
        'description': 'Read a text file by absolute path. If output says it is partial, call again with offset/limit.',
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'Absolute path to the file to read.',
                },
                'offset': {
                    'type': 'integer',
                    'description': 'Byte offset to start reading from.',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Maximum number of bytes to read.',
                },
            },
            'required': ['file_path'],
        },
    }
    return [
        server.Tool(
            type='function',
            function=server.FunctionDefinition(**schema),
        )
    ]


def convert_tool_calls(tool_calls: list[dict]) -> list[dict]:
    converted = []
    for call in tool_calls or []:
        name = call.get('name') or call.get('function', {}).get('name')
        args = call.get('args')
        if args is None:
            args = call.get('function', {}).get('arguments', {})
        converted.append(
            {
                'id': call.get('id') or f'tool-{id(call)}',
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': args,
                },
            }
        )
    return converted


def trim_final_assistant(messages: list[dict]) -> list[dict]:
    trimmed = list(messages)
    while trimmed:
        last = trimmed[-1]
        if last.get('type') == 'ai' and not last.get('tool_calls'):
            trimmed.pop()
            continue
        break
    return trimmed


def build_agentic_prompt(server, tools, message_trace: list[dict]) -> str:
    trimmed = trim_final_assistant(message_trace)
    formatted = []
    for msg in trimmed:
        mtype = msg.get('type')
        if mtype == 'human':
            formatted.append(server.ChatMessage(role='user', content=msg.get('content')))
        elif mtype == 'ai':
            formatted.append(
                server.ChatMessage(
                    role='assistant',
                    content=msg.get('content'),
                    tool_calls=convert_tool_calls(msg.get('tool_calls', [])),
                )
            )
        elif mtype == 'tool':
            formatted.append(
                server.ChatMessage(
                    role='tool',
                    content=msg.get('content'),
                    tool_call_id=msg.get('tool_call_id'),
                )
            )
    return server.build_prompt(formatted, tools)


def build_assisted_prompt(server, user_prompt: str) -> str:
    messages = [server.ChatMessage(role='user', content=user_prompt)]
    return server.build_prompt(messages, tools=None)


def add_context_marker(text: str) -> str:
    marker = '#Function definition and docstring:\n'
    target = 'Function definition and docstring:\n'
    if marker in text:
        return text
    if target in text:
        return text.replace(target, marker, 1)
    return text


def insert_tool_response_marker(prompt: str) -> str:
    marker = '#Function definition and docstring:\n'
    if marker in prompt:
        return prompt
    needle = '<tool_response>\n'
    idx = prompt.find(needle)
    if idx == -1:
        return prompt
    insert_at = idx + len(needle)
    return prompt[:insert_at] + marker + prompt[insert_at:]


def normalize_body(text: str) -> str:
    if not text:
        return ''
    cleaned = text.replace('\t', ' ' * 4)
    lines = cleaned.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ''
    if lines[0].startswith('def ') and lines[0].lstrip().startswith('def '):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return ''

    indents = [len(line) - len(line.lstrip(' ')) for line in lines if line.strip()]
    min_indent = min(indents) if indents else 0
    if min_indent >= 4:
        shift = min_indent - 4
        normalized = [line[shift:] if line.strip() else '' for line in lines]
    else:
        shift = 4 - min_indent
        normalized = [(' ' * shift + line.lstrip(' ')) if line.strip() else '' for line in lines]
    return '\n'.join(normalized)


def build_continuation(canonical_solution: str) -> str:
    body = normalize_body(canonical_solution)
    return f"```python\n{body}\n```"


token_build = load_module(TOKEN_BUILD_PATH, 'token_build')
server = load_module(SERVER_PATH, 'qserver')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
server.tokenizer = tokenizer

records: List[dict] = []
assisted_output: Dict[str, dict] = {}
agentic_output: Dict[str, dict] = {}
canonical_by_key: Dict[str, str] = {}

with open(APPS_TEST, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        is_cb, io_obj = is_call_based(obj)
        if not is_cb:
            continue
        fn_name = io_obj.get('fn_name')
        question = obj.get('question', '')
        starter_code = obj.get('starter_code', '')
        solutions = json.loads(obj.get('solutions') or '[]')
        solution = ''
        for sol in solutions:
            if re.search(rf'\bdef\s+{re.escape(fn_name)}\s*\(', sol):
                solution = sol
                break
        canonical_solution = extract_canonical_body(solution, fn_name)
        prompt = build_prompt(starter_code, fn_name, question)

        task_id = f"APPS/{obj.get('id')}"
        task_num = task_id.split('/')[1]
        key = f"humaneval_{task_num}"

        project_dir = PROJECTS_DIR / key
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / 'main.py').write_text(prompt)

        assisted_user_prompt = USER_TEMPLATE_ASSISTED.format(context=prompt)
        assisted_output[key] = {
            'prompt': assisted_user_prompt,
            'prompt_chars': len(assisted_user_prompt),
            'completion_raw': '',
            'completion': '',
            'completion_chars': 0,
            'duration_s': 0.0,
            'model': 'synthetic',
            'base_url': 'synthetic',
        }

        agentic_user_prompt = USER_TEMPLATE_AGENTIC.format(project_dir=str(project_dir))
        tool_call_id = f"tool-{key}"
        tool_call = {
            'name': 'read_file',
            'args': {'file_path': str(project_dir / 'main.py')},
            'id': tool_call_id,
            'type': 'tool_call',
        }
        message_trace = [
            {'type': 'human', 'content': agentic_user_prompt},
            {'type': 'ai', 'content': '', 'tool_calls': [tool_call]},
            {'type': 'tool', 'content': prompt.rstrip(), 'tool_call_id': tool_call_id},
            {'type': 'ai', 'content': ''},
        ]
        agentic_output[key] = {
            'prompt': agentic_user_prompt,
            'prompt_chars': len(agentic_user_prompt),
            'final_response': '',
            'completion_chars': 0,
            'duration_s': 0.0,
            'tool_calls': [tool_call],
            'tool_calls_count': 1,
            'message_trace': message_trace,
            'model': 'synthetic',
            'base_url': 'synthetic',
        }

        records.append({
            'task_id': task_id,
            'prompt': prompt.rstrip(),
            'canonical_solution': canonical_solution.rstrip(),
            'test': '',
            'entry_point': fn_name,
        })
        canonical_by_key[key] = canonical_solution.rstrip()

if len(records) != 38:
    raise RuntimeError(f'Expected 38 call-based records, got {len(records)}')

parquet_path = DATA_DIR / 'apps_callbased_test.parquet'
pd.DataFrame(records).to_parquet(parquet_path, index=False)

agentic_path = MESSAGE_DIR / 'agentic_output.json'
agentic_path.write_text(json.dumps(agentic_output, ensure_ascii=False, indent=2))

pair_prompts: Dict[str, Dict[str, str]] = {}

for key in canonical_by_key:
    assisted_entry = assisted_output[key]
    assisted_user_prompt = assisted_entry['prompt']
    assisted_prefix = build_assisted_prompt(server, assisted_user_prompt)
    assisted_prefix = add_context_marker(assisted_prefix)

    message_trace = agentic_output[key]['message_trace']
    project_dir = PROJECTS_DIR / key
    tools = build_read_file_tool(server, base_dir=str(project_dir))
    agentic_prefix = build_agentic_prompt(server, tools, message_trace)
    agentic_prefix = insert_tool_response_marker(agentic_prefix)

    continuation = build_continuation(canonical_by_key[key])
    pair_prompts[key] = {
        'assisted': assisted_prefix + continuation,
        'agentic': agentic_prefix + continuation,
    }

pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    raise RuntimeError('Tokenizer has no pad_token_id')

token_output: Dict[str, Any] = {}

for key, pair in pair_prompts.items():
    assisted = pair.get('assisted', '')
    agentic = pair.get('agentic', '')
    segments = token_build.split_assisted_prompt(assisted)

    agentic_ids, agentic_offsets = token_build.tokenize_with_offsets(tokenizer, agentic)
    assisted_ids, _ = token_build.tokenize_with_offsets(tokenizer, assisted)

    segment_meta: Dict[str, Dict[str, Any]] = {}
    if segments is None:
        segment_meta['error'] = {'message': 'marker or assistant tag not found'}
    else:
        tool_resp_open_char, tool_resp_open_char_end, _, _ = token_build.find_marker_range_any(
            agentic, [token_build.TOOL_RESPONSE_OPEN + "\n", token_build.TOOL_RESPONSE_OPEN], agentic_offsets
        )
        tool_resp_close_char, _, _, _ = token_build.find_marker_range_any(
            agentic, [token_build.TOOL_RESPONSE_CLOSE], agentic_offsets
        )
        for name, text in segments.items():
            seg_ids = tokenizer(text, add_special_tokens=False).input_ids
            if name == 'function_docstring' and tool_resp_open_char != -1:
                search_start = tool_resp_open_char_end
                search_end = tool_resp_close_char if tool_resp_close_char != -1 else None
            else:
                search_start = 0
                search_end = None
            char_start, char_end, tok_start, tok_end = token_build.find_segment_range(
                agentic, text, agentic_offsets, search_start, search_end
            )
            found = tok_start >= 0
            segment_meta[name] = {
                'char_start': char_start,
                'char_end': char_end,
                'start': tok_start,
                'end': tok_end,
                'length': (tok_end - tok_start) if found else 0,
                'found': found,
                'token_ids': seg_ids,
            }

    canonical_positions: List[int] = []
    canonical_tokens: List[str] = []
    canonical_fence_start = -1
    canonical_fence_end = -1
    canonical_code_start = -1
    canonical_first_non_ws = -1
    canonical_info = segment_meta.get('canonical_solution')
    if canonical_info and canonical_info.get('found'):
        canonical_text = segments['canonical_solution']
        prefix, code, _ = token_build.find_code_block(canonical_text)
        canonical_char_start = canonical_info.get('char_start', -1)

        fence_idx = canonical_text.find('```python')
        if fence_idx == -1:
            fence_idx = canonical_text.find('```')
        if fence_idx != -1 and canonical_char_start != -1:
            fence_line_end = canonical_text.find('\n', fence_idx)
            if fence_line_end != -1:
                fence_char_start = canonical_char_start + fence_idx
                fence_char_end = canonical_char_start + fence_line_end + 1
                canonical_fence_start, canonical_fence_end = token_build.char_range_to_token_range(
                    agentic_offsets, fence_char_start, fence_char_end
                )

        if canonical_char_start != -1:
            code_char_start = canonical_char_start + len(prefix)
            code_char_end = code_char_start + len(code)
            canonical_code_start, code_end = token_build.char_range_to_token_range(
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

    system_char_start, _, system_start, system_start_end = token_build.find_marker_range_any(
        agentic, [token_build.SYSTEM_START + "\n", token_build.SYSTEM_START], agentic_offsets
    )
    system_char_end, _, system_end_start, system_end_end = token_build.find_marker_range(
        agentic, token_build.IM_END, agentic_offsets, system_char_start
    )
    system_content_start = system_start_end
    system_content_end = system_end_start

    user_char_start, _, user_start, user_start_end = token_build.find_marker_range_any(
        agentic, [token_build.USER_START + "\n", token_build.USER_START], agentic_offsets, system_char_end
    )
    _, _, user_end_start, user_end_end = token_build.find_marker_range(
        agentic, token_build.IM_END, agentic_offsets, user_char_start
    )

    _, _, assistant_start, assistant_end = token_build.find_marker_range_any(
        agentic, [token_build.ASSISTANT_TAG + "\n", token_build.ASSISTANT_TAG], agentic_offsets
    )

    _, _, tools_start, tools_end = token_build.find_marker_range_any(
        agentic, [token_build.TOOLS_OPEN], agentic_offsets
    )
    _, _, tools_close_start, tools_close_end = token_build.find_marker_range_any(
        agentic, [token_build.TOOLS_CLOSE], agentic_offsets
    )
    _, _, important_start, important_end = token_build.find_marker_range_any(
        agentic, [token_build.IMPORTANT_OPEN], agentic_offsets
    )
    _, _, important_close_start, important_close_end = token_build.find_marker_range_any(
        agentic, [token_build.IMPORTANT_CLOSE], agentic_offsets
    )

    _, _, tool_call_example_start, tool_call_example_end = token_build.find_marker_range(
        agentic, token_build.TOOL_CALL_OPEN, agentic_offsets, system_char_start
    )
    _, _, tool_call_example_close_start, tool_call_example_close_end = token_build.find_marker_range(
        agentic, token_build.TOOL_CALL_CLOSE, agentic_offsets, system_char_start
    )
    _, _, tool_call_actual_start, tool_call_actual_end = token_build.find_marker_range(
        agentic, token_build.TOOL_CALL_OPEN, agentic_offsets, system_char_end
    )
    _, _, tool_call_actual_close_start, tool_call_actual_close_end = token_build.find_marker_range(
        agentic, token_build.TOOL_CALL_CLOSE, agentic_offsets, system_char_end
    )

    _, _, tool_call_start, tool_call_end = token_build.find_marker_range_any(
        agentic, [token_build.TOOL_CALL_OPEN], agentic_offsets
    )
    _, _, tool_call_close_start, tool_call_close_end = token_build.find_marker_range_any(
        agentic, [token_build.TOOL_CALL_CLOSE], agentic_offsets
    )
    tool_response_char_start, tool_response_char_end, tool_response_start, tool_response_end = token_build.find_marker_range_any(
        agentic, [token_build.TOOL_RESPONSE_OPEN + "\n", token_build.TOOL_RESPONSE_OPEN], agentic_offsets
    )
    _, _, tool_response_close_start, tool_response_close_end = token_build.find_marker_range_any(
        agentic, [token_build.TOOL_RESPONSE_CLOSE], agentic_offsets
    )
    tool_response_content_start = tool_response_end if tool_response_end != -1 else -1

    aligned_ids, aligned_mask, aligned_ok, align_errors, align_warnings = token_build.build_aligned_ids(
        len(agentic_ids), pad_token_id, segment_meta
    )

    token_output[key] = {
        'agentic_ids': agentic_ids,
        'assisted_ids': assisted_ids,
        'assisted_aligned_ids': aligned_ids,
        'assisted_aligned_attention_mask': aligned_mask,
        'pad_token_id': pad_token_id,
        'agentic_length': len(agentic_ids),
        'assisted_length': len(assisted_ids),
        'segment_positions': {
            name: {
                'start': info.get('start'),
                'end': info.get('end'),
                'length': info.get('length'),
                'found': info.get('found'),
            }
            for name, info in segment_meta.items()
            if name != 'error'
        },
        'tool_positions': {
            'tool_call_first_open': {'start': tool_call_start, 'end': tool_call_end},
            'tool_call_first_close': {'start': tool_call_close_start, 'end': tool_call_close_end},
            'tool_call_example_open': {'start': tool_call_example_start, 'end': tool_call_example_end},
            'tool_call_example_close': {'start': tool_call_example_close_start, 'end': tool_call_example_close_end},
            'tool_call_actual_open': {'start': tool_call_actual_start, 'end': tool_call_actual_end},
            'tool_call_actual_close': {'start': tool_call_actual_close_start, 'end': tool_call_actual_close_end},
            'tool_response_open': {'start': tool_response_start, 'end': tool_response_end},
            'tool_response_content_start': tool_response_content_start,
            'tool_response_close': {'start': tool_response_close_start, 'end': tool_response_close_end},
            'tools_block_open': {'start': tools_start, 'end': tools_end},
            'tools_block_close': {'start': tools_close_start, 'end': tools_close_end},
            'important_open': {'start': important_start, 'end': important_end},
            'important_close': {'start': important_close_start, 'end': important_close_end},
        },
        'message_positions': {
            'system_start': system_start,
            'system_start_end': system_start_end,
            'system_end': system_end_start,
            'system_end_end': system_end_end,
            'system_content_start': system_content_start,
            'system_content_end': system_content_end,
            'user_start': user_start,
            'user_start_end': user_start_end,
            'user_end': user_end_start,
            'user_end_end': user_end_end,
            'assistant_start': assistant_start,
            'assistant_start_end': assistant_end,
        },
        'canonical_positions': {
            'assistant_start': canonical_info['start'] if canonical_info else -1,
            'fence_start': canonical_fence_start,
            'fence_end': canonical_fence_end,
            'code_start': canonical_code_start,
            'first_non_ws': canonical_first_non_ws,
        },
        'canonical_non_ws_positions': canonical_positions,
        'canonical_non_ws_tokens': canonical_tokens,
        'aligned_ok': aligned_ok,
        'aligned_errors': align_errors,
        'aligned_warnings': align_warnings,
    }

pair_tokens_path = PAIR_DIR / 'pair_tokens.json'
pair_tokens_path.write_text(json.dumps(token_output, ensure_ascii=False, indent=2))

print('DONE')
print('parquet:', parquet_path)
print('agentic_output:', agentic_path)
print('pair_tokens:', pair_tokens_path)
