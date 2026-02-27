#!/usr/bin/env python3
"""
OpenAI-Compatible Server for Qwen3-Coder-30B
"""

import asyncio
import gc
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


def _fix_tool_arguments(arguments: str | dict) -> str:
    """
    Fix tool call arguments by parsing JSON-encoded string values back to their proper types.

    The qwen3 chat template uses |tojson which converts arrays/objects to JSON strings.
    The vLLM parser doesn't convert them back, so we need to do it here.

    Args:
        arguments: Either a JSON string or a dict

    Returns:
        A JSON string with properly typed values
    """
    if isinstance(arguments, dict):
        args_dict = arguments
    else:
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments

    # Parse JSON string values back to their original types
    for key, value in args_dict.items():
        if isinstance(value, str) and value.startswith(('[', '{', '"')):
            try:
                parsed = json.loads(value)
                args_dict[key] = parsed
            except (json.JSONDecodeError, ValueError):
                pass

    return json.dumps(args_dict, ensure_ascii=False)

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/root/autodl-tmp/models/qwen3-coder-30b"
MODEL_NAME = "qwen3-coder-30b"
MAX_MODEL_LEN = 163840  # Maximum context length (matches vLLM setting)
HIDDEN_DIR = "/root/autodl-tmp/xai/exp/hidden"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None

# vLLM tool parser - imported lazily to avoid early import errors
tool_parser_class = None


# ============================================================================
# Pydantic Models (OpenAI-Compatible)
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None  # 直接使用 dict，与 vLLM 保持一致


class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = "auto"  # Default to auto for tool calling compatibility
    return_hidden_states: bool = False
    hidden_tag: str | None = None
    return_attentions: bool = False
    attention_mode: Literal["full", "last_token", "mean_heads"] | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: dict


class ChatMessageResponse(BaseModel):
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo
    hidden_state_path: str | None = None
    hidden_state_meta: dict | None = None


def _safe_hidden_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def _build_token_spans(tok: AutoTokenizer, token_ids: list[int]) -> tuple[list[str], list[list[int]], str]:
    token_texts: list[str] = []
    spans: list[list[int]] = []
    cursor = 0
    for tid in token_ids:
        text = tok.decode([tid], skip_special_tokens=False)
        token_texts.append(text)
        start = cursor
        cursor += len(text)
        spans.append([start, cursor])
    full_text = "".join(token_texts)
    return token_texts, spans, full_text


def _cleanup_cuda_oom() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _raise_oom(exc: Exception) -> None:
    _cleanup_cuda_oom()
    logger.error(f"CUDA OOM: {exc}")
    raise HTTPException(status_code=500, detail="CUDA out of memory") from exc




class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None


class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class ChatCompletionStreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelsList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ============================================================================
# Token Accumulator (for proper incremental decoding)
# ============================================================================

class TokenAccumulator:
    """Accumulates tokens and produces delta text without corruption."""

    def __init__(self, tok: AutoTokenizer, special_token_ids: set[int] | None = None):
        self.tokenizer = tok
        self.all_tokens: list[int] = []
        self.last_decoded_len = 0
        self.special_token_ids = special_token_ids or set()

    def add_tokens(self, new_tokens: list[int]) -> tuple[str, str]:
        """
        Add new tokens and return (delta_text, full_text).

        Filters out special tokens and decodes incrementally.
        """
        # Filter out special tokens
        filtered_tokens = [t for t in new_tokens if t not in self.special_token_ids]
        self.all_tokens.extend(filtered_tokens)

        # Decode full sequence
        full_text = self.tokenizer.decode(
            self.all_tokens,
            skip_special_tokens=True,
        )

        # Extract delta (new text since last decode)
        delta_text = full_text[self.last_decoded_len:]
        self.last_decoded_len = len(full_text)

        return delta_text, full_text

    def get_full_text(self) -> str:
        """Get the full decoded text so far."""
        return self.tokenizer.decode(
            self.all_tokens,
            skip_special_tokens=True,
        )


# ============================================================================
# Tool Parser Integration
# ============================================================================

def get_tool_parser():
    """Get or initialize the vLLM tool parser."""
    global tool_parser_class

    if tool_parser_class is None:
        try:
            import sys
            sys.path.insert(0, '/root/miniconda3/envs/qwen/lib/python3.12/site-packages')
            from vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser import (
                StreamingXMLToolCallParser
            )
            tool_parser_class = StreamingXMLToolCallParser
            logger.info("Successfully imported vLLM tool parser")
        except ImportError as e:
            logger.warning(f"Failed to import vLLM tool parser: {e}")
            tool_parser_class = None

    return tool_parser_class


# ============================================================================
# Server Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global model, tokenizer

    logger.info(f"Loading model from {MODEL_PATH}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # Load model (manual device placement without accelerate)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
        trust_remote_code=True,
    )
    model = model.to("cuda")  # Move to GPU manually

    model.eval()  # Set to evaluation mode

    # Compile model with torch.compile for faster inference
    logger.info("Compiling model with torch.compile() (this may take a minute)...")
    try:
        # Using fullmode=False allows dynamic shapes (for varying input lengths)
        # Using mode="reduce-overhead" for faster startup with good speedup
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=False,
        )
        logger.info("Model compiled successfully!")
    except Exception as e:
        logger.warning(f"Failed to compile model: {e}. Continuing without compilation.")
        logger.warning("This is not critical - the server will work, just slower.")

    # Get special token IDs
    special_tokens = {
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|tool_start|>",
        "<|tool_end|>",
    }

    special_token_ids = set()
    for tok in special_tokens:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid not in [tokenizer.unk_token_id, None]:
                special_token_ids.add(tid)
        except Exception:
            pass

    # Store special token IDs in app state
    app.state.special_token_ids = special_token_ids
    logger.info(f"Model loaded successfully. Special token IDs: {special_token_ids}")
    logger.info("Server ready!")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
    del model
    del tokenizer


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(lifespan=lifespan)


# ============================================================================
# Helper Functions
# ============================================================================

def _pydantic_to_dict(obj: Any) -> Any:
    """
    递归地将 Pydantic 模型转换为字典。

    这个函数确保所有嵌套的 Pydantic 对象都被完全转换为字典，
    避免 Jinja2 模板中的 .items() 调用失败。
    """
    from pydantic import BaseModel

    # 处理 None
    if obj is None:
        return None

    # 处理 Pydantic 模型
    if isinstance(obj, BaseModel):
        # 先转换为字典，然后递归处理所有值
        result = {}
        for key, value in obj.model_dump(exclude_none=True).items():
            result[key] = _pydantic_to_dict(value)
        return result

    # 处理字典
    if isinstance(obj, dict):
        return {k: _pydantic_to_dict(v) for k, v in obj.items()}

    # 处理列表
    if isinstance(obj, (list, tuple)):
        return [_pydantic_to_dict(item) for item in obj]

    # 处理集合
    if isinstance(obj, set):
        return {_pydantic_to_dict(item) for item in obj}

    # 基本类型直接返回
    return obj


def format_messages_for_template(messages: list[ChatMessage]) -> list[dict]:
    """
    Convert ChatMessage objects to the format expected by chat template.
    使用递归转换确保所有嵌套的 Pydantic 对象都被转换为字典。
    """
    formatted = []
    for msg in messages:
        # 使用递归转换函数完全转换 Pydantic 对象
        msg_dict = _pydantic_to_dict(msg)

        # 处理 content（可以是字符串或数组）
        if msg_dict.get("content") is not None:
            content = msg_dict["content"]
            if isinstance(content, list):
                # 从 content 数组中提取文本
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                msg_dict["content"] = "".join(text_parts)

        formatted.append(msg_dict)

    return formatted


def format_tools_for_template(tools: list[Tool] | None) -> list[dict] | None:
    """Convert Tool objects to the format expected by chat template."""
    if tools is None:
        return None

    formatted = []
    for i, tool in enumerate(tools):
        logger.debug(f"DEBUG: Tool {i} - raw tool type: {type(tool)}")
        logger.debug(f"DEBUG: Tool {i} - raw tool.function type: {type(tool.function)}")

        # 使用递归转换确保所有嵌套的 Pydantic 对象都被转换为字典
        func_dict = _pydantic_to_dict(tool.function)

        logger.debug(f"DEBUG: Tool {i} - func_dict type after conversion: {type(func_dict)}")
        logger.debug(f"DEBUG: Tool {i} - func_dict keys: {list(func_dict.keys()) if isinstance(func_dict, dict) else 'N/A'}")

        if "parameters" in func_dict:
            params = func_dict["parameters"]
            logger.debug(f"DEBUG: Tool {i} - parameters type: {type(params)}")
            logger.debug(f"DEBUG: Tool {i} - parameters is dict: {isinstance(params, dict)}")
            logger.debug(f"DEBUG: Tool {i} - parameters content: {params}")

        formatted.append({
            "type": "function",
            "function": func_dict
        })

    return formatted


def _postprocess_tool_calls(messages: list[dict]) -> None:
    """
    Convert tool_call.arguments from JSON string to dict.

    Per Transformers docs, chat templates expect arguments as dict,
    not JSON string (OpenAI format). This matches vLLM's behavior.
    """
    for message in messages:
        if (
            message.get("role") == "assistant"
            and "tool_calls" in message
            and isinstance(message["tool_calls"], list)
        ):
            for item in message["tool_calls"]:
                func = item.get("function", {})
                if isinstance(func, dict):
                    args = func.get("arguments")
                    if args and isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            func["arguments"] = {}
                    elif not args:
                        func["arguments"] = {}


def build_prompt(
    messages: list[ChatMessage],
    tools: list[Tool] | None = None
) -> str:
    """Build prompt from messages using chat template."""
    formatted_messages = format_messages_for_template(messages)
    formatted_tools = format_tools_for_template(tools)

    # Postprocess: convert tool_call.arguments from JSON string to dict
    _postprocess_tool_calls(formatted_messages)

    # Debug: 打印传入模板的数据
    if formatted_tools:
        logger.debug(f"DEBUG: Tools count: {len(formatted_tools)}")
        for i, tool in enumerate(formatted_tools):
            logger.debug(f"DEBUG: Tool {i}: {tool}")
            # 检查 parameters 的类型
            if "function" in tool and "parameters" in tool["function"]:
                params = tool["function"]["parameters"]
                logger.debug(f"DEBUG: Tool {i} parameters type: {type(params)}, is dict: {isinstance(params, dict)}")
                if isinstance(params, dict):
                    logger.debug(f"DEBUG: Tool {i} parameters keys: {list(params.keys()) if params else 'None'}")

    prompt = tokenizer.apply_chat_template(
        formatted_messages,
        tools=formatted_tools,
        add_generation_prompt=True,
        tokenize=False,
        return_dict=False,
    )

    return prompt


def format_sse_chunk(
    chunk_id: str,
    model: str,
    created: int,
    delta: DeltaMessage,
    finish_reason: str | None = None,
) -> str:
    """Format a streaming response chunk as SSE."""
    chunk = ChatCompletionStreamChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[
            StreamChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
    )

    return f"data: {chunk.model_dump_json()}\n\n"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/v1/models", response_model=ModelsList)
async def list_models():
    """List available models."""
    return ModelsList(
        data=[
            ModelInfo(
                id=MODEL_NAME,
                created=int(time.time()),
                owned_by="qwen",
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint.

    Supports both streaming and non-streaming modes.
    """
    # Log request for debugging
    logger.info("=" * 60)
    logger.info(f"NEW REQUEST: stream={request.stream}, messages={len(request.messages)}, tools={len(request.tools) if request.tools else 0}")
    logger.info("=" * 60)

    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream",
        )
    else:
        return await generate_non_stream(request)


async def generate_non_stream(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate non-streaming response."""
    # Build prompt
    prompt = build_prompt(request.messages, request.tools)

    # Tokenize (keep attention_mask for reliable generation)
    offsets = None
    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = inputs.get("offset_mapping")
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    prompt_tokens = input_ids.shape[1]

    # Check if input exceeds max length
    if prompt_tokens > MAX_MODEL_LEN:
        logger.warning(f"Input length {prompt_tokens} exceeds MAX_MODEL_LEN {MAX_MODEL_LEN}")

    # Generation parameters
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": min(request.max_tokens or 2048, MAX_MODEL_LEN - prompt_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    # Generate
    try:
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
    except torch.OutOfMemoryError as exc:
        _raise_oom(exc)

    # Decode output (only the generated part)
    generated_ids = output_ids[0][prompt_tokens:]
    special_token_ids = app.state.special_token_ids

    accumulator = TokenAccumulator(tokenizer, special_token_ids)
    delta_text, full_text = accumulator.add_tokens(generated_ids.tolist())

    # Parse tool calls if tools were provided
    tool_calls = None
    tool_parser_class = get_tool_parser()

    if request.tools and tool_parser_class:
        parser = tool_parser_class()
        # Use Pydantic tool objects so vLLM parser can read parameter types.
        parser.set_tools(request.tools)

        # Parse the full output
        try:
            result = parser.parse_single_streaming_chunks(full_text)
            if result.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=f"chatcmpl-tool-{uuid.uuid4().hex}",
                        function={
                            "name": tc.function.name,
                            "arguments": _fix_tool_arguments(tc.function.arguments),
                        },
                    )
                    for tc in result.tool_calls
                    if tc.function and tc.function.name
                ]
        except Exception as e:
            logger.warning(f"Failed to parse tool calls: {e}")

    # Build response
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    finish_reason = "tool_calls" if tool_calls else "stop"

    hidden_state_path = None
    hidden_state_meta = None
    if request.return_hidden_states or request.return_attentions:
        hidden_path = Path(HIDDEN_DIR)
        hidden_path.mkdir(parents=True, exist_ok=True)
        tag = request.hidden_tag or completion_id
        safe_tag = _safe_hidden_tag(tag)
        file_path = hidden_path / f"{safe_tag}.pt"

        full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)
        full_attention = None
        if attention_mask is not None:
            gen_mask = torch.ones(
                (attention_mask.shape[0], generated_ids.shape[0]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention = torch.cat([attention_mask, gen_mask], dim=1)

        try:
            with torch.no_grad():
                outputs = model(
                    full_ids,
                    attention_mask=full_attention,
                    output_hidden_states=request.return_hidden_states,
                    output_attentions=request.return_attentions,
                    use_cache=False,
                )
        except torch.OutOfMemoryError as exc:
            _raise_oom(exc)
        stacked = None
        if request.return_hidden_states:
            hidden_states = outputs.hidden_states or ()
            if hidden_states:
                stacked = torch.stack(
                    [hs[0].detach().cpu().to(torch.float16) for hs in hidden_states[1:]],
                    dim=0,
                )
            else:
                stacked = torch.empty((0, full_ids.shape[1], 0), dtype=torch.float16)

        attn_data = None
        attn_mode = None
        if request.return_attentions:
            attn_mode = request.attention_mode or "last_token"
            attentions = outputs.attentions or ()
            if attentions:
                if attn_mode == "full":
                    attn_data = torch.stack(
                        [att[0].detach().cpu().to(torch.float16) for att in attentions],
                        dim=0,
                    )
                elif attn_mode == "mean_heads":
                    attn_data = torch.stack(
                        [att[0].mean(dim=0).detach().cpu().to(torch.float16) for att in attentions],
                        dim=0,
                    )
                else:
                    # Default: last token attention, keep head dimension
                    attn_data = torch.stack(
                        [att[0, :, -1, :].detach().cpu().to(torch.float16) for att in attentions],
                        dim=0,
                    )
            else:
                attn_data = torch.empty((0,), dtype=torch.float16)

        prompt_token_ids = input_ids[0].tolist()
        gen_token_ids = generated_ids.tolist()
        full_token_ids = prompt_token_ids + gen_token_ids
        token_texts, token_spans, full_text = _build_token_spans(tokenizer, full_token_ids)
        prompt_offsets = None
        if offsets is not None:
            prompt_offsets = offsets[0].tolist()

        payload = {
            "completion_id": completion_id,
            "created": created,
            "model": MODEL_NAME,
            "prompt": prompt,
            "messages": [m.model_dump() for m in request.messages],
            "tools": [t.model_dump() for t in request.tools] if request.tools else None,
            "prompt_token_ids": prompt_token_ids,
            "prompt_offsets": prompt_offsets,
            "generated_token_ids": gen_token_ids,
            "full_token_ids": full_token_ids,
            "token_texts": token_texts,
            "token_spans": token_spans,
            "full_text_from_tokens": full_text,
        }
        if request.return_hidden_states:
            payload["hidden_states"] = stacked
        if request.return_attentions:
            payload["attentions"] = attn_data
            payload["attention_mode"] = attn_mode
        torch.save(payload, file_path)
        hidden_state_path = str(file_path)
        hidden_state_meta = {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(gen_token_ids),
            "total_tokens": len(full_token_ids),
            "num_layers": int(stacked.shape[0]) if stacked is not None else 0,
            "hidden_size": int(stacked.shape[2]) if stacked is not None and stacked.numel() else 0,
            "attention_mode": attn_mode,
            "attention_shape": list(attn_data.shape) if attn_data is not None else None,
        }

    response = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatMessageResponse(
                    role="assistant",
                    content=delta_text if not tool_calls else None,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated_ids),
            total_tokens=prompt_tokens + len(generated_ids),
        ),
        hidden_state_path=hidden_state_path,
        hidden_state_meta=hidden_state_meta,
    )

    logger.info(f"RESPONSE: {response.model_dump_json(exclude_none=True)}")
    return response


async def generate_stream(request: ChatCompletionRequest):
    """Generate streaming response."""
    logger.info(f"Starting stream generation...")

    # Build prompt
    prompt = build_prompt(request.messages, request.tools)

    # Tokenize (keep attention_mask for reliable generation)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    prompt_tokens = input_ids.shape[1]

    # Check if input exceeds max length
    if prompt_tokens > MAX_MODEL_LEN:
        logger.warning(f"Input length {prompt_tokens} exceeds MAX_MODEL_LEN {MAX_MODEL_LEN}")

    # Generation parameters
    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": min(request.max_tokens or 2048, MAX_MODEL_LEN - prompt_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    # Initialize
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    special_token_ids = app.state.special_token_ids
    accumulator = TokenAccumulator(tokenizer, special_token_ids)

    # Initialize tool parser if tools provided
    tool_parser_class = get_tool_parser()
    tool_parser = None

    if request.tools and tool_parser_class:
        tool_parser = tool_parser_class()
        # Use Pydantic tool objects so vLLM parser can read parameter types.
        tool_parser.set_tools(request.tools)

    completion_tokens = 0
    current_tool_call_id = None  # Track current tool call ID
    saw_tool_calls = False

    # Generate with streaming
    try:
        with torch.no_grad():
            # Use model.generate with streaming
            # Note: We'll simulate streaming by generating all at once
            # then yielding chunks, since transformers doesn't have true streaming
            output_ids = model.generate(**gen_kwargs)
    except torch.OutOfMemoryError as exc:
        _raise_oom(exc)

    generated_ids = output_ids[0][prompt_tokens:]

    # Process tokens
    for i, token_id in enumerate(generated_ids.tolist()):
        delta_text, full_text = accumulator.add_tokens([token_id])
        completion_tokens += 1

        if not delta_text:
            continue

        if tool_parser:
            try:
                result = tool_parser.parse_single_streaming_chunks(delta_text)
            except Exception as e:
                logger.debug(f"Tool parse error: {e}")
                result = None

            if result:
                delta_content = result.content
                delta_tool_calls = None
                if result.tool_calls:
                    saw_tool_calls = True
                    delta_tool_calls = []
                    for tc in result.tool_calls:
                        tc_id = tc.id or current_tool_call_id or f"chatcmpl-tool-{uuid.uuid4().hex}"
                        if tc.id and current_tool_call_id is None:
                            current_tool_call_id = tc.id
                        elif current_tool_call_id is None:
                            current_tool_call_id = tc_id
                        delta_tool_calls.append({
                            "index": tc.index or 0,
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            } if tc.function else None,
                        })

                if delta_content is None and not delta_tool_calls:
                    continue

                delta = DeltaMessage(
                    role="assistant",
                    content=delta_content,
                    tool_calls=delta_tool_calls,
                )
                yield format_sse_chunk(completion_id, MODEL_NAME, created, delta).encode()
                continue

        # No tool parser or fallback path: stream raw text
        delta = DeltaMessage(role="assistant", content=delta_text)
        yield format_sse_chunk(completion_id, MODEL_NAME, created, delta).encode()

    # Send final chunk with finish_reason
    yield format_sse_chunk(
        completion_id,
        MODEL_NAME,
        created,
        DeltaMessage(),
        finish_reason="tool_calls" if saw_tool_calls else "stop",
    ).encode()

    logger.info(f"Streaming complete. Tokens: {completion_tokens}")
    logger.info("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
