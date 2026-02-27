#!/usr/bin/env python3
"""
OpenAI-Compatible Server for Qwen3-Coder-30B
"""

import argparse
import ast
import asyncio
import contextvars
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
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


def _normalize_mask_value(value: Any) -> set[tuple[int, int]]:
    """Parse router mask from a set-of-tuples string, list of pairs, or dict."""
    if value is None:
        return set(DEFAULT_ROUTER_MASK_POINTS)

    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except Exception as e:
            raise ValueError(f"Invalid router_mask string: {e}")

    points: set[tuple[int, int]] = set()
    if isinstance(parsed, dict):
        for layer_key, routers in parsed.items():
            try:
                layer_idx = int(layer_key)
            except Exception:
                raise ValueError("router_mask dict keys must be ints")
            if routers is None:
                continue
            if not isinstance(routers, (list, tuple, set)):
                raise ValueError("router_mask dict values must be list/tuple/set")
            for r in routers:
                points.add((layer_idx, int(r)))
        return points

    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                points.add((int(item[0]), int(item[1])))
            else:
                raise ValueError("router_mask must contain (layer, router) pairs")
        return points

    raise ValueError("Unsupported router_mask format")


def _build_layer_router_mask(points: set[tuple[int, int]]) -> dict[int, set[int]]:
    mask: dict[int, set[int]] = {}
    for layer_idx, router_idx in points:
        if layer_idx < 0 or router_idx < 0:
            continue
        mask.setdefault(layer_idx, set()).add(router_idx)
    return mask


def _validate_router_mask(mask: dict[int, set[int]], num_experts: int, top_k: int) -> None:
    for layer_idx, routers in mask.items():
        if len(routers) >= num_experts:
            raise ValueError(f"Mask removes all experts at layer {layer_idx}")
        if num_experts - len(routers) < top_k:
            raise ValueError(f"Mask leaves fewer than top_k experts at layer {layer_idx}")

def _mask_keep_topk(out: torch.Tensor, router_ids: set[int], top_k: int) -> torch.Tensor:
    orig = out
    k = min(top_k, orig.shape[-1])
    topk_vals, topk_idx = orig.topk(k=k, dim=-1)

    neg_inf = torch.tensor(float("-inf"), device=orig.device, dtype=orig.dtype)
    kept = torch.full_like(orig, neg_inf)
    kept.scatter_(-1, topk_idx, topk_vals)

    if router_ids:
        kept[..., list(router_ids)] = neg_inf

    # If all logits are masked, fall back to original top-k without masking.
    all_masked = torch.isinf(kept).all(dim=-1, keepdim=True)
    if all_masked.any():
        fallback = torch.full_like(orig, neg_inf)
        fallback.scatter_(-1, topk_idx, topk_vals)
        kept = torch.where(all_masked, fallback, kept)

    return kept


def _make_gate_hook(layer_idx: int, top_k: int):
    def hook(module, inp, out):
        mask = router_mask_var.get()
        if not mask:
            return out
        router_ids = mask.get(layer_idx)
        if not router_ids:
            return out
        if not torch.is_tensor(out):
            return out
        if ROUTER_MASK_MODE == "keep_top8":
            return _mask_keep_topk(out, router_ids, top_k)
        out = out.clone()
        out[..., list(router_ids)] = -1e9
        return out

    return hook

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/root/autodl-tmp/models"
MODEL_NAME = "qwen3-coder-30b"
MAX_MODEL_LEN = 163840  # Maximum context length (matches vLLM setting)
ENABLE_TORCH_COMPILE = False  # Disable compile to ensure router masking hooks are applied reliably

# Masking mode:
# - "replace_topk": mask logits then reselect top-k from remaining experts
# - "keep_top8": keep only original top-k, then mask within them and renormalize
DEFAULT_MASK_MODE = "keep_top8"

# Default router mask points (layer, router) from Image #1
DEFAULT_ROUTER_MASK_POINTS = {
    (30, 24),
    (34, 68),
    (45, 1),
    (47, 28),
}

# Global router mask points (settable at startup via CLI)
ROUTER_MASK_POINTS = set(DEFAULT_ROUTER_MASK_POINTS)
ROUTER_MASK_MODE = DEFAULT_MASK_MODE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None
router_mask_var = contextvars.ContextVar("router_mask", default=None)

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
            try:
                # vLLM >= 0.13 moved tool parsers to vllm.tool_parsers
                from vllm.tool_parsers.qwen3xml_tool_parser import (
                    StreamingXMLToolCallParser
                )
                tool_parser_class = StreamingXMLToolCallParser
                logger.info("Successfully imported vLLM tool parser (vllm.tool_parsers)")
            except ImportError:
                # Fallback for older vLLM layout
                from vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser import (
                    StreamingXMLToolCallParser
                )
                tool_parser_class = StreamingXMLToolCallParser
                logger.info("Successfully imported vLLM tool parser (entrypoints)")
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

    if ENABLE_TORCH_COMPILE:
        logger.info("Compiling model with torch.compile() (this may take a minute)...")
        try:
            model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=False,
            )
            logger.info("Model compiled successfully!")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}. Continuing without compilation.")
            logger.warning("This is not critical - the server will work, just slower.")

    # Register router-mask hooks (MoE gate)
    num_experts = getattr(model.config, "num_experts", None)
    top_k = getattr(model.config, "num_experts_per_tok", 8)
    if num_experts is None:
        logger.warning("Model config missing num_experts; router masking validation may be limited.")
    app.state.num_experts = num_experts
    app.state.router_top_k = top_k

    hooks = []
    try:
        for i, layer in enumerate(model.model.layers):
            gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
            if gate is not None:
                hooks.append(gate.register_forward_hook(_make_gate_hook(i, int(top_k))))
        app.state.router_hooks = hooks
        logger.info(f"Registered router mask hooks for {len(hooks)} layers.")
    except Exception as e:
        logger.warning(f"Failed to register router mask hooks: {e}")

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

    # Router mask from startup configuration
    mask = _build_layer_router_mask(ROUTER_MASK_POINTS)
    num_experts = getattr(app.state, "num_experts", None)
    top_k = getattr(app.state, "router_top_k", 8)
    if num_experts is not None:
        _validate_router_mask(mask, int(num_experts), int(top_k))

    token = router_mask_var.set(mask)
    try:
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
    finally:
        router_mask_var.reset(token)

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

    # Router mask from startup configuration
    mask = _build_layer_router_mask(ROUTER_MASK_POINTS)
    num_experts = getattr(app.state, "num_experts", None)
    top_k = getattr(app.state, "router_top_k", 8)
    if num_experts is not None:
        _validate_router_mask(mask, int(num_experts), int(top_k))

    # Generate with streaming
    token = router_mask_var.set(mask)
    try:
        with torch.no_grad():
            # Use model.generate with streaming
            # Note: We'll simulate streaming by generating all at once
            # then yielding chunks, since transformers doesn't have true streaming
            output_ids = model.generate(**gen_kwargs)
    finally:
        router_mask_var.reset(token)

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

    parser = argparse.ArgumentParser(description="Qwen3-Coder server with router mask intervention.")
    parser.add_argument(
        "router_mask",
        nargs="?",
        default=None,
        help="Mask points, e.g. \"{(30,24),(34,68),(45,1),(47,28)}\"",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["replace_topk", "keep_top8"],
        default=DEFAULT_MASK_MODE,
        help="Masking mode: replace_topk or keep_top8 (default).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.router_mask is not None:
        try:
            ROUTER_MASK_POINTS = _normalize_mask_value(args.router_mask)
        except Exception as e:
            raise SystemExit(f"Invalid router_mask: {e}")
    ROUTER_MASK_MODE = args.mask_mode

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
