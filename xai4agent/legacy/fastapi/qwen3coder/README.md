# FastAPI OpenAI-Compatible Server for Qwen3-Coder-30B

一个基于 FastAPI + Transformers 的 OpenAI 兼容服务器，专为 qwen-code 设计。

## 目录

- [快速开始](#快速开始)
- [功能特性](#功能特性)
- [性能对比](#性能对比)
- [配置说明](#配置说明)
- [API 使用](#api-使用)
- [qwen-code 集成](#qwen-code-集成)
- [故障排查](#故障排查)
- [vLLM 原生交互参考](#vllm-原生交互参考)
- [开发历史](#开发历史)
- [技术架构](#技术架构)

---

## 快速开始

### 启动服务器

```bash
cd /root/autodl-tmp/FastAPI/qwen3coder

# 方式 1: 使用快速启动脚本（推荐）
./quick_start.sh

# 方式 2: 后台运行
nohup python server.py > server.log 2>&1 &

# 方式 3: 直接运行
python server.py
```

### 验证运行

```bash
# 测试模型列表
curl http://127.0.0.1:8000/v1/models

# 运行完整测试
python test_client.py
```

### 查看日志

```bash
# 实时查看日志
tail -f server.log

# 查看最近错误
tail -50 server.log | grep ERROR
```

### 停止服务器

```bash
ps aux | grep "python server.py" | grep -v grep | awk '{print $2}' | xargs -r kill
```

---

## 功能特性

### ✅ 已实现

| 功能 | 状态 | 说明 |
|------|------|------|
| **GET /v1/models** | ✅ | 列出可用模型 |
| **POST /v1/chat/completions** | ✅ | 聊天补全 API |
| **非流式响应** | ✅ | 标准完整响应 |
| **流式响应** | ✅ | SSE 格式流式输出 |
| **工具调用** | ✅ | 使用 vLLM qwen3_xml 解析器 |
| **tool_choice="auto"** | ✅ | 自动工具调用支持 |
| **特殊 token 过滤** | ✅ | 过滤 `<|im_start|>` 等 |
| **增量 decode** | ✅ | TokenAccumulator 避免乱码 |
| **请求日志** | ✅ | 完整请求体记录 |

### 🔧 技术栈

- **Python 3.12+**
- **FastAPI** - Web 框架
- **Transformers** - 模型加载
- **PyTorch** - 深度学习框架
- **vLLM Tool Parser** - qwen3_xml 工具解析

### ⚙️ 默认配置

```python
MODEL_PATH = "/root/autodl-tmp/models/qwen3-coder-30b"  # 57GB bf16
MODEL_NAME = "qwen3-coder-30b"
MAX_MODEL_LEN = 163840
HOST = "127.0.0.1"
PORT = 8000
```

---

## 性能对比

### 与 vLLM 对比

| 指标 | vLLM | FastAPI Server | 说明 |
|------|------|----------------|------|
| **模型** | qwen3-coder-30b (bf16) | qwen3-coder-30b (bf16) | 完全相同 |
| **内存占用** | ~80 GB | ~57 GB | 更低内存 |
| **Prompt 吞吐量** | 1180 tok/s | ~50-100 tok/s | 慢 10-20 倍 |
| **Generation 吞吐量** | 24.6 tok/s | ~10-20 tok/s | 慢 2-3 倍 |
| **并发支持** | ✅ 支持 | ❌ 单请求 | 开发环境限制 |
| **max_model_len** | 163840 | 163840 | 完全相同 |
| **tool_call_parser** | qwen3_xml | qwen3_xml | 复用 vLLM |
| **真流式生成** | ✅ | ❌ 伪流式 | Transformers 限制 |

### 适用场景

**本服务器适用于：**
- ✅ 单用户开发/调试
- ✅ 功能验证和测试
- ✅ 内存受限环境
- ✅ 快速原型开发

**vLLM 适用于：**
- ✅ 生产环境高并发
- ✅ 高吞吐量需求
- ✅ 多用户服务

---

## 配置说明

### 模型路径

编辑 `server.py` 中的 `MODEL_PATH`：

```python
# 使用 bf16 模型（57GB，推荐）
MODEL_PATH = "/root/autodl-tmp/models/qwen3-coder-30b"

# 或使用 fp8 量化模型（30GB，需要修改 dtype）
MODEL_PATH = "/root/autodl-tmp/models/qwen3-coder-30b-fp8"
```

### 服务器设置

```bash
# 自定义 host/port
python server.py --host 0.0.0.0 --port 8080

# 或修改 server.py 中的配置
HOST = "0.0.0.0"  # 允许外部访问
PORT = 8080
```

### 环境变量

```bash
# 设置 PYTHONPATH 以导入 vLLM
export PYTHONPATH="/root/miniconda3/envs/qwen/lib/python3.12/site-packages:$PYTHONPATH"
```

---

## API 使用

### Python SDK

#### 安装

```bash
pip install openai
```

#### 基本对话

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"  # 不实际验证
)

response = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print(response.choices[0].message.content)
# 输出: 2+2 equals 4.
```

#### 流式输出

```python
stream = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=[{"role": "user", "content": "Count from 1 to 10"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
# 输出: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

#### 工具调用

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Tool: {tc.function.name}")
        print(f"Args: {tc.function.arguments}")
        # 输出:
        # Tool: get_weather
        # Args: {"city":"Beijing"}
```

#### 多轮对话

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Remember my favorite color: blue"}
]

response = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=messages
)

messages.append({"role": "assistant", "content": response.choices[0].message.content})
messages.append({"role": "user", "content": "What's my favorite color?"})

response = client.chat.completions.create(
    model="qwen3-coder-30b",
    messages=messages
)

print(response.choices[0].message.content)
# 输出: Your favorite color is blue.
```

### cURL

#### 非流式请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### 流式请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

#### 工具调用

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          }
        }
      }
    }]
  }'
```

### API 参考

#### GET /v1/models

列出可用模型。

**响应示例：**
```json
{
  "object": "list",
  "data": [{
    "id": "qwen3-coder-30b",
    "object": "model",
    "created": 1234567890,
    "owned_by": "qwen"
  }]
}
```

#### POST /v1/chat/completions

创建聊天补全。

**请求参数：**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | ✅ | 模型名称 |
| messages | array | ✅ | 对话消息列表 |
| stream | boolean | ❌ | 是否流式输出（默认 false）|
| temperature | number | ❌ | 采样温度（0-2）|
| max_tokens | integer | ❌ | 最大生成 token 数 |
| tools | array | ❌ | 工具定义列表 |
| tool_choice | string/object | ❌ | 工具选择策略（默认 "auto"）|

**消息格式：**
```json
{
  "role": "user|assistant|system|tool",
  "content": "string or array",
  "tool_calls": [...],  // assistant 消息可选
  "tool_call_id": "string"  // tool 消息必需
}
```

---

## qwen-code 集成

### 配置 settings.json

在 `/root/.qwen/settings.json` 中配置：

```json
{
  "security": {
    "auth": {
      "selectedType": "openai",
      "apiKey": "EMPTY",
      "baseUrl": "http://127.0.0.1:8000/v1"
    }
  },
  "model": {
    "name": "qwen3-coder-30b",
    "enableOpenAILogging": true,
    "openAILoggingDir": "/root/autodl-tmp/tmp"
  }
}
```

### 日志位置

qwen-code 的请求日志保存在：
```
/root/autodl-tmp/tmp/openai-*.json
```

查看最新日志：
```bash
ls -lt /root/autodl-tmp/tmp/openai-*.json | head -1
```

---

## 故障排查

### 常见问题

#### 1. 内存不足 (OOM)

**错误信息：**
```
CUDA out of memory
```

**解决方案：**
- 使用 fp8 量化模型（30GB vs 57GB）
- 降低 `max_tokens` 参数
- 减少输入消息长度
- 清理 GPU 缓存：`torch.cuda.empty_cache()`

#### 2. accelerate 库错误

**错误信息：**
```
ValueError: Using a device_map requires accelerate
```

**解决方案：**
- 已修复：使用 `.to("cuda")` 替代 `device_map="auto"`
- 无需安装 accelerate

#### 3. torch_dtype 弃用警告

**错误信息：**
```
torch_dtype is deprecated! Use dtype instead
```

**解决方案：**
- 已修复：改用 `dtype=torch.bfloat16`

#### 4. TypeError: Can only get item pairs from a mapping

**错误信息：**
```
TypeError: Can only get item pairs from a mapping
  at "<template>", line 87, in top-level template code
```

**原因：** 多种可能原因
1. Pydantic 模型未完全转换为 dict
2. `tool_call.arguments` 是 JSON 字符串但模板期望 dict

**解决方案：**
- 已修复：递归转换 Pydantic 模型为 dict
- 已修复：添加 `_postprocess_tool_calls()` 将 arguments 从 JSON 字符串转为 dict

#### 5. 流式工具调用 terminated

**错误信息：**
```
OpenAI API Streaming Error: terminated
```

**原因：** 每个 token 生成新的 tool_call_id

**解决方案：**
- 已修复：添加 `current_tool_call_id` 变量保持一致性

#### 6. 工具调用不工作

**可能原因：**
- vLLM 未正确安装
- PYTHONPATH 未设置
- 工具定义格式错误

**解决方案：**
```bash
# 检查 vLLM 安装
python -c "import vllm; print(vllm.__version__)"

# 设置 PYTHONPATH
export PYTHONPATH="/root/miniconda3/envs/qwen/lib/python3.12/site-packages:$PYTHONPATH"

# 检查工具定义格式
# 确保 parameters 是有效的 JSON Schema
```

#### 7. 工具参数类型错误（数组变字符串）

**症状：** `{"todos":"[...]"}` 被解析为字符串，而不是数组。  
**原因：** 工具解析器没有读取到参数类型，全部退化为 string。  
**解决方案：** 确保 vLLM 的工具解析器接收到 Pydantic 工具对象（而非 dict），并使用解析器输出的流式内容，避免 `<tool_call>` 泄露。

### 日志调试

#### 查看服务器日志

```bash
# 实时日志
tail -f server.log

# 查看最近 50 行
tail -50 server.log

# 只看错误
grep ERROR server.log

# 查看完整请求
grep "Request:" server.log | tail -10
```

#### 查看 qwen-code 日志

```bash
# 查看最新的 OpenAI 请求日志
ls -lt /root/autodl-tmp/tmp/openai-*.json | head -1 | xargs cat

# 查看最近的 5 个请求
ls -lt /root/autodl-tmp/tmp/openai-*.json | head -5 | xargs -I {} sh -c 'echo "=== {} ===" && cat {}'
```

### 性能优化

#### 当前限制

- **单请求处理** - 不支持并发
- **伪流式** - 生成完后分块发送，非真流式
- **无缓存** - 每次请求都重新处理

#### 可选优化

##### 1. 使用 vLLM（生产环境推荐）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/qwen3-coder-30b \
    --host 127.0.0.1 --port 8000 \
    --served-model-name qwen3-coder-30b \
    --max-model-len 163840 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

##### 2. 模型量化

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    ...
)
```

##### 3. 添加缓存层

对常见请求缓存结果以提升响应速度。

---

## vLLM 原生交互参考

### 项目目标

本项目的核心目标是 **用 Transformers + FastAPI 替代 vLLM** 来加载 Qwen3-Coder 模型，同时保持与 qwen-code 的完全兼容性。

**为什么替代 vLLM？**
- ✅ 更低的内存占用（57GB vs 80GB）
- ✅ 更简单的依赖关系（不需要完整的 vLLM）
- ✅ 更易于调试和定制
- ✅ 适合单用户开发环境

**保留什么？**
- ✅ 完全复用 vLLM 的 qwen3_xml 工具解析器
- ✅ 完全兼容 OpenAI API 格式
- ✅ 相同的 chat template 处理
- ✅ 相同的模型输出

### vLLM 实现解析

#### 1. 模型注册机制

vLLM 通过 `ModelRegistry` 管理所有模型架构：

**vLLM 源码：**
```python
# /vllm/model_executor/models/registry.py
_VLLM_MODELS = {
    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
}
```

**我们的实现：**
```python
# server.py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = model.to("cuda")  # 手动设备映射
```

#### 2. Qwen3 模型架构

**vLLM 源码：**
```python
# /vllm/model_executor/models/qwen3.py
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer
        )

class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
```

**关键特性：**
- 继承自 Qwen2Model
- 支持 Packed attention（QKV 合并）
- 支持 Pipeline Parallelism（PP）
- 支持 LoRA

**我们的实现：** 使用标准 Transformers 加载，自动支持这些特性。

#### 3. qwen3_xml 工具解析器

vLLM 的工具解析器是核心组件，我们**完全复用**：

**vLLM 源码：**
```python
# /vllm/entrypoints/openai/tool_parsers/qwen3xml_tool_parser.py

class StreamingXMLToolCallParser:
    """流式 XML 工具调用解析器"""

    # XML 标记定义
    tool_call_start_token = "<|tool_call|>"
    tool_call_end_token = "<|end_tool_call|>"
    function_start_token = "<function="
    function_end_token = "</function>"
    # ... 更多实现
```

**我们的实现（完全复用）：**
```python
# server.py - 从 vLLM 完全复用
from vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser import (
    StreamingXMLToolCallParser
)

tool_parser = StreamingXMLToolCallParser()
tool_parser.set_tools(request.tools)

# 流式解析
result = tool_parser.parse_single_streaming_chunks(delta_text)
```

**关键功能：**
- 解析 Qwen3 的 XML 格式工具调用
- 支持流式增量解析
- 自动处理参数类型转换
- 容错处理不完整的 XML

#### 4. Chat Template 处理

**vLLM 实现：**
```python
# /vllm/entrypoints/chat_utils.py
from transformers import PreTrainedTokenizer

# 使用 tokenizer 的 chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    tokenize=False,
    return_dict=False
)
```

**我们的实现：**
```python
# server.py - 完全相同
prompt = tokenizer.apply_chat_template(
    formatted_messages,  # 递归转换后的 messages
    tools=formatted_tools,  # 递归转换后的 tools
    add_generation_prompt=True,
    tokenize=False,
    return_dict=False
)
```

**关键转换：**
```python
def format_messages_for_template(messages: list[ChatMessage]) -> list[dict]:
    """递归转换 Pydantic 模型为 dict，适配 Jinja2 模板"""
    formatted = []
    for msg in messages:
        # 转换为 dict
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump(exclude_none=True)
        else:
            msg_dict = dict(msg)
        
        # 递归处理 tool_calls
        if msg_dict.get("tool_calls"):
            formatted_tool_calls = []
            for tc in msg_dict["tool_calls"]:
                if hasattr(tc, "model_dump"):
                    tc_dict = tc.model_dump(exclude_none=True)
                formatted_tool_calls.append(tc_dict)
            msg_dict["tool_calls"] = formatted_tool_calls
        
        formatted.append(msg_dict)
    return formatted
```

### vLLM 与本服务对比

| 组件 | vLLM | 本服务 | 说明 |
|------|------|--------|------|
| **模型加载** | ModelRegistry + 自定义类 | AutoModelForCausalLM | Transformers 标准 API |
| **设备映射** | device_map="auto" | .to("cuda") | 避免依赖 accelerate |
| **dtype 参数** | torch_dtype | dtype | 适配新版本 Transformers |
| **工具解析器** | StreamingXMLToolCallParser | 完全复用 | 从 vLLM 导入 |
| **Chat Template** | apply_chat_template | 完全相同 | Transformers 内置 |
| **流式生成** | 真·流式（KV Cache） | 伪流式（生成后分块）| Transformers 限制 |
| **并发处理** | 支持（AsyncLLMEngine） | 不支持 | 单用户环境 |

### 参考实现路径

**vLLM 源码位置：**
```
/root/miniconda3/envs/qwen/lib/python3.12/site-packages/vllm/
├── model_executor/models/
│   ├── qwen3.py                    # Qwen3 模型实现
│   └── registry.py                 # 模型注册
├── entrypoints/
│   ├── openai/
│   │   └── tool_parsers/
│   │       └── qwen3xml_tool_parser.py  # 工具解析器
│   └── chat_utils.py               # Chat 工具函数
└── transformers_utils/
    └── tokenizer.py                # Tokenizer 包装
```

**关键代码片段：**

1. **模型注册** - `registry.py:163-164`
2. **Qwen3 类** - `qwen3.py:258-299`
3. **工具解析器** - `qwen3xml_tool_parser.py:31-1317`
4. **Chat Template** - `chat_utils.py:1-100`

### 兼容性验证

本服务器已通过以下兼容性测试：

| 测试项 | vLLM 行为 | 本服务器 | 结果 |
|--------|-----------|----------|------|
| 工具调用格式 | XML | XML（复用解析器） | ✅ 完全相同 |
| 流式输出 | SSE | SSE（伪流式） | ✅ 格式兼容 |
| 非流式输出 | JSON | JSON | ✅ 完全相同 |
| Chat Template | Jinja2 | Jinja2 | ✅ 完全相同 |
| tool_choice="auto" | 支持 | 支持 | ✅ 完全相同 |
| 多轮对话 | 支持 | 支持 | ✅ 完全相同 |
| 特殊 token 过滤 | 自动 | 手动过滤 | ✅ 效果相同 |

---

## 开发历史

### 修复记录

#### 修复 1: device_map 需要 accelerate
**问题：** `ValueError: Using a device_map requires accelerate`
**解决：** 改用 `.to("cuda")` 手动设备映射
**文件：** server.py:238
**日期：** 2026-01-11

#### 修复 2: fp8 模型需要 accelerate
**问题：** `ImportError: Loading an FP8 quantized model requires accelerate`
**解决：** 改用 bf16 模型 `qwen3-coder-30b`（57GB）
**文件：** server.py:25
**日期：** 2026-01-11

#### 修复 3: torch_dtype deprecated
**问题：** `torch_dtype is deprecated! Use dtype instead`
**解决：** 改用 `dtype=torch.bfloat16`
**文件：** server.py:235
**日期：** 2026-01-11

#### 修复 4: tool_call_id 不一致
**问题：** 流式工具调用每个 token 生成新 ID
**解决：** 添加 `current_tool_call_id` 变量保持一致性
**文件：** server.py:549, 575-576
**日期：** 2026-01-11

#### 修复 5: Pydantic 模型未转换为 dict
**问题：** `TypeError: Can only get item pairs from a mapping`
**解决：**
- 递归转换 `format_tools_for_template` 中的 parameters
- 递归转换 `format_messages_for_template` 中的 tool_calls
- 使用 `model_dump(exclude_none=True)` 完全转换
**文件：** server.py:320-341, 286-335
**日期：** 2026-01-11

#### 修复 6: tool_call.arguments 类型不匹配
**问题：** qwen3 聊天模板期望 `tool_call.arguments` 为 dict，但 OpenAI 格式为 JSON 字符串
**错误信息：**
```
TypeError: Can only get item pairs from a mapping
  at "<template>", line 87, in top-level template code
  {%- for args_name, args_value in tool_call.arguments|items %}
```
**根本原因：**
- qwen code 发送包含之前工具调用历史的请求时，`arguments` 是 JSON 字符串（OpenAI 格式）
- qwen3 聊天模板使用 `tool_call.arguments|items` 遍历参数，期望 dict 类型
- 当模板尝试对字符串调用 `.items()` 时失败
**解决：**
- 添加 `_postprocess_tool_calls()` 函数，在调用 `apply_chat_template()` 前转换
- 将 assistant 消息中的 `tool_call.function.arguments` 从 JSON 字符串转为 dict
- 这与 vLLM 的 `_postprocess_messages()` 行为一致（chat_utils.py:1425-1443）
**文件：** server.py:373-396, 408
**日期：** 2026-01-11

### 已知限制

1. **伪流式** - Transformers 不支持真流式，生成完后分块发送
2. **单请求** - 不支持并发处理
3. **性能较慢** - 比 vLLM 慢 10-20 倍

### 测试状态

| 测试项 | 状态 | 说明 |
|--------|------|------|
| GET /v1/models | ✅ 通过 | - |
| 非流式 chat completions | ✅ 通过 | - |
| 流式 chat completions | ✅ 通过 | - |
| 工具调用（非流式） | ✅ 通过 | - |
| 工具调用（流式） | ✅ 通过 | - |
| 多轮对话 | ✅ 通过 | 含工具调用历史 |
| qwen-code 13工具测试 | ✅ 通过 | 修复后兼容 |

---

## 技术架构

### 核心组件

```
server.py
├── FastAPI app                  # Web 框架
├── Pydantic 模型                # OpenAI 兼容的数据模型
│   ├── ChatCompletionRequest    # 请求模型
│   ├── ChatCompletionResponse   # 响应模型
│   └── Tool / Function          # 工具定义模型
├── Tokenizer                    # 分词器
│   └── Chat template 支持       # 自动处理工具注入
├── Model Loader                 # 模型加载
│   └── AutoModelForCausalLM     # Transformers 模型
├── Tool Parser                  # 工具解析器
│   └── StreamingXMLToolCallParser  # vLLM qwen3_xml
├── Token Accumulator            # token 累加器
│   └── 避免增量 decode 乱码     # 完整解码后提取增量
├── Special Token Filter         # 特殊 token 过滤
│   └── 过滤 <|im_start|> 等     # 清理输出
└── SSE Formatter               # 流式格式化
    └── Server-Sent Events       # 兼容 OpenAI 格式
```

### 关键设计

#### 1. 增量 Decode 策略

```python
class TokenAccumulator:
    """避免增量 decode 导致的乱码"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.all_tokens = []
        self.last_decoded_len = 0

    def add_tokens(self, new_tokens):
        self.all_tokens.extend(new_tokens)
        full_text = self.tokenizer.decode(
            self.all_tokens,
            skip_special_tokens=True
        )
        delta_text = full_text[self.last_decoded_len:]
        self.last_decoded_len = len(full_text)
        return delta_text, full_text
```

#### 2. Pydantic 递归转换

```python
def format_messages_for_template(messages: list[ChatMessage]) -> list[dict]:
    """递归转换 Pydantic 模型为 dict"""
    formatted = []
    for msg in messages:
        # 转换为 dict
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump(exclude_none=True)
        else:
            msg_dict = dict(msg)

        # 递归处理 tool_calls
        if msg_dict.get("tool_calls"):
            formatted_tool_calls = []
            for tc in msg_dict["tool_calls"]:
                if hasattr(tc, "model_dump"):
                    tc_dict = tc.model_dump(exclude_none=True)
                    # 处理嵌套的 function 字段
                    if "function" in tc_dict and hasattr(tc_dict["function"], "model_dump"):
                        tc_dict["function"] = tc_dict["function"].model_dump(exclude_none=True)
                formatted_tool_calls.append(tc_dict)
            msg_dict["tool_calls"] = formatted_tool_calls

        formatted.append(msg_dict)
    return formatted
```

#### 3. 工具调用流式处理

```python
async def generate_stream(request: ChatCompletionRequest):
    current_tool_call_id = None  # 跟踪 ID

    for token_id in generated_ids.tolist():
        delta_text, full_text = accumulator.add_tokens([token_id])

        # 解析工具调用
        result = tool_parser.parse_single_streaming_chunks(delta_text)

        if result.tool_calls:
            # 保持 tool_call_id 一致
            if current_tool_call_id is None:
                current_tool_call_id = f"chatcmpl-tool-{uuid.uuid4().hex}"

            # 构建 delta
            delta = {
                "content": result.content,
                "tool_calls": [{
                    "index": 0,
                    "id": current_tool_call_id,
                    "function": {
                        "name": result.tool_calls[0].function.name,
                        "arguments": result.tool_calls[0].function.arguments
                    }
                }]
            }
            yield format_sse(delta)
```

### 文件结构

```
FastAPI/
└── qwen3coder/
    ├── server.py          # 主服务器
    ├── quick_start.sh     # 快速启动脚本
    ├── test_client.py     # 完整测试套件
    └── README.md          # 本文档
```

---

## 参考资料

### 相关文档

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [vLLM 文档](https://docs.vllm.ai/)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [Qwen 模型文档](https://huggingface.co/Qwen)

### 源码参考

- **vLLM Tool Parser**: `/root/miniconda3/envs/qwen/lib/python3.12/site-packages/vllm/entrypoints/openai/tool_parsers/qwen3xml_tool_parser.py`
- **vLLM Qwen3 Model**: `/root/miniconda3/envs/qwen/lib/python3.12/site-packages/vllm/model_executor/models/qwen3.py`
- **vLLM Model Registry**: `/root/miniconda3/envs/qwen/lib/python3.12/site-packages/vllm/model_executor/models/registry.py`
- **vLLM Chat Utils**: `/root/miniconda3/envs/qwen/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py`

---

## 许可证

本项目使用以下开源组件：

| 组件 | 许可证 |
|------|--------|
| Qwen3-Coder-30B | Apache 2.0 |
| vLLM tool parser | Apache 2.0 |
| Transformers | Apache 2.0 |
| FastAPI | MIT |
| PyTorch | BSD-style |

---

**最后更新：** 2026-01-11
**版本：** 1.0.0
**维护者：** Claude Code
