# HumanEval 代码补全任务

本目录包含使用 Qwen3-Coder 模型对 HumanEval 数据集进行代码补全的脚本和配置。

## 目录结构

```
humaneval/
├── data/                           # 数据集目录
│   └── test-00000-of-00001.parquet # HumanEval 测试数据集
├── agentic/                        # Agentic 模式（使用 qwen CLI）
│   ├── agentic.sh                  # Agentic 模式执行脚本
│   └── project/                    # 164 个项目目录
│       ├── humaneval_0/
│       ├── humaneval_1/
│       └── ...
├── assisted/                       # FIM 模式（使用 vLLM API）
│   ├── fim_completion.py           # FIM 补全脚本
│   ├── output.json                 # 输出结果
│   └── completion_log.tsv          # 执行日志
└── README.md                       # 本文件
```

## 模式说明

### 1. Agentic 模式 (`agentic/`)

使用 `qwen` CLI 工具，让模型自主地调用文件读写工具来完成代码补全。

**特点**：
- 模型可以读取文件、分析代码、编辑文件
- 更接近真实开发场景
- 需要安装 qwen CLI 工具

### 2. FIM 模式 (`assisted/`)

使用 vLLM API，采用 Fill-In-Middle (FIM) 格式直接生成代码补全。

**特点**：
- 直接生成代码，速度更快
- 使用 vLLM OpenAI 兼容 API
- 适合批量处理

## 使用方法

### Agentic 模式

```bash
cd /root/autodl-tmp/xai/dataset/humaneval/agentic
./agentic.sh
```

**配置说明**（编辑 `agentic.sh`）：

```bash
# 并行任务数，根据机器性能调整
PARALLEL_JOBS=6

# 单个任务超时时间（秒）
TIMEOUT=300
```

**输出**：
- 每个项目目录下会生成 `qwen_output.json`
- 执行日志：`agentic/agentic_run_log.tsv`

### FIM 模式

**前置要求**：
1. 启动 vLLM 服务：
```bash
vllm serve qwen3-coder-30b \
  --port 8000 \
  --host 0.0.0.0
```

2. 安装依赖：
```bash
pip install openai pandas
```

**运行脚本**：
```bash
cd /root/autodl-tmp/xai/dataset/humaneval/assisted
python fim_completion.py
```

**配置说明**（编辑 `fim_completion.py`）：

```python
# vLLM API 配置
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "qwen3-coder-30b"

# 生成参数
MAX_TOKENS = 512
TEMPERATURE = 0.0
```

**输出**：
- `assisted/output.json` - 所有补全结果
  ```json
  {
    "humaneval_0": "生成的代码...",
    "humaneval_1": "生成的代码...",
    ...
  }
  ```
- `assisted/completion_log.tsv` - 执行日志

## FIM 格式说明

FIM (Fill-In-Middle) 是 Qwen3-Coder 支持的代码补全格式：

```
<|fim_prefix|>前缀代码<|fim_suffix|>后缀代码<|fim_middle|>
```

对于 HumanEval 任务：
- **prefix**: 函数签名 + 文档字符串
- **suffix**: 空（生成整个函数体）
- **模型输出**: 函数体实现代码

**示例**：
```python
<|fim_prefix|>def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    """<|fim_suffix|><|fim_middle|>

# 模型会生成：
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

## 依赖

- Python 3.8+
- pandas
- openai (vLLM API 客户端)
- qwen CLI (Agentic 模式)

## 参考

- [Qwen3-Coder GitHub](https://github.com/QwenLM/Qwen3-Coder)
- [HumanEval 数据集](https://github.com/openai/human-eval)
- [vLLM 文档](https://docs.vllm.ai/)
