# XAI4Agent - AGENTS Guide

## 代码结构树（详略版）
```text
/root/autodl-tmp/XAI4Agent                        # 项目总根目录与入口
├── configs/                                      # 配置统一管理根目录
│   ├── models/                                   # 模型规格与路径配置层
│   │   ├── qwen3-8b.yaml                         # 8B模型默认推理参数
│   │   └── qwen3-coder-30b-a3b.yaml              # 30B模型兼容参数模板
│   ├── datasets/                                 # 数据集映射与字段定义
│   │   └── humaneval.yaml                        # HumanEval数据字段映射
│   ├── pipelines/                                # 流水线默认参数入口集
│   │   ├── pair_build.yaml                       # 配对提示构建参数模板
│   │   ├── token_build.yaml                      # 配对token对齐参数模板
│   │   ├── pair_forward.yaml                     # 前向抽隐状态参数模板
│   │   ├── pair_decode.yaml                      # 双分支解码参数模板
│   │   ├── router_ablation.yaml                  # 路由消融实验参数模板
│   │   └── attribution_patching.yaml             # 归因补丁实验参数模板
│   └── runtime/                                  # 服务端运行时配置集合
│       ├── local_fastapi.yaml                    # FastAPI本地服务配置
│       └── local_vllm.yaml                       # vLLM本地服务配置模板
├── experiments/                                  # 实验输出时间戳目录根
│   ├── pair/                                     # pair链路运行结果归档
│   ├── router_ablation/                          # 路由消融结果与报表区
│   ├── attribution_patching/                     # 归因补丁实验输出目录
│   └── smoke/                                    # 快速冒烟测试结果区
├── artifacts/                                    # 汇总产物目录可选使用
│   └── ...                                       # 图表日志表格细节略写
├── references/                                   # 方法论文与说明文档区
├── scripts/                                      # 辅助脚本预留目录位
├── idea.md                                       # 研究思路与计划草稿
├── AGENTS.md                                     # 本地协作规范说明书
└── xai4agent/                                    # 主代码包命名空间根
    ├── pipelines/                                # 实验脚本主入口集合
    │   ├── pair_build.py                         # 构建agentic/assisted配对
    │   ├── token_build.py                        # 对齐并生成pair_tokens
    │   ├── pair_forward.py                       # 前向提取隐藏态数据
    │   ├── pair_decode.py                        # 双分支无mask解码入口
    │   ├── pair_decode_eval.py                   # 双分支解码评测入口
    │   ├── pair_decode_mask.py                   # 指定router屏蔽解码
    │   ├── pair_decode_mask_sweep.py             # top-N路由消融扫描脚本
    │   ├── agentic/                              # agentic范式脚本目录
    │   │   ├── agent.py                          # read_file工具agent定义
    │   │   ├── generate_agentic_projects.py      # 生成任务工程数据目录
    │   │   ├── agentic_run.py                    # agentic主执行入口脚本
    │   │   ├── agentic_run1.py                   # 带隐藏态记录执行脚本
    │   │   └── agentic_eval.py                   # agentic结果评测脚本
    │   ├── assisted/                             # assisted范式脚本目录
    │   │   ├── assisted_run.py                   # assisted主执行入口脚本
    │   │   └── assisted_eval.py                  # assisted结果评测脚本
    │   ├── real/                                 # 三工具真实链路目录
    │   │   ├── agent.py                          # read/write/run工具定义
    │   │   ├── run.py                            # real链路主执行入口脚本
    │   │   ├── eval.py                           # real结果评测入口脚本
    │   │   ├── extract_tool_prompts.py           # 提取tool调用提示脚本
    │   │   ├── verify_router_mask.py             # 验证router屏蔽有效性
    │   │   └── ...                               # 其余分析脚本按需使用
    ├── analysis/                                 # 可解释分析脚本集合
    │   ├── core/                                 # 核心分析主流程目录
    │   └── model/                                # 各模型分析适配目录
    ├── data/                                     # 数据集与基准脚本目录
    │   ├── dataset/                              # Humaneval/MBPP等脚本
    │   └── benchmark/                            # BFCL等基准资料目录
    ├── serving/                                  # 推理服务代码目录层
    │   ├── fastapi/                              # FastAPI服务适配代码
    │   └── vllm/                                 # vLLM服务与启动脚本
    ├── legacy/                                   # 原仓库快照仅回溯用
    ├── utils/                                    # 通用工具预留目录位
    └── interventions/                            # 干预实验预留目录位
```

## 启动 vLLM（8B + hermes）

对于启动参数`--tool-call-parser`，若是qwen3-8B则用`hermes`；若是qwen3coder-30B-A3b则用`qwen3_xml`。目前默认不开启thinking模式。

## 当前选择qwen3-8B与humaneval数据进行实验

## 外部依赖路径
- 旧实验代码（仅兜底兼容）：`/root/autodl-tmp/xai`
- FastAPI：`/root/autodl-tmp/FastAPI/qwen3coder`
- vLLM 运行配置：`/root/autodl-tmp/XAI4Agent/configs/runtime/local_vllm.yaml`
- 30B 模型：`/root/autodl-tmp/models`
- 8B 模型：`/root/autodl-tmp/qwen3-8B`

## 6. 环境 Conda：`qwen`，Python：`/root/miniconda3/envs/qwen/bin/python`

## 7. 运行注意事项
2. 默认端口常用 `8000`，启动前先查占用。
6. `xai4agent/serving/fastapi/qwen3coder/*` 为当前迁移后的服务代码；`legacy/*` 仅用于回溯。
7. Pair 主链路脚本（`pair_build/token_build/pair_forward/pair_decode/pair_decode_eval`）已内置时间戳输出。
8. 直接运行 `python 脚本.py` 即可；不传 `--run-id` 时会自动写入时间戳目录，并自动优先读取最近一次 run 的输入文件。


## 9. VPN代理（翻墙，访问国外资源）
- 程序：`/opt/clash/mihomo`，配置：`/etc/clash/config.yaml`，HTTP：`127.0.0.1:7890`，SOCKS5：`127.0.0.1:7891`

翻墙注意检查启动，并使用环境变量：export http_proxy=http://127.0.0.1:7890 export https_proxy=http://127.0.0.1:7890
