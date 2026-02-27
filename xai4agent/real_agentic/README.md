# Real Tool-Use Experiment Workspace

This directory is for the **real tool-use** variant of the coding experiment.

Unlike pure text completion settings, this setup requires the model to actually:

1. read file content
2. write updated code back to disk
3. run the Python file

That is why this directory is named `real`: it captures real execution behavior of tools, not only final text output.

## Why `real` Exists

The older experiment modes (`agentic`, `assisted`) mainly focus on completion quality from prompt variants.
`real` was added to answer a different question:

- Is the model's tool-calling process itself correct and stable?
- What exact prompt did the model see right before each tool call?
- How do router/expert patterns look at those true tool-call decision points?

So this folder is both:

- an execution benchmark (`run.py`, `eval.py`)
- and a trace-analysis workspace (`extract_tool_prompts.py`, `analyze_*`, `verify_*`)

## Main Files

- `agent.py`
  - Defines tool implementations and agent builder.
  - Tools are restricted to:
    - `read_file`
    - `write_file`
    - `run` (Python file only)
  - Includes path safety checks and escaped-content decode handling for writes.

- `run.py`
  - Main real experiment runner over all `projects/humaneval_*`.
  - Enforces tool-use workflow via prompt instructions.
  - Saves:
    - `real_output.json` (includes `message_trace`)
    - `real_log.tsv`

- `eval.py`
  - Evaluates `projects/*/main.py` directly using HumanEval tests.
  - Saves `real_eval_results.json`.

- `extract_tool_prompts.py`
  - Reconstructs the exact tool-call prompts from `message_trace`.
  - Saves `real_tool_prompts.json` with rows:
    - `{id, prompt, sample_id, tool}`

- `analyze_tool_prompt_top8.py`
  - For one or more tools, runs forward on last token and aggregates top-8 router weights per layer.
  - Produces tool-specific heatmaps and npz outputs.

- `analyze_toolmode_point_activation.py`
  - Computes frequency/activation on selected tool-mode points for `read_file/write_file/run`.
  - Produces heatmaps and per-tool summary stats.

- `intersect_tool_points.py`
  - Computes intersection between tool-mode points and read/write/run top-8 points.
  - Saves intersection json and a heatmap.

- `verify_router_mask.py`
  - Verifies whether masked routers are removed from top-k in practice.
  - Outputs verification json.

- `make_tool_call_ppt.py`
  - Builds a review deck (`tool_call_prompt_review.pptx`) for debugging and communication.

## Data and Result Files

- `projects/`
  - Working directories (`humaneval_0` ... `humaneval_163`), each with `main.py`.
  - This is where the agent reads/writes/runs.

- `real_output.json`
  - Per-sample run output.
  - Includes:
    - `prompt`
    - `final_response`
    - `tool_calls`
    - `message_trace`

- `real_log.tsv`
  - Per-sample run log with timing and tool call count.

- `real_eval_results.json`
  - Per-sample evaluation pass/fail.

- `real_tool_prompts.json`
  - Extracted prompts at tool-call decision points.

- `verify_router_mask.json`
- `verify_router_mask_full.json`
  - Router mask verification reports.

## Typical Workflow

## 1) Run real tool-use experiment

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen

python /root/autodl-tmp/xai/exp/real/run.py \
  --project-root /root/autodl-tmp/xai/exp/real/projects \
  --output /root/autodl-tmp/xai/exp/real/real_output.json \
  --log /root/autodl-tmp/xai/exp/real/real_log.tsv \
  --max-steps 20
```

## 2) Evaluate resulting code files

```bash
python /root/autodl-tmp/xai/exp/real/eval.py \
  --project-root /root/autodl-tmp/xai/exp/real/projects \
  --parquet /root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet \
  --results /root/autodl-tmp/xai/exp/real/real_eval_results.json
```

## 3) Extract real tool-call prompts

```bash
python /root/autodl-tmp/xai/exp/real/extract_tool_prompts.py \
  --input /root/autodl-tmp/xai/exp/real/real_output.json \
  --output /root/autodl-tmp/xai/exp/real/real_tool_prompts.json \
  --agent /root/autodl-tmp/xai/exp/real/agent.py \
  --model /root/autodl-tmp/models/qwen3-coder-30b
```

## 4) Tool prompt router analysis (example)

```bash
python /root/autodl-tmp/xai/exp/real/analyze_tool_prompt_top8.py \
  --data_path /root/autodl-tmp/xai/exp/real/real_tool_prompts.json \
  --tools read_file \
  --device cuda
```

## 5) Tool-mode point activation analysis

```bash
python /root/autodl-tmp/xai/exp/real/analyze_toolmode_point_activation.py \
  --data_path /root/autodl-tmp/xai/exp/real/real_tool_prompts.json \
  --toolmode_npz /root/autodl-tmp/xai/output/humaneval/tables/01-21-09.47/expert_top8_sum.npz \
  --device cuda
```

## 6) Router-mask verification

```bash
python /root/autodl-tmp/xai/exp/real/verify_router_mask.py \
  --data_path /root/autodl-tmp/xai/exp/real/real_tool_prompts.json \
  --server_path /root/autodl-tmp/FastAPI/qwen3coder/server_intervene.py \
  --device cuda \
  --output /root/autodl-tmp/xai/exp/real/verify_router_mask_full.json
```

## Environment Notes

- Recommended environment: `qwen`
- Required service endpoint for LLM API:
  - default `http://127.0.0.1:8000/v1`
- `run.py` sets `MAX_TOKENS` to `65536` by default.

## Operational Notes

- `real` mode modifies `projects/*/main.py` in place by design.
- For strict reproducibility, regenerate clean `projects/` before reruns.
- `message_trace` is the canonical source for reconstructing true tool-call prompts.
- If you only need evaluation, `real_eval_results.json` is enough.
- If you need interpretability diagnostics, keep:
  - `real_output.json`
  - `real_tool_prompts.json`
  - analysis npz/png outputs under `/root/autodl-tmp/xai/output/humaneval`.

