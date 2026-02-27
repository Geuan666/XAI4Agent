# Pair Experiment Workspace

This directory contains the paired experiment pipeline that compares two prompting styles on the same tasks:

- `agentic` prompt style
- `assisted` prompt style

The core idea is:

1. Build two prompts per sample (`pair_prompts.json`)
2. Align both prompts in token space (`pair_tokens.json`)
3. Run decode/eval and optional router-mask interventions
4. Analyze performance drops as masked router count increases

## Directory Purpose

Use this directory for:

- Building pair data (`pair_build.py`, `token_build.py`)
- Running forward hidden-state dumps (`pair_forward.py`)
- Running decode and evaluation (`pair_decode.py`, `pair_decode_eval.py`)
- Running masked decode and sweeps (`pair_decode_mask.py`, `pair_decode_mask_sweep.py`)

Do not use this directory as the final analysis sink. The main analysis outputs are usually written under:

- `/root/autodl-tmp/xai/output/humaneval`

## Current Important Files

- `pair_build.py`
  - Builds `pair_prompts.json` from assisted output + agentic message trace.
- `token_build.py`
  - Builds `pair_tokens.json` (and optional cache `pair_tokens.pt`) with alignment metadata.
- `pair_forward.py`
  - Runs model forward and stores hidden states to output directory.
- `pair_decode.py`
  - Baseline decode without router masking.
- `pair_decode_eval.py`
  - Evaluates decode outputs against HumanEval tests.
- `pair_decode_mask.py`
  - Decode with router masking; supports `replace_topk` and `keep_topk`.
- `pair_decode_mask_sweep.py`
  - Runs a top-N sweep by rewriting `DEFAULT_ROUTER_MASK_POINTS` in `pair_decode_mask.py`.
- `pair_decode_mask_sweep_agentic_last.py`
  - Builds ordering from `avg_last_agentic`, then launches sweep.
- `plot_keep_topk_curves.py`
  - Draws accuracy-vs-masked-router curves from sweep outputs.
- `pair_prompts.json`
  - Per-sample paired prompt text.
- `pair_tokens.json`
  - Per-sample tokenized/aligned inputs and metadata.
- `pair_tokens.pt`
  - Optional cache copy for faster loading.

## End-to-End Pipeline

### Step 1: Build paired prompts

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen

python /root/autodl-tmp/xai/exp/pair/pair_build.py \
  --parquet /root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet \
  --assisted /root/autodl-tmp/xai/exp/assisted/assisted_output.json \
  --agentic /root/autodl-tmp/xai/exp/agentic/agentic_output1.json \
  --agentic-fallback /root/autodl-tmp/xai/exp/agentic/agentic_output.json \
  --output /root/autodl-tmp/xai/exp/pair/pair_prompts.json
```

### Step 2: Build aligned tokens

```bash
python /root/autodl-tmp/xai/exp/pair/token_build.py \
  --pairs /root/autodl-tmp/xai/exp/pair/pair_prompts.json \
  --output /root/autodl-tmp/xai/exp/pair/pair_tokens.json
```

### Step 3: Forward hidden states (optional but needed for NLL/logit-lens analysis)

```bash
python /root/autodl-tmp/xai/exp/pair/pair_forward.py \
  --tokens /root/autodl-tmp/xai/exp/pair/pair_tokens.json \
  --model /root/autodl-tmp/models/qwen3-coder-30b \
  --output-dir /root/autodl-tmp/xai/output/humaneval/hidden_states \
  --log /root/autodl-tmp/xai/exp/pair/pair_forward.tsv \
  --device cuda \
  --dtype bfloat16 \
  --save-mode full \
  --overwrite
```

### Step 4: Baseline decode

```bash
python /root/autodl-tmp/xai/exp/pair/pair_decode.py \
  --prompts /root/autodl-tmp/xai/exp/pair/pair_prompts.json \
  --tokens /root/autodl-tmp/xai/exp/pair/pair_tokens.json \
  --output-dir /root/autodl-tmp/xai/output/humaneval/decode \
  --log /root/autodl-tmp/xai/exp/pair/pair_decode.tsv \
  --device cuda \
  --dtype bfloat16 \
  --overwrite
```

### Step 5: Baseline eval

```bash
python /root/autodl-tmp/xai/exp/pair/pair_decode_eval.py \
  --decode-dir /root/autodl-tmp/xai/output/humaneval/decode \
  --results /root/autodl-tmp/xai/output/humaneval/tables/pair_decode_eval_results.json
```

### Step 6: Single mask decode

```bash
python /root/autodl-tmp/xai/exp/pair/pair_decode_mask.py \
  --output-dir /root/autodl-tmp/xai/output/humaneval/decode_mask \
  --log /root/autodl-tmp/xai/exp/pair/pair_decode_mask.tsv \
  --device cuda \
  --mask-mode keep_topk \
  --overwrite
```

### Step 7: Sweep over top-N mask points

```bash
python /root/autodl-tmp/xai/exp/pair/pair_decode_mask_sweep.py \
  --points-json /root/autodl-tmp/xai/output/humaneval/tables/01-21-09.47/last_token_agentic_minus_assisted_points.json \
  --out-base /root/autodl-tmp/xai/output/humaneval/decode_mask_keep_topk_agentic_minus_assisted_full \
  --log-base /root/autodl-tmp/xai/exp/pair/keep_topk_agentic_minus_assisted_full \
  --results-dir /root/autodl-tmp/xai/output/humaneval/tables/keep_topk_agentic_minus_assisted_full \
  --mask-mode keep_topk \
  --device cuda \
  --overwrite
```

## Mask Modes

`pair_decode_mask.py` supports:

- `replace_topk`
  - Mask selected routers first, then re-select top-k from remaining routers.
- `keep_topk`
  - Keep the original top-k candidate set, then mask inside this set and renormalize.

## Inputs and Outputs

### Main Inputs

- HumanEval parquet:
  - `/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet`
- Assisted output:
  - `/root/autodl-tmp/xai/exp/assisted/assisted_output.json`
- Agentic output:
  - `/root/autodl-tmp/xai/exp/agentic/agentic_output1.json` (fallback to `agentic_output.json`)

### Main Outputs

- Pair data:
  - `pair_prompts.json`
  - `pair_tokens.json`
- Decode/eval artifacts:
  - Usually under `/root/autodl-tmp/xai/output/humaneval/...`
- Sweep summaries:
  - `pair_decode_mask_sweep_summary.json`
  - `pair_decode_mask_sweep_summary.csv`

## Current Directory State Notes

- Legacy `*.tsv` logs were intentionally removed during cleanup.
- Only script/code and core json artifacts are retained in this directory.
- Some historical empty folders may remain if they were used as experiment placeholders.

## Recommended Minimal Retention

If you want this directory to stay lightweight while preserving reproducibility, keep:

- `pair_build.py`
- `token_build.py`
- `pair_forward.py`
- `pair_decode.py`
- `pair_decode_eval.py`
- `pair_decode_mask.py`
- `pair_decode_mask_sweep.py`
- `pair_prompts.json` (optional, can rebuild)
- `pair_tokens.json`

You can usually regenerate all decode/eval artifacts from these files plus upstream assisted/agentic outputs.

