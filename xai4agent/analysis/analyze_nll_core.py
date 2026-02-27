#!/usr/bin/env python3
"""Core NLL / logit-lens analysis for paired (agentic vs assisted) hidden states.

Focus:
1) all canonical_nonws tokens
2) last canonical_nonws token
3) NLL / logit-lens / deltas per layer

Input expectations:
- pair_tokens_json: produced by token_build.py (contains aligned ids + canonical positions)
- hidden_states_dir: contains per-sample .pt files with full hidden_states for both variants

Outputs (under --out_dir):
- figures/nll_core/*.png
- tables/nll_core_layer_metrics.csv
- tables/nll_core_summary.json
- tables/nll_core_matrices.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_PAIR_TOKENS = "/root/autodl-tmp/xai/exp/pair/pair_tokens.json"
DEFAULT_HIDDEN = "/root/autodl-tmp/xai/output/humaneval/hidden_states"
DEFAULT_OUT = "/root/autodl-tmp/xai/output/humaneval"

EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Core NLL/logit-lens analysis (all vs last token).")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--pair_tokens_json", default=DEFAULT_PAIR_TOKENS)
    p.add_argument("--hidden_states_dir", default=DEFAULT_HIDDEN)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--max_tokens", type=int, default=None, help="Cap for all-token canonical_nonws positions")
    p.add_argument("--js_mode", default="none", choices=["none", "topk", "exact"])
    p.add_argument("--js_topk", type=int, default=512)
    return p.parse_args()


def load_json(path: Path) -> Any:
    try:
        import orjson

        return orjson.loads(path.read_bytes())
    except Exception:
        return json.loads(path.read_text())


def resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def get_module_by_path(root: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
    cur = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_final_norm(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    candidates = [
        "model.norm",
        "model.final_norm",
        "model.ln_f",
        "model.model.norm",
        "model.model.final_norm",
        "transformer.ln_f",
        "transformer.norm",
        "norm",
    ]
    for name in candidates:
        mod = get_module_by_path(model, name)
        if mod is not None:
            return mod
    return None


def find_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    head = model.get_output_embeddings()
    if head is not None:
        return head
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "output_layer"):
        return model.output_layer
    raise RuntimeError("Could not locate lm_head/output embeddings")


class LightHead:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        norm = find_final_norm(model)
        lm_head = find_lm_head(model)

        if norm is None:
            self.norm_type = "none"
            self.norm_weight = None
            self.norm_bias = None
            self.norm_eps = 1e-6
        else:
            nm = norm.__class__.__name__.lower()
            self.norm_type = "rms" if "rms" in nm else "layernorm"
            self.norm_weight = norm.weight.detach().to(device=device, dtype=dtype)
            self.norm_bias = (
                norm.bias.detach().to(device=device, dtype=dtype)
                if hasattr(norm, "bias") and norm.bias is not None
                else None
            )
            self.norm_eps = float(getattr(norm, "eps", 1e-6))

        self.lm_head_weight = lm_head.weight.detach().to(device=device, dtype=dtype)
        self.lm_head_bias = (
            lm_head.bias.detach().to(device=device, dtype=dtype)
            if hasattr(lm_head, "bias") and lm_head.bias is not None
            else None
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_norm(self, h: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "none":
            return h
        if self.norm_type == "rms":
            variance = h.pow(2).mean(-1, keepdim=True)
            h = h * torch.rsqrt(variance + self.norm_eps)
            if self.norm_weight is not None:
                h = h * self.norm_weight
            return h
        mean = h.mean(-1, keepdim=True)
        var = (h - mean).pow(2).mean(-1, keepdim=True)
        h = (h - mean) * torch.rsqrt(var + self.norm_eps)
        if self.norm_weight is not None:
            h = h * self.norm_weight
        if self.norm_bias is not None:
            h = h + self.norm_bias
        return h

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.lm_head_bias is None:
            return h @ self.lm_head_weight.t()
        return h @ self.lm_head_weight.t() + self.lm_head_bias



def gather_hidden(hidden: torch.Tensor, pred_positions: List[int]) -> torch.Tensor:
    pos = torch.tensor(pred_positions, dtype=torch.long)
    return hidden.index_select(1, pos)


def compute_nll_and_logp(
    head: LightHead,
    hidden_batch: torch.Tensor,
    target_ids: torch.Tensor,
    chunk_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = hidden_batch.shape[0]
    nll = torch.empty((n,), dtype=torch.float32, device="cpu")
    logp = torch.empty((n,), dtype=torch.float32, device="cpu")
    for i in range(0, n, chunk_size):
        h = hidden_batch[i : i + chunk_size].to(device)
        h = head.apply_norm(h)
        logits = head.logits(h)
        lp = torch.log_softmax(logits.float(), dim=-1)
        ids = target_ids[i : i + chunk_size].to(device)
        chosen = lp.gather(1, ids[:, None]).squeeze(1)
        nll[i : i + chunk_size] = (-chosen).detach().cpu()
        logp[i : i + chunk_size] = chosen.detach().cpu()
    return nll, logp


def compute_js_batch(
    head: LightHead,
    hidden_a: torch.Tensor,
    hidden_b: torch.Tensor,
    chunk_size: int,
    device: str,
    mode: str,
    topk: int,
) -> torch.Tensor:
    n = hidden_a.shape[0]
    out = torch.empty((n,), dtype=torch.float32, device="cpu")
    for i in range(0, n, chunk_size):
        ha = hidden_a[i : i + chunk_size].to(device)
        hb = hidden_b[i : i + chunk_size].to(device)
        la = head.logits(head.apply_norm(ha)).float()
        lb = head.logits(head.apply_norm(hb)).float()

        if mode == "exact":
            pa = torch.softmax(la, dim=-1)
            pb = torch.softmax(lb, dim=-1)
            m = 0.5 * (pa + pb)
            js = 0.5 * (torch.sum(pa * (pa / (m + EPS)).log(), dim=-1) + torch.sum(pb * (pb / (m + EPS)).log(), dim=-1))
            out[i : i + chunk_size] = js.detach().cpu()
            continue

        # topk union
        k = min(topk, la.shape[-1])
        ia = la.topk(k, dim=-1).indices
        ib = lb.topk(k, dim=-1).indices
        vals = torch.empty((la.shape[0],), dtype=torch.float32, device=la.device)
        for r in range(la.shape[0]):
            idx = torch.unique(torch.cat([ia[r], ib[r]], dim=0))
            xa = la[r].index_select(0, idx)
            xb = lb[r].index_select(0, idx)
            pa = torch.softmax(xa, dim=-1)
            pb = torch.softmax(xb, dim=-1)
            m = 0.5 * (pa + pb)
            vals[r] = 0.5 * (torch.sum(pa * (pa / (m + EPS)).log()) + torch.sum(pb * (pb / (m + EPS)).log()))
        out[i : i + chunk_size] = vals.detach().cpu()
    return out


def safe_mean(arrs: List[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.array([])
    return np.stack(arrs, axis=0).mean(axis=0)


def save_curve_two(a: np.ndarray, b: np.ndarray, title: str, ylabel: str, out: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(a, label="agentic")
    plt.plot(b, label="assisted")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def save_curve_single(a: np.ndarray, title: str, ylabel: str, out: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(a)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    out_base = Path(args.out_dir)
    figs_dir = out_base / "figures" / "nll_core"
    tables_dir = out_base / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(Path(args.pair_tokens_json))
    hidden_dir = Path(args.hidden_states_dir)
    if not hidden_dir.exists():
        raise SystemExit(f"hidden_states_dir not found: {hidden_dir}")

    pt_files = sorted(hidden_dir.glob("humaneval_*.pt"))
    if args.max_samples is not None:
        pt_files = pt_files[: args.max_samples]

    head = LightHead(args.model_path, device=device, dtype=(torch.float32 if device == "cpu" else torch.float16))

    curves: Dict[str, List[np.ndarray]] = {
        "nll_agentic_all": [],
        "nll_assisted_all": [],
        "delta_nll_all": [],
        "logitlens_agentic_all": [],
        "logitlens_assisted_all": [],
        "delta_logitlens_all": [],
        "nll_agentic_last": [],
        "nll_assisted_last": [],
        "delta_nll_last": [],
        "logitlens_agentic_last": [],
        "logitlens_assisted_last": [],
        "delta_logitlens_last": [],
    }
    if args.js_mode != "none":
        curves["js_all"] = []
        curves["js_last"] = []

    processed = 0
    skipped = 0

    for pt in pt_files:
        task_id = pt.stem
        entry = data.get(task_id)
        if entry is None:
            skipped += 1
            continue

        obj = torch.load(pt, map_location="cpu")
        agentic_hidden = obj.get("agentic", {}).get("hidden_states")
        assisted_hidden = obj.get("assisted", {}).get("hidden_states")
        token_meta = obj.get("token_meta", {})

        if agentic_hidden is None or assisted_hidden is None:
            skipped += 1
            continue
        if token_meta.get("save_mode") not in (None, "full"):
            skipped += 1
            continue

        token_ids = entry.get("agentic_ids") or entry.get("assisted_aligned_ids")
        if not token_ids:
            skipped += 1
            continue

        canonical = entry.get("canonical_non_ws_positions") or entry.get("canonical_nonws_positions") or []
        mask = entry.get("assisted_aligned_attention_mask")
        if mask:
            canonical = [p for p in canonical if 0 <= p < len(mask) and mask[p] == 1]
        canonical = [p for p in canonical if p > 0 and p < len(token_ids)]

        if args.max_tokens is not None and len(canonical) > args.max_tokens:
            canonical = canonical[: args.max_tokens]
        if not canonical:
            skipped += 1
            continue

        all_target_positions = canonical
        all_pred_positions = [p - 1 for p in all_target_positions]
        all_target_ids = [token_ids[p] for p in all_target_positions]

        all_ah = gather_hidden(agentic_hidden, all_pred_positions)
        all_bh = gather_hidden(assisted_hidden, all_pred_positions)
        layers, k_all, dim = all_ah.shape
        flat_a = all_ah.reshape(layers * k_all, dim)
        flat_b = all_bh.reshape(layers * k_all, dim)
        target_ids = torch.tensor(all_target_ids, dtype=torch.long).repeat(layers)

        nll_a, logp_a = compute_nll_and_logp(head, flat_a, target_ids, args.chunk_size, device)
        nll_b, logp_b = compute_nll_and_logp(head, flat_b, target_ids, args.chunk_size, device)

        nll_curve_a = nll_a.reshape(layers, k_all).mean(dim=1).numpy()
        nll_curve_b = nll_b.reshape(layers, k_all).mean(dim=1).numpy()
        log_curve_a = logp_a.reshape(layers, k_all).mean(dim=1).numpy()
        log_curve_b = logp_b.reshape(layers, k_all).mean(dim=1).numpy()

        curves["nll_agentic_all"].append(nll_curve_a)
        curves["nll_assisted_all"].append(nll_curve_b)
        curves["delta_nll_all"].append(nll_curve_a - nll_curve_b)
        curves["logitlens_agentic_all"].append(log_curve_a)
        curves["logitlens_assisted_all"].append(log_curve_b)
        curves["delta_logitlens_all"].append(log_curve_a - log_curve_b)

        if args.js_mode != "none":
            js_vals = compute_js_batch(head, flat_a, flat_b, args.chunk_size, device, args.js_mode, args.js_topk)
            curves["js_all"].append(js_vals.reshape(layers, k_all).mean(dim=1).numpy())

        # last token
        lp = all_target_positions[-1]
        last_pred = [lp - 1]
        last_id = [token_ids[lp]]

        last_ah = gather_hidden(agentic_hidden, last_pred)
        last_bh = gather_hidden(assisted_hidden, last_pred)
        flat_la = last_ah.reshape(layers, dim)
        flat_lb = last_bh.reshape(layers, dim)
        last_ids = torch.tensor(last_id, dtype=torch.long).repeat(layers)

        nll_la, logp_la = compute_nll_and_logp(head, flat_la, last_ids, args.chunk_size, device)
        nll_lb, logp_lb = compute_nll_and_logp(head, flat_lb, last_ids, args.chunk_size, device)

        nll_last_a = nll_la.numpy()
        nll_last_b = nll_lb.numpy()
        log_last_a = logp_la.numpy()
        log_last_b = logp_lb.numpy()

        curves["nll_agentic_last"].append(nll_last_a)
        curves["nll_assisted_last"].append(nll_last_b)
        curves["delta_nll_last"].append(nll_last_a - nll_last_b)
        curves["logitlens_agentic_last"].append(log_last_a)
        curves["logitlens_assisted_last"].append(log_last_b)
        curves["delta_logitlens_last"].append(log_last_a - log_last_b)

        if args.js_mode != "none":
            js_last = compute_js_batch(head, flat_la, flat_lb, args.chunk_size, device, args.js_mode, args.js_topk)
            curves["js_last"].append(js_last.numpy())

        processed += 1

    if processed == 0:
        raise SystemExit("No valid samples processed. Check hidden_states save_mode/fullness and pair tokens.")

    mean_curves = {k: safe_mean(v) for k, v in curves.items()}

    # plots
    save_curve_two(
        mean_curves["nll_agentic_all"],
        mean_curves["nll_assisted_all"],
        "NLL per layer (all canonical_nonws tokens)",
        "NLL",
        figs_dir / "nll_all_agentic_vs_assisted.png",
    )
    save_curve_single(
        mean_curves["delta_nll_all"],
        "Delta NLL per layer (agentic - assisted, all)",
        "Delta NLL",
        figs_dir / "delta_nll_all.png",
    )

    save_curve_two(
        mean_curves["nll_agentic_last"],
        mean_curves["nll_assisted_last"],
        "NLL per layer (last canonical_nonws token)",
        "NLL",
        figs_dir / "nll_last_agentic_vs_assisted.png",
    )
    save_curve_single(
        mean_curves["delta_nll_last"],
        "Delta NLL per layer (agentic - assisted, last)",
        "Delta NLL",
        figs_dir / "delta_nll_last.png",
    )

    save_curve_two(
        mean_curves["logitlens_agentic_all"],
        mean_curves["logitlens_assisted_all"],
        "Logit-lens score per layer (all canonical_nonws tokens)",
        "Mean log-prob(target)",
        figs_dir / "logitlens_all_agentic_vs_assisted.png",
    )
    save_curve_single(
        mean_curves["delta_logitlens_all"],
        "Delta logit-lens per layer (agentic - assisted, all)",
        "Delta log-prob(target)",
        figs_dir / "delta_logitlens_all.png",
    )

    save_curve_two(
        mean_curves["logitlens_agentic_last"],
        mean_curves["logitlens_assisted_last"],
        "Logit-lens score per layer (last canonical_nonws token)",
        "Mean log-prob(target)",
        figs_dir / "logitlens_last_agentic_vs_assisted.png",
    )
    save_curve_single(
        mean_curves["delta_logitlens_last"],
        "Delta logit-lens per layer (agentic - assisted, last)",
        "Delta log-prob(target)",
        figs_dir / "delta_logitlens_last.png",
    )

    if args.js_mode != "none":
        save_curve_single(
            mean_curves["js_all"],
            f"JS per layer (all canonical_nonws, mode={args.js_mode})",
            "JS",
            figs_dir / "js_all.png",
        )
        save_curve_single(
            mean_curves["js_last"],
            f"JS per layer (last token, mode={args.js_mode})",
            "JS",
            figs_dir / "js_last.png",
        )

    # csv
    csv_path = tables_dir / "nll_core_layer_metrics.csv"
    n_layers = len(mean_curves["nll_agentic_all"])
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "layer",
            "nll_agentic_all",
            "nll_assisted_all",
            "delta_nll_all",
            "nll_agentic_last",
            "nll_assisted_last",
            "delta_nll_last",
            "logitlens_agentic_all",
            "logitlens_assisted_all",
            "delta_logitlens_all",
            "logitlens_agentic_last",
            "logitlens_assisted_last",
            "delta_logitlens_last",
        ]
        if args.js_mode != "none":
            header += ["js_all", "js_last"]
        writer.writerow(header)

        for l in range(n_layers):
            row = [
                l,
                float(mean_curves["nll_agentic_all"][l]),
                float(mean_curves["nll_assisted_all"][l]),
                float(mean_curves["delta_nll_all"][l]),
                float(mean_curves["nll_agentic_last"][l]),
                float(mean_curves["nll_assisted_last"][l]),
                float(mean_curves["delta_nll_last"][l]),
                float(mean_curves["logitlens_agentic_all"][l]),
                float(mean_curves["logitlens_assisted_all"][l]),
                float(mean_curves["delta_logitlens_all"][l]),
                float(mean_curves["logitlens_agentic_last"][l]),
                float(mean_curves["logitlens_assisted_last"][l]),
                float(mean_curves["delta_logitlens_last"][l]),
            ]
            if args.js_mode != "none":
                row += [float(mean_curves["js_all"][l]), float(mean_curves["js_last"][l])]
            writer.writerow(row)

    npz_path = tables_dir / "nll_core_matrices.npz"
    np_payload = {k: np.stack(v, axis=0) for k, v in curves.items() if len(v) > 0}
    for k, v in mean_curves.items():
        np_payload[f"mean_{k}"] = v
    np.savez(npz_path, **np_payload)

    summary = {
        "processed_samples": processed,
        "skipped_samples": skipped,
        "js_mode": args.js_mode,
        "js_topk": args.js_topk,
        "device": device,
        "dtype": str(dtype),
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "inputs": {
            "pair_tokens_json": str(Path(args.pair_tokens_json).resolve()),
            "hidden_states_dir": str(Path(args.hidden_states_dir).resolve()),
            "model_path": args.model_path,
        },
        "outputs": {
            "figures_dir": str(figs_dir.resolve()),
            "layer_csv": str(csv_path.resolve()),
            "matrices_npz": str(npz_path.resolve()),
        },
    }
    summary_path = tables_dir / "nll_core_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Processed samples: {processed} (skipped={skipped})")
    print(f"Saved figures to: {figs_dir}")
    print(f"Saved layer metrics: {csv_path}")
    print(f"Saved matrices: {npz_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
