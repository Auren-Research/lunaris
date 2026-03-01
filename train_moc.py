from __future__ import annotations

import argparse
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from model_moc import LunarisCodex, LunarisCodexConfig
except Exception:
    from model import LunarisCodex, LunarisCodexConfig

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class TrainConfig:
    """Training configuration loaded from YAML."""

    model: LunarisCodexConfig = field(default_factory=LunarisCodexConfig)
    data_dir: str = "data"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    max_steps: int = 600_000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    grad_clip: float = 1.0
    device: str = "cuda"
    compile_model: bool = True
    out_dir: str = "checkpoints"
    log_interval: int = 20
    save_interval: int = 1000
    save_latest_always: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    wandb_project: Optional[str] = "lunaris-codex-moc"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    val_interval: int = 500
    val_batches: int = 50
    early_stopping_patience: int = 0
    log_routing_every: int = 1
    rich_terminal: bool = True
    save_best: bool = True

    @property
    def sequence_length(self) -> int:
        """Alias used by dataset setup."""
        return int(self.model.max_seq_len)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Build TrainConfig from YAML and force routing diagnostics on."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid YAML root (expected mapping): {config_path}")

        model_cfg_raw = payload.pop("model", {})
        if not isinstance(model_cfg_raw, dict):
            raise ValueError("`model` must be a mapping in YAML config")
        model_cfg = LunarisCodexConfig(**model_cfg_raw)

        # Always on for training diagnostics.
        model_cfg.return_routing_diagnostics = True
        model_cfg.track_routing_stats = True

        payload["model"] = model_cfg
        cfg = cls(**payload)

        # Normalize numeric user inputs.
        cfg.learning_rate = float(cfg.learning_rate)
        cfg.weight_decay = float(cfg.weight_decay)
        cfg.beta1 = float(cfg.beta1)
        cfg.beta2 = float(cfg.beta2)
        cfg.grad_clip = float(cfg.grad_clip)
        cfg.warmup_steps = int(cfg.warmup_steps)
        cfg.max_steps = int(cfg.max_steps)
        cfg.batch_size = int(cfg.batch_size)
        cfg.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        cfg.num_epochs = int(cfg.num_epochs)
        cfg.log_interval = int(cfg.log_interval)
        cfg.save_interval = int(cfg.save_interval)
        cfg.num_workers = int(cfg.num_workers)
        cfg.prefetch_factor = int(cfg.prefetch_factor)
        cfg.val_interval = int(cfg.val_interval)
        cfg.val_batches = int(cfg.val_batches)
        cfg.early_stopping_patience = int(cfg.early_stopping_patience)
        cfg.log_routing_every = max(1, int(cfg.log_routing_every))

        return cfg


class ShardDataset(Dataset):
    """Memory-mapped token dataset over .npy shards."""

    def __init__(self, data_dir: str | Path, sequence_length: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sequence_length = int(sequence_length)

        self.shards = sorted(self.data_dir.glob("*.npy"))
        if len(self.shards) == 0:
            raise ValueError(f"No .npy shards found in: {self.data_dir}")

        self.mmap_shards: List[np.ndarray] = [np.load(str(p), mmap_mode="r") for p in self.shards]
        self.shard_lengths = [int(len(x)) for x in self.mmap_shards]
        self.total_tokens = int(sum(self.shard_lengths))
        self.total_samples = self.total_tokens // self.sequence_length
        self.cumulative_lengths = np.cumsum(self.shard_lengths)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (x, y, valid_len_y) for one fixed-length training sample."""
        seq_len = self.sequence_length
        token_start = idx * seq_len
        shard_idx = int(np.searchsorted(self.cumulative_lengths, token_start, side="right"))
        local_start = token_start if shard_idx == 0 else token_start - int(self.cumulative_lengths[shard_idx - 1])

        needed = seq_len + 1
        current = self.mmap_shards[shard_idx]
        current_len = self.shard_lengths[shard_idx]

        if local_start + needed <= current_len:
            seq = current[local_start : local_start + needed]
        else:
            first = current[local_start:]
            remain = needed - len(first)
            if shard_idx + 1 < len(self.mmap_shards):
                second = self.mmap_shards[shard_idx + 1][:remain]
                seq = np.concatenate((first, second))
            else:
                seq = first

        original_len = int(len(seq))
        valid_len_y = int(max(0, min(seq_len, original_len - 1)))

        if original_len < needed:
            pad_len = needed - original_len
            pad_val = np.array(0, dtype=seq.dtype)
            seq = np.pad(seq, (0, pad_len), mode="constant", constant_values=pad_val)

        seq_t = torch.from_numpy(seq)
        x = seq_t[:-1]
        y = seq_t[1:]
        return x, y, valid_len_y


def get_lr(step: int, config: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to 1% base LR."""
    if step < config.warmup_steps:
        return config.learning_rate * (step / max(1, config.warmup_steps))
    if step >= config.max_steps:
        return config.learning_rate * 0.01
    ratio = (step - config.warmup_steps) / max(1, (config.max_steps - config.warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return (config.learning_rate * 0.01) + coeff * (config.learning_rate * 0.99)


def unwrap_model_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip common wrappers from checkpoint keys."""
    out: Dict[str, Any] = {}
    prefixes = ("_orig_mod.module.", "module.", "_orig_mod.")
    for k, v in state_dict.items():
        new_key = k
        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p) :]
                break
        out[new_key] = v
    return out


def to_float(x: Any) -> float:
    """Convert scalar tensor-like values to Python float."""
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


def apply_ignore_mask(y: torch.Tensor, valid_len_y: Optional[torch.Tensor | Sequence[int]]) -> torch.Tensor:
    """Apply ignore_index=-1 to padded targets on GPU."""
    if valid_len_y is None:
        return y

    if not torch.is_tensor(valid_len_y):
        valid_len_y = torch.as_tensor(valid_len_y, device=y.device)
    valid_len_y = valid_len_y.to(device=y.device, dtype=torch.long)

    t = torch.arange(y.shape[1], device=y.device).unsqueeze(0)
    pad_mask = t >= valid_len_y.unsqueeze(1)
    if torch.any(pad_mask):
        y = y.clone()
        y[pad_mask] = -1
    return y


def compute_active_params_per_token(model: LunarisCodex) -> Tuple[int, int]:
    """Estimate active parameters per token under top-k expert routing."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction = 0
    for block in model.transformer.h:
        if not getattr(block, "is_moe", False):
            continue
        ff = block.feed_forward
        if len(ff.experts) == 0:
            continue
        per_expert = sum(p.numel() for p in ff.experts[0].parameters() if p.requires_grad)
        n_experts = int(getattr(ff, "n_experts", len(ff.experts)))
        top_k = int(getattr(ff, "top_k", 1))
        reduction += max(0, (n_experts - top_k)) * per_expert
    return int(total - reduction), int(total)


def format_eta(seconds: float) -> str:
    """Format ETA in h/m style."""
    sec = max(0, int(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h}h {m:02d}m"


def gini_coefficient(x: torch.Tensor) -> float:
    """Compute Gini coefficient for non-negative values."""
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return 0.0
    total = x.sum().item()
    if total <= 0:
        return 1.0
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    cumsum = torch.cumsum(x_sorted, dim=0)
    g = (n + 1 - 2.0 * (cumsum.sum().item() / cumsum[-1].item())) / n
    return float(max(0.0, min(1.0, g)))


def supports_color() -> bool:
    """Return True if stdout supports ANSI colors."""
    return sys.stdout.isatty()


def colorize(text: str, level: str) -> str:
    """Color text for terminal readability."""
    if not supports_color():
        return text
    code = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
    }.get(level, "")
    reset = "\033[0m" if code else ""
    return f"{code}{text}{reset}"


def register_gamma_hooks(raw_model: LunarisCodex) -> Tuple[Dict[int, float], List[Any]]:
    """Register hooks to capture sigmoid(fuse_gate(x)) mean per MoE layer."""
    gamma_tracker: Dict[int, float] = {}
    handles: List[Any] = []

    def make_hook(layer_idx: int):
        def hook(_: torch.nn.Module, __: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            try:
                gamma_tracker[layer_idx] = float(torch.sigmoid(output).detach().float().mean().item())
            except Exception:
                pass

        return hook

    for i, block in enumerate(raw_model.transformer.h):
        ff = getattr(block, "feed_forward", None)
        fuse_gate = getattr(ff, "fuse_gate", None)
        if isinstance(fuse_gate, torch.nn.Linear):
            handles.append(fuse_gate.register_forward_hook(make_hook(i)))

    return gamma_tracker, handles


def parse_debug_payload(debug_payload: Any) -> Tuple[Optional[List[Dict[str, torch.Tensor]]], Optional[List[torch.Tensor]]]:
    """Extract routing diagnostics and expert indices from model debug payload."""
    if debug_payload is None:
        return None, None

    if isinstance(debug_payload, dict):
        routing = debug_payload.get("routing_diagnostics", None)
        experts = debug_payload.get("expert_indices", None)
        if isinstance(experts, list) and len(experts) > 0 and isinstance(experts[0], list):
            experts = experts[0]
        if not isinstance(routing, list):
            routing = None
        if not isinstance(experts, list):
            experts = None
        return routing, experts

    if isinstance(debug_payload, list) and len(debug_payload) > 0 and isinstance(debug_payload[0], list):
        return None, debug_payload[0]

    return None, None


def init_routing_entry(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Create accumulator entry for one routing layer."""
    req = torch.as_tensor(diag["requested_hist"]).detach().float().cpu()
    kept = torch.as_tensor(diag["kept_hist"]).detach().float().cpu()
    return {
        "requested_hist": torch.zeros_like(req),
        "kept_hist": torch.zeros_like(kept),
        "drop_sum": 0.0,
        "entropy_sum": 0.0,
        "capacity_sum": 0.0,
        "reasoning_sum": 0.0,
        "collab_sum": 0.0,
        "count": 0,
    }


def update_routing_window(
    debug_payload: Any,
    routing_window: Dict[int, Dict[str, Any]],
    agreement_window: Dict[Tuple[int, int], Tuple[float, int]],
    layer0_indices_window: List[torch.Tensor],
    reasoning_samples: List[float],
    collab_samples: List[float],
    gamma_tracker: Dict[int, float],
    gamma_window: Dict[int, List[float]],
) -> None:
    """Accumulate routing/debug metrics for one forward pass."""
    routing_diags, expert_indices = parse_debug_payload(debug_payload)

    if routing_diags is not None:
        for layer_idx, diag in enumerate(routing_diags):
            if not isinstance(diag, dict):
                continue
            if layer_idx not in routing_window:
                routing_window[layer_idx] = init_routing_entry(diag)

            entry = routing_window[layer_idx]
            req = torch.as_tensor(diag["requested_hist"]).detach().float().cpu()
            kept = torch.as_tensor(diag["kept_hist"]).detach().float().cpu()

            entry["requested_hist"] += req
            entry["kept_hist"] += kept
            entry["drop_sum"] += to_float(diag.get("drop_rate", 0.0))
            entry["entropy_sum"] += to_float(diag.get("router_entropy", 0.0))
            entry["capacity_sum"] += to_float(diag.get("capacity_per_expert", 0.0))
            entry["reasoning_sum"] += to_float(diag.get("avg_reasoning_steps", 0.0))
            entry["collab_sum"] += to_float(diag.get("avg_collab_steps", 0.0))
            entry["count"] += 1

            reasoning_samples.append(to_float(diag.get("avg_reasoning_steps", 0.0)))
            collab_samples.append(to_float(diag.get("avg_collab_steps", 0.0)))

    if expert_indices is not None and len(expert_indices) > 0:
        try:
            layer0_indices_window.append(expert_indices[0].detach().cpu())
        except Exception:
            pass

        for i in range(len(expert_indices) - 1):
            try:
                top1_i = expert_indices[i][:, :, 0]
                top1_j = expert_indices[i + 1][:, :, 0]
                agreement = float((top1_i == top1_j).float().mean().item())
                prev_sum, prev_count = agreement_window.get((i, i + 1), (0.0, 0))
                agreement_window[(i, i + 1)] = (prev_sum + agreement, prev_count + 1)
            except Exception:
                continue

    for layer_idx, gamma_val in gamma_tracker.items():
        gamma_window.setdefault(layer_idx, []).append(float(gamma_val))


def summarize_routing(
    routing_window: Dict[int, Dict[str, Any]],
    agreement_window: Dict[Tuple[int, int], Tuple[float, int]],
    gamma_window: Dict[int, List[float]],
    dead_expert_streaks: Dict[Tuple[int, int], int],
    model_cfg: LunarisCodexConfig,
) -> Dict[str, Any]:
    """Compute aggregated routing statistics for one log window."""
    layer_ids = sorted(routing_window.keys())
    per_layer: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []

    for layer_idx in layer_ids:
        entry = routing_window[layer_idx]
        count = max(1, int(entry["count"]))
        kept_hist: torch.Tensor = entry["kept_hist"]
        requested_hist: torch.Tensor = entry["requested_hist"]

        util = kept_hist / kept_hist.sum().clamp_min(1.0)
        gini = gini_coefficient(util)

        dead_count = 0
        for expert_idx in range(int(kept_hist.numel())):
            key = (layer_idx, expert_idx)
            if float(kept_hist[expert_idx].item()) <= 0.0:
                dead_expert_streaks[key] = dead_expert_streaks.get(key, 0) + 1
            else:
                dead_expert_streaks[key] = 0

            if dead_expert_streaks[key] >= 3:
                dead_count += 1
                if dead_expert_streaks[key] == 3:
                    warnings.append(f"Layer {layer_idx}: expert {expert_idx} has received 0 tokens for 3 log windows")

        per_layer[layer_idx] = {
            "requested_hist": requested_hist,
            "kept_hist": kept_hist,
            "utilization": util,
            "drop_rate": float(entry["drop_sum"] / count),
            "router_entropy": float(entry["entropy_sum"] / count),
            "capacity_per_expert": float(entry["capacity_sum"] / count),
            "avg_reasoning_steps": float(entry["reasoning_sum"] / count),
            "avg_collab_steps": float(entry["collab_sum"] / count),
            "gini": gini,
            "dead_experts": dead_count,
        }

    agreements: Dict[str, float] = {}
    for (i, j), (sum_val, cnt) in agreement_window.items():
        if cnt > 0:
            agreements[f"{i}_{j}"] = float(sum_val / cnt)

    gamma_means: Dict[int, float] = {}
    for layer_idx, values in gamma_window.items():
        if len(values) > 0:
            gamma_means[layer_idx] = float(sum(values) / len(values))

    avg_reasoning = 0.0
    avg_collab = 0.0
    efficiency = 0.0
    if len(layer_ids) > 0:
        avg_reasoning = float(sum(per_layer[i]["avg_reasoning_steps"] for i in layer_ids) / len(layer_ids))
        avg_collab = float(sum(per_layer[i]["avg_collab_steps"] for i in layer_ids) / len(layer_ids))

        max_reason = max(1, int(getattr(model_cfg, "n_reasoning_steps", 1)))
        max_collab = int(getattr(model_cfg, "moc_collab_steps", 0))
        if not bool(getattr(model_cfg, "use_moc_collab", True)):
            max_collab = 0

        denom = len(layer_ids) * max(1, (max_reason + max_collab))
        used = sum(per_layer[i]["avg_reasoning_steps"] + per_layer[i]["avg_collab_steps"] for i in layer_ids)
        efficiency = 1.0 - (used / float(denom))
        efficiency = max(0.0, min(1.0, efficiency))

    return {
        "layer_ids": layer_ids,
        "per_layer": per_layer,
        "agreements": agreements,
        "gamma_means": gamma_means,
        "avg_reasoning": avg_reasoning,
        "avg_collab": avg_collab,
        "efficiency": efficiency,
        "warnings": warnings,
    }


def select_routing_layers(layer_ids: List[int]) -> List[int]:
    """Pick first/mid/last layers for compact terminal display."""
    if len(layer_ids) <= 2:
        return layer_ids
    mid = layer_ids[len(layer_ids) // 2]
    out = [layer_ids[0], mid, layer_ids[-1]]
    dedup: List[int] = []
    for x in out:
        if x not in dedup:
            dedup.append(x)
    return dedup


def cooccurrence_figure(layer_indices: List[torch.Tensor], n_experts: int) -> Optional[Any]:
    """Create layer-0 utilization + co-occurrence visualization."""
    if plt is None or n_experts <= 0 or len(layer_indices) == 0:
        return None

    try:
        all_idx = torch.cat(layer_indices, dim=0)  # [N, T, K]
        k = int(all_idx.shape[-1])
        if k < 1:
            return None

        flat = all_idx.reshape(-1)
        util_counts = torch.bincount(flat, minlength=n_experts).float()
        util = (util_counts / util_counts.sum().clamp_min(1.0)).cpu().numpy()

        cooc = torch.zeros((n_experts, n_experts), dtype=torch.float32)
        if k >= 2:
            idx2d = all_idx.view(-1, k)
            for i in range(k):
                for j in range(i + 1, k):
                    u = idx2d[:, i]
                    v = idx2d[:, j]
                    pairs = (u * n_experts + v)
                    counts = torch.bincount(pairs, minlength=n_experts * n_experts).view(n_experts, n_experts)
                    cooc += counts.float()
                    cooc += counts.t().float()

        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.bar(np.arange(n_experts), util)
        ax0.set_title("Layer 0 Expert Utilization")
        ax0.set_xlabel("Expert")
        ax0.set_ylabel("Fraction")

        ax1 = fig.add_subplot(gs[0, 1])
        im = ax1.imshow(cooc.cpu().numpy(), aspect="auto", interpolation="nearest")
        ax1.set_title("Layer 0 Expert Pair Co-occurrence")
        ax1.set_xlabel("Expert v")
        ax1.set_ylabel("Expert u")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        return fig
    except Exception:
        return None


def make_autocast_context(device_type: str, amp_dtype: Optional[torch.dtype]) -> Any:
    """Build autocast context manager for the current precision mode."""
    if device_type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    raw_model: LunarisCodex,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    step: int,
    is_master: bool,
    gamma_tracker: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Run fixed-batch validation and return averaged metrics."""
    if len(val_loader) == 0:
        return {}

    was_training = model.training
    model.eval()

    saved_gamma: Dict[int, float] = {}
    if gamma_tracker is not None:
        saved_gamma = dict(gamma_tracker)
        gamma_tracker.clear()

    try:
        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        aux_loss_sum = 0.0
        n = 0

        routing_window: Dict[int, Dict[str, Any]] = {}
        agreement_window: Dict[Tuple[int, int], Tuple[float, int]] = {}
        layer0_indices_window: List[torch.Tensor] = []
        reasoning_samples: List[float] = []
        collab_samples: List[float] = []
        gamma_window: Dict[int, List[float]] = {}
        dead_dummy: Dict[Tuple[int, int], int] = {}

        val_iter = iter(val_loader)
        for _ in range(config.val_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)

            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, y, valid_len_y = batch
            else:
                x, y = batch
                valid_len_y = None

            x = x.to(device, dtype=torch.long, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)
            y = apply_ignore_mask(y, valid_len_y)

            with make_autocast_context("cuda" if device.type == "cuda" else "cpu", amp_dtype):
                _, loss_tuple, _, debug_payload = model(x, targets=y, past_key_values=None)

            if loss_tuple is None:
                continue

            total_loss, ce_loss, aux_loss = loss_tuple
            total_loss_sum += float(total_loss.detach().item())
            ce_loss_sum += float(ce_loss.detach().item())
            aux_loss_sum += float(aux_loss.detach().item())
            n += 1

            update_routing_window(
                debug_payload=debug_payload,
                routing_window=routing_window,
                agreement_window=agreement_window,
                layer0_indices_window=layer0_indices_window,
                reasoning_samples=reasoning_samples,
                collab_samples=collab_samples,
                gamma_tracker=gamma_tracker if gamma_tracker is not None else {},
                gamma_window=gamma_window,
            )

        summary = summarize_routing(
            routing_window=routing_window,
            agreement_window=agreement_window,
            gamma_window=gamma_window,
            dead_expert_streaks=dead_dummy,
            model_cfg=raw_model.config,
        )

        if n == 0:
            return {}

        val_total = total_loss_sum / n
        val_ce = ce_loss_sum / n
        val_aux = aux_loss_sum / n
        val_ppl = math.exp(val_ce) if val_ce < 20 else float("inf")

        out: Dict[str, Any] = {
            "step": step,
            "val_loss": val_total,
            "val_ce_loss": val_ce,
            "val_aux_loss": val_aux,
            "val_perplexity": val_ppl,
            "val_avg_reasoning": summary["avg_reasoning"],
            "val_avg_collab": summary["avg_collab"],
            "val_compute_efficiency": summary["efficiency"],
        }

        for layer_idx, stats in summary["per_layer"].items():
            out[f"val_routing/layer_{layer_idx}/entropy"] = stats["router_entropy"]
            out[f"val_routing/layer_{layer_idx}/drop_rate"] = stats["drop_rate"]
            out[f"val_routing/layer_{layer_idx}/avg_reasoning_steps"] = stats["avg_reasoning_steps"]
            out[f"val_routing/layer_{layer_idx}/avg_collab_steps"] = stats["avg_collab_steps"]
            out[f"val_routing/gini_layer_{layer_idx}"] = stats["gini"]

        for pair_name, agreement in summary["agreements"].items():
            out[f"val_routing/layer_agreement_{pair_name}"] = agreement

        if is_master:
            print(
                f"[VAL] step={step} loss={val_total:.4f} ce={val_ce:.4f} aux={val_aux:.4f} "
                f"ppl={val_ppl:.2f}"
            )

        return out
    finally:
        if gamma_tracker is not None:
            gamma_tracker.clear()
            gamma_tracker.update(saved_gamma)

        if was_training:
            model.train()


def serialize_dead_expert_streaks(streaks: Dict[Tuple[int, int], int]) -> Dict[str, int]:
    """Serialize tuple-keyed dead-expert state for checkpointing."""
    return {f"{k[0]}:{k[1]}": int(v) for k, v in streaks.items()}


def deserialize_dead_expert_streaks(payload: Dict[str, int]) -> Dict[Tuple[int, int], int]:
    """Deserialize dead-expert state from checkpoint."""
    out: Dict[Tuple[int, int], int] = {}
    for key, value in payload.items():
        try:
            a, b = key.split(":")
            out[(int(a), int(b))] = int(value)
        except Exception:
            continue
    return out


def gather_rng_state() -> Dict[str, Any]:
    """Capture Python/NumPy/Torch RNG state."""
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state when resuming."""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_monitor_block(
    step: int,
    max_steps: int,
    epoch: int,
    lr: float,
    eta_s: float,
    total_loss: float,
    ce_loss: float,
    aux_loss: float,
    ppl: float,
    val_metrics: Optional[Dict[str, Any]],
    tok_per_s: float,
    samples_per_s: float,
    ms_per_step: float,
    routing_summary: Dict[str, Any],
    grad_norm: float,
    grad_scale: float,
    mem_used_gib: float,
    mem_total_gib: float,
    model_cfg: LunarisCodexConfig,
) -> str:
    """Render a compact formatted monitor block for terminal logging."""
    width = 74

    def fit(text: str) -> str:
        return text[:width].ljust(width)

    def row(text: str) -> str:
        return f"║{fit(text)}║"

    sep = f"╠{'═' * width}╣"
    top = f"╔{'═' * width}╗"
    bot = f"╚{'═' * width}╝"

    layer_ids: List[int] = routing_summary["layer_ids"]
    per_layer = routing_summary["per_layer"]
    selected = select_routing_layers(layer_ids)

    max_reason = max(1, int(getattr(model_cfg, "n_reasoning_steps", 1)))
    max_collab = int(getattr(model_cfg, "moc_collab_steps", 0))
    if not bool(getattr(model_cfg, "use_moc_collab", True)):
        max_collab = 0

    avg_reason = float(routing_summary.get("avg_reasoning", 0.0))
    avg_collab = float(routing_summary.get("avg_collab", 0.0))
    compute_saved = float(routing_summary.get("efficiency", 0.0) * 100.0)

    entropy_vals: List[str] = []
    drop_vals: List[str] = []
    for idx in selected:
        ent = float(per_layer[idx]["router_entropy"])
        drp = float(per_layer[idx]["drop_rate"])
        ent_s = f"{ent:.2f}"
        drp_s = f"{drp:.2f}"
        if ent < 1.0:
            ent_s = colorize(ent_s, "red")
        else:
            ent_s = colorize(ent_s, "green")
        if drp > 0.1:
            drp_s = colorize(drp_s, "yellow")
        else:
            drp_s = colorize(drp_s, "green")
        entropy_vals.append(ent_s)
        drop_vals.append(drp_s)

    util_line = "n/a"
    if len(selected) > 0:
        util = per_layer[selected[0]]["utilization"]
        util_line = "[" + " ".join(f"{float(v):.2f}" for v in util[: min(8, util.numel())]) + "]"

    val_line = "Val: n/a"
    if val_metrics:
        vloss = float(val_metrics.get("val_loss", float("nan")))
        vppl = float(val_metrics.get("val_perplexity", float("nan")))
        val_line = f"Val: {vloss:.4f} | Val PPL: {vppl:.2f}"

    block = [
        top,
        row("  LUNARIS CODEX v2 - MoC Training Monitor"),
        sep,
        row(f"  Step: {step}/{max_steps}  |  Epoch: {epoch}  |  LR: {lr:.2e}  |  ETA: {format_eta(eta_s)}"),
        sep,
        row("  LOSSES"),
        row(f"    Total: {total_loss:.4f}  |  CE: {ce_loss:.4f}  |  Aux: {aux_loss:.4f}  |  PPL: {ppl:.2f}"),
        row(f"    {val_line}"),
        sep,
        row("  THROUGHPUT"),
        row(f"    Tokens/s: {tok_per_s:,.0f}  |  Samples/s: {samples_per_s:.2f}  |  ms/step: {ms_per_step:.0f}"),
        sep,
        row("  ADAPTIVE COMPUTE (averaged across layers)"),
        row(
            f"    IRL depth: {avg_reason:.2f}/{max_reason}  |  Collab rounds: {avg_collab:.2f}/{max_collab}  "
            f"|  Compute saved: {compute_saved:.1f}%"
        ),
        sep,
        row(
            "  ROUTING "
            + (
                "(" + " / ".join(f"layer {i}" for i in selected) + ")"
                if len(selected) > 0
                else "(unavailable this window)"
            )
        ),
        row(f"    Entropy: {' / '.join(entropy_vals) if entropy_vals else 'n/a'}"),
        row(f"    Drop rate: {' / '.join(drop_vals) if drop_vals else 'n/a'}"),
        row(f"    Expert util: {util_line}" + (f"  (layer {selected[0]})" if len(selected) > 0 else "")),
        sep,
        row("  MEMORY & GRAD"),
        row(
            f"    VRAM: {mem_used_gib:.2f} / {mem_total_gib:.2f} GiB  |  "
            f"Grad norm: {grad_norm:.3f}  |  Grad scale: {grad_scale:.1f}"
        ),
        bot,
    ]
    return "\n".join(block)


def make_dataloader(dataset: Dataset, config: TrainConfig, shuffle: bool, drop_last: bool) -> DataLoader:
    """Create DataLoader with pinned/persistent/prefetch settings."""
    kwargs: Dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": bool(config.pin_memory),
        "drop_last": drop_last,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = bool(config.persistent_workers)
        kwargs["prefetch_factor"] = int(config.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def train(config_path: str) -> None:
    """Main training entrypoint."""
    config = TrainConfig.from_yaml(config_path)
    is_master = True

    # Repro + CUDA perf knobs.
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False")

    device = torch.device(config.device)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype: Optional[torch.dtype] = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else None)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not use_bf16))

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy W&B import.
    wandb = None
    wandb_run = None
    if is_master and config.wandb_project:
        try:
            import wandb as _wandb

            wandb = _wandb
            if config.wandb_run_name is None:
                config.wandb_run_name = f"lunaris-moc-{time.strftime('%Y%m%d-%H%M%S')}"
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=asdict(config),
            )
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")
            wandb = None
            wandb_run = None

    if is_master:
        print("-" * 90)
        print("LUNARIS CODEX v2 :: Single-GPU MoC Training")
        print("-" * 90)
        print(f"Device: {device} | bf16: {use_bf16} | compile: {config.compile_model}")
        print(f"Data dir: {config.data_dir} | SeqLen: {config.sequence_length}")
        print(
            f"Batch: {config.batch_size} | Accum: {config.gradient_accumulation_steps} | "
            f"Effective: {config.batch_size * config.gradient_accumulation_steps}"
        )
        print("-" * 90)

    train_dataset = ShardDataset(data_dir=config.data_dir, sequence_length=config.sequence_length)
    train_loader = make_dataloader(train_dataset, config=config, shuffle=True, drop_last=True)
    if is_master:
        print(
            f"[DATA] train={Path(config.data_dir)} | shards={len(train_dataset.shards)} "
            f"| tokens={train_dataset.total_tokens:,} | samples={train_dataset.total_samples:,}"
        )

    val_loader: Optional[DataLoader] = None
    val_dir = Path(config.data_dir) / "val"
    if val_dir.is_dir() and any(val_dir.glob("*.npy")):
        val_dataset = ShardDataset(data_dir=val_dir, sequence_length=config.sequence_length)
        val_loader = make_dataloader(val_dataset, config=config, shuffle=False, drop_last=False)
        if is_master:
            print(
                f"[VAL] Enabled {val_dir} | shards={len(val_dataset.shards)} "
                f"| tokens={val_dataset.total_tokens:,} | samples={val_dataset.total_samples:,}"
            )
    elif is_master:
        print(f"[VAL] No validation shard dir found at {val_dir}; validation disabled")

    raw_model = LunarisCodex(config.model).to(device)

    active_params, total_params = compute_active_params_per_token(raw_model)
    active_ratio = active_params / max(1, total_params)
    if is_master:
        print(
            f"[MODEL] Trainable params: {total_params:,} | "
            f"Active/token (est): {active_params:,} ({active_ratio:.2%})"
        )

    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )

    gamma_tracker, gamma_handles = register_gamma_hooks(raw_model)

    model: torch.nn.Module = raw_model
    if config.compile_model and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            if is_master:
                print("[MODEL] torch.compile enabled (mode=max-autotune)")
        except Exception as e:
            if is_master:
                print(f"[WARN] torch.compile failed; continuing eager. Error: {e}")

    latest_ckpt = out_dir / "latest_checkpoint.pt"
    current_step = 0
    current_epoch = 0
    best_val_loss = float("inf")
    no_improve_windows = 0
    dead_expert_streaks: Dict[Tuple[int, int], int] = {}

    if latest_ckpt.exists():
        if is_master:
            print(f"[RESUME] Loading {latest_ckpt}")
        state = torch.load(latest_ckpt, map_location=device)
        raw_model.load_state_dict(unwrap_model_keys(state["model"]), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        current_step = int(state.get("step", 0))
        current_epoch = int(state.get("epoch", 0))
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        no_improve_windows = int(state.get("no_improve_windows", 0))
        dead_expert_streaks = deserialize_dead_expert_streaks(state.get("dead_expert_streaks", {}))
        if scaler.is_enabled() and "scaler" in state and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        if "rng_state" in state:
            restore_rng_state(state["rng_state"])

    optimizer.zero_grad(set_to_none=True)
    model.train()

    if wandb is not None:
        wandb.log(
            {
                "meta/active_params_per_token": active_params,
                "meta/total_trainable_params": total_params,
                "meta/active_params_ratio": active_ratio,
                "step": current_step,
            }
        )

    pbar = tqdm(total=config.max_steps, initial=current_step, desc="train", dynamic_ncols=True) if is_master else None

    train_iter = iter(train_loader)
    last_log_time = time.time()
    steps_since_log = 0
    tokens_since_log = 0
    samples_since_log = 0
    log_counter = 0
    last_val_metrics: Optional[Dict[str, Any]] = None

    # Window accumulators (reset every log_interval).
    window_total_loss = 0.0
    window_ce_loss = 0.0
    window_aux_loss = 0.0
    routing_window_log: Dict[int, Dict[str, Any]] = {}
    agreement_window_log: Dict[Tuple[int, int], Tuple[float, int]] = {}
    layer0_indices_window_log: List[torch.Tensor] = []
    reasoning_samples_log: List[float] = []
    collab_samples_log: List[float] = []
    gamma_window_log: Dict[int, List[float]] = {}

    training_start = time.time()

    while current_step < config.max_steps:
        current_step += 1
        steps_since_log += 1

        lr = get_lr(current_step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        step_total_loss_sum = 0.0
        step_ce_loss_sum = 0.0
        step_aux_loss_sum = 0.0

        stop_for_epoch_end = False

        for micro in range(config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                current_epoch += 1
                train_iter = iter(train_loader)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    stop_for_epoch_end = True
                    break

            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, y, valid_len_y = batch
            else:
                x, y = batch
                valid_len_y = None

            x = x.to(device, dtype=torch.long, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)
            y = apply_ignore_mask(y, valid_len_y)

            tokens_since_log += int((y != -1).sum().item())
            samples_since_log += int(x.size(0))

            autocast_ctx = make_autocast_context(device_type, amp_dtype)
            with autocast_ctx:
                _, loss_tuple, _, debug_payload = model(x, targets=y, past_key_values=None)

            if loss_tuple is None:
                continue

            total_loss, ce_loss, aux_loss = loss_tuple
            scaled_loss = total_loss / config.gradient_accumulation_steps

            step_total_loss_sum += float(total_loss.detach().item()) / config.gradient_accumulation_steps
            step_ce_loss_sum += float(ce_loss.detach().item()) / config.gradient_accumulation_steps
            step_aux_loss_sum += float(aux_loss.detach().item()) / config.gradient_accumulation_steps

            update_routing_window(
                debug_payload=debug_payload,
                routing_window=routing_window_log,
                agreement_window=agreement_window_log,
                layer0_indices_window=layer0_indices_window_log,
                reasoning_samples=reasoning_samples_log,
                collab_samples=collab_samples_log,
                gamma_tracker=gamma_tracker,
                gamma_window=gamma_window_log,
            )

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if stop_for_epoch_end:
            optimizer.zero_grad(set_to_none=True)
            break

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        window_total_loss += step_total_loss_sum
        window_ce_loss += step_ce_loss_sum
        window_aux_loss += step_aux_loss_sum

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{step_total_loss_sum:.3f}", "lr": f"{lr:.2e}"})

        do_validate = val_loader is not None and config.val_interval > 0 and (current_step % config.val_interval == 0)
        if do_validate:
            last_val_metrics = run_validation(
                model=model,
                raw_model=raw_model,
                val_loader=val_loader,
                config=config,
                device=device,
                amp_dtype=amp_dtype,
                step=current_step,
                is_master=is_master,
                gamma_tracker=gamma_tracker,
            )

            if len(last_val_metrics) > 0:
                val_loss = float(last_val_metrics["val_loss"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_windows = 0
                    if config.save_best:
                        best_path = out_dir / "best_checkpoint.pt"
                        best_state = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                            "step": current_step,
                            "epoch": current_epoch,
                            "config": asdict(config),
                            "best_val_loss": best_val_loss,
                            "no_improve_windows": no_improve_windows,
                            "dead_expert_streaks": serialize_dead_expert_streaks(dead_expert_streaks),
                            "rng_state": gather_rng_state(),
                        }
                        torch.save(best_state, best_path)
                else:
                    no_improve_windows += 1

                if config.early_stopping_patience > 0 and no_improve_windows >= config.early_stopping_patience:
                    if is_master:
                        print(
                            f"[EARLY STOP] No val improvement for {no_improve_windows} validations "
                            f"(patience={config.early_stopping_patience})."
                        )
                    break

        do_checkpoint = config.save_interval > 0 and (current_step % config.save_interval == 0)
        if do_checkpoint:
            ckpt_state = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "step": current_step,
                "epoch": current_epoch,
                "config": asdict(config),
                "best_val_loss": best_val_loss,
                "no_improve_windows": no_improve_windows,
                "dead_expert_streaks": serialize_dead_expert_streaks(dead_expert_streaks),
                "rng_state": gather_rng_state(),
            }
            numbered = out_dir / f"ckpt_{current_step}.pt"
            latest = out_dir / "latest_checkpoint.pt"
            torch.save(ckpt_state, numbered)
            torch.save(ckpt_state, latest)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if is_master:
                print(f"\n[CKPT] Saved: {numbered} and {latest}")

        do_log = config.log_interval > 0 and (current_step % config.log_interval == 0)
        if do_log:
            log_counter += 1
            now = time.time()
            elapsed = max(1e-6, now - last_log_time)
            sec_per_step = elapsed / max(1, steps_since_log)
            eta_s = max(0.0, (config.max_steps - current_step) * sec_per_step)
            tok_per_s = tokens_since_log / elapsed
            samples_per_s = samples_since_log / elapsed
            ms_per_step = sec_per_step * 1000.0

            mem_used_gib = 0.0
            mem_total_gib = 0.0
            if device.type == "cuda":
                mem_used_gib = torch.cuda.memory_allocated(device) / (1024**3)
                mem_total_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)

            routing_summary = summarize_routing(
                routing_window=routing_window_log,
                agreement_window=agreement_window_log,
                gamma_window=gamma_window_log,
                dead_expert_streaks=dead_expert_streaks,
                model_cfg=raw_model.config,
            )

            avg_total_loss = window_total_loss / max(1, steps_since_log)
            avg_ce_loss = window_ce_loss / max(1, steps_since_log)
            avg_aux_loss = window_aux_loss / max(1, steps_since_log)
            ppl = math.exp(avg_ce_loss) if avg_ce_loss < 20 else float("inf")
            grad_scale = scaler.get_scale() if scaler.is_enabled() else 1.0

            if is_master and config.rich_terminal:
                block = build_monitor_block(
                    step=current_step,
                    max_steps=config.max_steps,
                    epoch=current_epoch,
                    lr=lr,
                    eta_s=eta_s,
                    total_loss=avg_total_loss,
                    ce_loss=avg_ce_loss,
                    aux_loss=avg_aux_loss,
                    ppl=ppl,
                    val_metrics=last_val_metrics,
                    tok_per_s=tok_per_s,
                    samples_per_s=samples_per_s,
                    ms_per_step=ms_per_step,
                    routing_summary=routing_summary,
                    grad_norm=float(grad_norm),
                    grad_scale=float(grad_scale),
                    mem_used_gib=mem_used_gib,
                    mem_total_gib=mem_total_gib,
                    model_cfg=raw_model.config,
                )
                tqdm.write(block)
                for warning in routing_summary["warnings"]:
                    tqdm.write(colorize(f"WARNING: {warning}", "yellow"))

            if wandb is not None:
                log_data: Dict[str, Any] = {
                    "step": current_step,
                    "epoch": current_epoch,
                    "lr": lr,
                    "loss/total": avg_total_loss,
                    "loss/ce": avg_ce_loss,
                    "loss/aux": avg_aux_loss,
                    "perplexity": ppl,
                    "timing/ms_per_step": ms_per_step,
                    "timing/sec_per_step": sec_per_step,
                    "timing/eta_sec": eta_s,
                    "throughput/tokens_per_s": tok_per_s,
                    "throughput/samples_per_s": samples_per_s,
                    "memory/used_gib": mem_used_gib,
                    "memory/total_gib": mem_total_gib,
                    "grad/norm": float(grad_norm),
                    "grad/scale": float(grad_scale),
                    "routing/avg_reasoning_steps": routing_summary["avg_reasoning"],
                    "routing/avg_collab_steps": routing_summary["avg_collab"],
                    "routing/adaptive_compute_efficiency": routing_summary["efficiency"],
                    "routing/compute_saved_pct": routing_summary["efficiency"] * 100.0,
                    "meta/active_params_per_token": active_params,
                    "meta/total_trainable_params": total_params,
                    "meta/active_params_ratio": active_ratio,
                }

                if last_val_metrics is not None:
                    for key, value in last_val_metrics.items():
                        if key in {"step"}:
                            continue
                        log_data[f"val/{key}"] = value

                routing_due = (log_counter % config.log_routing_every) == 0
                if routing_due:
                    for layer_idx, stats in routing_summary["per_layer"].items():
                        log_data[f"routing/layer_{layer_idx}/entropy"] = stats["router_entropy"]
                        log_data[f"routing/layer_{layer_idx}/drop_rate"] = stats["drop_rate"]
                        log_data[f"routing/layer_{layer_idx}/capacity_per_expert"] = stats["capacity_per_expert"]
                        log_data[f"routing/layer_{layer_idx}/avg_reasoning_steps"] = stats["avg_reasoning_steps"]
                        log_data[f"routing/layer_{layer_idx}/avg_collab_steps"] = stats["avg_collab_steps"]
                        log_data[f"routing/gini_layer_{layer_idx}"] = stats["gini"]
                        log_data[f"routing/dead_experts_layer_{layer_idx}"] = stats["dead_experts"]

                        util_np = stats["utilization"].detach().cpu().numpy()
                        log_data[f"routing/layer_{layer_idx}/expert_util_hist"] = wandb.Histogram(util_np)

                    for pair_name, agreement in routing_summary["agreements"].items():
                        log_data[f"routing/layer_agreement_{pair_name}"] = agreement

                    for layer_idx, gamma_mean in routing_summary["gamma_means"].items():
                        log_data[f"routing/layer_{layer_idx}/gamma_mean"] = gamma_mean

                    if len(reasoning_samples_log) > 0:
                        log_data["routing/reasoning_depth_hist"] = wandb.Histogram(np.array(reasoning_samples_log))
                    if len(collab_samples_log) > 0:
                        log_data["routing/collab_depth_hist"] = wandb.Histogram(np.array(collab_samples_log))

                    fig = cooccurrence_figure(
                        layer0_indices_window_log, int(getattr(raw_model.config, "n_experts", 0) or 0)
                    )
                    if fig is not None:
                        try:
                            log_data["routing/layer0_utilization_cooccurrence"] = wandb.Image(fig)
                        finally:
                            plt.close(fig)

                wandb.log(log_data)

            last_log_time = now
            steps_since_log = 0
            tokens_since_log = 0
            samples_since_log = 0
            window_total_loss = 0.0
            window_ce_loss = 0.0
            window_aux_loss = 0.0
            routing_window_log = {}
            agreement_window_log = {}
            layer0_indices_window_log = []
            reasoning_samples_log = []
            collab_samples_log = []
            gamma_window_log = {}

    if pbar is not None:
        pbar.close()

    for handle in gamma_handles:
        try:
            handle.remove()
        except Exception:
            pass

    if wandb_run is not None:
        try:
            wandb_run.summary["best_val_loss"] = best_val_loss
            wandb_run.summary["best_val_perplexity"] = math.exp(best_val_loss) if best_val_loss < 20 else float("inf")
            wandb_run.summary["final_step"] = current_step
            wandb_run.summary["train_wall_time_sec"] = time.time() - training_start
        except Exception:
            pass

    if wandb is not None:
        wandb.finish()

    if is_master:
        elapsed = time.time() - training_start
        print(f"[DONE] step={current_step} | best_val_loss={best_val_loss:.4f} | wall_time={elapsed/3600:.2f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LunarisCodex (single GPU, MoC diagnostics).")
    parser.add_argument("config", type=str, help="Path to config.yaml")
    args = parser.parse_args()
    train(args.config)
