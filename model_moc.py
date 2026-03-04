import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except Exception:
    sdpa_kernel = None
    SDPBackend = None


# -----------------------------------------------------------------------------
# Global backend hints (safe no-ops on unsupported devices)
# -----------------------------------------------------------------------------
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
except Exception:
    pass


# -----------------------------------------------------------------------------
# Norm fallback for older torch builds
# -----------------------------------------------------------------------------
class _FallbackRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


RMSNorm = nn.RMSNorm if hasattr(nn, "RMSNorm") else _FallbackRMSNorm


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class LunarisCodexConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    vocab_size: int = 50257
    multiple_of: int = 256
    ffn_hidden_multiplier: float = 4.0
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0

    # MoE / MoC
    n_experts: Optional[int] = 8
    top_k: int = 2
    aux_loss_weight: float = 1e-2
    capacity_factor: float = 1.25
    router_z_loss_weight: float = 1e-3
    router_noise_std: float = 0.0
    drop_penalty_weight: float = 1e-3

    # Engineering
    use_gradient_checkpointing: bool = True
    grad_ckpt_policy: str = "ffn"  # "none" | "ffn" | "block"
    save_attn_weights: bool = False

    # Collaboration flags (legacy compatibility + vNext defaults)
    use_simple_collab: bool = False
    use_moc_collab: bool = True
    simple_collab_dropout: float = 0.1
    moc_collab_steps: int = 2
    moc_collab_heads: int = 4  # kept for compatibility (unused by vNext MoC-Lite)
    moc_collab_dropout: float = 0.0
    moc_use_mediator: bool = True
    moc_expert_feedback: bool = True
    moc_low_rank_message_dim: int = 0

    # IRL reasoning depth (interpreted as max steps)
    n_reasoning_steps: int = 1

    # Adaptive compute
    adaptive_reasoning: bool = True
    adaptive_collaboration: bool = True
    adaptive_compute_mode: str = "soft"  # "soft" (compile-friendly) or "hard" (faster sparse execution)
    reasoning_gate_temperature: float = 1.0
    reasoning_gate_noise_std: float = 0.0
    reasoning_step_penalty_weight: float = 0.0
    collaboration_gate_temperature: float = 1.0
    collaboration_gate_noise_std: float = 0.0
    collaboration_step_penalty_weight: float = 0.0

    # Diagnostics / debug payload
    track_routing_stats: bool = True
    return_routing_diagnostics: bool = False

    # Optional aggressive variant flags
    aggressive_router_quant_bits: int = 0
    aggressive_ffn_quant_bits: int = 0


# -----------------------------------------------------------------------------
# Rotary embeddings
# -----------------------------------------------------------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: [B, H, T, D]
    bsz, n_heads, seqlen, dim = xq.shape
    xq_ = torch.view_as_complex(xq.reshape(bsz, n_heads, seqlen, dim // 2, 2).to(torch.float32))
    xk_ = torch.view_as_complex(xk.reshape(bsz, xk.size(1), seqlen, dim // 2, 2).to(torch.float32))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(bsz, n_heads, seqlen, dim).to(dtype=xq.dtype)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(bsz, xk.size(1), seqlen, dim).to(dtype=xk.dtype)
    return xq_out, xk_out


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _argsort_stable(x: torch.Tensor, descending: bool = False) -> torch.Tensor:
    try:
        return torch.argsort(x, descending=descending, stable=True)
    except TypeError:
        return torch.argsort(x, descending=descending)


def _fake_quantize_ste(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return x
    qmax = float((1 << (bits - 1)) - 1)
    scale = x.detach().abs().amax()
    scale = torch.clamp(scale / qmax, min=1e-8)
    x_q = torch.clamp(torch.round(x / scale), min=-qmax, max=qmax) * scale
    return x + (x_q - x).detach()


def _get_sdpa_backends(device: torch.device) -> List[Any]:
    if SDPBackend is None:
        return []

    flash = getattr(SDPBackend, "FLASH_ATTENTION", None)
    efficient = getattr(SDPBackend, "EFFICIENT_ATTENTION", None)
    if efficient is None:
        efficient = getattr(SDPBackend, "MEM_EFFICIENT", None)
    math_backend = getattr(SDPBackend, "MATH", None)

    backends: List[Any] = []
    if device.type in ("cuda", "hip") and flash is not None:
        backends.append(flash)
    if efficient is not None:
        backends.append(efficient)
    if math_backend is not None:
        backends.append(math_backend)

    deduped: List[Any] = []
    for b in backends:
        if b not in deduped:
            deduped.append(b)
    return deduped


def _scaled_dot_product_attention_safe(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    backends = _get_sdpa_backends(q.device)
    if sdpa_kernel is not None and len(backends) > 0:
        try:
            with sdpa_kernel(backends=backends):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
                )
        except Exception:
            pass
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p
    )


def _compute_step_activity(
    logits: torch.Tensor,
    max_steps: int,
    mode: str,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert logits over {0..max_steps} into per-step activity.

    Returns:
      activity: [N, max_steps] in [0,1], where activity[:, s] means "run step s+1"
      expected_steps: [N]
      probs: [N, max_steps+1]
    """
    temperature = max(float(temperature), 1e-4)
    probs = F.softmax(logits / temperature, dim=-1, dtype=torch.float32)
    step_values = torch.arange(max_steps + 1, device=logits.device, dtype=torch.float32)
    expected_steps = (probs * step_values.unsqueeze(0)).sum(dim=-1)

    mode = mode.lower()
    if mode == "hard":
        hard_steps = torch.argmax(probs, dim=-1)  # [N]
        thresholds = torch.arange(1, max_steps + 1, device=logits.device).unsqueeze(0)
        activity = (hard_steps.unsqueeze(-1) >= thresholds).to(torch.float32)
        return activity, expected_steps, probs

    # soft / compile-friendly mode
    tail = torch.flip(torch.cumsum(torch.flip(probs[:, 1:], dims=[-1]), dim=-1), dims=[-1])
    return tail, expected_steps, probs


# -----------------------------------------------------------------------------
# Attention (RoPE + GQA + safe SDPA backend fallback)
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        assert config.n_heads % config.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.wqkv = nn.Linear(config.d_model, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seqlen, d_model = x.shape
        qkv = self.wqkv(x)
        q, k, v = torch.split(
            qkv,
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        q, k = apply_rotary_emb(q, k, freqs_cis)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(-2)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        present_kv = (k, v)

        if self.n_kv_heads < self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        attn_dropout = self.dropout.p if (self.training and self.dropout.p > 0) else 0.0

        # Correct causal masking with KV cache offsets.
        if past_len == 0:
            attn_mask = None
            is_causal = True
        else:
            total_kv = k.size(-2)
            q_pos = torch.arange(seqlen, device=x.device).unsqueeze(-1) + past_len
            kv_pos = torch.arange(total_kv, device=x.device).unsqueeze(0)
            attn_mask = kv_pos <= q_pos
            is_causal = False

        y = _scaled_dot_product_attention_safe(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=attn_dropout,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.dropout(self.o_proj(y))
        return y, present_kv


# -----------------------------------------------------------------------------
# Reasoning FFN (IRL inside) with adaptive depth gate
# -----------------------------------------------------------------------------
class ReasoningFeedForward(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_multiplier * config.d_model)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w13 = nn.Linear(config.d_model, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.d_model, eps=1e-6)

        self.max_reasoning_steps = max(1, int(getattr(config, "n_reasoning_steps", 1)))
        self.alpha = 1.0 / math.sqrt(float(self.max_reasoning_steps))

        self.adaptive_reasoning = bool(getattr(config, "adaptive_reasoning", True)) and self.max_reasoning_steps > 1
        self.adaptive_mode = str(getattr(config, "adaptive_compute_mode", "soft")).lower()
        self.reasoning_gate_temperature = float(getattr(config, "reasoning_gate_temperature", 1.0))
        self.reasoning_gate_noise_std = float(getattr(config, "reasoning_gate_noise_std", 0.0))
        self.reasoning_step_penalty_weight = float(getattr(config, "reasoning_step_penalty_weight", 0.0))

        self.ffn_quant_bits = int(getattr(config, "aggressive_ffn_quant_bits", 0))

        if self.adaptive_reasoning:
            self.step_gate = nn.Linear(config.d_model, self.max_reasoning_steps + 1, bias=True)
            with torch.no_grad():
                self.step_gate.bias.zero_()
                self.step_gate.bias.data[-1] = 2.0
        else:
            self.step_gate = None

        self.last_expected_steps = torch.tensor(1.0)
        self.last_step_penalty = torch.tensor(0.0)

    def _linear(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        w = layer.weight
        if self.ffn_quant_bits > 0:
            w = _fake_quantize_ste(w, self.ffn_quant_bits)
        return F.linear(x, w, layer.bias)

    def _ffn_logic(self, z: torch.Tensor) -> torch.Tensor:
        gate, up = self._linear(z, self.w13).chunk(2, dim=-1)
        swiglu = F.silu(gate) * up
        return self.dropout(self._linear(swiglu, self.w2))

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        if self.step_gate is None:
            for _ in range(self.max_reasoning_steps):
                upd = self._ffn_logic(self.norm(h + x))
                h = h + self.alpha * upd
            expected = x.new_full((x.size(0),), float(self.max_reasoning_steps), dtype=torch.float32)
        else:
            step_logits = self.step_gate(x.to(torch.float32))
            if self.training and self.reasoning_gate_noise_std > 0.0:
                step_logits = step_logits + torch.randn_like(step_logits) * self.reasoning_gate_noise_std

            activity, expected, _ = _compute_step_activity(
                step_logits,
                max_steps=self.max_reasoning_steps,
                mode=self.adaptive_mode,
                temperature=self.reasoning_gate_temperature,
            )

            if self.adaptive_mode == "hard":
                # Sparse execution path: runs deeper IRL only for selected tokens.
                for step in range(self.max_reasoning_steps):
                    active = activity[:, step] > 0.5
                    if not torch.any(active):
                        break
                    active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
                    h_sel = h.index_select(0, active_idx)
                    x_sel = x.index_select(0, active_idx)
                    upd = self._ffn_logic(self.norm(h_sel + x_sel))
                    h_sel = h_sel + self.alpha * upd
                    h.index_copy_(0, active_idx, h_sel)
            else:
                activity = activity.to(dtype=x.dtype)
                for step in range(self.max_reasoning_steps):
                    upd = self._ffn_logic(self.norm(h + x))
                    h = h + activity[:, step : step + 1] * self.alpha * upd

        self.last_expected_steps = expected.detach().mean()
        if self.reasoning_step_penalty_weight > 0.0 and self.max_reasoning_steps > 0:
            self.last_step_penalty = (
                expected.mean() / float(self.max_reasoning_steps)
            ) * self.reasoning_step_penalty_weight
        else:
            self.last_step_penalty = x.new_zeros((), dtype=torch.float32)

        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = x.shape
        x_flat = x.reshape(-1, in_shape[-1])
        y_flat = self._forward_flat(x_flat)
        return y_flat.view(in_shape)


# -----------------------------------------------------------------------------
# MoC Top-K Experts (vectorized dispatch + MoC-Lite collaboration)
# -----------------------------------------------------------------------------
class MoCTopKExperts(nn.Module):
    """
    MoC vNext core:
    - Router in fp32 (optionally fake-quantized for aggressive experiments)
    - Vectorized top-k dispatch + vectorized capacity masking
    - One unavoidable per-expert loop for executing expert-specific parameters
    - MoC-Lite mediator-only collaboration in O(K) per token
    - Adaptive reasoning depth inside experts and adaptive collaboration depth
    """

    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        assert config.n_experts is not None and config.n_experts > 0
        assert config.top_k >= 1
        assert config.top_k <= config.n_experts, "top_k cannot exceed n_experts"

        self.n_experts = int(config.n_experts)
        self.top_k = int(config.top_k)
        self.aux_loss_weight = float(config.aux_loss_weight)
        self.capacity_factor = float(config.capacity_factor)
        self.z_loss_weight = float(config.router_z_loss_weight)
        self.drop_penalty_weight = float(getattr(config, "drop_penalty_weight", 0.0))

        self.config = config
        self.track_routing_stats = bool(getattr(config, "track_routing_stats", True))
        self.save_attn_weights = bool(getattr(config, "save_attn_weights", False))
        self.last_attn_weights: Optional[torch.Tensor] = None
        self.last_routing_diagnostics: Optional[Dict[str, torch.Tensor]] = None

        self.router_quant_bits = int(getattr(config, "aggressive_router_quant_bits", 0))

        # Router
        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)

        # Experts (IRL FFNs)
        self.experts = nn.ModuleList([ReasoningFeedForward(config) for _ in range(self.n_experts)])

        # MoC-Lite collaboration modules
        d_model = config.d_model
        self.use_simple_collab = bool(getattr(config, "use_simple_collab", False))
        self.use_moc_collab = bool(getattr(config, "use_moc_collab", True))
        self.max_collab_steps = max(0, int(getattr(config, "moc_collab_steps", 0)))
        self.collab_alpha = 1.0 / math.sqrt(float(max(1, self.max_collab_steps)))
        self.adaptive_collaboration = bool(getattr(config, "adaptive_collaboration", True)) and self.max_collab_steps > 1
        self.adaptive_mode = str(getattr(config, "adaptive_compute_mode", "soft")).lower()
        self.collab_gate_temperature = float(getattr(config, "collaboration_gate_temperature", 1.0))
        self.collab_gate_noise_std = float(getattr(config, "collaboration_gate_noise_std", 0.0))
        self.collab_step_penalty_weight = float(getattr(config, "collaboration_step_penalty_weight", 0.0))

        self.moc_use_mediator = bool(getattr(config, "moc_use_mediator", True))
        self.moc_expert_feedback = bool(getattr(config, "moc_expert_feedback", True))

        self.mediator_norm = RMSNorm(d_model, eps=1e-6)
        self.expert_norm = RMSNorm(d_model, eps=1e-6)
        self.med_q = nn.Linear(d_model, d_model, bias=False)
        self.exp_k = nn.Linear(d_model, d_model, bias=False)

        rank = int(getattr(config, "moc_low_rank_message_dim", 0))
        if rank > 0 and rank < d_model:
            self.exp_v_down = nn.Linear(d_model, rank, bias=False)
            self.exp_v_up = nn.Linear(rank, d_model, bias=False)
        else:
            self.exp_v_down = None
            self.exp_v_up = None
            self.exp_v = nn.Linear(d_model, d_model, bias=False)

        self.med_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        self.collab_dropout = nn.Dropout(float(getattr(config, "moc_collab_dropout", 0.0)))

        if self.moc_use_mediator:
            self.mediator = nn.Parameter(torch.empty(1, d_model))
            nn.init.normal_(self.mediator, mean=0.0, std=0.02)
        else:
            self.mediator = None

        self.med_context = nn.Linear(d_model, d_model, bias=False)
        self.fuse_gate = nn.Linear(d_model, 1, bias=True)
        with torch.no_grad():
            self.fuse_gate.bias.fill_(-2.0)

        if self.moc_expert_feedback:
            self.med_to_exp = nn.Linear(d_model, d_model, bias=False)
            self.exp_feedback_e = nn.Linear(d_model, 1, bias=True)
            self.exp_feedback_m = nn.Linear(d_model, 1, bias=False)

        if self.adaptive_collaboration:
            self.collab_step_gate = nn.Linear(d_model, self.max_collab_steps + 1, bias=True)
            with torch.no_grad():
                self.collab_step_gate.bias.zero_()
                self.collab_step_gate.bias.data[-1] = 2.0
        else:
            self.collab_step_gate = None

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _load_balance_loss_topk(
        router_probs: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_probs: torch.Tensor,
        n_experts: int,
    ) -> torch.Tensor:
        # Fully vectorized top-k assignment histogram.
        n_tokens, n_experts_total = router_probs.shape
        assign = router_probs.new_zeros(n_tokens, n_experts_total)
        assign.scatter_add_(1, topk_idx, topk_probs)
        fraction_tokens = assign.mean(dim=0)
        prob_mass = router_probs.mean(dim=0)
        return (prob_mass * fraction_tokens).sum() * float(n_experts)

    def _capacity_limit(self, total_slots: int) -> int:
        avg = total_slots / max(1, self.n_experts)
        target = int(math.ceil(avg * self.capacity_factor))
        if target <= 0:
            return 0
        # power-of-two padding helps allocator behavior on some accelerators
        return 1 << int(math.ceil(math.log2(target)))

    @staticmethod
    def _vectorized_capacity_keep(
        target_expert: torch.Tensor,
        priority: torch.Tensor,
        capacity: int,
        n_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          keep_mask: [N*K] bool in original order
          sorted_perm: permutation sorted by expert (and by descending priority inside each expert)
          keep_sorted: keep_mask in sorted order
        """
        nk = target_expert.numel()
        device = target_expert.device

        keep = torch.zeros(nk, dtype=torch.bool, device=device)
        if nk == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return keep, empty, keep

        # 1) priority descending, stable
        perm_prio = _argsort_stable(priority, descending=True)
        # 2) stable group by expert while preserving priority order inside each expert
        experts_after_prio = target_expert.index_select(0, perm_prio)
        regroup = _argsort_stable(experts_after_prio, descending=False)
        sorted_perm = perm_prio.index_select(0, regroup)
        experts_sorted = target_expert.index_select(0, sorted_perm)

        if capacity <= 0:
            return keep, sorted_perm, keep.index_select(0, sorted_perm)

        idx = torch.arange(nk, device=device)
        start = torch.ones(nk, dtype=torch.bool, device=device)
        if nk > 1:
            start[1:] = experts_sorted[1:] != experts_sorted[:-1]

        first_pos = torch.zeros(n_experts, dtype=torch.long, device=device)
        first_pos[experts_sorted[start]] = idx[start]
        rank_in_expert = idx - first_pos[experts_sorted]

        keep_sorted = rank_in_expert < capacity
        keep.index_copy_(0, sorted_perm, keep_sorted)
        return keep, sorted_perm, keep_sorted

    def _project_values(self, expert_states: torch.Tensor) -> torch.Tensor:
        if self.exp_v_down is not None and self.exp_v_up is not None:
            return self.exp_v_up(self.exp_v_down(expert_states))
        return self.exp_v(expert_states)

    def _moc_lite_step(
        self,
        mediator: torch.Tensor,
        expert_states: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        One O(K) mediator step:
          mediator -> attention over experts -> message -> mediator update
          optional mediator -> experts gated feedback
        """
        # Normalize states before projections for stability.
        m_norm = self.mediator_norm(mediator)
        e_norm = self.expert_norm(expert_states)

        q = self.med_q(m_norm)  # [N, D]
        k = self.exp_k(e_norm)  # [N, K, D]
        v = self._project_values(e_norm)  # [N, K, D]

        logits = (k * q.unsqueeze(1)).sum(dim=-1) * (1.0 / math.sqrt(k.size(-1)))
        logits = logits.masked_fill(~keep_mask, -1e4)

        attn = F.softmax(logits, dim=-1, dtype=torch.float32)
        attn = attn * keep_mask.to(attn.dtype)
        denom = attn.sum(dim=-1, keepdim=True)
        attn = torch.where(denom > 0, attn / denom, torch.zeros_like(attn))

        msg = torch.sum(attn.to(v.dtype).unsqueeze(-1) * v, dim=1)
        med_delta = self.med_mlp(m_norm + msg)
        mediator_new = mediator + self.collab_alpha * self.collab_dropout(med_delta)

        experts_new = expert_states
        if self.moc_expert_feedback:
            med_msg = self.med_to_exp(self.mediator_norm(mediator_new)).unsqueeze(1)
            gate = torch.sigmoid(
                self.exp_feedback_e(self.expert_norm(expert_states))
                + self.exp_feedback_m(self.mediator_norm(mediator_new)).unsqueeze(1)
            )
            exp_delta = gate * med_msg
            experts_new = expert_states + self.collab_alpha * self.collab_dropout(exp_delta)
            experts_new = experts_new * keep_mask.unsqueeze(-1)

        return mediator_new, experts_new, (attn if self.save_attn_weights else None)

    def _moc_lite_collaboration(
        self,
        x_flat: torch.Tensor,
        expert_states: torch.Tensor,
        keep_mask: torch.Tensor,
        weights: torch.Tensor,
        no_kept: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_tokens, _, d_model = expert_states.shape

        weighted = torch.sum(expert_states * weights.unsqueeze(-1), dim=1)

        if self.mediator is not None:
            mediator = self.mediator.unsqueeze(0).expand(n_tokens, -1, -1).squeeze(1).to(expert_states.dtype)
            mediator = mediator + self.med_context(x_flat.to(expert_states.dtype))
        else:
            mediator = weighted + self.med_context(x_flat.to(expert_states.dtype))

        if (not self.use_moc_collab) or self.max_collab_steps <= 0:
            avg_steps = x_flat.new_zeros((), dtype=torch.float32)
            return weighted, x_flat.new_zeros((), dtype=torch.float32), avg_steps

        if self.collab_step_gate is None:
            activity = torch.ones(n_tokens, self.max_collab_steps, device=x_flat.device, dtype=torch.float32)
            expected_steps = torch.full(
                (n_tokens,), float(self.max_collab_steps), device=x_flat.device, dtype=torch.float32
            )
        else:
            step_logits = self.collab_step_gate(x_flat.to(torch.float32))
            if self.training and self.collab_gate_noise_std > 0.0:
                step_logits = step_logits + torch.randn_like(step_logits) * self.collab_gate_noise_std
            activity, expected_steps, _ = _compute_step_activity(
                step_logits,
                max_steps=self.max_collab_steps,
                mode=self.adaptive_mode,
                temperature=self.collab_gate_temperature,
            )

        # Never spend collaboration compute when all experts were dropped.
        activity = activity * (~no_kept.squeeze(-1)).to(activity.dtype).unsqueeze(-1)

        last_attn = None
        if self.adaptive_mode == "hard":
            for step in range(self.max_collab_steps):
                active = activity[:, step] > 0.5
                if not torch.any(active):
                    break
                idx = torch.nonzero(active, as_tuple=False).squeeze(-1)

                med_sel = mediator.index_select(0, idx)
                exp_sel = expert_states.index_select(0, idx)
                keep_sel = keep_mask.index_select(0, idx)

                med_new, exp_new, attn = self._moc_lite_step(med_sel, exp_sel, keep_sel)
                mediator.index_copy_(0, idx, med_new)
                expert_states.index_copy_(0, idx, exp_new)
                if attn is not None:
                    last_attn = attn
        else:
            for step in range(self.max_collab_steps):
                med_new, exp_new, attn = self._moc_lite_step(mediator, expert_states, keep_mask)
                a = activity[:, step : step + 1].to(mediator.dtype)
                mediator = mediator + a * (med_new - mediator)
                expert_states = expert_states + a.unsqueeze(-1) * (exp_new - expert_states)
                if attn is not None:
                    last_attn = attn

        if self.save_attn_weights:
            self.last_attn_weights = last_attn

        expert_states = expert_states * keep_mask.unsqueeze(-1)
        weighted_refined = torch.sum(expert_states * weights.unsqueeze(-1), dim=1)

        gamma = torch.sigmoid(self.fuse_gate(x_flat.to(weighted_refined.dtype)))
        fused = gamma * mediator + (1.0 - gamma) * weighted_refined
        fused = torch.where(no_kept, x_flat.to(fused.dtype), fused)

        avg_steps = expected_steps.mean()
        if self.collab_step_penalty_weight > 0.0 and self.max_collab_steps > 0:
            collab_penalty = (
                expected_steps.mean() / float(self.max_collab_steps)
            ) * self.collab_step_penalty_weight
        else:
            collab_penalty = x_flat.new_zeros((), dtype=torch.float32)

        return fused, collab_penalty, avg_steps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, d_model = x.shape
        n_tokens = bsz * seqlen
        x_flat = x.view(n_tokens, d_model)

        gate_w = self.gate.weight
        if self.router_quant_bits > 0:
            gate_w = _fake_quantize_ste(gate_w, self.router_quant_bits)

        # Router in fp32 for stable top-k and aux losses.
        logits = F.linear(x_flat.to(torch.float32), gate_w.to(torch.float32), None)
        if self.training and getattr(self.config, "router_noise_std", 0.0) > 0.0:
            logits = logits + torch.randn_like(logits) * float(self.config.router_noise_std)

        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        topk_probs = F.softmax(topk_vals, dim=-1, dtype=torch.float32)
        router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        balance_loss = self._load_balance_loss_topk(router_probs, topk_idx, topk_probs, self.n_experts)
        z = torch.logsumexp(logits, dim=-1)
        z_loss = (z * z).mean()
        aux_loss = self.aux_loss_weight * balance_loss + self.z_loss_weight * z_loss

        nk = n_tokens * self.top_k
        x_expanded = x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(nk, d_model)
        target_expert = topk_idx.reshape(-1)
        priority = topk_vals.reshape(-1)

        capacity = min(self._capacity_limit(nk), nk)
        keep_mask, sorted_perm, keep_sorted = self._vectorized_capacity_keep(
            target_expert,
            priority,
            capacity,
            self.n_experts,
        )

        total_dropped = (~keep_mask).sum()

        expert_out_selected = x_expanded.new_zeros((nk, d_model))
        reasoning_penalty = x_flat.new_zeros((), dtype=torch.float32)
        avg_reasoning_steps = x_flat.new_zeros((), dtype=torch.float32)

        kept_perm = sorted_perm[keep_sorted]
        if kept_perm.numel() > 0:
            kept_experts = target_expert.index_select(0, kept_perm)
            kept_inputs = x_expanded.index_select(0, kept_perm)
            kept_counts = torch.bincount(kept_experts, minlength=self.n_experts)
            ends = kept_counts.cumsum(dim=0)
            starts = torch.cat([torch.zeros(1, device=ends.device, dtype=ends.dtype), ends[:-1]], dim=0)

            outputs_kept = torch.empty_like(kept_inputs)
            total_kept = float(max(1, kept_inputs.size(0)))

            # This loop is intentionally kept: each expert has distinct parameters.
            for e in range(self.n_experts):
                s = int(starts[e].tolist())
                t = int(ends[e].tolist())
                chunk_size = t - s
                if chunk_size == 0:
                    continue
                chunk = kept_inputs[s:t]
                out_chunk = self.experts[e](chunk)
                outputs_kept[s:t] = out_chunk

                avg_reasoning_steps = avg_reasoning_steps + (
                    self.experts[e].last_expected_steps.to(avg_reasoning_steps.dtype)
                    * (float(chunk_size) / total_kept)
                )
                if self.experts[e].last_step_penalty is not None:
                    reasoning_penalty = reasoning_penalty + (
                        self.experts[e].last_step_penalty.to(reasoning_penalty.dtype)
                        * (float(chunk_size) / total_kept)
                    )

            expert_out_selected.index_copy_(0, kept_perm, outputs_kept)

        if self.drop_penalty_weight > 0.0 and nk > 0:
            drop_frac = total_dropped.to(torch.float32) / float(nk)
            aux_loss = aux_loss + self.drop_penalty_weight * drop_frac

        expert_states = expert_out_selected.view(n_tokens, self.top_k, d_model)
        kept_mask_nk = keep_mask.view(n_tokens, self.top_k)
        expert_states = expert_states * kept_mask_nk.unsqueeze(-1)

        topk_probs_masked_f32 = topk_probs * kept_mask_nk.to(topk_probs.dtype)
        denom_f32 = topk_probs_masked_f32.sum(dim=-1, keepdim=True)
        no_kept = denom_f32 <= 0
        safe_denom_f32 = torch.where(no_kept, torch.ones_like(denom_f32), denom_f32)
        weights_f32 = topk_probs_masked_f32 / safe_denom_f32
        weights_f32 = torch.where(no_kept, torch.zeros_like(weights_f32), weights_f32)
        weights = weights_f32.to(expert_states.dtype)

        if self.use_simple_collab and not self.use_moc_collab:
            fused = torch.sum(expert_states * weights.unsqueeze(-1), dim=1)
            avg_collab_steps = x_flat.new_zeros((), dtype=torch.float32)
            collab_penalty = x_flat.new_zeros((), dtype=torch.float32)
            fused = torch.where(no_kept, x_flat.to(fused.dtype), fused)
        else:
            fused, collab_penalty, avg_collab_steps = self._moc_lite_collaboration(
                x_flat=x_flat,
                expert_states=expert_states,
                keep_mask=kept_mask_nk,
                weights=weights,
                no_kept=no_kept,
            )

        aux_loss = aux_loss + reasoning_penalty + collab_penalty

        fused = self.o_proj(fused).view(bsz, seqlen, d_model)
        expert_indices = topk_idx.view(bsz, seqlen, self.top_k).to(torch.long)

        if self.track_routing_stats:
            req_hist = torch.bincount(target_expert, minlength=self.n_experts).to(torch.float32)
            kept_hist = torch.bincount(target_expert[keep_mask], minlength=self.n_experts).to(torch.float32)
            router_entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-9))).sum(dim=-1).mean()
            self.last_routing_diagnostics = {
                "requested_hist": req_hist.detach(),
                "kept_hist": kept_hist.detach(),
                "drop_rate": (total_dropped.to(torch.float32) / float(max(1, nk))).detach(),
                "capacity_per_expert": torch.tensor(float(capacity), device=x.device).detach(),
                "router_entropy": router_entropy.detach(),
                "avg_reasoning_steps": avg_reasoning_steps.detach(),
                "avg_collab_steps": avg_collab_steps.detach(),
            }
        else:
            self.last_routing_diagnostics = None

        return fused, aux_loss.to(fused.dtype), expert_indices


# -----------------------------------------------------------------------------
# Transformer block
# -----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.attn_norm = RMSNorm(config.d_model, eps=1e-6)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=1e-6)

        if config.n_experts is not None and config.n_experts > 0 and config.top_k >= 1:
            self.feed_forward = MoCTopKExperts(config)
            self.is_moe = True
        else:
            self.feed_forward = ReasoningFeedForward(config)
            self.is_moe = False

        self.last_routing_diagnostics: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        def _inner_full(
            x_inner: torch.Tensor,
            freqs_cis_inner: torch.Tensor,
            past_kv_inner: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
            attn_output, new_kv = self.attention(self.attn_norm(x_inner), freqs_cis_inner, past_kv_inner)
            h = x_inner + attn_output

            ffn_input = self.ffn_norm(h)
            if self.is_moe:
                ffn_output, aux_loss, expert_indices = self.feed_forward(ffn_input)
                self.last_routing_diagnostics = self.feed_forward.last_routing_diagnostics
            else:
                ffn_output = self.feed_forward(ffn_input)
                aux_loss = ffn_output.new_zeros(())
                expert_indices = None
                self.last_routing_diagnostics = None

            out = h + ffn_output
            return out, new_kv, aux_loss, expert_indices

        if self.training and self.config.use_gradient_checkpointing:
            if self.config.grad_ckpt_policy == "block":
                # Keep checkpoint inputs tensor-only; close over past_kv.
                def _inner_ckpt(x_inner: torch.Tensor, freqs_cis_inner: torch.Tensor):
                    return _inner_full(x_inner, freqs_cis_inner, past_kv)

                return checkpoint(_inner_ckpt, x, freqs_cis, use_reentrant=False)

            if self.config.grad_ckpt_policy == "ffn":
                attn_output, new_kv = self.attention(self.attn_norm(x), freqs_cis, past_kv)
                h = x + attn_output
                ffn_input = self.ffn_norm(h)

                def _ffn_only(ffn_in: torch.Tensor):
                    if self.is_moe:
                        y, aux, experts = self.feed_forward(ffn_in)
                    else:
                        y = self.feed_forward(ffn_in)
                        aux = y.new_zeros(())
                        experts = None
                    return y, aux, experts

                ffn_output, aux_loss, expert_indices = checkpoint(_ffn_only, ffn_input, use_reentrant=False)
                self.last_routing_diagnostics = (
                    self.feed_forward.last_routing_diagnostics if self.is_moe else None
                )
                out = h + ffn_output
                return out, new_kv, aux_loss, expert_indices

        return _inner_full(x, freqs_cis, past_kv)


# -----------------------------------------------------------------------------
# Top-level model
# -----------------------------------------------------------------------------
class LunarisCodex(nn.Module):
    def __init__(self, config: LunarisCodexConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=RMSNorm(config.d_model, eps=1e-6),
                drop=nn.Dropout(config.dropout),
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # tie weights
        self.transformer.wte.weight = self.lm_head.weight

        freqs_cis = precompute_freqs_cis(
            self.config.d_model // self.config.n_heads,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)
        self._rescale_out_projections()

        self.last_routing_diagnostics: Optional[List[Dict[str, torch.Tensor]]] = None

        num_params = self.get_num_params()
        print(f"Number of parameters: {num_params / 1e6:.2f}M")
        if config.n_experts is not None and config.n_experts > 0:
            print("Note: Parameter count includes all experts. Only top_k experts are active per token.")

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rescale_out_projections(self):
        denom = math.sqrt(2 * self.config.n_layers)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, Attention):
                    m.o_proj.weight.mul_(1.0 / denom)
                elif isinstance(m, ReasoningFeedForward):
                    m.w2.weight.mul_(1.0 / denom)
                elif isinstance(m, MoCTopKExperts):
                    m.o_proj.weight.mul_(1.0 / denom)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]], Optional[Any]]:
        bsz, seqlen = idx.shape
        start_pos = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        assert (
            start_pos + seqlen <= self.config.max_seq_len
        ), f"Sequence length {start_pos + seqlen} exceeds max_seq_len {self.config.max_seq_len}"

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(device=x.device)

        new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_aux_loss = x.new_zeros(())
        expert_indices_list: List[torch.Tensor] = []
        routing_diagnostics: List[Dict[str, torch.Tensor]] = []

        for i, block in enumerate(self.transformer.h):
            past_kv_for_block = past_key_values[i] if past_key_values is not None else None
            x, new_kv, aux_loss, expert_indices = block(x, freqs_cis, past_kv_for_block)
            total_aux_loss = total_aux_loss + aux_loss.to(x.dtype)
            if expert_indices is not None:
                expert_indices_list.append(expert_indices)
            if block.last_routing_diagnostics is not None:
                routing_diagnostics.append(block.last_routing_diagnostics)
            new_past_key_values.append(new_kv)

        self.last_routing_diagnostics = routing_diagnostics if len(routing_diagnostics) > 0 else None

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) if targets is not None else self.lm_head(x[:, [-1], :])

        if targets is None:
            if self.config.return_routing_diagnostics:
                debug_payload = {"routing_diagnostics": self.last_routing_diagnostics}
                return logits, None, new_past_key_values, debug_payload
            return logits, None, new_past_key_values, None

        main_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )

        num_moe_layers = sum(1 for block in self.transformer.h if getattr(block, "is_moe", False))
        final_aux_loss = total_aux_loss / max(1, num_moe_layers)
        total_loss = main_loss + final_aux_loss.to(main_loss.dtype)
        loss = (total_loss, main_loss, final_aux_loss)

        if self.config.return_routing_diagnostics:
            debug_payload = {
                "expert_indices": [expert_indices_list],
                "routing_diagnostics": self.last_routing_diagnostics,
            }
            return logits, loss, new_past_key_values, debug_payload

        return logits, loss, new_past_key_values, [expert_indices_list]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        router_params, decay_params, nodecay_params = [], [], []
        router_param_ids = set()

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "feed_forward.gate" in n and isinstance(p, torch.nn.Parameter):
                router_params.append(p)
                router_param_ids.add(id(p))
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        # Keep compatibility with naming expectations.
        for n, p in self.named_parameters():
            if n.endswith("feed_forward.gate.weight") and id(p) not in router_param_ids:
                router_params.append(p)
                router_param_ids.add(id(p))

        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {sum(p.numel() for p in decay_params):,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {sum(p.numel() for p in nodecay_params):,} parameters"
        )
        print(
            f"num router parameter tensors: {len(router_params)}, "
            f"with {sum(p.numel() for p in router_params):,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        optim_kwargs = dict(lr=learning_rate, betas=betas)
        if fused_available:
            optim_kwargs["fused"] = use_fused

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
                {"params": router_params, "weight_decay": 0.0, "lr": learning_rate * 0.5},
            ],
            **optim_kwargs,
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()

        past_key_values = None
        for _ in range(max_new_tokens):
            current_len = past_key_values[0][0].shape[-2] if past_key_values else idx.shape[1]
            if current_len >= self.config.max_seq_len:
                break

            idx_cond = idx if past_key_values is None else idx[:, -1:]
            logits, _, past_key_values, _ = self(idx_cond, past_key_values=past_key_values)
            logits = logits[:, -1, :] / max(float(temperature), 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()
        return idx


# Backward-compatibility aliases requested by downstream codebases.
LunarisMoC = LunarisCodex
MoELayer = MoCTopKExperts


# -----------------------------------------------------------------------------
# Compile helper
# -----------------------------------------------------------------------------
def compile_model_if_available(model: nn.Module, mode: str = "max-autotune"):
    try:
        model = torch.compile(model, mode=mode)
        print(f"Model compiled with torch.compile ({mode}).")
    except Exception as e:
        print(f"torch.compile not enabled or failed: {e}")
    return model


if __name__ == "__main__":
    # Minimal shape sanity (torch must be available in runtime environment)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = LunarisCodexConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=503,
        multiple_of=16,
        ffn_hidden_multiplier=4.0,
        max_seq_len=64,
        dropout=0.0,
        n_experts=4,
        top_k=2,
        aux_loss_weight=1e-2,
        capacity_factor=1.25,
        router_z_loss_weight=1e-3,
        router_noise_std=0.02,
        drop_penalty_weight=1e-3,
        use_gradient_checkpointing=True,
        grad_ckpt_policy="ffn",
        use_simple_collab=False,
        use_moc_collab=True,
        moc_collab_steps=2,
        moc_use_mediator=True,
        n_reasoning_steps=3,
        adaptive_reasoning=True,
        adaptive_collaboration=True,
        adaptive_compute_mode="soft",
        return_routing_diagnostics=True,
    )

    model = LunarisCodex(cfg).to(device)

    bsz, seqlen = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (bsz, seqlen), device=device)
    targets = torch.randint(0, cfg.vocab_size, (bsz, seqlen), device=device)

    model.train()
    logits, loss_tuple, _, debug = model(idx, targets=targets)
    total_loss, ce_loss, aux_loss = loss_tuple
    print(f"Losses -> total: {total_loss.tolist():.4f}, ce: {ce_loss.tolist():.4f}, aux: {aux_loss.tolist():.6f}")
    print("Debug keys:", None if debug is None else list(debug.keys()))
    assert logits.shape == (bsz, seqlen, cfg.vocab_size)

    model.eval()
    out = model.generate(idx[:, :4], max_new_tokens=5)
    print("Generate output shape:", out.shape)
    print("Sanity complete.")
