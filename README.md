<div align="center">

# Lunaris MoC — Mixture-of-Collaboration with Adaptive Compute

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Paper](https://img.shields.io/badge/Technical%20Report-PDF-red.svg)](paper/lunaris-moc.pdf)

**A decoder-only Transformer with collaborative expert routing, iterative reasoning, and optional adaptive per-token compute allocation.**

Developed independently by [Francisco Antonio](https://github.com/MeryylleA) — a 17-year-old researcher from Brazil 🇧🇷

[📄 Technical Report](paper/lunaris-moc.pdf) · [📈 Weights & Biases Dashboard](https://wandb.ai/smeryylle-moon-cloud-services-/lunaris-moc-validation) · [🤗 Datasets](https://huggingface.co/meryyllebr543)

</div>

## What Is This?

Lunaris MoC is a decoder-only Transformer architecture where selected experts **reason, collaborate, and then fuse their outputs**, while the model can optionally **learn how much computation each token needs**.

Standard Mixture-of-Experts (MoE) routes each token to K experts, runs them independently, and merges the results. MoC does something different: the selected experts exchange information through a learned mediator before fusion. Inside each expert, an Iterative Reasoning Loop (IRL) can refine the representation across multiple steps. When adaptive compute is enabled, the model learns per-token reasoning depth and collaboration depth instead of spending the same amount of compute on every token.

Core files in the repository:

- `model_moc.py` — the model architecture
- `train_moc.py` — the single-GPU training script with diagnostics and W&B logging
- `optimizer_lr.py` — parameter-group learning-rate utilities used by training

The `main` branch is intended to track the current implementation. Historical experiments and older code paths can live in separate branches, but the dashboard below is the easiest place to inspect raw runs and routing diagnostics.

---

## 📊 Experiments & Tracking

Latest runs, routing diagnostics, and ablation logs:

**W&B dashboard:** https://wandb.ai/smeryylle-moon-cloud-services-/lunaris-moc-validation

A few grounded takeaways from the current pilot experiments:

- In small-budget ablations, **MoC beats both the dense baseline and a standard MoE baseline** on validation loss / perplexity.
- The current implementation keeps routing healthy in pilot runs: no dead experts in the reported experiments, low drop rates, and non-collapsed utilization.
- A larger sparse run (~197M total trainable parameters) scaled without routing collapse and reached substantially lower validation perplexity than the earlier small-scale pilot.

Because experiments so far span different hardware and model scales, the W&B panel should be treated as the source of truth for raw results. Static screenshots age like milk, so this README now links directly to the dashboard instead of embedding images.

---

## Architecture

### The Forward Pass of a Single Token

```
Input token
    │
    ▼
┌─────────────┐
│  Embedding  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  Transformer Block (× N layers)                  │
│                                                  │
│  1. RMSNorm → Attention (RoPE + GQA) → Residual  │
│                                                  │
│  2. RMSNorm → MoC Block:                         │
│     ┌──────────────────────────────────────┐     │
│     │ Router (fp32) → select top-K experts │     │
│     │                                      │     │
│     │ For each selected expert:            │     │
│     │   IRL: h = h + α·FFN(h + x)          │     │
│     │   (repeated S times, S is learned)   │     │
│     │                                      │     │
│     │ Collaboration:                       │     │
│     │   Mediator attends over experts      │     │
│     │   Experts receive gated feedback     │     │
│     │   (repeated R times, R is learned)   │     │
│     │                                      │     │
│     │ Fusion:                              │     │
│     │   γ·mediator + (1-γ)·weighted_sum    │     │
│     │   (γ is learned per token)           │     │
│     └──────────────────────────────────────┘     │
│     → Residual                                   │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ RMSNorm → LM │
│    Head       │
└──────────────┘
```

### Component Details

#### Router

Each token is routed to the top-K experts based on learned logits computed in fp32 for numerical stability. The router uses softmax over the selected experts to produce probability weights. Three auxiliary losses keep the routing healthy:

- **Balance loss** encourages uniform expert utilization across the batch
- **Z-loss** regularizes the magnitude of router logits via `mean(logsumexp(logits)²)`
- **Drop penalty** adds cost proportional to tokens dropped by capacity limits

The router supports optional noise injection during training (`router_noise_std`) to encourage exploration.

#### Iterative Reasoning Loop (IRL)

Inside each expert, the hidden state is iteratively refined:

```python
h = x
for step in range(S):
    h = h + α * FFN(norm(h + x))
# α = 1/√S for magnitude control
```

The skip connection `h + x` anchors each step to the original input, preventing drift. The scaling factor `1/√S` controls magnitude growth across steps. This increases effective network depth without adding parameters — each expert with S=3 steps effectively performs 3× the computation of a standard FFN.

With **adaptive reasoning** enabled, the model learns a categorical distribution over `{0, 1, ..., S}` steps for each token independently. A small linear gate projects the input to `S+1` logits, and the resulting probabilities determine per-step activity. Tokens that need deeper processing receive more steps; trivial tokens can skip the loop entirely.

Two execution modes are supported:

| Mode | Behavior | torch.compile | Best For |
|------|----------|---------------|----------|
| `soft` | All steps run; outputs weighted by learned activity | ✓ Compatible | Training |
| `hard` | Only steps with activity > 0.5 execute | ✗ Incompatible | Inference |

A step penalty `(expected_steps / max_steps) × penalty_weight` prevents the model from always choosing maximum steps.

#### MoC-Lite Collaboration

After experts process their tokens, the collaboration mechanism lets them exchange information through a mediator. This is O(K) per token, not O(K²):

**Step 1 — Mediator reads from experts:**
```
query = project(mediator)
key   = project(expert_states)
value = project(expert_states)
attention_weights = softmax(query · key^T / √d)
message = weighted_sum(value, attention_weights)
mediator = mediator + MLP(mediator + message)
```

**Step 2 — Optional expert feedback:**
```
feedback = project(mediator)
gate = sigmoid(linear(expert) + linear(mediator))
expert = expert + gate * feedback
```

Each expert independently decides how much to accept from the mediator via the learned gate. Experts that are confident in their representation can close the gate; uncertain experts open it.

With **adaptive collaboration** enabled, the number of collaboration rounds is learned per token using the same mechanism as adaptive IRL — a gate network produces a distribution over `{0, 1, ..., R}` rounds.

**Mediator initialization:** The mediator starts as a learned parameter vector, conditioned on the current token via additive projection: `mediator = param + linear(x)`.

**Fusion:** The final output combines mediator and weighted expert states:
```python
gamma = sigmoid(linear(x))
output = gamma * mediator + (1 - gamma) * weighted_expert_sum
```

#### Capacity Control

Each expert has a capacity limit: `capacity = ceil((N × K / E) × capacity_factor)` where N is tokens, K is top-k, E is number of experts. Token-expert pairs exceeding capacity are dropped based on priority (router probability). Dropped tokens fall back to the input representation. The capacity is rounded up to the nearest power of two for memory allocator efficiency.

### Backbone

- **Normalization:** Pre-LN with RMSNorm
- **Positional encoding:** Rotary Position Embeddings (RoPE)
- **Attention:** Grouped Query Attention (GQA) via PyTorch SDPA with automatic backend selection (Flash Attention → Memory Efficient → Math)
- **FFN:** SwiGLU activation with `2/3 × 4d` hidden dimension, rounded to `multiple_of`
- **Weight tying:** Embedding and LM head weights are shared
- **Initialization:** Normal(0, 0.02) with output projection rescaling by `1/√(2·n_layers)`

---

## Configuration Reference

All model configuration is handled through `LunarisCodexConfig`. Training configuration is loaded from YAML via `TrainConfig.from_yaml()`.

### Model Configuration

#### Core Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 768 | Model hidden dimension |
| `n_layers` | 12 | Number of transformer blocks |
| `n_heads` | 12 | Number of attention heads |
| `n_kv_heads` | 12 | Number of key/value heads (set < n_heads for GQA) |
| `vocab_size` | 50257 | Vocabulary size |
| `max_seq_len` | 2048 | Maximum sequence length |
| `multiple_of` | 256 | FFN hidden dim rounded to this multiple |
| `ffn_hidden_multiplier` | 4.0 | FFN hidden dimension multiplier |
| `rope_theta` | 10000.0 | RoPE frequency base |
| `dropout` | 0.0 | Dropout rate (0.0 recommended for pre-training) |

#### Expert Routing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_experts` | 8 | Number of experts (set to `null` for dense model) |
| `top_k` | 2 | Number of experts selected per token |
| `capacity_factor` | 1.25 | Expert capacity multiplier (higher = fewer drops, more memory) |
| `aux_loss_weight` | 1e-2 | Weight for load balance loss |
| `router_z_loss_weight` | 1e-3 | Weight for router logit regularization |
| `drop_penalty_weight` | 1e-3 | Weight for capacity overflow penalty |
| `router_noise_std` | 0.0 | Gaussian noise added to router logits during training |

#### Iterative Reasoning Loop

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_reasoning_steps` | 1 | Maximum IRL steps per expert (1 = standard FFN) |
| `adaptive_reasoning` | true | Learn per-token step count (requires `n_reasoning_steps` > 1) |
| `reasoning_gate_temperature` | 1.0 | Temperature for step gate softmax |
| `reasoning_gate_noise_std` | 0.0 | Noise for step gate exploration |
| `reasoning_step_penalty_weight` | 0.0 | Penalty for using more steps (prevents always-max) |

#### Collaboration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_moc_collab` | true | Enable MoC-Lite collaboration |
| `use_simple_collab` | false | Use simple weighted average instead (no mediator) |
| `moc_collab_steps` | 2 | Maximum collaboration rounds |
| `moc_use_mediator` | true | Use learned mediator parameter |
| `moc_expert_feedback` | true | Enable bidirectional mediator→expert feedback |
| `moc_low_rank_message_dim` | 0 | Low-rank projection for expert values (0 = full rank) |
| `moc_collab_dropout` | 0.0 | Dropout in collaboration layers |
| `adaptive_collaboration` | true | Learn per-token collaboration rounds |
| `collaboration_gate_temperature` | 1.0 | Temperature for collaboration gate softmax |
| `collaboration_gate_noise_std` | 0.0 | Noise for collaboration gate exploration |
| `collaboration_step_penalty_weight` | 0.0 | Penalty for using more collaboration rounds |

#### Adaptive Compute

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adaptive_compute_mode` | `"soft"` | `"soft"` (compile-friendly, weighted) or `"hard"` (sparse execution) |

When both `adaptive_reasoning` and `adaptive_collaboration` are enabled, the model learns to allocate computation across two dimensions independently: how deeply each expert processes its tokens (IRL) and how much experts communicate (collaboration). The combined compute savings are reported in training logs as "Compute saved %".

#### Engineering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_gradient_checkpointing` | true | Enable activation checkpointing |
| `grad_ckpt_policy` | `"ffn"` | `"none"`, `"ffn"` (checkpoint FFN only), or `"block"` (full block) |
| `track_routing_stats` | true | Compute routing diagnostics per forward pass |
| `return_routing_diagnostics` | false | Return diagnostics in model output (set true for training) |
| `save_attn_weights` | false | Save collaboration attention weights (debug only) |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `"data"` | Directory containing `.npy` token shards |
| `learning_rate` | 3e-4 | Peak learning rate |
| `weight_decay` | 0.1 | AdamW weight decay |
| `beta1` / `beta2` | 0.9 / 0.95 | Adam beta parameters |
| `warmup_steps` | 2000 | Linear warmup steps |
| `max_steps` | 600000 | Total training steps |
| `batch_size` | 16 | Per-device batch size |
| `gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `grad_clip` | 1.0 | Maximum gradient norm |
| `compile_model` | true | Use `torch.compile` (NVIDIA only) |
| `device` | `"cuda"` | Training device |
| `out_dir` | `"checkpoints"` | Checkpoint output directory |
| `log_interval` | 20 | Steps between terminal/wandb logs |
| `save_interval` | 1000 | Steps between checkpoints |
| `val_interval` | 500 | Steps between validation runs |
| `val_batches` | 50 | Number of validation batches |
| `early_stopping_patience` | 0 | Validations without improvement before stopping (0 = disabled) |
| `wandb_project` | `"lunaris-codex-moc"` | Weights & Biases project name (`null` = disabled) |

---

## Training Guide

### Data Format

The training script expects pre-tokenized data stored as NumPy `.npy` arrays of token IDs. The data loader memory-maps shards and emits fixed-length sequences.

**Directory structure:**

```
data/
├── shard_0000.npy      # Training data
├── shard_0001.npy
├── ...
└── val/
    └── shard_0000.npy  # Validation data
```

**Creating shards:**

1. Tokenize your corpus with any tokenizer (BPE, SentencePiece, tiktoken, etc.)
2. Append an end-of-text token after each document
3. Concatenate all token IDs into a flat array
4. Split into shard files of ~500M-1B tokens each
5. Save as `.npy` with an integer dtype (`uint16` for vocab < 65536, `uint32` otherwise)

The data loader handles cross-shard boundaries automatically — sequences can span two adjacent shards.

### Running Training

**Basic usage:**

```bash
python train_moc.py config.yaml
```

**Example configuration (small validation run):**

```yaml
model:
  d_model: 384
  n_layers: 6
  n_heads: 6
  n_kv_heads: 6
  vocab_size: 65536
  multiple_of: 64
  max_seq_len: 512
  dropout: 0.0

  n_experts: 4
  top_k: 2
  capacity_factor: 1.5
  aux_loss_weight: 0.01
  router_z_loss_weight: 0.001
  router_noise_std: 0.02
  drop_penalty_weight: 0.001

  n_reasoning_steps: 2
  adaptive_reasoning: true
  adaptive_compute_mode: "soft"
  reasoning_step_penalty_weight: 0.001

  use_moc_collab: true
  moc_collab_steps: 2
  moc_use_mediator: true
  moc_expert_feedback: true
  adaptive_collaboration: true
  collaboration_step_penalty_weight: 0.001

  use_gradient_checkpointing: false

data_dir: "data"
learning_rate: 3.0e-4
weight_decay: 0.1
warmup_steps: 500
max_steps: 20000
batch_size: 16
gradient_accumulation_steps: 2
grad_clip: 1.0
compile_model: false
device: "cuda"
out_dir: "checkpoints/moc"
log_interval: 100
save_interval: 5000
val_interval: 2000
val_batches: 20
wandb_project: null
```

### Resuming Training

Training automatically resumes from `latest_checkpoint.pt` if it exists in the output directory. The checkpoint contains model weights, optimizer state, step counter, RNG states, and training metadata. No additional flags are needed — just run the same command again.

### Monitoring

The training script outputs a formatted monitor block every `log_interval` steps:

```
╔══════════════════════════════════════════════════════════════════════════╗
║  LUNARIS MoC Training Monitor                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Step: 10000/20000  |  Epoch: 0  |  LR: 1.57e-04  |  ETA: 1h 52m         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  LOSSES                                                                  ║
║    Total: 4.30  |  CE: 4.29  |  Aux: 0.010  |  PPL: 73.06                ║
║    Val: 4.21 | Val PPL: 66.55                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  ADAPTIVE COMPUTE (averaged across layers)                               ║
║    IRL depth: 2.50/3  |  Collab rounds: 0.43/2  |  Compute saved: 41.5%  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  ROUTING (layer 0 / layer 3 / layer 5)                                   ║
║    Entropy: 1.17 / 0.94 / 1.07                                          ║
║    Drop rate: 0.00 / 0.00 / 0.00                                        ║
║    Expert util: [0.30 0.22 0.24 0.24]  (layer 0)                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  MEMORY & GRAD                                                           ║
║    VRAM: 15.2 / 22.1 GiB  |  Grad norm: 1.023  |  Grad scale: 1.0       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Key metrics to watch:**

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| Router entropy | > 1.0 (for 4 experts) | < 0.5 indicates expert collapse |
| Drop rate | < 5% | > 20% means capacity_factor is too low |
| IRL depth | Between 0 and max | Always 0 or always max means penalty is miscalibrated |
| Collab rounds | Between 0 and max | Same as above |
| Compute saved | 10-60% | 0% or >90% suggests gates are not learning |
| Grad norm | 0.5-2.0 | Sustained > 5.0 or < 0.01 indicates instability |

**Entropy color coding in terminal:**
- 🟢 Green: entropy ≥ 1.0 (healthy diversity)
- 🔴 Red: entropy < 1.0 (increasing specialization — monitor but not necessarily bad)

### Weights & Biases Integration

Enable W&B logging by setting `wandb_project` in the config:

```yaml
wandb_project: "lunaris-moc-validation"
wandb_run_name: "moc-384d-4exp-adaptive"
```

Public dashboard used for current experiments:

**https://wandb.ai/smeryylle-moon-cloud-services-/lunaris-moc-validation**

The following metrics are logged automatically:

- Loss curves (total, cross-entropy, auxiliary)
- Perplexity (train and validation)
- Per-layer routing entropy, drop rate, expert utilization histograms
- Per-layer Gini coefficient (routing inequality)
- Adaptive compute: IRL depth and collaboration round distributions
- Compute efficiency percentage
- Expert co-occurrence heatmaps (layer 0)
- Inter-layer routing agreement
- Fusion gate (gamma) values per layer
- Throughput, memory, gradient statistics

---

## Recommended Hardware

This training script is **single-GPU only**. No multi-GPU or distributed training is implemented.

### GPU Requirements by Model Size

| Total Params | Active/Token | Recommended GPU | Batch Size | Notes |
|-------------|-------------|-----------------|------------|-------|
| ~35M (dense baseline) | 35M | Any GPU ≥ 8GB | 8-32 | RTX 3060 and up |
| ~65M (4 experts, top-2) | ~50M | ≥ 16GB | 8-16 | RTX 4080, A10, V100 |
| ~65M + collab + IRL | ~50M | ≥ 20GB | 8-16 | A10 (24GB), RTX 4090 |
| ~200M (8 experts, top-2) | ~120M | ≥ 24GB | 4-8 | A10, A100 (40GB) |
| ~500M+ | ~250M+ | ≥ 40GB | 2-8 | A100 (40/80GB), H100 |

**VRAM considerations for MoC:**

The collaboration mechanism allocates intermediate tensors for mediator states, attention over experts, and expert feedback. This adds ~30-50% memory overhead compared to standard MoE with the same expert count. If you encounter OOM:

1. Reduce `batch_size` and increase `gradient_accumulation_steps` (keeps effective batch the same)
2. Enable `use_gradient_checkpointing: true` with `grad_ckpt_policy: "ffn"`
3. Reduce `capacity_factor` (increases drops but saves memory)
4. Use `moc_low_rank_message_dim` to compress collaboration messages

### AMD GPU Support

Training has been tested on AMD GPUs via ROCm. Set the following environment variable before running:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RDNA 2 (RX 6000 series)
```

Notes for AMD:
- Set `compile_model: false` (torch.compile is unreliable on ROCm consumer GPUs)
- bf16 is supported on RDNA 3+ (RX 7000 series); RDNA 2 uses fp16 automatically
- Flash Attention may not be available; the code falls back to the math SDPA backend
- VRAM reporting via `torch.cuda.memory_allocated()` may be inaccurate on ROCm; use system tools like `rocm-smi` or sysfs for accurate readings

### CPU Training

For architecture validation without a GPU, set `device: "cpu"` in the config. Training will run in fp32 (no mixed precision). Reduce `max_steps` and `batch_size` accordingly. Expect ~50-100× slower throughput compared to a modern GPU.

---

## Training Practices

### Hyperparameter Guidelines

**Learning rate:** Start with `3e-4` for models under 200M parameters. For larger models, reduce to `1e-4` or `6e-5`. The scheduler uses linear warmup followed by cosine decay to 1% of peak.

**Auxiliary loss weights:** The defaults (`aux_loss_weight: 0.01`, `router_z_loss_weight: 0.001`) work well across scales. If you observe expert collapse (entropy dropping below 0.5 in any layer), increase `aux_loss_weight` to 0.02. If auxiliary losses dominate total loss (aux > 10% of CE), reduce the weights.

**Adaptive compute penalties:** Start with `0.001` for both `reasoning_step_penalty_weight` and `collaboration_step_penalty_weight`. If the model always uses maximum steps (compute saved = 0%), increase to `0.005`. If it barely uses any steps (compute saved > 80%), decrease to `0.0005`. The goal is 20-50% compute savings.

**Capacity factor:** `1.25` for training, `1.5` for stability during fine-tuning. Higher values waste memory but prevent token drops. Zero drops is ideal; under 5% is acceptable.

### Running Ablation Experiments

To isolate the contribution of each MoC component, train models with identical configurations except for one variable:

| Experiment | n_experts | n_reasoning_steps | use_moc_collab | adaptive | Tests |
|------------|-----------|-------------------|----------------|----------|-------|
| Dense baseline | null | 1 | false | false | Baseline transformer |
| MoE only | 4 | 1 | false | false | Expert routing alone |
| + IRL | 4 | 2-3 | false | false | Iterative reasoning contribution |
| + Collaboration | 4 | 1 | true | false | Collaboration contribution |
| Full MoC | 4 | 2-3 | true | false | Combined effect |
| Full MoC + Adaptive | 4 | 2-3 | true | true | Adaptive compute contribution |

Keep all other parameters identical: same data, same batch size, same learning rate, same number of steps. Compare using **validation cross-entropy loss** (not total loss, which includes auxiliary terms).

For speed claims, keep the hardware fixed as well. A faster GPU can make a training script look magically better even when the architecture is unchanged — delightful for morale, terrible for attribution.

### Interpreting Routing Diagnostics

**Expert utilization:** Watch the Gini coefficient per layer. Values below 0.3 indicate healthy distribution. Values above 0.5 suggest dominant experts. Some specialization is expected and healthy — the concern is when experts die (receive zero tokens for multiple log windows).

**Layer-wise patterns:** It is normal for middle layers to show more routing specialization (lower entropy) than early or late layers. Early layers tend to route more uniformly because they process general features. Middle layers specialize on content types. Late layers often re-broaden.

**Adaptive compute patterns:** During early training, the gates may fluctuate. By ~20% of training, they typically stabilize. If IRL depth stabilizes near maximum and collaboration near zero (or vice versa), this is the model discovering which component provides more value at your scale.

**Dead expert warnings:** The training script alerts when an expert receives zero tokens for 3 consecutive log windows. Occasional warnings during early training are normal. Persistent dead experts indicate routing collapse — increase `aux_loss_weight` or `router_noise_std`.

---

## Inference

### Basic Generation

```python
import torch
from model_moc import LunarisCodex, LunarisCodexConfig

# Load from checkpoint
checkpoint = torch.load("checkpoints/moc/best_checkpoint.pt", map_location="cuda")
config = LunarisCodexConfig(**checkpoint["config"]["model"])
model = LunarisCodex(config).cuda().eval()

# Clean state dict (handles DDP/compile prefixes)
state_dict = checkpoint["model"]
cleaned = {}
for k, v in state_dict.items():
    clean_key = k.replace("module.", "").replace("_orig_mod.", "")
    cleaned[clean_key] = v
model.load_state_dict(cleaned)

# Generate
prompt_tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")  # your tokenized prompt
output = model.generate(prompt_tokens, max_new_tokens=128, temperature=0.8, top_k=50)
```

### Inference Optimization Tips

- Use `adaptive_compute_mode: "hard"` for faster sparse execution during inference (skips inactive steps entirely instead of multiplying by near-zero weights)
- The KV cache is fully supported — generation uses cached key/value states automatically
- For latency-sensitive applications, consider reducing `n_reasoning_steps` and `moc_collab_steps` at inference time via config modification before loading

---

## Technical Notes

### Loss Components

| Loss | Formula | Purpose |
|------|---------|---------|
| Cross-entropy | Standard next-token prediction | Primary training objective |
| Balance loss | `sum(prob_mass × fraction_tokens) × n_experts` | Uniform expert utilization |
| Z-loss | `mean(logsumexp(logits)²)` | Router logit regularization |
| Drop penalty | `fraction_dropped × weight` | Discourage capacity overflow |
| IRL step penalty | `mean(expected_steps / max_steps) × weight` | Discourage always-max reasoning |
| Collab step penalty | `mean(expected_steps / max_steps) × weight` | Discourage always-max collaboration |

The total loss is: `CE + aux_weight × balance + z_weight × z_loss + drop_penalty + irl_penalty + collab_penalty`

Auxiliary losses are averaged across MoE layers before being added to CE.

### Numerical Stability

- Router logits are computed in fp32 regardless of model precision
- Capacity masking uses stable argsort for deterministic token ordering
- RoPE embeddings use complex arithmetic in fp32, cast back to model dtype
- Gradient checkpointing uses `use_reentrant=False` for correctness with autograd

### Parameter Counting

The model reports two parameter counts:

- **Total parameters:** All trainable parameters, including all experts
- **Active parameters per token:** Parameters used for a single token, accounting for top-K expert selection

For a model with E experts and top-K routing, `active ≈ total - (E - K) × params_per_expert`. The ratio `active/total` indicates parameter efficiency — lower ratios mean more total capacity with less per-token compute.

---

## Project Status

- [x] Architecture implementation (MoC + IRL + adaptive compute)
- [x] Production training script with full diagnostics
- [x] Technical paper
- [ ] Large-scale pre-training validation
- [ ] Downstream task evaluation
- [ ] Multi-GPU training support

---

## Citation

```bibtex
@article{LunarisMoC,
  title={Lunaris MoC: Mixture-of-Collaboration with Iterative Reasoning Loops},
  author={Auren Research},
  year={2026},
  url={https://github.com/MeryylleA/lunariscodex-MoC}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
