"""
MambaLM — Attention-free Language Model using Selective State Space Model (SSM).

Key difference from Transformer/GPT:
  - NO attention mechanism (no Q, K, V, no O(L²) cost)
  - Uses a selective state space scan — each token is processed via a
    learnable recurrence whose parameters depend on the INPUT itself.
  - This makes it fundamentally different from both RNNs (fixed weights)
    and Transformers (all-pairs attention).

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
           State Spaces" (2023).

Architecture per layer:
    x (B, L, d_model)
        │
    RMSNorm
        │
    MambaBlock:
        ├─ in_proj  → split into [x_ssm, z_gate]
        ├─ DepthwiseConv1d(x_ssm)  (local context)
        ├─ SSM selective scan(x_ssm) → y
        ├─ y * SiLU(z_gate)         (gating)
        └─ out_proj
        │
    + residual
        │
    (repeat n_layer times)
        │
    RMSNorm → LM Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm  (used in Llama, Mamba — faster than LayerNorm, no mean subtraction)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# MambaBlock  (the core SSM block)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Selective State Space block.

    Parameters
    ----------
    d_model  : model dimension
    d_state  : SSM hidden state size (N in the paper, typically 16)
    d_conv   : depthwise conv kernel size (typically 4)
    expand   : inner dim factor  (d_inner = d_model * expand)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand          # D in the paper

        # ── Input projection ──────────────────────────────────────────────
        # Project to 2 * d_inner: first half goes through SSM, second is gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ── Depthwise Conv1d (causal local mixing) ────────────────────────
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,        # depthwise
            padding=d_conv - 1,         # causal: trim right later
            bias=True,
        )

        # ── SSM input-dependent projections ──────────────────────────────
        # Projects x → [B (N), C (N), dt (1)]  (all input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # dt rank→1 → d_inner  (learned time-step scaling)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # ── SSM fixed parameters ──────────────────────────────────────────
        # A: (d_inner, d_state) — log-space, always negative after exp
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_init = A_init.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A_init))
        self.A_log._no_weight_decay = True          # type: ignore[attr-defined]

        # D: skip connection coefficient  (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True              # type: ignore[attr-defined]

        # ── Output projection ─────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model)
        returns : (B, L, d_model)
        """
        B, L, _ = x.shape

        # ── 1. Input projection + split ───────────────────────────────────
        xz = self.in_proj(x)                             # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                  # each (B, L, d_inner)

        # ── 2. Causal depthwise conv ──────────────────────────────────────
        x_conv = x_ssm.transpose(1, 2)                  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]          # trim future padding
        x_conv = F.silu(x_conv.transpose(1, 2))         # (B, L, d_inner)

        # ── 3. Selective SSM scan ─────────────────────────────────────────
        y = self._ssm(x_conv)                            # (B, L, d_inner)

        # ── 4. Gating ─────────────────────────────────────────────────────
        y = y * F.silu(z)

        return self.out_proj(y)                          # (B, L, d_model)

    # ------------------------------------------------------------------
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective scan (sequential, O(L·D·N)).
        Inputs are *input-dependent*, unlike classical fixed-weight SSMs.

        x : (B, L, d_inner)
        """
        B_batch, L, D = x.shape
        N = self.d_state

        # A is always negative (stability)
        A = -torch.exp(self.A_log.float())               # (D, N)

        # ── Input-dependent B, C, dt ──────────────────────────────────────
        x_proj = self.x_proj(x)                          # (B, L, N*2+1)
        dt_raw, B_vals, C_vals = x_proj.split([1, N, N], dim=-1)
        # dt_raw: (B, L, 1), B_vals: (B, L, N), C_vals: (B, L, N)

        dt = F.softplus(self.dt_proj(dt_raw))            # (B, L, D)

        # ── Zero-Order Hold discretization ───────────────────────────────
        # dA[b, t] = exp(dt[b,t] * A)  →  (B, L, D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A)

        # dB[b, t] = dt[b,t] * B[b,t]  →  (B, L, D, N)
        dB = dt.unsqueeze(-1) * B_vals.unsqueeze(2)

        # ── Sequential causal scan ────────────────────────────────────────
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            # h: (B, D, N)
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # y: (B, D)  via C projection
            y_t = (h * C_vals[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                  # (B, L, D)

        # Skip connection (D parameter)
        return y + x * self.D


# ---------------------------------------------------------------------------
# MambaLM  (full language model)
# ---------------------------------------------------------------------------

class MambaLM(nn.Module):
    """
    Mamba Language Model — fully attention-free.

    Compatible interface with MiniGPT:
      forward(idx, targets=None)  →  (logits, loss)
      generate(idx, max_new_tokens, temperature, top_k)  →  idx
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": RMSNorm(d_model),
                "ssm":  MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand),
            })
            for _ in range(n_layer)
        ])

        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (embedding ↔ lm_head)
        self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)
        print(f"MambaLM  params: {self._count_params()/1e6:.2f}M  "
              f"(d_model={d_model}, n_layer={n_layer}, d_state={d_state})")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """
        idx     : (B, T)  token indices
        targets : (B, T)  next-token targets  (optional, for training)
        returns : (logits, loss)
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        x = self.drop(self.embedding(idx))      # (B, T, d_model)

        for layer in self.layers:
            x = x + layer["ssm"](layer["norm"](x))   # pre-norm + residual

        x = self.norm_f(x)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation — same interface as MiniGPT."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx
