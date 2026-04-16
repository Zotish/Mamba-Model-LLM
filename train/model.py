# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CausalSelfAttention(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.n_head = n_head
#         self.head_dim = n_embd // n_head

#         self.qkv = nn.Linear(n_embd, 3 * n_embd)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(dropout)

#         # causal mask as buffer
#         mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
#         self.register_buffer("mask", mask)

#     def forward(self, x):
#         B, T, C = x.shape
#         qkv = self.qkv(x)  # (B, T, 3C)
#         q, k, v = qkv.split(C, dim=2)

#         # reshape to heads
#         q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
#         k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

#         att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
#         att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
#         att = F.softmax(att, dim=-1)
#         att = self.dropout(att)

#         y = att @ v  # (B, nh, T, hd)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
#         y = self.dropout(self.proj(y))
#         return y

# class MLP(nn.Module):
#     def __init__(self, n_embd: int, dropout: float):
#         super().__init__()
#         self.fc1 = nn.Linear(n_embd, 4 * n_embd)
#         self.fc2 = nn.Linear(4 * n_embd, n_embd)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

# class Block(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
#         self.ln2 = nn.LayerNorm(n_embd)
#         self.mlp = MLP(n_embd, dropout)

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x

# class MiniGPT(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,
#         block_size: int = 256,
#         n_layer: int = 4,
#         n_head: int = 4,
#         n_embd: int = 256,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.block_size = block_size

#         self.tok_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         self.drop = nn.Dropout(dropout)

#         self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.head = nn.Linear(n_embd, vocab_size, bias=False)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, mean=0.0, std=0.02)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Embedding):
#             nn.init.normal_(m.weight, mean=0.0, std=0.02)

#     def forward(self, idx, targets=None):
#         B, T = idx.shape
#         if T > self.block_size:
#             raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

#         pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
#         x = self.tok_emb(idx) + self.pos_emb(pos)
#         x = self.drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         x = self.ln_f(x)
#         logits = self.head(x)  # (B, T, vocab)

#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

#         return logits, loss

#     @torch.no_grad()
#     def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
#         for _ in range(max_new_tokens):
#             idx_cond = idx[:, -self.block_size :]
#             logits, _ = self(idx_cond)
#             logits = logits[:, -1, :] / max(temperature, 1e-6)

#             if top_k is not None and top_k > 0:
#                 v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
#                 logits[logits < v[:, [-1]]] = -float("inf")

#             probs = F.softmax(logits, dim=-1)
#             next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
#             idx = torch.cat([idx, next_id], dim=1)
#         return idx


# model.py
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CausalSelfAttention(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, dropout: float):
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.n_head = n_head
#         self.head_dim = n_embd // n_head

#         self.qkv = nn.Linear(n_embd, 3 * n_embd)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.attn_drop = nn.Dropout(dropout)
#         self.resid_drop = nn.Dropout(dropout)

#     def forward(self, x):
#         B, T, C = x.size()
#         qkv = self.qkv(x)  # (B,T,3C)
#         q, k, v = qkv.split(C, dim=2)

#         q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hd)
#         k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

#         att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nh,T,T)
#         mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
#         att = att.masked_fill(~mask, float("-inf"))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)

#         y = att @ v  # (B,nh,T,hd)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_drop(self.proj(y))
#         return y


# class MLP(nn.Module):
#     def __init__(self, n_embd: int, dropout: float):
#         super().__init__()
#         self.fc1 = nn.Linear(n_embd, 4 * n_embd)
#         self.fc2 = nn.Linear(4 * n_embd, n_embd)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return self.drop(x)

# class Block(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, dropout: float):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.attn = CausalSelfAttention(n_embd, n_head, dropout)
#         self.ln2 = nn.LayerNorm(n_embd)
#         self.mlp = MLP(n_embd, dropout)

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x

# class MiniGPT(nn.Module):
#     def __init__(self, vocab_size: int, block_size: int, n_layer=4, n_head=4, n_embd=256, dropout=0.1):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.block_size = block_size

#         self.tok_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         self.drop = nn.Dropout(dropout)
#         self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.head = nn.Linear(n_embd, vocab_size, bias=False)

#         self.apply(self._init)

#     def _init(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, mean=0.0, std=0.02)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         if isinstance(m, nn.Embedding):
#             nn.init.normal_(m.weight, mean=0.0, std=0.02)

#     def forward(self, idx, targets=None, loss_mask=None):
#         # idx: (B,T)
#         B, T = idx.size()
#         if T > self.block_size:
#             idx = idx[:, -self.block_size:]
#             if targets is not None:
#                 targets = targets[:, -self.block_size:]
#             if loss_mask is not None:
#                 loss_mask = loss_mask[:, -self.block_size:]

#         pos = torch.arange(0, idx.size(1), device=idx.device).unsqueeze(0)  # (1,T)
#         x = self.tok_emb(idx) + self.pos_emb(pos)
#         x = self.drop(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.ln_f(x)
#         logits = self.head(x)  # (B,T,V)

#         loss = None
#         if targets is not None:
#             # shift is handled by caller; this computes token loss at each position
#             flat_logits = logits.reshape(-1, logits.size(-1))
#             flat_targets = targets.reshape(-1)
#             ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # (B*T,)

#             if loss_mask is not None:
#                 m = loss_mask.reshape(-1).float()
#                 denom = m.sum().clamp(min=1.0)
#                 loss = (ce * m).sum() / denom
#             else:
#                 loss = ce.mean()

#         return logits, loss
#     @torch.no_grad()
#     def generate(
#         self,
#         idx: torch.Tensor,
#         max_new_tokens: int,
#         temperature: float = 1.0,
#         top_k: int = 0,
#     ):
#         """
#         Autoregressive text generation
#         idx: (1, T) token ids
#         """
#         self.eval()

#         for _ in range(max_new_tokens):
#             # context trim
#             idx_cond = idx[:, -self.block_size :]

#             # forward
#             logits, _ = self(idx_cond)

#             # last token logits
#             logits = logits[:, -1, :] / max(temperature, 1e-6)

#             # top-k filtering
#             if top_k > 0:
#                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#                 logits[logits < v[:, [-1]]] = -float("inf")

#             probs = F.softmax(logits, dim=-1)
#             next_id = torch.multinomial(probs, num_samples=1)

#             idx = torch.cat((idx, next_id), dim=1)

#         return idx




# train/model.py
# from __future__ import annotations

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CausalSelfAttention(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.n_head = n_head
#         self.head_dim = n_embd // n_head
#         self.block_size = block_size

#         self.qkv = nn.Linear(n_embd, 3 * n_embd)
#         self.proj = nn.Linear(n_embd, n_embd)

#         self.attn_drop = nn.Dropout(dropout)
#         self.resid_drop = nn.Dropout(dropout)

#         # causal mask (buffer)
#         mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
#         self.register_buffer("mask", mask)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, T, C = x.size()
#         qkv = self.qkv(x)  # (B,T,3C)
#         q, k, v = qkv.split(C, dim=2)

#         # (B, nh, T, hs)
#         q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

#         # scaled dot-product attention
#         att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

#         # causal mask to prevent looking ahead
#         att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)

#         y = att @ v  # (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_drop(self.proj(y))
#         return y


# class MLP(nn.Module):
#     def __init__(self, n_embd: int, dropout: float):
#         super().__init__()
#         self.fc = nn.Linear(n_embd, 4 * n_embd)
#         self.proj = nn.Linear(4 * n_embd, n_embd)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc(x)
#         x = F.gelu(x)
#         x = self.proj(x)
#         x = self.drop(x)
#         return x


# class Block(nn.Module):
#     def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
#         self.ln2 = nn.LayerNorm(n_embd)
#         self.mlp = MLP(n_embd, dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x


# class MiniGPT(nn.Module):
#     def __init__(
#         self,
#         vocab_size: int,
#         block_size: int,
#         n_layer: int = 4,
#         n_head: int = 4,
#         n_embd: int = 256,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.block_size = block_size

#         self.tok_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         self.drop = nn.Dropout(dropout)

#         self.blocks = nn.ModuleList(
#             [Block(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)]
#         )
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.head = nn.Linear(n_embd, vocab_size, bias=False)

#         self.apply(self._init_weights)

#     def _init_weights(self, module: nn.Module):
#         if isinstance(module, nn.Linear):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
#         B, T = idx.size()
#         if T > self.block_size:
#             idx = idx[:, -self.block_size :]
#             T = idx.size(1)
#             if targets is not None:
#                 targets = targets[:, -self.block_size :]

#         pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
#         x = self.tok_emb(idx) + self.pos_emb(pos)
#         x = self.drop(x)

#         for b in self.blocks:
#             x = b(x)

#         x = self.ln_f(x)
#         logits = self.head(x)  # (B,T,V)

#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
#         return logits, loss

#     @torch.no_grad()
#     def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
#         # idx: (B, T)
#         for _ in range(max_new_tokens):
#             idx_cond = idx[:, -self.block_size :]
#             logits, _ = self(idx_cond)
#             logits = logits[:, -1, :]  # last step (B,V)

#             if temperature <= 0:
#                 temperature = 1.0
#             logits = logits / temperature

#             if top_k and top_k > 0:
#                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#                 logits[logits < v[:, [-1]]] = -float("inf")

#             probs = F.softmax(logits, dim=-1)
#             next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
#             idx = torch.cat([idx, next_id], dim=1)
#         return idx





import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Option B: separate Q/K/V (এই architecture-এ train করলে ckpt ঠিকমতো load হবে)
        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask (T x T)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)               # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                                              # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                         # (B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
        super().__init__()
        self.block_size = int(block_size)

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(self.block_size, n_embd)
        self.drop    = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, self.block_size, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"T={T} exceeds block_size={self.block_size}")

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
