"""
╔══════════════════════════════════════════════════════════════╗
║   MambaLM × Alpaca — Kaggle Training Script                 ║
║   Standalone file — no external local imports needed        ║
║                                                              ║
║   Dataset : tatsu-lab/alpaca (52K instruction examples)     ║
║   Model   : MambaLM (Attention-Free SSM)                    ║
║   Output  : /kaggle/working/out_alpaca/ckpt.pt              ║
║             /kaggle/working/out_alpaca/meta.json            ║
╚══════════════════════════════════════════════════════════════╝

HOW TO USE ON KAGGLE:
  1. Create new notebook → File → Upload → this file
  2. Settings → Accelerator → GPU T4 x2  (or P100)
  3. Run all cells
  4. After training: Download ckpt.pt + meta.json from /kaggle/working/out_alpaca/

TRAINING TIME RECOMMENDATION:
  ┌─────────────┬───────────┬──────────────────────────────┐
  │ Time        │ Steps     │ Quality                      │
  ├─────────────┼───────────┼──────────────────────────────┤
  │  3 hours    │  ~1,500   │ Basic — simple answers       │
  │  6 hours    │  ~3,000   │ Good — coherent responses    │
  │ 12 hours    │  ~6,000   │ Best — follow instructions   │
  └─────────────┴───────────┴──────────────────────────────┘
  → Recommended: run_hours = 11.5  (Kaggle max session = 12h)
"""

# ── 0. Install dependencies ──────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

# ── 1. Imports ───────────────────────────────────────────────────────────────
import os, json, time, math, signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset

# ── 2. Config ────────────────────────────────────────────────────────────────
RUN_HOURS    = 11.5       # ← Kaggle session ~12h max; set lower if you want less
BLOCK_SIZE   = 256        # context window
BATCH_SIZE   = 2          # per GPU (T4 has 16GB — can increase to 4 if no OOM)
ACCUM        = 8          # gradient accumulation steps
LR           = 3e-4
WEIGHT_DECAY = 0.01

# Mamba model size
D_MODEL  = 256
N_LAYER  = 6
D_STATE  = 16
D_CONV   = 4
EXPAND   = 2
DROPOUT  = 0.1

LOG_EVERY  = 50           # print loss every N steps
SAVE_EVERY = 300          # save checkpoint every N steps

OUT_DIR  = "/kaggle/working/out_alpaca"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
META_PATH = os.path.join(OUT_DIR, "meta.json")

# ── 3. Device ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TOKENIZER  (Character-level)
# ═══════════════════════════════════════════════════════════════════════════
class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi      = stoi
        self.itos      = itos
        self.vocab_size = len(stoi)

    @staticmethod
    def from_text(text: str):
        vocab = sorted(set(text))
        stoi  = {ch: i for i, ch in enumerate(vocab)}
        itos  = {i: ch for ch, i in stoi.items()}
        return CharTokenizer(stoi, itos)

    @staticmethod
    def from_vocab_map(vocab_map: dict):
        stoi = {str(k): int(v) for k, v in vocab_map.items()}
        itos = {int(v): str(k) for k, v in stoi.items()}
        return CharTokenizer(stoi, itos)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAMBA MODEL  (Attention-Free SSM)
# ═══════════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                  kernel_size=d_conv, groups=self.d_inner,
                                  padding=d_conv - 1, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz      = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        xc = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xc = F.silu(xc)

        y = self._ssm(xc)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x):
        B, L, D  = x.shape
        N        = self.d_state
        A        = -torch.exp(self.A_log.float())        # (D, N)

        xp       = self.x_proj(x)                        # (B, L, N*2+1)
        dt_r, Bv, Cv = xp.split([1, N, N], dim=-1)
        dt       = F.softplus(self.dt_proj(dt_r))        # (B, L, D)

        dA = torch.exp(dt.unsqueeze(-1) * A)             # (B, L, D, N)
        dB = dt.unsqueeze(-1) * Bv.unsqueeze(2)          # (B, L, D, N)

        h   = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys  = []
        for t in range(L):
            h    = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            ys.append((h * Cv[:, t].unsqueeze(1)).sum(-1))

        return torch.stack(ys, dim=1) + x * self.D


class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=6,
                 d_state=16, d_conv=4, expand=2,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop  = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": RMSNorm(d_model),
                "ssm" : MambaBlock(d_model, d_state, d_conv, expand),
            }) for _ in range(n_layer)
        ])
        self.norm_f  = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight          # weight tying
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        print(f"MambaLM  {n/1e6:.2f}M params  "
              f"(d_model={d_model}, n_layer={n_layer}, d_state={d_state})")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets=None):
        x = self.drop(self.embed(idx))
        for layer in self.layers:
            x = x + layer["ssm"](layer["norm"](x))
        x      = self.norm_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_c  = idx[:, -self.block_size:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _  = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


# ═══════════════════════════════════════════════════════════════════════════
# 6. ALPACA DATASET
# ═══════════════════════════════════════════════════════════════════════════
STOP_TAG = "\n### Instruction:"      # model stops here during inference

def format_example(ex: dict) -> str:
    """
    Convert one Alpaca example to training text.

    Format:
        ### Instruction:
        {instruction}

        ### Input:          ← only if input is non-empty
        {input}

        ### Response:
        {output}
    """
    parts = [f"### Instruction:\n{ex['instruction'].strip()}"]
    if ex.get("input", "").strip():
        parts.append(f"### Input:\n{ex['input'].strip()}")
    parts.append(f"### Response:\n{ex['output'].strip()}")
    return "\n\n".join(parts)


def build_corpus() -> str:
    print("Loading Alpaca dataset from HuggingFace...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  {len(ds):,} examples loaded")

    examples = []
    for ex in ds:
        text = format_example(ex)
        if len(text) > 20:
            examples.append(text)

    corpus = ("\n\n" + STOP_TAG.lstrip() + "\n").join(examples)
    print(f"  Corpus size: {len(corpus):,} chars  ({len(corpus)/1e6:.1f} MB)")
    return corpus


# ═══════════════════════════════════════════════════════════════════════════
# 7. TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def batchify(data: torch.Tensor, batch_size: int, block_size: int):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x  = torch.stack([data[i : i + block_size]     for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, data, iters=10):
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = batchify(data, BATCH_SIZE, BLOCK_SIZE)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_ckpt(model, opt, step):
    raw = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "model":      raw.state_dict(),
        "opt":        opt.state_dict(),
        "step":       step,
        "vocab_size": raw.vocab_size,
        "block_size": raw.block_size,
    }, CKPT_PATH)


def load_ckpt(model, opt):
    if not os.path.exists(CKPT_PATH):
        return 0
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if (ckpt.get("vocab_size") != model.vocab_size or
            ckpt.get("block_size") != model.block_size):
        print("⚠️  Checkpoint mismatch — starting fresh")
        return 0
    model.load_state_dict(ckpt["model"])
    try: opt.load_state_dict(ckpt["opt"])
    except Exception: pass
    step = int(ckpt.get("step", 0))
    print(f"🔄 Resumed from step {step}")
    return step


def save_meta(tok, config):
    meta = {"vocab": tok.stoi, "config": config}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"💾 meta saved → {META_PATH}")


def fmt(sec):
    if sec < 60:   return f"{sec:.0f}s"
    if sec < 3600: return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── Build dataset ──────────────────────────────────────────────────────
    corpus = build_corpus()
    tok    = CharTokenizer.from_text(corpus)
    ids    = torch.tensor(tok.encode(corpus), dtype=torch.long)

    n          = int(0.9 * len(ids))
    train_ids  = ids[:n]
    val_ids    = ids[n:]
    print(f"Vocab: {tok.vocab_size}  Train: {len(train_ids):,}  Val: {len(val_ids):,}")

    # ── Config ─────────────────────────────────────────────────────────────
    config = {
        "model_type": "mamba",
        "block_size": BLOCK_SIZE,
        "d_model":    D_MODEL,
        "n_layer":    N_LAYER,
        "d_state":    D_STATE,
        "d_conv":     D_CONV,
        "expand":     EXPAND,
        "dropout":    DROPOUT,
        "vocab_size": tok.vocab_size,
        "stop_tag":   STOP_TAG,
    }
    save_meta(tok, config)

    # ── Model ──────────────────────────────────────────────────────────────
    model = MambaLM(
        vocab_size = tok.vocab_size,
        d_model    = D_MODEL,
        n_layer    = N_LAYER,
        d_state    = D_STATE,
        d_conv     = D_CONV,
        expand     = EXPAND,
        block_size = BLOCK_SIZE,
        dropout    = DROPOUT,
    ).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    opt  = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step = load_ckpt(model, opt)

    # ── Ctrl+C handler ─────────────────────────────────────────────────────
    stop = {"flag": False}
    def on_sigint(sig, frame):
        stop["flag"] = True
        print("\n🛑 Stopping — saving checkpoint...")
    signal.signal(signal.SIGINT, on_sigint)

    # ── Training ───────────────────────────────────────────────────────────
    deadline = time.time() + RUN_HOURS * 3600
    t0       = time.time()
    model.train()

    print(f"\n{'='*55}")
    print(f"  Training for {RUN_HOURS}h on {DEVICE}")
    print(f"  Steps est: ~{int(RUN_HOURS * 3600 / 2):,}  (2s/step on T4)")
    print(f"{'='*55}\n")

    while time.time() < deadline and not stop["flag"]:
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(ACCUM):
            xb, yb = batchify(train_ids, BATCH_SIZE, BLOCK_SIZE)
            _, loss = model(xb, yb)
            (loss / ACCUM).backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % LOG_EVERY == 0:
            elapsed   = time.time() - t0
            remaining = max(0, deadline - time.time())
            avg_loss  = total_loss / ACCUM
            print(f"step {step:5d} | loss {avg_loss:.4f} | "
                  f"elapsed {fmt(elapsed)} | left {fmt(remaining)}")

        if step % SAVE_EVERY == 0:
            save_ckpt(model, opt, step)
            va = estimate_loss(model, val_ids)
            print(f"💾 saved  step {step} | val_loss {va:.4f}")

    # ── Final save ─────────────────────────────────────────────────────────
    save_ckpt(model, opt, step)
    print(f"\n✅ Training complete!")
    print(f"   Steps   : {step}")
    print(f"   Elapsed : {fmt(time.time() - t0)}")
    print(f"   ckpt    : {CKPT_PATH}")
    print(f"   meta    : {META_PATH}")
    print(f"\n→ Download both files from Kaggle Output panel")

    # ── Quick generation test ───────────────────────────────────────────────
    print("\n── Sample generation ──────────────────────────────────")
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    test_prompt = "### Instruction:\nWhat is a language model?\n\n### Response:\n"
    ids_t = torch.tensor([tok.encode(test_prompt)], dtype=torch.long, device=DEVICE)
    out   = raw_model.generate(ids_t, max_new_tokens=120, temperature=0.7, top_k=50)
    reply = tok.decode(out[0].tolist()[len(tok.encode(test_prompt)):])
    if STOP_TAG in reply:
        reply = reply.split(STOP_TAG)[0]
    print(f"Prompt : {test_prompt.strip()}")
    print(f"Reply  : {reply.strip()}")


if __name__ == "__main__":
    main()
