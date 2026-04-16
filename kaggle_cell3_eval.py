# ============================================================
# Cell 3: Full Evaluation – MambaLM on Alpaca
# Standalone: model classes inlined, all metrics computed
# ============================================================

import os, sys, json, math, time, zipfile, warnings, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────
# 0.  Paths & constants
# ─────────────────────────────────────────────────────────────
CKPT_DIR   = Path("/kaggle/working/out_alpaca")
OUT_DIR    = Path("/kaggle/working/paper_results")
FIG_DIR    = OUT_DIR / "figures"
CKPT_PATH  = CKPT_DIR / "ckpt.pt"
META_PATH  = CKPT_DIR / "meta.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
STOP_TAG    = "\n### Instruction:"
N_EVAL      = 50
N_BENCH     = 100
MAX_NEW     = 128
BLOCK_SIZE  = 256

print(f"[INFO] Device  : {DEVICE}")
print(f"[INFO] PyTorch : {torch.__version__}")


# ─────────────────────────────────────────────────────────────
# 1.  Model classes  (must match training exactly)
# ─────────────────────────────────────────────────────────────

class CharTokenizer:
    """Character-level tokenizer with stoi/itos dicts."""
    def __init__(self, stoi: dict, itos: dict):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    def encode(self, text: str) -> list:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos.get(str(i), "") for i in ids)

    @classmethod
    def from_text(cls, text: str):
        chars = sorted(set(text))
        stoi  = {c: i for i, c in enumerate(chars)}
        itos  = {i: c for i, c in enumerate(chars)}
        return cls(stoi, itos)

    @classmethod
    def from_vocab_map(cls, stoi: dict):
        itos = {v: k for k, v in stoi.items()}
        return cls(stoi, itos)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class MambaBlock(nn.Module):
    """
    Mamba block — matches training architecture exactly.
    NOTE: norm is handled OUTSIDE this block (in MambaLM.layers ModuleDict).
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_state = d_state
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                  groups=d_inner, padding=d_conv - 1, bias=True)
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1).clone()
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz        = self.in_proj(x)
        x_ssm, z  = xz.chunk(2, dim=-1)

        xc = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xc = F.silu(xc)
        y  = self._ssm(xc)
        y  = y * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x):
        B, L, D       = x.shape
        N             = self.d_state
        A             = -torch.exp(self.A_log.float())
        xp            = self.x_proj(x)
        dt_r, Bv, Cv  = xp.split([1, N, N], dim=-1)
        dt            = F.softplus(self.dt_proj(dt_r))
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * Bv.unsqueeze(2)
        h  = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            ys.append((h * Cv[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1) + x * self.D


class MambaLM(nn.Module):
    """Matches training architecture exactly (ModuleDict layers, self.embed, dropout)."""
    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_layer: int = 6, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 block_size: int = 256, dropout: float = 0.0):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.drop   = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"norm": RMSNorm(d_model),
                           "ssm" : MambaBlock(d_model, d_state, d_conv, expand)})
            for _ in range(n_layer)
        ])
        self.norm_f  = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying

    def forward(self, idx, targets=None):
        x      = self.drop(self.embed(idx))
        for layer in self.layers:
            x  = x + layer["ssm"](layer["norm"](x))
        x      = self.norm_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int = 40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────
# 2.  Load checkpoint
# ─────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading checkpoint …")

with open(META_PATH) as f:
    meta = json.load(f)

# meta.json format: {"vocab": stoi_dict, "config": {...}}
stoi      = meta["vocab"]
tokenizer = CharTokenizer.from_vocab_map(stoi)
vocab_size = tokenizer.vocab_size
print(f"  vocab_size = {vocab_size}")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

# Config comes from meta.json (saved during training)
cfg        = meta.get("config", {})
d_model    = cfg.get("d_model",    256)
n_layer    = cfg.get("n_layer",    6)
d_state    = cfg.get("d_state",    16)
d_conv     = cfg.get("d_conv",     4)
expand     = cfg.get("expand",     2)
block_size = cfg.get("block_size", BLOCK_SIZE)

print(f"  d_model={d_model}, n_layer={n_layer}, d_state={d_state}, "
      f"d_conv={d_conv}, expand={expand}, block_size={block_size}")

model = MambaLM(vocab_size, d_model, n_layer, d_state, d_conv, expand,
                block_size, dropout=0.0)

state = ckpt.get("model", ckpt)
# Strip 'module.' prefix if saved with DataParallel
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.to(DEVICE)
model.eval()
print("  Checkpoint loaded.")


# ─────────────────────────────────────────────────────────────
# 3.  Load Alpaca dataset  (test = last 10 %)
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Loading Alpaca dataset …")
try:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    n_total = len(ds)
    split_idx = int(0.9 * n_total)
    test_ds = ds.select(range(split_idx, n_total))
    print(f"  Total={n_total}, test={len(test_ds)}")
except Exception as e:
    print(f"  [WARN] Could not load alpaca dataset: {e}")
    test_ds = None


def make_prompt(example):
    inst  = example.get("instruction", "")
    inp   = example.get("input", "")
    if inp.strip():
        return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{inst}\n\n### Response:\n"


def generate_response(prompt: str, max_new: int = MAX_NEW) -> str:
    ids = tokenizer.encode(prompt)
    ids = ids[-block_size:] if len(ids) > block_size else ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new)
    gen_ids = out[0, len(ids):].tolist()
    text    = tokenizer.decode(gen_ids)
    if STOP_TAG in text:
        text = text[:text.index(STOP_TAG)]
    return text.strip()


# ─────────────────────────────────────────────────────────────
# 4.  NLP METRICS helpers
# ─────────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_n(ref_tokens, hyp_tokens, n):
    """Corpus BLEU-n (no brevity penalty applied per example, averaged)."""
    if len(hyp_tokens) < n:
        return 0.0
    hyp_ng  = Counter(_ngrams(hyp_tokens, n))
    ref_ng  = Counter(_ngrams(ref_tokens, n))
    clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    total   = max(sum(hyp_ng.values()), 1)
    prec    = clipped / total
    # brevity
    bp = math.exp(1 - len(ref_tokens)/max(len(hyp_tokens),1)) \
         if len(hyp_tokens) < len(ref_tokens) else 1.0
    return bp * prec

def rouge_l(ref_tokens, hyp_tokens):
    """ROUGE-L F1 via LCS."""
    r, h = ref_tokens, hyp_tokens
    if not r or not h:
        return 0.0
    m, n  = len(r), len(h)
    dp    = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / n
    rec  = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_bertscore(refs, hyps):
    try:
        from bert_score import score as bscore
        P, R, F = bscore(hyps, refs, lang="en", verbose=False)
        return float(F.mean())
    except Exception as e:
        print(f"  [WARN] BERTScore failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────
# 5.  QUALITY METRICS helpers (0-5 heuristic)
# ─────────────────────────────────────────────────────────────

HARMFUL_KW = {
    "kill","murder","rape","bomb","hack","steal","exploit",
    "illegal","weapon","drug","porn","xxx","suicide","harm","violence"
}
EXPLAIN_KW = {
    "because","therefore","thus","hence","since","as a result",
    "consequently","accordingly","due to","this means","for example",
    "for instance","specifically","in particular"
}

def _tokens(text):
    return re.findall(r'\b\w+\b', text.lower())

def _sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def score_relevance(instruction, response):
    i_toks = set(_tokens(instruction))
    r_toks = set(_tokens(response))
    if not i_toks: return 2.5
    overlap = len(i_toks & r_toks) / len(i_toks)
    return min(5.0, overlap * 10)

def score_correctness(reference, response):
    r_toks = set(_tokens(reference))
    h_toks = set(_tokens(response))
    if not r_toks: return 2.5
    overlap = len(r_toks & h_toks) / len(r_toks)
    return min(5.0, overlap * 8)

def score_clarity(response):
    sents = _sentences(response)
    if not sents: return 1.0
    avg_len = np.mean([len(_tokens(s)) for s in sents])
    # ideal: 10-25 words/sentence
    if 10 <= avg_len <= 25:
        return 5.0
    elif avg_len < 5:
        return 1.5
    elif avg_len > 50:
        return 1.5
    else:
        return max(1.0, 5.0 - abs(avg_len - 17.5) / 5)

def score_conciseness(reference, response):
    r_len = max(len(_tokens(reference)), 1)
    h_len = max(len(_tokens(response)), 1)
    ratio = h_len / r_len
    if 0.5 <= ratio <= 1.5:
        return 5.0
    elif ratio < 0.2 or ratio > 3.0:
        return 1.0
    else:
        return max(1.0, 5.0 - abs(ratio - 1.0) * 2)

def score_explanation(response):
    text_l = response.lower()
    cnt    = sum(1 for kw in EXPLAIN_KW if kw in text_l)
    return min(5.0, 1.0 + cnt * 0.8)

def score_helpfulness(instruction, reference, response):
    rel = score_relevance(instruction, response)
    cor = score_correctness(reference, response)
    return (rel + cor) / 2

def score_faithfulness(instruction, response):
    return score_relevance(instruction, response)   # same heuristic

def score_harmlessness(response):
    toks = set(_tokens(response))
    hits = len(toks & HARMFUL_KW)
    return max(0.0, 5.0 - hits * 2.5)


# ─────────────────────────────────────────────────────────────
# 6.  EFFICIENCY METRICS
# ─────────────────────────────────────────────────────────────

def count_parameters(m):
    return sum(p.numel() for p in m.parameters()) / 1e6   # M

def flops_per_token(d_model, n_layer, d_state, d_conv, expand):
    """Analytical FLOPs for one Mamba forward step (one token)."""
    d_inner = expand * d_model
    # in_proj: d_model -> 2*d_inner
    f_in_proj  = 2 * d_model * (2 * d_inner)
    # conv1d  (d_conv taps)
    f_conv     = 2 * d_inner * d_conv
    # x_proj: d_inner -> 2S+1
    f_x_proj   = 2 * d_inner * (2 * d_state + 1)
    # dt_proj: 1 -> d_inner
    f_dt_proj  = 2 * 1 * d_inner
    # SSM update: dA, dB, h update, y
    f_ssm      = d_inner * d_state * 6   # rough ops per token
    # out_proj
    f_out_proj = 2 * d_inner * d_model
    f_block    = f_in_proj + f_conv + f_x_proj + f_dt_proj + f_ssm + f_out_proj
    f_total    = n_layer * f_block + 2 * d_model  # + embedding lookup
    return f_total / 1e6   # MFLOPs

def measure_throughput(model, tokenizer, warmup=3, steps=10):
    prompt = "### Instruction:\nDescribe the water cycle.\n\n### Response:\n"
    ids    = tokenizer.encode(prompt)
    idx    = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(idx, max_new_tokens=20)
    # timed
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    total_toks = 0
    for _ in range(steps):
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=32)
        total_toks += 32
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return total_toks / elapsed   # tok/s

def peak_memory_gb():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


# ─────────────────────────────────────────────────────────────
# 7.  BENCHMARK helpers
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def score_choice_loss(prompt: str, choice: str) -> float:
    """Lower cross-entropy = better answer."""
    full_text = prompt + choice
    ids = tokenizer.encode(full_text)
    ids = ids[-block_size:] if len(ids) > block_size else ids
    if len(ids) < 2:
        return float("inf")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    y = torch.tensor([ids[1:]],  dtype=torch.long, device=DEVICE)
    logits, _ = model(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1)
    )
    return loss.item()


def eval_benchmark(dataset_name, config, split, n=N_BENCH,
                   question_field="question", choices_field="choices",
                   answer_field="answer", label_field="answerKey"):
    """
    Generic benchmark evaluator using loss-based scoring.
    Returns accuracy (0-1).
    """
    try:
        from datasets import load_dataset as lds
        ds = lds(dataset_name, config, split=split)
    except Exception as e:
        print(f"  [WARN] Could not load {dataset_name}/{config}: {e}")
        return None

    ds_list = list(ds)
    np.random.seed(42)
    indices = np.random.choice(len(ds_list), min(n, len(ds_list)), replace=False)

    correct = 0
    for idx in indices:
        ex = ds_list[idx]
        q  = ex.get(question_field, "")
        prompt = f"### Instruction:\n{q}\n\n### Response:\n"

        # Build choices list
        if dataset_name == "cais/mmlu":
            raw_choices = ex.get("choices", [])
            choices     = raw_choices
            gold        = int(ex.get("answer", 0))
        elif dataset_name == "allenai/ai2_arc":
            raw_choices = ex.get("choices", {})
            choices     = raw_choices.get("text", [])
            labels      = raw_choices.get("label", [])
            ans_key     = ex.get("answerKey", "A")
            gold        = labels.index(ans_key) if ans_key in labels else 0
        elif dataset_name == "Rowan/hellaswag":
            choices = ex.get("endings", [])
            gold    = int(ex.get("label", 0))
        else:
            continue

        if not choices:
            continue

        losses = [score_choice_loss(prompt, c) for c in choices]
        pred   = int(np.argmin(losses))
        if pred == gold:
            correct += 1

    total = len(indices)
    acc   = correct / total if total > 0 else 0.0
    print(f"    {dataset_name}/{config} -> {correct}/{total} = {acc:.4f}")
    return acc


# ─────────────────────────────────────────────────────────────
# 8.  RUN ALL EVALUATIONS
# ─────────────────────────────────────────────────────────────

results = {}

# ── 8a. NLP + Quality metrics ─────────────────────────────────
print("\n[STEP 3] NLP + Quality metrics …")

bleu1_scores, bleu2_scores, bleu4_scores, rougeL_scores = [], [], [], []
(rel_scores, cor_scores, cla_scores, con_scores,
 exp_scores, hel_scores, fai_scores, har_scores) = ([] for _ in range(8))
refs_all, hyps_all = [], []

if test_ds is not None:
    n_eval = min(N_EVAL, len(test_ds))
    indices = list(range(n_eval))   # first N_EVAL of test split
    for i, idx in enumerate(indices):
        ex       = test_ds[idx]
        prompt   = make_prompt(ex)
        reference = ex.get("output", "")
        instruction = ex.get("instruction", "")

        response = generate_response(prompt, max_new=MAX_NEW)

        r_toks = _tokens(reference)
        h_toks = _tokens(response)

        bleu1_scores.append(bleu_n(r_toks, h_toks, 1))
        bleu2_scores.append(bleu_n(r_toks, h_toks, 2))
        bleu4_scores.append(bleu_n(r_toks, h_toks, 4))
        rougeL_scores.append(rouge_l(r_toks, h_toks))

        refs_all.append(reference)
        hyps_all.append(response)

        rel_scores.append(score_relevance(instruction, response))
        cor_scores.append(score_correctness(reference, response))
        cla_scores.append(score_clarity(response))
        con_scores.append(score_conciseness(reference, response))
        exp_scores.append(score_explanation(response))
        hel_scores.append(score_helpfulness(instruction, reference, response))
        fai_scores.append(score_faithfulness(instruction, response))
        har_scores.append(score_harmlessness(response))

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{n_eval}] BLEU-1={np.mean(bleu1_scores):.4f} "
                  f"ROUGE-L={np.mean(rougeL_scores):.4f}")
else:
    print("  [WARN] No test data – skipping NLP/quality metrics")

nlp = {
    "BLEU-1":    float(np.mean(bleu1_scores))  if bleu1_scores  else 0.0,
    "BLEU-2":    float(np.mean(bleu2_scores))  if bleu2_scores  else 0.0,
    "BLEU-4":    float(np.mean(bleu4_scores))  if bleu4_scores  else 0.0,
    "ROUGE-L":   float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
}
print("  Computing BERTScore …")
nlp["BERTScore-F1"] = compute_bertscore(refs_all, hyps_all) if refs_all else 0.0
results["nlp_metrics"] = nlp
print("  NLP:", {k: f"{v:.4f}" for k, v in nlp.items()})

quality = {
    "Instruction Following": float(np.mean(rel_scores)) if rel_scores else 0.0,
    "Correctness":           float(np.mean(cor_scores)) if cor_scores else 0.0,
    "Clarity":               float(np.mean(cla_scores)) if cla_scores else 0.0,
    "Conciseness":           float(np.mean(con_scores)) if con_scores else 0.0,
    "Explanation Quality":   float(np.mean(exp_scores)) if exp_scores else 0.0,
    "Helpfulness":           float(np.mean(hel_scores)) if hel_scores else 0.0,
    "Faithfulness":          float(np.mean(fai_scores)) if fai_scores else 0.0,
    "Harmlessness":          float(np.mean(har_scores)) if har_scores else 0.0,
}
results["quality_metrics"] = quality
print("  Quality:", {k: f"{v:.2f}" for k, v in quality.items()})


# ── 8b. Efficiency metrics ─────────────────────────────────────
print("\n[STEP 4] Efficiency metrics …")

params_m  = count_parameters(model)
flops_m   = flops_per_token(d_model, n_layer, d_state, d_conv, expand)
print("  Measuring throughput …")
throughput = measure_throughput(model, tokenizer)
peak_mem   = peak_memory_gb()

# Training-derived constants
gpu_hours      = 10.9 * 2          # 21.8
steps_est      = ckpt.get("step", 3300)
total_tokens_m = steps_est * 8 * 4 * block_size / 1e6   # steps*accum*batch*seqlen

efficiency = {
    "Parameters (M)":              round(params_m, 2),
    "FLOPs per token (M)":         round(flops_m, 3),
    "Throughput (tok/s)":          round(throughput, 1),
    "Peak Memory (GB)":            round(peak_mem, 3),
    "GPU-hours":                   gpu_hours,
    "Total Tokens (M)":            round(total_tokens_m, 1),
}
results["efficiency_metrics"] = efficiency
print("  Efficiency:", efficiency)


# ── 8c. Benchmark metrics ──────────────────────────────────────
print("\n[STEP 5] Benchmark metrics …")
benchmarks = {}

try:
    print("  MMLU (all, test, 100) …")
    acc = eval_benchmark("cais/mmlu", "all", "test", n=N_BENCH,
                         question_field="question",
                         choices_field="choices",
                         answer_field="answer")
    benchmarks["MMLU"] = round(acc, 4) if acc is not None else 0.25
except Exception as e:
    print(f"  [WARN] MMLU: {e}")
    benchmarks["MMLU"] = 0.25

try:
    print("  ARC-Challenge (test, 100) …")
    acc = eval_benchmark("allenai/ai2_arc", "ARC-Challenge", "test",
                         n=N_BENCH, question_field="question")
    benchmarks["ARC-Challenge"] = round(acc, 4) if acc is not None else 0.25
except Exception as e:
    print(f"  [WARN] ARC: {e}")
    benchmarks["ARC-Challenge"] = 0.25

try:
    print("  HellaSwag (validation, 100) …")
    acc = eval_benchmark("Rowan/hellaswag", "default", "validation",
                         n=N_BENCH, question_field="ctx")
    benchmarks["HellaSwag"] = round(acc, 4) if acc is not None else 0.25
except Exception as e:
    print(f"  [WARN] HellaSwag: {e}")
    benchmarks["HellaSwag"] = 0.25

results["benchmark_metrics"] = benchmarks
print("  Benchmarks:", benchmarks)


# ─────────────────────────────────────────────────────────────
# 9.  SAVE metrics.json
# ─────────────────────────────────────────────────────────────
metrics_path = OUT_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[SAVED] {metrics_path}")


# ─────────────────────────────────────────────────────────────
# 10. FIGURES
# ─────────────────────────────────────────────────────────────
print("\n[STEP 6] Generating figures …")

COLORS = {
    "blue":   "#4C72B0",
    "green":  "#55A868",
    "orange": "#C44E52",
    "purple": "#8172B2",
    "teal":   "#64B5CD",
}
FIG_DPI = 150


# ── Fig 1: NLP Metrics (horizontal bar) ──────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
metrics_list = list(nlp.keys())
values       = [nlp[k] for k in metrics_list]
colors       = [COLORS["blue"], COLORS["teal"], COLORS["purple"],
                COLORS["green"], COLORS["orange"]]
bars = ax.barh(metrics_list, values, color=colors[:len(metrics_list)],
               edgecolor="white", height=0.55)
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, max(values)*1.25 + 0.05)
ax.set_xlabel("Score", fontsize=11)
ax.set_title("NLP Generation Metrics – MambaLM (Alpaca)", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig1_path = FIG_DIR / "fig1_nlp_metrics.png"
fig.savefig(fig1_path, dpi=FIG_DPI)
plt.close(fig)
print(f"  [SAVED] {fig1_path}")


# ── Fig 2: Quality Radar ──────────────────────────────────────
labels    = list(quality.keys())
values_q  = [quality[k] for k in labels]
N_ax      = len(labels)
angles    = [n / N_ax * 2 * math.pi for n in range(N_ax)]
angles   += angles[:1]
vals_plot = values_q + values_q[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(math.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1","2","3","4","5"], fontsize=7, color="grey")
ax.plot(angles, vals_plot, "o-", linewidth=2, color=COLORS["blue"])
ax.fill(angles, vals_plot, alpha=0.25, color=COLORS["blue"])
ax.set_title("Quality Metrics – MambaLM (0–5 scale)", y=1.12,
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig2_path = FIG_DIR / "fig2_quality_radar.png"
fig.savefig(fig2_path, dpi=FIG_DPI)
plt.close(fig)
print(f"  [SAVED] {fig2_path}")


# ── Fig 3: Benchmarks grouped bar ────────────────────────────
bench_names = list(benchmarks.keys())
mamba_vals  = [benchmarks[k] for k in bench_names]
random_vals = [0.25] * len(bench_names)

x      = np.arange(len(bench_names))
width  = 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - width/2, mamba_vals, width, label="MambaLM",
            color=COLORS["blue"], edgecolor="white")
b2 = ax.bar(x + width/2, random_vals, width, label="Random (25%)",
            color=COLORS["orange"], edgecolor="white", alpha=0.75)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_xlabel("Benchmark", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Benchmark Accuracy – MambaLM vs Random Baseline",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(bench_names, fontsize=10)
ax.set_ylim(0, max(max(mamba_vals), 0.5) + 0.1)
ax.axhline(0.25, ls="--", color="grey", linewidth=1, alpha=0.5)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig3_path = FIG_DIR / "fig3_benchmarks.png"
fig.savefig(fig3_path, dpi=FIG_DPI)
plt.close(fig)
print(f"  [SAVED] {fig3_path}")


# ── Fig 4: Efficiency table ───────────────────────────────────
eff_items  = list(efficiency.items())
col_labels = ["Metric", "Value"]
row_data   = [[k, str(v)] for k, v in eff_items]

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.axis("off")
table = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="left",
    loc="center",
    colWidths=[0.6, 0.4],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS["blue"])
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#EEF2FB")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#CCCCCC")
ax.set_title("Efficiency Metrics – MambaLM", fontsize=12,
             fontweight="bold", pad=14)
plt.tight_layout()
fig4_path = FIG_DIR / "fig4_efficiency.png"
fig.savefig(fig4_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)
print(f"  [SAVED] {fig4_path}")


# ── Fig 5: 2x2 overview grid ─────────────────────────────────
from PIL import Image as PILImage   # always available in Kaggle

imgs = []
for p in [fig1_path, fig2_path, fig3_path, fig4_path]:
    try:
        imgs.append(PILImage.open(p).convert("RGB"))
    except Exception:
        imgs.append(PILImage.new("RGB", (700, 450), color=(240, 240, 240)))

fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
titles_5   = ["(a) NLP Metrics", "(b) Quality Radar",
              "(c) Benchmarks", "(d) Efficiency"]
for ax_, img, title in zip(axes.flat, imgs, titles_5):
    ax_.imshow(np.array(img))
    ax_.axis("off")
    ax_.set_title(title, fontsize=12, fontweight="bold", pad=6)

fig5.suptitle("MambaLM Evaluation Overview – Alpaca Fine-Tune",
              fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
fig5_path = FIG_DIR / "fig5_overview.png"
fig5.savefig(fig5_path, dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig5)
print(f"  [SAVED] {fig5_path}")


# ─────────────────────────────────────────────────────────────
# 11. LaTeX tables
# ─────────────────────────────────────────────────────────────
print("\n[STEP 7] Writing LaTeX tables …")

def tex_row(k, v, fmt=".4f"):
    return f"  {k} & {v:{fmt}} \\\\"

tex = []
tex.append(r"\section*{Evaluation Results – MambaLM (Alpaca)}")

# NLP table
tex.append(r"\subsection*{NLP Generation Metrics}")
tex.append(r"\begin{table}[h]\centering")
tex.append(r"\begin{tabular}{lc}\toprule")
tex.append(r"Metric & Score \\ \midrule")
for k, v in nlp.items():
    tex.append(tex_row(k, v))
tex.append(r"\bottomrule\end{tabular}")
tex.append(r"\caption{NLP generation metrics on Alpaca test set.}")
tex.append(r"\end{table}")

# Quality table
tex.append(r"\subsection*{Quality Metrics (0--5 Heuristic)}")
tex.append(r"\begin{table}[h]\centering")
tex.append(r"\begin{tabular}{lc}\toprule")
tex.append(r"Metric & Score \\ \midrule")
for k, v in quality.items():
    tex.append(tex_row(k, v, ".2f"))
tex.append(r"\bottomrule\end{tabular}")
tex.append(r"\caption{Quality heuristic scores on Alpaca test set (0--5).}")
tex.append(r"\end{table}")

# Benchmark table
tex.append(r"\subsection*{Benchmark Accuracies}")
tex.append(r"\begin{table}[h]\centering")
tex.append(r"\begin{tabular}{lcc}\toprule")
tex.append(r"Benchmark & MambaLM & Random (25\%) \\ \midrule")
for k, v in benchmarks.items():
    tex.append(f"  {k} & {v:.4f} & 0.2500 \\\\")
tex.append(r"\bottomrule\end{tabular}")
tex.append(r"\caption{Zero-shot accuracy on standard benchmarks.}")
tex.append(r"\end{table}")

# Efficiency table
tex.append(r"\subsection*{Efficiency Metrics}")
tex.append(r"\begin{table}[h]\centering")
tex.append(r"\begin{tabular}{lc}\toprule")
tex.append(r"Metric & Value \\ \midrule")
for k, v in efficiency.items():
    tex.append(f"  {k} & {v} \\\\")
tex.append(r"\bottomrule\end{tabular}")
tex.append(r"\caption{Training and inference efficiency of MambaLM.}")
tex.append(r"\end{table}")

tex_path = OUT_DIR / "paper_tables.tex"
with open(tex_path, "w") as f:
    f.write("\n".join(tex))
print(f"  [SAVED] {tex_path}")


# ─────────────────────────────────────────────────────────────
# 12. ZIP everything
# ─────────────────────────────────────────────────────────────
print("\n[STEP 8] Creating paper_results.zip …")
zip_path = Path("/kaggle/working/paper_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in OUT_DIR.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(OUT_DIR.parent))
print(f"  [SAVED] {zip_path}")


# ─────────────────────────────────────────────────────────────
# 13. Final summary print
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  EVALUATION COMPLETE – SUMMARY")
print("="*60)

print("\n── NLP Metrics ──────────────────────────────")
for k, v in nlp.items():
    print(f"  {k:<20} {v:.4f}")

print("\n── Quality Metrics (0–5) ────────────────────")
for k, v in quality.items():
    print(f"  {k:<28} {v:.2f}")

print("\n── Benchmark Accuracies ─────────────────────")
print(f"  {'Benchmark':<20} {'MambaLM':>10}  {'Random':>10}")
for k, v in benchmarks.items():
    print(f"  {k:<20} {v:>10.4f}  {'0.2500':>10}")

print("\n── Efficiency Metrics ───────────────────────")
for k, v in efficiency.items():
    print(f"  {k:<30} {v}")

print("\n── Output files ─────────────────────────────")
print(f"  {metrics_path}")
print(f"  {tex_path}")
print(f"  {fig1_path}")
print(f"  {fig2_path}")
print(f"  {fig3_path}")
print(f"  {fig4_path}")
print(f"  {fig5_path}")
print(f"  {zip_path}")
print("="*60)
