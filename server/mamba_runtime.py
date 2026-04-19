"""
mamba_runtime.py — Inference runtime for MambaLM.

Supports two tokenizer modes (auto-detected from meta.json):
  • "bpe"  — Custom BPE tokenizer (new, redesigned training)
  • "char" — Character-level tokenizer (legacy)
"""

import os
import re
import json
import torch
import torch.nn.functional as F

from train.mamba_model import MambaLM


DEFAULT_STOP_TAG = "### Instruction:"


# ─────────────────────────────────────────────────────────────────────────────
#  Inline BPE Tokenizer (mirrors Kaggle training code exactly)
# ─────────────────────────────────────────────────────────────────────────────

class _BPETokenizer:
    PAD_ID, UNK_ID = 0, 1

    def __init__(self, bpe_data: dict):
        merges             = bpe_data["merges"]
        self.merges        = [(a, b) for a, b in merges]
        self.vocab         = bpe_data["vocab"]
        self.inv_vocab     = {v: k for k, v in self.vocab.items()}
        self.merge_priority = {(a, b): r for r, (a, b) in enumerate(self.merges)}
        self._cache        = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def _split(text):
        return re.findall(r'\n|\S+', text)

    def _encode_word(self, word):
        prefixed = '\n' if word == '\n' else 'Ġ' + word
        syms = list(prefixed)
        if len(syms) == 1:
            return syms
        while len(syms) > 1:
            best_rank, best_i = float('inf'), -1
            for i in range(len(syms) - 1):
                rank = self.merge_priority.get((syms[i], syms[i+1]), float('inf'))
                if rank < best_rank:
                    best_rank, best_i = rank, i
            if best_i == -1:
                break
            a, b = syms[best_i], syms[best_i + 1]
            merged = a + b
            new_syms, k = [], 0
            while k < len(syms):
                if k < len(syms)-1 and syms[k] == a and syms[k+1] == b:
                    new_syms.append(merged); k += 2
                else:
                    new_syms.append(syms[k]); k += 1
            syms = new_syms
        return syms

    def encode(self, text: str):
        ids = []
        for word in self._split(text):
            if word not in self._cache:
                toks = self._encode_word(word)
                self._cache[word] = [self.vocab.get(t, self.UNK_ID) for t in toks]
            ids.extend(self._cache[word])
        return ids

    def decode(self, ids):
        tokens = [self.inv_vocab.get(i, '') for i in ids]
        text   = ''.join(tokens)
        text   = text.replace('\nĠ', '\n')
        text   = text.replace('Ġ', ' ')
        return text.lstrip()


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy char tokenizer wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _CharTokenizer:
    def __init__(self, vocab_map: dict):
        self.stoi      = {str(k): int(v) for k, v in vocab_map.items()}
        self.itos      = {int(v): str(k) for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos.get(int(i), '') for i in ids)


# ─────────────────────────────────────────────────────────────────────────────
#  MambaRuntime
# ─────────────────────────────────────────────────────────────────────────────

class MambaRuntime:
    def __init__(self, ckpt_path: str, meta_path: str, device: str | None = None):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt.pt not found: {ckpt_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load meta ─────────────────────────────────────────────────────────
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = meta.get("config", {})
        self.stop_tag = (cfg.get("stop_tag")
                         or meta.get("stop_tag")
                         or DEFAULT_STOP_TAG)
        print(f"  stop_tag = {repr(self.stop_tag)}")

        # ── Tokenizer (auto-detect) ────────────────────────────────────────────
        tok_type = meta.get("tokenizer_type", "char")
        if tok_type == "bpe" and "bpe_data" in meta:
            self.tokenizer = _BPETokenizer(meta["bpe_data"])
            print(f"  tokenizer = BPE  (vocab={self.tokenizer.vocab_size})")
        else:
            # Legacy char mode — try multiple meta.json layouts
            vocab_map = (meta.get("vocab")
                         or meta.get("stoi")
                         or meta.get("vocab_stoi"))
            if vocab_map is None:
                raise KeyError("meta.json: no vocab/stoi/bpe_data found")
            self.tokenizer = _CharTokenizer(vocab_map)
            print(f"  tokenizer = char  (vocab={self.tokenizer.vocab_size})")

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = MambaLM(
            vocab_size  = self.tokenizer.vocab_size,
            d_model     = int(cfg.get("d_model",    256)),
            n_layer     = int(cfg.get("n_layer",      6)),
            d_state     = int(cfg.get("d_state",     16)),
            d_conv      = int(cfg.get("d_conv",       4)),
            expand      = int(cfg.get("expand",       2)),
            block_size  = int(cfg.get("block_size", 256)),
            dropout     = float(cfg.get("dropout",  0.0)),
        ).to(self.device)

        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # Key remap: Kaggle training uses "embed.", local model uses "embedding."
        remapped = {}
        for k, v in state.items():
            nk = k.replace("embed.", "embedding.", 1) if k.startswith("embed.") else k
            remapped[nk] = v

        self.model.load_state_dict(remapped)
        self.model.eval()
        print(f"✅ MambaRuntime loaded: {ckpt_path}")

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 320,
        temperature: float  = 0.8,
        top_k: int          = 50,
        stop_tag: str       = None,
    ) -> str:
        stop_tag = stop_tag or self.stop_tag
        prompt   = str(prompt)

        ids = self.tokenizer.encode(prompt)
        ids = ids[-self.model.block_size:]
        x   = torch.tensor([ids], dtype=torch.long, device=self.device)

        new_ids = []

        for _ in range(int(max_new_tokens)):
            ctx    = x[:, -self.model.block_size:]
            logits, _ = self.model(ctx)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / float(temperature)
                if top_k and top_k > 0:
                    v, _  = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs   = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_id], dim=1)
            new_ids.append(int(next_id.item()))

            # Stop check on decoded output
            decoded = self.tokenizer.decode(new_ids)
            if stop_tag and stop_tag in decoded:
                decoded = decoded.split(stop_tag, 1)[0]
                return self._clean(decoded)

        return self._clean(self.tokenizer.decode(new_ids))

    def _clean(self, text: str) -> str:
        text = text.strip("\n\r\t ")
        return self._trim_incomplete(text)

    def _trim_incomplete(self, text: str) -> str:
        if not text:
            return text
        ENDS = {'.', '!', '?', '।', '…'}
        stripped = text.rstrip()
        if stripped and stripped[-1] in ENDS:
            return text
        last = max((i for i, c in enumerate(stripped) if c in ENDS), default=-1)
        if last == -1:
            return text
        if len(stripped[last + 1:].strip()) > 12:
            return stripped[:last + 1].strip()
        return text
