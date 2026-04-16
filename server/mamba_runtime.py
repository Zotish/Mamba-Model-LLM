"""
mamba_runtime.py — Inference runtime for MambaLM.

Loads out_mamba/ckpt.pt + out_mamba/meta.json and exposes the same
generate() interface as model_runtime.py so the FastAPI layer stays
identical.
"""

import os
import json
import torch
import torch.nn.functional as F

from train.tokenizer    import CharTokenizer
from train.mamba_model  import MambaLM


DEFAULT_STOP_TAG = "\n### Instruction:"


class MambaRuntime:
    def __init__(self, ckpt_path: str, meta_path: str, device: str | None = None):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt.pt not found: {ckpt_path}")

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Tokenizer
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        vocab_map = meta["vocab"]
        cfg       = meta["config"]

        # Read stop_tag from meta.json config (set during training)
        self.stop_tag = cfg.get("stop_tag", DEFAULT_STOP_TAG)
        print(f"  stop_tag = {repr(self.stop_tag)}")

        self.tokenizer = CharTokenizer.from_vocab_map(vocab_map)

        # Model
        self.model = MambaLM(
            vocab_size  = self.tokenizer.vocab_size,
            d_model     = int(cfg["d_model"]),
            n_layer     = int(cfg["n_layer"]),
            d_state     = int(cfg.get("d_state", 16)),
            d_conv      = int(cfg.get("d_conv",  4)),
            expand      = int(cfg.get("expand",  2)),
            block_size  = int(cfg["block_size"]),
            dropout     = float(cfg.get("dropout", 0.0)),
        ).to(self.device)

        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # Remap keys: Kaggle training used "embed." but local model uses "embedding."
        remapped = {}
        for k, v in state.items():
            new_k = k.replace("embed.", "embedding.", 1) if k.startswith("embed.") else k
            remapped[new_k] = v

        self.model.load_state_dict(remapped)
        self.model.eval()

        print(f"✅ MambaRuntime loaded: {ckpt_path}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 320,
        temperature: float  = 0.7,
        top_k: int          = 50,
        stop_tag: str       = None,
    ) -> str:
        if stop_tag is None:
            stop_tag = self.stop_tag
        prompt   = str(prompt)
        stop_len = len(stop_tag)

        ids     = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
        )
        decoded = prompt

        for _ in range(int(max_new_tokens)):
            # crop to block_size
            if ids.size(1) > self.model.block_size:
                ids = ids[:, -self.model.block_size:]

            logits, _ = self.model(ids)
            logits     = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits  = logits / float(temperature)
                if top_k and top_k > 0:
                    v, _    = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs   = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids     = torch.cat([ids, next_id], dim=1)
            ch      = self.tokenizer.decode([int(next_id.item())])
            decoded += ch

            if stop_len > 0 and decoded.endswith(stop_tag):
                break

        # Extract reply (new tokens only)
        reply = decoded[len(prompt):]

        # Cut at stop tag
        if stop_tag and stop_tag in reply:
            reply = reply.split(stop_tag, 1)[0]

        reply = reply.strip("\n\r\t ")
        reply = self._trim_incomplete_sentence(reply)
        return reply

    def _trim_incomplete_sentence(self, text: str) -> str:
        if not text:
            return text
        ENDS = {'.', '!', '?', '।', '…'}
        stripped = text.rstrip()
        if stripped and stripped[-1] in ENDS:
            return text
        last_end = -1
        for i, ch in enumerate(stripped):
            if ch in ENDS:
                last_end = i
        if last_end == -1:
            return text
        fragment = stripped[last_end + 1:].strip()
        if len(fragment) > 12:
            return stripped[:last_end + 1].strip()
        return text
