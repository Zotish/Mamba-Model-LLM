

# import os, json, math
# import torch
# import torch.nn.functional as F

# # imports from your train package
# from train.model import MiniGPT
# from train.tokenizer import CharTokenizer

# END_TAG = "\n</ASS>\n"

# def detect_lang(text: str) -> str:
#     # very lightweight language detection for fallback replies
#     for ch in text:
#         o = ord(ch)
#         if 0x0980 <= o <= 0x09FF:
#             return "bn"
#         if 0x0900 <= o <= 0x097F:
#             return "hi"
#         if 0x4E00 <= o <= 0x9FFF:
#             return "zh"
#     # naive Spanish hint
#     if any(x in text.lower() for x in ["qué", "cómo", "acción", "apalancamiento", "hola"]):
#         return "es"
#     return "en"

# def fallback_unknown(user_text: str) -> str:
#     lang = detect_lang(user_text)
#     if lang == "bn":
#         return "আমি নিশ্চিত না / জানি না। আমি এই বিষয়ে নির্ভুল উত্তর দিতে পারি না।"
#     if lang == "hi":
#         return "मुझे नहीं पता। मैं इस बारे में भरोसेमंद जवाब नहीं दे सकता।"
#     if lang == "es":
#         return "No lo sé. No puedo darte una respuesta fiable sobre eso."
#     if lang == "zh":
#         return "我不知道。我无法给出可靠的答案。"
#     return "I don’t know. I can’t give a reliable answer to that."

# class Runtime:
#     def __init__(self, ckpt_path: str, meta_path: str):
#         self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

#         with open(meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)

#         stoi = meta["vocab"]
#         itos = {int(v): k for k, v in stoi.items()}
#         self.tok = CharTokenizer(stoi=stoi, itos=itos)

#         cfg = meta["config"]
#         self.model = MiniGPT(
#             vocab_size=len(self.tok.stoi),
#             block_size=cfg["block_size"],
#             n_layer=cfg["n_layer"],
#             n_head=cfg["n_head"],
#             n_embd=cfg["n_embd"],
#             dropout=cfg.get("dropout", 0.0),
#         ).to(self.device)

#         ckpt = torch.load(ckpt_path, map_location=self.device)
#         self.model.load_state_dict(ckpt["model"], strict=True)
#         self.model.eval()

#         # pre-encode end tag ids
#         self.end_ids = self.tok.encode(END_TAG)

#     @torch.no_grad()
#     def _sample_next(
#         self,
#         logits: torch.Tensor,
#         prev_ids: list[int],
#         temperature: float,
#         top_k: int,
#         top_p: float,
#         repetition_penalty: float,
#         no_repeat_ngram: int,
#     ) -> tuple[int, float]:
#         # logits: (vocab,)
#         logits = logits.float()

#         # repetition penalty
#         if repetition_penalty and repetition_penalty > 1.0 and prev_ids:
#             for tid in set(prev_ids[-256:]):
#                 logits[tid] /= repetition_penalty

#         # no-repeat ngram (simple trigram)
#         if no_repeat_ngram and no_repeat_ngram >= 2 and len(prev_ids) >= no_repeat_ngram:
#             n = no_repeat_ngram
#             # block tokens that would repeat last n-1 + token sequence found before
#             prefix = prev_ids[-(n - 1):]
#             banned = set()
#             for i in range(len(prev_ids) - n):
#                 if prev_ids[i:i + n - 1] == prefix:
#                     banned.add(prev_ids[i + n - 1])
#             if banned:
#                 logits[list(banned)] = -1e10

#         # temperature
#         temperature = max(0.05, float(temperature))
#         logits = logits / temperature

#         # top-k
#         if top_k and top_k > 0:
#             k = min(int(top_k), logits.numel())
#             v, _ = torch.topk(logits, k)
#             cutoff = v[-1]
#             logits = torch.where(logits < cutoff, torch.tensor(-1e10, device=logits.device), logits)

#         # top-p nucleus
#         if top_p and 0 < top_p < 1.0:
#             sorted_logits, sorted_idx = torch.sort(logits, descending=True)
#             probs = F.softmax(sorted_logits, dim=-1)
#             cum = torch.cumsum(probs, dim=-1)
#             mask = cum > top_p
#             # keep at least 1
#             mask[0] = False
#             sorted_logits = torch.where(mask, torch.tensor(-1e10, device=logits.device), sorted_logits)
#             # unsort back
#             new_logits = torch.empty_like(logits)
#             new_logits[sorted_idx] = sorted_logits
#             logits = new_logits

#         probs = F.softmax(logits, dim=-1)
#         next_id = torch.multinomial(probs, num_samples=1).item()

#         # logprob for confidence
#         lp = float(torch.log(probs[next_id] + 1e-12).item())
#         return next_id, lp

#     @torch.no_grad()
#     def generate_chat(
#         self,
#         user_text: str,
#         history_pairs: list[tuple[str, str]],
#         max_new_tokens: int = 220,
#         temperature: float = 0.8,
#         top_k: int = 60,
#         top_p: float = 0.9,
#         repetition_penalty: float = 1.12,
#         no_repeat_ngram: int = 3,
#     ) -> str:
#         # Build prompt with tags (prevents system leak + gives consistent format)
#         sys = (
#             "<SYS>\n"
#             "You are a helpful, calm, educational assistant.\n"
#             "You explain coin, token, stock, trading safely.\n"
#             "NOT financial advice. No guaranteed profit.\n"
#             "Reply in user's language.\n"
#             "If you do not know, say you don't know in user's language.\n"
#             "Do NOT repeat system text or tags.\n"
#             "</SYS>\n"
#         )

#         prompt = sys
#         # keep last few turns
#         for u, a in history_pairs[-6:]:
#             prompt += f"<USR>\n{u}\n</USR>\n<ASS>\n{a}\n</ASS>\n"
#         prompt += f"<USR>\n{user_text}\n</USR>\n<ASS>\n"

#         ids = self.tok.encode(prompt)
#         if not ids:
#             return ""

#         x = torch.tensor([ids], dtype=torch.long, device=self.device)

#         out_ids = ids[:]  # python list for penalties
#         total_lp = 0.0
#         gen_count = 0

#         for _ in range(int(max_new_tokens)):
#             idx_cond = x[:, -self.model.block_size:]
#             logits, _ = self.model(idx_cond)          # (B,T,V)
#             next_logits = logits[0, -1, :]            # (V,)

#             next_id, lp = self._sample_next(
#                 next_logits,
#                 out_ids,
#                 temperature=temperature,
#                 top_k=top_k,
#                 top_p=top_p,
#                 repetition_penalty=repetition_penalty,
#                 no_repeat_ngram=no_repeat_ngram,
#             )

#             out_ids.append(next_id)
#             total_lp += lp
#             gen_count += 1

#             x = torch.cat([x, torch.tensor([[next_id]], device=self.device)], dim=1)

#             # stop if END_TAG appears
#             if len(out_ids) >= len(self.end_ids):
#                 if out_ids[-len(self.end_ids):] == self.end_ids:
#                     break

#         full = self.tok.decode(out_ids)

#         # Extract only assistant answer between <ASS> and </ASS>
#         # We prompted with "<ASS>\n" at the end.
#         marker = "<ASS>\n"
#         start = full.rfind(marker)
#         if start == -1:
#             # if parsing fails, fallback unknown
#             return fallback_unknown(user_text)

#         answer = full[start + len(marker):]
#         end_pos = answer.find("</ASS>")
#         if end_pos != -1:
#             answer = answer[:end_pos]

#         answer = answer.strip()

#         # Confidence / unknown handling:
#         # If avg logprob too low OR answer empty OR contains tags/system → fallback
#         avg_lp = total_lp / max(gen_count, 1)
#         leaked = any(t in answer for t in ["<SYS>", "</SYS>", "<USR>", "</USR>", "<ASS>", "</ASS>", "System:"])
#         too_short = len(answer) < 3

#         if leaked or too_short or avg_lp < -5.5:
#             return fallback_unknown(user_text)

#         return answer





# import json
# import torch
# from typing import Tuple


# class Runtime:
#     """
#     Loads:
#       - meta.json (vocab + config)
#       - ckpt.pt   (model weights)
#     Generates:
#       - ONLY the completion (new tokens), not the whole prompt.
#       - Applies simple stop rules to avoid "User:" / "System:" echo.
#     """

#     def __init__(self, ckpt_path: str, meta_path: str):
#         self.device = self._get_device()

#         tok, cfg = self._load_meta(meta_path)
#         self.tok = tok
#         self.cfg = cfg

#         # Local imports to avoid path issues
#         from train.model import MiniGPT  # <-- your train/model.py

#         self.model = MiniGPT(
#             vocab_size=self.tok.vocab_size,
#             block_size=int(cfg["block_size"]),
#             n_layer=int(cfg["n_layer"]),
#             n_head=int(cfg["n_head"]),
#             n_embd=int(cfg["n_embd"]),
#             dropout=float(cfg.get("dropout", 0.0)),
#         ).to(self.device)

#         ckpt = torch.load(ckpt_path, map_location=self.device)
#         self.model.load_state_dict(ckpt["model"], strict=True)
#         self.model.eval()

#     def _get_device(self):
#         # inference: MPS ok (training CPU cool mode আলাদা)
#         if torch.backends.mps.is_available():
#             return torch.device("mps")
#         if torch.cuda.is_available():
#             return torch.device("cuda")
#         return torch.device("cpu")

#     def _load_meta(self, meta_path: str):
#         from train.tokenizer import CharTokenizer

#         with open(meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)

#         stoi = {k: int(v) for k, v in meta["vocab"].items()}
#         itos = {int(v): k for k, v in meta["vocab"].items()}
#         tok = CharTokenizer(stoi=stoi, itos=itos)
#         return tok, meta["config"]

#     def _cut_on_stop(self, text: str) -> str:
#         # Common “prompt markers” – এগুলো দেখলেই থামিয়ে দিব
#         stops = [
#             "\nUser:", "\nuser:",
#             "\nSystem:", "\nsystem:",
#             "\n###", "\nInstruction:", "\nResponse:",
#         ]
#         cut = len(text)
#         for s in stops:
#             i = text.find(s)
#             if i != -1:
#                 cut = min(cut, i)
#         return text[:cut].strip()

#     @torch.no_grad()
#     def generate_completion(
#         self,
#         prompt: str,
#         max_new_tokens: int = 220,
#         temperature: float = 0.9,
#         top_k: int = 60,
#     ) -> str:
#         prompt_ids = self.tok.encode(prompt)
#         if len(prompt_ids) == 0:
#             return ""

#         x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
#         y = self.model.generate(
#             x,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k,
#         )

#         # ✅ only new tokens:
#         new_ids = y[0].tolist()[len(prompt_ids):]
#         out = self.tok.decode(new_ids)
#         return self._cut_on_stop(out)


# from __future__ import annotations
# import os
# import json
# import re
# from typing import Dict, Any, Optional

# import torch

# # IMPORTANT: ensure imports point to train/*
# from train.tokenizer import CharTokenizer
# from train.model import MiniGPT


# ASS_TAG = "<|assistant|>\n"
# USER_TAG = "<|user|>\n"
# SYS_TAG = "<|system|>\n"
# END_TAG = "\n<|end|>\n"


# def detect_lang(text: str) -> str:
#     # very small heuristic (no extra deps)
#     for ch in text:
#         o = ord(ch)
#         if 0x0980 <= o <= 0x09FF:
#             return "bn"
#         if 0x0900 <= o <= 0x097F:
#             return "hi"
#         if (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF):
#             return "zh"
#     if "¿" in text or "¡" in text or any(w in text.lower() for w in ["qué", "acción", "bolsa", "cript", "token"]):
#         return "es"
#     return "en"


# def unknown_reply(lang: str) -> str:
#     if lang == "bn":
#         return "দুঃখিত—এটার উত্তর আমি জানি না। আমি coin/token/stock এবং basic trading বিষয়ে সাহায্য করতে পারি।"
#     if lang == "hi":
#         return "माफ़ कीजिए—मुझे इसका जवाब नहीं पता। मैं coin/token/stock और basic trading में मदद कर सकता हूँ।"
#     if lang == "zh":
#         return "抱歉——我不知道这个问题的答案。我可以帮助解释 coin/token/stock 和基础交易概念。"
#     if lang == "es":
#         return "Lo siento—no sé la respuesta. Puedo ayudar con coin/token/stock y conceptos básicos de trading."
#     return "Sorry—I don’t know the answer. I can help with coin/token/stock and basic trading concepts."


# def looks_in_domain(text: str) -> bool:
#     t = text.lower()

#     # greetings / identity should be allowed
#     if any(k in t for k in ["who are you", "who am i", "tumi ke", "ami ke", "আপনি কে", "তুমি কে", "আমি কে"]):
#         return True
#     if any(k in t for k in ["hello", "hi", "hey", "হাই", "হ্যালো", "namaste", "hola", "你好"]):
#         return True

#     # finance domain keywords
#     keys = [
#         "coin", "token", "stock", "trading", "trade", "exchange", "wallet", "kyc",
#         "leverage", "spot", "futures", "broker", "platform", "crypto", "bitcoin", "ethereum",
#         "buy", "sell", "fees", "risk", "scam", "regulation",
#         "কয়েন", "টোকেন", "স্টক", "ট্রেডিং", "এক্সচেঞ্জ", "ওয়ালেট", "লিভারেজ", "ফিউচার", "স্পট",
#         "कॉइन", "टोकन", "स्टॉक", "ट्रेडिंग", "एक्सचेंज", "वॉलेट", "लीवरेज",
#         "交易", "股票", "代币", "币", "交易所", "钱包", "杠杆",
#         "acción", "bolsa", "intercambio", "billetera", "apalancamiento",
#     ]
#     return any(k.lower() in t for k in keys)


# class Runtime:
#     def __init__(self, ckpt_path: str, meta_path: str, device: Optional[str] = None):
#         self.ckpt_path = ckpt_path
#         self.meta_path = meta_path

#         self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))

#         tok, cfg = CharTokenizer.load_meta(meta_path)
#         self.tok = tok
#         self.cfg = cfg

#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         # config fallback
#         c = ckpt.get("config", {}) or cfg
#         self.cfg = c

#         self.model = MiniGPT(
#             vocab_size=int(c["vocab_size"]),
#             block_size=int(c["block_size"]),
#             n_layer=int(c["n_layer"]),
#             n_head=int(c["n_head"]),
#             n_embd=int(c["n_embd"]),
#             dropout=float(c.get("dropout", 0.1)),
#         ).to(self.device)

#         self.model.load_state_dict(ckpt["model"])
#         self.model.eval()

#     def build_prompt(self, system: str, messages: list[dict[str, str]]) -> str:
#         # tags-based prompt; model learns to answer after <|assistant|>
#         out = SYS_TAG + system.strip() + END_TAG
#         for m in messages:
#             role = m.get("role")
#             content = (m.get("content") or "").strip()
#             if not content:
#                 continue
#             if role == "user":
#                 out += USER_TAG + content + END_TAG
#             elif role == "assistant":
#                 out += ASS_TAG + content + END_TAG
#         out += ASS_TAG  # now generate assistant
#         return out

#     @torch.no_grad()
#     def generate_completion(self, prompt: str, max_new_tokens: int = 240, temperature: float = 0.9, top_k: int = 60) -> str:
#         ids = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=self.device)
#         y = self.model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
#         text = self.tok.decode(y[0].tolist())

#         # Keep only text after the last "<|assistant|>\n"
#         if ASS_TAG in text:
#             text = text.split(ASS_TAG)[-1]

#         # Stop at <|end|>
#         if "<|end|>" in text:
#             text = text.split("<|end|>")[0]

#         # If model leaks tags/system, cut them
#         text = re.split(r"<\|system\|>|<\|user\|>|<\|assistant\|>", text)[0]

#         # Clean minor junk
#         text = text.strip()
#         return text


# import os
# import json
# import sys
# import torch


# # Ensure project root is on sys.path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from train.model import MiniGPT
# from train.tokenizer import CharTokenizer


# class Runtime:
#     def __init__(self, ckpt_path: str, meta_path: str):
#         self.device = self._get_device()
#         self.tok, cfg = self._load_meta(meta_path)

#         self.model = MiniGPT(
#             vocab_size=self.tok.vocab_size,
#             block_size=cfg["block_size"],
#             n_layer=cfg["n_layer"],
#             n_head=cfg["n_head"],
#             n_embd=cfg["n_embd"],
#             dropout=cfg["dropout"],
#         ).to(self.device)

#         ckpt = torch.load(ckpt_path, map_location=self.device)
#         self.model.load_state_dict(ckpt["model"])
#         self.model.eval()

#     def _get_device(self):
#         # For M1 Air cool mode, you can force CPU by env if you want
#         force_cpu = os.getenv("FORCE_CPU", "0") == "1"
#         if force_cpu:
#             return torch.device("cpu")
#         if torch.backends.mps.is_available():
#             return torch.device("mps")
#         if torch.cuda.is_available():
#             return torch.device("cuda")
#         return torch.device("cpu")

#     def _load_meta(self, meta_path: str):
#         with open(meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)
#         stoi = {k: int(v) for k, v in meta["vocab"].items()}
#         itos = {int(v): k for k, v in meta["vocab"].items()}
#         tok = CharTokenizer(stoi=stoi, itos=itos)
#         return tok, meta["config"]

#     def _apply_stop(self, text: str, stop: list[str]) -> str:
#         if not stop:
#             return text
#         cut = None
#         for s in stop:
#             if not s:
#                 continue
#             i = text.find(s)
#             if i != -1:
#                 cut = i if cut is None else min(cut, i)
#         return text if cut is None else text[:cut]

#     @torch.no_grad()
#     def generate(
#         self,
#         prompt: str,
#         max_new_tokens: int = 200,
#         temperature: float = 0.9,
#         top_k: int = 50,
#         stop: list[str] | None = None,
#     ) -> str:
#         # encode prompt
#         ids = self.tok.encode(prompt)
#         if len(ids) == 0:
#             return ""

#         x = torch.tensor([ids], dtype=torch.long, device=self.device)

#         # generate full sequence (prompt + new tokens)
#         y = self.model.generate(
#             x,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k,
#         )

#         # ✅ return ONLY newly generated tokens (not the prompt)
#         new_ids = y[0].tolist()[len(ids):]
#         out = self.tok.decode(new_ids)

#         # optional stop strings to cut junk
#         if stop:
#             out = self._apply_stop(out, stop)

#         return out.strip()


# import os
# import json
# import torch
# import torch.nn.functional as F

# # Must match training imports
# from train.model import MiniGPT
# from train.tokenizer import CharTokenizer

# END_TAG = "\n</ASS>\n"

# def detect_lang(text: str) -> str:
#     # very lightweight language detection for fallback replies
#     for ch in text:
#         o = ord(ch)
#         if 0x0980 <= o <= 0x09FF:
#             return "bn"
#         if 0x0900 <= o <= 0x097F:
#             return "hi"
#         if 0x4E00 <= o <= 0x9FFF:
#             return "zh"
#     # naive Spanish hint
#     if any(x in text.lower() for x in ["qué", "cómo", "acción", "apalancamiento", "hola"]):
#         return "es"
#     return "en"

# def fallback_unknown(user_text: str) -> str:
#     lang = detect_lang(user_text)
#     if lang == "bn":
#         return "আমি নিশ্চিত না / জানি না। আমি এই বিষয়ে নির্ভুল উত্তর দিতে পারি না।"
#     if lang == "hi":
#         return "मुझे नहीं पता। मैं इस बारे में भरोसेमंद जवाब नहीं दे सकता।"
#     if lang == "es":
#         return "No lo sé. No puedo darte una respuesta fiable sobre eso."
#     if lang == "zh":
#         return "我不知道。我无法给出可靠的答案。"
#     return "I don’t know. I can’t give a reliable answer to that."


# def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
#     if k <= 0:
#         return logits
#     v, ix = torch.topk(logits, k)
#     out = logits.clone()
#     out[out < v[..., [-1]]] = -float("inf")
#     return out


# class Runtime:
#     def __init__(self, ckpt_path: str, meta_path: str, device: str | None = None):
#         self.ckpt_path = ckpt_path
#         self.meta_path = meta_path

#         if device is None:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device

#         if not os.path.exists(self.meta_path):
#             raise FileNotFoundError(f"meta.json not found: {self.meta_path}")
#         if not os.path.exists(self.ckpt_path):
#             raise FileNotFoundError(f"ckpt.pt not found: {self.ckpt_path}")

#         with open(self.meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)

#         vocab_map = meta["vocab"]          # stoi
#         config = meta["config"]
#         self.tokenizer = CharTokenizer.from_vocab_map(vocab_map)

#         self.model = MiniGPT(
#             vocab_size=self.tokenizer.vocab_size,
#             block_size=int(config["block_size"]),
#             n_layer=int(config["n_layer"]),
#             n_head=int(config["n_head"]),
#             n_embd=int(config["n_embd"]),
#             dropout=float(config.get("dropout", 0.0)),
#         ).to(self.device)

#         ckpt = torch.load(self.ckpt_path, map_location="cpu")
#         state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
#         self.model.load_state_dict(state, strict=True)
#         self.model.eval()

#     @torch.no_grad()
#     def generate(
#         self,
#         prompt: str,
#         max_new_tokens: int = 220,
#         temperature: float = 0.8,
#         top_k: int = 60,
#         stop_tag: str = END_TAG,
#     ) -> str:
#         """
#         Returns ONLY assistant reply (no prompt echo), stops at stop_tag if present.
#         """
#         prompt = str(prompt)

#         # Encode prompt
#         ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

#         # We'll build decoded output incrementally for stop detection
#         decoded = prompt
#         stop_len = len(stop_tag)

#         for _ in range(int(max_new_tokens)):
#             # crop to block size
#             if ids.size(1) > self.model.block_size:
#                 ids = ids[:, -self.model.block_size :]

#             logits, _ = self.model(ids)
#             logits = logits[:, -1, :]  # (B=1, vocab)

#             if temperature <= 0:
#                 next_id = torch.argmax(logits, dim=-1, keepdim=True)
#             else:
#                 logits = logits / float(temperature)
#                 logits = _top_k_logits(logits, int(top_k))
#                 probs = F.softmax(logits, dim=-1)
#                 next_id = torch.multinomial(probs, num_samples=1)

#             ids = torch.cat([ids, next_id], dim=1)

#             # decode only last char for efficiency
#             ch = self.tokenizer.decode([int(next_id.item())])
#             decoded += ch

#             # stop if END_TAG appears at the end (or anywhere near the end)
#             if stop_len > 0 and decoded.endswith(stop_tag):
#                 break

#         # Remove prompt part (reply only)
#         reply = decoded[len(prompt):]

#         # Cut at END_TAG if present
#         if stop_tag and stop_tag in reply:
#             reply = reply.split(stop_tag, 1)[0]

#         # Clean common junk
#         reply = reply.strip("\n\r\t ")

#         return reply









import os
import json
import torch
import torch.nn.functional as F

# Must match training imports
from train.model import MiniGPT
from train.tokenizer import CharTokenizer

END_TAG = "\nUser:"

def detect_lang(text: str) -> str:
    # very lightweight language detection for fallback replies
    for ch in text:
        o = ord(ch)
        if 0x0980 <= o <= 0x09FF:
            return "bn"
        if 0x0900 <= o <= 0x097F:
            return "hi"
        if 0x4E00 <= o <= 0x9FFF:
            return "zh"
    # naive Spanish hint
    if any(x in text.lower() for x in ["qué", "cómo", "acción", "apalancamiento", "hola"]):
        return "es"
    return "en"

def fallback_unknown(user_text: str) -> str:
    lang = detect_lang(user_text)
    if lang == "bn":
        return "আমি নিশ্চিত না / জানি না। আমি এই বিষয়ে নির্ভুল উত্তর দিতে পারি না।"
    if lang == "hi":
        return "मुझे नहीं पता। मैं इस बारे में भरोसेमंद जवाब नहीं दे सकता।"
    if lang == "es":
        return "No lo sé. No puedo darte una respuesta fiable sobre eso."
    if lang == "zh":
        return "我不知道。我无法给出可靠的答案。"
    return "I don’t know. I can’t give a reliable answer to that."


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float("inf")
    return out


class Runtime:
    def __init__(self, ckpt_path: str, meta_path: str, device: str | None = None):
        self.ckpt_path = ckpt_path
        self.meta_path = meta_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"meta.json not found: {self.meta_path}")
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"ckpt.pt not found: {self.ckpt_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        vocab_map = meta["vocab"]          # stoi
        config = meta["config"]
        self.tokenizer = CharTokenizer.from_vocab_map(vocab_map)

        self.model = MiniGPT(
            vocab_size=self.tokenizer.vocab_size,
            block_size=int(config["block_size"]),
            n_layer=int(config["n_layer"]),
            n_head=int(config["n_head"]),
            n_embd=int(config["n_embd"]),
            dropout=float(config.get("dropout", 0.0)),
        ).to(self.device)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.8,
        top_k: int = 60,
        stop_tag: str = END_TAG,
    ) -> str:
        """
        Returns ONLY assistant reply (no prompt echo), stops at stop_tag if present.
        """
        prompt = str(prompt)

        # Encode prompt
        ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

        # We'll build decoded output incrementally for stop detection
        decoded = prompt
        stop_len = len(stop_tag)

        for _ in range(int(max_new_tokens)):
            # crop to block size
            if ids.size(1) > self.model.block_size:
                ids = ids[:, -self.model.block_size :]

            logits, _ = self.model(ids)
            logits = logits[:, -1, :]  # (B=1, vocab)

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / float(temperature)
                logits = _top_k_logits(logits, int(top_k))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)

            # decode only last char for efficiency
            ch = self.tokenizer.decode([int(next_id.item())])
            decoded += ch

            # stop if END_TAG appears at the end (or anywhere near the end)
            if stop_len > 0 and decoded.endswith(stop_tag):
                break

        # Remove prompt part (reply only)
        reply = decoded[len(prompt):]

        # Cut at END_TAG if present
        if stop_tag and stop_tag in reply:
            reply = reply.split(stop_tag, 1)[0]

        # Clean common junk
        reply = reply.strip("\n\r\t ")

        # Trim to last complete sentence if reply seems cut off mid-sentence
        reply = self._trim_incomplete_sentence(reply)

        return reply

    def _trim_incomplete_sentence(self, text: str) -> str:
        """
        If the reply was cut off mid-sentence (hit max_new_tokens),
        trim back to the last sentence-ending punctuation so the
        response always ends cleanly.
        """
        if not text:
            return text

        SENTENCE_ENDS = {'.', '!', '?', '।', '…', '"', "'"}

        stripped = text.rstrip()
        if stripped and stripped[-1] in SENTENCE_ENDS:
            return text  # already ends cleanly

        # find the last sentence boundary
        last_end = -1
        for i, ch in enumerate(stripped):
            if ch in SENTENCE_ENDS:
                last_end = i

        if last_end == -1:
            # no sentence boundary found → short answer, return as-is
            return text

        # only trim if the dangling fragment after last boundary is substantial
        # (> 12 chars = likely a real cut-off, not trailing whitespace)
        fragment = stripped[last_end + 1:].strip()
        if len(fragment) > 12:
            return stripped[:last_end + 1].strip()

        return text
