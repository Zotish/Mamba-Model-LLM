# from dataclasses import dataclass
# from typing import Dict, List

# @dataclass
# class CharTokenizer:
#     stoi: Dict[str, int]
#     itos: Dict[int, str]

#     @classmethod
#     def from_text(cls, text: str) -> "CharTokenizer":
#         chars = sorted(list(set(text)))
#         stoi = {ch: i for i, ch in enumerate(chars)}
#         itos = {i: ch for ch, i in stoi.items()}
#         return cls(stoi=stoi, itos=itos)

#     @property
#     def vocab_size(self) -> int:
#         return len(self.stoi)

#     def encode(self, s: str) -> List[int]:
#         # Unknown chars are dropped; you can handle differently if you want.
#         return [self.stoi[ch] for ch in s if ch in self.stoi]

#     def decode(self, ids: List[int]) -> str:
#         return "".join(self.itos[i] for i in ids if i in self.itos)



# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from typing import Dict, List, Any, Tuple


# @dataclass
# class CharTokenizer:
#     stoi: Dict[str, int]
#     itos: Dict[int, str]

#     @classmethod
#     def from_text(cls, text: str) -> "CharTokenizer":
#         chars = sorted(list(set(text)))
#         stoi = {ch: i for i, ch in enumerate(chars)}
#         itos = {i: ch for ch, i in stoi.items()}
#         return cls(stoi=stoi, itos=itos)

#     @property
#     def vocab_size(self) -> int:
#         return len(self.stoi)

#     def encode(self, s: str) -> List[int]:
#         # unknown chars skip (safe)
#         return [self.stoi[ch] for ch in s if ch in self.stoi]

#     def decode(self, ids: List[int]) -> str:
#         return "".join(self.itos[i] for i in ids if i in self.itos)

#     # ✅ train.py calls: tok.save_meta(META_PATH, config)
#     def save_meta(self, meta_path: str, config: Dict[str, Any]) -> None:
#         meta = {
#             "vocab": self.stoi,  # ✅ server expects meta["vocab"]
#             "itos": {str(k): v for k, v in self.itos.items()},  # optional but useful
#             "config": config,
#         }
#         with open(meta_path, "w", encoding="utf-8") as f:
#             json.dump(meta, f, ensure_ascii=False, indent=2)

#     @classmethod
#     def load_meta(cls, meta_path: str) -> Tuple["CharTokenizer", Dict[str, Any]]:
#         with open(meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)

#         cfg = meta.get("config") or {}

#         # main format
#         if isinstance(meta.get("vocab"), dict):
#             stoi = {str(k): int(v) for k, v in meta["vocab"].items()}
#             itos = {int(v): str(k) for k, v in stoi.items()}
#             return cls(stoi=stoi, itos=itos), cfg

#         # fallback formats
#         if isinstance(meta.get("stoi"), dict) and isinstance(meta.get("itos"), dict):
#             stoi = {str(k): int(v) for k, v in meta["stoi"].items()}
#             itos = {int(k): str(v) for k, v in meta["itos"].items()}
#             return cls(stoi=stoi, itos=itos), cfg

#         if isinstance(meta.get("tokenizer"), dict):
#             t = meta["tokenizer"]
#             if isinstance(t.get("stoi"), dict) and isinstance(t.get("itos"), dict):
#                 stoi = {str(k): int(v) for k, v in t["stoi"].items()}
#                 itos = {int(k): str(v) for k, v in t["itos"].items()}
#                 return cls(stoi=stoi, itos=itos), cfg

#         raise RuntimeError(f"meta.json vocab not found. keys={list(meta.keys())}")

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Any
# import json


# @dataclass
# class CharTokenizer:
#     stoi: Dict[str, int]
#     itos: Dict[int, str]

#     @classmethod
#     def from_text(cls, text: str) -> "CharTokenizer":
#         chars = sorted(list(set(text)))
#         stoi = {ch: i for i, ch in enumerate(chars)}
#         itos = {i: ch for ch, i in stoi.items()}
#         return cls(stoi=stoi, itos=itos)

#     @property
#     def vocab_size(self) -> int:
#         return len(self.stoi)

#     def encode(self, s: str) -> List[int]:
#         # Unknown chars dropped (OK for toy model).
#         return [self.stoi[ch] for ch in s if ch in self.stoi]

#     def decode(self, ids: List[int]) -> str:
#         return "".join(self.itos[i] for i in ids if i in self.itos)

#     # ---------- META IO ----------
#     def save_meta(self, path: str, config: Dict[str, Any]) -> None:
#         """
#         meta.json format:
#         {
#           "vocab": {"char": int, ...},
#           "config": {...}
#         }
#         """
#         meta = {"vocab": self.stoi, "config": config}
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(meta, f, ensure_ascii=False)

#     @classmethod
#     def load_meta(cls, path: str) -> Tuple["CharTokenizer", Dict[str, Any]]:
#         with open(path, "r", encoding="utf-8") as f:
#             meta = json.load(f)

#         # accept older keys too (backward compatible)
#         vocab = meta.get("vocab") or meta.get("stoi") or meta.get("vocab_stoi")
#         if vocab is None:
#             raise KeyError("meta.json missing vocab/stoi key")

#         stoi = {k: int(v) for k, v in vocab.items()}
#         itos = {i: ch for ch, i in stoi.items()}
#         config = meta.get("config", {})
#         return cls(stoi=stoi, itos=itos), config

# train/tokenizer.py
class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @staticmethod
    def from_text(text: str):
        vocab = sorted(list(set(text)))
        stoi = {ch:i for i,ch in enumerate(vocab)}
        itos = {i:ch for ch,i in stoi.items()}
        return CharTokenizer(stoi, itos)

    @staticmethod
    def from_vocab_map(vocab_map: dict):
        """
        vocab_map = {char: id}
        """
        stoi = {str(k): int(v) for k, v in vocab_map.items()}
        itos = {int(v): str(k) for k, v in stoi.items()}
        return CharTokenizer(stoi, itos)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
