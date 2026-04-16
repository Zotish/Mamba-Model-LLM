




# import os
# import json
# import time
# import argparse
# import torch
# from torch.optim import AdamW

# from tokenizer import CharTokenizer
# from model import MiniGPT

# # -----------------
# # Paths
# # -----------------
# DATA_PATH = os.path.join("data", "instruction.txt")
# OUT_DIR = "out"
# CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
# META_PATH = os.path.join(OUT_DIR, "meta.json")

# os.makedirs(OUT_DIR, exist_ok=True)

# # -----------------
# # Device (Mac-safe)
# # -----------------
# def pick_device():
#     if os.getenv("FORCE_CPU", "1") == "1":
#         return torch.device("cpu")
#     if torch.backends.mps.is_available():
#         return torch.device("mps")
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     return torch.device("cpu")

# # -----------------
# # Data
# # -----------------
# def load_text():
#     with open(DATA_PATH, "r", encoding="utf-8") as f:
#         return f.read()

# def batchify(data, batch_size, block_size, device):
#     ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     return x.to(device), y.to(device)

# # -----------------
# # Meta / checkpoint
# # -----------------
# def save_meta(tok, cfg):
#     meta = {"vocab": tok.stoi, "config": cfg}
#     with open(META_PATH, "w", encoding="utf-8") as f:
#         json.dump(meta, f, ensure_ascii=False)

# def load_meta():
#     if not os.path.exists(META_PATH):
#         return None
#     with open(META_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# def ckpt_compatible(model, ckpt):
#     try:
#         model.load_state_dict(ckpt["model"], strict=True)
#         return True
#     except RuntimeError:
#         return False

# def load_ckpt(model, opt):
#     if not os.path.exists(CKPT_PATH):
#         return 0

#     ckpt = torch.load(CKPT_PATH, map_location="cpu")

#     if not ckpt_compatible(model, ckpt):
#         print("⚠️ Checkpoint incompatible with current model. Ignoring old checkpoint.")
#         return 0

#     model.load_state_dict(ckpt["model"])
#     if "opt" in ckpt:
#         try:
#             opt.load_state_dict(ckpt["opt"])
#         except Exception:
#             pass

#     step = int(ckpt.get("step", 0))
#     print(f"🔄 Resumed from step {step}")
#     return step

# def save_ckpt(model, opt, step):
#     torch.save(
#         {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
#         CKPT_PATH
#     )

# # -----------------
# # Main
# # -----------------
# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--block_size", type=int, default=64)
#     parser.add_argument("--accum", type=int, default=16)

#     parser.add_argument("--n_layer", type=int, default=4)
#     parser.add_argument("--n_head", type=int, default=4)
#     parser.add_argument("--n_embd", type=int, default=256)
#     parser.add_argument("--dropout", type=float, default=0.1)

#     parser.add_argument("--lr", type=float, default=3e-4)
#     parser.add_argument("--weight_decay", type=float, default=0.01)

#     parser.add_argument("--max_steps", type=int, default=5000)
#     parser.add_argument("--log_every", type=int, default=50)
#     parser.add_argument("--save_every", type=int, default=200)
#     parser.add_argument("--sleep_ms", type=float, default=40)

#     args = parser.parse_args()

#     device = pick_device()
#     torch.set_num_threads(2)

#     print("✅ Device:", device)

#     # -----------------
#     # Load data
#     # -----------------
#     text = load_text()
#     tok = CharTokenizer.from_text(text)
#     ids = torch.tensor(tok.encode(text), dtype=torch.long)

#     n = int(0.9 * len(ids))
#     train_ids = ids[:n]
#     val_ids = ids[n:]

#     cfg = {
#         "block_size": args.block_size,
#         "n_layer": args.n_layer,
#         "n_head": args.n_head,
#         "n_embd": args.n_embd,
#         "dropout": args.dropout,
#     }

#     save_meta(tok, cfg)

#     model = MiniGPT(
#         vocab_size=tok.vocab_size,
#         block_size=cfg["block_size"],
#         n_layer=cfg["n_layer"],
#         n_head=cfg["n_head"],
#         n_embd=cfg["n_embd"],
#         dropout=cfg["dropout"],
#     ).to(device)

#     opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     step = load_ckpt(model, opt)

#     model.train()
#     t0 = time.time()

#     while step < args.max_steps:
#         opt.zero_grad()
#         total_loss = 0.0

#         for _ in range(args.accum):
#             xb, yb = batchify(train_ids, args.batch_size, cfg["block_size"], device)
#             _, loss = model(xb, yb)
#             (loss / args.accum).backward()
#             total_loss += loss.item()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         opt.step()
#         step += 1

#         if args.sleep_ms > 0:
#             time.sleep(args.sleep_ms / 1000)

#         if step % args.log_every == 0:
#             elapsed = time.time() - t0
#             print(f"step {step} | loss {total_loss/args.accum:.4f} | {elapsed/60:.1f} min")

#         if step % args.save_every == 0:
#             save_ckpt(model, opt, step)
#             print("💾 checkpoint saved")

#     save_ckpt(model, opt, step)
#     print("✅ Training finished")

# if __name__ == "__main__":
#     main()

# train/train.py
from __future__ import annotations

import os
import json
import time
import signal
import argparse
from typing import Dict, List

import torch
from torch.optim import AdamW

from tokenizer import CharTokenizer
from model import MiniGPT


# -----------------
# Paths
# -----------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "input.txt")

OUT_DIR = os.path.join(BASE_DIR, "out")
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
META_PATH = os.path.join(OUT_DIR, "meta.json")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------
# Device / cooling
# -----------------
def pick_device(force_cpu: bool = True) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_safe_threads(n: int):
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


# -----------------
# Data helpers
# -----------------
def load_text() -> str:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def batchify(data_ids: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    # random chunks
    if len(data_ids) <= block_size + 2:
        raise RuntimeError("Dataset too small for chosen block_size. Add more text or reduce block_size.")
    ix = torch.randint(0, len(data_ids) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i : i + block_size] for i in ix])
    y = torch.stack([data_ids[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: MiniGPT, data_ids: torch.Tensor, batch_size: int, block_size: int, device: torch.device, iters=10):
    model.eval()
    losses: List[float] = []
    for _ in range(iters):
        xb, yb = batchify(data_ids, batch_size, block_size, device)
        _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


# -----------------
# Meta / checkpoint
# -----------------
def save_meta(tokenizer: CharTokenizer, config: dict):
    meta = {"vocab": tokenizer.stoi, "config": config}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_meta_if_exists():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ckpt(model: MiniGPT, optimizer: AdamW, step: int):
    ckpt = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "step": step,
        "vocab_size": model.vocab_size,
        "block_size": model.block_size,
    }
    torch.save(ckpt, CKPT_PATH)


def is_ckpt_compatible(ckpt: dict, vocab_size: int, block_size: int) -> bool:
    try:
        return int(ckpt.get("vocab_size", -1)) == int(vocab_size) and int(ckpt.get("block_size", -1)) == int(block_size)
    except Exception:
        return False


def load_ckpt_if_exists(model: MiniGPT, optimizer: AdamW, allow_resume: bool = True) -> int:
    if not allow_resume or not os.path.exists(CKPT_PATH):
        return 0

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if not is_ckpt_compatible(ckpt, model.vocab_size, model.block_size):
        print("⚠️ Found checkpoint but config/vocab mismatch. Starting fresh (no resume).")
        return 0

    model.load_state_dict(ckpt["model"])
    try:
        optimizer.load_state_dict(ckpt["opt"])
    except Exception:
        pass
    return int(ckpt.get("step", 0))


def fmt(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


# -----------------
# Main
# -----------------
def main():
    p = argparse.ArgumentParser()

    # Training shape (cool defaults)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--accum", type=int, default=16)

    # Optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Model size (M1 Air friendly)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Cooling controls
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--mps", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--sleep_ms", type=float, default=40.0)
    p.add_argument("--micro_sleep_ms", type=float, default=8.0)

    # Auto-break / runtime controls
    p.add_argument("--run_hours", type=float, default=2.0)
    p.add_argument("--session_min", type=float, default=12.0)
    p.add_argument("--break_min", type=float, default=8.0)
    p.add_argument("--eval_every", type=int, default=240)
    p.add_argument("--save_every", type=int, default=120)
    p.add_argument("--log_every", type=int, default=40)

    # Resume behavior
    p.add_argument("--no_resume", action="store_true", help="Ignore checkpoint and start from scratch")

    args = p.parse_args()

    # Device
    force_cpu = True
    if args.mps:
        force_cpu = False
    if args.force_cpu:
        force_cpu = True

    device = pick_device(force_cpu=force_cpu)
    set_safe_threads(args.threads)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("✅ Device:", device)
    print("✅ Threads:", args.threads)
    print("✅ batch_size:", args.batch_size, "| block_size:", args.block_size, "| accum:", args.accum)
    print("✅ Auto-break:", f"{args.session_min}min train / {args.break_min}min break")
    print("✅ Total run_hours:", args.run_hours)
    print("✅ Resume:", "OFF" if args.no_resume else "ON")

    # Load dataset
    text = load_text()

    # Tokenizer and ids
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    # Split train/val
    n = int(0.9 * len(ids))
    train_ids = ids[:n]
    val_ids = ids[n:]

    # Config saved for runtime
    config = {
        "block_size": int(args.block_size),
        "n_layer": int(args.n_layer),
        "n_head": int(args.n_head),
        "n_embd": int(args.n_embd),
        "dropout": float(args.dropout),
    }
    save_meta(tok, config)

    model = MiniGPT(
        vocab_size=tok.vocab_size,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume safely
    step = load_ckpt_if_exists(model, opt, allow_resume=(not args.no_resume))
    if step > 0:
        print(f"🔄 Resumed from step {step}")

    # Ctrl+C safe stop
    stop = {"flag": False}

    def on_sigint(sig, frame):
        stop["flag"] = True
        print("\n🛑 Ctrl+C detected. Will save and exit safely...")

    signal.signal(signal.SIGINT, on_sigint)

    # Wall-time / break scheduling
    run_deadline = time.time() + args.run_hours * 3600.0
    session_seconds = args.session_min * 60.0
    break_seconds = args.break_min * 60.0
    session_start = time.time()

    t0 = time.time()
    model.train()

    while time.time() < run_deadline and not stop["flag"]:
        # Auto-break
        if time.time() - session_start >= session_seconds:
            save_ckpt(model, opt, step)
            elapsed = time.time() - t0
            print(f"\n🧊 Auto-break START at step {step} | elapsed {fmt(elapsed)}")
            print(f"💾 checkpoint saved -> {CKPT_PATH}")
            print(f"😴 cooling for {args.break_min} minutes...\n")
            time.sleep(break_seconds)
            session_start = time.time()
            print("🔥 Training RESUMED\n")

        # Train step (gradient accumulation)
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(args.accum):
            xb, yb = batchify(train_ids, args.batch_size, config["block_size"], device)
            _, loss = model(xb, yb)
            (loss / args.accum).backward()
            total_loss += float(loss.item())

            if args.micro_sleep_ms > 0:
                time.sleep(args.micro_sleep_ms / 1000.0)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / max(args.accum, 1)
            remaining = max(0.0, run_deadline - time.time())
            print(f"step {step} | loss {avg_loss:.4f} | elapsed {fmt(elapsed)} | remaining {fmt(remaining)}")

        if step % args.save_every == 0:
            save_ckpt(model, opt, step)
            print(f"💾 saved ckpt at step {step}")

        if step % args.eval_every == 0:
            tr = estimate_loss(model, train_ids, args.batch_size, config["block_size"], device, iters=8)
            va = estimate_loss(model, val_ids, args.batch_size, config["block_size"], device, iters=8)
            print(f"[eval] step {step} | train {tr:.4f} | val {va:.4f}")
            save_ckpt(model, opt, step)
            print("✅ eval ckpt saved")

    # Final save
    save_ckpt(model, opt, step)
    elapsed = time.time() - t0
    print(f"\n✅ Finished. Final step {step} | total elapsed {fmt(elapsed)}")
    print(f"💾 final checkpoint -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
