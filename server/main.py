import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.generate import router as generate_router, set_runtime

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Priority: out_alpaca > out_mamba > out (GPT)
ALPACA_CKPT = os.path.join(PROJECT_ROOT, "out_alpaca", "ckpt.pt")
ALPACA_META = os.path.join(PROJECT_ROOT, "out_alpaca", "meta.json")

MAMBA_CKPT  = os.path.join(PROJECT_ROOT, "out_mamba", "ckpt.pt")
MAMBA_META  = os.path.join(PROJECT_ROOT, "out_mamba", "meta.json")

GPT_CKPT    = os.path.join(PROJECT_ROOT, "out", "ckpt.pt")
GPT_META    = os.path.join(PROJECT_ROOT, "out", "meta.json")

if os.path.exists(ALPACA_CKPT) and os.path.exists(ALPACA_META):
    USE_MAMBA = True
    CKPT, META = ALPACA_CKPT, ALPACA_META
    print("📦 Using Alpaca checkpoint: out_alpaca/")
elif os.path.exists(MAMBA_CKPT) and os.path.exists(MAMBA_META):
    USE_MAMBA = True
    CKPT, META = MAMBA_CKPT, MAMBA_META
    print("📦 Using Mamba checkpoint: out_mamba/")
else:
    USE_MAMBA = False
    CKPT = GPT_CKPT
    META = GPT_META
    print("📦 Using GPT checkpoint: out/")

if USE_MAMBA:
    from server.mamba_runtime import MambaRuntime
else:
    from server.model_runtime import Runtime

app = FastAPI(title="Scratch Mini LLM (Chat-style)")

# ✅ Correct CORS:
# allow_credentials=True হলে "*" দেওয়া যাবে না।
# Dev frontend (Vite) origins গুলো explicitly দাও।
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)

@app.on_event("startup")
def startup():
    if USE_MAMBA:
        rt = MambaRuntime(ckpt_path=CKPT, meta_path=META)
        print("🐍 MambaLM loaded:", CKPT)
    else:
        rt = Runtime(ckpt_path=CKPT, meta_path=META)
        print("✅ MiniGPT loaded:", CKPT)
    set_runtime(rt)

@app.get("/health")
def health():
    return {"ok": True}
