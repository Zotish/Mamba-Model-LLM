from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from typing import Optional
import traceback   # ✅ add

router = APIRouter()
RUNTIME = None

def set_runtime(rt):
    global RUNTIME
    RUNTIME = rt

class GenerateReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = 220
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    no_repeat_ngram: int = 0

class GenerateResp(BaseModel):
    text: str

@router.post("/api/generate", response_model=GenerateResp)
def generate(req: GenerateReq):
    if RUNTIME is None:
        raise HTTPException(status_code=503, detail="Model runtime not loaded")

    try:
        out = RUNTIME.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        return {"text": out}
    except Exception as e:
        traceback.print_exc()  # ✅ terminal এ full stack দেখাবে
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
