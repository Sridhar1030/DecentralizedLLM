"""
Coordinator: chains layer nodes for decentralized inference.
No single process holds the full model - request flows node0 -> node1 -> node2.
"""
import base64
import os
from typing import List, Optional

import httpx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer

NODE0_URL = os.getenv("NODE0_URL", "http://node0:8001")
NODE1_URL = os.getenv("NODE1_URL", "http://node1:8002")
NODE2_URL = os.getenv("NODE2_URL", "http://node2:8003")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

app = FastAPI(title="DecentralizedLLM Coordinator")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


async def forward_chain(input_ids: List[int]) -> np.ndarray:
    """Single forward pass through all 3 nodes. Returns logits for next token."""
    async with httpx.AsyncClient() as client:
        r0 = await client.post(
            f"{NODE0_URL}/forward", json={"input_ids": input_ids}, timeout=60
        )
        r0.raise_for_status()
        h0 = r0.json()["hidden_states_b64"]

        r1 = await client.post(
            f"{NODE1_URL}/forward", json={"hidden_states_b64": h0}, timeout=60
        )
        r1.raise_for_status()
        h1 = r1.json()["hidden_states_b64"]

        r2 = await client.post(
            f"{NODE2_URL}/forward", json={"hidden_states_b64": h1}, timeout=60
        )
        r2.raise_for_status()
        logits_b64 = r2.json()["logits_b64"]
        logits = np.frombuffer(base64.b64decode(logits_b64), dtype=np.float32)
        return logits


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 32


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """API key validated by gateway. Accepts OpenAI-compatible payload."""
    text = req.messages[-1].content if req.messages else ""
    enc = tokenizer(text, return_tensors=None)
    input_ids = enc["input_ids"]

    # Greedy decode for max_tokens
    gen_ids = list(input_ids)
    for _ in range(req.max_tokens - 1):
        logits = await forward_chain(gen_ids)
        next_id = int(np.argmax(logits))
        gen_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

    out_text = tokenizer.decode(gen_ids[len(input_ids) :], skip_special_tokens=True)
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": out_text},
                "finish_reason": "stop",
            }
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "mode": "decentralized-layer-nodes"}


@app.get("/v1/models")
def models():
    """Minimal OpenAI-compatible models list for gateway health check."""
    return {"data": [{"id": MODEL_NAME}]}
