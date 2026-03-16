"""
Layer node: holds a subset of model layers. No single node has the full LLM.
Part of DecentralizedLLM - custom pipeline parallelism for CPU.
"""
import base64
import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM

# Which layers this node holds. Set via env: NODE_LAYERS="0-7" | "8-15" | "16-23"
LAYER_RANGE = os.getenv("NODE_LAYERS", "0-7")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

start_layer, end_layer = map(int, LAYER_RANGE.split("-"))
assert 0 <= start_layer < end_layer <= 24, f"Invalid NODE_LAYERS {LAYER_RANGE}"

app = FastAPI(title=f"LayerNode-{LAYER_RANGE}")

model = None


def load_model():
    global model
    print(f"Loading {MODEL_NAME} layers {start_layer}-{end_layer-1}...")
    full = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    config = full.config

    # Build minimal model with our layers only
    if start_layer == 0:
        # First node: embed + layers 0-7
        model = Qwen2ForCausalLM(config)
        model.model.embed_tokens = full.model.embed_tokens
        model.model.layers = full.model.layers[start_layer:end_layer]
        model.model.norm = torch.nn.Identity()  # No final norm on first/mid nodes
        model.lm_head = torch.nn.Identity()
    elif end_layer == 24:
        # Last node: layers 16-23 + norm + lm_head
        model = Qwen2ForCausalLM(config)
        model.model.embed_tokens = torch.nn.Identity()
        model.model.layers = full.model.layers[start_layer:end_layer]
        model.model.norm = full.model.norm
        model.lm_head = full.lm_head
    else:
        # Middle node: layers 8-15 only
        model = Qwen2ForCausalLM(config)
        model.model.embed_tokens = torch.nn.Identity()
        model.model.layers = full.model.layers[start_layer:end_layer]
        model.model.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()

    del full
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model.eval()
    print(f"Ready. Layers {start_layer}-{end_layer-1} loaded.")


class ForwardRequest(BaseModel):
    hidden_states_b64: Optional[str] = None  # Base64 numpy
    input_ids: Optional[list] = None  # For first node


@app.on_event("startup")
def startup():
    load_model()


@app.post("/forward")
def forward(req: ForwardRequest):
    """Run forward pass. Returns hidden_states (b64) or logits (last node)."""
    with torch.no_grad():
        if req.input_ids is not None:
            # First node: embed + layers
            ids = torch.tensor([req.input_ids], dtype=torch.long)
            out = model(ids, output_hidden_states=True)
            hidden = out.hidden_states[-1].numpy()
        else:
            # Middle/last node: hidden_states in
            buf = base64.b64decode(req.hidden_states_b64)
            hidden = np.frombuffer(buf, dtype=np.float32).reshape(
                -1, model.config.hidden_size
            )
            hidden = torch.tensor(hidden, dtype=torch.float32)
            out = model(inputs_embeds=hidden.unsqueeze(0), output_hidden_states=True)
            hidden = out.hidden_states[-1].numpy()

        if end_layer == 24:
            logits = out.logits[:, -1, :].numpy()
            return {"logits_b64": base64.b64encode(logits.tobytes()).decode()}
        else:
            return {"hidden_states_b64": base64.b64encode(hidden.tobytes()).decode()}


@app.get("/health")
def health():
    return {"status": "ok", "layers": LAYER_RANGE}
