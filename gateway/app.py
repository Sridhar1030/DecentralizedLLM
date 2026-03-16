"""
DecentralizedLLM Gateway: API key auth, circuit breaker, forwards to vLLM head.
"""
import os
import time

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

VALID_KEYS = {"sridhar-intern-2026"}
VLLM_URL = os.getenv("VLLM_URL", "http://head:8000")
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
CIRCUIT_COOLDOWN_SEC = int(os.getenv("CIRCUIT_COOLDOWN_SEC", "30"))

circuit_state = {"failures": 0, "open_until": 0}

GATEWAY_REQUESTS = Counter("sridhar_gateway_requests_total", "Total", ["status"])
GATEWAY_LATENCY = Histogram("sridhar_gateway_latency_seconds", "Latency")

app = FastAPI(title="DecentralizedLLM Gateway")


@app.post("/v1/chat/completions")
async def proxy_chat(payload: dict, x_api_key: str = Header(None)):
    if x_api_key not in VALID_KEYS:
        GATEWAY_REQUESTS.labels(status="auth_failed").inc()
        raise HTTPException(status_code=403, detail="Invalid API Key")
    if time.time() < circuit_state["open_until"]:
        GATEWAY_REQUESTS.labels(status="circuit_open").inc()
        raise HTTPException(status_code=503, detail="Inference cluster unavailable (circuit open)")
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # vLLM may not have /health; /v1/models is standard OpenAI-compatible
            ready = await client.get(f"{VLLM_URL}/v1/models")
            if ready.status_code != 200:
                raise Exception("vLLM not ready")
            resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
        circuit_state["failures"] = 0
        elapsed = time.perf_counter() - start
        GATEWAY_LATENCY.observe(elapsed)
        GATEWAY_REQUESTS.labels(status="success").inc()
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        circuit_state["failures"] += 1
        if circuit_state["failures"] >= CIRCUIT_FAILURE_THRESHOLD:
            circuit_state["open_until"] = time.time() + CIRCUIT_COOLDOWN_SEC
        GATEWAY_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
async def health():
    healthy = 0 if time.time() < circuit_state["open_until"] else 1
    return {"status": "ok", "healthy": healthy == 1, "vllm_url": VLLM_URL}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
