"""
Sridhar-Mesh Node: FastAPI proxy to vLLM backend.
Exposes /health, /metrics, /v1/chat/completions.
"""
import asyncio
import os
import random
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

VLLM_URL = os.getenv("VLLM_URL", "http://host.docker.internal:8000")
NODE_ID = os.getenv("NODE_ID", "node-1")
SIMULATED_LATENCY_MS = float(os.getenv("SIMULATED_LATENCY_MS", "0"))
FAILURE_RATE = float(os.getenv("FAILURE_RATE", "0"))

REQUEST_COUNT = Counter("sridhar_node_requests_total", "Total requests", ["node_id", "status"])
REQUEST_LATENCY = Histogram("sridhar_node_request_latency_seconds", "Request latency", ["node_id"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Sridhar-Mesh Node", lifespan=lifespan)


@app.get("/health")
async def health():
    if random.random() < FAILURE_RATE:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "node": NODE_ID})
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{VLLM_URL}/health")
            if r.status_code == 200:
                return {"status": "healthy", "node": NODE_ID}
            return JSONResponse(
                status_code=503, content={"status": "downstream_unhealthy", "node": NODE_ID}
            )
        except Exception as e:
            return JSONResponse(
                status_code=503, content={"status": "error", "error": str(e), "node": NODE_ID}
            )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start = time.perf_counter()
    if random.random() < FAILURE_RATE:
        REQUEST_COUNT.labels(node_id=NODE_ID, status="simulated_failure").inc()
        return JSONResponse(status_code=503, content={"error": "Simulated failure"})
    if SIMULATED_LATENCY_MS > 0:
        await asyncio.sleep(SIMULATED_LATENCY_MS / 1000.0)
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=body)
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(node_id=NODE_ID).observe(elapsed)
        status = "success" if r.status_code == 200 else "error"
        REQUEST_COUNT.labels(node_id=NODE_ID, status=status).inc()
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:
        REQUEST_COUNT.labels(node_id=NODE_ID, status="error").inc()
        return JSONResponse(status_code=503, content={"error": str(e)})