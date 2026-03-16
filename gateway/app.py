"""
Sridhar-Mesh Gateway: API key auth, circuit breaker, round-robin load balancing.
"""
import os
import time
from itertools import cycle

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

VALID_KEYS = {"sridhar-intern-2026"}
NODES = [
    "http://node-1:8000",
    "http://node-2:8000",
    "http://node-3:8000",
]
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
CIRCUIT_COOLDOWN_SEC = int(os.getenv("CIRCUIT_COOLDOWN_SEC", "30"))

circuit_state = {n: {"failures": 0, "open_until": 0} for n in NODES}
node_cycle = cycle(NODES)

GATEWAY_REQUESTS = Counter("sridhar_gateway_requests_total", "Total", ["status"])
GATEWAY_LATENCY = Histogram("sridhar_gateway_latency_seconds", "Latency")

app = FastAPI(title="Sridhar-Mesh Gateway")


def get_next_node():
    for _ in range(len(NODES)):
        node = next(node_cycle)
        state = circuit_state[node]
        if time.time() > state["open_until"]:
            return node
    return None


@app.post("/v1/chat/completions")
async def proxy_chat(payload: dict, x_api_key: str = Header(None)):
    if x_api_key not in VALID_KEYS:
        GATEWAY_REQUESTS.labels(status="auth_failed").inc()
        raise HTTPException(status_code=403, detail="Invalid API Key")
    node = get_next_node()
    if not node:
        GATEWAY_REQUESTS.labels(status="circuit_open").inc()
        raise HTTPException(status_code=503, detail="All nodes unavailable (circuit open)")
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            health = await client.get(f"{node}/health")
            if health.status_code != 200:
                raise Exception("Node unhealthy")
            resp = await client.post(f"{node}/v1/chat/completions", json=payload)
        circuit_state[node]["failures"] = 0
        elapsed = time.perf_counter() - start
        GATEWAY_LATENCY.observe(elapsed)
        GATEWAY_REQUESTS.labels(status="success").inc()
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        circuit_state[node]["failures"] += 1
        if circuit_state[node]["failures"] >= CIRCUIT_FAILURE_THRESHOLD:
            circuit_state[node]["open_until"] = time.time() + CIRCUIT_COOLDOWN_SEC
        GATEWAY_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
async def health():
    healthy = sum(1 for n in NODES if time.time() > circuit_state[n]["open_until"])
    return {"status": "ok", "healthy_nodes": healthy, "total_nodes": len(NODES)}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
