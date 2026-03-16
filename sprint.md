# Sridhar-Mesh – Decentralized Inference PoC

## Sprint-by-Sprint Implementation Guide

**Objective:** Simulate a 3-node decentralized LLM cluster on Apple Silicon with vllm-metal, automated failover, health-check gateway, and secure public tunneling.

**Prerequisites:** macOS on Apple Silicon (M1/M2/M3/M4), 32GB RAM recommended, Docker Desktop, Homebrew.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │  Cloudflare Tunnel (HTTPS public URL)  │
                    └─────────────────────┬─────────────────┘
                                          │
                    ┌─────────────────────▼─────────────────┐
                    │  Gateway :8080                         │
                    │  API Keys | Circuit Breaker | LB       │
                    └─────────────────────┬─────────────────┘
                                          │
         ┌────────────────┬────────────────┼────────────────┬────────────────┐
         │                │                │                │                │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐     ┌─────▼─────┐    ┌─────▼─────┐
    │ node-1  │      │ node-2  │      │ node-3  │     │ Prometheus │    │  Grafana  │
    │  :8001  │      │  :8002  │      │  :8003  │     │   :9090    │    │   :3000   │
    └────┬────┘      └────┬────┘      └────┬────┘     └─────┬───────┘    └─────┬─────┘
         │                │                │                │                 │
         └────────────────┴────────────────┴────────────────┘                 │
                                          │                                     │
                    ┌─────────────────────▼─────────────────┐                    │
                    │  vllm-metal (Host) :8000               │◄───────────────────┘
                    │  Qwen2.5-0.5B-Instruct | /metrics     │
                    └──────────────────────────────────────┘
```

---

# Sprint 0: Environment Setup (~15 min) — Done

## Step 0.1: Verify Docker Desktop

```bash
# Check Docker is installed and running
docker --version
docker info

# If Docker daemon is not running, start Docker Desktop from Applications
# Docker Desktop 4.62+ required for Model Runner (optional path)
# For this project we use vllm-metal standalone, so any recent Docker works
```

**Verification:** `docker info` should print system info without errors.

---

## Step 0.2: Install vllm-metal

```bash
# Install vllm-metal (creates ~/.venv-vllm-metal)
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

# Activate the environment (add to your shell profile for persistence)
source ~/.venv-vllm-metal/bin/activate

# Verify installation
which vllm
vllm --version
```

**Verification:** `vllm --version` prints a version string.

**Troubleshooting:**

- If `curl` fails: ensure you have internet; try `curl -v` for verbose output.
- If install fails: `rm -rf ~/.venv-vllm-metal` and re-run. Requires Python 3.10–3.12; if you have 3.14, consider `pyenv install 3.12` and use that.

---

## Step 0.3: Install cloudflared

```bash
brew install cloudflared

# Verify
cloudflared --version
```

**Verification:** Version string printed.

---

## Step 0.4: Create Project Structure

```bash
cd /Users/srpillai/CODING/DecentralizedLLM

mkdir -p node gateway prometheus grafana/provisioning/datasources grafana/provisioning/dashboards
```

**Verification:** `ls -la` shows `node`, `gateway`, `prometheus`, `grafana` directories.

---

# Sprint 1: Inference Engine (~20 min) — Done

## Step 1.1: Start vllm-metal Server

**In a dedicated terminal (keep it running):**

```bash
cd /Users/srpillai/CODING/DecentralizedLLM
source ~/.venv-vllm-metal/bin/activate

# Serve Qwen2.5-0.5B-Instruct (MLX format for Apple Silicon)
vllm serve mlx-community/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --host 0.0.0.0
```

**Note:** First run downloads the model (~600MB). Wait for "Application startup complete" or similar.

**Alternative model** (if above fails): `mlx-community/qwen2.5-0.5b-instruct-q2`

---

## Step 1.2: Verify OpenAI-Compatible API

```bash
# In a new terminal
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 32
  }' | jq .
```

**Verification:** JSON response with `choices[0].message.content` containing a reply.

---

## Step 1.3: Verify Metrics Endpoint

```bash
curl -s http://localhost:8000/metrics | head -30
```

**Verification:** Output starts with `# HELP` and `# TYPE` (Prometheus format).

---

# Sprint 2: Node Containers (~30 min) — Done

## Step 2.1: Create node/requirements.txt

```txt
fastapi==0.115.6
uvicorn[standard]==0.32.1
httpx==0.28.1
prometheus-client==0.21.0
```

---

## Step 2.2: Create node/app.py

Create `node/app.py` with this content (copy-paste as-is):

```python
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
```

---

## Step 2.3: Create node/Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 2.4: Create docker-compose.yml (Initial – Nodes Only)

```yaml
services:
  node-1:
    build: ./node
    container_name: sridhar-node-1
    environment:
      NODE_ID: node-1
      VLLM_URL: http://host.docker.internal:8000
    ports:
      - "8001:8000"
    networks:
      - decentralized-net
    extra_hosts:
      - "host.docker.internal:host-gateway"

  node-2:
    build: ./node
    container_name: sridhar-node-2
    environment:
      NODE_ID: node-2
      VLLM_URL: http://host.docker.internal:8000
    ports:
      - "8002:8000"
    networks:
      - decentralized-net
    extra_hosts:
      - "host.docker.internal:host-gateway"

  node-3:
    build: ./node
    container_name: sridhar-node-3
    environment:
      NODE_ID: node-3
      VLLM_URL: http://host.docker.internal:8000
    ports:
      - "8003:8000"
    networks:
      - decentralized-net
    extra_hosts:
      - "host.docker.internal:host-gateway"

networks:
  decentralized-net:
    driver: bridge
```

---

## Step 2.5: Build and Run Nodes

```bash
cd /Users/srpillai/CODING/DecentralizedLLM

# Ensure vllm-metal is running on host:8000 (Sprint 1)
# Build and start nodes
docker compose build
docker compose up -d node-1 node-2 node-3

# Verify
curl -s http://localhost:8001/health
curl -s http://localhost:8002/health
curl -s http://localhost:8003/health
```

**Verification:** Each returns `{"status":"healthy","node":"node-N"}`.

---

## Step 2.6: Test Inference Through Node

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 16
  }' | jq '.choices[0].message.content'
```

**Verification:** Returns generated text.

---

# Sprint 3: Gateway (~30 min) — Done

## Step 3.1: Create gateway/requirements.txt

```txt
fastapi==0.115.6
uvicorn[standard]==0.32.1
httpx==0.28.1
prometheus-client==0.21.0
```

---

## Step 3.2: Create gateway/app.py

```python
"""
Sridhar-Mesh Gateway: API key auth, circuit breaker, round-robin load balancing.
"""
import os
import time
from itertools import cycle

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, Response
import httpx
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

VALID_KEYS = {"sridhar-intern-2026"}
NODES = [
    "http://node-1:8000",
    "http://node-2:8000",
    "http://node-3:8000",
]
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
CIRCUIT_COOLDOWN_SEC = int(os.getenv("CIRCUIT_COOLDOWN_SEC", "30"))

# Circuit state: {node_url: {"failures": int, "open_until": float}}
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
```

---

## Step 3.3: Create gateway/Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Step 3.4: Update docker-compose.yml – Add Gateway

Add to `docker-compose.yml` (and fix node service names for internal DNS):

```yaml
  gateway:
    build: ./gateway
    container_name: sridhar-gateway
    ports:
      - "8080:8080"
    depends_on:
      - node-1
      - node-2
      - node-3
    networks:
      - decentralized-net
```

Update node services to use `container_name` for DNS. Docker Compose uses service name by default, so we need `node-1`, `node-2`, `node-3` as service names (they already are). The gateway connects to `http://node-1:8000` etc. But nodes expose port 8000 internally – correct.

---

## Step 3.5: Build and Test Gateway

```bash
docker compose build gateway
docker compose up -d gateway

# Without API key (should fail)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"Hi"}]}'
# Expect 403

# With API key (should succeed)
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: sridhar-intern-2026" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 32
  }' | jq '.choices[0].message.content'
```

**Verification:** Returns generated text.

---

# Sprint 4: Observability (~25 min) — Done

## Step 4.1: Create prometheus/prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "node-1"
    static_configs:
      - targets: ["node-1:8000"]
        labels:
          node: "node-1"
  - job_name: "node-2"
    static_configs:
      - targets: ["node-2:8000"]
        labels:
          node: "node-2"
  - job_name: "node-3"
    static_configs:
      - targets: ["node-3:8000"]
        labels:
          node: "node-3"
  - job_name: "gateway"
    static_configs:
      - targets: ["gateway:8080"]
        labels:
          service: "gateway"
  - job_name: "vllm"
    static_configs:
      - targets: ["host.docker.internal:8000"]
        labels:
          service: "vllm-metal"
```

**Note:** For `host.docker.internal` in Prometheus (running in Docker), use `host.docker.internal` – it works on Docker Desktop for Mac.

---

## Step 4.2: Create grafana/provisioning/datasources/prometheus.yml

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    uid: prometheus
```

---

## Step 4.3: Create grafana/provisioning/dashboards/dashboard.yml

```yaml
apiVersion: 1
providers:
  - name: "Sridhar-Mesh"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/provisioning/dashboards
```

---

## Step 4.4: Create grafana/provisioning/dashboards/sridhar-mesh.json

Put the dashboard JSON in the same folder as `dashboard.yml`. Key panels:

- Node request count by node_id
- Gateway latency histogram
- vLLM tokens/sec (if exposed)
- Circuit breaker status (from gateway health)

A minimal valid dashboard:

```json
{
  "annotations": {"list": []},
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {"defaults": {}, "overrides": []},
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
      "id": 1,
      "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
      "targets": [{"expr": "sum by (node_id) (rate(sridhar_node_requests_total[5m]))", "refId": "A"}],
      "title": "Node Requests/sec",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {"defaults": {}, "overrides": []},
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
      "id": 2,
      "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
      "targets": [{"expr": "rate(sridhar_gateway_latency_seconds_sum[5m]) / rate(sridhar_gateway_latency_seconds_count[5m])", "refId": "A"}],
      "title": "Gateway Avg Latency (s)",
      "type": "timeseries"
    },
    {
      "datasource": {"type": "prometheus", "uid": "prometheus"},
      "fieldConfig": {"defaults": {}, "overrides": []},
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
      "id": 3,
      "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
      "targets": [{"expr": "sum by (status) (sridhar_gateway_requests_total)", "refId": "A"}],
      "title": "Gateway Requests by Status",
      "type": "timeseries"
    }
  ],
  "schemaVersion": 38,
  "style": "dark",
  "tags": ["sridhar-mesh"],
  "templating": {"list": []},
  "time": {"from": "now-15m", "to": "now"},
  "timepicker": {},
  "timezone": "",
  "title": "Sridhar-Mesh Dashboard",
  "uid": "sridhar-mesh",
  "version": 1,
  "weekStart": ""
}
```

---

## Step 4.5: Update docker-compose.yml – Add Prometheus & Grafana

```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: sridhar-prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    command: ["--config.file=/etc/prometheus/prometheus.yml", "--web.enable-lifecycle"]
    networks:
      - decentralized-net
    extra_hosts:
      - "host.docker.internal:host-gateway"

  grafana:
    image: grafana/grafana:latest
    container_name: sridhar-grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: "false"
      GF_SERVER_HTTP_PORT: 3000
    volumes:
      - ./grafana/provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - decentralized-net
```

**Note:** Both `dashboard.yml` and `sridhar-mesh.json` live in `grafana/provisioning/dashboards/`. The provider loads JSON from that path.

---

## Step 4.6: Start Observability Stack

```bash
docker compose up -d prometheus grafana

# Verify Prometheus
curl -s "http://localhost:9090/api/v1/targets" | jq '.data.activeTargets | length'
# Should show 5 targets (3 nodes, gateway, vllm)

# Open Grafana: http://localhost:3000
# Login: admin / admin
# Dashboards -> Sridhar-Mesh Dashboard
```

**Verification:** Grafana shows the dashboard with panels.

---

# Sprint 5: Public Access and Polish (~15 min) — Done

## Step 5.1: Start Cloudflare Tunnel

```bash
# In a new terminal
cloudflared tunnel --url http://localhost:8080
```

**Verification:** You get a `*.trycloudflare.com` URL. Test:

```bash
export TUNNEL_URL="https://YOUR-URL.trycloudflare.com"
curl -s -X POST "$TUNNEL_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sridhar-intern-2026" \
  -d '{"model":"x","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' | jq .
```

---

## Step 5.2: Failover Demo

```bash
# 1. Send a few requests (should work)
for i in 1 2 3 4 5; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "x-api-key: sridhar-intern-2026" \
    -d '{"model":"x","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' | jq -r '.choices[0].message.content // .detail'
  echo "---"
done

# 2. Kill one node
docker compose stop node-2

# 3. Send more requests (gateway should route to node-1 and node-3)
for i in 1 2 3; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "x-api-key: sridhar-intern-2026" \
    -d '{"model":"x","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' | jq -r '.choices[0].message.content // .detail'
  echo "---"
done

# 4. Restore node
docker compose start node-2
```

**Verification:** Requests succeed with 2 nodes; after restore, all 3 work again.

---

# Sprint 6: Demo Script and README (~15 min) — Done

## Step 6.1: Create demo.sh

```bash
#!/bin/bash
# Sridhar-Mesh Demo Script for Red Hat Meetup
set -e
BASE="http://localhost:8080"
KEY="sridhar-intern-2026"

echo "=== 1. Health Check ==="
curl -s $BASE/health | jq .

echo -e "\n=== 2. Inference Request ==="
curl -s -X POST $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: $KEY" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32}' \
  | jq '.choices[0].message.content'

echo -e "\n=== 3. Invalid API Key (expect 403) ==="
curl -s -X POST $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: wrong-key" \
  -d '{"model":"x","messages":[]}' | jq .

echo -e "\n=== 4. Failover: Stopping node-2 ==="
docker compose stop node-2
sleep 2
curl -s -X POST $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: $KEY" \
  -d '{"model":"x","messages":[{"role":"user","content":"Still works?"}],"max_tokens":16}' \
  | jq -r '.choices[0].message.content // .detail'
echo "Restoring node-2..."
docker compose start node-2
```

---

## Step 6.2: Red Hat Talking Points

1. **Containerization:** 3 node containers + gateway + Prometheus + Grafana – all on a bridge network.
2. **Distributed Systems:** Independent nodes; gateway does round-robin + circuit breaker.
3. **Observability:** Prometheus scrapes nodes, gateway, vLLM; Grafana dashboard.
4. **Security:** API key validation at gateway; Cloudflare Tunnel for HTTPS.
5. **Failover:** Kill a node; gateway routes around it; circuit breaker prevents cascading failures.

---

# Quick Reference: Full Startup Sequence

```bash
# Terminal 1: vllm-metal
source ~/.venv-vllm-metal/bin/activate
vllm serve mlx-community/Qwen2.5-0.5B-Instruct --port 8000 --host 0.0.0.0

# Terminal 2: Docker stack
cd /Users/srpillai/CODING/DecentralizedLLM
docker compose up -d

# Terminal 3: Cloudflare (optional)
cloudflared tunnel --url http://localhost:8080
```

**URLs & UI Access:**

| Service | URL | Login / Notes |
|---------|-----|---------------|
| **Gateway** | http://localhost:8080 | API key: `sridhar-intern-2026` |
| **Grafana** | http://localhost:3000 | Username: `admin` / Password: `admin` — Dashboards → Sridhar-Mesh Dashboard |
| **Prometheus** | http://localhost:9090 | No login — Status → Targets to see scrape targets |

Open **Grafana** in your browser: http://localhost:3000 → log in with admin/admin → go to Dashboards → Sridhar-Mesh Dashboard.

---

# Troubleshooting


| Issue                               | Fix                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| `host.docker.internal` unreachable  | On Linux, add `extra_hosts: - "host.docker.internal:172.17.0.1"` or use host IP |
| vLLM model not found                | Try `mlx-community/qwen2.5-0.5b-instruct-q2`                                    |
| Node returns 503                    | Ensure vllm-metal is running on host:8000                                       |
| Gateway 503 "All nodes unavailable" | Check nodes: `docker compose ps`; ensure vLLM is up                             |
| Prometheus can't scrape vLLM        | vLLM must bind to 0.0.0.0; check firewall                                       |
| Grafana "No data"                   | Wait 1–2 min for scrape; check Prometheus targets                               |
| Docker daemon not running           | Start Docker Desktop from Applications                                          |


---

*End of Sprint Guide*