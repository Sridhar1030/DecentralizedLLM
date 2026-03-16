# DecentralizedLLM

**One LLM. Split across 3 devices. No single node holds the full model.**

A proof-of-concept where a single LLM (Qwen2.5-0.5B) is physically split across 3 containers. Each container holds ~1/3 of the model layers. A request flows **node0 → node1 → node2** over HTTP to produce output. True pipeline parallelism on CPU.

---

## Visual Overview

### Request Flow

```
   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
   │   Client    │────▶│   Gateway   │────▶│   Coordinator   │
   │  (curl/API) │     │ API key +   │     │  tokenize →     │
   └─────────────┘     │  circuit    │     │  chain nodes    │
                       │  breaker    │     └────────┬────────┘
                       └─────────────┘              │
                                                    │
         ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
         │                                          │                                          │
         ▼                                          ▼                                          ▼
   ┌───────────┐                              ┌───────────┐                              ┌───────────┐
   │  NODE 0   │  hidden_states (base64)      │  NODE 1   │  hidden_states (base64)      │  NODE 2   │
   │  :8001    │ ──────────────────────────▶  │  :8002    │ ──────────────────────────▶ │  :8003    │
   │           │                              │           │                              │           │
   │ embed +   │                              │ layers    │                              │ layers    │
   │ layers    │                              │ 8–15      │                              │ 16–23 +   │
   │ 0–7       │                              │           │                              │ norm +    │
   └───────────┘                              └───────────┘                              │ lm_head   │
                                                                                        └─────┬─────┘
                                                                                              │
                                                                                              ▼
                                                                                        logits → token
```

### How the Model is Split (24 layers total)

```
  Qwen2.5-0.5B (24 transformer layers)
  ═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  NODE 0 (container 1)          │  NODE 1 (container 2)    │  NODE 2 (container 3)  │
  ├────────────────────────────────┼──────────────────────────┼───────────────────────┤
  │  embed_tokens                   │                          │                       │
  │  layer 0  ████                  │  layer 8   ████          │  layer 16  ████       │
  │  layer 1  ████                  │  layer 9   ████          │  layer 17  ████       │
  │  layer 2  ████                  │  layer 10  ████          │  layer 18  ████       │
  │  layer 3  ████                  │  layer 11  ████          │  layer 19  ████       │
  │  layer 4  ████                  │  layer 12  ████          │  layer 20  ████       │
  │  layer 5  ████                  │  layer 13  ████          │  layer 21  ████       │
  │  layer 6  ████                  │  layer 14  ████          │  layer 22  ████       │
  │  layer 7  ████                  │  layer 15  ████          │  layer 23  ████       │
  │                                │                          │  norm                 │
  │                                │                          │  lm_head (vocab logits)│
  └────────────────────────────────┴──────────────────────────┴───────────────────────┘

  Each node = separate Docker container. No node has the full model.
```

### Per-Token Generation (autoregressive)

For each new token, the coordinator sends the full sequence so far through all 3 nodes:

```
  Step 1:  [token1]           → node0 → node1 → node2 → logits → sample token2
  Step 2:  [token1, token2]   → node0 → node1 → node2 → logits → sample token3
  Step 3:  [token1, token2, token3] → node0 → node1 → node2 → logits → sample token4
  ...
```

---

## Codebase Structure

```
DecentralizedLLM/
├── layer-nodes/                    # Core: custom pipeline parallelism
│   ├── node.py                    # Layer node service (runs 3×)
│   ├── coordinator.py             # Chains nodes, orchestrates inference
│   ├── requirements.txt
│   ├── Dockerfile.node
│   └── Dockerfile.coordinator
├── gateway/
│   ├── app.py                     # API key auth, circuit breaker, proxy
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml             # Orchestrates all services
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
├── demo.sh                         # Quick test script
└── README.md
```

---

## Codebase Walkthrough

### `layer-nodes/node.py` — Layer node

Each node holds a subset of the model. Configured via `NODE_LAYERS` env (e.g. `"0-7"`, `"8-15"`, `"16-23"`).

| Concept | Implementation |
|--------|------------------|
| **Load model** | Load full model, extract only our layers, `del full` to free memory |
| **First node** | `embed_tokens` + layers 0–7; replaces `norm` and `lm_head` with `Identity()` |
| **Middle node** | Layers 8–15 only; `embed_tokens` + `norm` + `lm_head` = `Identity()` |
| **Last node** | Layers 16–23 + `norm` + `lm_head` |
| **API** | `POST /forward` — accepts `input_ids` (first) or `hidden_states_b64` (middle/last) |
| **Serialization** | Hidden states and logits sent as base64-encoded numpy arrays |

```python
# First node: receives token IDs, returns hidden states
if req.input_ids is not None:
    out = model(ids, output_hidden_states=True)
    return {"hidden_states_b64": base64.b64encode(hidden.tobytes()).decode()}

# Middle/last: receives hidden states, returns hidden states or logits
if end_layer == 24:
    return {"logits_b64": base64.b64encode(logits.tobytes()).decode()}
```

---

### `layer-nodes/coordinator.py` — Orchestrator

Chains the three nodes and runs token generation.

| Concept | Implementation |
|--------|------------------|
| **Forward chain** | `forward_chain(ids)` → `node0/forward` → `node1/forward` → `node2/forward` → logits |
| **Generation** | Greedy decode: for each step, get logits from chain, `argmax` → next token |
| **Stop** | Stops when `eos_token_id` or `max_tokens` reached |
| **API** | `POST /v1/chat/completions` — OpenAI-compatible |

```python
async def forward_chain(input_ids):
    r0 = await client.post(NODE0_URL + "/forward", json={"input_ids": input_ids})
    r1 = await client.post(NODE1_URL + "/forward", json={"hidden_states_b64": r0.json()["hidden_states_b64"]})
    r2 = await client.post(NODE2_URL + "/forward", json={"hidden_states_b64": r1.json()["hidden_states_b64"]})
    return np.frombuffer(base64.b64decode(r2.json()["logits_b64"]), dtype=np.float32)
```

---

### `gateway/app.py` — API gateway

| Concept | Implementation |
|--------|------------------|
| **Auth** | Rejects requests without valid `x-api-key` |
| **Circuit breaker** | After 3 failures, blocks for 30s before retrying |
| **Proxy** | Forwards `POST /v1/chat/completions` to coordinator |
| **Health** | Uses `/v1/models` for readiness; `/health` reports circuit state |
| **Metrics** | Prometheus counters and latency histogram at `/metrics` |

---

### `docker-compose.yml` — Service layout

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `node0` | `layer-nodes` | 8001 | Layers 0–7 + embed |
| `node1` | `layer-nodes` | 8002 | Layers 8–15 |
| `node2` | `layer-nodes` | 8003 | Layers 16–23 + norm + lm_head |
| `coordinator` | `layer-nodes` | 8080 (internal) | Chains nodes |
| `gateway` | `gateway` | 8080 | Public API |
| `prometheus` | `prom/prometheus` | 9090 | Metrics |
| `grafana` | `grafana/grafana` | 3000 | Dashboards |

---

## Requirements

- **Docker**
- **Apple Silicon** or **Linux x86**
- **8GB+ RAM** recommended (2G per node)

---

## Quick Start

### 1. Build and start

```bash
cd /Users/srpillai/CODING/DecentralizedLLM
docker compose build
docker compose up -d
```

First run downloads the model (~1GB). Each node loads only its layers.

### 2. Test

```bash
# Health
curl -s http://localhost:8080/health | jq .

# Inference
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: sridhar-intern-2026" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}' \
  | jq '.choices[0].message.content'
```

### 3. Failover demo

Stop a node (one third of the model goes down):

```bash
docker compose stop node1
# Requests will fail—model is incomplete
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: sridhar-intern-2026" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' | jq .

docker compose start node1
```

---

## UI Access

| Service | URL | How to access |
|---------|-----|---------------|
| **Grafana** | http://localhost:3000 | Login: `admin` / `admin` → Dashboards |
| **Prometheus** | http://localhost:9090 | Status → Targets |
| **Gateway** | http://localhost:8080 | API; use `x-api-key: sridhar-intern-2026` |

---

## Configuration

- **Model:** Set `MODEL_NAME` in docker-compose (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- **Resources:** Each node 2 CPU / 2GB (adjust in docker-compose)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Out of memory | Increase node memory in docker-compose |
| Slow inference | CPU inference is slow; each token requires 3 HTTP round-trips |
| Node fails to load | Ensure 2GB+ RAM per node |
