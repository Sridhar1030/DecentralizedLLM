#!/bin/bash
# DecentralizedLLM Demo – Pipeline Parallelism
set -e
BASE="http://localhost:8080"
KEY="sridhar-intern-2026"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "=== 1. Health Check ==="
curl -s $BASE/health | jq .

echo -e "\n=== 2. Inference Request (model split across 3 devices) ==="
curl -s -X POST $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: $KEY" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":32}" \
  | jq '.choices[0].message.content'

echo -e "\n=== 3. Invalid API Key (expect 403) ==="
curl -s -X POST $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: wrong-key" \
  -d '{"model":"x","messages":[]}' | jq .

echo -e "\n=== 4. Failover Demo: Stopping node1 (1/3 of model goes down) ==="
docker compose stop node1
sleep 3
echo "Request with incomplete model (expect failure):"
curl -s -X POST $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: $KEY" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":16}" \
  | jq -r '.choices[0].message.content // .detail // .'
echo "Restoring node1..."
docker compose start node1
