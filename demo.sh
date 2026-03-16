#!/bin/bash
# Sridhar-Mesh Demo Script for Red Hat Meetup
set -e
BASE="http://localhost:8080"
KEY="sridhar-intern-2026"

# Use the model name you started vLLM with (e.g. mlx-community/Qwen2.5-0.5B-Instruct-8bit)
MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-8bit}"

echo "=== 1. Health Check ==="
curl -s $BASE/health | jq .

echo -e "\n=== 2. Inference Request ==="
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

echo -e "\n=== 4. Failover: Stopping node-2 ==="
docker compose stop node-2
sleep 2
curl -s -X POST $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: $KEY" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Still works?\"}],\"max_tokens\":16}" \
  | jq -r '.choices[0].message.content // .detail'
echo "Restoring node-2..."
docker compose start node-2
