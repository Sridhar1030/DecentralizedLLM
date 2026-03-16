# Sridhar-Mesh Quick Start

## 1. Start vLLM with a better English model

Use **8-bit** or **bf16** for cleaner English output (avoid q2—it produces garbled text):

```bash
source ~/.venv-vllm-metal/bin/activate

# Option A: Qwen 8-bit (good balance)
vllm serve mlx-community/Qwen2.5-0.5B-Instruct-8bit --port 8000 --host 0.0.0.0

# Option B: Llama 3.2 1B 4-bit (stronger English)
vllm serve mlx-community/Llama-3.2-1B-Instruct-4bit --port 8000 --host 0.0.0.0
```

## 2. Start the stack

```bash
cd /Users/srpillai/CODING/DecentralizedLLM
docker compose up -d
```

## 3. Test via Gateway (with API key)

```bash
# Use the SAME model name you started vLLM with
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: sridhar-intern-2026" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 32
  }' | jq '.choices[0].message.content'
```

For Llama: use `"model": "mlx-community/Llama-3.2-1B-Instruct-4bit"`

## 4. Check what model vLLM loaded

```bash
curl -s http://localhost:8000/v1/models | jq '.data[].id'
```

Use that ID in your requests.
