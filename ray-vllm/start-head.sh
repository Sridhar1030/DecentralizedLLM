#!/bin/bash
set -e

echo "Starting Ray head..."
ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0

echo "Waiting for workers to join (30s)..."
sleep 30

# Pipeline parallelism: model split across 2 stages (PP0, PP1).
# On CPU vLLM uses multiprocessing (stages on head); Ray workers used when GPU available.
# VLLM_CPU_OMP_THREADS_BIND=nobind bypasses NUMA detection (Docker reports no NUMA nodes).
echo "Starting vLLM with pipeline parallelism (2 stages)..."
exec vllm serve "${VLLM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --max-model-len 4096
