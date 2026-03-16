# Extend vLLM CPU image (ARM64 native on Apple Silicon)
FROM vllm/vllm-openai-cpu:latest

# Install Ray for distributed pipeline parallelism
RUN pip install --no-cache-dir "ray[default]>=2.9.0"

# Copy startup scripts
COPY start-head.sh /usr/local/bin/
COPY start-worker.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-head.sh /usr/local/bin/start-worker.sh

# Default model
ENV VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
