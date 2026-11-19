#!/bin/sh

exec "${VLLM_BIN:-vllm}" serve Qwen/Qwen3-8B \
    --quantization bitsandbytes \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --enforce-eager --generation-config vllm
