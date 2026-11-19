#!/bin/sh
set -eu

revision=1cfa9a7208912126459214e8b04321603b3df60c
exec "${VLLM_BIN:-vllm}" serve Qwen/Qwen3-4B \
    --revision "$revision" --tokenizer-revision "$revision" \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --max-num-seqs 2 \
    --generation-config vllm
