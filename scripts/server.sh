#!/bin/bash

export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch_inductor_cache
export SGLANG_FLASHINFER_WORKSPACE_SIZE=4294967296


python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-FP8 \
    --reasoning-parser qwen3 \
    --port 8000 \
    --host 127.0.0.1 \
    --log-level-http error \
    --enable-cache-report \
    --dp-size 8 \
    --schedule-conservativeness 0.1 \
    --random-seed 42 \
    --attention-backend flashinfer \
    --fp8-gemm-backend deep_gemm \
    --moe-runner-backend flashinfer_trtllm \
    --cuda-graph-max-bs 4096 \
    --enable-tokenizer-batch-encode \
    --mem-fraction-static 0.75 \
    --context-length 16384 \
    --max-running-requests 32768 \
    --chunked-prefill-size 65536 \
    --max-prefill-tokens 65536 \
    --schedule-policy lpm \
    --router-log-level error \
    --router-policy cache_aware \
    --router-dp-aware
