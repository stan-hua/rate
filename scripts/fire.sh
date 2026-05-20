#### TODO: change the init environment
source /mnt/nfs/users/zwcolin/pillar_vlm_parent_vbc/lib/LLaMA-Factory/shell_scripts/local_env.sh

# SGLANG_ENABLE_JIT_DEEPGEMM=false disables deep_gemm; on driver 575.57.08 with sglang 0.5.6
# freshly JIT-compiled FP8 GEMM cubins segfault at cuModuleLoad. Falls back to stable
# flashinfer/cutlass FP8 paths. --moe-runner-backend triton also avoids deep_gemm for MoE.
SGLANG_ENABLE_JIT_DEEPGEMM=false python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --reasoning-parser qwen3 \
  --port 8000 \
  --host 127.0.0.1 \
  --dp 8 \
  --schedule-conservativeness 0.1 \
  --moe-runner-backend triton
