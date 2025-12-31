./.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --reasoning-parser qwen3 \
  --port 8000 \
  --host 127.0.0.1 \
  --dp 8 \
  --schedule-conservativeness 0.1 