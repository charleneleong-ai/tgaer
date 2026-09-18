#!/bin/bash
# Launch vLLM for ARC-AGI-3 M1 codegen on a shared, multi-tenant A100.
#
# Uses ~/vllm_venv, NOT base python3: ~/.local is shared with other tenants and
# pins transformers 4.40.1, which vllm 0.26.0 cannot import
# (ALLOWED_LAYER_TYPES). Upgrading it would risk another tenant's job.
#
# PATH must carry the venv's bin: vLLM shells out to `ninja` to JIT the MoE
# kernels, and invoking the interpreter by absolute path does not put it there.
#
# Model: Qwen3.8-27B dense, bf16, ~54GB of weights. The 35B-A3B MoE is also
# cached but needs ~70GB and left no room for KV cache at a utilisation this
# card can spare; the FP8 27B that would have been the best match for the Kaggle
# kernel is only a 12K stub in the cache, never actually downloaded.
#
# Port 8011 (not 8000) and gpu-memory-utilization 0.88 leave room for the other
# tenant already on the card. max-num-seqs 16: codegen calls once per game, so
# concurrency buys nothing and the spare memory goes to KV cache instead.
export PATH="$HOME/vllm_venv/bin:$PATH"
LOG=$HOME/vllm_8011_$(date -u +%Y%m%dT%H%M%SZ).log
setsid nohup "$HOME/vllm_venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.8-27B \
  --served-model-name qwen-27b \
  --port 8011 \
  --max-num-seqs 16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  </dev/null >>"$LOG" 2>&1 &
disown
echo "LOG=$LOG"
