#!/bin/bash
# Shared serving defaults for the smoke test and full generation launcher.
# DeepSeek and Qwen-3.6 control workers reported CUDA illegal memory access.
# Eager, synchronous execution is a mitigation; the logs do not identify the
# faulty kernel. Limit the Qwen change to controls, whose small remaining
# workload does not need the prompted run's throughput settings.
VLLM_MODEL_ARGS=()
if [[ "$MODEL_SLUG" == deepseek_r1_distill_* ]] ||
   [[ "$MODEL_SLUG" == qwen3_6_35b_a3b && "${REGISTRY_VERSION:-v3}" == control ]]; then
    : "${VLLM_MAX_NUM_SEQS:=32}"
    : "${VLLM_MAX_NUM_BATCHED_TOKENS:=4096}"
    : "${GENERATION_MAX_WORKERS:=16}"
    VLLM_MODEL_ARGS+=(--enforce-eager --no-async-scheduling)
else
    : "${VLLM_MAX_NUM_SEQS:=512}"
    : "${VLLM_MAX_NUM_BATCHED_TOKENS:=32768}"
    : "${GENERATION_MAX_WORKERS:=64}"
fi
