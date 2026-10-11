#!/bin/bash
#SBATCH --job-name=pct_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/pct_%x_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/pct_%x_%j.err

# =============================================================================
# Smoke test: boot the vLLM server for one model, send a single chat
# completion request, and exit 0/1 based on success. No eval pipeline runs.
#
# Do NOT sbatch this directly — submit via `slurm/submit_model.sh <slug>`,
# which runs this first and only submits the full job
# (slurm/run_model_generic.sh) if this one exits 0 (via --dependency=afterok).
#
# Required env vars (same as run_model_generic.sh):
#   MODEL_DIR_NAME, MODEL_SLUG, TENSOR_PARALLEL_SIZE
# Optional: MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, SERVED_MODEL_NAME,
#           IS_GGUF, GGUF_FILE, IS_MOE, EXTRA_VLLM_ARGS
# =============================================================================

set -eo pipefail

: "${MODEL_DIR_NAME:?MODEL_DIR_NAME must be set}"
: "${MODEL_SLUG:?MODEL_SLUG must be set}"
: "${TENSOR_PARALLEL_SIZE:?TENSOR_PARALLEL_SIZE must be set}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"

echo "=========================================="
echo "SMOKE TEST ($MODEL_DIR_NAME)"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo ""

# ------------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/PromptControlText
MODELS_ROOT=/scratch/gpfs/JORDANAT/mg9965/models
MODEL_PATH="$MODELS_ROOT/$MODEL_DIR_NAME"

if [ "${IS_GGUF:-0}" = "1" ]; then
    : "${GGUF_FILE:?GGUF_FILE must be set when IS_GGUF=1}"
    MODEL_PATH="$MODELS_ROOT/$MODEL_DIR_NAME/$GGUF_FILE"
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_DIR_NAME")}"
CONDA_ENV="${CONDA_ENV:-PromptControlText}"
VLLM_PORT=8001   # distinct from the full job's port to avoid collisions if both ever overlap
source "$PROJECT_DIR/slurm/model_runtime.sh"

# ------------------------------------------------------------------
# 1. Environment setup
# ------------------------------------------------------------------
cd "$PROJECT_DIR"

module load anaconda3/2025.6

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f "$HOME/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "$HOME/.conda/envs/$CONDA_ENV/bin/activate"
else
    source activate "$CONDA_ENV"
fi

# Gemma's CUDA 13 stack must not load conflicting packages from ~/.local.
# This is equivalent to the successful `python -s` import diagnostic.
if [[ "$MODEL_SLUG" == gemma4_* ]]; then
    export PYTHONNOUSERSITE=1
    # FlashInfer sampling JIT cannot find curand.h in the cluster toolkit.
    # Use vLLM's native GPU sampler for startup warmup and generation.
    export VLLM_USE_FLASHINFER_SAMPLER=0
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# 2. Cache & offline
# ------------------------------------------------------------------
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export VLLM_USAGE_STATS_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/usage_stats
export TRITON_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/triton
export XDG_CACHE_HOME=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/xdg
export TIKTOKEN_CACHE_DIR=$HOME/.tiktoken_cache

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_DIR" "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------
# 3. GPU / Memory optimization
# ------------------------------------------------------------------
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))
fi
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ------------------------------------------------------------------
# 4. Validate prerequisites
# ------------------------------------------------------------------
if [ -e "$MODEL_PATH" ]; then
    echo "✅ Model found at: $MODEL_PATH"
else
    echo "❌ ERROR: Model not found at: $MODEL_PATH"
    echo "   Run slurm/download_models.sh $MODEL_SLUG on the login node first."
    exit 1
fi

mkdir -p logs

# ------------------------------------------------------------------
# 5. Start vLLM server
# ------------------------------------------------------------------
echo "Starting vLLM server ($SERVED_MODEL_NAME, TP=$TENSOR_PARALLEL_SIZE) for smoke test..."

VLLM_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype auto
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --disable-custom-all-reduce
)
VLLM_ARGS+=("${VLLM_MODEL_ARGS[@]}")
# Match the reduced limits whenever the shared runtime selects eager execution.
if [ "${#VLLM_MODEL_ARGS[@]}" -gt 0 ]; then
    VLLM_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS"
                --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
                --enable-chunked-prefill)
fi

if [ "${IS_GGUF:-0}" = "1" ]; then
    VLLM_ARGS+=(--quantization gguf --tokenizer "$MODELS_ROOT/$MODEL_DIR_NAME")
fi

if [ "${IS_MOE:-0}" = "1" ] && [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    VLLM_ARGS+=(--enable-expert-parallel)
fi

if [ -n "$EXTRA_VLLM_ARGS" ]; then
    # shellcheck disable=SC2206
    VLLM_ARGS+=($EXTRA_VLLM_ARGS)
fi

python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

RESULT=1  # assume failure until proven otherwise

cleanup() {
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    if [ "$RESULT" -eq 0 ]; then
        echo "✅ SMOKE TEST PASSED for $MODEL_SLUG"
    else
        echo "❌ SMOKE TEST FAILED for $MODEL_SLUG"
    fi
    exit "$RESULT"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 6. Wait for server readiness
# ------------------------------------------------------------------
echo "Waiting for vLLM server..."
MAX_WAIT=1200
WAIT_INTERVAL=15
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ ERROR: vLLM server exited unexpectedly during startup"
        exit 1
    fi
    if curl -fsS --max-time 10 "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM server ready after ${ELAPSED}s"
        break
    fi
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ ERROR: vLLM server timeout during smoke test."
    exit 1
fi

# ------------------------------------------------------------------
# 7. Send a single chat completion request
# ------------------------------------------------------------------
echo "Sending test chat completion request..."

RESPONSE=$(curl -s -w "\n%{http_code}" "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${SERVED_MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}],
        \"max_tokens\": 8,
        \"temperature\": 0
    }")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP status: $HTTP_CODE"
echo "Response body: $BODY"

if [ "$HTTP_CODE" = "200" ] && echo "$BODY" | grep -q '"content"'; then
    echo "✅ Received a valid chat completion response."
    RESULT=0
else
    echo "❌ Smoke test request failed or returned an unexpected response."
    RESULT=1
fi
