#!/bin/bash
#SBATCH --job-name=pct_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/pct_%x_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/pct_%x_%j.err

# =============================================================================
# Generic Prompt-Level Safety Controllers Pipeline runner.
#
# Do NOT sbatch this file directly (its #SBATCH --gres/--job-name defaults
# are placeholders). Submit it via `slurm/submit_model.sh <slug>`, which
# fills in the correct GPU count, tensor-parallel size, and model path for
# each of the new models (Gemma 4, Qwen3.6-35B-A3B, DeepSeek-R1 distills,
# NVIDIA Nemotron), and any future model added to configs/model_registry.yaml.
#
# Required environment variables (exported by submit_model.sh via --export):
#   MODEL_DIR_NAME          local dir name under $MODELS_ROOT (e.g.
#                           google--gemma-4-12b-it)
#   MODEL_SLUG              short slug used in output filenames
#   TENSOR_PARALLEL_SIZE    vLLM -tp value
#   MAX_MODEL_LEN           context length to serve
#   GPU_MEMORY_UTILIZATION  vLLM --gpu-memory-utilization
# Optional:
#   SERVED_MODEL_NAME       defaults to basename(MODEL_PATH)
#   IS_GGUF=1               serve a single .gguf file with --quantization gguf
#   GGUF_FILE               filename of the .gguf to load (required if IS_GGUF=1)
#   IS_MOE=1                add --enable-expert-parallel (for A3B MoE models)
#   EXTRA_VLLM_ARGS         extra raw args appended to the vllm command
#   HF_GATED=1              just informational; no download at serve-time
#   GENERATION_MAX_WORKERS  client concurrency (DeepSeek: 16; others: 64)
#   VLLM_MAX_NUM_SEQS       server concurrency (DeepSeek: 32; others: 512)
#   VLLM_MAX_NUM_BATCHED_TOKENS  prefill budget (DeepSeek: 4096; others: 32768)
#   REGISTRY_VERSION       v3 (default) or control (one no-system condition)
#   OUTPUT_FILE            defaults to a separate file for each registry
#   SAFETY_ONLY=1          generate only XSTest/HarmBench (850 control items)
# Existing generation checkpoints are resumed; completed responses are kept.
# =============================================================================

set -eo pipefail

: "${MODEL_DIR_NAME:?MODEL_DIR_NAME must be set}"
: "${MODEL_SLUG:?MODEL_SLUG must be set}"
: "${TENSOR_PARALLEL_SIZE:?TENSOR_PARALLEL_SIZE must be set}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${REGISTRY_VERSION:=v3}"
case "$REGISTRY_VERSION" in
    v3) DEFAULT_OUTPUT="artifacts/phase1_results_${MODEL_SLUG}.csv" ;;
    control) DEFAULT_OUTPUT="artifacts/phase1_results_control_${MODEL_SLUG}.csv" ;;
    *) echo "ERROR: REGISTRY_VERSION must be v3 or control"; exit 1 ;;
esac
OUTPUT_FILE="${OUTPUT_FILE:-$DEFAULT_OUTPUT}"

echo "=========================================="
echo "Prompt Control Pipeline ($MODEL_DIR_NAME)"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
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
if [ "$REGISTRY_VERSION" = control ]; then
    # Several small jobs can share a node, so each needs its own server port.
    VLLM_PORT="${VLLM_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}"
else
    VLLM_PORT="${VLLM_PORT:-8000}"
fi
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
export HF_DATASETS_DISK_DIR=$HF_DATASETS_CACHE
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
    # Build "0,1,...,N-1" for the requested tensor-parallel size
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TENSOR_PARALLEL_SIZE - 1)))
fi
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
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
    echo "   Run slurm/download_models.sh $MODEL_SLUG (or the matching slug) on the login node first."
    exit 1
fi

mkdir -p logs artifacts

# ------------------------------------------------------------------
# 5. Validate datasets
# ------------------------------------------------------------------
DATASETS_DIR="$PROJECT_DIR/benchmarks/artifacts/datasets"
for f in harmbench_behaviors.csv iheval.csv xstest_prompts.csv; do
    if [ ! -f "$DATASETS_DIR/$f" ]; then
        echo "❌ ERROR: Expected dataset not found: $DATASETS_DIR/$f"
        exit 1
    fi
done
echo "✅ All datasets found in $DATASETS_DIR"

# ------------------------------------------------------------------
# 6. Start vLLM server
# ------------------------------------------------------------------
echo "Starting vLLM server ($SERVED_MODEL_NAME, TP=$TENSOR_PARALLEL_SIZE)..."

VLLM_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype auto
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --enable-chunked-prefill
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --disable-custom-all-reduce
)
VLLM_ARGS+=("${VLLM_MODEL_ARGS[@]}")

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

cleanup() {
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 7. Wait for server readiness
# ------------------------------------------------------------------
echo "Waiting for vLLM server..."
MAX_WAIT=900
WAIT_INTERVAL=15
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ ERROR: vLLM server exited unexpectedly"
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
    echo "❌ ERROR: vLLM server timeout."
    exit 1
fi

# ------------------------------------------------------------------
# 8. Run Evaluation Pipeline
# ------------------------------------------------------------------
echo "=========================================="
echo "Phase 1: XSTest + HarmBench + IHEval ($REGISTRY_VERSION, resume)"
echo "Output: $OUTPUT_FILE"
echo "=========================================="
GENERATION_EXTRA_ARGS=()
if [ "${SAFETY_ONLY:-0}" = 1 ]; then
    GENERATION_EXTRA_ARGS+=(--safety-only)
fi
python -m experiments.run_phase1 \
    --generator-model    "$SERVED_MODEL_NAME" \
    --base-url           "http://localhost:${VLLM_PORT}/v1" \
    --output-file        "$OUTPUT_FILE" \
    --data-dir           "$PROJECT_DIR/benchmarks/artifacts/datasets" \
    --registry-version   "$REGISTRY_VERSION" \
    --max-workers        "$GENERATION_MAX_WORKERS" \
    --resume "${GENERATION_EXTRA_ARGS[@]}"

echo "✅ Job completed at $(date)"
