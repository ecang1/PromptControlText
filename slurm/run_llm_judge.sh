#!/bin/bash
#SBATCH --job-name=llm_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --array=0-5
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/llm_judge_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/llm_judge_%A_%a.err

# =============================================================================
# LLM Policy Judge — output-only behavioral classifier (PRD §19 / §20)
#
# Runs as a 6-task SLURM array (--array=0-5).  Each task starts its own
# vLLM server and processes one phase1 CSV through the 3-pass judge pipeline.
#
# Judge model: Llama-3.1-8B-Instruct
#   - already on cluster, no startup issues, strong JSON instruction following
#   - NOT gpt-oss-20b (openai_harmony startup failures + circularity risk)
#
# Each complete CSV has 242,640 responses, with three judge calls per response
# plus adjudication. Allow a full day and resume if a task times out.
# All tasks use --resume so they can be safely resubmitted if they time out.
#
# Submit complete models first: sbatch --array=0-3 slurm/run_llm_judge.sh
# After finishing DeepSeek generation: sbatch --array=4-5 slurm/run_llm_judge.sh
# Submit all six complete datasets: sbatch slurm/run_llm_judge.sh
# Resubmit (resume): sbatch slurm/run_llm_judge.sh   (--resume skips done rows)
# Indices: 0=gemma4_12b, 1=gemma4_31b, 2=qwen3_6_35b_a3b,
#          3=nemotron_3_nano_4b, 4=deepseek_qwen_32b, 5=deepseek_llama_70b.
# Inputs and output paths come from configs/llm_policy_jobs_new_models.yaml.
# New no-prompt controls use the same indices in a separate registry:
#   sbatch --export=ALL,JUDGE_JOBS_CONFIG=configs/llm_policy_jobs_new_controls.yaml,JUDGE_REQUIRE_COMPLETE=1 slurm/run_llm_judge.sh
# Or queue generation and dependent judging: bash slurm/submit_new_controls.sh
# Partial inputs fail validation before starting the vLLM server.
# =============================================================================

set -eo pipefail

# ------------------------------------------------------------------
# 0. Array task
# ------------------------------------------------------------------
TASK=${SLURM_ARRAY_TASK_ID:-0}
if ! [[ "$TASK" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be a nonnegative integer."
    exit 1
fi

# Each array task gets its own port so tasks can run on the same node if needed
VLLM_PORT=$((${JUDGE_PORT_BASE:-8000} + TASK))
JUDGE_JOBS_CONFIG=${JUDGE_JOBS_CONFIG:-configs/llm_policy_jobs_new_models.yaml}
if [[ "$JUDGE_JOBS_CONFIG" == *llm_policy_jobs_new_controls.yaml ]] && [ -z "${JUDGE_PORT_BASE:-}" ]; then
    VLLM_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 30000))
fi

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/PromptControlText
MODEL_DIR_NAME=meta-llama--Llama-3.1-8B-Instruct
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/$MODEL_DIR_NAME
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")
CONDA_ENV=PromptControlText
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

# Judge call settings
BATCH_SIZE=512
MAX_WORKERS=${JUDGE_MAX_WORKERS:-256}
TEMPERATURE=0.0
MAX_TOKENS=250

# ------------------------------------------------------------------
# 2. Environment setup
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

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Use the same job registry and completion check as the multi-job Python runner.
JOB_CONFIG=$(python - "$TASK" "$JUDGE_JOBS_CONFIG" <<'PY'
import sys
from pathlib import Path
from scoring.llm_policy_run_jobs import load_jobs, validate_job_input

jobs = load_jobs(sys.argv[2])
task = int(sys.argv[1])
if not 0 <= task < len(jobs):
    raise ValueError(f"Task {task} is outside the {len(jobs)} configured jobs")
job = jobs[task]
rows = validate_job_input(job)
print("\t".join([job["job_id"], str(Path(job["input"]).resolve()),
                 str(Path(job["output_dir"]).resolve()), str(rows or 'validated')]))
PY
)
IFS=$'\t' read -r JOB_ID INPUT_CSV OUTPUT_DIR ROW_COUNT <<< "$JOB_CONFIG"

echo "=========================================="
echo " LLM Policy Judge — task ${TASK} / ${JOB_ID}"
echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $TASK"
echo "Node:        $SLURMD_NODENAME"
echo "Time:        $(date)"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Input:       $INPUT_CSV ($ROW_COUNT responses)"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# ------------------------------------------------------------------
# 3. Cache & offline settings
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
# 4. GPU / Memory optimizations
# ------------------------------------------------------------------
# Preserve the GPU assigned by SLURM when array tasks share a node.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ------------------------------------------------------------------
# 5. Validate prerequisites
# ------------------------------------------------------------------
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Judge model found: $MODEL_PATH"
else
    echo "❌ ERROR: Judge model not found at: $MODEL_PATH"
    exit 1
fi

if [ -f "$INPUT_CSV" ]; then
    echo "✅ Complete input CSV found: $INPUT_CSV"
else
    echo "❌ ERROR: Input CSV not found: $PROJECT_DIR/$INPUT_CSV"
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR"

# ------------------------------------------------------------------
# 6. Start vLLM server
# ------------------------------------------------------------------
echo ""
echo "Starting vLLM server ($SERVED_MODEL_NAME, TP=$TENSOR_PARALLEL_SIZE, port=$VLLM_PORT)..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs 512 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --disable-custom-all-reduce \
    &

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
echo "Waiting for vLLM server on port $VLLM_PORT..."
MAX_WAIT=600
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
    echo "❌ ERROR: vLLM server failed to start within ${MAX_WAIT}s"
    exit 1
fi

# ------------------------------------------------------------------
# 8. Run LLM Policy Judge
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo " LLM Judge: ${JOB_ID}"
echo " Input:     ${INPUT_CSV}"
echo " Output:    ${OUTPUT_DIR}"
echo " Model:     ${SERVED_MODEL_NAME}"
echo "=========================================="

JUDGE_EXTRA_ARGS=()
if [ "${JUDGE_REQUIRE_COMPLETE:-0}" = "1" ]; then
    JUDGE_EXTRA_ARGS+=(--require-complete)
fi
python -m scoring.llm_policy_runner \
    --input          "$INPUT_CSV" \
    --job-id         "$JOB_ID" \
    --output-dir     "$OUTPUT_DIR" \
    --model          "$SERVED_MODEL_NAME" \
    --base-url       "http://localhost:${VLLM_PORT}/v1" \
    --prompt-a       "$PROJECT_DIR/scoring/llm_policy_judge_prompt_A_v1.txt" \
    --prompt-b       "$PROJECT_DIR/scoring/llm_policy_judge_prompt_B_v1.txt" \
    --prompt-c       "$PROJECT_DIR/scoring/llm_policy_judge_prompt_C_v1.txt" \
    --adjudicator-prompt "$PROJECT_DIR/scoring/llm_policy_adjudicator_prompt_v1.txt" \
    --batch-size     "$BATCH_SIZE" \
    --temperature    "$TEMPERATURE" \
    --max-tokens     "$MAX_TOKENS" \
    --max-workers    "$MAX_WORKERS" \
    --resume "${JUDGE_EXTRA_ARGS[@]}"

echo ""
echo "✅ Task ${TASK} (${JOB_ID}) completed at $(date)"
