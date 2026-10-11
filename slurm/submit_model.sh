#!/bin/bash
# =============================================================================
# Submit a smoke test + full vLLM serving/eval job for one of the new models.
#
# The smoke test job boots vLLM and sends a single chat completion request.
# The full job (run_model_generic.sh) is submitted immediately afterwards
# with --dependency=afterok:<smoke_job_id>, so Slurm will only start it once
# the smoke test has exited successfully (and will never start it — Slurm
# auto-cancels dependents — if the smoke test fails).
#
# Usage:
#   bash slurm/submit_model.sh <slug>                 # prompted run
#   bash slurm/submit_model.sh <slug> --control       # no-prompt baseline only
#   bash slurm/submit_model.sh <slug> --control --judge  # also queue its judge
# Controls run XSTest/HarmBench only (850 items) for refusal/compliance analysis,
# reuse the validated model setup, and skip the extra smoke-test boot.
# Add --smoke to a control submission to request a fresh smoke test first.
#
# Slugs (must match configs/model_registry.yaml):
#   gemma4_12b
#   gemma4_31b
#   qwen3_6_35b_a3b
#   deepseek_r1_distill_qwen_32b
#   deepseek_r1_distill_llama_70b
#   nemotron_3_5_lightning_30b_a3b
#   nemotron_3_nano_4b          (native safetensors — use this one)
#   nemotron_3_nano_4b_gguf     (BLOCKED: transformers/vLLM GGUF loader
#                                doesn't support the nemotron_h architecture
#                                yet — will always fail at startup)
# =============================================================================

set -eo pipefail

SLUG="${1:-}"
if [ -z "$SLUG" ]; then
    echo "Usage: $0 <slug> [--control] [--judge] [--smoke]"
    exit 1
fi
shift
REGISTRY_VERSION=v3
SUBMIT_JUDGE=0
CONTROL_SMOKE=0
for option in "$@"; do
    case "$option" in
        --control) REGISTRY_VERSION=control ;;
        --judge) SUBMIT_JUDGE=1 ;;
        --smoke) CONTROL_SMOKE=1 ;;
        *) echo "ERROR: Unknown option: $option"; exit 1 ;;
    esac
done
if [ "$SUBMIT_JUDGE" = 1 ] && [ "$REGISTRY_VERSION" != control ]; then
    echo "ERROR: --judge currently requires --control. Prompted judges use slurm/run_llm_judge.sh."
    exit 1
fi

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/PromptControlText}
SMOKE_SCRIPT="$PROJECT_DIR/slurm/run_smoke_test.sh"
GENERIC_SCRIPT="$PROJECT_DIR/slurm/run_model_generic.sh"

# Gemma 4's "gemma4_unified" architecture requires transformers>=5.0, but
# vllm==0.19.0 (used in the main "PromptControlText" env) breaks on
# transformers 5.x's stricter per-layer config access for ALL models
# (AmbiguousGlobalPerLayerAttributeError). Rather than re-breaking the
# already-working Qwen/DeepSeek/Nemotron jobs, Gemma 4 runs in a separate
# conda env with a newer vLLM release that supports transformers 5.x.
# One-time setup on the cluster:
#   conda create -n PromptControlText-gemma4 --clone PromptControlText
#   conda activate PromptControlText-gemma4
#   pip install --upgrade "transformers>=5.0" "vllm>=0.20"   # pick a version confirmed to support gemma4_unified
GEMMA4_CONDA_ENV="PromptControlText-gemma4"

# slug -> "MODEL_DIR_NAME|GPUS|TP|MAX_LEN|GPU_MEM_UTIL|IS_MOE|IS_GGUF|GGUF_FILE|CONDA_ENV"
CONDA_ENV_OVERRIDE=""   # default: use run_model_generic.sh's default env (PromptControlText)
case "$SLUG" in
  gemma4_12b)
    MODEL_DIR_NAME="google--gemma-4-12b-it"; GPUS=1; TP=1; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE=""; CONDA_ENV_OVERRIDE="$GEMMA4_CONDA_ENV" ;;
  gemma4_31b)
    MODEL_DIR_NAME="google--gemma-4-31b-it"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE=""; CONDA_ENV_OVERRIDE="$GEMMA4_CONDA_ENV" ;;
  qwen3_6_35b_a3b)
    MODEL_DIR_NAME="Qwen--Qwen3.6-35B-A3B-FP8"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=1; GGUF=0; GGUF_FILE="" ;;
  deepseek_r1_distill_qwen_32b)
    MODEL_DIR_NAME="deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE="" ;;
  deepseek_r1_distill_llama_70b)
    MODEL_DIR_NAME="deepseek-ai--DeepSeek-R1-Distill-Llama-70B"; GPUS=4; TP=4; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE="" ;;
  nemotron_3_5_lightning_30b_a3b)
    MODEL_DIR_NAME="nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=1; GGUF=0; GGUF_FILE="" ;;
  nemotron_3_nano_4b)
    MODEL_DIR_NAME="nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16"; GPUS=1; TP=1; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE="" ;;
  nemotron_3_nano_4b_gguf)
    echo "❌ nemotron_3_nano_4b_gguf is currently unsupported: transformers'/vLLM's"
    echo "   GGUF loader does not support the 'nemotron_h' architecture yet"
    echo "   (ValueError: GGUF model with architecture nemotron_h is not supported yet.)."
    echo "   Use 'nemotron_3_nano_4b' instead (native safetensors checkpoint)."
    exit 1 ;;
  *)
    echo "❌ Unknown slug: $SLUG"
    echo "   See configs/model_registry.yaml for valid slugs."
    exit 1 ;;
esac

# Indices match configs/llm_policy_jobs_new_controls.yaml, not GPU counts.
case "$SLUG" in
    gemma4_12b) JUDGE_TASK=0 ;;
    gemma4_31b) JUDGE_TASK=1 ;;
    qwen3_6_35b_a3b) JUDGE_TASK=2 ;;
    nemotron_3_nano_4b) JUDGE_TASK=3 ;;
    deepseek_r1_distill_qwen_32b) JUDGE_TASK=4 ;;
    deepseek_r1_distill_llama_70b) JUDGE_TASK=5 ;;
    *)
        if [ "$SUBMIT_JUDGE" = 1 ]; then
            echo "ERROR: No control judge is configured for $SLUG."
            exit 1
        fi ;;
esac

mkdir -p "$PROJECT_DIR/logs"

EXPORT_VARS="ALL,MODEL_DIR_NAME=$MODEL_DIR_NAME,MODEL_SLUG=$SLUG,TENSOR_PARALLEL_SIZE=$TP,MAX_MODEL_LEN=$MAX_LEN,GPU_MEMORY_UTILIZATION=$MEM,IS_MOE=$MOE,IS_GGUF=$GGUF,GGUF_FILE=$GGUF_FILE"
JOB_PREFIX=pct
RUN_ARGS=()
if [ "$REGISTRY_VERSION" = control ]; then
    JOB_PREFIX=pct_ctrl
    OUTPUT_FILE="artifacts/phase1_results_control_${SLUG}.csv"
    SAFETY_ONLY=1
    RUN_ARGS+=(--time="${CONTROL_TIME_LIMIT:-01:00:00}" --cpus-per-task=6)
else
    SAFETY_ONLY=0
    OUTPUT_FILE="artifacts/phase1_results_${SLUG}.csv"
fi
EXPORT_VARS="$EXPORT_VARS,REGISTRY_VERSION=$REGISTRY_VERSION,OUTPUT_FILE=$OUTPUT_FILE,SAFETY_ONLY=$SAFETY_ONLY"
if [ -n "$CONDA_ENV_OVERRIDE" ]; then
    EXPORT_VARS="$EXPORT_VARS,CONDA_ENV=$CONDA_ENV_OVERRIDE"
    echo "  (using conda env: $CONDA_ENV_OVERRIDE)"
fi

GENERATION_DEPENDENCY=()
if [ "$REGISTRY_VERSION" != control ] || [ "$CONTROL_SMOKE" = 1 ]; then
    echo "Submitting smoke test for $SLUG (GPUs=$GPUS, TP=$TP)..."
    SMOKE_JOB_ID=$(sbatch --parsable \
        --job-name="${JOB_PREFIX}_smoke_${SLUG}" \
        --gres="gpu:${GPUS}" \
        --export="$EXPORT_VARS" \
        "$SMOKE_SCRIPT")
    echo "  -> smoke test job id: $SMOKE_JOB_ID"
    GENERATION_DEPENDENCY+=(--dependency="afterok:${SMOKE_JOB_ID}")
else
    echo "Skipping repeated smoke test for the existing $SLUG setup."
fi

echo "Submitting $REGISTRY_VERSION job for $SLUG (GPUs=$GPUS, TP=$TP)..."
FULL_JOB_ID=$(sbatch --parsable \
    --job-name="${JOB_PREFIX}_${SLUG}" \
    --gres="gpu:${GPUS}" \
    "${GENERATION_DEPENDENCY[@]}" \
    --kill-on-invalid-dep=yes \
    --export="$EXPORT_VARS" \
    "${RUN_ARGS[@]}" \
    "$GENERIC_SCRIPT")
echo "  -> generation job id: $FULL_JOB_ID; output: $OUTPUT_FILE"

if [ "$SUBMIT_JUDGE" = 1 ]; then
    JUDGE_JOB_ID=$(sbatch --parsable \
        --job-name="pct_ctrl_judge_${SLUG}" \
        --array="$JUDGE_TASK" \
        --dependency="afterok:${FULL_JOB_ID}" \
        --kill-on-invalid-dep=yes \
        --time="${CONTROL_JUDGE_TIME_LIMIT:-00:30:00}" \
        --cpus-per-task=4 \
        --export="ALL,JUDGE_JOBS_CONFIG=configs/llm_policy_jobs_new_controls.yaml,JUDGE_REQUIRE_COMPLETE=1" \
        "$PROJECT_DIR/slurm/run_llm_judge.sh")
    echo "  -> control judge job id: $JUDGE_JOB_ID (after successful generation $FULL_JOB_ID)"
fi
