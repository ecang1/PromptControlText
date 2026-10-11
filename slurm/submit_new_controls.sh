#!/bin/bash
# Submit the six missing no-prompt baselines and their dependent Llama judges.
# Usage: bash slurm/submit_new_controls.sh [slug ...]
# Each model queues baseline generation -> baseline-only judge; no extra smoke.
# Only the 850 XSTest/HarmBench items used in refusal/compliance analysis run.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLUGS=(
    gemma4_12b
    gemma4_31b
    qwen3_6_35b_a3b
    nemotron_3_nano_4b
    deepseek_r1_distill_qwen_32b
    deepseek_r1_distill_llama_70b
)
if [ "$#" -gt 0 ]; then
    SLUGS=("$@")
fi
# Validate the whole request before submitting any jobs.
for slug in "${SLUGS[@]}"; do
    case "$slug" in
        gemma4_12b|gemma4_31b|qwen3_6_35b_a3b|nemotron_3_nano_4b|deepseek_r1_distill_qwen_32b|deepseek_r1_distill_llama_70b) ;;
        *) echo "ERROR: No new control job is configured for $slug"; exit 1 ;;
    esac
done
for slug in "${SLUGS[@]}"; do
    bash "$SCRIPT_DIR/submit_model.sh" "$slug" --control --judge
done
