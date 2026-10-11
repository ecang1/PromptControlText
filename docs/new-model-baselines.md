# Run the six missing no-prompt baselines

The six new open-model files each contain 242,640 prompted responses and no
`No-system-prompt` rows. Their baseline files were absent in the local snapshot
checked on 2026-10-11. Existing controls cover only the original four models.

After syncing the code changes to the cluster, run from the repository root:

```bash
bash slurm/submit_new_controls.sh
```

This submits baseline generation and a dependent Llama-3.1-8B policy judge for
each of these models. There are no GPT-4 or provider API grading calls.

| Model slug | Generation GPUs | Judge array index |
|---|---:|---:|
| `gemma4_12b` | 1 | 0 |
| `gemma4_31b` | 2 | 1 |
| `qwen3_6_35b_a3b` | 2 | 2 |
| `nemotron_3_nano_4b` | 1 | 3 |
| `deepseek_r1_distill_qwen_32b` | 2 | 4 |
| `deepseek_r1_distill_llama_70b` | 4 | 5 |

Only **850 responses per model**, or **5,100 total**, are generated:
450 XSTest and 400 HarmBench items, with one control condition. IHEval is excluded
because these jobs target refusal/compliance analysis, which also reduces the
workload by about 75% compared with running all three benchmarks. Every safety
item is retained, so these can be compared with the safety subset of the original
models' larger controls.
The system message is omitted entirely. The existing data loaders, generation
settings, and model configurations are reused so the control matches the
prompted runs.

## Short jobs and checkpointing

These already-used models skip the extra smoke-test startup. Baseline generation
requests **1 hour / 6 CPUs / 64 GB RAM**, using the GPU counts above to preserve
the existing serving setup. Each dependent judge requests **30 minutes / 4 CPUs /
32 GB RAM / 1 GPU**, and processes only its 850 baseline answers. GPU allocation
is released on exit; the time limit is not an estimate or a minimum runtime.
Queue time depends on availability, particularly for the four-GPU 70B model.

Gemma retains its separate environment and FlashInfer sampler workaround.
DeepSeek retains eager execution, synchronous scheduling, and reduced
concurrency. New control servers use job-specific ports so generation and
judging can share nodes without all attempting to bind port 8000. OpenMP thread
counts follow the allocated CPUs.

Both generation and judging resume saved progress. If generation fails or times
out, its dependent judge is canceled; resubmit the affected model to queue both:

```bash
bash slurm/submit_new_controls.sh deepseek_r1_distill_llama_70b
```

To generate only, or explicitly repeat the smoke test:

```bash
bash slurm/submit_model.sh gemma4_12b --control
bash slurm/submit_model.sh gemma4_12b --control --judge --smoke
```

If a job needs a longer limit, override it for that submission:

```bash
CONTROL_TIME_LIMIT=02:00:00 bash slurm/submit_new_controls.sh deepseek_r1_distill_llama_70b
```

The raw output paths are `artifacts/phase1_results_control_<slug>.csv`.
Judgments go into `artifacts/llm_policy_labels/control_<slug>/`.
Prompted files and their judgments retain separate paths.

To retry judging only, use the index from the table (or `0-5` for all six):

```bash
sbatch --array=0 --time=00:30:00 --cpus-per-task=4 \
  --export=ALL,JUDGE_JOBS_CONFIG=configs/llm_policy_jobs_new_controls.yaml,JUDGE_REQUIRE_COMPLETE=1 \
  slurm/run_llm_judge.sh
```

The judge validates that all 850 responses are present, unique, and nonempty
before starting its GPU server. It fails if any final policy label is unresolved.

If compilation reports missing Qwen-3.6 judgments, check its raw CSV first:
an incomplete generation checkpoint must be resumed with
`bash slurm/submit_new_controls.sh qwen3_6_35b_a3b` before judging can start.
For complete generation with unresolved DeepSeek labels, retry only judge
indices `4-5` using the command above. Resume preserves valid labels.
First-pass and adjudicator calls now share schema-constrained JSON output;
sync this fix before retrying older adjudications that repeatedly returned
invalid field values such as `"null"` or `"safety"`. Saved results are not relabeled
by the code update; unresolved rows require fresh judge calls.

## Add completed baselines to the analysis CSV

After all six judges finish, and wherever the saved labeled outputs are present:

```bash
python -m scoring.compile_results --safety-heuristic v2 --require-new-controls
```

This rebuilds `artifacts/phase1_results_combined_labeled.csv`, applying the same
v2 refusal scorer to old and new responses. It requires all six new controls to
have complete judge manifests and valid labels. With the existing original,
new prompted, and frontier main runs, the full table has **2,455,930 rows**.
Each new control has `experiment_group=control`,
`prompt_family=No-system-prompt`, and `source_run=control_<slug>`.

Without `--require-new-controls`, compilation includes completed new controls
and logs a warning for each unavailable or unfinished one. The strict option
helps prevent accidentally analyzing an incomplete set of baselines.

If R is pointed at the separately named v2 CSV, rebuild that same file:

```bash
python -m scoring.compile_results --safety-heuristic v2 --require-new-controls \
  --output artifacts/phase1_results_combined_labeled_heuristic_v2.csv
```

Reload `all_data` and rerun the notebook's cleaning steps after recompilation.
