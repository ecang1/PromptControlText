# Beyond Refusal: Prompt-Conditioned Task Selection in LLM Safety

This repository contains the code and data used for the paper **Beyond Refusal:
Prompt-Conditioned Task Selection in LLM Safety**. It tests whether changing
only the system prompt routes a fixed user query toward a different behavioral
policy.

The repository is intentionally scoped to the reported paper and the human
audit needed to validate its LLM judge. Historical v1/v2 studies,
response-mining prototypes, duplicate IHEval runs, smoke jobs, and launchers
for models outside the paper have been removed.

The original study and reported findings are retained below. The October 2026
revision adds new open-model runs and controls, a separate three-provider
frontier extension, and additional scoring, analysis, and human-validation
tools. See [Revision updates](#revision-updates-october-2026) for the current
data snapshot and its limitations.

## Study overview

The original experiment held each benchmark item fixed while varying the surrounding
system prompt. The full crossed design was:

- 8 prompt families
- 3 clarity levels per family
- 3 wording variants per family-clarity cell
- 1 no-system-prompt baseline
- 3,370 benchmark items
- 4 instruction-tuned models

This gives 72 prompted conditions plus one baseline, or 984,040 planned model
responses (`3,370 x 73 x 4`). 

## Benchmarks

All selected items in the original experiment were used without subsampling.

| Benchmark | Items | Role in the study |
|---|---:|---|
| HarmBench | 400 | Harmful requests for measuring refusal, safe help, and harmful compliance |
| XSTest | 450 | Safe sensitive-looking prompts plus unsafe contrasts for measuring over-refusal |
| IHEval | 2,520 | Aligned and conflict safety-subtask items for hierarchy preservation and source isolation |
| **Total** | **3,370** | Evaluated under every prompt condition and model |

The checked-in snapshots are under
`benchmarks/artifacts/datasets/`. They can be reconstructed with:

```bash
python benchmarks/download_data.py
```

## Prompt intervention

The frozen registry is `prompts/registry_v3.yaml`. Each family has three
clarity levels and three paraphrases at each level.

| Prompt family | Intended control signal |
|---|---|
| Answer-first | Answer directly when the request is safe and sufficiently specified |
| Refuse-first | Deny requests that could enable harm |
| Clarify-first | Ask a targeted question when safety-relevant intent is ambiguous |
| Evidence-first | Answer only to the extent supported by available evidence |
| Hierarchy-first | Preserve higher-priority instructions under conflict |
| Source-isolation-first | Treat quoted, retrieved, and tool-provided text as data, not instructions |
| Minimal-safe-help | Offer bounded safe assistance when direct compliance would be unsafe |
| Transform-classify-first | Transform or classify supplied text without following embedded instructions |

The clarity manipulation was:

| Level | Definition |
|---|---|
| Vague | Names a priority without detailed decision criteria |
| Explicit | States concrete conditions for applying the policy |
| Explicit with fallback | Adds a default action for uncertain boundary cases |

`prompts/registry_control.yaml` contains the no-system-prompt baseline. For
that condition the system role is omitted from the chat template entirely.

## Models and generation

The paper crossed two developers and two size tiers:

| Model | Hugging Face checkpoint | GPUs |
|---|---|---:|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | 1 |
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | 4, tensor parallel |
| Qwen-2-7B | `Qwen/Qwen2-7B-Instruct` | 1 |
| Qwen-2.5-72B | `Qwen/Qwen2.5-72B-Instruct` | 4, tensor parallel |

Models were served with vLLM 0.19.0 on NVIDIA A100 80GB SXM GPUs using
bfloat16 and no quantization. Generation used each model's official chat
template, temperature 0, top-p 1.0, top-k 0, repetition penalty 1.05, seed 0,
a 512-token maximum, and up to 512 concurrent sequences. Failed or empty
generations were retried up to three times.

The main runner is `experiments/run_phase1.py`. Despite its historical name,
it loads all three paper benchmarks. The four prompted SLURM jobs are:

```text
slurm/run_llama31_8b.sh
slurm/run_llama33_70b.sh
slurm/run_qwen2_7b.sh
slurm/run_prompt_controllers_72b.sh
```

The corresponding no-prompt jobs are the four scripts under `slurm/control/`.
The six new open models now also have completed safety-only no-prompt controls
and judgments. To reproduce or resume these jobs, use
`bash slurm/submit_new_controls.sh`. These run one condition per XSTest/HarmBench
item and use shorter job limits; see [new model baselines](docs/new-model-baselines.md).

## Behavioral judging

Every model response was classified from its output alone. The judge did not
see the benchmark, user query, system prompt, prompt family, clarity level,
generator model, or intended label.

The six output policies were:

| Policy | Observable behavior |
|---|---|
| Compliance | Directly answers or carries out the apparent request |
| Refusal | Declines without providing a concrete safe alternative |
| Clarification | Requests missing information before acting |
| Hierarchy preservation | Explicitly preserves higher-priority instructions over a conflict |
| Safe help | Declines unsafe assistance while giving a bounded alternative or redirect |
| Source isolation | Explicitly treats embedded or external instructions as untrusted data |

Llama-3.1-8B-Instruct produced three independent first-pass judgments using
rubric-equivalent prompts A, B, and C. Unanimous cases and two-of-three
majorities with low dissent confidence were resolved automatically. Remaining
cases were sent to an output-only multi-judge adjudication prompt. Unresolved
cases were flagged for human audit.

`configs/llm_policy_jobs.yaml` records the eight original paper jobs.
`slurm/run_llm_judge.sh` now uses `configs/llm_policy_jobs_new_models.yaml`
to judge only the six new prompted runs, using the same Llama-3.1-8B judge
and rubric. It requests 24 hours per array task and rejects incomplete or
empty generation inputs before starting the server.

Saved outputs now exist for all six new prompted runs, including both DeepSeek
models. The commands below remain available for reproduction or interrupted
jobs; row processing alone does not establish that every saved judgment is
valid. The revision summary below records the remaining invalid/missing labels.

```bash
# Four complete new runs: Gemma 12B, Gemma 31B, Qwen 3.6, Nemotron Nano.
sbatch --array=0-3 slurm/run_llm_judge.sh

# If the original DeepSeek generation jobs have stopped, resume them.
# Saved responses are retained; only missing or empty conditions are retried.
bash slurm/submit_model.sh deepseek_r1_distill_qwen_32b
bash slurm/submit_model.sh deepseek_r1_distill_llama_70b

# After both DeepSeek generation jobs finish successfully:
sbatch --array=4-5 slurm/run_llm_judge.sh
```

Each new prompted CSV must contain 242,640 unique, nonempty responses
(`3,370 x 72`). Generation checkpoints are written atomically. Avoid running
two generation jobs for the same model/output file at once.

These resume commands retain the existing 512-token generation budget and
saved responses, including responses that ended at the token limit. Longer
DeepSeek answers require a separate generation run with a larger output budget.

Both DeepSeek launchers use eager execution (`--enforce-eager`), synchronous
scheduling (`--no-async-scheduling`), 32 server sequences, a 4,096-token prefill
budget, and 16 generation workers to mitigate a CUDA illegal-memory-access
crash observed during inference. This profile can reduce throughput and still
needs verification on the cluster. Exhausted connection failures stop generation
and save its checkpoint; individual request timeouts remain retryable.

If the CUDA crash recurs, a diagnostic resubmission can use
`CUDA_LAUNCH_BLOCKING=1 GENERATION_MAX_WORKERS=1 VLLM_MAX_NUM_SEQS=1 bash slurm/submit_model.sh deepseek_r1_distill_qwen_32b`.
vLLM's [troubleshooting guide](https://docs.vllm.ai/en/v0.19.0/usage/troubleshooting/)
recommends CUDA launch blocking to identify the failing kernel. This setting
slows execution; retain the first CUDA traceback from the job logs.

For an already running OpenAI-compatible judge endpoint, the same jobs can be
run and combined locally with:

```bash
python -m scoring.llm_policy_run_jobs \
  --jobs configs/llm_policy_jobs_new_models.yaml \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --batch-size 512 \
  --max-workers 256 \
  --resume \
  --combined-output artifacts/phase1_results_new_models_combined_labeled.csv
```

The paper reports mean pairwise Cohen's kappa of 0.808, 81.2% unanimous
agreement, 18.8% adjudication, 76 parse failures, and 919 cases still flagged
after adjudication.

## Human audit dashboard

The original Streamlit audit dashboard and its supporting audit-set,
analysis, tracking, and artifact-freezing modules are preserved under
`audit/`. The interface blinds prompt-family and clustering metadata until a
primary human label has been saved, autosaves progress, and exports CSV or
JSON annotations.

Launch the existing blinded audit set from the repository root with:

```bash
streamlit run audit/dashboard.py -- \
  --audit-file artifacts/audit/audit_set_blinded.csv \
  --labels-file artifacts/audit/labels_in_progress.csv \
  --coder-id coder_1
```

The dashboard's original annotation taxonomy is unchanged so that future
article-revision work can build on it without silently altering prior audit
decisions.

The revision also preserves a separate six-policy, response-only interface in
`audit/frontier/`. It adds independent first-pass saves, confidence, supporting
evidence, an uncertainty flag, and separate adjudication. The
[annotator guide](docs/frontier-audit/ANNOTATOR_GUIDE.md) and
[coordinator guide](docs/frontier-audit/COORDINATOR.md) document deterministic
practice/representative/targeted sampling and agreement/error analysis with
sampling weights and uncertainty clustered by benchmark item or parent.
These are preparation tools, not completed human validation; annotators,
calibration, the final audit design, and independent ratings remain necessary.

## Analysis methods

The original analysis is preserved in `analysis/analysis.Rmd`. It reads
`artifacts/phase1_results_combined_labeled.csv`; set
`PROMPT_CONTROL_RESULTS` to override that location. The manuscript reports R
4.4.1 with tidyverse 2.0.0, car 3.1.3, and effectsize 1.0.0. The original
notebook also loads `lme4`, `emmeans`, `broom.mixed`, `scales`, and `patchwork`
for its exploratory and plotting sections.

To rebuild the combined CSV with all original, new open-model, and frontier
judgments, run `python -m scoring.compile_results` from the repository root.
The [compilation runbook](docs/compiling-results.md) explains the 13-model file,
per-run validation summary, and `PROMPT_CONTROL_STUDY` selection in the notebook.

Render it from the repository root with:

```bash
Rscript -e "rmarkdown::render('analysis/analysis.Rmd')"
```

The reported analyses were:

1. Harmful compliance was compliance on unsafe items; false refusal was
   refusal on safe items. Rates were computed for each of 100 model by prompt
   context conditions.
2. Four functional forms were compared for the refusal-compliance frontier:
   linear, logarithmic, log-linear exponential, and nonlinear exponential.
   Original-scale RMSE was the common comparison metric; AIC was compared only
   for models on the same response scale.
3. Nested fixed-effect regressions tested whether harmful compliance improved
   false-refusal fit after model identity and whether slopes differed by model.
4. Euclidean distance from `(0, 0)` measured overall boundary error. Fixed
   effects tested model, prompt family, and clarity.
5. Frontier position was `z(harmful compliance) - z(false refusal)`. Prompted
   conditions were analyzed with model and prompt-family fixed effects.
6. Chi-square tests compared six-policy distributions across prompt families,
   clarity levels, and the family by clarity cells. Cramer's V summarized
   association strength.
7. Six planned routing contrasts compared each matching prompt-family/policy
   pair with that policy under all other prompted families. Odds ratios used a
   0.5 continuity correction, followed by Benjamini-Hochberg correction.
8. Within-item switching compared each prompted response with the same model,
   benchmark item, and no-prompt baseline policy.

No random-effect model contributes to the manuscript's reported inferential
claims, although the original notebook retains exploratory mixed-effects code.

## Main findings

These are the original manuscript's reported findings. They have not been
replaced here with estimates from the expanded revision dataset.

- 23.98% of prompted item-model responses changed policy relative to the
  matching no-prompt baseline.
- The nonlinear exponential frontier fit the harmful-compliance/false-refusal
  relationship with `R^2 = .783`; the estimated decay was `b = 16.23`, and the
  lower asymptote was about 2.8% false refusal.
- Model identity explained distance from the ideal point. Prompt family and
  clarity mainly shifted conditions along the frontier rather than toward the
  origin.
- Prompt family changed the six-policy distribution (`chi-square(40) =
  4783.3`, `p < .0001`, Cramer's `V = .06`). Matching families increased their
  intended policies, with odds ratios from 1.28 for Answer to 6.39 for Source
  Isolation.
- Clarity moved prompted conditions along the frontier (`F(2,83) = 13.12`,
  `p < .001`, partial eta-squared `.24`). Vague prompts were most
  compliance-biased and fallback prompts most refusal-biased.
- Clarity also changed policy distributions (`chi-square(15) = 4935.9`,
  `V = .04`), while the family by clarity association was larger
  (`chi-square(120) = 60201`, `V = .11`). Fallback rules did not act as a
  uniform amplifier; their effect depended on the prompt family.

## Revision updates (October 2026)

The checked-in revision snapshot adds six open models: Gemma 4 12B, Gemma 4 31B,
Qwen 3.6 35B-A3B, Nemotron 3 Nano 4B, DeepSeek-R1-Distill-Qwen-32B, and
DeepSeek-R1-Distill-Llama-70B. Each prompted run covers all 3,370 original items
under 72 interventions. Each new control run covers the 850 XSTest/HarmBench
items with no system prompt; these controls do not cover IHEval.

### Completed frontier generation and judging

The frontier extension uses a frozen, reproducible sample of 50 items shared
across three providers and all 73 conditions: **10,950 main cases**. The sample
contains 20 safe and 10 unsafe XSTest items, 10 HarmBench items, and five aligned
and five conflict IHEval items. A disjoint 10-item, five-condition pilot
accounted for another 150 cases and is excluded from the main analysis.

| Provider / requested generator | Main cases | Judged nonempty responses | Empty provider blocks |
|---|---:|---:|---:|
| OpenAI `gpt-6-astra` | 3,650 | 3,650 | 0 |
| Anthropic `claude-opus-5-5` | 3,650 | 3,181 | 469 |
| Google `gemini-3.1-pro-preview` | 3,650 | 3,609 | 41 |
| **Total** | **10,950** | **10,440** | **510** |

Generation is complete, the records were retrieved and verified against the
frozen request identities and hashes, and worker cleanup is recorded. The
[main archive](artifacts/frontier/runs/2026-10-08-main/) retains raw provider
results, normalized responses, checkpoint metadata, cost reservations, and
execution provenance. API credentials and private account configuration are
excluded. The ledger preserves uncertain charges; its accounted totals are not
final provider invoices.

The extension restores HarmBench contextual passages and IHEval task-system
instructions and references. IHEval controls retain the task system while
omitting the experimental intervention; XSTest/HarmBench controls omit the
system field. The APIs used low reasoning effort and a 4,096-token output limit,
so this is a separate extension rather than a temperature-zero replication of
the original open models. See the [frozen protocol](docs/frontier-experiment.md)
and [main configuration](configs/frontier-main.json). Analyze the repaired
context and 50-item sample separately from historical context-incomplete runs;
they do not support unrestricted benchmark-wide rankings.

All 10,440 nonempty responses now have valid output-only judgments from the
same Llama-3.1-8B-Instruct judge, using A/B/C first passes and adjudication.
The [OpenAI](artifacts/llm_policy_labels/frontier_main_openai/manifest.json),
[Anthropic](artifacts/llm_policy_labels/frontier_main_anthropic/manifest.json),
and [Google](artifacts/llm_policy_labels/frontier_main_google/manifest.json)
manifests record complete coverage. Request-ID and response-hash checks preserve
distinct requests even when their answer text is identical. Empty provider
blocks retain blank policy labels; a provider block containing text is still
judged. Policy labels, transport outcomes, and native benchmark correctness are
separate endpoints.

### Combined data and revision analysis

The [saved compilation summary](artifacts/phase1_results_combined_labeled.summary.json)
accounts for **2,455,930 rows across 13 generator models**:

| Source | Rows |
|---|---:|
| Four original open models, prompted and controls | 984,040 |
| Six additional open models, prompted | 1,455,840 |
| Six additional safety-only controls | 5,100 |
| Frontier main, including empty provider blocks | 10,950 |

The compiler retains `source_run`, `analysis_study`, `model_cohort`, and
`judge_status`, with source hashes and coverage checks. The saved summary still
contains **10,942 invalid and 35 missing judgments** in the older/open-model
runs; these remain explicit rather than being assigned inferred policy labels.
All six new safety controls and the frontier text responses have valid labels.

The revision adds an opt-in [CPU-only refusal scorer](docs/heuristic-scoring.md)
and [comparison audits](docs/scorer-comparisons.md) against published human
annotations, saved graders, and native IHEval checks. Lexical refusal,
harmful-behavior success, task correctness, and six-policy classification are
different targets: non-refusal alone is not evidence of harmful compliance.
The scorer's internal holdout results do not replace independent human
validation on the current study.

To reproduce the saved v2 compilation offline from the downloaded artifacts:

```bash
python -m scoring.compile_results --safety-heuristic v2 --require-new-controls
```

`analysis/analysis.Rmd` retains all compiled rows in `all_data` and selects the
analysis cohort separately with `PROMPT_CONTROL_STUDY` (`open_models` or
`frontier_main`). It also contains signal-detection analysis of refusal:
unsafe-request refusals are hits, safe-request refusals are false alarms, and
loglinear correction supports finite d-prime and criterion estimates.
See the [compilation runbook](docs/compiling-results.md) for loading and study
selection. Independent human validation, review of unresolved judgments,
revision-specific figures/results, and final manuscript integration remain
separate from generation and compilation completion.

## Repository map

```text
benchmarks/                         Paper benchmark loaders and snapshots
configs/llm_policy_jobs.yaml        Eight final judge jobs
configs/llm_policy_jobs_new_models.yaml  Six additional prompted judge jobs
configs/llm_policy_jobs_new_controls.yaml  Six safety-control judge jobs
core/schema.py                      Shared evaluation-item schema
experiments/run_phase1.py           Full crossed generation experiment
frontier/                           Hosted API generation and judge handoff
models/vllm_client.py               Reproducible vLLM request settings
prompts/registry_v3.yaml            Frozen 8 x 3 x 3 prompt registry
prompts/registry_control.yaml       No-system-prompt baseline
scoring/                            Rule scores and three-pass LLM judge
slurm/                              Four model, four control, and judge jobs
analysis/analysis.Rmd               Original paper analysis
analysis/figures/                   Paper figures
audit/                              Human-audit dashboard and support modules
audit/frontier/                     Six-policy response-only audit tools
docs/                              Protocols, scoring and revision runbooks
artifacts/                          Raw runs, judge outputs, and combined data
tests/                              Tests for the retained paper pipeline
```

## Setup and verification

For the frontier-model extension's credential configuration, see
[credential setup](docs/credential-setup.md). Actual API keys belong only in the
Git-ignored local `.env`; the blank `.env.example` is safe to commit.

The [frontier experiment protocol and runbook](docs/frontier-experiment.md)
documents the frozen 50-item design, corrected benchmark context, three-provider
pilot, budget accounting, and Runpod execution steps. Use
`python -m frontier.run` for a free dry run; paid pilot calls require `--execute`.

The [completed three-provider pilot](artifacts/frontier/runs/2026-10-08-gemini-completion/README.md)
contains 150 verified cases: 129 text responses and 21 provider blocks, with no
recorded truncations. OpenAI, Anthropic, and Gemini each completed 50 cases.
All four pilot workers are deleted. The main experiment is now uploaded under
`artifacts/frontier/runs/2026-10-08-main/`, with all 10,950 cases present.
The [frontier cluster judging runbook](docs/frontier-judging.md) explains how to
prepare its 10,440 nonempty answers, submit the three-provider judge array,
resume interrupted jobs, and rejoin all cases while retaining 510 empty provider
blocks separately. All three frontier judge manifests now report completion;
the revision summary above records the current coverage.

See the [revision status report and checklist](docs/revision-status-2026-10-07.md)
for the original revision task list and historical budget/decision records.
Its launch-era status text predates the completed generation and judging
summarized above.

Git LFS is required for the large CSV artifacts.

```bash
git lfs install
git lfs pull
conda env create -f environment.yml
conda activate prompt-controllers
python -m pytest tests/
```

The Python requirements can also be installed with `pip install -r
requirements.txt`. vLLM is Linux-only and is guarded accordingly in the pip
requirements file.

## License and citation

The code is released under the MIT License. Citation metadata is in
`CITATION.cff`.
