# Compile all judged runs for analysis

Run from the repository root, with the project's Python environment active:

```bash
python -m scoring.compile_results
```

This rebuilds `artifacts/phase1_results_combined_labeled.csv` from saved labeled
outputs. It is offline: it makes no provider calls and does not rerun judging.
It includes all eight original jobs (four prompted and four controls), six new
open-model prompted jobs, completed new no-system controls, and the three
completed frontier main runs. The pilot is excluded. Before the six new controls,
the compilation contains 2,450,830 rows:

| Source | Rows |
|---|---:|
| Four original open models, prompted | 970,560 |
| Four original no-system controls | 13,480 |
| Six new open models, prompted | 1,455,840 |
| Six new safety no-system controls, once generated and judged | 5,100 |
| Frontier main, including provider blocks | 10,950 |

The CSV contains all 13 generator models and the union of their columns, keeping
answers, labels, prompt conditions, input context, references, and provider
outcomes. `source_run`, `analysis_study`, and `model_cohort` identify provenance.
The original Qwen-2.5-72B prompted file lacks `model_name`; compilation fills it
from its known run identity. Control/prompted groups remain distinct. There are
no new no-system controls in the earlier saved runs. To generate and judge the
missing controls, use [the baseline submission instructions](new-model-baselines.md).
These new controls cover the 850 XSTest/HarmBench items per model for refusal
analysis; they exclude IHEval. Once all six are complete, the full compilation
has 2,455,930 rows. Completed
controls are included automatically; missing or unfinished new controls are
skipped with warnings. Add `--require-new-controls` to require all six before
replacing the combined CSV.

For frontier answers, the compiler adds the same heuristic score columns and
output-quality flags used by the original generation pipeline. These remain
separate from the LLM policy labels and benchmark-native task correctness.
Empty provider blocks retain blank labels and blank scores. The compiler never
converts a provider block into a policy refusal.

For the improved CPU-only refusal scorer, use
`python -m scoring.compile_results --safety-heuristic v2`. This applies v2 to all
eligible XSTest/HarmBench answers, preserving the original scores in `legacy_*`
columns. See [validation, limitations, and R instructions](heuristic-scoring.md).
The default command retains the original scoring behavior.

Legacy missing and invalid labels are retained with explicit `judge_status`;
they are not inferred from majority votes or keyword scores. The file next to
the CSV, `phase1_results_combined_labeled.summary.json`, records per-run row
counts, label/status counts, source paths, and SHA-256 hashes. Review it for
legacy judgments that still need repair. Frontier coverage, labels, inputs,
and provenance must all pass `frontier.finish_judge` before inclusion.

Compilation streams the large CSVs, checks unique case identities and expected
row counts, and replaces the combined CSV only after all sources pass. It does
not append to a previous combined file, so repeating it does not duplicate rows.
Allow several GB of disk space and a few minutes. Generated combined files are
local analysis products; commit the compiler, rather than repeatedly uploading
the multi-GB CSV, and rebuild it after pulling new results.

To choose another output path or exclude the frontier experiment:

```bash
python -m scoring.compile_results --output .local/all_results.csv
python -m scoring.compile_results --open-models-only --output .local/open_models.csv
```

## Use in the analysis notebook

`analysis/analysis.Rmd` reads the same combined CSV path as before. `all_data`
contains all 13 models and all compiled rows, including empty provider blocks
and any completed new controls.
The loader uses explicit numeric score types and skips unused long answer/metadata
columns. Study selection happens separately in `analysis_data`, which excludes
empty provider blocks from response analysis. Legacy missing or
invalid judgments stay in the data, preserving the original heuristic safety-rate
denominators; their policy labels remain unclassified. For a fully judged subset,
use `select_analysis_results(all_data, include_unjudged = FALSE)`.
Its model factors now retain new models instead of converting them to `NA`.

The default analysis study is `open_models`, covering the ten open models;
`all_data` always remains complete. The frontier
experiment uses a 50-item subset with repaired IHEval context, so analyze it
separately. Before knitting, select it in R:

```r
Sys.setenv(PROMPT_CONTROL_STUDY = "frontier_main")
rmarkdown::render("analysis/analysis.Rmd")
```

To switch back:

```r
Sys.setenv(PROMPT_CONTROL_STUDY = "open_models")
```

`PROMPT_CONTROL_RESULTS` still overrides the input CSV path. To inspect all rows
or custom cohorts directly, read the combined CSV and filter `analysis_study`,
`model_cohort`, `experiment_group`, and `judge_status` explicitly. Native task
correctness requires separate scoring against the stored references.

To load everything interactively and inspect model coverage:

```r
all_data <- load_combined_results(data_path, study = "all")
all_data %>% count(analysis_study, model_name)
analysis_data <- select_analysis_results(all_data, study = "frontier_main")
```
