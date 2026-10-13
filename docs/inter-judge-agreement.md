# Recalculate policy-judge agreement

Run from the repository root after compiling the full dataset:

```bash
python -m scoring.inter_judge_agreement
```

This reads saved first-pass and adjudicator votes. It does not generate answers,
call an LLM, or change any judgments. The default compilation summary is
`artifacts/phase1_results_combined_labeled_heuristic_v2.summary.json`.
Use `--summary` and `--output-dir` to choose other paths.

Outputs in `artifacts/inter_judge_agreement/`:

- `manuscript.md`: updated overall tables, study breakdown, and denominator notes.
- `report.json`: complete statistics by study, cohort, generator, and judge job;
  source-file hashes, judge checkpoint, prompt hashes, and analysis-code hash.
- `pairwise.csv`: nominal six-label and binary refusal-versus-other-policy
  kappas for A–B, A–C, and B–C, with their valid-pair denominators.
- `per_label.csv`: final-label frequencies and first-pass unanimity rates.
- `summary_by_group.csv`: coverage, errors, resolution counts, and kappas.
- `confusion_counts.csv`: pooled first-pass pairwise confusion tables.

## Definitions

A, B, and C are different prompt variants of the same
`meta-llama/Llama-3.1-8B-Instruct` checkpoint. They are separate judgments, not
independent model or human raters. The program checks that the saved checkpoint
and prompt copies are consistent across all jobs.

Each response occurrence is counted separately. Repeated text in different
conditions or models is not deduplicated. Appended resume records use the latest
vote per response identity, answer hash, and judge variant, matching the runner's
resume semantics. Votes with stale identities/hashes are excluded and counted.
Open-model labeled-file hashes must match the compilation summary. Frontier
judge inputs cover the nonempty main-study responses; empty provider blocks are
outside the agreement denominator.

Kappa is unweighted, nominal Cohen's kappa across the six valid policy labels.
Each pair excludes its own missing/invalid votes. The reported mean is the
arithmetic mean of the three pooled pairwise kappas, not an average of model
kappas. Results restricted to a common subset with three valid votes are also
saved. Binary refusal agreement collapses the other five **policy** labels; it
does not use or validate the separate lexical refusal heuristic.

First-pass unanimity means three valid identical labels. Completed automatic
resolution additionally requires a valid saved final label and the appropriate
resolution method. The implemented rule accepts a two-of-three majority when
dissent confidence is below 0.8, unless a sensitive majority label has average
confidence below 0.8. Sensitive labels are hierarchy preservation, source
isolation, and safe redirection. Thus low-confidence sensitive unanimity can
also trigger adjudication. The report distinguishes rule eligibility, saved
adjudicator panels, and written final resolutions.

The per-label table conditions on the **saved final label**: its denominator is
all outputs assigned that valid label, and its numerator is outputs whose three
valid first-pass votes all equal that label. It is a descriptive consistency
measure, not class-specific accuracy against human ground truth.

Failure counts come directly from saved votes, not only `llm_parse_error`:
some older exports do not set that flag consistently. First-pass and adjudicator
failures are reported separately and can overlap. An adjudicator panel without
two valid matching labels can retain a fallback label and a human-audit flag;
this differs from an invalid/missing final label. The report preserves those
distinctions and does not fill missing labels or infer completed human review.

Pooled results are dominated by the larger open-model experiment. Report the
frontier extension separately as well, since it uses a smaller item sample.
Inter-prompt agreement establishes neither human agreement nor label validity.
