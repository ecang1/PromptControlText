# CPU-only refusal scoring

`scoring/safety_heuristic.py` supplies the opt-in `safety-heuristic-v2` scorer.
It uses shared lexical rules for all models and conditions, with no GPU, paid
grader, or inference calls. It estimates whether the answer expresses refusal,
including refusal mixed with an answer. It does not measure harmful-behavior
success or IHEval task correctness.

V2 normalizes curly apostrophes, recognizes more first-person and implicit
ethical refusals, and removes AI identity alone as a refusal signal. It excludes
block quotes, double-quoted passages and fenced code from refusal matching, and
limits indirect ethical objections to the opening. It treats an explanation
that a fictional entity lacks real records as an invalid-premise answer. Mixed
responses with an explicit refusal still count as refusal; safe redirections can
also contain refusal signals. These are lexical approximations, not semantic
guarantees. Each prediction includes its matched rule IDs.

## Rebuild the analysis data

From the repository root:

```bash
python -m scoring.compile_results --safety-heuristic v2
```

This regenerates the usual combined CSV offline, using v2 consistently for
eligible XSTest and HarmBench answers from both open and frontier models. Raw
answers, existing LLM judgments and IHEval scores are preserved. Empty answers
and malformed outputs have blank safety scores and explicit statuses.

`refusal_score`, `compliance_score`, `clarification_score`, and `abstention_score`
contain the v2 values. The previous values appear in `legacy_*` columns.
`safety_heuristic_version`, `safety_heuristic_status`, `safety_heuristic_rules`,
and the combined-file summary identify the scorer and its source-code hash.
Use the same version across all compared models. Reload the CSV and rerun the R
analysis; previously exported figures retain their old scores until regenerated.

The compiler defaults to legacy behavior unless the v2 flag is supplied.
To keep both compiled files:

```bash
python -m scoring.compile_results --safety-heuristic v2 \
  --output artifacts/phase1_results_combined_labeled_heuristic_v2.csv
```

From the repository root, select that second file before loading the R notebook.
Resolve the absolute path so knitting from `analysis/` cannot fall back to the
old CSV:

```r
Sys.setenv(PROMPT_CONTROL_RESULTS = normalizePath(
  "artifacts/phase1_results_combined_labeled_heuristic_v2.csv"
))
```

## Validation and interpretation

The released [XSTest responses](https://github.com/paul-rottger/xstest/tree/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/model_completions)
include human `final_label` annotations. These are human reference labels,
not saved predictions from the paper's GPT-4 grader. Full and partial refusal
are grouped together for the binary comparison.

Rules were developed on 315 shared item IDs across five models, then frozen
before evaluation on the other 135 IDs (675 responses). Grouping by item prevents
the same question from appearing in both splits through a different model.
The same public corpus was previously inspected in aggregate and through a few
error examples, so this is an internal holdout, not a fresh independent validation dataset. Performance on
these generators does not establish identical reliability on new generators.

| Published response set | Holdout N | V1 accuracy | V2 accuracy | V1 κ | V2 κ |
|---|---:|---:|---:|---:|---:|
| GPT-4 | 135 | 94.8% | 94.8% | 0.896 | 0.896 |
| Updated Llama-2 | 135 | 89.6% | 89.6% | 0.788 | 0.787 |
| Original Llama-2 | 135 | 91.9% | 93.3% | 0.786 | 0.821 |
| Mistral with safety prompt | 135 | 77.8% | 90.4% | 0.569 | 0.808 |
| Mistral instruction model | 135 | 81.5% | 94.1% | 0.286 | 0.820 |
| Pooled | 675 | 87.1% | 92.4% | 0.744 | 0.849 |

Pooled holdout refusal recall increased from 79.6% to 89.1%; precision increased
from 95.3% to 96.4%. Neither result justifies a claim of GPT-4 grader equivalence.

On HarmBench's 602 human-labeled classifier validation examples, lexical
compliance as a proxy for harmful success gives 70.8% accuracy/κ 0.436 for v1
and 68.9%/κ 0.401 for v2. That target is different from textual refusal. V2 is
an improvement for the XSTest refusal target; it is not an official HarmBench
success scorer and should not be described as one.

Rerun the audit with pinned public sources:

```bash
# Downloads annotated CSVs only. No model calls or charges.
python -m scoring.validate_heuristic --download
# Thereafter, run offline:
python -m scoring.validate_heuristic
```

The JSON report at `artifacts/scorer_validation_v2.json` records source hashes,
the split, scorer hashes, confusion matrices and model/category/safety results.
The accompanying `.predictions.csv` contains row-level labels and matched rules.

Additional saved GPT-4/HarmBench grader, official IHEval, and current policy-judge
agreements are documented in [the scorer comparison audit](scorer-comparisons.md).

Suggested methods wording:

> Refusal was measured with a fixed lexical heuristic, applied identically across
> models and prompt conditions. Full and partial refusal were grouped together.
> On an internal holdout of 675 published, human-annotated XSTest responses, the
> heuristic achieved 92.4% accuracy and Cohen's κ = 0.849. This evaluates expressed
> refusal and does not establish equivalence to an LLM grader or measure native
> HarmBench success. Scorer code, validation split, and prediction audit are retained.
