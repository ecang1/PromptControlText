# Compare heuristics, saved graders, and official task checks

Run `scoring.compare_scorers` to produce binary Cohen's κ, accuracy, marginal
positive rates, confusion counts and exclusions for every available comparison.
It makes no model calls and does not alter the heuristic or combined CSV.

```bash
python -m pip install -r requirements-scorer-audit.txt
# First use downloads public source code and human annotations only:
python -m scoring.compare_scorers --download
# After that, rerun offline:
python -m scoring.compare_scorers
```

The default input is the v2 combined CSV created by
`python -m scoring.compile_results --safety-heuristic v2 --output artifacts/phase1_results_combined_labeled_heuristic_v2.csv`.
Use `--combined PATH` for another v2 compiled file. The scorer hash must match
the summary beside that CSV. `--published-only` skips current-run policy and
IHEval comparisons, and does not require the ROUGE dependency.

Outputs are in `artifacts/scorer_comparisons/`:

- `README.md`: readable benchmark and per-model tables with interpretations.
- `kappas.csv`: every comparison, including reference, candidate, target, group,
  split, κ, sample count and exclusions.
- `harmbench_pairwise_kappa.csv`: all saved graders and heuristic proxies in a matrix.
- `comparisons.json`: the same results with source hashes and explicit label mappings.

## Reference labels and mappings

XSTest uses the released human annotations, grouping full and partial refusal
together. It compares both heuristic versions and the exact released
`strmatch_label` baseline. The earlier internal development/holdout split is
preserved, with the same item ID assigned to the same split across all five
generating models. There are no saved XSTest GPT-4 grader outputs in the
available files; human labels on GPT-4-generated answers are a different thing.

HarmBench's local validation snapshot contains saved GPT-4, native classifier,
other evaluator and human labels. No evaluator is rerun. Human majority vote
is the reference for harmful-behavior success. We compare lexical compliance
and, separately, nonrefusal as proxies for that outcome. We also calculate all
pairwise grader/annotator agreements, plus breakdowns by generating model and
attack method. PAIR ratings use the official default success threshold of 10;
GPT-3.5 labels map yes/no to 1/0. Invalid saved values remain missing. `cls` and
`cls_mistral` retain the names in the snapshot; the audit does not establish a
new model-checkpoint provenance for those previously saved predictions.

IHEval uses the exact pinned `eval_tensortrust` function and
`RougeScorer(["rouge1"], use_stemmer=False)` on the nonempty, context-complete
frontier main responses. The old hierarchy flag is compared to native task
correctness. A separate exploratory v2 proxy predicts correct protection from
refusal on wrong-password inputs and from nonrefusal on valid-password inputs.
This is not an adopted v2 IHEval scorer. Historical open-model IHEval generations
omitted the task system, so they are excluded rather than presented as valid
official runs. Empty provider blocks are excluded explicitly.

The existing Llama policy judgments are additional comparators, not independent
gold annotations. For all 13 current generators, we report agreement with
primary refusal and with primary refusal-or-safe-redirection separately. We use
only eligible safety responses with valid saved judgments, and retain study and
benchmark breakdowns. The two original Qwen-2.5 model-name aliases are grouped
together. This analysis does not establish equal validity across models.

κ is undefined when both marginals are the same constant. Such values are null,
not zero. Do not combine different benchmark targets into a single validation κ.
The explicitly labeled `safety_pooled` policy comparison combines only the
shared policy mapping across XSTest and HarmBench.

Official source references:
[XSTest string baseline](https://github.com/paul-rottger/xstest/blob/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/evaluation/classify_completions_strmatch.py),
[IHEval safety check](https://github.com/ytyz1307zzh/IHEval/blob/726a62924c3050045954df94347d53fe2bd1090d/src/safety/evaluate/eval_tensortrust.py),
[PAIR cutoff](https://github.com/centerforaisafety/HarmBench/blob/8e1604d1171fe8a48d8febecd22f600e462bdcdd/baselines/pair/PAIR.py).
