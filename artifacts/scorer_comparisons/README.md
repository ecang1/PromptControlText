# Scorer agreement audit

All comparisons use saved answers and labels or CPU-only official checks. No new LLM judgments were requested.

## XSTest refusal validation

Human full and partial refusal are grouped together. The 135 held-out item IDs are shared across the five generators (675 responses). The public corpus was previously inspected; this is internal validation, not a fresh independent test. There are no saved XSTest GPT-4 grader outputs in these inputs.

| Response set | N | V1 κ | V2 κ | Official string baseline κ |
|---|---:|---:|---:|---:|
| gpt4 | 135 | 0.896 | 0.896 | 0.867 |
| llama2new | 135 | 0.788 | 0.787 | 0.877 |
| llama2orig | 135 | 0.786 | 0.821 | 0.752 |
| mistralguard | 135 | 0.569 | 0.808 | 0.623 |
| mistralinstruct | 135 | 0.286 | 0.820 | 0.363 |
| all | 675 | 0.744 | 0.849 | 0.767 |

## HarmBench saved grader comparisons

Positive means harmful behavior success. The heuristic's compliance flag is only a proxy for this target; refusal detection does not establish harmful success. All 602 published validation examples are included. Grader labels are reused from the saved file; no grader models were rerun. `cls` and `cls_mistral` identify the saved classifier columns, not newly verified checkpoint versions.

| Reference | N | V1 proxy κ | V2 proxy κ | Reference vs human majority κ |
|---|---:|---:|---:|---:|
| human_majority | 602 | 0.436 | 0.401 | — |
| gpt-4-0613 | 602 | 0.510 | 0.466 | 0.821 |
| gpt-4-1106-preview | 602 | 0.375 | 0.342 | 0.763 |
| cls | 602 | 0.513 | 0.462 | 0.818 |
| cls_mistral | 602 | 0.513 | 0.462 | 0.811 |
| PAIR_gpt-4-0613 | 602 | 0.414 | 0.361 | 0.758 |
| llama_guard | 602 | 0.128 | 0.114 | 0.304 |
| advbench | 602 | 0.690 | 0.615 | 0.369 |
| GPTFuzz | 602 | 0.285 | 0.277 | 0.463 |
| mixtral | 602 | 0.330 | 0.292 | 0.688 |
| gpt-3.5-turbo-0613 | 602 | 0.139 | 0.131 | 0.312 |
| chatglm | 602 | 0.254 | 0.233 | 0.315 |

The CSV also includes the alternative nonrefusal mapping, every pair of saved graders and human annotators, and results by generating model and attack method. Human annotator agreement with a majority that includes their own vote is descriptive, not independent validation.

## IHEval official task correctness

Only the context-complete frontier main responses can be assessed as the intended task. Empty provider blocks are excluded. The historical open-model runs omitted the required task system and are not retrospectively treated as official IHEval runs. Native task correctness is compared to the old hierarchy flag and, separately, a v2 refusal proxy (refusal on wrong-password inputs; nonrefusal on valid-password inputs). V2 remains unused for IHEval in the combined analysis CSV.

| Frontier group | N | Old hierarchy flag vs native κ | V2 refusal proxy vs native κ |
|---|---:|---:|---:|
| frontier_main/Claude-Opus-5.5 | 688 | 0.000 | -0.006 |
| frontier_main/GPT-6-Astra | 730 | 0.000 | 0.049 |
| frontier_main/Gemini-3.1-Pro | 730 | 0.031 | 0.054 |
| frontier_main/all | 2148 | 0.006 | 0.050 |

These are comparisons of different constructs. A model can correctly protect a password without stating a refusal. The official task checks are CPU-only and cost no grader API budget.

## Agreement with existing policy judgments

These Llama policy judgments are comparators, not independent gold annotations. The two mappings retain only primary refusal, or combine primary refusal and safe redirection. Valid, nonempty XSTest/HarmBench answers are pooled for this mapping only; benchmark-specific results and both heuristic versions are in the full CSV. Open-model and frontier studies remain separate.

| Study/model | Paired N | V2 vs refusal-only κ | V2 vs refusal + safe redirection κ | Excluded cases |
|---|---:|---:|---:|---:|
| frontier_main/Claude-Opus-5.5 | 2493 | 0.305 | 0.666 | 427 |
| frontier_main/GPT-6-Astra | 2920 | 0.410 | 0.629 | 0 |
| frontier_main/Gemini-3.1-Pro | 2879 | 0.881 | 0.883 | 41 |
| frontier_main/all | 8292 | 0.587 | 0.742 | 468 |
| open_models/DeepSeek-R1-Distill-Llama-70B | 60137 | 0.340 | 0.488 | 1063 |
| open_models/DeepSeek-R1-Distill-Qwen-32B | 60160 | 0.441 | 0.531 | 1040 |
| open_models/Gemma-4-12B | 61091 | 0.831 | 0.866 | 109 |
| open_models/Gemma-4-31B | 61112 | 0.791 | 0.808 | 88 |
| open_models/Llama-3.1-8B | 61987 | 0.908 | 0.961 | 63 |
| open_models/Llama-3.3-70B | 61921 | 0.845 | 0.858 | 129 |
| open_models/Nemotron-3-Nano-4B | 60916 | 0.402 | 0.709 | 284 |
| open_models/Qwen-2-7B | 61889 | 0.851 | 0.887 | 161 |
| open_models/Qwen-2.5-72B | 61594 | 0.525 | 0.655 | 456 |
| open_models/Qwen-3.6-35B-A3B | 59851 | 0.215 | 0.403 | 1349 |
| open_models/all | 610658 | 0.637 | 0.740 | 4742 |

## Coverage

```json
{
  "rows": 2450830,
  "policy_safety_paired": 618950,
  "policy_safety_excluded": 5210,
  "iheval_legacy_missing_task_system": 1824480,
  "iheval_native_scored": 2148,
  "iheval_empty_excluded": 42
}
```

## Interpretation

V2 improves XSTest refusal agreement, but agreement with HarmBench harmful-success graders remains modest. The hierarchy/refusal proxies do not reproduce native IHEval task accuracy. These results do not establish GPT-4 grader equivalence or equal error rates across generators.

All metrics are unweighted binary Cohen's κ. Undefined constant-marginal comparisons remain missing, rather than zero. Missing and invalid grader outputs are excluded explicitly. The JSON records source/scorer hashes, label mappings, marginal positive rates and confusion matrices.

Files: `kappas.csv` (all comparisons), `harmbench_pairwise_kappa.csv` (matrix), `comparisons.json` (full provenance).
