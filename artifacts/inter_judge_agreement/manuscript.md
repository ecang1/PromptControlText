# Inter-judge agreement

Computed offline from the saved A/B/C first-pass votes and current adjudications. These are three prompt variants of the same Llama-3.1-8B-Instruct checkpoint, not three independent models. Each response occurrence is counted separately, including identical text in different conditions. Resumed append-only files use the latest vote per response identity, answer hash, and judge variant. Provider blocks without text do not enter the judge denominator.

Across 2,455,391 outputs with saved first-pass judgments, mean pairwise nominal Cohen’s κ was 0.747. Pairwise kappas exclude failed/invalid votes separately for each pair. First-pass unanimity occurred for 75.1% of outputs (n = 1,843,415); 611,710 outputs (24.9%) had a saved adjudicator panel. Unanimity and automatic resolution are distinct: low-confidence sensitive labels can be adjudicated, whereas a two-of-three majority with low dissent confidence can be accepted automatically.

| Metric | Value |
|---|---:|
| Outputs with saved first-pass judgments | 2,455,391 |
| First-pass prompt variants per output | 3 (A, B, C; same judge checkpoint) |
| Mean pairwise Cohen’s κ | 0.747 |
| A–B Cohen’s κ | 0.725 (n = 2,454,920 valid pairs) |
| A–C Cohen’s κ | 0.759 (n = 2,454,944 valid pairs) |
| B–C Cohen’s κ | 0.757 (n = 2,454,986 valid pairs) |
| First-pass unanimous agreement | 75.1% (n = 1,843,415) |
| Resolved automatically by unanimity | 75.1% (n = 1,843,412) |
| Resolved automatically by two-of-three majority | 0.011% (n = 266) |
| Sent to adjudication (saved current panel) | 24.9% (n = 611,710) |
| First-pass outputs with a failed/invalid vote | 0.024% (n = 583) |
| Adjudicated outputs with a failed/invalid adjudicator vote | 1.0% (n = 23,982) |
| Adjudications without a valid majority | 0.7% (n = 16,444) |
| Saved human-audit flag | 0.2% (n = 5,502) |
| Invalid final policy label | 0.4% (n = 10,942) |
| Missing final policy label | 0.001% (n = 35) |

Table 5. Overall agreement and adjudication statistics. Error counts denote response occurrences, not individual API attempts, and first-pass/adjudicator error categories can overlap. The full table contains 2,455,930 rows; 510 empty frontier blocks and 29 legacy rows without matching saved first-pass votes are outside the agreement denominator. 3 unanimous panels and 3 saved adjudications are not reflected in a final label in the current combined data. Those data are reported as saved, without filling missing labels.

| Final policy label | n | First-pass unanimous agreement with final label |
|---|---:|---:|
| COMPLIANCE | 1,072,710 | 80.2% |
| CLARIFICATION | 833,345 | 78.8% |
| REFUSAL | 395,026 | 62.3% |
| HIERARCHY_PRESERVATION | 55,862 | 70.6% |
| SOURCE_ISOLATION | 7,795 | 33.2% |
| SAFE_REDIRECTION | 79,705 | 48.3% |

Table 6. Denominator: outputs assigned each valid final policy label. Numerator: all three valid first-pass votes equal that final label. Missing/invalid final labels are excluded from this table, but retained in coverage/error totals.

| Study | Outputs with votes | Mean pairwise κ | First-pass unanimity | Adjudication |
|---|---:|---:|---:|---:|
| open_models | 2,444,951 | 0.747 | 75.1% | 24.9% |
| frontier_main | 10,440 | 0.753 | 76.3% | 23.7% |

The mean is the arithmetic mean of the three pooled pairwise kappas, not an average of per-job kappas. The JSON also reports kappas restricted to the common subset with three valid votes. No-majority adjudications can retain a fallback label and should not be described as necessarily missing final labels. Agreement measures consistency among prompt variants, not human validation or correctness. The v2 lexical refusal classifier is a separate endpoint and is not used in these policy-agreement calculations.
