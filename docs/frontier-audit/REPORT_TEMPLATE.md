# Human validation report — unfilled template

**Status: no human study findings entered.** Synthetic tests are software validation only.

- Generation archive / complete-frame checksum: [pending]
- Code commit, guide version, canonical model and prompt hashes: [pending]
- Calibration packet/checksum, timing (median, p90, actual total), and instruction decisions: [pending]
- Final sample-size rationale and date frozen before comparative inspection: [pending]
- Seeds, representative strata, inclusion probabilities, eligible frame and practice exclusions: [pending]
- Coverage by provider, benchmark stratum, condition and parent; gaps/truncation/nontext exclusions: [pending]
- Independent annotator recruitment/training and adjudicator procedure (no private identities): [pending]
- Missing annotations, uncertainty flags and reasons: [pending]

## Representative text-response component

Assigned N [ ]; A N [ ]; B N [ ]; paired N [ ]; adjudicated N [ ]; valid judge comparisons N [ ].

Report pre-adjudication percent agreement and nominal Cohen kappa, unweighted and inclusion-weighted, overall and per provider. State undefined values and prevalence sensitivity. Report approximate parent-cluster intervals and the number of parents/valid bootstrap replicates.

Insert confusion **counts** and weighted confusion estimates separately, with rows = adjudicated human and columns = canonical judge. For each of six labels report human support, judge support, precision, recall and uncertainty; flag absent/rare classes. Report resolved-case estimates alongside worst/best missingness bounds and exclusion-of-uncertainty sensitivity. Do not call missing classes zero-accuracy classes or pool provider blocks into textual refusals.

## Targeted diagnostic component

Selection rules/seed/candidate N [ ]; selected N [ ]; overlaps excluded [ ]; actual judge provenance [ ]. Report class-specific errors, human disagreements, unresolved cases and their reasons separately. No overall population error estimate may use the combined representative + targeted queue.

## Interpretation

The finite target is the frozen study's eligible text responses conditional on practice exclusions. It does not cover all benchmark populations or validate old-model labels automatically. Agreement is not taxonomy validity; identify disagreements caused by the refusal/fulfillment rubric ambiguity, insufficient output-only context, and other distinctions. Keep safety/harmful-compliance and native IHEval correctness endpoints separate. State annotation workload, missingness, parent dependence, approximate interval limitations and any post-freeze deviations.
