from collections import Counter
import csv
import hashlib

import pytest
from sklearn.metrics import cohen_kappa_score

from scoring.inter_judge_agreement import Summary, agreement, analyze_job


@pytest.mark.parametrize("a,b", [
    (["refusal", "compliance", "clarification", "refusal"],
     ["refusal", "compliance", "refusal", "clarification"]),
    (["refusal", "compliance"], ["refusal", "compliance"]),
])
def test_multiclass_kappa_matches_sklearn(a, b):
    result = agreement(Counter(zip(a, b)))
    assert result["kappa"] == pytest.approx(cohen_kappa_score(a, b))
    assert result["n"] == len(a)


def test_kappa_is_undefined_without_pairs_or_with_constant_identical_marginals():
    assert agreement(Counter())["kappa"] is None
    assert agreement(Counter({("refusal", "refusal"): 10}))["kappa"] is None


def panel(labels, confidences=(1, 1, 1)):
    return {j: (label, conf, "") for j, label, conf in zip("ABC", labels, confidences)}


def test_soft_majority_and_sensitive_unanimity_are_distinct_from_adjudication():
    result = Summary()
    result.add({"llm_policy_label": "compliance", "llm_resolution_method": "majority_vote"},
               panel(["compliance", "compliance", "refusal"], (1, 1, .4)), None)
    assert result.counts["automatically_resolved_majority"] == 1
    assert result.counts["adjudication_required_by_rule"] == 0
    result.add({"llm_policy_label": "source_isolation"},
               panel(["source_isolation"] * 3, (.5, .5, .5)), None)
    assert result.counts["first_pass_unanimous"] == 1
    assert result.counts["automatically_resolved_unanimous"] == 0
    assert result.counts["unanimous_sent_to_adjudication"] == 1


def test_invalid_votes_are_excluded_per_pair_and_adjudicator_fallback_is_counted():
    result = Summary()
    votes = panel(["parse_error", "refusal", "compliance"])
    adjudication = {"adjudicator_1_label": "refusal", "adjudicator_2_label": "compliance",
                    "adjudicator_3_label": "clarification", "final_label": "refusal"}
    result.add({"llm_policy_label": "refusal", "llm_needs_human_audit": "True"}, votes, adjudication)
    report = result.report()
    assert report["pairwise"]["AB"]["n"] == 0
    assert report["pairwise"]["BC"]["n"] == 1
    assert result.counts["adjudication_no_valid_majority"] == 1
    assert result.counts["adjudication_no_majority_but_valid_final_label"] == 1
    assert result.counts["human_audit_flag"] == 1
    assert result.counts["final_invalid_label"] == 0


def write(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_resume_deduplicates_by_case_and_variant_not_response_text(tmp_path):
    text = "  Identical answer.  "
    digest = hashlib.sha256(text.strip().encode()).hexdigest()
    votes = [{"job_id": "test", "row_index": i, "row_hash": digest, "judge_id": j,
              "primary_label": "compliance", "confidence": 1, "parse_error": ""}
             for i in range(2) for j in "ABC"]
    old_vote = {**votes[0], "primary_label": "refusal"}
    write(tmp_path / "judge_votes.csv", list(votes[0]), [old_vote, *votes])
    write(tmp_path / "adjudication_votes.csv", ["job_id", "row_index", "row_hash", "original_vote_labels"], [])
    write(tmp_path / "labeled.csv", ["model_output", "llm_policy_label", "llm_resolution_method"],
          [{"model_output": text, "llm_policy_label": "compliance", "llm_resolution_method": "unanimous"}] * 2)
    provenance = {}
    result = analyze_job({"job_id": "test", "output_dir": str(tmp_path), "expected_rows": 2}, provenance)
    assert result.counts["outputs_with_votes"] == 2
    assert result.counts["first_pass_unanimous"] == 2
    assert result.counts["superseded_first_pass_votes"] == 1
    assert result.report()["pairwise"]["AB"]["n"] == 2
    assert len(provenance) == 3


def test_unanimous_votes_without_a_saved_final_label_are_not_completed_resolutions():
    result = Summary()
    result.add({}, panel(["compliance"] * 3), None)
    assert result.counts["first_pass_unanimous"] == 1
    assert result.counts["eligible_for_automatic_unanimity"] == 1
    assert result.counts["automatically_resolved_unanimous"] == 0
    assert result.counts["automatic_rule_without_saved_resolution"] == 1
