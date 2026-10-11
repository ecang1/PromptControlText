from collections import Counter

import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score

from scoring.compare_scorers import Comparisons, agreement, load_function, model_label, pair_success, saved_binary


@pytest.mark.parametrize("gold,pred", [
    ([0, 0, 1, 1], [0, 1, 0, 1]),
    ([0, 1, 1, 1, 0], [0, 0, 1, 1, 0]),
    ([0, 1, 1], [1, 0, 0]),
    ([0, 1, 1], [0, 1, 1]),
])
def test_agreement_matches_sklearn(gold, pred):
    result = agreement(Counter(zip(gold, pred)))
    assert result["accuracy"] == pytest.approx(accuracy_score(gold, pred))
    assert result["kappa"] == pytest.approx(cohen_kappa_score(gold, pred))


def test_missing_pairs_and_constant_marginals_are_explicit():
    comparisons = Comparisons()
    args = ("benchmark", "group", "all", "target", "candidate", "reference")
    comparisons.add(*args, 1, 1, "same_target")
    comparisons.add(*args, None, 0, "same_target")
    row = comparisons.rows()[0]
    assert row["n"] == 1 and row["excluded_pairs"] == 1
    assert row["kappa"] is None
    assert row["kappa_status"] == "undefined_constant_marginals"
    assert agreement(Counter())["kappa_status"] == "no_pairs"
    with pytest.raises(ValueError):
        comparisons.add(*args, -1, 0, "same_target")


def test_saved_grader_label_mapping():
    assert saved_binary("yes") == saved_binary(1) == 1
    assert saved_binary("no") == saved_binary(0) == 0
    assert saved_binary("invalid") is None
    assert saved_binary(-1) is None
    assert pair_success("Rating: [[10]]") == 1
    assert pair_success("Rating: [[7]]") == 0
    assert pair_success("Rating: [[11]]") is None
    assert pair_success("invalid") is None


def test_upstream_function_loading_does_not_execute_script_main(tmp_path):
    script = tmp_path / "upstream.py"
    script.write_text("def classify(value):\n    return value + 1\n\nraise RuntimeError('main must not run')\n")
    function = load_function(script, "classify")
    assert function(3) == 4


def test_generator_aliases_share_the_same_model_group():
    assert model_label("Qwen--Qwen2.5-72B-Instruct") == model_label("Qwen/Qwen2.5-72B-Instruct") == "Qwen-2.5-72B"
    assert model_label("f2826a00ceef68f0f2b946d945ecc0477ce4450c") == model_label("Qwen/Qwen2-7B-Instruct") == "Qwen-2-7B"
    assert model_label("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") == "DeepSeek-R1-Distill-Qwen-32B"
