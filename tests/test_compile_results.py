import csv
import json
from pathlib import Path

import pytest

from scoring.compile_results import compile_sources
from scoring import compile_results as compiler


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def row(**extra):
    return {"benchmark": "XSTest", "item_id": "item", "prompt_family": "Answer-first",
            "clarity_level": "explicit", "prompt_variant": "v1", "input_text": "task",
            "model_output": 'An answer, with "quotes"\nand a newline.', "llm_policy_label": "compliance",
            "model_name": "model", **extra}


def source(path, study="open_models", **extra):
    return {"path": path, "job_id": path.stem, "expected_rows": 1, "analysis_study": study,
            "model_cohort": "new_open_models" if study == "open_models" else "frontier_models",
            "experiment_group": "prompted", **extra}


def test_compiler_retains_all_labels_metadata_controls_and_empty_blocks(tmp_path):
    old, new, frontier = [tmp_path / (name + ".csv") for name in ("old", "new", "frontier")]
    write_csv(old, [row(model_name="", refusal_score="0.0", compliance_score="1.0")])
    write_csv(new, [row(item_id="item2", llm_policy_label="parse_error", prompt_family="No-system-prompt")])
    write_csv(frontier, [row(request_id="r1", provider="google", experiment_group="control", transport_outcome="text"),
                         row(request_id="r2", provider="google", experiment_group="prompted", transport_outcome="blocked",
                             model_output="", llm_policy_label="", reference_json='{"answer": "gold"}')])
    output = tmp_path / "combined.csv"
    report = compile_sources([source(old, default_model="Qwen2.5-72B", model_cohort="original_open_models"),
                              source(new, experiment_group="control"),
                              source(frontier, "frontier_main", expected_rows=2)], output)
    rows = read_csv(output)
    assert len(rows) == report["total_rows"] == 4
    assert rows[0]["model_output"] == row()["model_output"]
    assert rows[0]["model_name"] == "Qwen2.5-72B"
    assert rows[0]["compliance_score"] == "1.0"
    assert rows[1]["llm_policy_label"] == "parse_error" and rows[1]["judge_status"] == "invalid_judgment"
    assert rows[1]["experiment_group"] == rows[2]["experiment_group"] == "control"
    assert rows[2]["source_run"] == "frontier_main_google" and rows[2]["score_source"] == "original_heuristic_scorers"
    assert rows[3]["judge_status"] == "not_applicable_empty_provider_block"
    assert rows[3]["llm_policy_label"] == rows[3]["refusal_score"] == ""
    assert rows[3]["reference_json"] == '{"answer": "gold"}'
    assert report["by_study"] == {"open_models": 2, "frontier_main": 2}
    assert json.loads(output.with_suffix(".summary.json").read_text())["total_rows"] == 4


@pytest.mark.parametrize("failure", ["duplicate", "count", "model", "frontier_label"])
def test_compiler_failure_preserves_previous_combined_csv(tmp_path, failure):
    path, output = tmp_path / "source.csv", tmp_path / "combined.csv"
    output.write_text("previous data\n")
    rows, config = [row()], source(path)
    if failure == "duplicate":
        rows.append(row())
        config["expected_rows"] = 2
    elif failure == "count":
        config["expected_rows"] = 2
    elif failure == "model":
        rows[0]["model_name"] = ""
    else:
        config = source(path, "frontier_main")
        rows[0].update(request_id="r", provider="openai", transport_outcome="text", llm_policy_label="parse_error")
    write_csv(path, rows)
    with pytest.raises(ValueError):
        compile_sources([config], output)
    assert output.read_text() == "previous data\n"
    assert not list(tmp_path.glob("combined.csv.*.tmp"))


def test_duplicate_sources_and_self_input_are_rejected(tmp_path):
    path = tmp_path / "source.csv"
    write_csv(path, [row()])
    with pytest.raises(ValueError):
        compile_sources([source(path), source(path)], tmp_path / "out.csv")
    with pytest.raises(ValueError):
        compile_sources([source(path)], path)


def test_v2_rescores_all_cohorts_preserves_legacy_and_skips_unavailable_answers(tmp_path):
    old, frontier = tmp_path / "old.csv", tmp_path / "frontier.csv"
    write_csv(old, [row(model_output="I can’t help with that.", refusal_score="0.0", compliance_score="1.0"),
                    row(item_id="malformed", model_output="<tool_call>garbage", malformed_output="tool_call_token", refusal_score=""),
                    row(item_id="iheval", benchmark="IHEval", hierarchy_following_score="0.5")])
    write_csv(frontier, [row(request_id="r1", provider="openai", transport_outcome="text", model_output="I can’t help with that."),
                         row(request_id="r2", provider="google", transport_outcome="blocked", model_output="", llm_policy_label="")])
    output = tmp_path / "v2.csv"
    report = compile_sources([source(old, expected_rows=3), source(frontier, "frontier_main", expected_rows=2)],
                             output, safety_heuristic="v2")
    rows = read_csv(output)
    assert rows[0]["refusal_score"] == rows[3]["refusal_score"] == "1.0"
    assert rows[0]["legacy_refusal_score"] == "0.0" and rows[0]["legacy_compliance_score"] == "1.0"
    assert rows[0]["llm_policy_label"] == "compliance"
    assert rows[0]["score_source"] == "safety-heuristic-v2"
    assert rows[1]["refusal_score"] == "" and rows[1]["safety_heuristic_status"] == "not_applicable_malformed_output"
    assert rows[2]["hierarchy_following_score"] == "0.5" and rows[2]["safety_heuristic_status"] == "not_applicable_benchmark"
    assert rows[4]["refusal_score"] == "" and rows[4]["safety_heuristic_status"] == "not_applicable_empty_response"
    assert report["by_safety_heuristic_status"]["scored"] == 2


def test_discovery_adds_only_finished_new_controls_and_can_require_all(tmp_path, monkeypatch):
    monkeypatch.setattr(compiler, "ROOT", tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("llm_policy_jobs.yaml", "llm_policy_jobs_new_models.yaml"):
        (configs / name).write_text("jobs: []\n")
    jobs = [{"job_id": "control_gemma4_12b", "output_dir": "labels/control_gemma4_12b",
             "experiment_group": "control", "expected_rows": 850}]
    import yaml
    (configs / "llm_policy_jobs_new_controls.yaml").write_text(yaml.safe_dump({"jobs": jobs}))
    assert compiler.discover_sources(include_frontier=False) == []
    with pytest.raises(ValueError, match="judgments are missing"):
        compiler.discover_sources(include_frontier=False, require_new_controls=True)
    directory = tmp_path / jobs[0]["output_dir"]
    directory.mkdir(parents=True)
    manifest_path = directory / "manifest.json"
    manifest = {"job_id": jobs[0]["job_id"], "total_rows": 850,
                "valid_labeled_rows": 800, "complete": False}
    manifest_path.write_text(json.dumps(manifest))
    assert compiler.discover_sources(include_frontier=False) == []
    with pytest.raises(ValueError, match="judging is incomplete"):
        compiler.discover_sources(include_frontier=False, require_new_controls=True)
    manifest.update(complete=True, valid_labeled_rows=850)
    manifest_path.write_text(json.dumps(manifest))
    sources = compiler.discover_sources(include_frontier=False, require_new_controls=True)
    assert len(sources) == 1
    assert sources[0]["path"] == directory / "labeled.csv"
    assert sources[0]["model_cohort"] == "new_open_models"
    assert sources[0]["experiment_group"] == "control"
    assert sources[0]["require_complete"] is True


def test_new_control_with_invalid_labels_cannot_replace_combined_csv(tmp_path):
    path, output = tmp_path / "control.csv", tmp_path / "combined.csv"
    write_csv(path, [row(prompt_family="No-system-prompt", clarity_level="", llm_policy_label="parse_error")])
    output.write_text("previous data\n")
    with pytest.raises(ValueError, match="control CSV has missing or invalid judgments"):
        compile_sources([source(path, experiment_group="control", require_complete=True)], output)
    assert output.read_text() == "previous data\n"
