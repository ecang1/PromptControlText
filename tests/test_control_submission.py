"""Check cluster job wiring without submitting jobs or starting GPU servers."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scoring.llm_policy_run_jobs import load_jobs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def submit(tmp_path):
    executable = tmp_path / "sbatch"
    log = tmp_path / "submissions.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "path = pathlib.Path(os.environ['SUBMISSION_LOG'])\n"
        "calls = json.loads(path.read_text()) if path.exists() else []\n"
        "calls.append(sys.argv[1:])\n"
        "path.write_text(json.dumps(calls))\n"
        "print(1000 + len(calls))\n"
    )
    executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
           "PROJECT_DIR": str(tmp_path), "SUBMISSION_LOG": str(log)}
    # Test defaults regardless of the shell that invokes pytest.
    env.pop("CONTROL_TIME_LIMIT", None)
    env.pop("CONTROL_JUDGE_TIME_LIMIT", None)

    def run(script, *args, success=True):
        result = subprocess.run(["bash", str(ROOT / "slurm" / script), *args],
                                env=env, capture_output=True, text=True)
        if success:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0
        return json.loads(log.read_text()) if log.exists() else []

    return run


def exports(args):
    export_arg = next(arg for arg in args if arg.startswith("--export="))
    return dict(part.split("=", 1) for part in export_arg[len("--export="):].split(",") if "=" in part)


def test_six_controls_queue_only_baselines_and_dependent_judges(submit):
    jobs = load_jobs(ROOT / "configs/llm_policy_jobs_new_controls.yaml")
    calls = submit("submit_new_controls.sh")
    assert len(calls) == 12  # no redundant smoke-test jobs or prompted jobs
    gpu_counts = [1, 2, 2, 1, 2, 4]
    for task, job in enumerate(jobs):
        generation, judge = calls[task * 2:task * 2 + 2]
        slug = job["job_id"].removeprefix("control_")
        env = exports(generation)
        assert env["MODEL_SLUG"] == slug
        assert env["REGISTRY_VERSION"] == "control"
        assert env["SAFETY_ONLY"] == "1"
        assert env["OUTPUT_FILE"] == job["input"]
        assert int(env["TENSOR_PARALLEL_SIZE"]) == gpu_counts[task]
        assert f"--gres=gpu:{gpu_counts[task]}" in generation
        assert "--time=01:00:00" in generation and "--cpus-per-task=6" in generation
        assert not any(arg.startswith("--dependency=") for arg in generation)
        assert generation[-1].endswith("/slurm/run_model_generic.sh")
        if slug.startswith("gemma4_"):
            assert env["CONDA_ENV"] == "PromptControlText-gemma4"
        assert f"--array={task}" in judge
        assert f"--dependency=afterok:{1001 + task * 2}" in judge
        assert "--kill-on-invalid-dep=yes" in judge
        assert "--time=00:30:00" in judge and "--cpus-per-task=4" in judge
        assert exports(judge)["JUDGE_JOBS_CONFIG"] == "configs/llm_policy_jobs_new_controls.yaml"
        assert exports(judge)["JUDGE_REQUIRE_COMPLETE"] == "1"
        assert judge[-1].endswith("/slurm/run_llm_judge.sh")
        assert job["expected_rows"] == 850 and job["experiment_group"] == "control"


def test_existing_prompted_submission_keeps_smoke_test_and_output(submit):
    smoke, generation = submit("submit_model.sh", "deepseek_r1_distill_qwen_32b")
    assert smoke[-1].endswith("/slurm/run_smoke_test.sh")
    assert "--dependency=afterok:1001" in generation
    assert exports(generation)["REGISTRY_VERSION"] == "v3"
    assert exports(generation)["SAFETY_ONLY"] == "0"
    assert exports(generation)["OUTPUT_FILE"] == "artifacts/phase1_results_deepseek_r1_distill_qwen_32b.csv"
    assert not any(arg.startswith("--time=") for arg in generation)


def test_control_can_explicitly_request_smoke_test(submit):
    smoke, generation, judge = submit("submit_model.sh", "gemma4_12b", "--control", "--judge", "--smoke")
    assert smoke[-1].endswith("/slurm/run_smoke_test.sh")
    assert "--dependency=afterok:1001" in generation
    assert "--dependency=afterok:1002" in judge


def test_unknown_control_is_rejected_before_submitting_any_jobs(submit):
    assert submit("submit_new_controls.sh", "gemma4_12b", "not-a-model", success=False) == []


def runtime_settings(slug, registry=None, **overrides):
    env = dict(os.environ)
    for key in ("REGISTRY_VERSION", "VLLM_MAX_NUM_SEQS",
                "VLLM_MAX_NUM_BATCHED_TOKENS", "GENERATION_MAX_WORKERS"):
        env.pop(key, None)
    env["MODEL_SLUG"] = slug
    if registry is not None:
        env["REGISTRY_VERSION"] = registry
    env.update(overrides)
    result = subprocess.run(
        ["bash", "-ec", 'source "$1"; printf "%s\\n" "$VLLM_MAX_NUM_SEQS" '
         '"$VLLM_MAX_NUM_BATCHED_TOKENS" "$GENERATION_MAX_WORKERS" '
         '"${VLLM_MODEL_ARGS[@]}"', "runtime", str(ROOT / "slurm/model_runtime.sh")],
        env=env, capture_output=True, text=True, check=True,
    )
    lines = result.stdout.splitlines()
    return tuple(map(int, lines[:3])), [line for line in lines[3:] if line]


@pytest.mark.parametrize("slug,registry,conservative", [
    ("qwen3_6_35b_a3b", "control", True),
    ("qwen3_6_35b_a3b", "v3", False),
    ("qwen3_6_35b_a3b", None, False),
    ("deepseek_r1_distill_qwen_32b", "v3", True),
    ("deepseek_r1_distill_llama_70b", "control", True),
    ("gemma4_12b", "control", False),
    ("nemotron_3_nano_4b", "control", False),
])
def test_qwen_control_cuda_mitigation_preserves_other_runtime_defaults(slug, registry, conservative):
    limits, flags = runtime_settings(slug, registry)
    if conservative:
        assert limits == (32, 4096, 16)
        assert flags == ["--enforce-eager", "--no-async-scheduling"]
    else:
        assert limits == (512, 32768, 64)
        assert flags == []


def test_qwen_control_respects_explicit_runtime_limits():
    limits, flags = runtime_settings(
        "qwen3_6_35b_a3b", "control", VLLM_MAX_NUM_SEQS="8",
        VLLM_MAX_NUM_BATCHED_TOKENS="2048", GENERATION_MAX_WORKERS="4",
    )
    assert limits == (8, 2048, 4)
    assert flags == ["--enforce-eager", "--no-async-scheduling"]
