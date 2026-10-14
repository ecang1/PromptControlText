"""Blindly label a composition-generation CSV with the existing policy judge."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scoring.llm_policy_runner import run_job


PAPER_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXPECTED_CONDITIONS_PER_ITEM = 67


def validate_composition_results(input_path: str | Path) -> pd.DataFrame:
    """Reject incomplete or mock generation files before any judge calls."""
    path = Path(input_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "benchmark", "item_id", "condition_id", "model_output",
        "item_stratum", "registry_sha256", "item_manifest_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Composition input is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Composition input contains no rows.")
    if frame["model_output"].str.strip().eq("").any():
        raise ValueError("Composition input contains empty model outputs.")
    if frame["model_output"].eq("Mocked response").any():
        raise ValueError("Mock generation outputs cannot be sent to the policy judge.")
    if frame["item_manifest_sha256"].nunique() != 1 or not frame["item_manifest_sha256"].iloc[0]:
        raise ValueError("Composition input must come from one fixed, hashed item manifest.")
    if frame["registry_sha256"].nunique() != 1 or not frame["registry_sha256"].iloc[0]:
        raise ValueError("Composition input must use one frozen prompt registry.")

    key = ["benchmark", "item_id", "condition_id"]
    if frame.duplicated(key).any():
        raise ValueError("Composition input contains duplicate item-condition rows.")
    condition_sets = frame.groupby(["benchmark", "item_id"])["condition_id"].agg(set)
    if condition_sets.map(len).ne(EXPECTED_CONDITIONS_PER_ITEM).any():
        raise ValueError(f"Each item must have exactly {EXPECTED_CONDITIONS_PER_ITEM} conditions.")
    canonical_conditions = condition_sets.iloc[0]
    if condition_sets.map(lambda values: values == canonical_conditions).eq(False).any():
        raise ValueError("Items do not share the same complete condition set.")
    return frame


def score_composition(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = PAPER_JUDGE_MODEL,
    base_url: str = "http://localhost:8000/v1",
    max_workers: int = 16,
    resume: bool = False,
    limit: int | None = None,
) -> None:
    """Run three blind labels and automated adjudication for composition rows."""
    validate_composition_results(input_path)
    run_job(
        input_path=input_path,
        job_id="composition_v1_pilot",
        output_dir=output_dir,
        model=model,
        base_url=base_url,
        max_workers=max_workers,
        resume=resume,
        limit=limit,
        require_complete=True,
        human_audit_sample=False,
        unresolved_label="unresolved",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blindly score a composition result CSV with the paper's policy judge."
    )
    parser.add_argument("--input", required=True, help="Composition generation CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for judge outputs.")
    parser.add_argument("--model", default=PAPER_JUDGE_MODEL)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional leading-row subset for a small pilot.")
    args = parser.parse_args()
    score_composition(
        args.input,
        args.output_dir,
        model=args.model,
        base_url=args.base_url,
        max_workers=args.max_workers,
        resume=args.resume,
        limit=args.limit,
    )
