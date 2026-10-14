"""
scoring/llm_policy_runner.py

Run one CSV job through the three-pass LLM judge pipeline.

Usage
-----
python -m scoring.llm_policy_runner \\
  --input artifacts/phase1_results_qwen2_7b.csv \\
  --job-id phase1_qwen2_7b \\
  --output-dir artifacts/llm_policy_labels/phase1_qwen2_7b \\
  --model Qwen2.5-72B-Instruct \\
  --base-url http://localhost:8000/v1 \\
  --batch-size 128 \\
  --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pathlib
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set

import pandas as pd

from models.vllm_client import VLLMClient
from scoring.llm_policy_prompts import (
    get_model_output,
    load_default_prompts,
    load_prompt,
)
from scoring.llm_policy_judge import judge_row
from scoring.llm_policy_adjudicate import resolve_first_pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Column names added to labeled.csv ──────────────────────────────────────
LLM_COLS = [
    "llm_policy_label", "llm_secondary_label", "llm_confidence",
    "llm_resolution_method", "llm_num_agree", "llm_disagreement_type",
    "llm_needs_human_audit", "llm_evidence", "llm_reason",
    "llm_judge_model", "llm_adjudicator_model",
    "llm_schema_version", "llm_prompt_set_version", "llm_parse_error",
]

SCHEMA_VERSION = "v1"
PROMPT_SET_VERSION = "v1"


def _row_hash(model_output: str) -> str:
    return hashlib.sha256(model_output.encode("utf-8")).hexdigest()


def _load_completed_hashes(judge_votes_path: pathlib.Path) -> Set[str]:
    """Return row hashes that already have 3 complete first-pass votes."""
    if not judge_votes_path.exists():
        return set()
    try:
        df = pd.read_csv(judge_votes_path)
        counts = df.groupby("row_hash")["judge_id"].nunique()
        return set(counts[counts >= 3].index.tolist())
    except Exception:
        return set()


def _identity(row, index):
    request_id = row.get("request_id")
    return "request:" + str(request_id) if request_id else "row:" + str(index)


def _saved_identity(row, use_request_ids):
    if use_request_ids:
        return "request:" + row["request_id"] if row.get("request_id") else None
    return "row:" + str(row["row_index"]) if row.get("row_index") is not None else None


def _csv_records(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_completed_resolutions(judge_votes_path, adj_path, df, unresolved_label=None):
    """Restore each request's votes and saved adjudication without LLM calls."""
    from collections import Counter, defaultdict
    from scoring.llm_policy_adjudicate import _needs_adjudication
    from scoring.llm_policy_schema import VALID_LABELS

    use_ids = "request_id" in df.columns
    groups, adjudications = defaultdict(list), defaultdict(list)
    for vote in _csv_records(judge_votes_path):
        key = (_saved_identity(vote, use_ids), vote.get("row_hash"))
        groups[key].append(vote)
    for row in _csv_records(adj_path):
        key = (_saved_identity(row, use_ids), row.get("row_hash"))
        adjudications[key].append(row)
    completed = {}
    for idx, row in df.iterrows():
        try:
            text = get_model_output(row.to_dict())
        except ValueError:
            continue
        key = (_identity(row, idx), _row_hash(text))
        by_judge = {v["judge_id"]: v for v in groups[key] if v.get("judge_id") in {"A", "B", "C"}}
        if set(by_judge) != {"A", "B", "C"}:
            continue
        votes = [dict(by_judge[jid]) for jid in "ABC"]
        try:
            for vote in votes:
                vote["confidence"] = float(vote.get("confidence") or 0)
                vote["secondary_label"] = vote.get("secondary_label") or None
        except (TypeError, ValueError):
            continue  # An interrupted CSV append is not a completed vote set.
        needs_adj, _ = _needs_adjudication(votes)
        if not needs_adj:
            resolution, _ = resolve_first_pass(
                row_index=idx, job_id=votes[0].get("job_id", ""), model_output=text,
                votes=votes, prompts={}, client=None, model="",
                unresolved_label=unresolved_label,
            )
        else:
            labels = "|".join(v["primary_label"] for v in votes)
            saved = [a for a in adjudications[key] if a.get("original_vote_labels") == labels]
            if not saved:
                continue  # The previous job stopped before adjudication finished.
            adj = saved[-1]
            label = adj["final_label"]
            rep = next((v for v in votes if v["primary_label"] == label), votes[0])
            panel = [adj.get(f"adjudicator_{i}_label") for i in range(1, 4)]
            resolution = {
                "llm_policy_label": label, "llm_secondary_label": rep.get("secondary_label"),
                "llm_confidence": 0.0 if label == unresolved_label else rep["confidence"],
                "llm_resolution_method": "adjudication",
                "llm_num_agree": Counter(panel)[label], "llm_disagreement_type": "adjudicated",
                "llm_needs_human_audit": str(adj.get("needs_human_audit", "")).lower() == "true",
                "llm_evidence": "" if label == unresolved_label else rep.get("evidence", ""),
                "llm_reason": ("Automated adjudication did not reach a majority."
                               if label == unresolved_label else rep.get("reason", "")),
            }
        resolution["llm_parse_error"] = any(v.get("parse_error") for v in votes) or (
            needs_adj and any(adj.get(f"adjudicator_{i}_label") == "parse_error" for i in range(1, 4))
        )
        if resolution["llm_policy_label"] in VALID_LABELS or (
            unresolved_label is not None and resolution["llm_policy_label"] == unresolved_label
        ):
            completed[idx] = resolution
    return completed


def _load_resolution_checkpoint(path, df, unresolved_label=None):
    """Replay exact saved resolutions using request identity AND answer hash."""
    from scoring.llm_policy_schema import VALID_LABELS
    if not path.exists():
        return {}
    saved = {}
    lines = path.read_bytes().splitlines(keepends=True)
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if number != len(lines) - 1:
                raise ValueError("Corrupt resolution checkpoint; invalid interior record")
            # A killed process can leave one unfinished append. Preserve all
            # complete records and remove that tail before appending new ones.
            with path.open("r+b") as stream:
                stream.truncate(sum(len(part) for part in lines[:number]))
            break
        saved[(record["identity"], record["row_hash"])] = record["resolution"]
        if number == len(lines) - 1 and not line.endswith(b"\n"):
            # JSON may be complete even when the final newline was interrupted.
            with path.open("ab") as stream:
                stream.write(b"\n")
    complete = {}
    for idx, row in df.iterrows():
        try:
            text = get_model_output(row.to_dict())
        except ValueError:
            continue
        resolution = saved.get((_identity(row, idx), _row_hash(text)))
        if resolution and (
            resolution.get("llm_policy_label") in VALID_LABELS
            or (unresolved_label is not None and resolution.get("llm_policy_label") == unresolved_label)
        ):
            complete[idx] = resolution
    return complete


def _exclusive_job(function):
    import fcntl
    from functools import wraps

    @wraps(function)
    def locked(**kwargs):
        directory = pathlib.Path(kwargs["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".judge.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another judge job is already writing {directory}") from exc
            handlers = set(logger.handlers)
            try:
                return function(**kwargs)
            finally:
                for handler in list(logger.handlers):
                    if handler not in handlers:
                        logger.removeHandler(handler)
                        handler.close()
    return locked


def _append_rows(path: pathlib.Path, rows: List[Dict], fieldnames: List[str]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _build_audit_sample(labeled_df: pd.DataFrame, n_per_stratum: int = 50) -> pd.DataFrame:
    """Build a stratified audit sample."""
    parts = []
    for label in labeled_df["llm_policy_label"].dropna().unique():
        subset = labeled_df[labeled_df["llm_policy_label"] == label]
        parts.append(subset.sample(min(n_per_stratum, len(subset)), random_state=42))
    # Extra: disagreement, low-confidence, human-audit flagged
    disagree = labeled_df[labeled_df["llm_disagreement_type"].isin(
        ["soft_2_1", "hard_2_1", "1_1_1", "adjudicated"])]
    parts.append(disagree.sample(min(100, len(disagree)), random_state=42))
    low_conf = labeled_df[labeled_df["llm_confidence"].fillna(1.0) < 0.70]
    parts.append(low_conf.sample(min(100, len(low_conf)), random_state=42))
    human_audit = labeled_df[labeled_df["llm_needs_human_audit"] == True]
    parts.append(human_audit.sample(min(100, len(human_audit)), random_state=42))
    parts.append(labeled_df.sample(min(100, len(labeled_df)), random_state=99))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).drop_duplicates()


@_exclusive_job
def run_job(
    *,
    input_path: str | pathlib.Path,
    job_id: str,
    output_dir: str | pathlib.Path,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_a: Optional[str] = None,
    prompt_b: Optional[str] = None,
    prompt_c: Optional[str] = None,
    adjudicator_prompt: Optional[str] = None,
    batch_size: int = 128,
    temperature: float = 0.0,
    max_tokens: int = 250,
    max_workers: int = 16,
    resume: bool = True,
    limit: Optional[int] = None,
    sample_strategy: Optional[str] = None,
    stratify_by: Optional[str] = None,
    require_complete: bool = False,
    human_audit_sample: bool = True,
    unresolved_label: Optional[str] = None,
) -> None:
    input_path = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── File paths ────────────────────────────────────────────────────────
    labeled_path       = output_dir / "labeled.csv"
    labels_only_path   = output_dir / "labels_only.csv"
    votes_path         = output_dir / "judge_votes.csv"
    adj_path           = output_dir / "adjudication_votes.csv"
    manifest_path      = output_dir / "manifest.json"
    log_path           = output_dir / "run.log"
    parse_errors_path  = output_dir / "parse_errors.csv"
    audit_path         = output_dir / "audit_sample.csv"
    summary_label_path = output_dir / "summary_by_label.csv"
    summary_dis_path   = output_dir / "summary_by_disagreement.csv"
    resolutions_path  = output_dir / "resolutions.jsonl"
    settings_path     = output_dir / "settings.json"

    # Add file handler to logger for this job
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)

    logger.info("Job %s started.  Input: %s", job_id, input_path)
    t0 = time.time()

    # ── Load prompts ──────────────────────────────────────────────────────
    prompts = load_default_prompts()
    if prompt_a:
        prompts["A"] = load_prompt(prompt_a)
    if prompt_b:
        prompts["B"] = load_prompt(prompt_b)
    if prompt_c:
        prompts["C"] = load_prompt(prompt_c)
    if adjudicator_prompt:
        prompts["adjudicator"] = load_prompt(adjudicator_prompt)

    settings = {
        "job_id": job_id, "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "unresolved_label": unresolved_label,
        "human_audit_sample": human_audit_sample,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": {key: _row_hash(value) for key, value in prompts.items()},
    }
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Judge model/settings/prompts changed; use a separate output directory")
    if not settings_path.exists() and votes_path.exists():
        if manifest_path.exists() and json.loads(manifest_path.read_text()).get("model") != model:
            raise ValueError("Saved votes belong to a different judge model")
        for key, filename in [("A", "judge_prompt_A.txt"), ("B", "judge_prompt_B.txt"),
                              ("C", "judge_prompt_C.txt"), ("adjudicator", "adjudicator_prompt.txt")]:
            previous = output_dir / filename
            if previous.exists() and previous.read_text().strip() != prompts[key].strip():
                raise ValueError("Saved judge prompts changed; use a separate output directory")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # Save prompt copies
    for key, fname in [("A", "judge_prompt_A.txt"), ("B", "judge_prompt_B.txt"),
                       ("C", "judge_prompt_C.txt"), ("adjudicator", "adjudicator_prompt.txt")]:
        (output_dir / fname).write_text(prompts[key])

    # ── Load input CSV ────────────────────────────────────────────────────
    df = pd.read_csv(input_path, keep_default_na=False, dtype={"request_id": str})
    if "request_id" in df.columns and (df["request_id"].eq("").any() or df["request_id"].duplicated().any()):
        raise ValueError("Judge input must have unique, nonempty request_id values")
    logger.info("Loaded %d rows from %s", len(df), input_path)

    # Sampling / limit
    if sample_strategy == "stratified" and stratify_by and limit:
        cols = [c.strip() for c in stratify_by.split(",") if c.strip() in df.columns]
        if cols:
            df = (df.groupby(cols, group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), max(1, limit // df[cols].drop_duplicates().shape[0])), random_state=42)))
            df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
    elif limit:
        df = df.head(limit)

    # Resume: skip completed rows and replay their resolutions into all_resolutions
    all_resolutions: Dict[int, Dict] = {}
    checkpoints = _load_resolution_checkpoint(resolutions_path, df, unresolved_label)
    if resume:
        all_resolutions = _load_completed_resolutions(
            votes_path, adj_path, df, unresolved_label
        )
        all_resolutions.update(checkpoints)
        logger.info("Replayed %d completed request resolutions.", len(all_resolutions))

    # ── Client ────────────────────────────────────────────────────────────
    client = VLLMClient(
        model_name=model,
        base_url=base_url,
        max_retries=2,
        enable_cache=False,
    )

    # ── Vote column definitions ───────────────────────────────────────────
    vote_fields = [
        "job_id", "row_index", "row_hash", "judge_id", "judge_prompt_variant",
        "judge_model", "primary_label", "secondary_label", "confidence",
        "contains_answer", "contains_refusal", "contains_clarifying_question",
        "contains_safe_redirect", "mentions_instruction_priority",
        "treats_external_text_as_data", "evidence", "reason",
        "raw_json", "parse_error",
    ]
    adj_fields = [
        "job_id", "row_index", "row_hash", "original_vote_labels",
        "adjudicator_1_label", "adjudicator_2_label", "adjudicator_3_label",
        "final_label", "final_confidence", "adjudication_reason", "needs_human_audit",
    ]
    parse_err_fields = ["job_id", "row_index", "row_hash", "judge_id", "parse_error", "model_output_snippet"]
    if "request_id" in df.columns:
        for fields in (vote_fields, adj_fields, parse_err_fields):
            fields.append("request_id")

    # ── Per-row processor ─────────────────────────────────────────────────
    def process_row(idx_row):
        idx, row = idx_row
        row_dict = row.to_dict()
        try:
            model_output = get_model_output(row_dict)
        except ValueError as e:
            logger.warning("Row %d: skipped — %s", idx, e)
            return idx, None, None, None

        if idx in all_resolutions:
            return idx, None, None, None  # already done

        votes = judge_row(
            row_index=idx,
            job_id=job_id,
            model_output=model_output,
            prompts=prompts,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        resolution, adj_rows = resolve_first_pass(
            row_index=idx,
            job_id=job_id,
            model_output=model_output,
            votes=votes,
            prompts=prompts,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            unresolved_label=unresolved_label,
        )
        if "request_id" in row_dict:
            for saved in [*votes, *adj_rows]:
                saved["request_id"] = row_dict["request_id"]
        resolution["llm_parse_error"] = any(v.get("parse_error") for v in votes) or any(
            adj.get(f"adjudicator_{i}_label") == "parse_error"
            for adj in adj_rows for i in range(1, 4)
        )

        return idx, votes, resolution, adj_rows

    # ── Main loop ─────────────────────────────────────────────────────────
    rows_to_process = [(idx, row) for idx, row in df.iterrows() if idx not in all_resolutions]
    logger.info("%d rows to process.", len(rows_to_process))

    parse_error_rows: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_row, item): item[0] for item in rows_to_process}
        done = 0
        for fut in as_completed(futures):
            try:
                idx, votes, resolution, adj_rows = fut.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            done += 1
            if votes is None:
                continue

            # Write votes incrementally
            _append_rows(votes_path, votes, vote_fields)
            if adj_rows:
                _append_rows(adj_path, adj_rows, adj_fields)

            # Collect parse errors
            for v in votes:
                if v.get("parse_error"):
                    row_dict = df.iloc[idx].to_dict()
                    snippet = str(row_dict.get("model_output", ""))[:120]
                    parse_error_rows.append({
                        "job_id": job_id, "row_index": idx,
                        "row_hash": v["row_hash"], "judge_id": v["judge_id"],
                        "parse_error": v["parse_error"], "model_output_snippet": snippet,
                        **({"request_id": row_dict["request_id"]} if "request_id" in row_dict else {}),
                    })

            all_resolutions[idx] = resolution
            with resolutions_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "identity": _identity(df.loc[idx], idx), "row_hash": votes[0]["row_hash"],
                    "resolution": resolution,
                }) + "\n")

            if done % 500 == 0:
                logger.info("Progress: %d / %d rows processed.", done, len(rows_to_process))

    # ── Build labeled.csv ─────────────────────────────────────────────────
    result_df = df.copy()
    for col in LLM_COLS:
        result_df[col] = None

    for idx, res in all_resolutions.items():
        for col, val in res.items():
            result_df.at[idx, col] = val
        result_df.at[idx, "llm_judge_model"] = model
        result_df.at[idx, "llm_adjudicator_model"] = model
        result_df.at[idx, "llm_schema_version"] = SCHEMA_VERSION
        result_df.at[idx, "llm_prompt_set_version"] = PROMPT_SET_VERSION
        result_df.at[idx, "llm_parse_error"] = res.get("llm_parse_error", False)

    result_df.to_csv(labeled_path, index=False)
    logger.info("Saved labeled.csv (%d rows).", len(result_df))

    # labels_only.csv
    id_cols = [c for c in ["request_id", "item_id", "condition_id", "provider", "row_index"] if c in result_df.columns]
    result_df[id_cols + LLM_COLS].to_csv(labels_only_path, index=False)

    # parse_errors.csv
    _append_rows(parse_errors_path, parse_error_rows, parse_err_fields)

    # summaries
    if "llm_policy_label" in result_df.columns:
        result_df["llm_policy_label"].value_counts().rename_axis("label").reset_index(name="count").to_csv(
            summary_label_path, index=False)
        result_df["llm_disagreement_type"].value_counts().rename_axis("type").reset_index(name="count").to_csv(
            summary_dis_path, index=False)

    # Optional human-audit sample. Composition runs disable this and preserve
    # unresolved rows for a future study instead of creating an audit queue.
    if human_audit_sample:
        try:
            audit_df = _build_audit_sample(result_df)
            audit_df.to_csv(audit_path, index=False)
            logger.info("Audit sample: %d rows.", len(audit_df))
        except Exception as e:
            logger.warning("Audit sample failed: %s", e)

    # manifest
    elapsed = time.time() - t0
    from scoring.llm_policy_schema import VALID_LABELS
    valid_rows = int(result_df["llm_policy_label"].isin(VALID_LABELS).sum())
    unresolved_rows = int(
        result_df["llm_policy_label"].eq(unresolved_label).sum()
    ) if unresolved_label else 0
    completed_rows = valid_rows + unresolved_rows
    manifest = {
        "job_id": job_id,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "model": model,
        "schema_version": SCHEMA_VERSION,
        "prompt_set_version": PROMPT_SET_VERSION,
        "total_rows": len(df),
        "processed_rows": len(all_resolutions),
        "parse_errors": len(parse_error_rows),
        "valid_labeled_rows": valid_rows,
        "unresolved_rows": unresolved_rows,
        "human_audit_sample_created": human_audit_sample,
        "complete": completed_rows == len(df),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "resume_identity": "request_id" if "request_id" in df.columns else "row_index_and_answer_hash",
        "elapsed_seconds": round(elapsed, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Job %s finished in %.0fs: %d policy labels, %d unresolved, %d total.",
                job_id, elapsed, valid_rows, unresolved_rows, len(df))
    logger.removeHandler(fh)
    fh.close()
    if require_complete and completed_rows != len(df):
        raise RuntimeError(
            f"Judge incomplete: {completed_rows}/{len(df)} policy labels or unresolved rows; "
            "resubmit with --resume"
        )


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run one CSV through the LLM policy judge.")
    p.add_argument("--input", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--prompt-a")
    p.add_argument("--prompt-b")
    p.add_argument("--prompt-c")
    p.add_argument("--adjudicator-prompt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-complete", action="store_true", help="Fail if any input lacks a valid policy label")
    p.add_argument("--no-human-audit-sample", action="store_true",
                   help="Do not create the stratified audit_sample.csv artifact.")
    p.add_argument("--unresolved-label", choices=["unresolved"],
                   help="Store cases without an adjudication majority as unresolved.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-strategy", default=None)
    p.add_argument("--stratify-by", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_job(
        input_path=args.input,
        job_id=args.job_id,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        prompt_c=args.prompt_c,
        adjudicator_prompt=args.adjudicator_prompt,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        resume=args.resume,
        require_complete=args.require_complete,
        limit=args.limit,
        sample_strategy=args.sample_strategy,
        stratify_by=args.stratify_by,
        human_audit_sample=not args.no_human_audit_sample,
        unresolved_label=args.unresolved_label,
    )
