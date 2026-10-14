"""Generate responses for the composition_v1 prompt-module study.

This runner intentionally records raw responses and legacy benchmark scores but
does not assign a single output-policy label. Composition-specific policy labels
must be audited before they are used for confirmatory claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

from benchmarks.harmbench import load_harmbench
from benchmarks.iheval import load_iheval
from benchmarks.xstest import load_xstest
from core.schema import EvalItem
from experiments.run_phase1 import _is_valid_output, score_item
from models.client import LLMClient
from models.vllm_client import ServerUnavailableError, VLLMClient
from prompts.composition import build_conditions, load_composition_registry, registry_sha256

GENERATION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "composition_v1_generation_config.json"
GENERATION_CONFIG = json.loads(GENERATION_CONFIG_PATH.read_text(encoding="utf-8"))
GENERATION_SETTINGS = dict(GENERATION_CONFIG["sampling"])
MAX_GENERATION_RETRIES = int(GENERATION_CONFIG["retry_policy"]["max_retries"])
MAX_GENERATION_ATTEMPTS = int(GENERATION_CONFIG["retry_policy"]["max_attempts_total"])
RETRY_BACKOFF_SECONDS = list(GENERATION_CONFIG["retry_policy"]["backoff_seconds"])


def _generate_one(
    client, item, condition, generator_model, registry_hash,
    item_manifest_hash, prior_attempts=0,
):
    output, metadata, error = "", {}, ""
    attempts = prior_attempts
    server_unavailable_exhausted = False
    while attempts < MAX_GENERATION_ATTEMPTS:
        attempts += 1
        try:
            output, metadata = client.generate(
                system_prompt=condition["system_prompt"],
                user_prompt=item.input_text,
                model=generator_model,
                **GENERATION_SETTINGS,
            )
        except ServerUnavailableError as exc:
            error = f"{type(exc).__name__}: {exc}"
            server_unavailable_exhausted = attempts >= MAX_GENERATION_ATTEMPTS
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            if output and str(output).strip():
                error = ""
                break
            error = "Empty response"

        if attempts >= MAX_GENERATION_ATTEMPTS:
            break
        time.sleep(RETRY_BACKOFF_SECONDS[min(attempts - 1, len(RETRY_BACKOFF_SECONDS) - 1)])

    metadata = dict(metadata)
    metadata["generation_attempts"] = attempts
    metadata["max_generation_retries"] = MAX_GENERATION_RETRIES
    metadata["server_unavailable_exhausted"] = server_unavailable_exhausted

    valid, invalid_reason = _is_valid_output(output)
    scores = score_item(item.benchmark, output) if output and valid and item.benchmark in {"XSTest", "HarmBench", "IHEval"} else {}
    record = item.model_dump()
    record.update({
        "condition_id": condition["condition_id"],
        "condition_type": condition["condition_type"],
        "module_ids": json.dumps(condition["modules"]),
        "module_order": condition["module_order"],
        "module_variants": condition["module_variants"],
        "priority_state": condition["priority_state"],
        "system_prompt": condition["system_prompt"],
        "system_prompt_sha256": hashlib.sha256(
            condition["system_prompt"].encode("utf-8")
        ).hexdigest(),
        "registry_sha256": registry_hash,
        "item_manifest_sha256": item_manifest_hash,
        "item_stratum": (item.metadata or {}).get("stratum", ""),
        "item_source": (item.metadata or {}).get("source", ""),
        "model_name": generator_model,
        "model_output": output,
        "generation_error": error,
        "generation_attempts": attempts,
        "malformed_output": "" if valid else invalid_reason,
        "metadata": json.dumps(metadata, sort_keys=True),
    })
    record.update({f"{name}_score": value for name, value in scores.items()})
    return record


def _condition_key(record):
    return record["benchmark"], record["item_id"], record["condition_id"]


def _load_items(data_dir, base_dir, mock_mode):
    if mock_mode:
        fixture_dir = Path(base_dir) / "tests" / "fixtures"
        return (
            load_xstest(str(fixture_dir / "xstest_mock.csv"))
            + load_harmbench(str(fixture_dir / "harmbench_mock.csv"))
            + load_iheval(str(fixture_dir / "iheval_mock.csv"))
        )

    dataset_dir = Path(data_dir)
    items = []
    for benchmark, filename, loader in (
        ("XSTest", "xstest_prompts.csv", load_xstest),
        ("HarmBench", "harmbench_behaviors.csv", load_harmbench),
        ("IHEval", "iheval.csv", load_iheval),
    ):
        path = dataset_dir / filename
        if path.exists():
            items.extend(loader(str(path)))
        else:
            print(f"Warning: {path} not found; skipping {benchmark}.")
    return items


def _load_item_manifest(path):
    """Load the fixed composition item cohort and verify its source snapshot."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("items"), list):
        raise ValueError(f"Unsupported or invalid item manifest: {manifest_path}")
    for source in manifest.get("sources", []):
        source_path = manifest_path.parent.parent / source["path"]
        if not source_path.is_file():
            raise ValueError(f"Manifest source is missing: {source_path}")
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != source["sha256"]:
            raise ValueError(f"Manifest source changed since freeze: {source_path}")

    items = []
    seen_ids = set()
    strata_counts = {}
    required = {"item_id", "stratum", "benchmark", "domain", "input_text", "gold_label", "source"}
    for row in manifest["items"]:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Manifest item is missing fields {sorted(missing)}")
        if row["item_id"] in seen_ids:
            raise ValueError(f"Duplicate manifest item ID: {row['item_id']}")
        seen_ids.add(row["item_id"])
        strata_counts[row["stratum"]] = strata_counts.get(row["stratum"], 0) + 1
        items.append(EvalItem(
            item_id=row["item_id"], benchmark=row["benchmark"], domain=row["domain"],
            input_text=row["input_text"], gold_label=row["gold_label"],
            metadata={"stratum": row["stratum"], "source": row["source"],
                      "source_item_id": row.get("source_item_id")},
        ))
    if strata_counts != manifest.get("quotas"):
        raise ValueError(
            f"Manifest stratum counts {strata_counts} do not match declared quotas "
            f"{manifest.get('quotas')}"
        )
    return items, hashlib.sha256(manifest_path.read_bytes()).hexdigest(), manifest


def run_composition(
    output_filepath,
    generator_model="Qwen/Qwen2.5-72B-Instruct",
    mock_mode=False,
    limit=0,
    max_workers=int(GENERATION_CONFIG["max_concurrent_sequences"]),
    data_dir=None,
    resume=False,
    base_url="http://localhost:8000/v1",
    manifest_filepath=None,
):
    base_dir = Path(__file__).resolve().parents[1]
    registry_path = base_dir / "prompts" / "composition_registry.json"
    registry = load_composition_registry(registry_path)
    registry_hash = registry_sha256(registry_path)
    conditions = build_conditions(registry)
    data_dir = Path(data_dir) if data_dir else base_dir / "benchmarks" / "artifacts" / "datasets"
    item_manifest_hash = ""
    if manifest_filepath:
        items, item_manifest_hash, manifest = _load_item_manifest(manifest_filepath)
        print(f"Loaded frozen manifest {item_manifest_hash}: {len(items)} items; "
              f"strata {manifest['quotas']}.")
    elif mock_mode:
        items = _load_items(data_dir, base_dir, mock_mode)
    else:
        items = _load_items(data_dir, base_dir, mock_mode)
    if limit > 0:
        items = items[:limit]
    if not items:
        raise ValueError("No benchmark items loaded for the composition study.")
    if not mock_mode and generator_model not in GENERATION_CONFIG["models"]:
        raise ValueError(
            f"Generator model {generator_model!r} is not in the frozen model set: "
            f"{GENERATION_CONFIG['models']}"
        )

    tasks = [(item, condition, 0) for item in items for condition in conditions]
    planned_total = len(tasks)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    output_filepath = os.fspath(output_filepath)
    records = []
    completed = set()
    prior_attempt_counts = {}
    if resume and os.path.isfile(output_filepath):
        old = pd.read_csv(output_filepath, dtype=str, keep_default_na=False)
        required = {
            "benchmark", "item_id", "condition_id", "input_text", "model_name",
            "model_output", "registry_sha256",
            "item_manifest_sha256",
        }
        if required - set(old.columns):
            raise ValueError("Cannot resume: checkpoint schema is incomplete.")
        expected = {
            (item.benchmark, item.item_id, condition["condition_id"]): (
                item.input_text,
                hashlib.sha256(
                    condition["system_prompt"].encode("utf-8")
                ).hexdigest(),
            )
            for item, condition, _ in tasks
        }
        for row in old.to_dict(orient="records"):
            key = _condition_key(row)
            if key not in expected or row["input_text"] != expected[key][0]:
                raise ValueError(f"Cannot resume: unexpected or changed condition {key}.")
            if row["model_name"] != generator_model or row["registry_sha256"] != registry_hash:
                raise ValueError("Cannot resume: model or frozen registry hash differs.")
            if row["item_manifest_sha256"] != item_manifest_hash:
                raise ValueError("Cannot resume: item manifest differs.")
            if row.get("system_prompt_sha256") != expected[key][1]:
                raise ValueError(f"Cannot resume: rendered prompt differs for {key}.")
            if key in completed:
                raise ValueError(f"Cannot resume: duplicate condition {key}.")
            attempts = int(row.get("generation_attempts", 0) or 0)
            if row["model_output"].strip() or attempts >= MAX_GENERATION_ATTEMPTS:
                completed.add(key)
                records.append(row)
            else:
                prior_attempt_counts[key] = attempts
        tasks = [
            (
                item,
                condition,
                prior_attempt_counts.get(
                    (item.benchmark, item.item_id, condition["condition_id"]), 0
                ),
            )
            for item, condition, _ in tasks
            if (item.benchmark, item.item_id, condition["condition_id"]) not in completed
        ]
        print(f"Resuming: {len(records)} saved responses; {len(tasks)} conditions remain.")

    client = (
        LLMClient(mock_mode=True, mock_response="Mocked response", enable_cache=True)
        if mock_mode else VLLMClient(
            model_name=generator_model, base_url=base_url, enable_cache=True,
            max_retries=1, openai_max_retries=0,
            retry_delay=RETRY_BACKOFF_SECONDS[0],
            timeout=float(GENERATION_CONFIG["request_timeout_seconds"]),
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

    def checkpoint():
        if records:
            temp = output_filepath + ".tmp"
            pd.DataFrame(records).to_csv(temp, index=False)
            os.replace(temp, output_filepath)

    total = len(tasks)
    print(f"Composition registry {registry_hash}; {len(conditions)} conditions/item.")
    print(f"Generating {total}/{planned_total} remaining cases with {max_workers} workers.")
    pending = set()
    task_iterator = iter(tasks)
    completed_count = 0
    server_error = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        def submit_next():
            next_task = next(task_iterator, None)
            if next_task is not None:
                item, condition, prior_attempts = next_task
                pending.add(executor.submit(
                    _generate_one, client, item, condition, generator_model, registry_hash,
                    item_manifest_hash, prior_attempts,
                ))

        for _ in range(min(max_workers, total)):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                try:
                    record = future.result()
                    records.append(record)
                    details = json.loads(record.get("metadata") or "{}")
                    if details.get("server_unavailable_exhausted"):
                        server_error = ServerUnavailableError(
                            record.get("generation_error", "Server remained unavailable.")
                        )
                        for waiting in pending:
                            waiting.cancel()
                        break
                except ServerUnavailableError as exc:
                    server_error = exc
                    for waiting in pending:
                        waiting.cancel()
                    break
                completed_count += 1
                if completed_count % 100 == 0 or completed_count == total:
                    print(f"Progress: {completed_count}/{total}; checkpointing.")
                    checkpoint()
                submit_next()
            if server_error:
                break

    if server_error:
        for future in pending:
            if future.cancelled():
                continue
            try:
                records.append(future.result())
            except Exception:
                continue

    checkpoint()
    if server_error:
        raise RuntimeError(
            f"Generation stopped on server unavailability; saved {len(records)} rows. "
            "Resume with --resume after checking the endpoint."
        ) from server_error

    finalized = {
        _condition_key(row) for row in records
        if row.get("model_output", "").strip()
        or int(row.get("generation_attempts", 0) or 0) >= MAX_GENERATION_ATTEMPTS
    }
    if len(finalized) != planned_total:
        raise RuntimeError(
            f"Generation incomplete: {len(finalized)}/{planned_total} cases completed "
            "or exhausted under the retry policy. "
            "Use --resume to retry missing conditions."
        )
    print(f"Saved {len(records)} rows to {output_filepath}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compositional safety-task selection study.")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--output-file", default="artifacts/composition_v1_results.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-workers", type=int,
        default=int(GENERATION_CONFIG["max_concurrent_sequences"]),
    )
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--manifest", default="artifacts/composition_v1_pilot_manifest.json",
        help="Frozen item manifest to use (default: composition pilot manifest).",
    )
    parser.add_argument(
        "--all-benchmarks", action="store_true",
        help="Load all available benchmark rows instead of a fixed item manifest.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_file)
    if not output.is_absolute():
        output = root / output
    run_composition(
        output, args.generator_model, args.mock, args.limit, args.max_workers,
        args.data_dir, args.resume, args.base_url,
        None if args.all_benchmarks else root / args.manifest,
    )
