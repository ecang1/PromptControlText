import os
import unicodedata
import argparse
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from core.schema import EvalItem
from benchmarks.xstest import load_xstest
from benchmarks.harmbench import load_harmbench
from benchmarks.iheval import load_iheval
from prompts.registry import load_registry, iter_prompt_triples
from models.vllm_client import VLLMClient, ServerUnavailableError
from models.client import LLMClient
from scoring.harmbench_scorer import parse_harmbench_response
from scoring.hierarchy_scorer import parse_hierarchy_response

BENCHMARK_SCORERS = {
    "HarmBench": parse_harmbench_response,
    "XSTest": parse_harmbench_response,
    "IHEval": parse_hierarchy_response,
}

# ---------------------------------------------------------------------------
# Output quality validation
# ---------------------------------------------------------------------------

def _is_valid_output(text: str) -> tuple[bool, str]:
    """
    Return (is_valid, reason) for a raw model output string.

    Catches two known failure modes:
      1. Literal ``<tool_call>`` tokens — the model serialised a function call
         instead of producing a plain-text response.
      2. Garbled Unicode — >10 % of characters carry combining diacritics,
         which indicates the model hallucinated character substitutions on
         adversarial / malformed inputs.
    """
    if "<tool_call>" in text:
        return False, "tool_call_token"
    # Count characters that carry combining diacritics after NFD decomposition
    nfd = unicodedata.normalize("NFD", text)
    combining = sum(1 for c in nfd if unicodedata.category(c) == "Mn")
    if len(text) > 0 and combining / len(nfd) > 0.10:
        return False, "garbled_unicode"
    return True, ""

def score_item(benchmark: str, response: str):
    scorer = BENCHMARK_SCORERS.get(benchmark, parse_harmbench_response)
    return scorer(response)

def _generate_one(client, item: EvalItem, family: str, variant: str, prompt_text: str, generator_model: str, clarity: str = None):
    """Single unit of work: one (item, prompt_family, prompt_variant) triple."""
    try:
        output, metadata = client.generate(
            system_prompt=prompt_text,
            user_prompt=item.input_text,
            model=generator_model,
            temperature=0.0
        )
    except ServerUnavailableError:
        raise
    except Exception as e:
        print(f"Failed generation for {item.item_id} [{family}/{variant}]: {e}")
        return None

    valid, invalid_reason = _is_valid_output(output)
    scores = score_item(item.benchmark, output) if valid else {}

    record = item.model_dump()
    record["prompt_family"] = family
    record["prompt_variant"] = variant
    record["clarity_level"] = clarity
    record["model_name"] = generator_model
    record["model_output"] = output
    record["malformed_output"] = "" if valid else invalid_reason
    record["metadata"] = metadata
    for k, v in scores.items():
        record[f"{k}_score"] = v

    return record

def _generation_key(record):
    return (
        record["benchmark"], record["item_id"], record["prompt_family"],
        record.get("clarity_level") or None, record["prompt_variant"],
    )


def _load_generation_checkpoint(filepath, generator_model, tasks):
    """Keep saved nonempty responses only when they match the current run."""
    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    required = {
        "benchmark", "item_id", "prompt_family", "clarity_level",
        "prompt_variant", "model_name", "model_output", "input_text",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot resume: checkpoint is missing columns {sorted(missing)}")
    expected = {
        (item.benchmark, item.item_id, family, clarity, variant): item.input_text
        for item, family, clarity, variant, _ in tasks
    }
    records = []
    completed = set()
    seen = set()
    for record in df.to_dict(orient="records"):
        key = _generation_key(record)
        if record["model_name"] != generator_model:
            raise ValueError("Cannot resume: checkpoint belongs to a different generator model")
        if key not in expected or record["input_text"] != expected[key]:
            raise ValueError(f"Cannot resume: checkpoint condition does not match this run: {key}")
        if key in seen:
            raise ValueError(f"Cannot resume: duplicate checkpoint condition: {key}")
        seen.add(key)
        if record["model_output"].strip():
            records.append(record)
            completed.add(key)
    return records, completed


def run_experiment(output_filepath: str, generator_model: str = "Qwen2.5-72B-Instruct", mock_mode: bool = False, limit: int = 0, max_workers: int = 32, data_dir: str = None, registry_version: str = "v3", resume: bool = False, base_url: str = "http://localhost:8000/v1", safety_only: bool = False):
    output_filepath = os.fspath(output_filepath)
    print("Loading Registry...")
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "benchmarks", "artifacts", "datasets")
    data_dir = os.path.abspath(data_dir)
    print(f"Dataset directory: {data_dir}")

    _REGISTRY_FILES = {
        "v3":      "registry_v3.yaml",
        "control": "registry_control.yaml",
    }
    if registry_version not in _REGISTRY_FILES:
        raise ValueError(f"Unknown registry_version '{registry_version}'. Choose from: {list(_REGISTRY_FILES)}")
    registry_path = os.path.join(base_dir, "prompts", _REGISTRY_FILES[registry_version])
    print(f"Registry: {registry_path}")

    # Load registry
    registry = load_registry(registry_path)
    
    print("Loading Benchmarks...")
    items: List[EvalItem] = []
    
    # Real dataset paths (populated by benchmarks/download_data.py)
    xstest_real_path   = os.path.join(data_dir, "xstest_prompts.csv")
    xstest_mock_path   = os.path.join(base_dir, "tests", "fixtures", "xstest_mock.csv")
    harmbench_path     = os.path.join(data_dir, "harmbench_behaviors.csv")
    iheval_path        = os.path.join(data_dir, "iheval.csv")

    if mock_mode:
        harmbench_path = os.path.join(base_dir, "tests", "fixtures", "harmbench_mock.csv")

    # Prefer real XSTest when available; fall back to mock fixture for tests
    if mock_mode:
        xstest_path = xstest_mock_path
    elif os.path.exists(xstest_real_path):
        xstest_path = xstest_real_path
    elif os.path.exists(xstest_mock_path):
        print("Warning: real xstest_prompts.csv not found; using mock fixture. "
              "Run benchmarks/download_data.py to get the full dataset.")
        xstest_path = xstest_mock_path
    else:
        xstest_path = None

    if xstest_path and os.path.exists(xstest_path):
        items.extend(load_xstest(xstest_path))
    else:
        print("Warning: No XSTest data found. Run benchmarks/download_data.py")

    if os.path.exists(harmbench_path):
        items.extend(load_harmbench(harmbench_path))
    else:
        print(f"Warning: {harmbench_path} not found. Run download_data.py")

    if os.path.exists(iheval_path):
        items.extend(load_iheval(iheval_path))
    else:
        print(f"Warning: {iheval_path} not found. Run download_data.py")
        
    if safety_only:
        items = [item for item in items if item.benchmark in {"XSTest", "HarmBench"}]
        print("Safety-only run: retaining XSTest and HarmBench; excluding IHEval.")
    if not items:
        raise ValueError("No benchmark items loaded.")

    if limit and limit > 0:
        items = items[:limit]

    print(f"Loaded {len(items)} items. Setting up model {generator_model}...")
    if mock_mode:
         client = LLMClient(mock_mode=True, mock_response="I cannot help with that.", enable_cache=True)
    else:
         client = VLLMClient(model_name=generator_model, base_url=base_url, enable_cache=True)

    # Build the full list of (item, family, clarity, variant, prompt_text) tasks.
    # iter_prompt_triples() handles the v3 crossed design and control schema.
    tasks = []
    for item in items:
        for family, clarity, variant, prompt_text in iter_prompt_triples(registry):
            tasks.append((item, family, clarity, variant, prompt_text))

    planned_total = len(tasks)
    evaluated_records = []
    if resume and os.path.isfile(output_filepath):
        evaluated_records, completed_keys = _load_generation_checkpoint(
            output_filepath, generator_model, tasks
        )
        tasks = [
            task for task in tasks
            if (task[0].benchmark, task[0].item_id, task[1], task[2], task[3])
            not in completed_keys
        ]
        print(f"Resuming: preserving {len(evaluated_records)} saved responses; "
              f"{len(tasks)} conditions remain.")
        if not tasks:
            print("All generation conditions are already complete.")
            return
    total = len(tasks)
    print(f"Submitting {total} generation tasks with max_workers={max_workers}...")

    completed = 0

    # Checkpoint every CHECKPOINT_EVERY completed futures (saves partial progress
    # so a SLURM timeout doesn't lose everything).
    CHECKPOINT_EVERY = 500
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

    def _checkpoint(records, filepath):
        if records:
            # A killed job must not truncate its last complete checkpoint.
            temp_path = filepath + ".tmp"
            pd.DataFrame(records).to_csv(temp_path, index=False)
            os.replace(temp_path, filepath)

    server_error = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Keep only one request per worker in flight. A dead server must not
        # trigger retries for every remaining condition in a large run.
        pending = set()
        task_iterator = iter(tasks)

        def _submit_next():
            task = next(task_iterator, None)
            if task is not None:
                item, family, clarity, variant, prompt_text = task
                pending.add(executor.submit(
                    _generate_one, client, item, family, variant, prompt_text,
                    generator_model, clarity
                ))

        for _ in range(min(max_workers, total)):
            _submit_next()

        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.remove(future)
                    record = future.result()
                    completed += 1
                    if record is not None:
                        evaluated_records.append(record)
                    if completed % CHECKPOINT_EVERY == 0 or completed == total:
                        print(f"  Progress: {completed}/{total} attempted — checkpointing {len(evaluated_records)} records")
                        _checkpoint(evaluated_records, output_filepath)
                    _submit_next()
        except ServerUnavailableError as exc:
            server_error = exc
            for future in pending:
                future.cancel()

    if server_error is not None:
        # The executor has finished the small number of in-flight requests.
        # Retain any successful responses received during shutdown as well.
        for future in pending:
            if future.cancelled():
                continue
            try:
                record = future.result()
            except Exception:
                continue
            if record is not None:
                evaluated_records.append(record)
        _checkpoint(evaluated_records, output_filepath)
        raise RuntimeError(
            f"vLLM server unreachable: saved {len(evaluated_records)}/{planned_total} "
            "responses and stopped generation. Check the server logs, then "
            "resubmit with --resume."
        ) from server_error

    print(f"Saving {len(evaluated_records)} evaluations to {output_filepath}")
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    _checkpoint(evaluated_records, output_filepath)
    nonempty = sum(bool(record["model_output"].strip()) for record in evaluated_records)
    if nonempty != planned_total:
        raise RuntimeError(
            f"Generation incomplete: {nonempty}/{planned_total} nonempty responses saved. "
            "Resubmit with --resume to retry missing or empty responses."
        )
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model", type=str, default="Qwen2.5-72B-Instruct")
    parser.add_argument("--output-file", type=str, default="artifacts/phase1_results.csv")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of benchmark items to evaluate.")
    parser.add_argument("--mock", action="store_true", help="Use mock client and fixture paths for testing.")
    parser.add_argument("--resume", action="store_true", help="Keep matching saved responses and generate only missing or empty conditions.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Local vLLM endpoint; use separate ports for jobs sharing a node.")
    parser.add_argument("--safety-only", action="store_true", help="Generate only XSTest/HarmBench answers for refusal/compliance analysis.")
    parser.add_argument("--max-workers", type=int, default=32, help="Number of concurrent generation threads.")
    parser.add_argument("--data-dir", type=str, default=None, help="Absolute path to dataset directory (overrides default).")
    parser.add_argument("--registry-version", type=str, default="v3",
                        choices=["v3", "control"],
                        help="Use the paper's v3 prompt registry or the no-prompt control.")
    
    args = parser.parse_args()
    
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base, args.output_file)
    run_experiment(out, args.generator_model, mock_mode=args.mock, limit=args.limit, max_workers=args.max_workers, data_dir=args.data_dir, registry_version=args.registry_version, resume=args.resume, base_url=args.base_url, safety_only=args.safety_only)
