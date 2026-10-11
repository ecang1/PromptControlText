"""Validate lexical refusal scoring against published XSTest human labels.

No inference calls. First download the pinned public data with --download, then
rerun offline. Development/holdout splitting groups the same item across models.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from scoring.harmbench_scorer import parse_harmbench_response
from scoring.safety_heuristic import classify_response

ROOT = Path(__file__).resolve().parents[1]
XSTEST_REVISION = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
MODELS = ("gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct")
SPLIT_SEED = "heuristic-v2-dev-v1"


def metrics(gold, predictions):
    counts = Counter(zip(gold, predictions))
    tn, fp, fn, tp = (counts[p] for p in ((0, 0), (0, 1), (1, 0), (1, 1)))
    n = len(gold)
    if not n:
        return {"n": 0}
    accuracy = (tp + tn) / n
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n**2
    return {"n": n, "accuracy": accuracy,
            "kappa": (accuracy - expected) / (1 - expected) if expected != 1 else None,
            "refusal_precision": tp / (tp + fp) if tp + fp else None,
            "refusal_recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "confusion_matrix_rows_gold_cols_pred_0_1": [[tn, fp], [fn, tp]]}


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        filename = f"xstest_v2_completions_{model}.csv"
        url = f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REVISION}/model_completions/{filename}"
        with urlopen(url, timeout=30) as stream:
            data = stream.read()
        path = directory / filename
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Existing file differs from pinned upstream: {path}")
        path.write_bytes(data)


def evaluate(directory):
    datasets, sources = {}, []
    for model in MODELS:
        path = directory / f"xstest_v2_completions_{model}.csv"
        if not path.exists():
            raise ValueError(f"Missing {path}; run with --download once")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 450 or len({r["id"] for r in rows}) != 450:
            raise ValueError(f"Expected 450 distinct XSTest IDs in {path}")
        if any(r["final_label"] not in {"1_full_compliance", "2_full_refusal", "3_partial_refusal"} for r in rows):
            raise ValueError(f"Unknown human label in {path}")
        datasets[model] = rows
        sources.append({"model": model, "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "url": f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REVISION}/model_completions/{path.name}"})
    ids = {r["id"] for r in datasets["gpt4"]}
    if any({r["id"] for r in rows} != ids for rows in datasets.values()):
        raise ValueError("Published model response sets must have identical item IDs")
    ranked = sorted(ids, key=lambda item_id: hashlib.sha256((SPLIT_SEED + "/" + item_id).encode()).hexdigest())
    development, holdout = set(ranked[:315]), set(ranked[315:])
    report = {"target": "Expressed refusal: human full or partial refusal versus full compliance",
              "reference": "Published human final_label, not GPT-4 grader outputs",
              "limitation": "Published aggregate metrics and a few error examples were previously inspected. Rules were developed on 315 designated item IDs and frozen before evaluation on the remaining 135; this is internal validation, not a fresh independent corpus. It does not establish GPT-4 grader equivalence or validity on new generators.",
              "upstream_revision": XSTEST_REVISION, "sources": sources,
              "scorer_sha256": {name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                for name, path in (("v1", "scoring/harmbench_scorer.py"), ("v2", "scoring/safety_heuristic.py"))},
              "split": {"seed": SPLIT_SEED, "unit": "item ID shared across all generators",
                        "development_ids": ranked[:315], "holdout_ids": ranked[315:]},
              "by_model": {}, "pooled": {}}
    predictions = []
    for model, rows in datasets.items():
        for row in rows:
            old = parse_harmbench_response(row["completion"])
            new = classify_response(row["completion"])
            predictions.append({"model": model, "id": row["id"], "type": row["type"],
                                "split": "development" if row["id"] in development else "holdout",
                                "gold_class": row["final_label"],
                                "gold_refusal": int(row["final_label"] != "1_full_compliance"),
                                "v1_refusal": int(old["refusal"] == 1), "v1_compliance": int(old["compliance"] == 1),
                                "v2_refusal": int(new["label"] == "refusal"), "v2_label": new["label"],
                                "v2_rules": ";".join(new["rules"]), "completion": row["completion"]})
    for name, subset in [(m, [r for r in predictions if r["model"] == m]) for m in MODELS] + [("pooled", predictions)]:
        result = {}
        for split in ("development", "holdout", "all"):
            selected = [r for r in subset if split == "all" or r["split"] == split]
            result[split] = {version: metrics([r["gold_refusal"] for r in selected],
                                            [r[f"{version}_refusal"] for r in selected]) for version in ("v1", "v2")}
        result["by_prompt_safety"] = {safety: {
            version: metrics([r["gold_refusal"] for r in subset if r["type"].startswith("contrast_") == unsafe],
                             [r[f"{version}_refusal"] for r in subset if r["type"].startswith("contrast_") == unsafe])
            for version in ("v1", "v2")} for safety, unsafe in (("safe", False), ("unsafe", True))}
        result["by_type"] = {kind: {version: metrics([r["gold_refusal"] for r in subset if r["type"] == kind],
                                                   [r[f"{version}_refusal"] for r in subset if r["type"] == kind])
                                  for version in ("v1", "v2")} for kind in sorted({r["type"] for r in subset})}
        if name == "pooled":
            report["pooled"] = result
        else:
            report["by_model"][name] = result
    hb = ROOT / "benchmarks/artifacts/gold_labels/harmbench_text_val.json"
    raw = json.loads(hb.read_text())
    rows = [r for group in raw.values() for r in group]
    gold = [int(sum(r[f"human_{i}"] for i in range(3)) >= 2) for r in rows]
    report["harmbench_proxy_only"] = {
        "warning": "Human labels indicate harmful-behavior success, not refusal. Lexical compliance is a proxy; these results do not validate a native HarmBench scorer.",
        "source_sha256": hashlib.sha256(hb.read_bytes()).hexdigest(),
        "v1": metrics(gold, [int(parse_harmbench_response(r["generation"])["compliance"] == 1) for r in rows]),
        "v2": metrics(gold, [int(classify_response(r["generation"])["label"] == "compliance") for r in rows])}
    for version in ("v1", "v2"):
        result = report["harmbench_proxy_only"][version]
        result["harmful_success_precision"] = result.pop("refusal_precision")
        result["harmful_success_recall"] = result.pop("refusal_recall")
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xstest-dir", type=Path, default=ROOT / ".local/heuristic-validation/xstest")
    parser.add_argument("--download", action="store_true", help="Download public annotated CSVs only; no model calls")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/scorer_validation_v2.json")
    args = parser.parse_args()
    if args.download:
        download(args.xstest_dir)
    report, predictions = evaluate(args.xstest_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    prediction_path = args.output.with_suffix(".predictions.csv")
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print("Holdout: 135 shared item IDs per model; full and partial refusal grouped.")
    for model, result in report["by_model"].items():
        old, new = result["holdout"]["v1"], result["holdout"]["v2"]
        print(f"{model}: accuracy {old['accuracy']:.1%} -> {new['accuracy']:.1%}; "
              f"kappa {old['kappa']:.3f} -> {new['kappa']:.3f}")
    print("Pooled holdout:", json.dumps(report["pooled"]["holdout"]))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
