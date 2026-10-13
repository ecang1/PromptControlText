"""Recalculate first-pass policy agreement from saved votes, without inference.

python -m scoring.inter_judge_agreement
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import logging
from pathlib import Path

from scoring.llm_policy_adjudicate import _needs_adjudication
from scoring.llm_policy_prompts import get_model_output
from scoring.llm_policy_run_jobs import load_jobs
from scoring.llm_policy_schema import VALID_LABELS

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))
LABELS = ("compliance", "clarification", "refusal", "hierarchy_preservation",
          "source_isolation", "safe_redirection")
MODEL_NAMES = dict(zip(
    ("qwen25_72b", "llama31_8b", "llama33_70b", "qwen2_7b", "gemma4_12b", "gemma4_31b",
     "qwen3_6_35b_a3b", "nemotron_3_nano_4b", "deepseek_r1_distill_qwen_32b",
     "deepseek_r1_distill_llama_70b", "frontier_main_openai", "frontier_main_anthropic", "frontier_main_google"),
    ("Qwen-2.5-72B", "Llama-3.1-8B", "Llama-3.3-70B", "Qwen-2-7B", "Gemma-4-12B", "Gemma-4-31B",
     "Qwen-3.6-35B-A3B", "Nemotron-3-Nano-4B", "DeepSeek-R1-Distill-Qwen-32B",
     "DeepSeek-R1-Distill-Llama-70B", "GPT-6-Astra", "Claude-Opus-5.5", "Gemini-3.1-Pro")))
LOG = logging.getLogger(__name__)


def agreement(counts):
    """Nominal Cohen kappa from a multiclass paired-label confusion table."""
    n = sum(counts.values())
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    left, right = Counter(), Counter()
    for (a, b), count in counts.items():
        left[a] += count
        right[b] += count
    observed = sum(count for (a, b), count in counts.items() if a == b) / n
    expected = sum(left[label] * right[label] for label in left.keys() | right.keys()) / n ** 2
    return {"n": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def truth(value):
    return str(value).lower() in {"true", "1"}


def identity(row, use_ids, index=None):
    value = row.get("request_id") if use_ids else (index if index is not None else row["row_index"])
    if value is None or str(value) == "":
        raise ValueError("Missing saved judgment identity")
    return str(value)


def read_records(path, provenance):
    """Stream CSV records while hashing the exact bytes and checking stability."""
    before = path.stat()
    hasher = hashlib.sha256()
    def lines(raw):
        for line in raw:
            hasher.update(line)
            yield line.decode("utf-8")
    with path.open("rb") as raw:
        for row in csv.DictReader(lines(raw)):
            if None in row:
                raise ValueError(f"Malformed CSV: {path}")
            yield row
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during audit: {path}")
    provenance[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = hasher.hexdigest()


def load_votes(directory, job_id, use_ids, provenance):
    """Use the latest saved A/B/C vote per identity AND answer hash, as resume does."""
    votes, adjudications = defaultdict(dict), {}
    counts = Counter()
    for row in read_records(directory / "judge_votes.csv", provenance):
        if row["job_id"] != job_id or row["judge_id"] not in "ABC" or len(row["judge_id"]) != 1:
            raise ValueError(f"Foreign first-pass vote in {job_id}")
        key = (identity(row, use_ids), row["row_hash"])
        judge = row["judge_id"]
        counts["saved_first_pass_vote_records"] += 1
        counts["superseded_first_pass_votes"] += judge in votes[key]
        votes[key][judge] = (row["primary_label"], float(row.get("confidence") or 0), row.get("parse_error", ""))
    for row in read_records(directory / "adjudication_votes.csv", provenance):
        if row["job_id"] != job_id:
            raise ValueError(f"Foreign adjudication in {job_id}")
        key = (identity(row, use_ids), row["row_hash"], row["original_vote_labels"])
        counts["saved_adjudication_records"] += 1
        counts["superseded_adjudications"] += key in adjudications
        adjudications[key] = row
    return votes, adjudications, counts


class Summary:
    def __init__(self):
        self.counts = Counter()
        self.pairs = {"".join(pair): Counter() for pair in PAIRS}
        self.common_pairs = {"".join(pair): Counter() for pair in PAIRS}
        self.labels = {label: Counter() for label in LABELS}
        self.triggers = Counter()

    def add(self, final, votes, adjudication):
        self.counts["input_rows"] += 1
        label = final.get("llm_policy_label", "")
        method = final.get("llm_resolution_method", "")
        self.counts["final_valid_label" if label in VALID_LABELS else "final_invalid_label" if label else "final_missing_label"] += 1
        self.counts["saved_resolution_" + (method or "missing")] += 1
        self.counts["human_audit_flag"] += truth(final.get("llm_needs_human_audit"))
        self.counts["saved_pipeline_error_flag"] += truth(final.get("llm_parse_error"))
        if label in VALID_LABELS:
            self.labels[label]["n_final_label"] += 1
        if not votes:
            self.counts["no_first_pass_votes"] += 1
            return
        self.counts["outputs_with_votes"] += 1
        valid = {judge: vote[0] for judge, vote in votes.items()
                 if vote[0] in VALID_LABELS and not vote[2]}
        for a, b in PAIRS:
            if a in valid and b in valid:
                self.pairs[a + b][valid[a], valid[b]] += 1
        if set(votes) != set("ABC"):
            self.counts["incomplete_first_pass_panel"] += 1
            return
        self.counts["complete_first_pass_panel"] += 1
        all_valid = len(valid) == 3
        self.counts["all_three_valid" if all_valid else "first_pass_failed_outputs"] += 1
        self.counts["first_pass_failed_votes"] += 3 - len(valid)
        unanimous = all_valid and len(set(valid.values())) == 1
        self.counts["first_pass_unanimous"] += unanimous
        self.counts["first_pass_nonunanimous_valid"] += all_valid and not unanimous
        if all_valid:
            for a, b in PAIRS:
                self.common_pairs[a + b][valid[a], valid[b]] += 1
        if label in VALID_LABELS:
            self.labels[label]["all_three_valid"] += all_valid
            self.labels[label]["unanimous_any_label"] += unanimous
            self.labels[label]["unanimous_final_label"] += unanimous and valid["A"] == label
        records = [{"primary_label": votes[j][0], "confidence": votes[j][1],
                    "parse_error": votes[j][2] or ("invalid_label" if votes[j][0] not in VALID_LABELS else "")}
                   for j in "ABC"]
        needed, trigger = _needs_adjudication(records)
        self.counts["adjudication_required_by_rule"] += needed
        if needed:
            self.triggers[trigger] += 1
        self.counts["unanimous_sent_to_adjudication"] += unanimous and needed
        self.counts["eligible_for_automatic_unanimity"] += unanimous and not needed
        self.counts["eligible_for_automatic_majority"] += not unanimous and not needed
        self.counts["automatically_resolved_unanimous"] += unanimous and not needed and method == "unanimous" and label in VALID_LABELS
        self.counts["automatically_resolved_majority"] += not unanimous and not needed and method == "majority_vote" and label in VALID_LABELS
        self.counts["automatic_rule_without_saved_resolution"] += not needed and (not method or label not in VALID_LABELS)
        if not adjudication:
            self.counts["required_but_no_saved_adjudication"] += needed
            return
        self.counts["saved_current_adjudication"] += 1
        self.counts["adjudication_without_rule_trigger"] += not needed
        panel = [adjudication.get(f"adjudicator_{i}_label", "") for i in range(1, 4)]
        good = [vote for vote in panel if vote in VALID_LABELS]
        self.counts["adjudication_failed_outputs"] += len(good) != 3
        self.counts["adjudication_failed_votes"] += 3 - len(good)
        majority = max(Counter(good).values(), default=0) >= 2
        self.counts["adjudication_no_valid_majority"] += not majority
        self.counts["adjudication_no_majority_but_valid_final_label"] += not majority and label in VALID_LABELS
        self.counts["adjudication_final_label_mismatch"] += adjudication["final_label"] != label

    def merge(self, other):
        self.counts.update(other.counts)
        self.triggers.update(other.triggers)
        for pair in self.pairs:
            self.pairs[pair].update(other.pairs[pair])
            self.common_pairs[pair].update(other.common_pairs[pair])
        for label in LABELS:
            self.labels[label].update(other.labels[label])

    def report(self):
        pairs = {pair: agreement(counts) for pair, counts in self.pairs.items()}
        common = {pair: agreement(counts) for pair, counts in self.common_pairs.items()}
        refusal = {}
        for pair, counts in self.pairs.items():
            binary = Counter()
            for (a, b), n in counts.items():
                binary[a == "refusal", b == "refusal"] += n
            refusal[pair] = agreement(binary)
        def mean(values):
            kappas = [value["kappa"] for value in values.values() if value["kappa"] is not None]
            return sum(kappas) / len(kappas) if len(kappas) == 3 else None
        return {"counts": dict(self.counts), "pairwise": pairs, "mean_pairwise_kappa": mean(pairs),
                "policy_refusal_vs_other_pairwise": refusal, "policy_refusal_vs_other_mean_kappa": mean(refusal),
                "complete_valid_triplets_pairwise": common,
                "complete_valid_triplets_mean_kappa": mean(common),
                "adjudication_triggers": dict(self.triggers),
                "per_label": {label: {**dict(counts), "unanimous_rate": counts["unanimous_final_label"] / counts["n_final_label"]
                                      if counts["n_final_label"] else None} for label, counts in self.labels.items()}}


def analyze_job(job, provenance):
    directory = ROOT / job["output_dir"]
    path = directory / "labeled.csv"
    with path.open(newline="") as stream:
        use_ids = "request_id" in (csv.DictReader(stream).fieldnames or [])
    votes, adjudications, archive_counts = load_votes(directory, job["job_id"], use_ids, provenance)
    summary, seen = Summary(), set()
    for index, row in enumerate(read_records(path, provenance)):
        row_id = identity(row, use_ids, index)
        if row_id in seen:
            raise ValueError(f"Duplicate output identity in {job['job_id']}: {row_id}")
        seen.add(row_id)
        try:
            text = get_model_output(row)
        except ValueError:
            summary.add(row, {}, None)
            summary.counts["empty_output_rows"] += 1
            continue
        key = row_id, hashlib.sha256(text.encode()).hexdigest()
        panel = votes.pop(key, {})
        original_labels = "|".join(panel[j][0] for j in "ABC") if set(panel) == set("ABC") else ""
        saved_adj = adjudications.pop((*key, original_labels), None)
        summary.add(row, panel, saved_adj)
        if not panel:
            summary.counts["nonempty_output_without_votes"] += 1
    summary.counts.update(archive_counts)
    summary.counts["stale_vote_panels_excluded"] += len(votes)
    summary.counts["stale_adjudications_excluded"] += len(adjudications)
    if summary.counts["input_rows"] != job["expected_rows"]:
        raise ValueError(f"Unexpected labeled row count for {job['job_id']}")
    if job.get("expected_sha256") and provenance[str(path.relative_to(ROOT))] != job["expected_sha256"]:
        raise ValueError(f"Judgments differ from compiled dataset: {job['job_id']}")
    return summary


def discover_jobs(compilation):
    jobs = []
    registries = [("llm_policy_jobs.yaml", "original_open_models"),
                  ("llm_policy_jobs_new_models.yaml", "new_open_prompted"),
                  ("llm_policy_jobs_new_controls.yaml", "new_open_controls")]
    for registry, cohort in registries:
        for job in load_jobs(ROOT / "configs" / registry):
            source = compilation["by_source"][job["job_id"]]
            jobs.append({**job, "cohort": cohort, "study": "open_models",
                         "expected_rows": source["rows"], "expected_sha256": source["sha256"]})
    for provider in ("openai", "anthropic", "google"):
        job_id = "frontier_main_" + provider
        directory = "artifacts/llm_policy_labels/" + job_id
        manifest = json.loads((ROOT / directory / "manifest.json").read_text())
        if manifest["job_id"] != job_id or not manifest.get("complete"):
            raise ValueError(f"Incomplete frontier judge: {job_id}")
        jobs.append({"job_id": job_id, "output_dir": directory, "cohort": "frontier_models",
                     "study": "frontier_main", "expected_rows": manifest["total_rows"]})
    expected = compilation["total_rows"] - compilation["by_judge_status"].get("not_applicable_empty_provider_block", 0)
    if sum(job["expected_rows"] for job in jobs) != expected:
        raise ValueError("Judge inputs do not match the compiled dataset universe")
    return jobs


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manuscript(report):
    result = report["overall"]
    c = Counter(result["counts"])
    n = c["outputs_with_votes"]
    def percent(value, denominator=n):
        if not denominator:
            return "undefined"
        percentage = 100 * value / denominator
        return f"{percentage:.3f}%" if 0 < percentage < .1 else f"{percentage:.1f}%"
    pairs = result["pairwise"]
    rows = [
        ("Outputs with saved first-pass judgments", f"{n:,}"),
        ("First-pass prompt variants per output", "3 (A, B, C; same judge checkpoint)"),
        ("Mean pairwise Cohen’s κ", f"{result['mean_pairwise_kappa']:.3f}"),
        *[(f"{pair[0]}–{pair[1]} Cohen’s κ", f"{stat['kappa']:.3f} (n = {stat['n']:,} valid pairs)") for pair, stat in pairs.items()],
        ("First-pass unanimous agreement", f"{percent(c['first_pass_unanimous'])} (n = {c['first_pass_unanimous']:,})"),
        ("Resolved automatically by unanimity", f"{percent(c['automatically_resolved_unanimous'])} (n = {c['automatically_resolved_unanimous']:,})"),
        ("Resolved automatically by two-of-three majority", f"{percent(c['automatically_resolved_majority'])} (n = {c['automatically_resolved_majority']:,})"),
        ("Sent to adjudication (saved current panel)", f"{percent(c['saved_current_adjudication'])} (n = {c['saved_current_adjudication']:,})"),
        ("First-pass outputs with a failed/invalid vote", f"{percent(c['first_pass_failed_outputs'])} (n = {c['first_pass_failed_outputs']:,})"),
        ("Adjudicated outputs with a failed/invalid adjudicator vote", f"{percent(c['adjudication_failed_outputs'])} (n = {c['adjudication_failed_outputs']:,})"),
        ("Adjudications without a valid majority", f"{percent(c['adjudication_no_valid_majority'])} (n = {c['adjudication_no_valid_majority']:,})"),
        ("Saved human-audit flag", f"{percent(c['human_audit_flag'])} (n = {c['human_audit_flag']:,})"),
        ("Invalid final policy label", f"{percent(c['final_invalid_label'])} (n = {c['final_invalid_label']:,})"),
        ("Missing final policy label", f"{percent(c['final_missing_label'])} (n = {c['final_missing_label']:,})"),
    ]
    text = ["# Inter-judge agreement", "",
            "Computed offline from the saved A/B/C first-pass votes and current adjudications. "
            "These are three prompt variants of the same Llama-3.1-8B-Instruct checkpoint, not three independent models. "
            "Each response occurrence is counted separately, including identical text in different conditions. "
            "Resumed append-only files use the latest vote per response identity, answer hash, and judge variant. "
            "Provider blocks without text do not enter the judge denominator.", "",
            f"Across {n:,} outputs with saved first-pass judgments, mean pairwise nominal Cohen’s κ was "
            f"{result['mean_pairwise_kappa']:.3f}. Pairwise kappas exclude failed/invalid votes separately for each pair. "
            f"First-pass unanimity occurred for {percent(c['first_pass_unanimous'])} of outputs "
            f"(n = {c['first_pass_unanimous']:,}); {c['saved_current_adjudication']:,} outputs "
            f"({percent(c['saved_current_adjudication'])}) had a saved adjudicator panel. "
            "Unanimity and automatic resolution are distinct: low-confidence sensitive labels can be adjudicated, "
            "whereas a two-of-three majority with low dissent confidence can be accepted automatically.", "",
            "| Metric | Value |", "|---|---:|", *[f"| {key} | {value} |" for key, value in rows], "",
            "Table 5. Overall agreement and adjudication statistics. Error counts denote response occurrences, "
            "not individual API attempts, and first-pass/adjudicator error categories can overlap. "
            f"The full table contains {report['compiled_total_rows']:,} rows; "
            f"{report['empty_provider_blocks_not_submitted']:,} empty frontier blocks and "
            f"{c['no_first_pass_votes']:,} legacy rows without matching saved first-pass votes are outside the agreement denominator. "
            f"{c['automatic_rule_without_saved_resolution']:,} unanimous panels and "
            f"{c['adjudication_final_label_mismatch']:,} saved adjudications are not reflected in a final label in the current combined data. "
            "Those data are reported as saved, without filling missing labels.", "",
            "| Final policy label | n | First-pass unanimous agreement with final label |", "|---|---:|---:|"]
    for label in LABELS:
        values = result["per_label"][label]
        text.append(f"| {label.upper()} | {values.get('n_final_label', 0):,} | {100 * values['unanimous_rate']:.1f}% |")
    text += ["", "Table 6. Denominator: outputs assigned each valid final policy label. "
             "Numerator: all three valid first-pass votes equal that final label. "
             "Missing/invalid final labels are excluded from this table, but retained in coverage/error totals.", "",
             "| Study | Outputs with votes | Mean pairwise κ | First-pass unanimity | Adjudication |", "|---|---:|---:|---:|---:|"]
    for study, values in report["by_study"].items():
        counts = Counter(values["counts"])
        total = counts["outputs_with_votes"]
        text.append(f"| {study} | {total:,} | {values['mean_pairwise_kappa']:.3f} | "
                    f"{percent(counts['first_pass_unanimous'], total)} | {percent(counts['saved_current_adjudication'], total)} |")
    text += ["", "The mean is the arithmetic mean of the three pooled pairwise kappas, not an average of per-job kappas. "
             "The JSON also reports kappas restricted to the common subset with three valid votes. "
             "No-majority adjudications can retain a fallback label and should not be described as necessarily missing final labels. "
             "Agreement measures consistency among prompt variants, not human validation or correctness. "
             "The v2 lexical refusal classifier is a separate endpoint and is not used in these policy-agreement calculations.", ""]
    return "\n".join(text)


def run(summary_path, output_dir):
    csv.field_size_limit(16 * 1024 * 1024)
    compilation = json.loads(summary_path.read_text())
    overall, studies, cohorts, models, by_job = Summary(), defaultdict(Summary), defaultdict(Summary), defaultdict(Summary), {}
    provenance = {}
    prompt_hashes, judge_models = defaultdict(set), set()
    for job in discover_jobs(compilation):
        LOG.info("Auditing %s", job["job_id"])
        directory = ROOT / job["output_dir"]
        judge_models.add(json.loads((directory / "manifest.json").read_text())["model"])
        for role in "ABC":
            prompt_hashes[role].add(hashlib.sha256((directory / f"judge_prompt_{role}.txt").read_text().strip().encode()).hexdigest())
        result = analyze_job(job, provenance)
        overall.merge(result)
        studies[job["study"]].merge(result)
        cohorts[job["cohort"]].merge(result)
        model = MODEL_NAMES[job["job_id"].removeprefix("phase1_").removeprefix("control_")]
        models[model].merge(result)
        by_job[job["job_id"]] = result.report()
        LOG.info("%s: %s", job["job_id"], dict(result.counts))
    if len(judge_models) != 1 or any(len(values) != 1 for values in prompt_hashes.values()):
        raise ValueError("Judge checkpoints or A/B/C prompts differ between jobs; report separate protocols")
    report = {"overall": overall.report(), "by_study": {key: value.report() for key, value in studies.items()},
              "by_cohort": {key: value.report() for key, value in cohorts.items()}, "by_job": by_job,
              "by_model": {key: value.report() for key, value in models.items()},
              "judge_models": sorted(judge_models), "judge_prompt_sha256": {key: next(iter(values)) for key, values in prompt_hashes.items()},
              "compilation_summary": str(summary_path), "compiled_total_rows": compilation["total_rows"],
              "empty_provider_blocks_not_submitted": compilation["by_judge_status"].get("not_applicable_empty_provider_block", 0),
              "source_sha256": provenance,
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "manuscript.md").write_text(manuscript(report))
    write_csv(output_dir / "per_label.csv", [{"label": key, **value} for key, value in report["overall"]["per_label"].items()])
    pair_rows, cohort_rows = [], []
    for scope, values in [("overall", {"all": report["overall"]}), ("study", report["by_study"]),
                          ("cohort", report["by_cohort"]), ("model", report["by_model"]), ("job", by_job)]:
        for group, value in values.items():
            cohort_rows.append({"scope": scope, "group": group, "mean_pairwise_kappa": value["mean_pairwise_kappa"], **value["counts"]})
            for pair, stat in value["pairwise"].items():
                pair_rows.append({"scope": scope, "group": group, "target": "six_policy_labels", "pair": pair, **stat})
            for pair, stat in value["policy_refusal_vs_other_pairwise"].items():
                pair_rows.append({"scope": scope, "group": group, "target": "policy_refusal_vs_other", "pair": pair, **stat})
    write_csv(output_dir / "pairwise.csv", pair_rows)
    write_csv(output_dir / "summary_by_group.csv", cohort_rows)
    write_csv(output_dir / "confusion_counts.csv", [{"pair": pair, "left": a, "right": b, "n": n}
              for pair, counts in overall.pairs.items() for (a, b), n in sorted(counts.items())])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts/phase1_results_combined_labeled_heuristic_v2.summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/inter_judge_agreement")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run(args.summary, args.output_dir)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
