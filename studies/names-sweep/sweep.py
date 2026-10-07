"""Ecosystem sweep: can a program tell what each channel of a hub dataset is?
(huggingface/lerobot#4784)

Downloads ONE file per dataset - meta/info.json, a few KB - and classifies
`names` of every vector feature with the N-rules. Unauthenticated works;
set HF_TOKEN for higher rate limits. Resumable: rows already in results.jsonl
are skipped.

    .venv/bin/python studies/names-sweep/sweep.py 500          # sweep + summary
    .venv/bin/python studies/names-sweep/sweep.py --summary    # summary only
    .venv/bin/python studies/names-sweep/sweep.py --reclassify # same datasets, current rules
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lerobot_lint.findings import Report  # noqa: E402
from lerobot_lint.names import check_names, dataset_names_class  # noqa: E402

OUT = Path(__file__).parent / "results.jsonl"
CLASSES = ("semantic", "placeholder", "whole-vector", "axis-names", "absent", "mismatch",
           "duplicate", "invalid", "no-vector-features")


def classify_info(repo: str, info: dict) -> dict:
    """Pure: info.json -> one result row. Tested offline."""
    report = Report(repo=repo)
    facts = check_names(info, report)
    return {
        "repo": repo,
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "classification": dataset_names_class(facts),
        "features": [f.to_dict() for f in facts],
        "rules": sorted({f.rule for f in report.findings}),
    }


def sweep(limit: int) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    print(f"listing up to {limit} datasets tagged LeRobot (by downloads) ...")
    datasets = list(api.list_datasets(filter="LeRobot", sort="downloads", limit=limit))
    print(f"got {len(datasets)}")
    done = set()
    if OUT.exists():
        done = {json.loads(line)["repo"] for line in OUT.read_text().splitlines() if line}
        print(f"resuming: {len(done)} already recorded")
    consecutive_errors = 0
    with OUT.open("a") as out:
        for i, d in enumerate(datasets):
            if d.id in done:
                continue
            try:
                path = hf_hub_download(d.id, "meta/info.json", repo_type="dataset")
                row = classify_info(d.id, json.loads(Path(path).read_text()))
                consecutive_errors = 0
            except Exception as exc:
                row = {"repo": d.id, "classification": "FETCH_ERROR",
                       "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                consecutive_errors += 1
                if "429" in str(exc) or "rate" in str(exc).lower():
                    print("  rate limited - backing off 60 s")
                    time.sleep(60)
            row["downloads"] = getattr(d, "downloads", None)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"  [{i}/{len(datasets)}] {d.id} -> {row['classification']}")
            if consecutive_errors >= 15:
                print("aborting: 15 consecutive fetch errors")
                break
            time.sleep(0.2)


def reclassify() -> None:
    """Re-score every row already in results.jsonl with the current N-rules.
    Same datasets, same order; info.json comes from the HF cache (one network
    call only if it was evicted). Fetch errors stay as they are."""
    from huggingface_hub import hf_hub_download

    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line]
    new = []
    for r in rows:
        if r["classification"] == "FETCH_ERROR":
            new.append(r)
            continue
        path = hf_hub_download(r["repo"], "meta/info.json", repo_type="dataset")
        row = classify_info(r["repo"], json.loads(Path(path).read_text()))
        row["downloads"] = r.get("downloads")
        new.append(row)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in new))
    print(f"reclassified {sum(r['classification'] != 'FETCH_ERROR' for r in new)} rows")


def summarize(rows: list[dict]) -> str:
    """Markdown summary. Shares are over datasets whose info.json was read."""
    ok = [r for r in rows if r["classification"] != "FETCH_ERROR"]
    errors = len(rows) - len(ok)
    by = Counter(r["classification"] for r in ok)
    lines = [f"Datasets read: {len(ok)} (fetch errors: {errors})", "",
             "| names of action / observation.state | datasets | share |",
             "|---|---|---|"]
    for c in CLASSES:
        if by.get(c):
            lines.append(f"| {c} | {by[c]} | {by[c] / len(ok):.1%} |")
    unusable = sum(v for k, v in by.items() if k != "semantic")
    if ok:
        lines += ["", f"Not programmatically usable (anything but semantic): "
                      f"{unusable}/{len(ok)} = {unusable / len(ok):.1%}"]
    rule_hits = Counter(rule for r in ok for rule in r.get("rules", []))
    if rule_hits:
        lines += ["", "Datasets with at least one finding, per rule:", ""]
        lines += [f"- {rule}: {n}" for rule, n in sorted(rule_hits.items())]
    by_robot: dict[str, Counter] = {}
    for r in ok:
        by_robot.setdefault(str(r.get("robot_type")), Counter())[r["classification"]] += 1
    top = sorted(by_robot.items(), key=lambda kv: -sum(kv[1].values()))[:8]
    if top:
        lines += ["", "| robot_type | datasets | semantic |", "|---|---|---|"]
        for robot, c in top:
            n = sum(c.values())
            lines.append(f"| {robot} | {n} | {c['semantic'] / n:.0%} |")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--reclassify":
        reclassify()
    elif argv and argv[0] != "--summary":
        sweep(int(argv[0]))
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line] \
        if OUT.exists() else []
    print(summarize(rows) if rows else "no results yet - run a sweep first")


if __name__ == "__main__":
    main(sys.argv[1:] or ["300"])
