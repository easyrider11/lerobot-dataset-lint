"""Show the raw `names` behind a sweep verdict, to audit the N-rules for
false positives before quoting numbers.

    .venv/bin/python studies/names-sweep/explain.py N005        # by rule
    .venv/bin/python studies/names-sweep/explain.py mismatch 15 # by status, max 15 datasets

Reads results.jsonl, re-reads meta/info.json (already in the HF cache after
a sweep) and prints feature, dtype, shape and the names exactly as stored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lerobot_lint.names import classify, is_vector_feature  # noqa: E402

OUT = Path(__file__).parent / "results.jsonl"
RULE_STATUS = {"N001": "mismatch", "N002": "duplicate", "N003": "placeholder",
               "N004": "absent", "N005": "invalid"}


def describe(repo: str, info: dict, status: str, width: int = 160) -> list[str]:
    """Pure: one line per vector feature of `info` whose status matches."""
    lines = []
    for key, ft in (info.get("features") or {}).items():
        if not isinstance(ft, dict) or not is_vector_feature(key, ft):
            continue
        fact = classify(key, ft)
        if fact.status != status:
            continue
        raw = json.dumps(ft.get("names"))
        if len(raw) > width:
            raw = raw[: width - 3] + "..."
        lines.append(f"{repo}  {key}  {ft.get('dtype')} shape={ft.get('shape')}  names={raw}")
    return lines


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    status = RULE_STATUS.get(argv[0], argv[0])
    limit = int(argv[1]) if len(argv) > 1 else 40
    from huggingface_hub import hf_hub_download

    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line]
    hits = [r for r in rows if any(f["status"] == status for f in r.get("features", []))]
    print(f"{len(hits)} dataset(s) with a '{status}' feature; showing {min(limit, len(hits))}")
    for r in hits[:limit]:
        path = hf_hub_download(r["repo"], "meta/info.json", repo_type="dataset")
        for line in describe(r["repo"], json.loads(Path(path).read_text()), status):
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
