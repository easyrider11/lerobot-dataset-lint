"""Ecosystem sweep: how many public lerobot checkpoints ship self-contradicting
feature specs (huggingface/lerobot#4517)?

Metadata only - config.json, processor json/stats (KB), safetensors headers
via Range requests. Unauthenticated; polite pacing; 429s recorded, not hidden.

    .venv/bin/python studies/checkpoint-sweep/sweep.py 300
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lerobot_lint.checkpoint import CheckpointView, check_checkpoint  # noqa: E402
from lerobot_lint.findings import Report  # noqa: E402

OUT = Path(__file__).parent / "results.jsonl"


def main(limit: int) -> None:
    api = HfApi()
    print(f"listing up to {limit} models with library=lerobot ...")
    models = list(api.list_models(filter="lerobot", sort="downloads", limit=limit))
    print(f"got {len(models)}")
    done = set()
    if OUT.exists():
        done = {json.loads(line)["repo"] for line in OUT.read_text().splitlines() if line}
        print(f"resuming: {len(done)} already recorded")
    consecutive_errors = 0
    with OUT.open("a") as out:
        for i, m in enumerate(models):
            if m.id in done:
                continue
            row = {"repo": m.id, "downloads": getattr(m, "downloads", None)}
            try:
                report = Report(repo=m.id)
                view = CheckpointView(m.id)
                facts = check_checkpoint(view, report)
                row.update(facts)
                row["rules"] = sorted({f.rule for f in report.findings})
                consecutive_errors = 0
            except Exception as exc:
                row["classification"] = "FETCH_ERROR"
                row["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
                consecutive_errors += 1
                if "429" in str(exc) or "rate" in str(exc).lower():
                    print("  rate limited - backing off 60 s"); time.sleep(60)
            out.write(json.dumps(row) + "\n"); out.flush()
            if i % 20 == 0:
                print(f"  [{i}/{len(models)}] {m.id} -> {row['classification']}")
            if consecutive_errors >= 15:
                print("aborting: 15 consecutive fetch errors"); break
            time.sleep(0.3)
    summarize()


def summarize() -> None:
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line]
    by = {}
    for r in rows:
        by.setdefault(r["classification"], []).append(r)
    print(f"\n═══ {len(rows)} checkpoints ═══")
    for k in sorted(by, key=lambda k: -len(by[k])):
        print(f"  {k:15s} {len(by[k])}")
    verified = [r for r in rows if r["classification"] in ("CLEAN", "CONTRADICTORY", "SUSPECT")]
    bad = [r for r in rows if r["classification"] == "CONTRADICTORY"]
    sus = [r for r in rows if r["classification"] == "SUSPECT"]
    if verified:
        print(f"\n  contradiction rate among VERIFIABLE: {len(bad)}/{len(verified)} "
              f"= {len(bad)/len(verified):.1%}  (+{len(sus)} suspect)")
    ft = [r for r in verified if r.get("finetuned_from")]
    ft_bad = [r for r in ft if r["classification"] == "CONTRADICTORY"]
    scratch = [r for r in verified if not r.get("finetuned_from")]
    scratch_bad = [r for r in scratch if r["classification"] == "CONTRADICTORY"]
    if ft:
        print(f"  among FINETUNES (train_config has pretrained_path): "
              f"{len(ft_bad)}/{len(ft)} = {len(ft_bad)/len(ft):.1%}")
    if scratch:
        print(f"  among from-scratch: {len(scratch_bad)}/{len(scratch)} "
              f"= {len(scratch_bad)/len(scratch):.1%}")
    for r in bad[:15]:
        print(f"    CONTRADICTORY: {r['repo']}  (type={r.get('policy_type')}, "
              f"state cfg={r.get('state_cfg')} vs stats={r.get('state_stats')}, "
              f"from={str(r.get('finetuned_from'))[:40]})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
