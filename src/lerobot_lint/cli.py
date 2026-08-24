"""lerobot-lint <repo_id_or_path> - contract linter for LeRobot datasets."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .checks import run_all
from .findings import Report
from .loader import open_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="lerobot-lint",
        description="Lint a LeRobot dataset (local path or hub repo id) for "
                    "contract violations: convention drift, dropped frames, "
                    "stale stats, dead recordings.")
    ap.add_argument("target", help="hub repo id (e.g. lerobot/pusht) or local dataset dir")
    ap.add_argument("--sample", type=int, default=50,
                    help="episodes to deep-check at frame level (default 50, spread evenly)")
    ap.add_argument("--revision", default=None, help="hub revision/branch")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    report = Report(repo=args.target)
    try:
        view = open_dataset(args.target, sample=args.sample, revision=args.revision)
    except FileNotFoundError as exc:
        report.add("M001", "ERROR", str(exc))
        print(report.to_json() if args.json else report.render())
        return 1
    report.version = view.version
    report.episodes_total = len(view.episodes)
    report.episodes_checked = len(view.sampled)
    report.notes.append(f"frame-level checks on {len(view.sampled)} sampled episode(s); "
                        f"videos checked by listing + index, never decoded")
    if view.stats_source:
        report.notes.append(f"stats: {view.stats_source}")
    run_all(view, report)
    print(report.to_json() if args.json else report.render())
    return report.exit_code(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
