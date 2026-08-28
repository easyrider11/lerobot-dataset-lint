"""lerobot-lint <repo_id_or_path> - contract linter for LeRobot datasets."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .checks import run_all
from .findings import Report
from .loader import open_dataset


def _detect_kind(target: str, revision: str | None) -> str:
    """A dataset has meta/info.json; a checkpoint has a top-level config.json."""
    from pathlib import Path

    local = Path(target).expanduser()
    if local.is_dir():
        if (local / "meta" / "info.json").exists():
            return "dataset"
        return "checkpoint" if (local / "config.json").exists() else "dataset"
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        files = set(api.list_repo_files(target, repo_type="dataset", revision=revision))
        if "meta/info.json" in files:
            return "dataset"
    except Exception:
        pass
    return "checkpoint"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="lerobot-lint",
        description="Lint a LeRobot dataset (local path or hub repo id) for "
                    "contract violations: convention drift, dropped frames, "
                    "stale stats, dead recordings.")
    ap.add_argument("target", help="hub repo id or local dir: a DATASET "
                                   "(lerobot/pusht) or a policy CHECKPOINT "
                                   "(lerobot/smolvla_libero) - auto-detected")
    ap.add_argument("--kind", choices=["auto", "dataset", "checkpoint"], default="auto",
                    help="force the target kind (default: auto-detect)")
    ap.add_argument("--sample", type=int, default=50,
                    help="episodes to deep-check at frame level (default 50, spread evenly)")
    ap.add_argument("--revision", default=None, help="hub revision/branch")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    report = Report(repo=args.target)
    kind = args.kind
    if kind == "auto":
        kind = _detect_kind(args.target, args.revision)
    if kind == "checkpoint":
        from .checkpoint import CheckpointView, check_checkpoint

        try:
            view = CheckpointView(args.target, revision=args.revision)
        except Exception as exc:
            report.add("K000", "ERROR", f"cannot open checkpoint: {type(exc).__name__}: {exc}")
            print(report.to_json() if args.json else report.render())
            return 1
        check_checkpoint(view, report)
        print(report.to_json() if args.json else report.render())
        return report.exit_code(strict=args.strict)
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
