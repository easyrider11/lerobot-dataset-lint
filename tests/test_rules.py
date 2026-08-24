"""One corrupted fixture per rule; a clean fixture must produce zero problems."""

from __future__ import annotations

from conftest import build_dataset

from lerobot_lint.checks import run_all
from lerobot_lint.findings import Report
from lerobot_lint.loader import open_dataset


def lint(root) -> Report:
    view = open_dataset(str(root), sample=10)
    report = Report(repo=str(root), version=view.version)
    report.episodes_total = len(view.episodes)
    report.episodes_checked = len(view.sampled)
    run_all(view, report)
    return report


def rules(report: Report, severity: str | None = None) -> set[str]:
    return {f.rule for f in report.findings if severity is None or f.severity == severity}


def test_clean_dataset_has_no_problems(tmp_path):
    r = lint(build_dataset(tmp_path / "d"))
    assert r.count("ERROR") == 0, [f.message for f in r.findings]
    assert r.count("WARN") == 0, [f.message for f in r.findings]
    assert rules(r) == {"G001"}  # convention report is informational and always on


def test_total_frames_mismatch(tmp_path):
    assert "M003" in rules(lint(build_dataset(tmp_path / "d", total_frames_off=True)), "ERROR")


def test_stale_stats_dims_like_the_checkpoint_bug(tmp_path):
    assert "M004" in rules(lint(build_dataset(tmp_path / "d", stats_wrong_dims=True)), "ERROR")


def test_stale_stats_envelope(tmp_path):
    assert "D004" in rules(lint(build_dataset(tmp_path / "d", stats_stale_max=True)), "ERROR")


def test_zero_std_dim(tmp_path):
    assert "M005" in rules(lint(build_dataset(tmp_path / "d", stats_zero_std=True)), "WARN")


def test_empty_task_string(tmp_path):
    assert "M006" in rules(lint(build_dataset(tmp_path / "d", empty_task=True)), "WARN")


def test_unknown_task(tmp_path):
    assert "E004" in rules(lint(build_dataset(tmp_path / "d", unknown_task=True)), "ERROR")


def test_short_and_long_episodes(tmp_path):
    r = lint(build_dataset(tmp_path / "d", lengths=(5, 300, 20)))
    assert "E001" in rules(r, "WARN")   # 5 frames @ 10 fps < 1 s
    assert "E002" in rules(r, "WARN")   # 300 > 10x median


def test_global_index_gap(tmp_path):
    assert "E003" in rules(lint(build_dataset(tmp_path / "d", index_gap=True)), "ERROR")


def test_missing_video_file(tmp_path):
    assert "V001" in rules(lint(build_dataset(tmp_path / "d", missing_video=True)), "ERROR")


def test_video_span_shorter_than_frames(tmp_path):
    assert "V002" in rules(lint(build_dataset(tmp_path / "d", short_video_span=True)), "ERROR")


def test_nan_in_action(tmp_path):
    assert "D001" in rules(lint(build_dataset(tmp_path / "d", nan_action=True)), "ERROR")


def test_frame_index_not_contiguous(tmp_path):
    assert "D002" in rules(lint(build_dataset(tmp_path / "d", shuffled_frame_index=True)), "ERROR")


def test_dropped_frames_timestamp_gap(tmp_path):
    assert "D003" in rules(lint(build_dataset(tmp_path / "d", timestamp_gap=True)), "WARN")


def test_mixed_gripper_convention_is_an_error(tmp_path):
    r = lint(build_dataset(tmp_path / "d", mixed_gripper=True))
    assert "G001" in rules(r, "ERROR")


def test_consistent_gripper_is_reported_as_info(tmp_path):
    r = lint(build_dataset(tmp_path / "d"))
    g = [f for f in r.findings if f.rule == "G001"]
    assert len(g) == 1 and g[0].severity == "INFO" and "binary{-1,+1}" in g[0].message


def test_frozen_action_episode(tmp_path):
    assert "D006" in rules(lint(build_dataset(tmp_path / "d", frozen_action=True)), "WARN")


def test_saturated_action_dim(tmp_path):
    assert "D005" in rules(lint(build_dataset(tmp_path / "d", saturated_action=True)), "WARN")


def test_exit_codes(tmp_path):
    clean = lint(build_dataset(tmp_path / "a"))
    assert clean.exit_code() == 0 and clean.exit_code(strict=True) == 0
    warn = lint(build_dataset(tmp_path / "b", stats_zero_std=True))
    assert warn.exit_code() == 0 and warn.exit_code(strict=True) == 1
    err = lint(build_dataset(tmp_path / "c", nan_action=True))
    assert err.exit_code() == 1


def test_libero_style_frame_task_association_is_not_a_warning(tmp_path):
    from conftest import strip_episode_tasks

    root = build_dataset(tmp_path / "d")
    strip_episode_tasks(root)
    r = lint(root)
    assert r.count("ERROR") == 0 and r.count("WARN") == 0, [f.message for f in r.findings]
    assert "G002" in rules(r, "INFO")


def test_out_of_range_frame_task_index(tmp_path):
    from conftest import corrupt_frame_task_index, strip_episode_tasks

    root = build_dataset(tmp_path / "d")
    strip_episode_tasks(root)
    corrupt_frame_task_index(root)
    assert "E004" in rules(lint(root), "ERROR")


def test_v21_per_episode_stats_aggregation():
    """Exact combined std: two episodes with known moments."""
    import numpy as np

    from lerobot_lint.loader import aggregate_stats

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0])
    full = np.concatenate([a, b])

    def st(x):
        return {"min": [x.min()], "max": [x.max()], "mean": [x.mean()],
                "std": [x.std()], "count": [len(x)]}

    agg = aggregate_stats([{"action": st(a)}, {"action": st(b)}])["action"]
    assert np.isclose(agg["mean"][0], full.mean())
    assert np.isclose(agg["std"][0], full.std())
    assert agg["min"][0] == 1.0 and agg["max"][0] == 20.0 and agg["count"] == [6]
