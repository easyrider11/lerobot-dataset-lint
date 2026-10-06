"""The rules. Each takes (view, report) and appends findings.

Rule numbering: M=metadata, E=episode index, D=frame data, G=gripper, V=video,
N=feature names (names.py).
Every rule exists because something real breaks: the docstrings say what.
"""

from __future__ import annotations

import numpy as np

from .findings import Report
from .loader import DatasetView
from .names import check_names

EPS_STD = 1e-8


# -- metadata ----------------------------------------------------------------


def check_info(view: DatasetView, report: Report) -> None:
    """M001/M002/M008: info.json must carry the keys everything else trusts."""
    info = view.info
    for key in ("fps", "features", "total_episodes", "total_frames"):
        if key not in info:
            report.add("M001", "ERROR", f"meta/info.json missing required key '{key}'")
    fps = info.get("fps") or 0
    if fps and not (0 < float(fps) <= 240):
        report.add("M008", "WARN", f"suspicious fps={fps}")
    v = str(info.get("codebase_version", ""))
    if not (v.startswith("v2") or v.startswith("v3")):
        report.add("M002", "WARN", f"unknown codebase_version '{v}' - linted best-effort")


def check_counts(view: DatasetView, report: Report) -> None:
    """M003: the episode index must agree with the declared totals. A mismatch
    means an interrupted upload or a hand-edited meta - training will either
    crash or silently drop data."""
    info = view.info
    n_eps = len(view.episodes)
    if "total_episodes" in info and n_eps and n_eps != int(info["total_episodes"]):
        report.add("M003", "ERROR",
                   f"episode index has {n_eps} episodes but info.json declares "
                   f"{info['total_episodes']}")
    total_len = sum(e.length for e in view.episodes)
    if "total_frames" in info and n_eps and total_len != int(info["total_frames"]):
        report.add("M003", "ERROR",
                   f"episode lengths sum to {total_len} but info.json declares "
                   f"total_frames={info['total_frames']}")


def check_stats(view: DatasetView, report: Report) -> None:
    """M004/M005: stats are the model's window onto the data - normalization
    reads them, not the parquet. Stale dims (the shape in stats disagreeing
    with the declared feature shape) is exactly the failure found in a
    published policy checkpoint: config said state[6], weights said state[8].
    Zero-std dims make MEAN_STD normalization explode or freeze a dim."""
    if not view.stats:
        report.add("M005", "WARN", "no stats found (neither meta/stats.json nor "
                                   "meta/episodes_stats.jsonl) - normalization must "
                                   "be recomputed by every consumer")
        return
    feats = view.features
    for key, st in view.stats.items():
        if key not in feats or not isinstance(st, dict):
            continue
        declared = feats[key].get("shape") or []
        mean = np.asarray(st.get("mean", []), dtype=float).ravel()
        if feats[key].get("dtype") in ("float32", "float64") and declared:
            want = int(np.prod(declared))
            if mean.size and mean.size != want:
                report.add("M004", "ERROR",
                           f"stats for '{key}' have {mean.size} dims but features "
                           f"declare shape {declared} - stale metadata", feature=key)
        std = np.asarray(st.get("std", []), dtype=float).ravel()
        dead = np.where(std < EPS_STD)[0]
        if std.size and dead.size and feats[key].get("dtype") in ("float32", "float64"):
            report.add("M005", "WARN",
                       f"'{key}' std≈0 in dim(s) {dead.tolist()} - MEAN_STD "
                       f"normalization will blow up or pin these dims", feature=key)
        lo = np.asarray(st.get("min", []), dtype=float).ravel()
        hi = np.asarray(st.get("max", []), dtype=float).ravel()
        if lo.size and hi.size and lo.size == hi.size and np.any(lo > hi):
            report.add("M005", "ERROR", f"'{key}' stats min > max", feature=key)


def check_tasks(view: DatasetView, report: Report) -> None:
    """M006/E004: empty task strings starve language conditioning; episodes
    pointing at tasks that do not exist crash dataloaders."""
    for idx, text in view.tasks.items():
        if not text.strip():
            report.add("M006", "WARN", f"task {idx} is an empty string")
    known = set(view.tasks.values())
    if not view.episode_tasks_present:
        # libero-style: association lives in the per-frame task_index column;
        # its range is validated in check_frames on the sampled episodes
        report.add("G002", "INFO",
                   "episodes index has no 'tasks' column - task association is "
                   "per-frame task_index (libero-style); validated on sampled episodes")
        return
    seen_missing = 0
    for e in view.episodes:
        if not e.tasks:
            report.add("E004", "WARN", "episode has no task", episode=e.index)
        for t in e.tasks:
            if known and t not in known and seen_missing < 20:
                seen_missing += 1
                report.add("E004", "ERROR",
                           f"episode task {t!r} not in meta tasks table", episode=e.index)


# -- episode index -----------------------------------------------------------


def check_episode_lengths(view: DatasetView, report: Report) -> None:
    """E001/E002: zero-length episodes crash samplers; sub-second ones are
    almost always aborted recordings; extreme outliers skew batch mixing."""
    fps = float(view.info.get("fps") or 30)
    lengths = np.array([e.length for e in view.episodes])
    if not lengths.size:
        report.add("E001", "ERROR", "no episodes found")
        return
    for e in view.episodes:
        if e.length == 0:
            report.add("E001", "ERROR", "episode length 0", episode=e.index)
        elif e.length < fps:
            report.add("E001", "WARN",
                       f"episode shorter than 1 second ({e.length} frames @ {fps:g} fps)",
                       episode=e.index)
    med = float(np.median(lengths))
    for e in view.episodes:
        if med > 0 and e.length > 10 * med:
            report.add("E002", "WARN",
                       f"episode length {e.length} is >10x the median ({med:.0f}) - "
                       f"forgotten stop button?", episode=e.index)


def check_global_index(view: DatasetView, report: Report) -> None:
    """E003 (v3): dataset_from/to_index must tile the dataset exactly -
    overlaps double-serve frames, gaps drop them, and length must match."""
    eps = [e for e in view.episodes if e.from_index is not None]
    if not eps:
        return
    prev_end = 0
    for e in eps:
        if e.to_index - e.from_index != e.length:
            report.add("E003", "ERROR",
                       f"index span {e.to_index - e.from_index} != length {e.length}",
                       episode=e.index)
        if e.from_index != prev_end:
            report.add("E003", "ERROR",
                       f"global index gap/overlap: episode starts at {e.from_index}, "
                       f"previous ended at {prev_end}", episode=e.index)
        prev_end = e.to_index


def check_video_alignment(view: DatasetView, report: Report) -> None:
    """V001/V002: every referenced video file must exist, and (v3) the video
    timestamp span must cover length/fps - a shorter span means frames with no
    matching video frame: the classic A/V desync that surfaces as 'index out
    of range' three weeks into a training run. No decoding needed."""
    fps = float(view.info.get("fps") or 30)
    for e in view.episodes:
        for key, rel in e.video_files.items():
            if view.files and rel not in view.files:
                report.add("V001", "ERROR", f"missing video file {rel}",
                           episode=e.index, feature=key)
        for key, (t0, t1) in e.video_spans.items():
            want = e.length / fps
            got = t1 - t0
            if got < want - 1.5 / fps:
                report.add("V002", "ERROR",
                           f"video span {got:.2f}s covers less than {want:.2f}s of "
                           f"frames ({key})", episode=e.index, feature=key)


# -- frame data (sampled episodes) -------------------------------------------


def _frame_columns(view: DatasetView) -> list[str]:
    return ["frame_index", "timestamp", "index", "task_index", *view.float_features()]


def check_frames(view: DatasetView, report: Report) -> None:
    """D001-D006 on the sampled episodes:
    D001 NaN/Inf; D002 frame_index/index continuity; D003 timestamp regularity
    (dropped frames); D004 data outside its own stats envelope (stale stats);
    D005 action saturation (teleop clipping); D006 whole-episode frozen action
    (dead recording)."""
    fps = float(view.info.get("fps") or 30)
    cols = _frame_columns(view)
    by_index = {e.index: e for e in view.episodes}
    stats = view.stats or {}
    sat_counts: dict[str, np.ndarray] = {}
    sat_total = 0
    dim_values: list[set] = []  # per action dim, capped unique-value sample

    for ep_idx in view.sampled:
        e = by_index.get(ep_idx)
        if e is None or e.data_file is None:
            continue
        try:
            data = view.read_episode(e, cols)
        except FileNotFoundError:
            report.add("D000", "WARN", f"data file not available locally ({e.data_file})",
                       episode=ep_idx)
            continue

        fi = data.get("frame_index")
        if fi is not None and len(fi):
            fi = np.asarray(fi, dtype=float).ravel()
            if len(fi) != e.length:
                report.add("D002", "ERROR",
                           f"parquet has {len(fi)} frames, index declares {e.length}",
                           episode=ep_idx)
            if not np.array_equal(fi, np.arange(len(fi))):
                report.add("D002", "ERROR", "frame_index not contiguous from 0", episode=ep_idx)

        ti = data.get("task_index")
        if ti is not None and len(ti) and view.tasks:
            ti = np.asarray(ti, dtype=float).ravel()
            bad_ti = set(int(x) for x in np.unique(ti)) - set(view.tasks)
            if bad_ti:
                report.add("E004", "ERROR",
                           f"frame task_index {sorted(bad_ti)} not in meta tasks table",
                           episode=ep_idx)

        ts = data.get("timestamp")
        if ts is not None and len(ts) > 1:
            ts = np.asarray(ts, dtype=float).ravel()
            d = np.diff(ts)
            if np.any(d <= 0):
                report.add("D003", "ERROR", "timestamps not strictly increasing",
                           episode=ep_idx)
            big = d > 1.5 / fps
            if np.any(big):
                worst = float(d.max() * fps)
                report.add("D003", "WARN",
                           f"{int(big.sum())} inter-frame gap(s) > 1.5/fps "
                           f"(worst {worst:.1f} frame periods) - dropped frames",
                           episode=ep_idx)

        for key in view.float_features():
            arr = data.get(key)
            if arr is None or not len(arr):
                continue
            try:
                a = np.asarray(np.stack(list(arr)), dtype=float)
            except Exception:
                continue
            if a.ndim == 1:
                a = a[:, None]
            bad = ~np.isfinite(a)
            if bad.any():
                report.add("D001", "ERROR",
                           f"{int(bad.sum())} NaN/Inf value(s) in '{key}'",
                           episode=ep_idx, feature=key)
                a = np.where(bad, np.nan, a)
            st = stats.get(key)
            if st and "min" in st and "max" in st:
                lo = np.asarray(st["min"], dtype=float).ravel()
                hi = np.asarray(st["max"], dtype=float).ravel()
                if lo.size == a.shape[1]:
                    span = np.maximum(hi - lo, 1e-6)
                    out = (a < lo - 0.001 * span) | (a > hi + 0.001 * span)
                    if np.any(np.nan_to_num(out)):
                        report.add("D004", "ERROR",
                                   f"'{key}' values outside meta/stats.json min/max - "
                                   f"stats are stale for this data",
                                   episode=ep_idx, feature=key)
            if key == "action":
                sat_total += a.shape[0]
                st_a = stats.get("action", {})
                lo = np.asarray(st_a.get("min", []), dtype=float).ravel()
                hi = np.asarray(st_a.get("max", []), dtype=float).ravel()
                if lo.size == a.shape[1]:
                    counts = sat_counts.setdefault(key, np.zeros(a.shape[1]))
                    counts += np.nansum((a <= lo + 1e-9) | (a >= hi - 1e-9), axis=0)
                while len(dim_values) < a.shape[1]:
                    dim_values.append(set())
                for dim in range(a.shape[1]):
                    s = dim_values[dim]
                    if len(s) <= 4:  # enough to prove "not binary"
                        s.update(np.round(a[:, dim], 6)[:50].tolist())
                if a.shape[0] > 1 and np.all(np.nanstd(a, axis=0) < EPS_STD):
                    report.add("D006", "WARN",
                               "action constant for the whole episode - dead recording?",
                               episode=ep_idx)

    if sat_total and "action" in sat_counts:
        frac = sat_counts["action"] / sat_total
        for dim, f in enumerate(frac):
            # a genuinely binary dim (e.g. the gripper) lives at its bounds by
            # definition - saturation is only meaningful for continuous dims
            is_binary = dim < len(dim_values) and len(dim_values[dim]) <= 3
            if f > 0.6 and not is_binary:
                report.add("D005", "WARN",
                           f"action dim {dim} sits at its min/max bound in "
                           f"{f:.0%} of sampled frames - teleop clipping or a "
                           f"binary dim mislabelled as continuous", feature="action")


def check_gripper(view: DatasetView, report: Report) -> None:
    """G001: report the gripper convention as a fact, and flag MIXED
    conventions as an error. Same-looking datasets disagree: LIBERO-style
    data uses {-1 open, +1 close}, OpenVLA emits [0,1] with 1=open. A dataset
    that mixes both trains a policy that grasps on half the episodes. This
    linter exists because that bug shipped once with every test green."""
    feats = view.features
    if "action" not in feats:
        return
    by_index = {e.index: e for e in view.episodes}
    values: list[np.ndarray] = []
    per_ep_kind: dict[int, str] = {}
    for ep_idx in view.sampled:
        e = by_index.get(ep_idx)
        if e is None or e.data_file is None:
            continue
        try:
            data = view.read_episode(e, ["action"])
        except FileNotFoundError:
            continue
        arr = data.get("action")
        if arr is None or not len(arr):
            continue
        try:
            a = np.asarray(np.stack(list(arr)), dtype=float)
        except Exception:
            continue
        g = a[:, -1]
        g = g[np.isfinite(g)]
        if not g.size:
            continue
        values.append(g)
        uniq = np.unique(np.round(g, 6))
        if uniq.size <= 3 and np.all(np.isin(uniq, (-1.0, 0.0, 1.0))):
            has_neg, has_01 = bool((uniq == -1).any()), bool((uniq == 0).any())
            kind = "binary{-1,+1}" if has_neg and not has_01 else (
                "binary{0,1}" if has_01 and not has_neg else "binary-mixed")
        else:
            kind = "continuous"
        per_ep_kind[ep_idx] = kind
    if not values:
        return
    kinds = set(per_ep_kind.values())
    allv = np.concatenate(values)
    desc = (f"last action dim over {len(per_ep_kind)} sampled episodes: "
            f"range [{allv.min():.3g}, {allv.max():.3g}], "
            f"convention(s): {', '.join(sorted(kinds))}")
    if len(kinds) > 1 or "binary-mixed" in kinds:
        report.add("G001", "ERROR",
                   f"inconsistent gripper convention across episodes - {desc}",
                   feature="action")
    else:
        report.add("G001", "INFO", desc + " - verify this matches your policy's "
                   "output convention before training", feature="action")


def check_feature_names(view: DatasetView, report: Report) -> None:
    """N001-N005: see names.py - metadata only, no frames read."""
    check_names(view.info, report)


ALL_CHECKS = (
    check_info, check_counts, check_stats, check_tasks, check_feature_names,
    check_episode_lengths, check_global_index, check_video_alignment,
    check_frames, check_gripper,
)


def run_all(view: DatasetView, report: Report) -> None:
    for check in ALL_CHECKS:
        check(view, report)
