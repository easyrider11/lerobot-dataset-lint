"""Synthetic v3 LeRobot dataset builder with targeted corruptions.

Builds the smallest dataset that exercises every rule: 3 episodes x ~20
frames, state[2] + action[3] (last dim = binary gripper), one declared video
feature whose files are created empty (the linter never decodes). Each
corruption flag breaks exactly one contract so each test pins one rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 10


def build_dataset(
    root: Path,
    *,
    lengths: tuple[int, ...] = (20, 20, 20),
    total_frames_off: bool = False,
    stats_wrong_dims: bool = False,
    stats_stale_max: bool = False,
    stats_zero_std: bool = False,
    empty_task: bool = False,
    unknown_task: bool = False,
    index_gap: bool = False,
    missing_video: bool = False,
    short_video_span: bool = False,
    nan_action: bool = False,
    shuffled_frame_index: bool = False,
    timestamp_gap: bool = False,
    mixed_gripper: bool = False,
    frozen_action: bool = False,
    saturated_action: bool = False,
) -> Path:
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    vkey = "observation.images.cam"
    rows: dict[str, list] = {k: [] for k in (
        "episode_index", "frame_index", "index", "timestamp", "task_index",
        "action", "observation.state")}
    ep_rows: dict[str, list] = {k: [] for k in (
        "episode_index", "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index", "tasks", "length",
        f"videos/{vkey}/chunk_index", f"videos/{vkey}/file_index",
        f"videos/{vkey}/from_timestamp", f"videos/{vkey}/to_timestamp")}

    g = 0
    for ep, n in enumerate(lengths):
        state = rng.normal(0, 0.5, (n, 2)).astype(np.float32)
        act = rng.uniform(-0.8, 0.8, (n, 3)).astype(np.float32)
        grip = np.where(np.arange(n) < n // 2, -1.0, 1.0)
        if mixed_gripper and ep == 2:
            grip = np.where(np.arange(n) < n // 2, 0.0, 1.0)  # other convention
        act[:, 2] = grip
        if frozen_action and ep == 1:
            act[:] = 0.25
        if saturated_action:
            # realistic teleop clipping: most frames pinned at the bound, the
            # rest continuous (a constant dim would read as binary and be skipped)
            pin = rng.uniform(size=n) < 0.75
            act[pin, 0] = 0.8
        if nan_action and ep == 0:
            act[3, 1] = np.nan
        fi = np.arange(n)
        if shuffled_frame_index and ep == 1:
            fi = fi[::-1].copy()
        ts = np.arange(n) / FPS
        if timestamp_gap and ep == 2:
            ts[n // 2:] += 5 / FPS  # 5-frame hole
        rows["episode_index"] += [ep] * n
        rows["frame_index"] += fi.tolist()
        rows["index"] += list(range(g, g + n))
        rows["timestamp"] += ts.astype(np.float32).tolist()
        rows["task_index"] += [0] * n
        rows["action"] += [r.tolist() for r in act]
        rows["observation.state"] += [r.tolist() for r in state]

        ep_rows["episode_index"].append(ep)
        ep_rows["data/chunk_index"].append(0)
        ep_rows["data/file_index"].append(0)
        start = g + (3 if index_gap and ep == 1 else 0)
        ep_rows["dataset_from_index"].append(start)
        ep_rows["dataset_to_index"].append(start + n)
        ep_rows["tasks"].append(["missing task" if unknown_task and ep == 1 else "push the block"])
        ep_rows["length"].append(n)
        ep_rows[f"videos/{vkey}/chunk_index"].append(0)
        ep_rows[f"videos/{vkey}/file_index"].append(0)
        ep_rows[f"videos/{vkey}/from_timestamp"].append(float(g) / FPS)
        span = (n / FPS) * (0.5 if short_video_span and ep == 0 else 1.0)
        ep_rows[f"videos/{vkey}/to_timestamp"].append(g / FPS + span)
        g += n

    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table(ep_rows),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table({"task_index": [0], "task": ["" if empty_task else "push the block"]}),
        root / "meta" / "tasks.parquet")

    if not missing_video:
        vdir = root / "videos" / vkey / "chunk-000"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "file-000.mp4").write_bytes(b"\0")

    act_all = np.array(rows["action"], dtype=float)
    state_all = np.array(rows["observation.state"], dtype=float)

    def st(a: np.ndarray) -> dict:
        return {"min": np.nanmin(a, 0).tolist(), "max": np.nanmax(a, 0).tolist(),
                "mean": np.nanmean(a, 0).tolist(), "std": np.nanstd(a, 0).tolist(),
                "count": [int(a.shape[0])]}

    stats = {"action": st(act_all), "observation.state": st(state_all)}
    if stats_stale_max:
        stats["action"]["max"] = (np.array(stats["action"]["max"]) - 0.5).tolist()
    if stats_wrong_dims:
        stats["observation.state"]["mean"] = [0.0] * 6  # checkpoint config.json bug
        stats["observation.state"]["std"] = [1.0] * 6
    if stats_zero_std:
        stats["observation.state"]["std"] = [0.0, 1.0]

    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "total_episodes": len(lengths),
        "total_frames": g + (7 if total_frames_off else 0),
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            vkey: {"dtype": "video", "shape": [64, 64, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    return root


def strip_episode_tasks(root: Path) -> None:
    """Convert the fixture to libero-style: no tasks column in the episodes
    index; association only via the per-frame task_index column."""
    p = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(p)
    pq.write_table(t.drop_columns(["tasks"]), p)


def corrupt_frame_task_index(root: Path) -> None:
    p = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(p)
    ti = np.asarray(t.column("task_index").to_pylist())
    ti[5] = 99  # not in the tasks table
    t = t.drop_columns(["task_index"]).append_column("task_index", pa.array(ti))
    pq.write_table(t, p)
