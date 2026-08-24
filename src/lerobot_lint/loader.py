"""Load a LeRobot dataset (local dir or hub repo) into a uniform view.

Supports the two layouts found in the wild:

    v3.0   meta/info.json, meta/stats.json, meta/tasks.parquet,
           meta/episodes/chunk-XXX/file-XXX.parquet (index + per-episode stats
           + data/video file mapping), data/chunk-XXX/file-XXX.parquet
           (multi-episode files), videos/{key}/chunk-XXX/file-XXX.mp4
    v2.x   meta/info.json, meta/stats.json?, meta/episodes.jsonl,
           meta/tasks.jsonl, data/chunk-XXX/episode_XXXXXX.parquet,
           videos/chunk-XXX/{key}/episode_XXXXXX.mp4

Hub mode downloads meta/** always and ONLY the data files that contain the
sampled episodes - a 58k-dataset ecosystem needs a linter that does not pull
gigabytes of video to say "your timestamps jitter". Videos are never
downloaded; their existence is checked against the repo file listing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Episode:
    index: int
    length: int
    tasks: list[str]
    data_file: str | None = None          # repo-relative parquet holding this episode
    from_index: int | None = None         # global first frame index (v3)
    to_index: int | None = None           # global last+1 (v3)
    video_files: dict[str, str] = field(default_factory=dict)      # key -> repo path
    video_spans: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class DatasetView:
    repo: str
    root: Path                 # local directory (download cache in hub mode)
    info: dict
    version: str
    stats: dict | None
    tasks: dict[int, str]
    episodes: list[Episode]
    files: set[str]            # full repo file listing (even files not downloaded)
    sampled: list[int]         # episode indices selected for frame-level checks
    # two task-association conventions exist in the wild, both in OFFICIAL
    # datasets: lerobot/pusht stores a `tasks` list in the episodes index;
    # lerobot/libero has no such column and associates via the per-frame
    # `task_index` column instead. Rules must know which one they are looking at.
    episode_tasks_present: bool = True
    stats_source: str | None = None

    @property
    def features(self) -> dict:
        return self.info.get("features", {})

    def float_features(self) -> list[str]:
        return [k for k, v in self.features.items()
                if v.get("dtype") in ("float32", "float64")
                and k not in ("timestamp",) and not k.startswith("next.")]

    def read_episode(self, ep: Episode, columns: list[str]) -> dict[str, np.ndarray]:
        """Frame-level table for one episode, non-video columns only."""
        import pyarrow.parquet as pq

        path = self.root / ep.data_file
        if not path.exists():
            raise FileNotFoundError(ep.data_file)
        avail = pq.read_schema(path).names
        cols = [c for c in columns if c in avail]
        table = pq.read_table(path, columns=list({*cols, "episode_index"}))
        mask = np.asarray(table.column("episode_index")) == ep.index
        out = {}
        for c in cols:
            col = table.column(c)
            arr = np.asarray(col.to_pylist(), dtype=object)
            # fixed-size list columns come back as lists; stack to 2-D floats
            try:
                out[c] = np.array([np.asarray(x) for x in arr[mask]])
            except Exception:
                out[c] = arr[mask]
        return out


def _sample(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total))
    idx = np.linspace(0, total - 1, n).round().astype(int)
    return sorted(set(int(i) for i in idx))


def _load_v3_episodes(root: Path, files: set[str], info: dict) -> list[Episode]:
    import pyarrow.parquet as pq

    eps: list[Episode] = []
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    data_tpl = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    video_tpl = info.get(
        "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    index_files = sorted(
        f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    for rel in index_files:
        path = root / rel
        if not path.exists():
            continue
        t = pq.read_table(path)
        cols = t.column_names
        _load_v3_episodes.tasks_col_seen = "tasks" in cols  # type: ignore[attr-defined]
        for i in range(t.num_rows):
            row = {c: t.column(c)[i].as_py() for c in cols if not c.startswith("stats/")}
            ep = Episode(
                index=int(row["episode_index"]),
                length=int(row.get("length", 0)),
                tasks=list(row.get("tasks") or []),
                from_index=row.get("dataset_from_index"),
                to_index=row.get("dataset_to_index"),
            )
            if "data/chunk_index" in row:
                ep.data_file = data_tpl.format(
                    chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
            for k in video_keys:
                ci, fi = row.get(f"videos/{k}/chunk_index"), row.get(f"videos/{k}/file_index")
                if ci is not None:
                    ep.video_files[k] = video_tpl.format(video_key=k, chunk_index=ci, file_index=fi)
                ts0 = row.get(f"videos/{k}/from_timestamp")
                ts1 = row.get(f"videos/{k}/to_timestamp")
                if ts0 is not None and ts1 is not None:
                    ep.video_spans[k] = (float(ts0), float(ts1))
            eps.append(ep)
    return sorted(eps, key=lambda e: e.index)


def _load_v2_episodes(root: Path, files: set[str], info: dict) -> list[Episode]:
    eps: list[Episode] = []
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    data_tpl = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_tpl = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    p = root / "meta" / "episodes.jsonl"
    if not p.exists():
        return eps
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["episode_index"])
        chunk = i // chunks_size
        ep = Episode(index=i, length=int(row.get("length", 0)),
                     tasks=list(row.get("tasks") or []))
        ep.data_file = data_tpl.format(episode_chunk=chunk, episode_index=i)
        for k in video_keys:
            ep.video_files[k] = video_tpl.format(
                episode_chunk=chunk, video_key=k, episode_index=i)
        eps.append(ep)
    return sorted(eps, key=lambda e: e.index)


def aggregate_stats(per_episode: list[dict]) -> dict:
    """Fold v2.1 per-episode stats (meta/episodes_stats.jsonl) into global
    stats: min of mins, max of maxes, count-weighted mean, and the exact
    combined std via E[x^2] = std^2 + mean^2. v2.1 dropped the global
    stats.json in favour of this file, so a linter that only looks for
    stats.json false-positives on every v2.1 dataset."""
    out: dict[str, dict] = {}
    keys: set[str] = set()
    for ep in per_episode:
        keys.update(ep)
    for key in keys:
        rows = [ep[key] for ep in per_episode if key in ep and isinstance(ep[key], dict)]
        rows = [r for r in rows if "mean" in r and "count" in r]
        if not rows:
            continue
        try:
            counts = np.array([float(np.asarray(r["count"]).ravel()[0]) for r in rows])
            means = np.stack([np.asarray(r["mean"], dtype=float).ravel() for r in rows])
            stds = np.stack([np.asarray(r["std"], dtype=float).ravel() for r in rows])
            mins = np.stack([np.asarray(r["min"], dtype=float).ravel() for r in rows])
            maxs = np.stack([np.asarray(r["max"], dtype=float).ravel() for r in rows])
        except (ValueError, KeyError, TypeError):
            continue  # ragged shapes (e.g. image stats stored per-channel arrays)
        n = counts.sum()
        w = (counts / n)[:, None]
        mean = (w * means).sum(0)
        ex2 = (w * (stds ** 2 + means ** 2)).sum(0)
        var = np.maximum(ex2 - mean ** 2, 0.0)
        out[key] = {"min": mins.min(0).tolist(), "max": maxs.max(0).tolist(),
                    "mean": mean.tolist(), "std": np.sqrt(var).tolist(),
                    "count": [int(n)]}
    return out


def _load_tasks(root: Path) -> dict[int, str]:
    import pyarrow.parquet as pq

    tasks: dict[int, str] = {}
    pt = root / "meta" / "tasks.parquet"
    if pt.exists():
        t = pq.read_table(pt)
        names = t.column_names
        # the task string column is "__index_level_0__" (pandas index artifact)
        # or "task" depending on writer version
        str_col = "task" if "task" in names else "__index_level_0__"
        for i in range(t.num_rows):
            tasks[int(t.column("task_index")[i].as_py())] = str(t.column(str_col)[i].as_py())
        return tasks
    pj = root / "meta" / "tasks.jsonl"
    if pj.exists():
        for line in pj.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                tasks[int(row["task_index"])] = str(row.get("task", ""))
    return tasks


def open_dataset(target: str, sample: int = 50, revision: str | None = None) -> DatasetView:
    """`target` is a local directory or a hub repo id like 'lerobot/pusht'."""
    local = Path(target).expanduser()
    if local.is_dir():
        root = local
        files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        repo = str(local)
    else:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        files = set(api.list_repo_files(target, repo_type="dataset", revision=revision))
        root = Path(snapshot_download(
            target, repo_type="dataset", revision=revision, allow_patterns=["meta/*", "meta/**"]))
        repo = target

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{repo}: meta/info.json not found - not a LeRobot dataset?")
    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", "?"))

    stats = None
    stats_source = None
    sp = root / "meta" / "stats.json"
    if sp.exists():
        stats = json.loads(sp.read_text())
        stats_source = "meta/stats.json"
    else:
        es = root / "meta" / "episodes_stats.jsonl"
        if es.exists():
            per_ep = []
            for line in es.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    per_ep.append(row.get("stats", row))
            if per_ep:
                stats = aggregate_stats(per_ep)
                stats_source = "aggregated from meta/episodes_stats.jsonl (v2.1)"


    _load_v3_episodes.tasks_col_seen = True  # type: ignore[attr-defined]
    if version.startswith("v3"):
        episodes = _load_v3_episodes(root, files, info)
        tasks_present = bool(getattr(_load_v3_episodes, "tasks_col_seen", True))
    else:
        episodes = _load_v2_episodes(root, files, info)
        tasks_present = True
    tasks = _load_tasks(root)

    sampled = [episodes[i].index for i in _sample(len(episodes), sample)] if episodes else []

    if not local.is_dir():
        # pull only the data files that hold the sampled episodes
        need = sorted({e.data_file for e in episodes
                       if e.index in set(sampled) and e.data_file})
        missing = [f for f in need if not (root / f).exists()]
        if missing:
            from huggingface_hub import snapshot_download

            snapshot_download(target, repo_type="dataset", revision=revision,
                              allow_patterns=missing)

    return DatasetView(repo=repo, root=root, info=info, version=version, stats=stats,
                       tasks=tasks, episodes=episodes, files=files, sampled=sampled,
                       episode_tasks_present=tasks_present, stats_source=stats_source)
