"""Checkpoint contract checks.

Does a policy repo's config.json agree with its own shipped normalization
statistics? Found in the wild: lerobot/smolvla_libero declares
observation.state shape [6] and three cameras while its normalizer carries
8-dim state stats and its preprocessor renames exactly two cameras - stale
metadata inherited from the base checkpoint (huggingface/lerobot#4517).

Everything here works from metadata only:
- config.json / *_preprocessor.json: small JSON files
- processor normalizer .safetensors: a few KB (stats only), downloaded
- model.safetensors (old-style checkpoints keep stats inside it as
  normalize_* buffers): NEVER downloaded - the safetensors header (8-byte
  little-endian length + JSON) is fetched alone via an HTTP Range request.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .findings import Report

STATE_T, VISUAL_T, ACTION_T = "STATE", "VISUAL", "ACTION"


# -- safetensors headers ------------------------------------------------------


def read_safetensors_header(path: Path) -> dict[str, list[int]]:
    """name -> shape from a local .safetensors file (header only)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: v.get("shape", []) for k, v in header.items() if k != "__metadata__"}


def _range_get(url: str, first: int, last: int, timeout: float) -> bytes:
    import ssl
    import urllib.request

    try:  # framework Pythons on macOS ship no CA bundle for stdlib ssl
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover
        ctx = ssl.create_default_context()
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={first}-{last}", "User-Agent": "lerobot-lint"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


def hub_safetensors_header(repo: str, filename: str, revision: str | None = None
                           ) -> dict[str, list[int]]:
    """Same, for a hub file, via HTTP Range requests - no full download.

    stdlib urllib on purpose: huggingface_hub 1.x dropped `requests`, and this
    must not care which HTTP client generation is installed."""
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, filename, revision=revision)
    (n,) = struct.unpack("<Q", _range_get(url, 0, 7, 30))
    if n > 50_000_000:  # a header this size means we misread the file
        raise ValueError(f"implausible safetensors header size {n}")
    header = json.loads(_range_get(url, 8, 7 + n, 60))
    return {k: v.get("shape", []) for k, v in header.items() if k != "__metadata__"}


# -- checkpoint view ----------------------------------------------------------


class CheckpointView:
    """Lazy accessor over a checkpoint repo (hub id or local dir)."""

    def __init__(self, target: str, revision: str | None = None) -> None:
        self.repo = target
        self.revision = revision
        local = Path(target).expanduser()
        self.local = local if local.is_dir() else None
        if self.local:
            self.files = {str(p.relative_to(local)) for p in local.rglob("*") if p.is_file()}
        else:
            from huggingface_hub import HfApi

            self.files = set(HfApi().list_repo_files(target, revision=revision))

    def _json(self, name: str) -> dict | None:
        if name not in self.files:
            return None
        if self.local:
            return json.loads((self.local / name).read_text())
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(self.repo, name, revision=self.revision))
                          .read_text())

    def config(self) -> dict | None:
        return self._json("config.json")

    def train_config(self) -> dict | None:
        return self._json("train_config.json")

    def preprocessor(self) -> dict | None:
        for name in self.files:
            if name.endswith("preprocessor.json") and "/" not in name:
                return self._json(name)
        return None

    def rename_map(self) -> dict[str, str]:
        pre = self.preprocessor() or {}
        for step in pre.get("steps", []):
            if step.get("registry_name") == "rename_observations_processor":
                return dict(step.get("config", {}).get("rename_map", {}))
        return {}

    def stats_shapes(self) -> tuple[dict[str, list[int]], str] | None:
        """(feature -> stat dims, source). Prefers processor stats files."""
        proc = sorted(f for f in self.files
                      if f.endswith(".safetensors") and "normalizer_processor" in f
                      and "unnormalizer" not in f)
        unproc = sorted(f for f in self.files
                        if f.endswith(".safetensors") and "unnormalizer_processor" in f)
        shapes: dict[str, list[int]] = {}
        for name in proc + unproc:
            if self.local:
                header = read_safetensors_header(self.local / name)
            else:
                from huggingface_hub import hf_hub_download

                header = read_safetensors_header(
                    Path(hf_hub_download(self.repo, name, revision=self.revision)))
            # keys look like "observation.state.mean" / "action.std"
            for k, shape in header.items():
                base, stat = k.rsplit(".", 1)
                if stat in ("mean", "std", "min", "max") and shape:
                    prev = shapes.get(base)
                    if prev is None or prev == shape:
                        shapes[base] = shape
                    else:
                        shapes[base] = ["INCONSISTENT", prev, shape]
        if shapes:
            return shapes, "processor stats"

        # old-style: normalize_* buffers inside model.safetensors (header only)
        if "model.safetensors" in self.files:
            if self.local:
                header = read_safetensors_header(self.local / "model.safetensors")
            else:
                header = hub_safetensors_header(self.repo, "model.safetensors",
                                                self.revision)
            for k, shape in header.items():
                if ".buffer_" not in k or k.split(".")[-1] not in ("mean", "std", "min", "max"):
                    continue
                buf = k.split(".buffer_", 1)[1].rsplit(".", 1)[0]  # observation_state
                if shape and shape != [1]:
                    shapes[buf] = shape
            if shapes:
                return shapes, "model.safetensors normalize buffers"
        return None


def _flat(shape) -> int | None:
    try:
        n = 1
        for s in shape:
            n *= int(s)
        return n
    except (TypeError, ValueError):
        return None


def _stat_dims(stats: dict[str, list[int]], key: str) -> int | None:
    """Match a config feature key against stats keys (dots or underscores)."""
    for cand in (key, key.replace(".", "_")):
        if cand in stats:
            return _flat(stats[cand])
    return None


# -- the checks ---------------------------------------------------------------


def check_checkpoint(view: CheckpointView, report: Report) -> dict:
    """Runs all checkpoint rules; returns facts for sweep aggregation."""
    facts: dict = {"repo": view.repo}
    cfg = view.config()
    if cfg is None or "type" not in cfg:
        report.add("K000", "ERROR", "no config.json with a policy 'type' - not a "
                                    "lerobot policy checkpoint?")
        facts["classification"] = "NOT_A_POLICY"
        return facts
    facts["policy_type"] = cfg.get("type")
    tc = view.train_config() or {}
    pretrained = (tc.get("policy") or {}).get("pretrained_path")
    facts["finetuned_from"] = pretrained
    report.version = f"policy:{cfg.get('type')}"

    inputs = cfg.get("input_features") or {}
    outputs = cfg.get("output_features") or {}
    got = view.stats_shapes()
    if got is None:
        report.add("K005", "WARN", "no normalization stats found (no processor stats "
                                   "files, no normalize_* buffers) - config cannot be "
                                   "verified against the shipped weights")
        facts["classification"] = "UNVERIFIABLE"
        return facts
    stats, source = got
    report.notes.append(f"stats source: {source}")
    contradiction = False

    # only canonical rank-1 proprio vectors are safe to compare by size:
    # normalizers keep PER-CHANNEL stats for pointclouds/images-as-state, so a
    # config shape [8192] vs 4-dim stats is a convention, not a contradiction
    CANONICAL = ("observation.state", "observation.environment_state")
    for key, ft in inputs.items():
        if ft.get("type") != STATE_T:
            continue
        want = _flat(ft.get("shape") or [])
        have = _stat_dims(stats, key)
        if key not in CANONICAL or len(ft.get("shape") or []) != 1:
            if want and have and want != have:
                report.add("K004", "INFO",
                           f"'{key}' config shape {ft.get('shape')} vs {have}-dim stats - "
                           f"not compared (per-channel stats conventions vary for "
                           f"non-canonical state features)", feature=key)
            continue
        facts["state_cfg"], facts["state_stats"] = want, have
        if want and have and want != have:
            contradiction = True
            report.add("K001", "ERROR",
                       f"config declares '{key}' shape [{want}] but the shipped "
                       f"normalization stats are {have}-dimensional - stale feature "
                       f"spec (the huggingface/lerobot#4517 failure mode)",
                       feature=key)

    for key, ft in outputs.items():
        if ft.get("type") != ACTION_T:
            continue
        want = _flat(ft.get("shape") or [])
        have = _stat_dims(stats, key)
        facts["action_cfg"], facts["action_stats"] = want, have
        if want and have and want != have:
            contradiction = True
            report.add("K002", "ERROR",
                       f"config declares '{key}' shape [{want}] but the shipped "
                       f"action stats are {have}-dimensional", feature=key)

    rename = view.rename_map()
    if rename:
        produced = set(rename.values())
        cams = {k for k, ft in inputs.items() if ft.get("type") == VISUAL_T}
        phantom = {c for c in cams if c not in produced and c not in rename}
        facts["cameras_cfg"] = sorted(cams)
        facts["cameras_produced"] = sorted(produced)
        if phantom and produced:
            report.add("K003", "WARN",
                       f"config declares camera(s) {sorted(phantom)} that the "
                       f"preprocessor's rename map never produces "
                       f"({sorted(produced)} are the real ones) - phantom cameras "
                       f"inherited from a base checkpoint", feature=",".join(sorted(phantom)))

    facts["classification"] = ("CONTRADICTORY" if contradiction
                               else "SUSPECT" if any(f.rule == "K003" for f in report.findings)
                               else "CLEAN")
    return facts
