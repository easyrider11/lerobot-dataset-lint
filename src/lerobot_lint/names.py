"""N-rules: can a program answer "what is channel i of this feature?"

`names` in meta/info.json is the only place a LeRobot dataset says which
column of `action` is the gripper and which is shoulder_pan. Code downstream
relies on it without checking:

- lerobot/policies/utils.py builds the robot command as
  ``{name: action[i] for i, name in enumerate(names)}`` - duplicate names
  silently drop motors, a short list silently drops trailing dims.
- relative-action stats (compute_stats.py) and ``relative_exclude_joints``
  pick dims by name - placeholder names make that impossible.

Proposed upstream as huggingface/lerobot#4784. These rules read meta/info.json
only, so they also run on a metadata-only sweep of the hub (see
studies/names-sweep/).

Rule  Sev    Meaning
N001  ERROR  len(names) != prod(shape)
N002  ERROR  the same name appears twice inside one feature
N003  WARN   names are placeholders (motor_0, joint_1, ...) - no identity
N004  INFO   multi-dim feature has no names at all
N005  ERROR  names present but not in any layout LeRobot can read
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .findings import Report

# columns LeRobot itself writes; their identity is the key, not a names list
INDEX_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
VECTOR_DTYPES = {"float16", "float32", "float64", "int8", "int16", "int32", "int64", "bool"}

# "motor_0", "joint-3", "dim7", "12", "action.4" - a generic stem plus an index
PLACEHOLDER = re.compile(
    r"^(?:(?:motor|motors|joint|joints|dim|axis|channel|state|action|value|feature"
    r"|idx|index|j|m|q|x)[ _.\-]?)?\d+$",
    re.IGNORECASE,
)
PLACEHOLDER_SHARE = 0.5  # a feature counts as placeholder-named above this share


@dataclass
class NamesFact:
    feature: str
    dims: int
    layout: str               # absent | list | indexed-dict | grouped | invalid
    names: list[str] | None
    status: str               # semantic | placeholder | absent | mismatch | duplicate | invalid

    def to_dict(self) -> dict:
        return {"feature": self.feature, "dims": self.dims, "layout": self.layout,
                "n_names": None if self.names is None else len(self.names),
                "status": self.status}


def flatten_names(raw: Any) -> tuple[list[str] | None, str]:
    """Return (flat names, layout). Mirrors the layouts LeRobot reads:
    a list (``["shoulder_pan.pos", ...]``), an indexed dict
    (``{"delta_x": 0, ...}``, teleop_gamepad.py) and a grouped dict
    (``{"motors": [...]}``, older ALOHA-style datasets)."""
    if raw is None:
        return None, "absent"
    if isinstance(raw, str):
        return [raw], "list"
    if isinstance(raw, dict):
        if not raw:
            return None, "invalid"
        vals = list(raw.values())
        if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            if sorted(vals) != list(range(len(vals))):
                return None, "invalid"
            return [str(k) for k, _ in sorted(raw.items(), key=lambda kv: kv[1])], "indexed-dict"
        flat: list[str] = []
        for v in vals:
            sub, layout = flatten_names(v)
            if sub is None or layout != "list":
                return None, "invalid"
            flat.extend(sub)
        return flat, "grouped"
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None, "invalid"
        flat = []
        for v in raw:
            if isinstance(v, (list, tuple)):
                sub, layout = flatten_names(v)
                if sub is None:
                    return None, "invalid"
                flat.extend(sub)
            elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
                flat.append(str(v))
            else:
                return None, "invalid"
        return flat, "list"
    return None, "invalid"


def is_vector_feature(key: str, ft: dict) -> bool:
    """Features whose dims are channels a policy reads one by one. Images and
    videos name their axes (height/width/channel), not channels - skipped."""
    if key in INDEX_KEYS or key.startswith("next."):
        return False
    return str(ft.get("dtype", "")) in VECTOR_DTYPES


def classify(key: str, ft: dict) -> NamesFact:
    shape = ft.get("shape") or [1]
    try:
        dims = int(math.prod(int(s) for s in shape))
    except (TypeError, ValueError):
        dims = 0
    names, layout = flatten_names(ft.get("names"))
    if layout == "invalid":
        status = "invalid"
    elif names is None:
        status = "absent"
    elif len(names) != dims:
        status = "mismatch"
    elif len(set(names)) != len(names):
        status = "duplicate"
    elif sum(bool(PLACEHOLDER.match(n.strip())) for n in names) > PLACEHOLDER_SHARE * len(names):
        status = "placeholder"
    else:
        status = "semantic"
    return NamesFact(key, dims, layout, names, status)


def names_facts(info: dict) -> list[NamesFact]:
    feats = info.get("features") or {}
    return [classify(k, ft) for k, ft in feats.items()
            if isinstance(ft, dict) and is_vector_feature(k, ft)]


def check_names(info: dict, report: Report) -> list[NamesFact]:
    """Emit N001-N005 for every vector feature; return the per-feature facts
    so callers (``--json``, the hub sweep) get a machine-readable verdict."""
    facts = names_facts(info)
    for f in facts:
        if f.status == "invalid":
            report.add("N005", "ERROR",
                       f"'{f.feature}' names are not a list, an index dict or a grouped "
                       f"dict of lists - LeRobot cannot map them to dims", feature=f.feature)
        elif f.status == "mismatch":
            report.add("N001", "ERROR",
                       f"'{f.feature}' has {len(f.names or [])} names for {f.dims} dims - "
                       f"code that zips names with values drops or shifts channels",
                       feature=f.feature)
        elif f.status == "duplicate":
            dup = sorted(n for n, c in Counter(f.names or []).items() if c > 1)
            report.add("N002", "ERROR",
                       f"'{f.feature}' repeats name(s) {dup[:5]} - a name->value dict "
                       f"keeps only one of them", feature=f.feature)
        elif f.status == "placeholder":
            report.add("N003", "WARN",
                       f"'{f.feature}' names are placeholders ({', '.join((f.names or [])[:3])}"
                       f"{', ...' if f.dims > 3 else ''}) - channel identity unknown",
                       feature=f.feature)
        elif f.status == "absent" and f.dims > 1:
            report.add("N004", "INFO",
                       f"'{f.feature}' ({f.dims} dims) has no names - channel identity "
                       f"cannot be queried", feature=f.feature)
    report.facts["names"] = [f.to_dict() for f in facts]
    return facts


def dataset_names_class(facts: list[NamesFact]) -> str:
    """One label per dataset, using the categories of lerobot#4784:
    the worst status among `action` and `observation.state` (falls back to
    all vector features if neither exists)."""
    core = [f for f in facts if f.feature in ("action", "observation.state")] or facts
    if not core:
        return "no-vector-features"
    order = ["invalid", "mismatch", "duplicate", "absent", "placeholder", "semantic"]
    return min((f.status for f in core), key=order.index)
