"""Checkpoint-mode rules against hand-built local checkpoint fixtures."""

from __future__ import annotations

import json
import struct

from lerobot_lint.checkpoint import CheckpointView, check_checkpoint
from lerobot_lint.findings import Report


def write_safetensors(path, tensors: dict[str, list[int]]):
    """Minimal valid safetensors: header + zero payload."""
    header, offset = {}, 0
    for name, shape in tensors.items():
        n = 4
        for s in shape:
            n *= s
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * offset)


def build(root, *, state_cfg=8, state_stats=8, phantom_camera=False,
          old_style=False, no_stats=False):
    root.mkdir(parents=True, exist_ok=True)
    cams = ["observation.images.camera1", "observation.images.camera2"]
    if phantom_camera:
        cams.append("observation.images.camera3")
    cfg = {
        "type": "smolvla",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [state_cfg]},
            **{c: {"type": "VISUAL", "shape": [3, 256, 256]} for c in cams},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "policy_preprocessor.json").write_text(json.dumps({
        "steps": [{"registry_name": "rename_observations_processor",
                   "config": {"rename_map": {
                       "observation.images.image": "observation.images.camera1",
                       "observation.images.image2": "observation.images.camera2"}}}]}))
    if no_stats:
        return root
    if old_style:
        write_safetensors(root / "model.safetensors", {
            "normalize_inputs.buffer_observation_state.mean": [state_stats],
            "normalize_inputs.buffer_observation_state.std": [state_stats],
            "unnormalize_outputs.buffer_action.mean": [7],
            "some.weight": [16, 16],
        })
    else:
        write_safetensors(root / "policy_preprocessor_step_5_normalizer_processor.safetensors", {
            "observation.state.mean": [state_stats],
            "observation.state.std": [state_stats],
            "action.mean": [7], "action.std": [7],
        })
    return root


def lint(root):
    report = Report(repo=str(root))
    facts = check_checkpoint(CheckpointView(str(root)), report)
    return report, facts


def rules(report, sev):
    return {f.rule for f in report.findings if f.severity == sev}


def test_clean_checkpoint(tmp_path):
    report, facts = lint(build(tmp_path / "c"))
    assert facts["classification"] == "CLEAN"
    assert not report.findings


def test_stale_state_dims_is_the_4517_bug(tmp_path):
    report, facts = lint(build(tmp_path / "c", state_cfg=6, state_stats=8))
    assert facts["classification"] == "CONTRADICTORY"
    assert "K001" in rules(report, "ERROR")


def test_phantom_camera(tmp_path):
    report, facts = lint(build(tmp_path / "c", phantom_camera=True))
    assert facts["classification"] == "SUSPECT"
    assert "K003" in rules(report, "WARN")


def test_old_style_normalize_buffers(tmp_path):
    report, facts = lint(build(tmp_path / "c", old_style=True, state_cfg=6, state_stats=8))
    assert "K001" in rules(report, "ERROR")
    assert "model.safetensors" in " ".join(report.notes)


def test_no_stats_is_unverifiable(tmp_path):
    report, facts = lint(build(tmp_path / "c", no_stats=True))
    assert facts["classification"] == "UNVERIFIABLE"
    assert "K005" in rules(report, "WARN")


def test_pointcloud_per_channel_stats_are_not_a_contradiction(tmp_path):
    """A pointcloud declared [8192] with 4-dim per-channel stats is a stats
    convention, not the stale-feature bug - must not fire K001."""
    root = build(tmp_path / "c")
    cfg = json.loads((root / "config.json").read_text())
    cfg["input_features"]["observation.pointcloud"] = {"type": "STATE", "shape": [8192]}
    (root / "config.json").write_text(json.dumps(cfg))
    write_safetensors(root / "policy_preprocessor_step_5_normalizer_processor.safetensors", {
        "observation.state.mean": [8], "observation.state.std": [8],
        "observation.pointcloud.mean": [4], "observation.pointcloud.std": [4],
        "action.mean": [7], "action.std": [7],
    })
    report, facts = lint(root)
    assert facts["classification"] == "CLEAN"
    assert "K001" not in rules(report, "ERROR")
    assert "K004" in {f.rule for f in report.findings}
