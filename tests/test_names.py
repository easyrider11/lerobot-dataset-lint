"""N-rules (feature names, lerobot#4784): layouts, statuses, end-to-end lint."""

from __future__ import annotations

import json

import pytest
from conftest import build_dataset
from test_rules import lint, rules

from lerobot_lint.findings import Report
from lerobot_lint.names import (
    check_names,
    classify,
    dataset_names_class,
    flatten_names,
    names_facts,
)

SO101 = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


# -- layouts LeRobot itself reads ---------------------------------------------


@pytest.mark.parametrize("raw, want, layout", [
    (SO101, SO101, "list"),
    ({"delta_x": 0, "delta_y": 1, "gripper": 2}, ["delta_x", "delta_y", "gripper"],
     "indexed-dict"),
    ({"gripper": 2, "delta_x": 0, "delta_y": 1}, ["delta_x", "delta_y", "gripper"],
     "indexed-dict"),
    ({"motors": ["waist", "shoulder"]}, ["waist", "shoulder"], "grouped"),
    ({"left": ["a", "b"], "right": ["c"]}, ["a", "b", "c"], "grouped"),
    ([["a", "b"], ["c"]], ["a", "b", "c"], "list"),
    (None, None, "absent"),
])
def test_flatten_layouts(raw, want, layout):
    assert flatten_names(raw) == (want, layout)


@pytest.mark.parametrize("raw", [{"a": 0, "b": 2}, {"motors": 3}, [{"x": 1}], 5])
def test_flatten_rejects_unreadable(raw):
    assert flatten_names(raw) == (None, "invalid")


# Real layouts from the 2026-10-07 hub sweep that the first N005 version
# wrongly called unreadable. LeRobot's own flattener (molmoact2,
# _flatten_feature_names) returns None for all of them: they mean "no names".
@pytest.mark.parametrize("raw", [
    [],                    # RoboCOIN/*: gripper_open_scale_state, shape [1]
    {},
    {"axes": None},        # lerobot/metaworld_mt50: observation.state
    [[]],
])
def test_empty_declarations_are_absent_not_invalid(raw):
    assert flatten_names(raw) == (None, "empty")
    assert classify("observation.state", ft(raw, shape=(4,))).status == "absent"


def test_grouped_dict_skips_null_groups():
    assert flatten_names({"motors": ["a", "b"], "axes": None}) == (["a", "b"], "grouped")


# -- statuses -------------------------------------------------------------------


def ft(names, shape=(6,), dtype="float32"):
    return {"dtype": dtype, "shape": list(shape), "names": names}


@pytest.mark.parametrize("names, status", [
    (SO101, "semantic"),
    ([f"motor_{i}" for i in range(6)], "placeholder"),
    ([f"joint{i}" for i in range(6)], "placeholder"),
    ([str(i) for i in range(6)], "placeholder"),
    (["x", "y", "z", "roll", "pitch", "yaw"], "semantic"),  # bare axes are real names
    (SO101[:5], "mismatch"),
    (SO101[:5] + ["gripper.pos", "extra"], "mismatch"),
    (SO101[:5] + ["shoulder_pan.pos"], "duplicate"),
    (None, "absent"),
    ({"a": 0, "b": 5}, "invalid"),
])
def test_classify(names, status):
    assert classify("action", ft(names)).status == status


# -- real layouts behind the first sweep's N001 hits (2026-10-07 audit) -------


@pytest.mark.parametrize("names, shape", [
    (["state"], (8,)),            # lerobot/libero observation.state
    (["actions"], (7,)),          # lerobot/libero action
    ("motion_token", (64,)),      # mncai/G1_Dex3_*: action.motion_token (bare string)
    (["channels"], (8,)),         # jasontchan/emg-*: observation.emg.lower
])
def test_one_label_for_whole_vector(names, shape):
    assert classify("observation.state", ft(names, shape=shape)).status == "whole-vector"


def test_one_name_for_one_dim_is_still_semantic():
    assert classify("action", ft(["gripper"], shape=(1,))).status == "semantic"


@pytest.mark.parametrize("names, shape", [
    (["way", "points"], (10, 2)),                       # yaak-ai/L2D waypoints
    (["hand", "finger", "height", "width", "flow_type"], (2, 5, 24, 32, 4)),  # tactile
])
def test_one_name_per_axis_is_axis_names(names, shape):
    assert classify("observation.state.waypoints", ft(names, shape=shape)).status == "axis-names"


def test_names_for_last_axis_of_action_chunk():
    # suz22/RoboTwin_*: action shape [20, 14], names [[14 names]]
    names = [[f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]]
    fact = classify("action", ft(names, shape=(20, 14)))
    assert fact.status == "semantic"
    short = [[f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(6)]]
    assert classify("action", ft(short, shape=(20, 14))).status == "mismatch"


@pytest.mark.parametrize("names, shape", [
    # cadene/droid_1.0.1: joint_position has the 6 cartesian axis names
    ({"axes": ["x", "y", "z", "roll", "pitch", "yaw"]}, (7,)),
    # mncai/G1_Dex3_*: observation.eef_state, 4 group names for 14 dims
    (["left_wrist_pos", "left_wrist_abs_quat", "right_wrist_pos", "right_wrist_abs_quat"],
     (14,)),
])
def test_real_mismatches_stay_mismatches(names, shape):
    assert classify("observation.state", ft(names, shape=shape)).status == "mismatch"


def test_whole_vector_is_info_n006():
    r = run({"observation.state": ft(["state"], shape=(8,))})
    assert {(f.rule, f.severity) for f in r.findings} == {("N006", "INFO")}


def test_placeholder_needs_majority():
    names = ["motor_0", "motor_1", "elbow", "wrist", "gripper", "base"]
    assert classify("action", ft(names)).status == "semantic"


def test_multidim_shape_counts_all_dims():
    assert classify("action", ft([f"j{i}" for i in range(6)], shape=(2, 3))).dims == 6
    assert classify("action", ft(SO101, shape=(2, 3))).status == "semantic"


def test_images_and_index_columns_are_not_vector_features():
    info = {"features": {
        "observation.images.top": {"dtype": "video", "shape": [480, 640, 3],
                                   "names": ["height", "width", "channels"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "action": ft(SO101),
    }}
    assert [f.feature for f in names_facts(info)] == ["action"]


# -- findings -------------------------------------------------------------------


def run(features: dict) -> Report:
    r = Report(repo="t")
    check_names({"features": features}, r)
    return r


def test_findings_per_status():
    r = run({
        "action": ft(SO101[:5]),                                   # N001
        "observation.state": ft(SO101),                            # semantic
        "observation.effort": ft(SO101[:5] + ["elbow_flex.pos"]),  # N002
        "observation.velocity": ft([f"motor_{i}" for i in range(6)]),  # N003
        "observation.env_state": ft(None, shape=(4,)),             # N004
        "observation.scalar": ft(None, shape=(1,)),                # nothing
        "observation.bad": ft({"a": 0, "b": 7}),                   # N005
    })
    got = {(f.rule, f.severity, f.feature) for f in r.findings}
    assert got == {
        ("N001", "ERROR", "action"),
        ("N002", "ERROR", "observation.effort"),
        ("N003", "WARN", "observation.velocity"),
        ("N004", "INFO", "observation.env_state"),
        ("N005", "ERROR", "observation.bad"),
    }


def test_facts_are_machine_readable():
    r = run({"action": ft([f"motor_{i}" for i in range(6)])})
    out = json.loads(r.to_json())
    assert out["facts"]["names"] == [{"feature": "action", "dims": 6, "layout": "list",
                                      "n_names": 6, "status": "placeholder"}]


@pytest.mark.parametrize("statuses, want", [
    ({"action": SO101, "observation.state": SO101}, "semantic"),
    ({"action": SO101, "observation.state": [f"motor_{i}" for i in range(6)]}, "placeholder"),
    ({"action": None, "observation.state": [f"motor_{i}" for i in range(6)]}, "absent"),
    ({"action": SO101[:3], "observation.state": SO101}, "mismatch"),
])
def test_dataset_class_is_worst_of_action_and_state(statuses, want):
    info = {"features": {k: ft(v) for k, v in statuses.items()}}
    assert dataset_names_class(names_facts(info)) == want


def test_dataset_class_without_vector_features():
    assert dataset_names_class([]) == "no-vector-features"


# -- end to end through the linter ---------------------------------------------


def test_clean_fixture_names_are_semantic(tmp_path):
    r = lint(build_dataset(tmp_path / "d"))
    assert not rules(r) & {"N001", "N002", "N003", "N004", "N005"}
    assert {f["status"] for f in r.facts["names"]} == {"semantic"}


def test_lint_flags_short_names_list(tmp_path):
    r = lint(build_dataset(tmp_path / "d", names={"action": ["shoulder_pan.pos", "gripper"]}))
    assert "N001" in rules(r, "ERROR")
    assert r.exit_code() == 1


def test_lint_placeholder_is_warning_not_error(tmp_path):
    r = lint(build_dataset(tmp_path / "d", names={"observation.state": ["motor_0", "motor_1"]}))
    assert "N003" in rules(r, "WARN")
    assert r.exit_code() == 0 and r.exit_code(strict=True) == 1


def test_lint_absent_names_is_info_only(tmp_path):
    r = lint(build_dataset(tmp_path / "d", names={"action": None, "observation.state": None}))
    assert "N004" in rules(r, "INFO")
    assert r.exit_code(strict=True) == 0  # #4784: never block on missing names


# -- hub sweep (offline: the pure classify/summarize halves) -------------------


def _sweep_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "studies" / "names-sweep" / "sweep.py"
    spec = importlib.util.spec_from_file_location("names_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sweep_classify_and_summarize():
    sweep = _sweep_module()
    rows = [
        sweep.classify_info("a/so101", {"robot_type": "so101_follower", "features": {
            "action": ft(SO101), "observation.state": ft(SO101)}}),
        sweep.classify_info("b/old", {"robot_type": "koch", "features": {
            "action": ft([f"motor_{i}" for i in range(6)])}}),
        sweep.classify_info("c/none", {"robot_type": None, "features": {
            "action": ft(None)}}),
        {"repo": "d/broken", "classification": "FETCH_ERROR", "error": "HTTPError"},
    ]
    assert [r["classification"] for r in rows[:3]] == ["semantic", "placeholder", "absent"]
    md = sweep.summarize(rows)
    assert "Datasets read: 3 (fetch errors: 1)" in md
    assert "| semantic | 1 | 33.3% |" in md
    assert "2/3 = 66.7%" in md
    assert "- N003: 1" in md and "- N004: 1" in md


def test_sweep_explain_prints_raw_names_for_one_status():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "studies" / "names-sweep" / "explain.py"
    spec = importlib.util.spec_from_file_location("names_explain", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = {"features": {"action": ft({"a": 0, "b": 5}),
                         "observation.state": ft(SO101),
                         "observation.images.top": {"dtype": "video", "shape": [3], "names": 7}}}
    lines = mod.describe("x/y", info, "invalid")
    assert lines == ['x/y  action  float32 shape=[6]  names={"a": 0, "b": 5}']
    assert mod.RULE_STATUS["N005"] == "invalid"
