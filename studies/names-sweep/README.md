# names sweep: can a program tell what each channel is?

Question from [huggingface/lerobot#4784](https://github.com/huggingface/lerobot/issues/4784):
for public LeRobot datasets, do `action` and `observation.state` carry
`names` that say which dim is which joint?

The issue reports, from a scan of 189 datasets: 25.0% semantic names,
40.0% placeholders (`motor_0`...), 30.6% no names. This sweep re-measures
that with the N-rules of this linter, and adds two checks the issue
proposes but did not count: names list length vs shape (N001) and
duplicate names (N002).

## Why it matters (LeRobot code that trusts `names`)

- `src/lerobot/policies/utils.py`: robot command built as
  `{name: action[i] for i, name in enumerate(names)}`. Duplicate names
  drop motors. A short list drops trailing dims. No error either way.
- `src/lerobot/datasets/compute_stats.py` / `policies/factory.py`:
  relative-action stats and `relative_exclude_joints` select dims by name.
  Placeholder names make that unusable.

(Paths checked at lerobot commit `200ee53`, 2026-10-07.)

## Run

```bash
.venv/bin/python studies/names-sweep/sweep.py 500        # one info.json per dataset
.venv/bin/python studies/names-sweep/sweep.py --summary  # re-print summary
```

Cost: one `meta/info.json` (a few KB) per dataset. Resumable.

## Classes

Per dataset, the worst status among `action` and `observation.state`:

| class | meaning |
|---|---|
| semantic | names present, right length, unique, not placeholders |
| placeholder | most names look like `motor_0`, `joint3`, `7` |
| whole-vector | one label for the whole vector (`["state"]` on 8 dims) |
| axis-names | one label per axis of a multi-axis feature (`["way", "points"]` on `[10, 2]`) |
| absent | no names, or an empty declaration (`[]`, `{"axes": null}`) |
| mismatch | names count matches neither the dims nor the last axis |
| duplicate | the same name twice in one feature |
| invalid | a layout LeRobot cannot read |

## Results

Run 2026-10-07: the 500 most-downloaded datasets with the `LeRobot` tag.
359 `meta/info.json` read; 141 not readable (127 gated, 14 with no
`meta/info.json`). Raw rows: `results.jsonl`. Re-score with
`sweep.py --reclassify`, re-print with `sweep.py --summary`.

| names of action / observation.state | datasets | share |
|---|---|---|
| semantic | 266 | 74.1% |
| placeholder | 40 | 11.1% |
| whole-vector | 20 | 5.6% |
| absent | 20 | 5.6% |
| duplicate | 10 | 2.8% |
| axis-names | 3 | 0.8% |

Not programmatically usable (anything but semantic): 93/359 = 25.9%.

Datasets with at least one finding (any vector feature, not only action/state):
N001 3, N002 10, N003 47, N004 45, N006 24. N005: 0.

| robot_type | datasets | semantic |
|---|---|---|
| aloha | 36 | 100% |
| AI2_Alphabot_2 | 35 | 100% |
| Leju_Kuavo_4 | 29 | 100% |
| unknown | 24 | 0% |
| None | 21 | 95% |
| franka | 19 | 63% |
| panda | 17 | 29% |
| G1_ALL | 9 | 0% |

### Audit: what changed between the first pass and these numbers

The first pass (same 359 datasets) reported N005 on 34 datasets and
`mismatch` on 26. Every hit was printed with `explain.py` and read by hand:

- **N005, 34 → 0. All false positives.** 33 RoboCOIN datasets declare
  `names: []` on 1-dim gripper scalars; `lerobot/metaworld_mt50` declares
  `{"axes": null}`. Both mean "no names" - LeRobot's own
  `_flatten_feature_names` (policies/molmoact2) returns None for them. Now
  `absent`.
- **mismatch, 26 → 0 at dataset level** (N001 now fires on 3 datasets, in
  non-core features). Of the 31 datasets with a mismatching feature:
  - whole-vector labels (`["state"]`, `["actions"]`, bare `"motion_token"`):
    LIBERO-style ports, CALVIN, mesa, EMG. Not a count error - a label for
    the vector. Now `whole-vector` (N006, INFO).
  - one name per axis (`yaak-ai/*` waypoints `[10, 2]`, tactile `[2,5,24,32,4]`).
    Now `axis-names`, no finding.
  - names for the last axis of an action chunk (`suz22/RoboTwin_*`, shape
    `[20, 14]`, 14 names). Now checked against that axis: semantic.
  - **real, kept:** `cadene/droid_1.0.1` `observation.state.joint_position`
    (7 dims, the 6 cartesian axis names copied over);
    `mncai/G1_Dex3_Trash_LocoManipulation` and
    `DaoyuanZhu/wb_pointed_chair_pull_push_rgb` `observation.eef_state`
    (14 dims, 4 group names).
- **duplicate (10) checked, all real:** 9 `unitreerobotics/G1_Dex1_*`
  datasets name the 6th end-effector dim `kLeftEEY`/`kRightEEY` - the same
  as the 2nd (yaw vs Y); `IPEC-COMMUNITY/libero_90_no_noops_lerobot` has
  `gripper` twice in `observation.state`.

Each real layout above has a test in `tests/test_names.py`.

### Comparing with lerobot#4784

The issue reports 25.0% semantic over 189 `lerobot/*` datasets. This sweep
gets 74.1% over 359 datasets chosen **by download count across all
owners**. The samples differ (org vs. popularity, different rules), so the
two numbers are not a trend and should not be quoted as one. Say which
sample a number comes from whenever both appear together.
