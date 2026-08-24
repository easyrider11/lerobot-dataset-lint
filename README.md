# lerobot-dataset-lint

**Contract linter for [LeRobot](https://github.com/huggingface/lerobot) datasets.**
Catch convention drift, dropped frames, stale stats and dead recordings *before*
they poison a training run.

```bash
uvx lerobot-dataset-lint lerobot/pusht          # any hub repo id
lerobot-lint ~/my_dataset --sample 100 --strict # or a local dataset dir
```

No torch, no lerobot import, no video decoding. Meta files are always fetched;
frame-level checks run on a sample of episodes and download **only the parquet
files that contain them** — linting a multi-GB dataset costs megabytes.
Supports both layouts found in the wild (`v2.x` jsonl-meta and `v3.0`
aggregated-parquet). Exit code is CI-friendly: non-zero on errors
(`--strict`: on warnings too).

## Why

The LeRobot Hub crossed 58,000 community datasets. Dataset bugs are the
quietest failure mode in robot learning: **nothing crashes — the policy just
doesn't grasp.** This tool exists because of a real one: a gripper sign
convention documented backwards, self-consistent in each half of a codebase,
every test green, discovered only when a third component forced the two
halves together. The same class of bug ships inside datasets, where no test
will ever see it. And it has company: we found a *published policy
checkpoint* whose `config.json` declared a 6-dim state while its
normalization weights carried 8 dims — stale metadata, silently trusted.

A linter can't know your intent, but it can prove your dataset agrees with
itself: data ↔ stats ↔ metadata ↔ index ↔ video timeline.

## The rules

| Rule | Sev | What it catches | Why it matters |
|---|---|---|---|
| M001 | ERROR | missing/invalid `meta/info.json` keys | everything downstream trusts them |
| M002 | WARN | unknown `codebase_version` | best-effort lint only |
| M003 | ERROR | episode index disagrees with `total_episodes` / `total_frames` | interrupted upload or hand-edited meta |
| M004 | ERROR | **stats dims ≠ declared feature shape** | stale metadata: normalization reads stats, not your intention |
| M005 | ERROR/WARN | stats `min>max`, `std≈0` dims, missing stats.json | MEAN_STD normalization explodes or pins dims |
| M006 | WARN | empty task strings | starves language conditioning |
| E001 | ERROR/WARN | zero-length / sub-second episodes | aborted recordings crash or dilute training |
| E002 | WARN | episode >10× median length | forgotten stop button |
| E003 | ERROR | v3 global index gaps/overlaps, span ≠ length | frames double-served or silently dropped |
| E004 | ERROR/WARN | episode task missing from tasks table / no task | dataloader crash |
| V001 | ERROR | referenced video file absent from repo | KeyError at decode time |
| V002 | ERROR | **video timestamp span shorter than frames/fps** | A/V desync, "index out of range" mid-training — caught without decoding a single frame |
| D001 | ERROR | NaN/Inf in float features | silent gradient poison |
| D002 | ERROR | `frame_index` not contiguous, parquet rows ≠ declared length | delta-timestamp lookups shift |
| D003 | ERROR/WARN | non-monotonic timestamps; inter-frame gaps >1.5/fps | dropped frames — action chunks span holes |
| D004 | ERROR | data outside its own stats min/max envelope | stats are stale for this data |
| D005 | WARN | action dim saturated at its bound in >60% of frames | teleop clipping / binary dim mislabelled continuous |
| D006 | WARN | action constant for a whole episode | dead recording |
| G001 | ERROR/INFO | **gripper convention report; mixed conventions across episodes** | `{-1,+1}` vs `{0,1}` vs continuous — the "arm never grasps" classic |

## Example

```text
lerobot-lint · lerobot/pusht  (v3.0, checked 50/206 episodes)
  · frame-level checks on 50 sampled episode(s); videos checked by listing + index, never decoded
  i INFO  G001 (1×)
      last action dim over 50 sampled episodes: range [25, 511], convention(s): continuous - verify this matches your policy's output convention before training
  = 0 error(s), 0 warning(s), 1 info
```

`--json` emits the full machine-readable report for CI.

## Validated against the real world

Development was test-driven on synthetic corrupted fixtures (22 tests, every
rule covered), then pointed at live hub datasets:

| Dataset | Layout | Result |
|---|---|---|
| `lerobot/pusht` | v3.0 | clean; G001 reports continuous action (no gripper dim) |
| `lerobot/libero` | v3.0 | clean; surfaced that **official v3 datasets disagree on task association** (pusht: `tasks` in the episodes index; libero: per-frame `task_index` only) — the linter now validates both |
| `qb1t/so101_teleop_cubes` | v2.1 | clean |
| `kaiserbuffle/so101_test`, `BasedLukas/so101_test_2` | v2.1 | clean after teaching the linter that v2.1's stats live in `episodes_stats.jsonl` (aggregated with exact combined std) |
| `DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task` | v3.0 | **D005: action dim 5 at its bound in 64% of sampled frames** — real teleop clipping in a real community dataset |

Two of the linter's own false positives were found and fixed by this sweep
(binary gripper dims "saturate" by definition; v2.1 stats location) — a
linter must hold itself to the standard it enforces.

## Non-goals (v0.1)

- Decoding videos (frame-content checks) — by design; everything here runs on
  metadata + parquet.
- Semantic checks (is the task string *true*?) — a linter proves
  self-consistency, not intent.
- Fixing datasets — it points, you decide.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q        # synthetic corrupted-dataset fixtures, every rule covered
ruff check src tests
```

Apache-2.0, matching lerobot. Built in response to
[huggingface/lerobot#2326](https://github.com/huggingface/lerobot/issues/2326)
(community call for LeRobotDataset tools).
