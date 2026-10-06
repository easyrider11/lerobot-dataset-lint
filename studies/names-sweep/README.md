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
| absent | no names |
| mismatch | `len(names) != prod(shape)` |
| duplicate | the same name twice in one feature |
| invalid | a layout LeRobot cannot read |

## Results

Not run yet. The sweep was written in a sandbox with no access to
huggingface.co. Run it locally and paste the summary here.
