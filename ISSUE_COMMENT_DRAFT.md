Answering the call for dataset tools — I built a **contract linter**: https://github.com/easyrider11/lerobot-dataset-lint

```bash
uvx lerobot-dataset-lint lerobot/pusht   # any hub repo id or local dir
```

19 rules: stats ↔ data ↔ metadata consistency (stale dims/envelopes, zero-std), dropped frames, v3 global-index tiling, video-timeline coverage (no decoding needed), NaN/Inf, dead recordings, teleop clipping, and a gripper-convention report (mixed conventions across episodes = error). No torch or lerobot import — it fetches meta plus only the parquet files holding the sampled episodes, so linting a multi-GB dataset costs megabytes. Supports v2.x and v3.0, CI-friendly exit codes, `--json`.

Two findings maintainers might care about:

1. Official v3 datasets disagree on task association: `lerobot/pusht` stores `tasks` in the episodes index, `lerobot/libero` only has per-frame `task_index`. The linter handles both — which one is canonical?
2. A first community sweep found real issues, e.g. an action dim sitting at its bound in 64% of frames (teleop clipping) in a public dataset.

Happy to adapt the rule set or donate the tool in whatever shape fits the roadmap. Apache-2.0.
