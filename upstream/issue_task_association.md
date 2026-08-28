### Question / inconsistency report

The v3 writer stores an episode's tasks in the **episodes index** (`meta/episodes/*.parquet` gets a `tasks` list column — [`lerobot_dataset.py` `_save_episode_metadata`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py), `"tasks": episode_tasks`), and `lerobot/pusht` follows that layout:

```
episodes index columns (lerobot/pusht):   [..., 'tasks', 'length', ...]
row 0: tasks=['Push the T-shaped block onto the T-shaped target.']
```

But `lerobot/libero` — also official, also `codebase_version: v3.0` — has **no `tasks` column in its episodes index at all**; task association exists only as the per-frame `task_index` column in the data parquet:

```
episodes index columns (lerobot/libero): ['episode_index', 'length', 'dataset_from_index',
  'dataset_to_index', 'data/chunk_index', 'data/file_index', 'videos/.../chunk_index', ...]
data parquet columns:                    [..., 'episode_index', 'index', 'task_index']
```

Both load and train fine today, so every downstream consumer has to treat both as valid — we hit this building a dataset linter ([#2326](https://github.com/huggingface/lerobot/issues/2326)): a rule that trusted the writer's layout false-positived "episode has no task" 1693× on `lerobot/libero` before we taught it the second convention.

### Ask

1. Is per-frame-`task_index`-only a **supported** v3 layout, or is `lerobot/libero` missing metadata it should have (i.e. worth regenerating)?
2. If both are supported, could the dataset format docs say so explicitly? Happy to send that docs PR once the intended semantics are confirmed.
