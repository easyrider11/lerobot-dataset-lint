Root cause found — this is a bug in `make_policy`, not in your command. `factory.py` refreshes `output_features` from the dataset unconditionally but `input_features` **only when empty**, and a config loaded via `--policy.path` is never empty — so your finetune kept `smolvla_base`'s `observation.state: [6]` (SO-100) while the normalizer stats were built from libero's 8-dim state. Training runs (state is padded internally); eval then trips over the self-contradicting spec.

That's also why the `--policy.input_features` workaround above works: it does manually what the refresh should do automatically.

Filed with evidence (including an official checkpoint, `lerobot/smolvla_libero`, that ships this exact contradiction in its `config.json`) and a proposed fix: see #4517.
