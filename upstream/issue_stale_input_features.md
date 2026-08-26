### Bug

When finetuning with `--policy.path=...`, `make_policy` refreshes `output_features` from the dataset **unconditionally**, but `input_features` **only when empty** ([`factory.py` L333-L335 @ `bf31dd7`](https://github.com/huggingface/lerobot/blob/bf31dd794ffb43a944ef4977da2f9350a3ddc960/src/lerobot/policies/factory.py#L333-L335)):

```python
cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
if not cfg.input_features:
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
```

A config loaded from a pretrained checkpoint always has non-empty `input_features`, so a finetune onto a different embodiment **keeps the base model's feature spec forever**, while the normalization stats come from the new dataset. The asymmetry with `output_features` suggests this is accidental.

### Evidence: an official checkpoint contradicts itself

[`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) (finetuned from `smolvla_base` on `lerobot/libero`, per its own `train_config.json`):

| Artifact | `observation.state` | cameras |
|---|---|---|
| `config.json` `input_features` | shape **[6]** | camera1, camera2, **camera3** |
| `policy_preprocessor_step_5_normalizer_processor.safetensors` | mean/std shape **(8,)** | rename step creates only camera1, camera2 |
| training dataset `lerobot/libero` features | shape **[8]** | 2 image keys |

The `[6]` + three cameras are `smolvla_base`'s SO-100 features, carried through verbatim. The checkpoint *works* (SmolVLA pads state to `max_state_dim` and iterates over the images actually present), but anything that trusts `config.json` — integration code, config-driven eval, model cards — reads a wrong spec. We hit this integrating the checkpoint into an external runtime and initially built the wrong observation vector from the config.

### This is the root cause of #2418

The reproduction there is exactly this path (`--policy.path=lerobot/smolvla_base` + `--dataset.repo_id=HuggingFaceVLA/libero` + correct `rename_map`): training runs, **eval crashes with a tensor shape mismatch**, and the workaround that thread converged on — manually passing `--policy.input_features` with the true shapes — is precisely the refresh that L334 skips.

### Proposed fix

Mirror the `output_features` behaviour, with a loud warning on mismatch:

```python
new_inputs = {key: ft for key, ft in features.items() if key not in cfg.output_features}
if cfg.input_features and cfg.input_features != new_inputs:
    logging.warning(
        f"input_features from the pretrained config {cfg.input_features} do not match "
        f"the dataset features {new_inputs}; using the dataset's."
    )
cfg.input_features = new_inputs
```

One design point for maintainers: an **explicit** `--policy.input_features` CLI override (the #2418 workaround) flows into the same `cfg`, so "dataset always wins" would clobber it. If CLI-provided features can be distinguished from checkpoint-loaded ones, precedence should be CLI > dataset > checkpoint; if not, even just warning loudly on the mismatch (instead of silently keeping stale features) plus regenerating the affected official checkpoint configs (`smolvla_libero`: state `[8]`, camera1/camera2 only) would prevent both the #2418 class of failures and the self-contradicting configs.

Happy to send a PR for whichever semantics you prefer.

---
Found while building a dataset/config contract linter (#2326): its "stale stats dims" rule fired on the checkpoint. lerobot 0.4.4, verified unchanged on `main` @ `bf31dd7`.
