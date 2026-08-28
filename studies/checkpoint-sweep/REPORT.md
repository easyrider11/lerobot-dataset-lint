# How many public lerobot checkpoints ship self-contradicting configs?

**2026-08-26 · sweep of the 300 most-downloaded `lerobot`-tagged model repos on the Hub**
Method: metadata only — `config.json`, processor JSONs, processor stats safetensors
(KB-sized, downloaded), `model.safetensors` headers (HTTP Range request, weights never
downloaded). Unauthenticated. Tool: `lerobot-lint` checkpoint mode (this repo, v0.2.0).

## Headline

**24 / 251 verifiable checkpoints (9.6%) provably carry a stale feature spec** —
their own `train_config.json` records a lerobot finetune (`policy.pretrained_path`
set) while `config.json`'s `observation.state` dims contradict the normalization
statistics shipped in the same repo ([huggingface/lerobot#4517](https://github.com/huggingface/lerobot/issues/4517):
`make_policy` refreshes `input_features` only when empty, so finetunes keep the
base checkpoint's spec forever).

**Seven are official `lerobot/` releases** (~38.5k combined downloads):

| checkpoint | config says | stats say | downloads |
|---|---|---|---|
| lerobot/smolvla_libero | state [6] | 8-dim | 17,735 |
| lerobot/smolvla_robotwin | state [6] | 14-dim | 10,646 |
| lerobot/smolvla_libero_plus | state [6] | 8-dim | 8,094 |
| lerobot/smolvla_metaworld | state [6] | 4-dim | 1,251 |
| lerobot/smolvla_robocasa | state [6] | 16-dim | 433 |
| lerobot/smolvla_vlabench | state [6] | 7-dim | 214 |
| lerobot/smolvla_robocerebra | state [6] | 8-dim | 117 |

All seven declare `smolvla_base`'s SO-100 spec (state [6], three cameras).

**Mechanism-specific rate: 39% (19/49) of all `smolvla_base` finetunes in the
sample are contradictory.** The sample even contains second-generation
propagation: `Sounderya/smolvla-ur3-mixed-20-80` inherits the stale `[6]` from
`Sounderya/smolvla-ur3-pickplace-2`, itself a stale finetune.

Secondary: **20 further checkpoints (SUSPECT)** declare phantom cameras their own
preprocessor rename map never produces (e.g. `camera3` inherited from base).

## Deliberately excluded from the headline

- **4 pi0-family repos** declaring `state [32]` (= `max_state_dim` padding): a
  different, deliberate-looking convention, not the finetune bug.
- **Per-channel stats conventions**: non-canonical STATE features (pointclouds)
  keep per-channel stats; size comparison is invalid there (rule K004 reports,
  never errors). One earlier false positive of this kind was found and fixed
  during the sweep.
- **1 case-by-case repo** (`zuoxingdong/evo1_libero`, custom arch, no
  train_config provenance).
- 40 UNVERIFIABLE (no stats found), 4 NOT_A_POLICY, 1 FETCH_ERROR.

## Reproduce

```bash
python studies/checkpoint-sweep/sweep.py 300   # writes results.jsonl (resumable)
python studies/checkpoint-sweep/analyze.py     # conservative scrub + this summary
```
