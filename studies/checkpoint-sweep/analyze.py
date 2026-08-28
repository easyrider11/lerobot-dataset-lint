"""Scrub + summarize the sweep, conservatively.

The headline number only counts checkpoints where the #4517 mechanism is
provable from the artifact itself: train_config records a lerobot finetune
(policy.pretrained_path set) AND the config state dims disagree with the
shipped normalizer stats. pi0/pi0-family checkpoints that declare the padded
max_state_dim as their feature shape are a different (deliberate-looking)
convention and are EXCLUDED from the headline, listed separately.
"""

from __future__ import annotations

import json
from pathlib import Path

rows = [json.loads(line) for line in
        (Path(__file__).parent / "results.jsonl").read_text().splitlines() if line]

verifiable = [r for r in rows if r["classification"] in ("CLEAN", "CONTRADICTORY", "SUSPECT")]
bad = [r for r in rows if r["classification"] == "CONTRADICTORY"]

# padded-convention family: declared state == 32 (pi0-style max_state_dim) or
# pi0 types - a deliberate convention, not the stale-finetune bug
padded = [r for r in bad if r.get("state_cfg") == 32 or str(r.get("policy_type")).startswith("pi0")]
core = [r for r in bad if r not in padded]
finetune_proven = [r for r in core if r.get("finetuned_from")]
official = [r for r in finetune_proven if r["repo"].startswith("lerobot/")]

print(f"swept: {len(rows)}   verifiable: {len(verifiable)}   contradictory: {len(bad)}")
print(f"\nHEADLINE (mechanism-proven stale finetunes): {len(finetune_proven)}"
      f" / {len(verifiable)} verifiable = {len(finetune_proven)/len(verifiable):.1%}")
print(f"  of which OFFICIAL lerobot/ checkpoints: {len(official)}")
for r in sorted(official, key=lambda r: -(r.get('downloads') or 0)):
    print(f"    {r['repo']}: cfg state [{r['state_cfg']}] vs stats {r['state_stats']}-dim"
          f"  ({r.get('downloads')} downloads)")
comm = [r for r in finetune_proven if not r["repo"].startswith("lerobot/")]
print(f"  community: {len(comm)}")
for r in sorted(comm, key=lambda r: -(r.get('downloads') or 0))[:10]:
    print(f"    {r['repo']}: [{r['state_cfg']}] vs {r['state_stats']}  <- {str(r['finetuned_from'])[:45]}")
print(f"\nEXCLUDED as padded-convention (pi0-family, declared==32): {len(padded)}")
for r in padded:
    print(f"    {r['repo']} ({r.get('policy_type')}): [{r.get('state_cfg')}] vs {r.get('state_stats')}")
other = [r for r in core if not r.get("finetuned_from")]
print(f"\nOTHER contradictory (no train_config provenance - case-by-case): {len(other)}")
for r in other:
    print(f"    {r['repo']} ({r.get('policy_type')}): state [{r.get('state_cfg')}] vs {r.get('state_stats')}"
          f"  action [{r.get('action_cfg')}] vs {r.get('action_stats')}  rules={r.get('rules')}")
sus = [r for r in rows if r["classification"] == "SUSPECT"]
print(f"\nSUSPECT (phantom cameras only): {len(sus)}")
smol_ft = [r for r in verifiable if str(r.get("finetuned_from") or "") == "lerobot/smolvla_base"]
smol_bad = [r for r in smol_ft if r["classification"] == "CONTRADICTORY"]
if smol_ft:
    print(f"\namong finetunes FROM lerobot/smolvla_base specifically: "
          f"{len(smol_bad)}/{len(smol_ft)} = {len(smol_bad)/len(smol_ft):.0%} contradictory")
