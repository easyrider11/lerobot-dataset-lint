"""Findings and the report container.

A rule emits Finding objects; the Report aggregates them per rule so a check
that fires on 4,000 frames reads as one line with a count and a few examples,
not 4,000 lines. Severity drives the exit code: ERROR means "this will break
or silently skew a training run", WARN means "probably hurts quality", INFO is
a fact worth knowing (e.g. which gripper convention the data uses).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

SEVERITIES = ("ERROR", "WARN", "INFO")


@dataclass
class Finding:
    rule: str          # e.g. "D003"
    severity: str      # ERROR | WARN | INFO
    message: str       # human sentence, self-contained
    episode: int | None = None
    feature: str | None = None

    def to_dict(self) -> dict:
        d = {"rule": self.rule, "severity": self.severity, "message": self.message}
        if self.episode is not None:
            d["episode"] = self.episode
        if self.feature is not None:
            d["feature"] = self.feature
        return d


@dataclass
class Report:
    repo: str
    version: str = "?"
    episodes_checked: int = 0
    episodes_total: int = 0
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # scan provenance, not findings
    facts: dict = field(default_factory=dict)  # machine-readable per-rule facts (--json)

    def add(self, rule: str, severity: str, message: str,
            episode: int | None = None, feature: str | None = None) -> None:
        assert severity in SEVERITIES, severity
        self.findings.append(Finding(rule, severity, message, episode, feature))

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def exit_code(self, strict: bool = False) -> int:
        if self.count("ERROR"):
            return 1
        if strict and self.count("WARN"):
            return 1
        return 0

    # -- rendering -----------------------------------------------------------

    MAX_EXAMPLES = 5

    def render(self) -> str:
        lines = [
            f"lerobot-lint · {self.repo}  ({self.version}, "
            f"checked {self.episodes_checked}/{self.episodes_total} episodes)"
        ]
        for note in self.notes:
            lines.append(f"  · {note}")
        if not self.findings:
            lines.append("  ✓ no findings")
            return "\n".join(lines)

        by_rule: dict[tuple[str, str], list[Finding]] = defaultdict(list)
        for f in self.findings:
            by_rule[(f.severity, f.rule)].append(f)

        order = {s: i for i, s in enumerate(SEVERITIES)}
        for (severity, rule) in sorted(by_rule, key=lambda k: (order[k[0]], k[1])):
            group = by_rule[(severity, rule)]
            mark = {"ERROR": "✗", "WARN": "!", "INFO": "i"}[severity]
            lines.append(f"  {mark} {severity:5s} {rule} ({len(group)}×)")
            for f in group[: self.MAX_EXAMPLES]:
                where = f" [ep {f.episode}]" if f.episode is not None else ""
                lines.append(f"      {f.message}{where}")
            if len(group) > self.MAX_EXAMPLES:
                lines.append(f"      … and {len(group) - self.MAX_EXAMPLES} more")
        lines.append(
            f"  = {self.count('ERROR')} error(s), {self.count('WARN')} warning(s), "
            f"{self.count('INFO')} info"
        )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "repo": self.repo,
            "version": self.version,
            "episodes_checked": self.episodes_checked,
            "episodes_total": self.episodes_total,
            "notes": self.notes,
            "counts": {s: self.count(s) for s in SEVERITIES},
            "findings": [f.to_dict() for f in self.findings],
            **({"facts": self.facts} if self.facts else {}),
        }, indent=2)
