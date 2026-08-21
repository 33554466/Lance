"""Decide which model tier a request deserves.

Most of what a household assistant is asked is trivial. Paying top-tier rates
for "what's the weather" is how a six-dollar month becomes a forty-dollar one.

This is a keyword-and-length heuristic on purpose. It is transparent, costs
nothing, and you can read the reason it chose a tier in the log. A learned
classifier is a refinement, not a prerequisite.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Route:
    tier: str
    model: str
    reason: str


class Router:
    def __init__(self, cfg: dict):
        r = cfg["router"]
        self.models = cfg["provider"]["models"]
        self.small_max_chars = r.get("small_max_chars", 80)
        self.escalate = [p.lower() for p in r.get("escalate_phrases", [])]
        self.markers = [m.lower() for m in r.get("complexity_markers", [])]

    def route(self, text: str) -> Route:
        t = text.lower().strip()

        # Explicit escalation always wins. Being able to say "think hard
        # about this" and have it mean something is worth more than any
        # amount of automatic classification.
        for phrase in self.escalate:
            if phrase in t:
                return Route("top", self.models["top"],
                             f"escalated by phrase: '{phrase}'")

        has_marker = any(m in t for m in self.markers)

        if len(t) <= self.small_max_chars and not has_marker:
            return Route("small", self.models["small"],
                         f"short ({len(t)} chars), no complexity markers")

        if has_marker:
            hit = next(m for m in self.markers if m in t)
            return Route("mid", self.models["mid"],
                         f"complexity marker: '{hit}'")

        return Route("mid", self.models["mid"], "default tier")
