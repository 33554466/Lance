"""The path that runs when nobody has said anything.

Every other part of this appliance is reactive: you speak, it answers. Even
the scheduler only fires things you asked for. This module is the first thing
here that goes looking for a problem on its own.

The problem it looks for has no trigger by construction. A fact that
CONTRADICTS something gets caught on arrival — you say the new thing, and the
old one is corrected. But a fact that simply outlived its truth produces no
message, no contradiction, and no event. "Brendan's SLA is four hours" was
right in September and may be wrong in March, and nothing in the conversation
will ever say so. An event-driven design is structurally blind to it.

So: a clock. Once a day, walk the store and ask which facts have gone quiet
long enough to doubt.

Two properties keep this affordable and safe:

  * DETECTION IS FREE. Both detectors are plain SQL against a table with a few
    dozen rows. A sweep over a store where nothing has drifted costs nothing
    at all — no model call, no network. That is what makes running it every
    day reasonable rather than something you switch on when you remember.

  * IT NEVER EDITS. The sweep FLAGS. A flagged fact is still retrieved and
    still answers questions; it just carries "unconfirmed since March" into
    the prompt so the model hedges instead of asserting. Anything that would
    actually change a fact goes to you. Silently rewriting what the assistant
    believes about your family is how you get corruption you would never spot.
"""
from __future__ import annotations

import logging
import time

from .clock import days_between, freshness

log = logging.getLogger("assistant.sweep")

# Half-lives in days. A fact's belief halves over this long without evidence.
#
#   stable  facts that effectively never age — a birthday, a vendor ID.
#           Ten years is "not in the sweep's field of view", not an exemption.
#   slow    six months. People change jobs, cities, kit. Two or three checks
#           per year is attention without nagging.
#   fast    a month. Roughly how long "my current project" stays true.
HALF_LIFE_DAYS = {"stable": 3650, "slow": 180, "fast": 30}

# Below this, a fact is flagged. Deliberately ONE number used by both the
# sweep and the prompt block.
#
# The guide this borrows from set theirs at 0.7 and found that on a freshly
# seeded store, eight of eight facts came back marked "may be out of date"
# while the sweep correctly reported nothing wrong. Two subsystems with two
# different definitions of stale will contradict each other in front of you.
# 0.35 is roughly one and a half half-lives — late enough to mean something.
FRESH_FLOOR = 0.35


def half_life(volatility: str) -> int:
    return HALF_LIFE_DAYS.get(volatility, HALF_LIFE_DAYS["slow"])


def score(row, now: float) -> float:
    """How much this fact should still be believed, 1.0 down to 0."""
    age = days_between(now, row["last_verified"])
    return freshness(age, half_life(row["volatility"]))


def age_days(row, now: float) -> float:
    return max(0.0, days_between(now, row["last_verified"]))


class Sweep:
    """Finds rot nobody reported. Flags it. Changes nothing else."""

    def __init__(self, store, cfg: dict, clock=None):
        m = (cfg.get("memory", {}) or {}).get("sweep", {}) or {}
        self.enabled = bool(m.get("enabled", False))
        self.every_hours = float(m.get("every_hours", 24))
        self.floor = float(m.get("fresh_floor", FRESH_FLOOR))
        self.store = store
        self.clock = clock
        self._last = 0.0

    def _now(self) -> float:
        return self.clock.now() if self.clock else time.time()

    def due(self) -> bool:
        return (self.enabled
                and self._now() - self._last >= self.every_hours * 3600)

    def run(self) -> dict:
        """One pass. Returns what it found, for logging and the endpoint."""
        now = self._now()
        self._last = now
        rows = self.store.list_memories()

        # Detector 1: aged. A fact nobody has confirmed for long enough that
        # its own volatility says to doubt it.
        aged = [r for r in rows
                if r["status"] == "active" and score(r, now) < self.floor]

        # Detector 2: ended. A fact whose validity window has a close date
        # that has now passed — a dated thing that is simply over.
        ended = [r for r in rows
                 if r["valid_to"] is not None and r["valid_to"] <= now
                 and r["status"] in ("active", "stale")]

        flagged = self.store.flag_stale([r["id"] for r in aged], now)
        expired = 0
        for r in ended:
            if self.store.expire_memory(r["id"], now, "its window closed"):
                expired += 1

        result = {
            "at": now,
            "checked": len(rows),
            "flagged": flagged,
            "expired": expired,
            "already_flagged": sum(1 for r in rows if r["status"] == "stale"),
        }
        if flagged or expired:
            log.info("sweep: %d checked, %d flagged, %d expired",
                     len(rows), flagged, expired)
        return result

    def review(self) -> list[dict]:
        """What is waiting for you to confirm or correct. Newest doubt first."""
        now = self._now()
        out = []
        for r in self.store.list_memories():
            if r["status"] != "stale":
                continue
            out.append({
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "volatility": r["volatility"],
                "days_quiet": round(age_days(r, now)),
                "belief": round(score(r, now), 2),
            })
        return sorted(out, key=lambda d: d["belief"])
