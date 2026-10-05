"""The status wall: the self-check, run continuously instead of once a day.

The daily diagnostic answers "is everything well?" when you ask. This answers
it before you ask, and keeps answering it. Same twenty-six probes, same Check
objects, same fixes — the only new idea here is WHEN each one runs.

That turns out to be the whole design problem. "Ping everything constantly" is
not a thing you can do to this appliance:

  * the printer probe talks to the device over USB. Hammering it while a
    receipt is going through is how you wedge a printer, and we have already
    spent two days on a wedged microphone.
  * the model API probe makes a real Anthropic call. At five-second intervals
    that is a bill, not a diagnostic.
  * the embedding model probe loads an ONNX model.
  * meanwhile "is the audio service connected" is a set lookup and costs
    nothing at all.

So every probe gets a tier, and a tier is just how stale its answer is allowed
to get. Cheap things are near-live; expensive things are minutes or hours old
and the screen says so rather than pretending otherwise. An age on every
reading is not decoration — a green dot that has not been re-checked since
Tuesday is a lie, and the cure is to show the timestamp, not to poll harder.

One probe runs at a time, on a worker thread, under the same deadline the
daily check uses. A probe that hangs costs its own tier and nothing else.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .selfcheck import FAIL, GROUPS, OK, SKIP, WARN, Check, _deadline

log = logging.getLogger("assistant.monitor")

# How often each tier is re-read, in seconds. These are defaults; the config
# can move them.
TIERS = {
    "fast": 5,
    "medium": 60,
    "slow": 900,
    "deep": 6 * 3600,
}

# Which tier each probe belongs to, by the name the self-check gives it.
#
# The rule: fast is memory or a set lookup, medium is a syscall or a socket,
# slow touches hardware or loads a model, deep costs money. Anything not
# listed falls to medium, which is the safe place for a probe nobody has
# thought about yet.
TIER_OF = {
    # fast — nothing but process memory and SQLite
    "microphone hearing":   "fast",
    "orchestrator":         "fast",
    "audio service":        "fast",
    "display":              "fast",
    "open sessions":        "fast",
    "spend":                "fast",
    "reminders":            "fast",

    # medium — a syscall, a stat, a socket, a subprocess
    "microphone on the bus": "medium",
    "speaker output":        "medium",
    "watchdog":              "medium",
    "network":               "medium",
    "mpv":                   "medium",
    "disk space":            "medium",
    "database":              "medium",
    "notes folder":          "medium",
    "workouts folder":       "medium",
    "playbooks":             "medium",
    "memory index":          "medium",
    "stale facts":           "medium",
    "wake word accuracy":    "medium",
    "transcription model":   "medium",
    "voice":                 "medium",
    "wake word model":       "medium",
    "yt-dlp freshness":      "slow",

    # slow — touches the hardware, or loads something
    "printer":               "slow",
    "embedding model":       "slow",

    # deep — a real network round trip that costs money or a rate limit
    "model API":             "deep",
    "web search":            "deep",
    "youtube":               "deep",
}

DEFAULT_TIER = "medium"


@dataclass
class Reading:
    """One probe's latest answer, and when it was taken."""
    check: Check
    at: float                      # wall clock, for display
    tier: str = DEFAULT_TIER
    due: float = 0.0               # monotonic, when to re-read
    runs: int = 0
    # The last time this probe was NOT ok. Lets the wall say "printer: ok,
    # but it was failing four minutes ago", which is the difference between
    # a healthy appliance and one that is flapping.
    last_bad: float = 0.0
    last_bad_detail: str = ""

    def age(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.at


class Monitor:
    """Runs the self-check's probes on a rolling schedule and caches them.

    Driven by the scheduler's one-second tick, like everything else that has
    to survive a restart: no task of its own to quietly disappear.
    """

    def __init__(self, checker, cfg: dict | None = None):
        self.checker = checker
        c = ((cfg or {}).get("status", {}) or {})
        self.enabled = bool(c.get("enabled", True))
        every = (c.get("every", {}) or {})
        self.tiers = {k: max(1, int(every.get(k, v))) for k, v in TIERS.items()}
        # Never more than this many probes started per tick. The tick is one
        # second; letting a whole tier come due at once would put twenty
        # probes on threads together and the printer would be one of them.
        self.per_tick = max(1, int(c.get("per_tick", 2)))
        # Whether the appliance's own screen rests on the full wall when
        # nothing else needs it. There is no keyboard on that machine, so
        # this is not a shortcut you can press — it is the default view or
        # it is nothing.
        self.resting = bool(c.get("ambient", True))
        # A probe that has just been read is not re-read for its whole
        # interval even if the tier is fast — but a FAILING probe is worth
        # looking at more often, because that is the one you are watching.
        self.retry_bad = max(1, int(c.get("retry_failed_seconds", 15)))

        self.readings: dict[str, Reading] = {}
        self.order: list[str] = []
        self._plan: dict[str, tuple[str, int, object]] = {}
        self._running = False
        self.started = time.time()
        self._build_plan()

    # -- the plan ------------------------------------------------------

    def _build_plan(self) -> None:
        """Borrow the self-check's own probe table, deep included.

        Deliberately not a second list of probes. A status wall that drifts
        out of step with the diagnostic is worse than no status wall, because
        the two would disagree and you would not know which to believe.
        """
        for name, group, budget, fn in self.checker._plan(deep=True):
            self._plan[name] = (group, budget, fn)
            self.order.append(name)
            tier = TIER_OF.get(name, DEFAULT_TIER)
            self.readings[name] = Reading(
                check=Check(name, group, SKIP, "not checked yet"),
                at=0.0, tier=tier, due=0.0)

    # -- the tick ------------------------------------------------------

    async def tick(self) -> None:
        """Run whatever is due, up to per_tick probes. Never raises."""
        if not self.enabled or self._running:
            return
        due = self._due()
        if not due:
            return
        self._running = True
        try:
            for name in due[:self.per_tick]:
                await self._run_one(name)
        finally:
            self._running = False

    def _due(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        ready = [(self.readings[n].due, n) for n in self.order
                 if self.readings[n].due <= now]
        # Oldest debt first, so a slow probe that came due during a burst is
        # not starved by fast ones coming round again.
        ready.sort()
        return [n for _due, n in ready]

    async def _run_one(self, name: str) -> None:
        group, budget, fn = self._plan[name]
        started = time.monotonic()
        try:
            check = await asyncio.to_thread(_deadline, fn, budget)
        except TimeoutError as exc:
            check = Check(name, group, FAIL, str(exc),
                          "Something is not answering. Look at the journal "
                          "for this subsystem.")
        except Exception as exc:  # noqa: BLE001
            log.exception("probe %r blew up", name)
            check = Check(name, group, FAIL,
                          f"the check itself failed: "
                          f"{type(exc).__name__}: {exc}",
                          "This is a bug in the diagnostic, not necessarily "
                          "in the thing it checks.")
        if check is None:
            check = Check(name, group, SKIP, "returned nothing")
        check.name, check.group = name, group
        check.seconds = time.monotonic() - started

        was = self.readings[name]
        now_wall, now_mono = time.time(), time.monotonic()
        # A probe that is failing gets looked at more often than its tier
        # says. That is the one you are standing in front of the screen
        # waiting to go green.
        interval = (self.retry_bad if check.status in (WARN, FAIL)
                    else self.tiers.get(was.tier, TIERS[DEFAULT_TIER]))
        reading = Reading(check=check, at=now_wall, tier=was.tier,
                          due=now_mono + interval, runs=was.runs + 1,
                          last_bad=was.last_bad,
                          last_bad_detail=was.last_bad_detail)
        if check.status in (WARN, FAIL):
            reading.last_bad = now_wall
            reading.last_bad_detail = check.detail
            if was.check.status not in (WARN, FAIL):
                log.warning("monitor: %s went %s — %s",
                            name, check.status, check.detail)
        elif was.check.status in (WARN, FAIL):
            log.info("monitor: %s recovered", name)
        self.readings[name] = reading

    # -- what the screen asks for --------------------------------------

    def counts(self) -> dict[str, int]:
        out = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for r in self.readings.values():
            out[r.check.status] = out.get(r.check.status, 0) + 1
        return out

    @property
    def verdict(self) -> str:
        """One word for the whole appliance."""
        counts = self.counts()
        if counts[FAIL]:
            return FAIL
        if counts[WARN]:
            return WARN
        if counts[OK]:
            return OK
        return SKIP

    def snapshot(self) -> dict:
        now = time.time()
        rows = []
        for name in self.order:
            r = self.readings[name]
            rows.append({
                "name": name,
                "group": r.check.group,
                "status": r.check.status,
                "detail": r.check.detail,
                "fix": r.check.fix,
                "tier": r.tier,
                "every": self.tiers.get(r.tier, 0),
                "age": round(r.age(now), 1) if r.at else None,
                "seconds": round(r.check.seconds, 3),
                "runs": r.runs,
                "recovered": bool(r.last_bad and r.check.status == OK),
                "last_bad": round(now - r.last_bad, 1) if r.last_bad else None,
            })
        counts = self.counts()
        return {
            "at": now,
            "verdict": self.verdict,
            "resting": self.resting,
            "uptime": round(now - self.started, 1),
            "counts": counts,
            "groups": list(GROUPS),
            "checks": rows,
            # Named separately so a screen can lead with the problems without
            # filtering twenty-six rows in JavaScript.
            "problems": [r for r in rows if r["status"] in (WARN, FAIL)],
        }
