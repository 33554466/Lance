"""The rolling status wall.

What matters here is not that the probes work — test_selfcheck covers that —
but that the SCHEDULE works. Specifically:

  * the expensive probes do not run often (the whole reason the tiers exist),
  * one slow probe cannot stop the others being read,
  * a probe that fails gets looked at more often than its tier says,
  * and the snapshot tells the truth about how old each reading is.

Time is controlled, not waited for.

    python -m tests.test_monitor
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

from core import monitor as mon                                # noqa: E402
from core.monitor import Monitor, TIER_OF                      # noqa: E402
from core.selfcheck import FAIL, OK, WARN, Check               # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeChecker:
    """Stands in for SelfCheck with a probe table of known cost."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.status = {}

    def _probe(self, name):
        def fn():
            self.calls[name] = self.calls.get(name, 0) + 1
            return Check(name, "", self.status.get(name, OK), f"{name} fine")
        return fn

    def _plan(self, deep: bool = True):
        rows = [
            ("orchestrator", "Services", 2, self._probe("orchestrator")),
            ("audio service", "Services", 2, self._probe("audio service")),
            ("spend", "State", 4, self._probe("spend")),
            ("network", "Access", 8, self._probe("network")),
            ("disk space", "Storage", 4, self._probe("disk space")),
            ("printer", "Hardware", 10, self._probe("printer")),
            ("embedding model", "Models", 20, self._probe("embedding model")),
            ("model API", "Access", 25, self._probe("model API")),
        ]
        return rows


def _monitor(clock, **over) -> Monitor:
    cfg = {"status": {"per_tick": 8, **over}}
    m = Monitor(FakeChecker(), cfg)
    return m


def run(m: Monitor) -> None:
    asyncio.run(m.tick())


def test_tiers() -> None:
    print("\nWho gets asked how often")
    check("a set lookup is fast", TIER_OF["audio service"] == "fast")
    check("a socket is medium", TIER_OF["network"] == "medium")
    check("the printer is slow, because it is USB",
          TIER_OF["printer"] == "slow", TIER_OF["printer"])
    check("the embedding model is slow", TIER_OF["embedding model"] == "slow")
    check("anything that costs money is deep",
          all(TIER_OF[n] == "deep"
              for n in ("model API", "web search", "youtube")))
    check("nothing is unclassified into fast by accident",
          mon.DEFAULT_TIER == "medium")


def test_schedule() -> None:
    print("\nThe schedule")
    clock = FakeClock()
    mon.time.monotonic = clock
    mon.time.time = clock
    m = _monitor(clock)

    check("nothing has been read yet",
          all(r.at == 0.0 for r in m.readings.values()))
    check("...so everything is due at once", len(m._due()) == 8)

    run(m)
    calls = m.checker.calls
    check("the first tick reads everything once",
          all(calls.get(n) == 1 for n in calls), str(calls))

    # Five minutes. fast(5s) -> many, medium(60s) -> 5, slow(900s) -> 0 more,
    # deep(6h) -> 0 more.
    for _ in range(60):
        clock.advance(5)
        run(m)

    check("a fast probe was re-read all the way through",
          calls["orchestrator"] >= 55, str(calls["orchestrator"]))
    check("a medium probe was read about five times",
          4 <= calls["network"] <= 7, str(calls["network"]))
    check("the printer was NOT hammered", calls["printer"] == 1,
          f'{calls["printer"]} times in five minutes')
    check("...nor was the embedding model",
          calls["embedding model"] == 1, str(calls["embedding model"]))
    check("...and the paid probe ran exactly once",
          calls["model API"] == 1, str(calls["model API"]))

    # Six hours in, the expensive ones have come round, still rarely.
    for _ in range(6 * 60):
        clock.advance(60)
        run(m)
    check("the printer comes round every fifteen minutes or so",
          20 <= calls["printer"] <= 30, str(calls["printer"]))
    check("the paid probe is still down at a handful a day",
          calls["model API"] <= 3, str(calls["model API"]))


def test_failing_probes_get_watched() -> None:
    print("\nA failing probe is looked at more often")
    clock = FakeClock()
    mon.time.monotonic = clock
    mon.time.time = clock
    m = _monitor(clock, retry_failed_seconds=15)
    run(m)
    before = m.checker.calls["printer"]

    m.checker.status["printer"] = FAIL
    # Move past the slow tier once so the failure is picked up.
    clock.advance(900)
    run(m)
    check("the failure was noticed",
          m.readings["printer"].check.status == FAIL)

    for _ in range(10):          # 150 seconds
        clock.advance(15)
        run(m)
    after = m.checker.calls["printer"] - before
    check("...and it is now re-read every fifteen seconds, not every fifteen "
          "minutes", after >= 10, f"{after} reads in 150s")

    m.checker.status["printer"] = OK
    clock.advance(15)
    run(m)
    check("recovery is noticed", m.readings["printer"].check.status == OK)
    check("...and it goes back to its slow tier",
          m.readings["printer"].due - clock.now > 800,
          str(m.readings["printer"].due - clock.now))
    check("...but the wall remembers it was bad",
          m.snapshot()["checks"][
              [c["name"] for c in m.snapshot()["checks"]].index("printer")
          ]["recovered"] is True)


def test_snapshot() -> None:
    print("\nWhat the screen is handed")
    clock = FakeClock()
    mon.time.monotonic = clock
    mon.time.time = clock
    m = _monitor(clock)
    run(m)
    snap = m.snapshot()
    check("a verdict for the whole appliance", snap["verdict"] == OK)
    check("counts add up to the probe count",
          sum(snap["counts"].values()) == len(snap["checks"]))
    check("every row carries its tier and interval",
          all(r["tier"] and r["every"] for r in snap["checks"]))
    check("every row carries an age",
          all(r["age"] is not None for r in snap["checks"]))
    check("nothing is a problem yet", snap["problems"] == [])

    m.checker.status["disk space"] = WARN
    clock.advance(60)
    run(m)
    snap = m.snapshot()
    check("a warning shows up as a problem",
          [p["name"] for p in snap["problems"]] == ["disk space"],
          str(snap["problems"]))
    check("...and the verdict follows the worst one",
          snap["verdict"] == WARN)

    m.checker.status["network"] = FAIL
    clock.advance(60)
    run(m)
    check("a failure outranks a warning", m.snapshot()["verdict"] == FAIL)

    # Age is the honest part: a slow probe read once is minutes old and must
    # say so rather than showing a confident dot.
    clock.advance(600)
    snap = m.snapshot()
    printer = next(r for r in snap["checks"] if r["name"] == "printer")
    check("an old reading reports its real age",
          printer["age"] > 600, str(printer["age"]))
    check("...against an interval the screen can compare it to",
          printer["every"] == 900, str(printer["every"]))


def test_a_broken_probe_does_not_stop_the_others() -> None:
    print("\nOne bad probe is not an outage")
    clock = FakeClock()
    mon.time.monotonic = clock
    mon.time.time = clock
    m = _monitor(clock)

    def explode():
        raise RuntimeError("usb is on fire")

    m._plan["printer"] = ("Hardware", 10, explode)
    run(m)
    check("the exception became a failed check, not a crash",
          m.readings["printer"].check.status == FAIL)
    check("...saying so in words",
          "usb is on fire" in m.readings["printer"].check.detail,
          m.readings["printer"].check.detail)
    check("...and carrying a fix", bool(m.readings["printer"].check.fix))
    check("everything else was still read",
          m.readings["orchestrator"].check.status == OK)



def test_every_real_probe_is_classified() -> None:
    """No probe may reach the wall without someone deciding its cost.

    The default tier is medium, which is a reasonable landing place but a
    terrible one to arrive at by accident: a new probe that shells out to the
    printer would quietly get polled every sixty seconds forever. So the real
    plan is walked and every name must appear in the table on purpose.
    """
    print("\nEvery probe has been thought about")
    import yaml
    from core.db import Store
    from core.selfcheck import SelfCheck
    from core.tools import build_tools

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    store = Store(":memory:")
    tools = build_tools(cfg, store)
    checker = SelfCheck(cfg, store, tools=tools, hub=None)

    names = [n for n, _g, _b, _f in checker._plan(deep=True)]
    missing = [n for n in names if n not in TIER_OF]
    check("every probe in the real plan has an explicit tier",
          not missing, str(missing))

    m = Monitor(checker, cfg)
    snap = m.snapshot()
    check("the wall covers the whole plan",
          len(snap["checks"]) == len(names),
          f'{len(snap["checks"])} vs {len(names)}')
    check("the paid probes are in it, at the deep tier",
          all(r["tier"] == "deep" for r in snap["checks"]
              if r["name"] in ("model API", "web search", "youtube")))
    check("the printer is not on a fast tier",
          next(r["every"] for r in snap["checks"] if r["name"] == "printer")
          >= 600)
    check("nothing has been read before the first tick",
          all(r["runs"] == 0 for r in snap["checks"]))


def main() -> int:
    real_mono, real_time = mon.time.monotonic, mon.time.time
    try:
        test_tiers()
        test_schedule()
        test_failing_probes_get_watched()
        test_snapshot()
        test_a_broken_probe_does_not_stop_the_others()
        test_every_real_probe_is_classified()
    finally:
        mon.time.monotonic, mon.time.time = real_mono, real_time
    print()
    if FAILURES:
        print(f"\033[1;31m{len(FAILURES)} failed\033[0m")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\033[1;32mall passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
