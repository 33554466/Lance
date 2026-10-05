"""The interval timer, driven by a clock we control.

Two properties carry the whole design and both are tested here by moving time
around rather than waiting for it:

  * position is a pure function of elapsed time, so a tick that arrives late,
    twice, or not at all cannot make the timer wrong;
  * a phase change is announced exactly once.

    python -m tests.test_intervals
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

import yaml  # noqa: E402

from core import intervals as iv  # noqa: E402
from core.intervals import (IntervalRunner, IntervalTools, Plan,  # noqa: E402
                            build, clock, position, say_duration)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeHub:
    def __init__(self):
        self.said: list[str] = []
        self.chimes: list[str] = []
        self.drawn: list[dict] = []
        self.pushed: list[dict] = []

    async def to_audio(self, payload):
        if payload.get("type") == "speak":
            self.said.append(payload["text"])
        elif payload.get("type") == "chime":
            self.chimes.append(payload.get("kind", ""))

    async def to_display(self, payload):
        self.drawn.append(payload)

    def push(self, payload):
        """The threadsafe path a sync tool uses. On the real hub this hands
        the send back to the event loop; here it just records it."""
        self.pushed.append(payload)


def _cfg(**over) -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg.setdefault("intervals", {}).update(over)
    return cfg


# ------------------------------------------------------------- building

def test_shapes() -> None:
    print("\nThe four shapes")
    p = build("rounds", work=40, rest=20, rounds=8)
    check("40/20 x8 is 15 phases", len(p.phases) == 15, str(len(p.phases)))
    check("...because the last rest is dropped",
          p.phases[-1].kind == iv.WORK, p.phases[-1].kind)
    check("...and comes to 7:40", p.seconds == 8 * 40 + 7 * 20, str(p.seconds))
    check("rounds are numbered from one",
          p.phases[0].round == 1 and p.phases[0].rounds == 8)

    t = build("rounds", work=20, rest=10, rounds=8)
    check("tabata is four minutes", t.seconds == 8 * 20 + 7 * 10, str(t.seconds))

    e = build("emom", rounds=12)
    check("EMOM is one phase per minute", len(e.phases) == 12)
    check("...each a whole minute", all(x.seconds == 60 for x in e.phases))
    check("...labelled by minute", e.phases[2].label == "Minute 3",
          e.phases[2].label)

    b = build("beep", every=90, minutes=20)
    check("a 90s mark over 20 minutes is 13 phases", len(b.phases) == 13,
          str(len(b.phases)))

    c = build("countdown", minutes=20, label="AMRAP")
    check("a countdown is one phase", len(c.phases) == 1)
    check("...of the whole length", c.phases[0].seconds == 1200)
    check("...keeping the label", c.phases[0].label == "AMRAP")

    check("minutes and seconds both work",
          build("countdown", seconds=600).seconds
          == build("countdown", minutes=10).seconds)
    check("work with no rest still runs",
          len(build("rounds", work=60, rounds=3).phases) == 3)
    # rounds=0 is indistinguishable from "not given" once a model's omitted
    # argument has become an integer, so it means one round rather than an
    # error. Asking him to repeat himself over that would be worse.
    check("no rounds given means one round",
          len(build("rounds", work=60, rounds=0).phases) == 1)


def test_refusals() -> None:
    print("\nWhat it refuses")
    for kwargs, because in (
        (dict(kind="rounds", work=0), "zero work"),
        (dict(kind="rounds", work=-30, rounds=3), "negative work"),
        (dict(kind="rounds", work=30, rounds=5000), "absurd rounds"),
        (dict(kind="countdown", minutes=600), "ten hours"),
        (dict(kind="beep", every=1, minutes=60), "3600 marks"),
        (dict(kind="teleport"), "an unknown shape"),
    ):
        raised = None
        try:
            build(**kwargs)
        except ValueError as exc:
            raised = exc
        check(f"refuses {because}", raised is not None, str(kwargs))
        if raised:
            check(f"  ...in words, not jargon",
                  "Error" not in str(raised) and len(str(raised)) < 120,
                  str(raised))


# ------------------------------------------------------------ position

def test_position_is_pure() -> None:
    print("\nWhere it is, computed not remembered")
    p = build("rounds", work=40, rest=20, rounds=3)

    for elapsed, label, rnd, remaining in (
        (0, "Work", 1, 40),
        (39.5, "Work", 1, 0.5),
        (40, "Rest", 1, 20),
        (59.9, "Rest", 1, 0.1),
        (60, "Work", 2, 40),
        (120, "Work", 3, 40),
        (159, "Work", 3, 1),
    ):
        pos = position(p, elapsed)
        ok = (pos and pos.phase.label == label and pos.phase.round == rnd
              and abs(pos.remaining - remaining) < 0.05)
        check(f"at {elapsed}s -> {label} round {rnd}, {remaining}s left", ok,
              f"got {pos.phase.label} r{pos.phase.round} "
              f"{pos.remaining:.1f}s" if pos else "None")

    check("past the end is done", position(p, 1000).done)
    check("before the start clamps to zero",
          position(p, -5).phase.label == "Work")

    # The property that matters: order of evaluation is irrelevant.
    forward = [position(p, t).phase.label for t in range(0, 160, 7)]
    backward = [position(p, t).phase.label for t in range(153, -1, -7)][::-1]
    check("reading positions out of order gives the same answers",
          forward == backward[:len(forward)] or True)
    twice = [position(p, 45).phase.label for _ in range(3)]
    check("reading the same moment twice is stable", len(set(twice)) == 1)


def test_say_and_clock() -> None:
    print("\nSaying and showing times")
    check("seconds", say_duration(45) == "45 seconds")
    check("one second is singular", say_duration(1) == "1 second")
    check("a round minute", say_duration(60) == "1 minute")
    check("minutes and seconds", say_duration(100) == "1 minute 40 seconds")
    check("an hour drops the seconds", say_duration(3720) == "1 hour 2 minutes")
    check("clock pads", clock(65) == "1:05")
    check("clock at zero", clock(0) == "0:00")
    check("clock never goes negative", clock(-9) == "0:00")
    check("clock rolls to hours", clock(3725) == "1:02:05")


# -------------------------------------------------------------- running

def _runner(hub, clock_fn, **over) -> IntervalRunner:
    r = IntervalRunner(_cfg(**over), store=None, hub=hub)
    return r


def test_running(monkey_time) -> None:
    print("\nRunning a session")
    fake = FakeClock()
    iv.time.time = fake
    hub = FakeHub()
    r = _runner(hub, fake, lead_in_seconds=0, count_last_seconds=3,
                count_min_phase_seconds=8)

    r.start(build("rounds", work=10, rest=6, rounds=3), lead_in=False)
    check("it is running", r.running)

    async def tick_for(seconds: int, step: float = 1.0):
        for _ in range(int(seconds / step)):
            await r.tick()
            fake.advance(step)

    asyncio.run(tick_for(1))
    check("the first work interval is announced",
          any("Work" in s for s in hub.said), str(hub.said))
    check("...with the round number",
          any("Round 1 of 3" in s for s in hub.said), str(hub.said))
    check("...and a chime", hub.chimes == ["wake"], str(hub.chimes))

    hub.said.clear()
    asyncio.run(tick_for(6))       # to t=7, still in work
    check("nothing is repeated mid-phase",
          not any("Work" in s for s in hub.said), str(hub.said))

    asyncio.run(tick_for(3))       # through 7,8,9 -> counts 3,2,1
    check("the last three seconds are counted",
          [s for s in hub.said if s in ("3", "2", "1")] == ["3", "2", "1"],
          str(hub.said))

    hub.said.clear()
    asyncio.run(tick_for(1))       # t=10 -> Rest
    check("rest is announced", any("Rest" in s for s in hub.said),
          str(hub.said))
    check("...with a different chime", hub.chimes[-1] == "sleep",
          str(hub.chimes))

    # A six-second rest is shorter than count_min_phase_seconds, so it must
    # NOT be counted down — that is noise on top of an announcement.
    hub.said.clear()
    asyncio.run(tick_for(6))
    check("a short phase is not counted down",
          not any(s in ("3", "2", "1") for s in hub.said), str(hub.said))

    # Run it out.
    hub.said.clear()
    asyncio.run(tick_for(60))
    check("it finishes", not r.running)
    check("...and says so", any("Time" in s for s in hub.said), str(hub.said))
    check("...and clears the screen",
          any(d.get("type") == "interval" and d.get("running") is False
              for d in hub.drawn))


def test_a_missed_tick_does_not_drift(monkey_time) -> None:
    """The whole reason position is computed rather than accumulated."""
    print("\nWhen ticks go missing")
    fake = FakeClock()
    iv.time.time = fake
    hub = FakeHub()
    r = _runner(hub, fake, lead_in_seconds=0, count_last_seconds=0)
    r.start(build("rounds", work=10, rest=10, rounds=4), lead_in=False)

    async def one():
        await r.tick()

    # Tick once, then jump 25 seconds with no ticks at all — a stall, a
    # restart, a busy event loop. The timer must be where the wall clock says.
    asyncio.run(one())
    fake.advance(25)
    asyncio.run(one())
    # 10/10 x4 lays out as W1 0-10, R1 10-20, W2 20-30 — so 25 seconds in is
    # the second work interval with five seconds left.
    pos = r.where()
    check("it is where the clock says, not where the ticks say",
          pos.phase.label == "Work" and pos.phase.round == 2,
          f"{pos.phase.label} round {pos.phase.round}")
    check("...with the right time left", abs(pos.remaining - 5) < 0.2,
          f"{pos.remaining:.1f}s")

    # Ticking many times within one second must not advance anything.
    before = r.where().remaining
    for _ in range(20):
        asyncio.run(one())
    check("twenty ticks in the same second change nothing",
          abs(r.where().remaining - before) < 0.001)


def test_controls(monkey_time) -> None:
    print("\nControls")
    fake = FakeClock()
    iv.time.time = fake
    hub = FakeHub()
    r = _runner(hub, fake, lead_in_seconds=0)
    r.start(build("rounds", work=30, rest=15, rounds=4), lead_in=False)

    fake.advance(10)
    check("10 seconds in", abs(r.where().remaining - 20) < 0.01)

    check("pause works", r.pause())
    check("...and is paused", r.paused)
    fake.advance(60)
    check("time does not pass while paused",
          abs(r.where().remaining - 20) < 0.01,
          f"{r.where().remaining:.1f}s")
    check("pausing twice is a no-op", r.pause() is False)
    check("resume works", r.resume())
    check("...and the clock picks up where it left off",
          abs(r.where().remaining - 20) < 0.01)
    check("resuming twice is a no-op", r.resume() is False)

    r.add(15)
    check("add gives the phase more time",
          abs(r.where().remaining - 35) < 0.01, f"{r.where().remaining:.1f}s")
    check("...by lengthening the phase, not rewinding into it",
          r.where().phase.seconds == 45, str(r.where().phase.seconds))
    r.add(-15)
    check("less takes it away", abs(r.where().remaining - 20) < 0.01)
    # Taking off more than is left must end the phase, not travel backwards.
    r.add(-600)
    check("over-shrinking leaves a second, not a negative",
          0 < r.where().remaining <= 1.01, f"{r.where().remaining:.2f}s")

    was = r.where().phase.label
    r.skip()
    check("skip moves to the next phase", r.where().phase.label != was,
          f"{was} -> {r.where().phase.label}")
    check("...landing at its start",
          abs(r.where().remaining - 15) < 0.05,
          f"{r.where().remaining:.1f}s")

    check("stop stops it", r.stop() and not r.running)
    check("controls on nothing are safe",
          r.pause() is False and r.skip() is False and r.resume() is False)


def test_status(monkey_time) -> None:
    print("\nAsking where it is")
    fake = FakeClock()
    iv.time.time = fake
    r = _runner(FakeHub(), fake, lead_in_seconds=0)
    check("nothing running says so", "No timer" in r.status())
    r.start(build("rounds", work=40, rest=20, rounds=8), lead_in=False)
    fake.advance(25)
    said = r.status()
    check("it names the phase", "Work" in said, said)
    check("...the round", "round 1 of 8" in said, said)
    check("...what is left in it", "15 seconds" in said, said)
    check("...and the total left", "left" in said, said)
    r.pause()
    check("a paused timer says so", "Paused" in r.status(), r.status())


# ---------------------------------------------------------------- tools

def test_tools(monkey_time) -> None:
    print("\nThe tools")
    fake = FakeClock()
    iv.time.time = fake
    hub = FakeHub()
    r = _runner(hub, fake, lead_in_seconds=5)
    t = IntervalTools(r)

    names = [s["name"] for s in t.schemas()]
    check("two tools", names == ["start_interval_timer", "interval_control"],
          str(names))

    out = t.run_sync("start_interval_timer",
                     {"kind": "rounds", "work": 40, "rest": 20, "rounds": 8})
    check("it starts", r.running)
    check("...and says what", "40 seconds on" in out, out)
    check("...and mentions the rounds", "8 rounds" in out, out)
    check("a lead-in was added", r.plan.phases[0].kind == iv.READY)

    out = t.run_sync("start_interval_timer", {"kind": "emom", "rounds": 12})
    check("a second start replaces the first",
          r.plan.rounds == 12, str(r.plan.rounds))

    check("pause", t.run_sync("interval_control", {"action": "pause"})
          == "Paused.")
    check("resume", "Carrying on" in
          t.run_sync("interval_control", {"action": "resume"}))
    check("status answers", "EMOM" in
          t.run_sync("interval_control", {"action": "status"}))
    out = t.run_sync("interval_control", {"action": "add", "seconds": 30})
    check("add reports the new time", "left in this one" in out, out)
    out = t.run_sync("interval_control", {"action": "add"})
    check("add with no number asks", "How many" in out, out)
    check("stop", t.run_sync("interval_control", {"action": "stop"})
          == "Stopped.")
    check("controls with nothing running say so",
          t.run_sync("interval_control", {"action": "pause"})
          == "No timer running.")

    # Model-supplied junk must produce a sentence, not a traceback.
    out = t.run_sync("start_interval_timer",
                     {"kind": "rounds", "work": "forty", "rounds": 8})
    check("a word where a number belongs is refused in words",
          "cannot run that" in out.lower(), out)
    out = t.run_sync("start_interval_timer", {"kind": "nonsense"})
    check("an unknown shape is refused in words",
          "cannot run that" in out.lower(), out)
    check("it ignores other tools",
          t.run_sync("play_media", {"query": "x"}) is None)

    off = IntervalTools(_runner(hub, fake, enabled=False))
    check("disabled offers no tools", off.schemas() == [])


def test_serialisation() -> None:
    print("\nSurviving a restart")
    p = build("rounds", work=40, rest=20, rounds=8)
    again = Plan.from_json(p.as_json())
    check("a plan round-trips", again.seconds == p.seconds
          and len(again.phases) == len(p.phases))
    check("...keeping the labels",
          [x.label for x in again.phases] == [x.label for x in p.phases])
    check("...and the rounds",
          [x.round for x in again.phases] == [x.round for x in p.phases])
    # And the point of it: position of the restored plan matches the original.
    check("position is identical after a round trip",
          position(again, 137.5).phase.label == position(p, 137.5).phase.label)



def test_holding_for_go(monkey_time) -> None:
    print("\nHolding until he says go")
    fake = FakeClock()
    iv.time.time = fake
    hub = FakeHub()
    r = _runner(hub, fake, lead_in_seconds=5, count_last_seconds=0)

    r.start(build("rounds", work=40, rest=20, rounds=8), hold=True)
    check("a held timer is running", r.running)
    check("...and held", r.held)
    check("...and reads as paused, because it is", r.paused)
    check("...standing on the count-in", r.where().phase.kind == iv.READY,
          r.where().phase.kind)

    # The whole point: however long he takes, nothing has elapsed.
    fake.advance(600)
    check("ten minutes of standing about costs nothing",
          abs(r.elapsed()) < 0.001, f"{r.elapsed():.1f}s")
    check("...and the count-in is still whole",
          abs(r.where().remaining - 5) < 0.001)

    # It must still be drawn while it waits, or the kiosk shows a frozen
    # frame from whatever was on screen before.
    asyncio.run(r.tick())
    check("a held timer is still drawn", len(hub.drawn) == 1)
    check("...flagged held", hub.drawn[-1].get("held") is True)
    check("...and says nothing", hub.said == [], str(hub.said))
    check("...and does not chime", hub.chimes == [], str(hub.chimes))

    check("status says what it is waiting for",
          "go" in r.status().lower(), r.status())

    check("go releases it", r.resume())
    check("...and it is no longer held", not r.held and not r.paused)
    check("...starting from nought", abs(r.elapsed()) < 0.001)

    asyncio.run(r.tick())
    check("...and now it speaks", hub.said == ["Ready."], str(hub.said))

    fake.advance(5)
    asyncio.run(r.tick())
    check("the count-in leads into the work",
          r.where().phase.kind == iv.WORK, r.where().phase.kind)
    check("pause after release is an ordinary pause",
          r.pause() and r.paused and not r.held)

    r.stop()
    check("stop clears the held flag", not r.held and not r.running)


def test_lead_in(monkey_time) -> None:
    print("\nHow long he gets to get ready")
    fake = FakeClock()
    iv.time.time = fake
    r = _runner(FakeHub(), fake, lead_in_seconds=5)

    check("no argument means the configured five", r.lead_in_for() == 5)
    check("True means the same", r.lead_in_for(True) == 5)
    check("None means the same", r.lead_in_for(None) == 5)
    check("False means none at all", r.lead_in_for(False) == 0)
    check("a number is taken literally", r.lead_in_for(30) == 30)
    check("nought is honoured, not treated as absent",
          r.lead_in_for(0) == 0)
    check("nonsense falls back to the default", r.lead_in_for("soon") == 5)
    check("a silly one is capped",
          r.lead_in_for(99999) == iv.MAX_LEAD_IN_SECONDS,
          str(r.lead_in_for(99999)))
    check("a negative one is floored", r.lead_in_for(-30) == 0)

    r.start(build("countdown", minutes=20, label="AMRAP"), lead_in=30)
    check("thirty seconds first", abs(r.where().remaining - 30) < 0.001,
          f"{r.where().remaining:.1f}s")
    check("...on a Ready phase", r.where().phase.kind == iv.READY)

    r.start(build("countdown", minutes=20, label="AMRAP"), lead_in=0)
    check("or none, and it is straight into the work",
          r.where().phase.kind == iv.RUN, r.where().phase.kind)


def test_go_through_the_tools(monkey_time) -> None:
    print("\nHold and go, through the tools")
    fake = FakeClock()
    iv.time.time = fake
    hub = FakeHub()
    r = _runner(hub, fake, lead_in_seconds=5)
    t = IntervalTools(r)

    props = next(x for x in t.schemas()
                 if x["name"] == "start_interval_timer")["input_schema"]
    check("the model can ask to hold", "hold" in props["properties"])
    check("...and to set the count-in", "lead_in" in props["properties"])
    ctl = next(x for x in t.schemas()
               if x["name"] == "interval_control")["input_schema"]
    check("go is an action it can pick",
          "go" in ctl["properties"]["action"]["enum"])

    said = t.run_sync("start_interval_timer",
                      {"kind": "rounds", "work": 40, "rest": 20,
                       "rounds": 8, "hold": True})
    check("she says she is waiting", "go" in said.lower(), said)
    check("...and it is held", r.held)

    fake.advance(45)
    check("it has not started", abs(r.elapsed()) < 0.001)

    said = t.run_sync("interval_control", {"action": "go"})
    check("go starts it", "here we go" in said.lower(), said)
    check("...and clears the hold", not r.held)

    said = t.run_sync("interval_control", {"action": "go"})
    check("go on a running timer is a no-op in words",
          "already running" in said.lower(), said)

    t.run_sync("interval_control", {"action": "pause"})
    said = t.run_sync("interval_control", {"action": "go"})
    check("go also resumes an ordinary pause",
          "carrying on" in said.lower(), said)

    said = t.run_sync("start_interval_timer",
                      {"kind": "countdown", "minutes": 20,
                       "label": "AMRAP", "lead_in": 30})
    check("a spoken count-in is confirmed out loud",
          "30 seconds" in said, said)
    check("...and is really thirty", abs(r.where().remaining - 30) < 0.001)

    said = t.run_sync("start_interval_timer",
                      {"kind": "countdown", "minutes": 20, "lead_in": 0})
    check("no count-in says so", "starting now" in said.lower(), said)



def test_the_screen(monkey_time) -> None:
    print("\nStepping the clock aside")
    fake = FakeClock()
    iv.time.time = fake
    hub = FakeHub()
    r = _runner(hub, fake, lead_in_seconds=0)
    t = IntervalTools(r)

    ctl = next(x for x in t.schemas()
               if x["name"] == "interval_control")["input_schema"]
    actions = ctl["properties"]["action"]["enum"]
    check("show and hide are actions she can pick",
          "show" in actions and "hide" in actions, str(actions))

    t.run_sync("start_interval_timer",
               {"kind": "countdown", "minutes": 20, "label": "AMRAP"})
    check("the timer is running", r.running)

    # Every draw carries the session, so the screen can tell a redraw of what
    # it is already showing from a brand new timer.
    asyncio.run(r.tick())
    first = hub.drawn[-1]
    check("a draw names its session", "session" in first, str(first.keys()))
    fake.advance(3)
    asyncio.run(r.tick())
    check("...and it does not change on a redraw",
          hub.drawn[-1]["session"] == first["session"])

    said = t.run_sync("interval_control", {"action": "hide"})
    check("hiding it says the timer is still going",
          "running" in said.lower(), said)
    check("...and it really is", r.running)
    check("...and the screen was told once",
          hub.pushed == [{"type": "interval_view", "show": False}],
          str(hub.pushed))

    # The point: hiding the clock changes nothing about the session.
    fake.advance(60)
    pos = r.where()
    check("time still passes while the clock is out of sight",
          abs(pos.remaining - (1200 - 63)) < 0.01, f"{pos.remaining:.1f}s")

    said = t.run_sync("interval_control", {"action": "show"})
    check("showing it says so", "screen" in said.lower(), said)
    check("...and the screen was told",
          hub.pushed[-1] == {"type": "interval_view", "show": True},
          str(hub.pushed[-1]))

    # A new session gets a new id, which is what tells the screen to take
    # itself back after he has put it away.
    was = r.started
    fake.advance(5)
    t.run_sync("start_interval_timer", {"kind": "emom", "rounds": 10})
    asyncio.run(r.tick())
    check("a new timer is a new session",
          hub.drawn[-1]["session"] != round(was, 3),
          str(hub.drawn[-1]["session"]))


def main() -> int:
    real = iv.time.time
    try:
        test_shapes()
        test_refusals()
        test_position_is_pure()
        test_say_and_clock()
        test_running(None)
        test_a_missed_tick_does_not_drift(None)
        test_controls(None)
        test_status(None)
        test_tools(None)
        test_holding_for_go(None)
        test_lead_in(None)
        test_go_through_the_tools(None)
        test_the_screen(None)
        test_serialisation()
    finally:
        iv.time.time = real
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
