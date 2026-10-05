"""An interval timer you can drive with your hands full.

Work, rest, repeat — and she calls it, so you are not squinting at a phone on
the floor between burpees. Four shapes cover almost everything:

    rounds     40 seconds on, 20 off, 8 rounds       (Tabata, circuits)
    emom       every minute on the minute, 12 minutes
    beep       a mark every 90 seconds for 20 minutes
    countdown  one 20-minute run                      (AMRAP)

The design decision that matters is that **a running timer holds no state**.
A plan is a fixed list of phases, so where you are is a pure function of how
long ago it started:

    position(plan, now - started - paused_for)

Nothing is incremented on a tick, so a tick that arrives late, early, or not
at all changes nothing. The orchestrator can restart in the middle of your
third round and pick it up exactly where it was, because "exactly where it
was" is recomputed from a timestamp rather than remembered. That is the same
reason the reminder scheduler stores absolute times instead of sleeping — a
timer that quietly loses its place is worse than no timer, because you stopped
counting the moment you delegated it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Iterable

log = logging.getLogger("assistant.intervals")

WORK, REST, MARK, RUN, READY = "work", "rest", "mark", "run", "ready"

# Longest single phase and longest whole session. Not arbitrary: these are the
# boundary between "a mistake I made saying it" and "a timer I meant".
MAX_PHASE_SECONDS = 60 * 60
MAX_TOTAL_SECONDS = 6 * 60 * 60
MAX_ROUNDS = 200
# Longest count-in he can ask for. Thirty seconds to get to the mat is a
# reasonable thing to want; ten minutes is a different tool.
MAX_LEAD_IN_SECONDS = 600


@dataclass(frozen=True)
class Phase:
    label: str          # what she says: "Work", "Rest", "Minute 3"
    seconds: int
    kind: str           # work | rest | mark | run | ready
    round: int = 0      # 1-based; 0 when rounds are not a thing
    rounds: int = 0     # total, for "round 3 of 8"

    def spoken(self) -> str:
        """What she says when this phase begins."""
        if self.kind == READY:
            return "Ready."
        if self.rounds and self.round:
            # The round number goes with WORK, not rest: it is the thing you
            # are about to do, and hearing "round four of eight" while you are
            # gasping through the rest is a different, worse message.
            if self.kind == WORK:
                return f"{self.label}. Round {self.round} of {self.rounds}."
            if self.kind == MARK:
                return f"{self.label}."
            return self.label + "."
        return self.label + "."


@dataclass
class Plan:
    name: str
    phases: list[Phase]

    @property
    def seconds(self) -> int:
        return sum(p.seconds for p in self.phases)

    @property
    def rounds(self) -> int:
        return max((p.rounds for p in self.phases), default=0)

    def as_json(self) -> str:
        return json.dumps({"name": self.name,
                           "phases": [asdict(p) for p in self.phases]})

    @classmethod
    def from_json(cls, blob: str) -> "Plan":
        raw = json.loads(blob)
        return cls(raw["name"], [Phase(**p) for p in raw["phases"]])

    def describe(self) -> str:
        return f"{self.name}, {say_duration(self.seconds)}"


# ------------------------------------------------------------- building

def _check(seconds: int, what: str) -> int:
    seconds = int(seconds)
    if seconds <= 0:
        raise ValueError(f"{what} must be more than zero")
    if seconds > MAX_PHASE_SECONDS:
        raise ValueError(f"{what} of {say_duration(seconds)} is too long for "
                         f"an interval timer")
    return seconds


def rounds_plan(work: int, rest: int = 0, rounds: int = 1,
                lead_in: int = 0, trailing_rest: bool = False) -> Plan:
    """Work, then rest, N times.

    The last rest is dropped by default. Standing there being told to rest
    when the session is over is the small stupidity that makes a timer feel
    like it was written by someone who has never used one.
    """
    work = _check(work, "the work interval")
    rounds = int(rounds)
    if rounds < 1 or rounds > MAX_ROUNDS:
        raise ValueError(f"{rounds} rounds is not a sensible number")
    if rest:
        rest = _check(rest, "the rest interval")

    phases: list[Phase] = []
    if lead_in:
        phases.append(Phase("Ready", int(lead_in), READY))
    for n in range(1, rounds + 1):
        phases.append(Phase("Work", work, WORK, n, rounds))
        if rest and (trailing_rest or n < rounds):
            phases.append(Phase("Rest", rest, REST, n, rounds))
    name = (f"{say_duration(work)} on, {say_duration(rest)} off, "
            f"{rounds} rounds" if rest else
            f"{rounds} x {say_duration(work)}")
    return Plan(name, phases)


def emom_plan(minutes: int, every: int = 60, lead_in: int = 0) -> Plan:
    """Every minute on the minute. Each phase is one whole minute."""
    every = _check(every, "the interval")
    rounds = int(minutes)
    if rounds < 1 or rounds > MAX_ROUNDS:
        raise ValueError(f"{minutes} minutes is not a sensible EMOM")
    phases: list[Phase] = []
    if lead_in:
        phases.append(Phase("Ready", int(lead_in), READY))
    for n in range(1, rounds + 1):
        phases.append(Phase(f"Minute {n}", every, WORK, n, rounds))
    unit = "minute" if every == 60 else say_duration(every)
    return Plan(f"EMOM, {rounds} rounds every {unit}", phases)


def beep_plan(every: int, total_seconds: int, lead_in: int = 0) -> Plan:
    """A mark every N seconds. No work/rest distinction — just a metronome
    slow enough to move on."""
    every = _check(every, "the interval")
    total = _check(total_seconds, "the total")
    rounds = max(1, int(total // every))
    if rounds > MAX_ROUNDS:
        raise ValueError(f"that is {rounds} intervals — too many to call out")
    phases: list[Phase] = []
    if lead_in:
        phases.append(Phase("Ready", int(lead_in), READY))
    for n in range(1, rounds + 1):
        phases.append(Phase("Next", every, MARK, n, rounds))
    return Plan(f"every {say_duration(every)} for {say_duration(total)}",
                phases)


def countdown_plan(total_seconds: int, label: str = "Go",
                   lead_in: int = 0) -> Plan:
    """One long run. An AMRAP, a plank, a row."""
    total = _check(total_seconds, "the timer")
    phases: list[Phase] = []
    if lead_in:
        phases.append(Phase("Ready", int(lead_in), READY))
    phases.append(Phase(label, total, RUN))
    return Plan(f"{say_duration(total)} {label.lower()}", phases)


def build(kind: str, *, work: int = 0, rest: int = 0, rounds: int = 0,
          minutes: int = 0, seconds: int = 0, every: int = 0,
          label: str = "", lead_in: int = 0) -> Plan:
    """One entry point, because the model picks the shape by name.

    Accepts minutes or seconds for the totals: a language model will produce
    either, and the difference between "twenty minutes" and 1200 is not
    something worth making it think about.
    """
    total = int(seconds) + int(minutes) * 60
    kind = (kind or "").strip().lower()

    if kind in ("rounds", "intervals", "tabata", "circuit"):
        plan = rounds_plan(work or total, rest, rounds or 1, lead_in)
    elif kind in ("emom", "every minute on the minute"):
        plan = emom_plan(rounds or int(total // 60) or 1, every or 60, lead_in)
    elif kind in ("beep", "mark", "metronome"):
        plan = beep_plan(every or work or 60, total, lead_in)
    elif kind in ("countdown", "amrap", "run", "single"):
        plan = countdown_plan(total or work, label or "Go", lead_in)
    else:
        raise ValueError(f"I do not know the timer shape {kind!r}")

    if plan.seconds > MAX_TOTAL_SECONDS:
        raise ValueError(f"that comes to {say_duration(plan.seconds)}, which "
                         f"is longer than I will run an interval timer for")
    return plan


# ------------------------------------------------------------- position

@dataclass
class Position:
    index: int          # which phase, 0-based
    phase: Phase
    remaining: float    # seconds left in this phase
    elapsed: float      # seconds into the whole session
    left: float         # seconds left in the whole session
    done: bool = False


def position(plan: Plan, elapsed: float) -> Position | None:
    """Where the session is, computed rather than remembered.

    Returns None once the plan has run out. Because this is a pure function of
    elapsed time, a tick that arrives late does not drift, a tick that never
    arrives loses nothing, and a restart mid-session resumes exactly.
    """
    if elapsed < 0:
        elapsed = 0.0
    total = plan.seconds
    if elapsed >= total:
        last = plan.phases[-1] if plan.phases else Phase("Done", 0, RUN)
        return Position(len(plan.phases) - 1, last, 0.0, total, 0.0, True)
    run = 0.0
    for i, phase in enumerate(plan.phases):
        if elapsed < run + phase.seconds:
            return Position(i, phase, (run + phase.seconds) - elapsed,
                            elapsed, total - elapsed)
        run += phase.seconds
    return None


def say_duration(seconds: float) -> str:
    """How a person says a length of time."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    bits = []
    if h:
        bits.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        bits.append(f"{m} minute{'s' if m != 1 else ''}")
    if s and not h:
        bits.append(f"{s} second{'s' if s != 1 else ''}")
    return " ".join(bits)


def clock(seconds: float) -> str:
    """MM:SS for the screen."""
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# --------------------------------------------------------------- runner

class IntervalRunner:
    """Runs one session at a time, driven by the scheduler's one-second tick.

    Holds almost nothing: the plan, when it started, and how long it has been
    paused. Everything else is derived. The only mutable bookkeeping is which
    phase was last announced, so a change is announced once rather than every
    tick — and if that is lost in a restart, the worst case is one missed
    "Work" call, not a timer that is wrong about where it is.
    """

    def __init__(self, cfg: dict, store, hub=None):
        self.cfg = cfg
        self.store = store
        self.hub = hub
        i = cfg.get("intervals", {}) or {}
        self.enabled = bool(i.get("enabled", True))
        self.lead_in = int(i.get("lead_in_seconds", 5))
        self.count_from = int(i.get("count_last_seconds", 3))
        # Do not count down a phase barely longer than the count itself.
        self.count_min_phase = int(i.get("count_min_phase_seconds", 8))
        self.chime = bool(i.get("chime", True))
        self.say_rounds = bool(i.get("announce_rounds", True))
        self.finish_words = str(i.get("finish_phrase", "Time. Well done."))

        self.plan: Plan | None = None
        self.started: float = 0.0
        self.paused_at: float | None = None
        self.paused_total: float = 0.0
        self._said_index: int | None = None
        self._said_count: int | None = None
        # Held is a paused timer that has never run: built, on screen, and
        # waiting for a word. It is kept apart from `paused` because the two
        # want different words on screen and in the ear — "paused" implies
        # something to carry on with, and there is nothing yet.
        self.held: bool = False

    # -- state ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.plan is not None

    @property
    def paused(self) -> bool:
        return self.paused_at is not None

    def elapsed(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        if self.paused_at is not None:
            return self.paused_at - self.started - self.paused_total
        return now - self.started - self.paused_total

    def where(self, now: float | None = None) -> Position | None:
        if self.plan is None:
            return None
        return position(self.plan, self.elapsed(now))

    # -- control -------------------------------------------------------

    def lead_in_for(self, lead_in: bool | int | None = True) -> int:
        """How long to count him in, given what he asked for.

        True or None means the configured default; False or 0 means start on
        the word; a number is that many seconds. `isinstance(True, int)` is
        True in Python, so the booleans have to be settled first or every
        `lead_in=True` would silently become a one-second count-in.
        """
        if lead_in is True or lead_in is None:
            return max(0, int(self.lead_in))
        if lead_in is False:
            return 0
        try:
            return max(0, min(MAX_LEAD_IN_SECONDS, int(float(lead_in))))
        except (TypeError, ValueError):
            return max(0, int(self.lead_in))

    def start(self, plan: Plan, lead_in: bool | int | None = True,
              hold: bool = False) -> str:
        """Put a plan on the clock.

        `hold` builds everything and then stands still: the plan is real, the
        screen shows it, and the clock does not move until `resume()`. It is
        implemented as a pause taken on the same instant the timer started, so
        elapsed time is exactly zero however long he takes to say the word —
        no special case in `position()`, which stays a pure function.
        """
        self.stop(announce=False)
        seconds = self.lead_in_for(lead_in)
        if seconds and plan.phases and plan.phases[0].kind != READY:
            plan = Plan(plan.name,
                        [Phase("Ready", seconds, READY)] + plan.phases)
        self.plan = plan
        self.started = time.time()
        self.paused_at = self.started if hold else None
        self.paused_total = 0.0
        self._said_index = None
        self._said_count = None
        self.held = bool(hold)
        log.info("interval timer: %s (%d phases, %s)%s", plan.name,
                 len(plan.phases), say_duration(plan.seconds),
                 " — held" if hold else "")
        return plan.describe()

    def pause(self) -> bool:
        if not self.running or self.paused:
            return False
        self.paused_at = time.time()
        return True

    def resume(self) -> bool:
        if not self.running or not self.paused:
            return False
        self.paused_total += time.time() - self.paused_at
        self.paused_at = None
        self.held = False
        # Re-announce the phase we are standing in, because you have almost
        # certainly forgotten which one it was.
        self._said_index = None
        return True

    def skip(self) -> bool:
        """Jump to the start of the next phase."""
        pos = self.where()
        if pos is None or pos.done:
            return False
        # Move the start time backwards by exactly the remainder. Still no
        # mutable position — just a different origin.
        self.started -= pos.remaining
        self._said_index = None
        return True

    def add(self, seconds: int) -> bool:
        """Give the current phase more time, or take some away.

        This lengthens the PHASE rather than moving the origin, and the
        difference matters. Moving the origin backwards rewinds you inside a
        phase of fixed length — "give me another thirty seconds" halfway
        through a thirty-second interval would put you back at the start with
        thirty left, not sixty. Extending the phase is what the words mean.

        Shrinking is clamped so at least a second remains: taking a minute off
        a phase you are fifty seconds into should end it, not travel backwards
        through it.
        """
        from dataclasses import replace
        pos = self.where()
        if pos is None or pos.done or self.plan is None:
            return False
        consumed = pos.phase.seconds - pos.remaining
        longer = max(int(consumed) + 1, pos.phase.seconds + int(seconds))
        self.plan.phases[pos.index] = replace(pos.phase, seconds=longer)
        return True

    def stop(self, announce: bool = True) -> bool:
        was = self.plan is not None
        self.plan = None
        self.started = 0.0
        self.paused_at = None
        self.paused_total = 0.0
        self._said_index = None
        self._said_count = None
        self.held = False
        return was

    # -- the tick ------------------------------------------------------

    async def tick(self) -> None:
        """Called once a second. Announces changes; never computes position
        by accumulation."""
        if not self.enabled or self.plan is None:
            return
        now = time.time()
        pos = self.where(now)
        if self.paused:
            # A paused or held timer still has to be on the screen — the
            # kiosk redraws from these messages and nothing else, so going
            # quiet leaves whatever was drawn last frozen and wrong. Draw,
            # but announce nothing and count nothing: the clock is not moving.
            if pos is not None and not pos.done:
                await self._draw(pos)
            return
        if pos is None or pos.done:
            await self._finish()
            return

        if pos.index != self._said_index:
            self._said_index = pos.index
            self._said_count = None
            await self._announce(pos)
            await self._draw(pos)
            return

        # The last few seconds, spoken one at a time. Skipped on a phase
        # barely longer than the count itself — being counted down from three
        # on a five-second rest is noise, not information.
        left = int(pos.remaining)
        if (self.count_from and pos.phase.seconds >= self.count_min_phase
                and 0 < left <= self.count_from and left != self._said_count):
            self._said_count = left
            await self._say(str(left))
        await self._draw(pos)

    async def _announce(self, pos: Position) -> None:
        phase = pos.phase
        if self.chime and self.hub is not None:
            # Work and rest get different tones. You learn them in one session
            # and stop needing the words at all.
            kind = "wake" if phase.kind in (WORK, RUN, MARK) else "sleep"
            await self.hub.to_audio({"type": "chime", "kind": kind})
        words = phase.spoken() if self.say_rounds else phase.label + "."
        await self._say(words)

    async def _say(self, text: str) -> None:
        if self.hub is None:
            return
        await self.hub.to_audio({"type": "speak", "text": text})
        await self.hub.to_audio({"type": "speak_done"})

    async def _draw(self, pos: Position) -> None:
        if self.hub is None:
            return
        await self.hub.to_display({
            "type": "interval",
            "running": True,
            # Which session this is. The screen uses it to tell a redraw of
            # the timer it is already showing from a brand new one — the
            # first takes the screen, the second must not, or asking for the
            # board mid-workout would lose the argument one second later.
            "session": round(self.started, 3),
            "paused": self.paused,
            "held": self.held,
            "name": self.plan.name if self.plan else "",
            "label": pos.phase.label,
            "kind": pos.phase.kind,
            "round": pos.phase.round,
            "rounds": pos.phase.rounds,
            "remaining": round(pos.remaining, 1),
            "phase_seconds": pos.phase.seconds,
            "left": round(pos.left, 1),
            "total": self.plan.seconds if self.plan else 0,
        })

    async def _finish(self) -> None:
        name = self.plan.name if self.plan else ""
        total = self.plan.seconds if self.plan else 0
        self.stop(announce=False)
        log.info("interval timer finished: %s", name)
        if self.hub is not None:
            await self.hub.to_display({"type": "interval", "running": False})
            if self.chime:
                await self.hub.to_audio({"type": "chime", "kind": "alert"})
            await self._say(f"{self.finish_words} "
                            f"{say_duration(total)} of work.")

    def screen(self, show: bool) -> bool:
        """Put the timer in front of the other panels, or step it aside.

        A one-shot instruction rather than a flag on the once-a-second draw.
        The screen owns which panel is in front — it has to, because he can
        also press a key — and a field repeated every tick would overrule
        that within a second, which is the bug this exists to stop.
        """
        push = getattr(self.hub, "push", None)
        if push is None:
            return False
        push({"type": "interval_view", "show": bool(show)})
        return True

    # -- what it says when asked --------------------------------------

    def status(self) -> str:
        pos = self.where()
        if self.plan is None or pos is None:
            return "No timer running."
        if pos.done:
            return "It just finished."
        if self.held:
            return (f"{self.plan.name}, ready when you are. "
                    'Say "go" to start it.')
        bits = [f"{self.plan.name}."]
        if pos.phase.rounds and pos.phase.round:
            bits.append(f"{pos.phase.label}, round {pos.phase.round} of "
                        f"{pos.phase.rounds}.")
        else:
            bits.append(f"{pos.phase.label}.")
        bits.append(f"{say_duration(pos.remaining)} left in this one, "
                    f"{say_duration(pos.left)} altogether.")
        if self.paused:
            bits.append("Paused.")
        return " ".join(bits)


# =====================================================================
# The tools
# =====================================================================

ACTIONS = ("pause", "resume", "go", "stop", "skip", "status", "add", "less",
           "show", "hide")


class IntervalTools:
    """Two tools: start one, and change one that is running."""

    def __init__(self, runner: IntervalRunner):
        self.runner = runner
        self.enabled = runner.enabled

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [
            {
                "name": "start_interval_timer",
                "description": (
                    "Start an interval timer for exercise and call it out "
                    "loud. Four shapes:\n"
                    "  rounds — work then rest, repeated. "
                    "\"forty on, twenty off, eight rounds\" is "
                    "kind=rounds work=40 rest=20 rounds=8. Tabata is "
                    "work=20 rest=10 rounds=8.\n"
                    "  emom — every minute on the minute. \"EMOM for "
                    "twelve\" is kind=emom rounds=12.\n"
                    "  beep — a mark every so often. \"every ninety "
                    "seconds for twenty minutes\" is kind=beep every=90 "
                    "minutes=20.\n"
                    "  countdown — one long run, for an AMRAP, a plank or a "
                    "row. \"twenty minute AMRAP\" is kind=countdown "
                    "minutes=20 label=AMRAP.\n"
                    "Work out the seconds yourself: \"a minute on\" is "
                    "work=60. Use this for exercise intervals, NOT for "
                    "set_reminder, which is for one-off timers and clock "
                    "times."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string",
                                 "enum": ["rounds", "emom", "beep",
                                          "countdown"]},
                        "work": {"type": "integer",
                                 "description": "Seconds of work per round."},
                        "rest": {"type": "integer",
                                 "description": "Seconds of rest per round."},
                        "rounds": {"type": "integer",
                                   "description": "How many rounds."},
                        "minutes": {"type": "integer",
                                    "description": "Total length, in minutes."},
                        "seconds": {"type": "integer",
                                    "description": "Total length, in seconds."},
                        "every": {"type": "integer",
                                  "description": "Seconds between marks, for "
                                                 "beep and emom."},
                        "lead_in": {
                            "type": "integer",
                            "description": (
                                "Seconds to count him in before the first "
                                "phase. Use it when he asks for time to get "
                                "set — \"give me thirty seconds first\" is "
                                "30. Leave it out for the usual short "
                                "count-in; 0 starts on the word."),
                        },
                        "hold": {
                            "type": "boolean",
                            "description": (
                                "Do not start the clock. Build the timer, "
                                "put it on the screen and wait for him. Use "
                                "it for \"start it when I say go\", \"wait "
                                "for me\", \"on my mark\". He releases it "
                                "by saying go, which is interval_control "
                                "with action=go."),
                        },
                        "label": {"type": "string",
                                  "description": "What to call the working "
                                                 "phase of a countdown."},
                    },
                    "required": ["kind"],
                },
            },
            {
                "name": "interval_control",
                "description": (
                    "Change the interval timer that is running. 'go' "
                    "releases one that is holding for him and also resumes a "
                    "paused one — use it for \"go\", \"begin\", "
                    "\"start\". 'skip' jumps to the next phase, 'add' "
                    "gives the current one more seconds, 'less' takes some "
                    "away, 'status' says where it is. 'hide' puts the clock "
                    "away without stopping it and 'show' brings it back — "
                    "the timer keeps running either way."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "seconds": {"type": "integer",
                                    "description": "For add and less."},
                    },
                    "required": ["action"],
                },
            },
        ]

    def run_sync(self, name: str, args: dict) -> str | None:
        if not self.enabled or name not in ("start_interval_timer",
                                            "interval_control"):
            return None
        if name == "start_interval_timer":
            return self._start(args)
        return self._control(args)

    def _start(self, args: dict) -> str:
        def num(key: str) -> int:
            value = args.get(key)
            if value is None or isinstance(value, bool):
                return 0
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        try:
            plan = build(
                str(args.get("kind") or "rounds"),
                work=num("work"), rest=num("rest"), rounds=num("rounds"),
                minutes=num("minutes"), seconds=num("seconds"),
                every=num("every"), label=str(args.get("label") or ""),
            )
        except ValueError as exc:
            # A refusal in words, not a stack trace. These are read out loud.
            return f"I cannot run that: {exc}."

        hold = bool(args.get("hold"))
        # An absent lead_in means "whatever is configured", which is not the
        # same as a lead_in of nought — so the default has to be True rather
        # than 0, and a value that will not parse falls back to the default
        # rather than to an instant start he did not ask for.
        raw = args.get("lead_in")
        if raw is None or isinstance(raw, bool):
            lead: bool | int = True
        else:
            try:
                lead = int(float(raw))
            except (TypeError, ValueError):
                lead = True

        said = self.runner.start(plan, lead_in=lead, hold=hold)
        seconds = self.runner.lead_in_for(lead)
        rounds = plan.rounds
        extra = f" {rounds} rounds." if rounds > 1 else ""
        if hold:
            tail = ' Say "go" when you are ready.'
        elif seconds:
            tail = f" Counting you in, {say_duration(seconds)}."
        else:
            tail = " Starting now."
        return f"Starting {said}.{extra}{tail}"

    def _control(self, args: dict) -> str:
        action = str(args.get("action") or "").strip().lower()
        r = self.runner
        if action == "status":
            return r.status()
        if not r.running:
            return "No timer running."
        if action == "pause":
            return "Paused." if r.pause() else "It is already paused."
        if action in ("resume", "go"):
            # Held and paused are the same mechanism and different sentences.
            # Read the flag before resuming, because resuming clears it.
            was_held = r.held
            if not r.resume():
                return "It is already running."
            return "Here we go." if was_held else "Carrying on."
        if action == "stop":
            r.stop()
            return "Stopped."
        if action == "skip":
            return "Next." if r.skip() else "Nothing left to skip to."
        if action in ("show", "hide"):
            # The clock, not the timer. Nothing about the session changes.
            r.screen(action == "show")
            return "On screen." if action == "show" else "It is still running."
        if action in ("add", "less"):
            secs = args.get("seconds")
            try:
                secs = int(float(secs))
            except (TypeError, ValueError):
                return "How many seconds?"
            if action == "less":
                secs = -secs
            r.add(secs)
            pos = r.where()
            left = say_duration(pos.remaining) if pos else "no time"
            return f"{left} left in this one."
        return f"I do not know how to {action}."
