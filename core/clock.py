"""The only module allowed to ask what time it is.

Ageing is measured against "now". If a dozen modules each call time.time(),
then "now" is whatever the wall clock says — and three things become
impossible at once: asking what the assistant would have believed in March,
demonstrating a year of decay without waiting a year, and writing a test for
staleness that finishes today.

So exactly one module knows the time and everything else receives a clock as
a parameter. It costs one argument and removes a whole class of problem.

Borrowed from @datasciencebrain's "Build an Agent That Repairs Its Own Stale
Memories", which is right that this file has to come first.
"""
from __future__ import annotations

import time
from typing import Protocol

DAY = 86400.0


class Clock(Protocol):
    """Anything that can say what "now" is, in epoch seconds."""

    frozen: bool

    def now(self) -> float: ...


class SystemClock:
    """Wall-clock time. What runs in the house."""

    frozen = False

    def now(self) -> float:
        return time.time()


class FrozenClock:
    """A clock the caller moves by hand.

    This is what makes silent decay testable in one second: seed a memory,
    advance six months, run the sweep, and watch a fast-moving fact go stale
    without anybody having waited for it.
    """

    frozen = True

    def __init__(self, at: float | None = None):
        self._at = float(at) if at is not None else time.time()

    def now(self) -> float:
        return self._at

    def advance(self, days: float = 0.0, seconds: float = 0.0) -> None:
        """Move forward. Refuses to go backwards.

        Superseding a fact writes "this stopped being true when that started".
        Run the clock backwards and you can produce a chain where the
        successor begins BEFORE its predecessor — an interval with negative
        length, which then answers point-in-time questions wrongly. Refusing
        here means the store never has to defend against it.
        """
        delta = days * DAY + seconds
        if delta < 0:
            raise ValueError("a clock that runs backwards breaks the "
                             "valid-time chain")
        self._at += delta

    def set(self, at: float) -> None:
        self._at = float(at)


def days_between(later: float, earlier: float) -> float:
    """Fractional and SIGNED days.

    Signed because a negative result is a real signal, not an error: it means
    a fact claims to have been verified in the future, which is what happens
    when a frozen clock and a live clock meet in the same database.
    """
    return (later - earlier) / DAY


def freshness(age_days: float, half_life_days: float) -> float:
    """Belief in a fact, decaying by half every half-life. 1.0 is today."""
    if half_life_days <= 0:
        return 1.0
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def make_clock(at: float | None = None) -> Clock:
    return FrozenClock(at) if at is not None else SystemClock()
