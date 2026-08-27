"""Timers and reminders — the only part of the system that speaks first.

Everything else in this appliance is reactive: you talk, it answers. This
module is the exception, and that difference is worth stating plainly because
it changes what "correct" means.

A reply can be a little late and nobody minds. A timer that fires four
minutes after the pasta is done is simply wrong, and a reminder that never
fires because the service restarted is worse than never having been set —
you stopped carrying the thing in your head because you delegated it.

So the design is deliberately dull:

  * Absolute timestamps in SQLite. Nothing lives only in memory, so a
    restart, a crash, or the USB wedge costs you nothing.
  * A poll loop, not sleeping tasks. A task sleeping for eight hours is a
    task that quietly disappears when anything goes wrong.
  * Overdue items still fire, with an apology, up to a grace window. Coming
    back from a two-minute restart should not lose your timer; being told
    about yesterday's is just noise.

The model does all the date arithmetic. It knows the current time — it is
in every request — so "in twenty minutes" and "tomorrow at seven" resolve
upstream into a unix timestamp. That keeps a whole class of daylight-saving
bugs out of this file.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import time

log = logging.getLogger("assistant.scheduler")

# How often to look for due items. One second is cheap — an indexed query
# against a table with a handful of rows — and it keeps timers honest.
TICK_SECONDS = 1.0

# Fire something overdue by up to this long, saying so. Beyond it, the moment
# has passed and announcing it is noise rather than help.
GRACE_SECONDS = 30 * 60


def _phrase_for(row, now: float) -> str:
    """What the assistant actually says when this fires."""
    late = now - row["due"]
    text = row["text"]

    if row["kind"] == "timer":
        base = f"Your {text} timer is up." if text else "Your timer is up."
    else:
        base = f"Reminder: {text}"

    # The user's own words end the sentence, and people rarely dictate a full
    # stop. Without this, "take the bins out" runs straight into the apology.
    if not base.endswith((".", "!", "?")):
        base += "."

    if late > 90:
        minutes = int(late // 60)
        unit = "minute" if minutes == 1 else "minutes"
        return f"{base} Sorry, this is {minutes} {unit} late."
    return base


class Scheduler:
    """Polls for due reminders and announces them through the hub."""

    def __init__(self, store, hub, cfg: dict):
        self.store = store
        self.hub = hub
        self.cfg = cfg
        self.chime = bool(cfg.get("behaviour", {}).get("reminder_chime", True))
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            pending = self.store.pending_reminders()
            if pending:
                log.info("scheduler started with %d pending", len(pending))
            else:
                log.info("scheduler started")

    async def _run(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception:  # noqa: BLE001
                # Degrade, never die. A bad row must not stop every future
                # timer from firing.
                log.exception("scheduler tick failed — continuing")
            await asyncio.sleep(TICK_SECONDS)

    async def _tick(self) -> None:
        now = time.time()
        for row in self.store.due_reminders(now):
            # Mark fired BEFORE announcing. If announcing throws, or the
            # process dies mid-sentence, the alternative is a reminder that
            # repeats every second forever — which is a genuinely alarming
            # thing to come home to.
            self.store.mark_fired(row["id"])

            if now - row["due"] > GRACE_SECONDS:
                log.info("skipping %r — overdue by %.0f minutes",
                         row["text"], (now - row["due"]) / 60)
                continue

            phrase = _phrase_for(row, now)
            log.info("firing %s: %s", row["kind"], phrase)

            await self.hub.to_display({
                "type": "reminder", "text": phrase, "kind": row["kind"],
            })
            if self.chime:
                await self.hub.to_audio({"type": "chime", "kind": "alert"})
            await self.hub.to_audio({"type": "speak", "text": phrase})
            # Tell the audio service the announcement is complete, so its
            # follow-up window opens — you can answer "thanks, remind me
            # again in ten" without saying the wake word.
            await self.hub.to_audio({"type": "speak_done"})


def describe_when(due: float) -> str:
    """Human phrasing for a future time, for tool confirmations."""
    now = time.time()
    delta = due - now
    if delta < 90:
        secs = max(1, int(round(delta)))
        return f"in {secs} second{'s' if secs != 1 else ''}"
    if delta < 3600:
        mins = int(round(delta / 60))
        return f"in {mins} minute{'s' if mins != 1 else ''}"
    when = _dt.datetime.fromtimestamp(due)
    today = _dt.date.today()
    if when.date() == today:
        return f"at {when.strftime('%-I:%M %p')}"
    if (when.date() - today).days == 1:
        return f"tomorrow at {when.strftime('%-I:%M %p')}"
    return when.strftime("on %A at %-I:%M %p")
