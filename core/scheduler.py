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

# How late an SLA warning may be and still be worth saying. A marker whose
# moment passed while the service was restarting has been claimed either way —
# the row goes in so it can never fire twice — but announcing "halfway through
# your SLA" two hours after halfway is misinformation, not a warning.
#
# Breach gets a longer window than the interim markers because it is the one
# you would still want to hear about late.
SLA_STALE_SECONDS = 10 * 60
SLA_STALE_BREACH_SECONDS = 60 * 60


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
        from .casework import Sla
        self.store = store
        self.hub = hub
        self.cfg = cfg
        self.chime = bool(cfg.get("behaviour", {}).get("reminder_chime", True))
        self.cases_on = bool((cfg.get("casework", {}) or {}).get("enabled", False))
        self.sla = Sla(cfg)
        self.sweep = None      # set by app.py, which owns the toolbox
        self.media = None      # likewise — needs a heartbeat, not a thread
        self.selfcheck = None  # the daily diagnostic, also set by app.py
        self.intervals = None  # the exercise interval timer, ditto
        self.monitor = None    # the rolling status wall, also from app.py
        sc = (cfg.get("selfcheck", {}) or {})
        self.check_enabled = bool(sc.get("enabled", True))
        self.check_daily = bool(sc.get("daily", True))
        self.check_hour = int(sc.get("daily_hour", 7))
        self.check_announce_ok = bool(sc.get("announce_when_healthy", False))
        self.check_print = bool(sc.get("print_daily", True))
        self.check_print_all = bool(sc.get("print_everything", False))
        self._check_last_day: str | None = None
        self._check_running = False
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
        await self._reminder_tick()
        if self.cases_on:
            await self._sla_tick()
        self._sweep_tick()
        self._media_tick()
        await self._interval_tick()
        await self._selfcheck_tick()
        await self._monitor_tick()

    async def _monitor_tick(self) -> None:
        """Let the status wall re-read whatever has gone stale.

        It decides what is due; this only has to call it once a second and
        never let it take the loop down with it.
        """
        if self.monitor is None:
            return
        try:
            await self.monitor.tick()
        except Exception:  # noqa: BLE001
            log.exception("monitor tick failed — continuing")

    async def _interval_tick(self) -> None:
        """Drive the exercise timer.

        It lives here rather than owning a task of its own for the same reason
        the reminders do: a sleeping task is a task that quietly disappears
        when anything goes wrong, and this is the one feature where being
        eight seconds late is the whole failure. The runner computes its own
        position from a start time, so a tick that arrives late corrects
        itself rather than drifting.
        """
        runner = getattr(self, "intervals", None)
        if runner is None or not runner.running:
            return
        try:
            await runner.tick()
        except Exception:  # noqa: BLE001
            log.exception("interval tick failed — stopping the timer")
            try:
                runner.stop()
            except Exception:  # noqa: BLE001
                pass

    async def _selfcheck_tick(self) -> None:
        """Once a day, check everything. Speak only if something is wrong.

        Silence is the design, not an oversight. A daily report that says "all
        thirty checks passed" is a report you stop hearing inside a week, and
        then the one that matters goes past you with it. The printed slip is
        the exception — it is a physical artefact you can glance at, and its
        existence is itself proof the printer still works.

        Keyed on the local date rather than an interval, so a restart does not
        re-run it and a machine that was off all morning still gets its check
        when it comes back.
        """
        checker = getattr(self, "selfcheck", None)
        if (checker is None or not self.check_enabled or not self.check_daily
                or self._check_running):
            return
        now = _dt.datetime.now()
        today = now.strftime("%Y-%m-%d")
        if self._check_last_day == today or now.hour < self.check_hour:
            return
        self._check_last_day = today
        self._check_running = True
        try:
            log.info("running the daily self check")
            report = await asyncio.to_thread(checker.run, True)
            await self.hub.to_display({"type": "selfcheck",
                                       **report.as_dict()})

            if self.check_print:
                printer = getattr(getattr(self, "tools", None), "printer", None)
                if printer is not None and printer.enabled:
                    only = "all" if self.check_print_all else "problems"
                    if not report.healthy or self.check_print_all:
                        try:
                            await asyncio.to_thread(
                                printer._send,
                                printer.build_selfcheck(report, only=only))
                        except Exception:  # noqa: BLE001
                            log.exception("could not print the daily check")

            if report.healthy and not self.check_announce_ok:
                log.info("daily self check: all clear, saying nothing")
                return
            said = report.spoken()
            log.info("daily self check: %s", said)
            if self.chime:
                await self.hub.to_audio({"type": "chime", "kind": "alert"})
            await self.hub.to_audio({"type": "speak", "text": said})
            await self.hub.to_audio({"type": "speak_done"})
        except Exception:  # noqa: BLE001
            log.exception("the daily self check failed — continuing")
        finally:
            self._check_running = False

    def _media_tick(self) -> None:
        """Notice a video that ended, and rescue a volume that never came back.

        Both are cheap — a poll() on a child process and a subtraction — and
        both matter for the same reason: the failure is silent. Without the
        first, "what's playing?" names something that finished an hour ago.
        Without the second, one dropped state message leaves everything you
        play afterwards at a murmur, with nothing in the logs to explain it.
        """
        media = getattr(self, "media", None)
        if media is None:
            return
        try:
            media.tick()
        except Exception:  # noqa: BLE001
            log.exception("media tick failed — continuing")

    def _sweep_tick(self) -> None:
        """Look for memories that have quietly gone stale.

        Cheap enough to sit on the one-second tick: `due()` is a subtraction,
        and the sweep itself only runs once a day. When it does run it is two
        SQL queries over a few dozen rows — no model call, no network. That is
        what makes an always-on sweep affordable rather than a thing you
        remember to switch on.

        It flags and never edits, so there is nothing here to announce. What
        it finds surfaces on the dashboard and when you ask.
        """
        sweep = getattr(self, "sweep", None)
        if sweep is None or not sweep.due():
            return
        try:
            sweep.run()
        except Exception:  # noqa: BLE001
            log.exception("memory sweep failed — continuing")

    async def _sla_tick(self) -> None:
        """Speak up as an investigation's deadline approaches.

        The deadline is absolute and lives in SQLite, so it survives a restart
        with the clock intact — which matters, because the SLA belongs to
        whoever wrote it and kept running while we were down.
        """
        now = time.time()
        for case in self.store.cases_open():
            if not case["sla_due"]:
                continue

            speak: tuple[str, str] | None = None
            for marker, at, phrase in self.sla.markers(case["opened"],
                                                       case["sla_due"]):
                if now < at:
                    continue
                # Claim it whether or not we end up speaking. An unclaimed
                # past marker is true on every subsequent tick, and the
                # scheduler ticks once a second.
                if not self.store.case_alert_once(case["id"], marker):
                    continue
                limit = (SLA_STALE_BREACH_SECONDS if marker == "breach"
                         else SLA_STALE_SECONDS)
                if now - at > limit:
                    log.info("case %d: %s marker passed while down — not "
                             "announcing", case["id"], marker)
                    continue
                # In normal running only one marker comes due per tick. Several
                # at once means we were down across them, and reading the whole
                # backlog out — "halfway through", then "75 percent through",
                # then the breach — tells him nothing the last one does not.
                speak = (marker, phrase)

            if not speak:
                continue
            marker, phrase = speak
            said = f"{case['title']}. {phrase}"
            log.info("case %d SLA: %s", case["id"], phrase)
            await self.hub.to_display({"type": "sla", "case_id": case["id"],
                                       "marker": marker, "text": said})
            # The checklist goes back up with the warning. Being told the clock
            # is running is only half of it; what is still unticked is the
            # other half, and it is the half you can act on.
            await self.hub.to_display({"type": "show_case"})
            if self.chime:
                await self.hub.to_audio({"type": "chime", "kind": "alert"})
            await self.hub.to_audio({"type": "speak", "text": said})
            await self.hub.to_audio({"type": "speak_done"})

    async def _reminder_tick(self) -> None:
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
