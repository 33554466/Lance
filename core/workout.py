"""Workouts as checklists with a clock — the casework machinery, reused.

A training session and a phishing investigation turn out to be the same shape:
an ordered list of steps, grouped into phases, worked through while your hands
and attention are on something else, with results recorded against individual
steps as you go. So this rides on the `cases` and `case_steps` tables rather
than inventing new ones, and inherits the screen, the tick-off matching, and
the persistence for free.

Two things make it different from a playbook, and both come from how workouts
actually arrive:

  * The SESSION IS THE FILE. A phishing playbook is the same every time; a
    workout is different every day. So there is no workouts.yaml — you drop a
    text file in and say you are starting. That is also the answer to "how do
    I get a workout in here without dictating it": scp a file, say go.

  * TICKING OFF STARTS A CLOCK. Finishing a set means resting, and the rest is
    part of the work. `log_set` records what you lifted and schedules the
    countdown in one move, because mid-set is exactly when you do not want to
    be saying a second sentence.

The file format is deliberately plain text rather than YAML. You will be
pasting these, editing them one-handed, and generating them elsewhere, and a
format with significant indentation is a bad choice for all three.

    # Push Day A            <- title
    ## Main                 <- phase
    Bench press, 5x5 @ 185 | rest 180
    Dips, 3 x AMRAP         <- rest falls back to the configured default
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

log = logging.getLogger("assistant.workout")

# A date anywhere in the filename turns a folder of files into a schedule.
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# "| rest 180", "| rest 180s", "| 180s", "| 180". The pipe is the marker: it
# never occurs in an exercise name, so nothing is stripped by accident.
_REST = re.compile(r"\|\s*(?:rest\s*)?(\d+)\s*s?\s*$", re.I)

MAX_SPOKEN = 3


def parse(text: str, default_rest: int = 90) -> tuple[str, list[dict]]:
    """Plain text -> (title, steps). Never raises on odd input.

    A workout file is written by a person or generated elsewhere; a parser
    that throws on a stray blank line or a missing header is a parser that
    loses you a session. Anything unrecognised becomes an exercise, which is
    the useful failure.
    """
    title, phase = "", None
    steps: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("## "):
            phase = line[3:].strip()
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        rest = default_rest
        m = _REST.search(line)
        if m:
            rest = int(m.group(1))
            line = line[:m.start()].strip()
        if not line:
            continue
        steps.append({
            "side": "workout",
            "phase": phase,
            "key": re.sub(r"[^a-z0-9]+", "-", line.lower())[:60].strip("-"),
            "text": line,
            # Aliases come from the exercise name with the set/rep tail
            # dropped, so "bench press" ticks off "Bench press, 5x5 @ 185".
            # Speech will never include the numbers.
            "aliases": _aliases(line),
            "rest": rest,
        })
    if not title:
        title = "Workout"
    return title, steps


def _aliases(line: str) -> list[str]:
    """Spoken forms of an exercise line."""
    head = line.split(",")[0].split("|")[0].strip()
    head = re.sub(r"\s*\d+\s*x\s*\d+.*$", "", head, flags=re.I).strip()
    head = re.sub(r"\s*@.*$", "", head).strip()
    out = {head.lower()}
    # Drop leading qualifiers so "incline dumbbell press" also answers to
    # "dumbbell press" — but never down to a single word. A one-word alias
    # steals from the exercise that word actually names: generating "bench"
    # for "Empty bar bench" means saying "bench" ticks off the warm-up
    # instead of the working set, which is exactly the silent misfile this
    # is meant to prevent.
    parts = head.lower().split()
    for i in range(1, len(parts) - 1):
        out.add(" ".join(parts[i:]))
    return [a for a in out if len(a) >= 3]


class WorkoutTools:
    """start_workout / log_set / next_exercise / last_time / finish_workout."""

    _last_next: int | None = None
    # The step most recently logged. "Rest" on its own has to mean "the rest
    # that belongs to what I just did", and that is the only way to know.
    _last_step: int | None = None

    def __init__(self, cfg: dict, store, cases=None):
        w = cfg.get("workouts", {}) or {}
        self.enabled = bool(w.get("enabled", False))
        self.dir = Path(w.get("path", "~/assistant/workouts")).expanduser()
        self.default_rest = int(w.get("default_rest_seconds", 90))
        self.announce_rest = bool(w.get("announce_rest", True))
        self.store = store
        self.cases = cases
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            log.info("workouts: %s (%d files)", self.dir, len(self._files()))

    def _files(self) -> list[Path]:
        """Newest first. Dropping a file in is how a session is chosen."""
        if not self.dir.is_dir():
            return []
        return sorted((p for p in self.dir.iterdir()
                       if p.is_file() and p.suffix.lower() in
                       (".txt", ".md", ".workout")),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    @staticmethod
    def _label(path: Path) -> str:
        """The spoken name of a file: no date, no hyphens, no extension."""
        stem = _DATE.sub("", path.stem).strip("-_ ")
        return (stem.replace("-", " ").replace("_", " ").strip()
                or path.stem.replace("-", " "))

    def _dated(self, path: Path) -> str | None:
        """The YYYY-MM-DD in a filename, if there is one."""
        m = _DATE.search(path.name)
        return m.group(0) if m else None

    def _pick(self, name: str | None) -> tuple[Path | None, str]:
        """(file, why). `why` is '' on a clean hit, or a sentence to say.

        Selection changes shape once a whole programme is on the box. With one
        or two ad-hoc files, "today's workout" sensibly means the newest one.
        With a month of them dropped in at once, every file has the same
        timestamp and "newest" is a coin toss — so a date in the filename wins
        over the clock whenever one is present.
        """
        files = self._files()
        if not files:
            return None, ""

        if name and name.strip():
            want = re.sub(r"[^a-z0-9]+", "", name.lower())
            for p in files:
                if want and want in re.sub(r"[^a-z0-9]+", "", p.stem.lower()):
                    return p, ""
            return None, ""

        today = time.strftime("%Y-%m-%d")
        dated = sorted((p for p in files if self._dated(p) == today),
                       key=lambda p: p.name)
        if dated:
            # Two sessions on one date is a double day, not a mistake. Take
            # the one not already finished, so the second "start today's
            # workout" opens the evening session rather than repeating the
            # morning.
            fresh = [p for p in dated if not self.store.workout_ran_today(p.name)]
            if fresh:
                return fresh[0], ""
            return dated[0], "That one is already done today, restarting it. "

        programme = [p for p in files if self._dated(p)]
        if programme:
            # A dated programme is installed and today is not in it — a rest
            # day. Say so and name the next one rather than silently starting
            # last Tuesday's session.
            ahead = sorted((p for p in programme if (self._dated(p) or "") > today),
                           key=lambda p: (self._dated(p) or "", p.name))
            if ahead:
                when = self._say_date(self._dated(ahead[0]))
                return None, (f"Nothing scheduled today. Next is "
                              f"{self._label(ahead[0])} {when}.")
            return None, "Nothing scheduled today, and the programme has run out."

        return files[0], ""      # undated folder: newest, as before

    @staticmethod
    def _say_date(iso: str | None) -> str:
        if not iso:
            return ""
        try:
            t = time.strptime(iso, "%Y-%m-%d")
        except ValueError:
            return ""
        days = (time.mktime(t) - time.mktime(
            time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))) / 86400
        if days <= 1:
            return "tomorrow"
        if days < 7:
            return f"on {time.strftime('%A', t)}"
        n = int(time.strftime("%-d", t))
        suffix = ("th" if 11 <= n % 100 <= 13
                  else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
        return f"on {time.strftime('%A', t)} the {n}{suffix}"

    # -- schemas -----------------------------------------------------

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [
            {
                "name": "start_workout",
                "description": (
                    "Begin a training session from a workout file. Use when "
                    "the user says they are starting a workout, training, "
                    "lifting, or names a session. With no `name` this takes "
                    "the most recently added file, which is what 'today's "
                    "workout' means. Puts the session on the monitor. Reply "
                    "in one short sentence — they want to start moving."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Part of the filename or title, if they named "
                                "one. Omit for the newest file."
                            ),
                        },
                    },
                },
            },
            {
                "name": "log_set",
                "description": (
                    "Record a completed set and tick the exercise off. Use "
                    "whenever the user says they finished something, or calls "
                    "out a lift: 'squats two twenty five for five', 'bench "
                    "done', 'that's the dips'. Put whatever they said about "
                    "weight and reps in `result` — it is kept against the "
                    "exercise and is what makes 'what did I lift last time' "
                    "work later. A rest countdown starts automatically, so do "
                    "not also set a timer. Keep the reply to a few words."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "exercise": {
                            "type": "string",
                            "description": (
                                "What they called it, in their words. Do not "
                                "translate it to the file's wording."
                            ),
                        },
                        "result": {
                            "type": "string",
                            "description": (
                                "Weight and reps as said: '225 for 5', "
                                "'bodyweight, 12 reps', '3 sets of 10'. Omit "
                                "if they only said they were done."
                            ),
                        },
                        "done": {
                            "type": "boolean",
                            "description": (
                                "False if there are more sets of this exercise "
                                "to come, so it stays open on the list. "
                                "Default true."
                            ),
                        },
                    },
                    "required": ["exercise"],
                },
            },
            {
                "name": "start_rest",
                "description": (
                    "Start the rest countdown. Use ONLY when the user asks "
                    "for it — 'rest', 'start the rest', 'start the timer', "
                    "'ninety seconds'. With no seconds given it uses the rest "
                    "written into the workout file for whatever they last "
                    "logged. Never call this on your own after a set; they "
                    "will say when they want it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "integer",
                            "description": (
                                "Only if they named a length. Otherwise omit "
                                "and the file's own rest is used."
                            ),
                        },
                    },
                },
            },
            {
                "name": "next_exercise",
                "description": (
                    "What is coming up in the session. Use for 'what's next', "
                    "'what have I got left'."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "last_time",
                "description": (
                    "What the user lifted for an exercise in previous "
                    "sessions. Use for 'what did I squat last time', 'what "
                    "was my bench last week'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"exercise": {"type": "string"}},
                    "required": ["exercise"],
                },
            },
            {
                "name": "finish_workout",
                "description": (
                    "End the session and stop the clock. Use for 'that's me "
                    "done', 'finished'."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "list_workouts",
                "description": "Which workout files are available to start.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    _NAMES = {"start_workout", "log_set", "next_exercise", "last_time",
              "finish_workout", "list_workouts", "start_rest"}

    def run_sync(self, name: str, args: dict) -> str | None:
        if name not in self._NAMES:
            return None
        if not self.enabled:
            return "Workouts are turned off in the configuration."
        if name == "start_workout":
            return self._start(args)
        if name == "list_workouts":
            files = self._files()
            if not files:
                return f"No workout files in {self.dir}."
            today = time.strftime("%Y-%m-%d")
            ahead = sorted((p for p in files
                            if (self._dated(p) or "") >= today),
                           key=lambda p: (self._dated(p) or "", p.name))
            if ahead:
                parts = []
                for p in ahead[:4]:
                    d = self._dated(p)
                    when = "today" if d == today else self._say_date(d)
                    parts.append(f"{when}, {self._label(p)}")
                more = len(ahead) - len(parts)
                return ("Coming up: " + "; ".join(parts)
                        + (f". And {more} more scheduled." if more > 0 else "."))
            return (f"{len(files)} on file: "
                    + ", ".join(self._label(p) for p in files[:6]))
        if name == "last_time":
            return self._last_time(args.get("exercise", ""))

        case = self._active()
        if not case:
            return "No workout is running. Say you are starting one."
        if name == "log_set":
            return self._log(case, args)
        if name == "next_exercise":
            return self._next(case)
        if name == "start_rest":
            return self._rest(case, args)
        return self._finish(case)

    # -- operations --------------------------------------------------

    def _active(self):
        case = self.store.case_active()
        return case if case and case["kind"] == "workout" else None

    def _start(self, args: dict) -> str:
        path, why = self._pick(args.get("name"))
        if path is None:
            if not self._files():
                return (f"There are no workout files. Drop one in "
                        f"{self.dir.name} and say it again.")
            if why:
                return why          # rest day, or the programme has ended
            # Name what IS there rather than a bare miss. A mis-heard session
            # name is the common case — "full workout A" for "Full Body A" —
            # and the useful reply is the list, not a refusal.
            have = ", ".join(self._label(p) for p in self._files()[:5])
            return (f"I have no workout called {args.get('name')!r}. "
                    f"I have: {have}.")
        try:
            title, steps = parse(path.read_text(), self.default_rest)
        except Exception as exc:  # noqa: BLE001
            return f"Could not read that workout: {type(exc).__name__}."
        if title == "Workout":
            # No "# " line in the file. Fall back to the filename with the
            # date stripped, so a dated programme still announces "Full Body
            # A" rather than eleven sessions all called "Workout".
            title = self._label(path).title() or "Workout"
        if not steps:
            return f"{path.name} has no exercises in it."

        # No SLA on a workout. The case machinery wants a deadline; a training
        # session does not have one, and being told you are 75% through your
        # workout's allotted time is not a thing anybody wants shouted at them.
        # The source filename goes in `ref`, which a workout has no other use
        # for. It is what lets a double day know the morning session is
        # already done.
        case_id = self.store.case_open("workout", title, "session", None,
                                       path.name, steps)
        self.store.case_steps_set_rest(
            case_id, {s["key"]: s["rest"] for s in steps})
        phases = [s["phase"] for s in steps if s["phase"]]
        n_phase = len(dict.fromkeys(phases))
        log.info("workout %d: %s (%d exercises from %s)",
                 case_id, title, len(steps), path.name)
        return (f"{why}{title}. {len(steps)} "
                + ("exercise" if len(steps) == 1 else "exercises")
                + (f" across {n_phase} sections" if n_phase > 1 else "")
                + ", on screen.")

    def _log(self, case, args: dict) -> str:
        if self.cases is None:
            return "Matching is unavailable."
        spoken = args.get("exercise", "")
        step, tied = self.cases.match_step(case["id"], spoken)
        if tied:
            return "Which one: " + " — or — ".join(
                r["text"].split(",")[0] for r in tied[:3]) + "?"
        if not step:
            return f"Nothing in this session matches {spoken!r}."

        result = (args.get("result") or "").strip()
        done = args.get("done", True)
        if not isinstance(done, bool):
            done = True
        # Keep every set, not just the last. "225 for 5 / 225 for 5 / 225 for
        # 3" is the useful record; overwriting leaves you thinking you hit
        # three clean sets.
        prior = (step["finding"] or "").strip()
        merged = f"{prior} / {result}" if prior and result else (result or prior)
        self.store.case_step_set(step["id"], done, merged or None)

        self._last_step = step["id"]
        rest = self.store.case_step_rest(step["id"])
        if rest is None:
            rest = self.default_rest
        tail = ""
        if done and rest > 0 and self.announce_rest:
            tail = " " + self._rest_for(step, rest)

        rows = self.store.case_steps(case["id"], "workout")
        nxt = next((r for r in rows if not r["done"]), None)
        head = "Logged." if result else "Done."
        if nxt is None:
            self._last_next = None
            return f"{head} That is the session finished."
        # Only name what is next when it has changed. Working out of order —
        # skipping a warm-up, coming back to it — otherwise means hearing the
        # same exercise called out after every single set.
        if nxt["id"] == self._last_next:
            return f"{head}{tail}"
        self._last_next = nxt["id"]
        return f"{head}{tail} Next, {nxt['text'].split(',')[0]}."

    def _rest_for(self, step, seconds: int) -> str:
        """Schedule the rest countdown for one exercise. Returns what to say."""
        label = step["text"].split(",")[0].strip().lower()
        # Replace, never stack. Three sets of bench would otherwise schedule
        # three timers, and the first two go off while you are under the bar
        # for the third — which trains you to ignore the one that matters.
        self.store.cancel_reminders(f"rest is up, {label}")
        self.store.add_reminder(time.time() + seconds,
                                f"rest is up, {label}", kind="timer")
        return f"{seconds} seconds."

    def _rest(self, case, args: dict) -> str:
        """Start the rest on request rather than automatically."""
        secs = args.get("seconds")
        rows = self.store.case_steps(case["id"], "workout")
        step = next((r for r in rows if r["id"] == self._last_step), None)
        if step is None:
            # Nothing logged this session yet — most likely a restart, or he
            # is resting before starting. Fall back to the last DONE step, and
            # then to the configured default.
            step = next((r for r in reversed(rows) if r["done"]), None)
        if step is None:
            if secs:
                self.store.add_reminder(time.time() + int(secs),
                                        "rest is up", kind="timer")
                return f"{int(secs)} seconds."
            return "Log a set first and I will know how long to rest."
        if secs:
            seconds = int(secs)
        else:
            seconds = self.store.case_step_rest(step["id"])
            if seconds is None or seconds <= 0:
                seconds = self.default_rest
        return self._rest_for(step, seconds)

    def _next(self, case) -> str:
        rows = [r for r in self.store.case_steps(case["id"], "workout")
                if not r["done"]]
        if not rows:
            return "Nothing left. That is the session."
        head = rows[:MAX_SPOKEN]
        more = len(rows) - len(head)
        out = "; ".join(r["text"] for r in head)
        return out + (f". And {more} more." if more > 0 else ".")

    def _last_time(self, exercise: str) -> str:
        if not exercise.strip():
            return "Which exercise?"
        rows = self.store.workout_history(exercise, limit=4)
        if not rows:
            return f"No record of {exercise} in a previous session."
        out = []
        for r in rows:
            when = time.strftime("%-d %b", time.localtime(r["opened"]))
            out.append(f"{when}, {r['finding']}")
        return f"{exercise}: " + "; ".join(out)

    def _finish(self, case) -> str:
        rows = self.store.case_steps(case["id"], "workout")
        done = sum(1 for r in rows if r["done"])
        mins = int((time.time() - case["opened"]) // 60)
        self.store.case_close(case["id"], "completed")
        # Cancel any rest timer still counting down. Nothing is worse than
        # being told to get back under the bar twenty minutes after you left.
        self.store.cancel_reminders("rest is up")
        left = len(rows) - done
        tail = f" {left} not done." if left else ""
        return (f"Session finished. {done} of {len(rows)} in "
                f"{mins} minute{'s' if mins != 1 else ''}.{tail}")
