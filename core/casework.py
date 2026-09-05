"""Investigations with a clock on them.

A case is not another named list, and the difference is worth being explicit
about because it drove every decision in this file.

A list is a bag of strings. You add to it, you tick things off, order does not
matter much and nothing bad happens if you never finish. A case is the
opposite: it is a template stamped out at a known moment, it has a deadline
someone else set, the steps came from a playbook rather than from you, and
findings hang off individual steps because the write-up at the end has to
reconstruct what you looked at and what you saw.

Three consequences follow, and they are the whole design:

  * Steps are COPIED onto the case at open time, never referenced. Edit the
    playbook tomorrow and yesterday's closed case is unchanged. A record that
    rewrites itself is not a record.
  * The SLA deadline is absolute and lives in SQLite. A restart mid-case must
    not reset the clock, because the clock is not ours — it belongs to whoever
    wrote the SLA, and it kept running while the service was down.
  * Warnings are claimed with an INSERT, not a SELECT-then-INSERT. The
    scheduler ticks once a second; anything less strict announces "halfway
    through your SLA" sixty times a minute.

The playbooks are YAML on disk rather than rows in the database, because the
investigation checklist is something you will want to edit in a text editor
at 8am after learning something at 5pm the day before.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import yaml

log = logging.getLogger("assistant.casework")

# Cap the read-back. Twenty-eight steps recited aloud is not a status update,
# it is a hostage situation.
MAX_SPOKEN_STEPS = 4


# ---------------------------------------------------------------- playbooks

def _flatten(spec: dict) -> list[dict]:
    """Playbook YAML -> the flat, ordered step list the database stores."""
    steps: list[dict] = []
    for side in ("investigation", "admin"):
        for group in spec.get(side) or []:
            phase = (group or {}).get("phase")
            for st in (group or {}).get("steps") or []:
                if not st or not st.get("text"):
                    continue
                steps.append({
                    "side": side,
                    "phase": phase,
                    "key": str(st.get("key") or st["text"])[:60],
                    "text": str(st["text"]).strip(),
                    "aliases": [str(a).lower() for a in (st.get("aliases") or [])],
                })
    return steps


class Playbooks:
    """Every YAML file in the playbook directory, reloaded on demand.

    Reloaded rather than cached because the point of keeping these as files is
    that you can fix a step and start the next case with it — without
    restarting a service that is holding a warm Whisper model in memory.
    """

    def __init__(self, path: Path):
        self.path = path

    def all(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not self.path.is_dir():
            return out
        for f in sorted(self.path.glob("*.yaml")) + sorted(self.path.glob("*.yml")):
            try:
                spec = yaml.safe_load(f.read_text()) or {}
            except Exception as exc:  # noqa: BLE001
                # A broken playbook must not take the assistant down with it.
                # It just is not offered until it parses.
                log.error("playbook %s did not parse: %s", f.name, exc)
                continue
            kind = str(spec.get("kind") or f.stem).lower()
            spec["_file"] = f.name
            out[kind] = spec
        return out

    def resolve(self, raw: str | None) -> tuple[str, dict] | None:
        """Map what was said to a playbook. Exact, then alias, then words."""
        books = self.all()
        if not books:
            return None
        if not raw or not str(raw).strip():
            # One playbook installed means there is nothing to disambiguate.
            if len(books) == 1:
                k = next(iter(books))
                return k, books[k]
            return None

        text = " ".join(str(raw).lower().translate(
            str.maketrans("", "", ".,!?;:\"'’")).split())
        for filler in ("the ", " investigation", " case", " incident"):
            text = text.replace(filler, " ")
        text = " ".join(text.split())

        if text in books:
            return text, books[text]
        words = set(text.split())
        best, best_len = None, 0
        for kind, spec in books.items():
            for alias in [kind] + [str(a).lower()
                                   for a in (spec.get("aliases") or [])]:
                parts = alias.split()
                if set(parts) <= words and len(alias) > best_len:
                    best, best_len = kind, len(alias)
        if best:
            return best, books[best]
        return None


# ---------------------------------------------------------------- SLA

class Sla:
    """Turns a severity word into an absolute deadline, and back into speech."""

    def __init__(self, cfg: dict):
        s = (cfg.get("casework", {}) or {}).get("sla", {}) or {}
        self.default_minutes = int(s.get("default", 240))
        self.tiers = {str(k).lower(): int(v)
                      for k, v in (s.get("tiers") or {}).items()}
        # Fractions of the window at which to speak up, plus a fixed final
        # warning. Fractions alone are wrong for a short SLA — 90% of an hour
        # is six minutes of notice — and a fixed warning alone is wrong for a
        # long one. Both, deduplicated, covers each.
        self.warn_at = [float(x) for x in (s.get("warn_at") or [0.5, 0.75])]
        self.final_minutes = int(s.get("warn_final_minutes", 15))

    def minutes_for(self, severity: str) -> int:
        return self.tiers.get((severity or "").lower(), self.default_minutes)

    def due_from(self, opened: float, severity: str) -> float:
        return opened + self.minutes_for(severity) * 60

    def markers(self, opened: float, due: float) -> list[tuple[str, float, str]]:
        """(marker id, absolute time, spoken phrase) for one case."""
        span = max(1.0, due - opened)
        out: list[tuple[str, float, str]] = []
        for frac in self.warn_at:
            if not 0 < frac < 1:
                continue
            at = opened + span * frac
            left = due - at
            out.append((f"frac{frac}", at,
                        f"{int(round(frac * 100))} percent through the S L A. "
                        f"{human_left(left)} left."))
        if self.final_minutes > 0:
            at = due - self.final_minutes * 60
            if at > opened:
                out.append((f"final{self.final_minutes}", at,
                            f"{self.final_minutes} minutes left on the S L A."))
        out.append(("breach", due, "The S L A has passed."))
        out.sort(key=lambda t: t[1])

        # On a short SLA the fractional and fixed warnings collide: 75% of an
        # hour and "fifteen minutes left" are the same instant, and hearing
        # both, a second apart, saying the same thing, is how you learn to
        # stop listening to the thing warning you about deadlines. Keep the
        # first of any cluster — it carries the percentage as well as the
        # time remaining, so it is the more informative of the two.
        kept: list[tuple[str, float, str]] = []
        for m in out:
            if kept and m[0] != "breach" and m[1] - kept[-1][1] < 60:
                continue
            kept.append(m)
        return kept


def human_left(seconds: float) -> str:
    """'2 hours 10 minutes', for saying out loud."""
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} seconds"
    mins = seconds // 60
    if mins < 60:
        return f"{mins} minute{'s' if mins != 1 else ''}"
    hours, rem = divmod(mins, 60)
    head = f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{head} {rem} minutes" if rem else head


def human_over(seconds: float) -> str:
    return human_left(seconds) + " over"


# ---------------------------------------------------------------- tools

_PUNCT = str.maketrans("", "", ".,!?;:\"'’")


def normalise(text: str) -> str:
    return " ".join(str(text).lower().translate(_PUNCT).split())


# Endings a word may pick up and still be the same word. Deliberately a closed
# list rather than "any short suffix": a length budget matches report/reporter
# and check/checksum, which are different steps in this very playbook, and a
# checklist that ticks off the wrong line is worse than one that ticks nothing.
_INFLECTIONS = ("s", "es", "d", "ed", "ing")


def _same_word(a: str, b: str) -> bool:
    """Loose word equality, so speech need not match the playbook's wording.

    People say "I purged it", not "purge". They say "hashes", "blocked",
    "notified", "checked the links". Demanding exact matches means writing
    every inflection of every verb into the alias lists forever, and still
    missing the one he used this morning.

    Words under four characters must match exactly, which is what keeps "the",
    "did", "it" and "url" from matching things they should not.
    """
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if long.startswith(short) and long[len(short):] in _INFLECTIONS:
        return True
    # A silent trailing e is dropped before -ing/-ed: purge -> purging.
    if short.endswith("e") and long.startswith(short[:-1]) \
            and long[len(short) - 1:] in _INFLECTIONS:
        return True
    # y -> ied/ies: notify -> notified, verify -> verifies.
    if short.endswith("y") and long in (short[:-1] + "ied", short[:-1] + "ies"):
        return True
    return False


def _covered(needles: set[str], haystack: set[str]) -> bool:
    """Every word in `needles` has a loose match somewhere in `haystack`."""
    return bool(needles) and all(
        any(_same_word(n, h) for h in haystack) for n in needles)


class CaseTools:
    """start_case / check_step / case_status / close_case, and friends."""

    def __init__(self, cfg: dict, store):
        c = cfg.get("casework", {}) or {}
        self.enabled = bool(c.get("enabled", False))
        root = Path(c.get("playbook_dir", "~/assistant/playbooks")).expanduser()
        self.books = Playbooks(root)
        self.sla = Sla(cfg)
        self.store = store
        docs = cfg.get("documents", {}) or {}
        self.notes_root = Path(docs.get("path", "~/Documents/Assistant")
                               ).expanduser()
        # The last step announced as "next", so a run of out-of-order ticks
        # does not say "Next: note who reported it" seven times running. It is
        # true every time and useless after the first.
        self._last_next: int | None = None
        if self.enabled:
            found = list(self.books.all())
            log.info("casework: %s (%s)", root,
                     ", ".join(found) if found else "no playbooks found")

    # -- matching ----------------------------------------------------

    def match_step(self, case_id: int, spoken: str,
                   side: str | None = None) -> tuple[object | None, list]:
        """Find the step someone just referred to out loud.

        Returns (best, ambiguous). `ambiguous` is populated only when two
        steps tie, in which case the caller asks rather than guessing — a
        wrongly ticked step in an investigation is worse than a question,
        because it reads afterwards as work that was done.
        """
        text = normalise(spoken)
        if not text:
            return None, []
        words = set(text.split())
        scored: list[tuple[int, object]] = []
        spoken_len = len(" ".join(sorted(words)))
        for row in self.store.case_steps(case_id, side):
            best = 0
            candidates = [a for a in (row["aliases"] or "").split("\n") if a]
            candidates.append(row["text"])
            for alias in candidates:
                a = normalise(alias)
                parts = set(a.split())
                if not parts:
                    continue
                # Containment in either direction, whole words only.
                #
                #   "tick off the hashes"         -> alias "hash" sits inside
                #                                    what was said
                #   "check the sender reputation" -> what was said sits inside
                #                                    the step's own wording
                #
                # A forward match always beats a reverse one, hence the
                # bonus rather than one shared scale. Otherwise the two are
                # measured with different rulers — the alias's length against
                # the whole spoken phrase's — and a long sentence that happens
                # to sit inside a step's wording outranks an alias the person
                # actually said. "The reporter" went to "tell the reporter and
                # thank them" instead of the intake step for exactly that
                # reason.
                if _covered(parts, words):
                    best = max(best, 1000 + len(a))
                elif _covered(words, parts):
                    best = max(best, spoken_len)
            if best:
                scored.append((best, row))
        if not scored:
            return None, []
        scored.sort(key=lambda t: -t[0])
        top = scored[0][0]
        tied = [r for s, r in scored if s == top]
        if len(tied) > 1:
            return None, tied
        return tied[0], []

    # -- schemas -----------------------------------------------------

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        kinds = ", ".join(self.books.all()) or "phishing"
        tiers = ", ".join(self.sla.tiers) or "standard"
        return [
            {
                "name": "start_case",
                "description": (
                    "Open an investigation. Use whenever the user says they "
                    "are starting, beginning, picking up, or working a case — "
                    "for example 'I'm starting a phishing email "
                    f"investigation'. Known playbooks: {kinds}. This starts "
                    "the SLA clock and puts the checklist on the monitor, so "
                    "call it immediately rather than asking questions first. "
                    "Keep your spoken reply to one sentence: the clock is "
                    "running and they want to start looking, not chat."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "description": f"Which playbook. One of: {kinds}",
                        },
                        "title": {
                            "type": "string",
                            "description": (
                                "Short label, if they gave one — the subject "
                                "line, the reporting user, the sender. Omit "
                                "rather than inventing one."
                            ),
                        },
                        "ref": {
                            "type": "string",
                            "description": "Ticket or case number, if said.",
                        },
                        "severity": {
                            "type": "string",
                            "description": (
                                f"SLA tier if stated: {tiers}. Omit unless "
                                "they actually said how urgent it is."
                            ),
                        },
                    },
                },
            },
            {
                "name": "check_step",
                "description": (
                    "Tick a checklist step off the active case. Use whenever "
                    "the user says they did, checked, ran, pulled, or finished "
                    "one of the steps. If they also said what they FOUND, pass "
                    "it as `finding` — 'the domain was registered four days "
                    "ago' belongs on the step, and it is what makes the report "
                    "at the end writable."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "string",
                            "description": (
                                "What they called it, in their words. Do not "
                                "translate it into the checklist's wording — "
                                "matching handles that."
                            ),
                        },
                        "finding": {
                            "type": "string",
                            "description": "What they found, if they said.",
                        },
                    },
                    "required": ["step"],
                },
            },
            {
                "name": "uncheck_step",
                "description": "Put a ticked step back to open.",
                "input_schema": {
                    "type": "object",
                    "properties": {"step": {"type": "string"}},
                    "required": ["step"],
                },
            },
            {
                "name": "case_status",
                "description": (
                    "How the active case stands: progress on both columns, "
                    "time left on the SLA, and what is next. Use for 'where am "
                    "I', 'how long have I got', 'what's left'."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "case_next",
                "description": (
                    "The next few unfinished steps. Use for 'what's next' or "
                    "'what else do I need to look at'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "side": {
                            "type": "string",
                            "enum": ["investigation", "admin"],
                            "description": "Limit to one column if they asked.",
                        },
                    },
                },
            },
            {
                "name": "case_note",
                "description": (
                    "Attach a free-text note to the case that does not belong "
                    "to any one step. Use for observations, decisions, and "
                    "anything they say to 'make a note of'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            {
                "name": "show_case",
                "description": (
                    "Put the case checklist back on the monitor. Say something "
                    "brief out loud too — they may not be looking."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "case_report",
                "description": (
                    "Write everything on the case out to a note file: every "
                    "step, whether it was done, the findings recorded against "
                    "each, and the notes. Use when they ask to write it up, "
                    "export it, or save the case."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "close_case",
                "description": (
                    "Close the active case and stop its SLA clock. Use for "
                    "'close it out', 'I'm done with this one'. If steps are "
                    "still open, say so in your reply — but close it anyway; "
                    "they know their job."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "description": (
                                "Verdict if stated: malicious, suspicious, "
                                "spam, benign, false positive."
                            ),
                        },
                    },
                },
            },
            {
                "name": "list_cases",
                "description": "Every open case, oldest first, with time left.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    # -- dispatch ----------------------------------------------------

    _NAMES = {"start_case", "check_step", "uncheck_step", "case_status",
              "case_next", "case_note", "show_case", "case_report",
              "close_case", "list_cases"}

    def run_sync(self, name: str, args: dict) -> str | None:
        if name not in self._NAMES:
            return None
        if not self.enabled:
            return "Casework is turned off in the configuration."
        if name == "start_case":
            return self._start(args)
        if name == "list_cases":
            return self._list_cases()

        case = self.store.case_active()
        if not case:
            return ("No case is open. Say you are starting one and I will "
                    "open it.")
        handler = {
            "check_step": lambda: self._check(case, args, True),
            "uncheck_step": lambda: self._check(case, args, False),
            "case_status": lambda: self._status(case),
            "case_next": lambda: self._next(case, args.get("side")),
            "case_note": lambda: self._note(case, args),
            "show_case": lambda: self._show(case),
            "case_report": lambda: self._report(case),
            "close_case": lambda: self._close(case, args),
        }[name]
        return handler()

    # -- operations --------------------------------------------------

    def _start(self, args: dict) -> str:
        found = self.books.resolve(args.get("kind"))
        if not found:
            have = ", ".join(self.books.all()) or "none installed"
            return (f"I do not have a playbook for that. I have: {have}.")
        kind, spec = found
        steps = _flatten(spec)
        if not steps:
            return f"The {kind} playbook has no steps in it."

        severity = (args.get("severity")
                    or spec.get("severity_default") or "standard")
        now = time.time()
        due = self.sla.due_from(now, severity)
        title = (args.get("title") or "").strip() or str(
            spec.get("title") or kind).strip()

        case_id = self.store.case_open(kind, title, severity, due,
                                       args.get("ref"), steps)
        inv = sum(1 for s in steps if s["side"] == "investigation")
        adm = len(steps) - inv
        ref = f" on {args['ref']}" if args.get("ref") else ""
        log.info("case %d opened: %s %r severity=%s due in %s",
                 case_id, kind, title, severity, human_left(due - now))
        return (f"Case open{ref}. {human_left(due - now)} on the "
                f"{severity} S L A. {inv} investigation steps and {adm} "
                f"of yours, on screen now.")

    def _check(self, case, args: dict, done: bool) -> str:
        step, tied = self.match_step(case["id"], args.get("step", ""))
        if tied:
            names = " — or — ".join(r["text"] for r in tied[:3])
            return f"Which one: {names}?"
        if not step:
            return (f"Nothing on the checklist matches "
                    f"{args.get('step', 'that')!r}.")
        if bool(step["done"]) == done and not args.get("finding"):
            return (f"{step['text']} is already "
                    f"{'ticked off' if done else 'open'}.")

        self.store.case_step_set(step["id"], done, args.get("finding"))
        if not done:
            return f"Reopened: {step['text']}."

        prog = self.store.case_progress(case["id"])
        d, t = prog.get(step["side"], (0, 0))
        left = self.store.case_steps(case["id"], step["side"])
        nxt = next((r for r in left if not r["done"]), None)
        note = " Noted." if args.get("finding") else ""
        if not nxt:
            self._last_next = None
            return f"Ticked off. {d} of {t}.{note} That side is complete."
        # Only name it if it is news. Working down the list out of order — and
        # everyone does — otherwise means hearing the same sentence every time.
        if nxt["id"] == self._last_next:
            return f"Ticked off. {d} of {t}.{note}"
        self._last_next = nxt["id"]
        return f"Ticked off. {d} of {t}.{note} Next: {nxt['text']}."

    def _status(self, case) -> str:
        prog = self.store.case_progress(case["id"])
        bits = []
        for side, label in (("investigation", "investigation"),
                            ("admin", "your tasks")):
            if side in prog:
                d, t = prog[side]
                bits.append(f"{label} {d} of {t}")
        head = "; ".join(bits) if bits else "nothing on the checklist"

        clock = self._clock_phrase(case)
        rows = [r for r in self.store.case_steps(case["id"]) if not r["done"]]
        nxt = f" Next: {rows[0]['text']}." if rows else " Everything is done."
        return f"{case['title']}: {head}. {clock}{nxt}"

    def _clock_phrase(self, case) -> str:
        if not case["sla_due"]:
            return "No S L A on this one."
        left = case["sla_due"] - time.time()
        if left <= 0:
            return f"S L A passed, {human_over(-left)}."
        return f"{human_left(left)} left on the S L A."

    def _next(self, case, side: str | None) -> str:
        side = side if side in ("investigation", "admin") else None
        rows = [r for r in self.store.case_steps(case["id"], side)
                if not r["done"]]
        if not rows:
            where = f"the {side}" if side else "the case"
            return f"Nothing open on {where}."
        head = rows[:MAX_SPOKEN_STEPS]
        more = len(rows) - len(head)
        out = "; ".join(r["text"] for r in head)
        return out + (f". And {more} more after that." if more > 0 else ".")

    def _note(self, case, args: dict) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            return "Nothing to note."
        self.store.case_note(case["id"], text)
        return "Noted on the case."

    def _show(self, case) -> str:
        prog = self.store.case_progress(case["id"])
        done = sum(d for d, _ in prog.values())
        total = sum(t for _, t in prog.values())
        return (f"{case['title']} on screen. {done} of {total} done. "
                f"{self._clock_phrase(case)}")

    def _list_cases(self) -> str:
        rows = self.store.cases_open()
        if not rows:
            return "No open cases."
        parts = []
        for r in rows:
            prog = self.store.case_progress(r["id"])
            done = sum(d for d, _ in prog.values())
            total = sum(t for _, t in prog.values())
            parts.append(f"{r['title']}, {done} of {total}, "
                         f"{self._clock_phrase(r).lower()}")
        return f"{len(rows)} open: " + "; ".join(parts)

    def _report(self, case) -> str:
        """Dump the case to a note file. This is the payoff for attaching
        findings to steps rather than keeping them in your head."""
        self.notes_root.mkdir(parents=True, exist_ok=True)
        opened = time.strftime("%Y-%m-%d %H:%M", time.localtime(case["opened"]))
        slug = re.sub(r"[^a-z0-9]+", "-", case["title"].lower()).strip("-")[:50]
        path = self.notes_root / (
            f"case-{case['id']}-{slug or case['kind']}.md")

        lines = [f"# {case['title']}", ""]
        lines.append(f"- Kind: {case['kind']}")
        if case["ref"]:
            lines.append(f"- Reference: {case['ref']}")
        lines.append(f"- Severity: {case['severity']}")
        lines.append(f"- Opened: {opened}")
        if case["sla_due"]:
            lines.append("- SLA due: " + time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(case["sla_due"])))
        if case["closed"]:
            lines.append("- Closed: " + time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(case["closed"])))
        if case["outcome"]:
            lines.append(f"- Outcome: {case['outcome']}")
        lines.append("")

        for side, heading in (("investigation", "Investigation"),
                              ("admin", "Tasks")):
            rows = self.store.case_steps(case["id"], side)
            if not rows:
                continue
            lines += [f"## {heading}", ""]
            phase = None
            for r in rows:
                if r["phase"] and r["phase"] != phase:
                    phase = r["phase"]
                    # The blank line before the heading is not cosmetic —
                    # CommonMark will not close the preceding list without it,
                    # and the heading renders as literal "### Attachments".
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines += [f"### {phase}", ""]
                box = "x" if r["done"] else " "
                lines.append(f"- [{box}] {r['text']}")
                if r["finding"]:
                    lines.append(f"  - {r['finding']}")
            lines.append("")

        if (case["notes"] or "").strip():
            lines += ["## Notes", "", "```", case["notes"].rstrip(), "```", ""]

        path.write_text("\n".join(lines))
        log.info("case %d written to %s", case["id"], path)
        return f"Written to {path.name}."

    def _close(self, case, args: dict) -> str:
        prog = self.store.case_progress(case["id"])
        open_n = sum(t - d for d, t in prog.values())
        self.store.case_close(case["id"], args.get("outcome"))
        elapsed = human_left(time.time() - case["opened"])
        verdict = f" Logged as {args['outcome']}." if args.get("outcome") else ""
        if case["sla_due"] and time.time() > case["sla_due"]:
            timing = " Outside the S L A."
        elif case["sla_due"]:
            timing = f" Inside the S L A with " \
                     f"{human_left(case['sla_due'] - time.time())} to spare."
        else:
            timing = ""
        tail = (f" {open_n} step{'s' if open_n != 1 else ''} left unticked."
                if open_n else "")
        log.info("case %d closed after %s", case["id"], elapsed)
        return f"Closed after {elapsed}.{verdict}{timing}{tail}"
