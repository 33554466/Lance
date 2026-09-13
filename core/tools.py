"""Tools that run on the appliance itself.

This is the other half of tool use. The web search tool runs on Anthropic's
servers and you never see it execute. Everything in this file runs HERE, on
your machine, with your filesystem — which is exactly why it is worth being
careful about.

Design rules, in order of importance:

  1. Everything is confined to one directory. A model that can write anywhere
     is a model that can overwrite ~/.bashrc because it misread a title. Every
     path is resolved and then checked to be inside the notes directory, after
     symlinks are followed. There is no escape hatch and no configuration that
     turns the check off.

  2. Filenames are generated, never taken. The model supplies a title; this
     module slugifies it. Whatever the model says, the filename is drawn from
     a small alphabet and cannot contain a slash or a dot-dot.

  3. Nothing is destroyed. Saving over an existing name writes "-2" instead.
     Appending is available as its own verb. There is no delete tool, because
     "Lance, throw that away" mis-transcribed is not a risk worth carrying for
     a convenience you can get from a file manager.

  4. Results are written for a voice. A tool that returns a wall of text makes
     the assistant read a wall of text. Results are short, factual, and say
     what happened rather than describing the data.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import time as _t
import re
from pathlib import Path

log = logging.getLogger("assistant.tools")

# Deliberately narrow. Anything outside this is dropped, so a title can never
# introduce a path separator, a leading dot, or shell-significant characters.
_SLUG_OK = re.compile(r"[^a-z0-9]+")

MAX_TITLE_SLUG = 60
MAX_LIST = 25
MAX_READ_CHARS = 4000


def slugify(title: str) -> str:
    slug = _SLUG_OK.sub("-", title.lower()).strip("-")[:MAX_TITLE_SLUG]
    return slug or "note"


class DocumentTools:
    """Plain-text notes in one directory. The Linux equivalent of Notepad."""

    def __init__(self, cfg: dict):
        d = cfg.get("documents", {}) or {}
        self.enabled = bool(d.get("enabled", False))
        self.root = Path(d.get("path", "~/Documents/Assistant")).expanduser()
        self.suffix = "." + str(d.get("format", "txt")).lstrip(".")
        self.max_chars = int(d.get("max_chars", 20000))
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            # Resolve once, at startup, so the containment check below is
            # comparing against a real path rather than one with symlinks in
            # it. /home being a symlink is common enough to matter.
            self.root = self.root.resolve()
            log.info("documents: %s", self.root)

    # -- safety ------------------------------------------------------

    def _path_for(self, title: str, unique: bool = True) -> Path:
        """Build a path inside the notes directory from a title. Never trusts
        the title for anything but its letters and digits."""
        base = slugify(title)
        path = (self.root / f"{base}{self.suffix}").resolve()
        self._assert_inside(path)
        if unique:
            n = 2
            while path.exists():
                path = (self.root / f"{base}-{n}{self.suffix}").resolve()
                self._assert_inside(path)
                n += 1
        return path

    def _existing(self, name: str) -> Path | None:
        """Find an existing note by title or filename. Returns None rather
        than raising, so the model gets a sentence it can act on."""
        slug = slugify(name)
        for candidate in (f"{slug}{self.suffix}", name):
            path = (self.root / candidate)
            try:
                path = path.resolve()
            except OSError:
                continue
            try:
                self._assert_inside(path)
            except ValueError:
                continue
            if path.is_file():
                return path
        # Fall back to a prefix match, which catches the "-2" duplicates.
        matches = sorted(p for p in self.root.glob(f"{slug}*{self.suffix}")
                         if p.is_file())
        return matches[0] if matches else None

    def _assert_inside(self, path: Path) -> None:
        """The whole security model, in three lines. Called on every path."""
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"refusing to touch {path}: outside {self.root}")

    # -- schemas -----------------------------------------------------

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [
            {
                "name": "save_note",
                "description": (
                    "Save a plain text note or document to the user's notes "
                    "folder. Use this whenever they ask you to write "
                    "something down, take a note, draft a document, or make "
                    "a list. Give it a short descriptive title — the "
                    "filename is derived from it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short title, a few words.",
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "The full text of the note. Write it as a "
                                "document to be read on screen, not spoken "
                                "aloud: line breaks and simple structure are "
                                "welcome here even though they are not in "
                                "your speech."
                            ),
                        },
                    },
                    "required": ["title", "content"],
                },
            },
            {
                "name": "append_note",
                "description": (
                    "Add text to the end of an existing note. Use this when "
                    "the user wants to add to something rather than start "
                    "over."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                 "description": "Title of the existing note."},
                        "content": {"type": "string",
                                    "description": "Text to append."},
                    },
                    "required": ["name", "content"],
                },
            },
            {
                "name": "list_notes",
                "description": (
                    "List the notes in the user's notes folder, newest first. "
                    "Use this when they ask what notes they have."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "read_note",
                "description": (
                    "Read back the contents of a note. Long notes are "
                    "truncated — summarise rather than reading the whole "
                    "thing aloud."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                 "description": "Title of the note."},
                    },
                    "required": ["name"],
                },
            },
        ]

    # -- execution ---------------------------------------------------

    async def run(self, name: str, args: dict) -> str:
        """Dispatch. Always returns a string; never raises into the caller.

        A tool that raises kills a reply mid-sentence. A tool that returns
        "I could not do that because X" lets the model tell the user
        something useful, which is the entire point of having it.
        """
        try:
            return await asyncio.to_thread(self._run_sync, name, args)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return f"That failed: {type(exc).__name__}: {exc}"

    def _run_sync(self, name: str, args: dict) -> str:
        if not self.enabled:
            return "Note saving is turned off in the configuration."

        if name == "save_note":
            title = (args.get("title") or "note").strip()
            content = args.get("content") or ""
            if not content.strip():
                return "Nothing to save — the note was empty."
            if len(content) > self.max_chars:
                return (f"That note is {len(content)} characters, over the "
                        f"{self.max_chars} limit. Nothing was saved.")
            path = self._path_for(title)
            header = f"{title}\n{_dt.datetime.now():%A %d %B %Y, %I:%M %p}\n\n"
            path.write_text(header + content.rstrip() + "\n", encoding="utf-8")
            log.info("wrote %s (%d chars)", path.name, len(content))
            words = len(content.split())
            return (f"Saved as {path.name} in the notes folder. "
                    f"{words} words.")

        if name == "append_note":
            path = self._existing(args.get("name") or "")
            if not path:
                return (f"There is no note called {args.get('name')!r}. "
                        f"Use save_note to start a new one.")
            content = (args.get("content") or "").rstrip()
            if not content:
                return "Nothing to add."
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n" + content + "\n")
            log.info("appended %d chars to %s", len(content), path.name)
            return f"Added to {path.name}."

        if name == "list_notes":
            files = sorted(
                (p for p in self.root.glob(f"*{self.suffix}") if p.is_file()),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )[:MAX_LIST]
            if not files:
                return "There are no notes yet."
            lines = [f"{p.stem} (saved {_dt.datetime.fromtimestamp(p.stat().st_mtime):%d %b})"
                     for p in files]
            return f"{len(files)} notes, newest first:\n" + "\n".join(lines)

        if name == "read_note":
            path = self._existing(args.get("name") or "")
            if not path:
                return f"There is no note called {args.get('name')!r}."
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_READ_CHARS:
                text = text[:MAX_READ_CHARS] + "\n[... truncated ...]"
            return text

        return f"There is no tool called {name!r}."


class MemoryTools:
    """Durable facts, and search over everything ever said.

    Two different things get called "memory" and they want opposite designs.

    FACTS are small, curated, and injected into every single request. They
    are what makes the assistant know you take your coffee black without
    being asked. Because they are always present, a bad one costs you on
    every exchange forever — so the bar for writing here is high and the
    model is told so explicitly.

    RECALL is search over the raw transcript. It is large, noisy, and only
    worth loading when asked. So it is a tool the model calls on demand
    rather than something that rides along.

    Getting this split wrong is the classic failure: dump the transcript into
    context and you pay for thousands of irrelevant tokens on every request
    and the model still cannot find anything.
    """

    def __init__(self, cfg: dict, store, index=None, clock=None):
        # One module knows what time it is; this receives it. Without that,
        # decay can only be observed by waiting for it.
        self.clock = clock
        # Semantic index, when one is available. Optional on purpose:
        # every path below has to keep working when the embedding
        # model is missing, because it is a 69 MB download that can
        # simply not be there yet.
        self.index = index
        m = cfg.get("memory", {}) or {}
        self.enabled = bool(m.get("enabled", False))
        self.max_facts = int(m.get("max_facts", 200))
        self.store = store

    def _now(self) -> float:
        return self.clock.now() if self.clock else _t.time()

    def _match_memory(self, phrase: str) -> int | None:
        """Find the live memory a phrase refers to. Three passes.

        Semantic first when the index is up, then a literal substring, then
        word overlap. The third pass matters more than it looks: the phrase
        the model supplies for `replaces` is a paraphrase of the old fact, not
        a quotation of it, so "voice is en_GB-semaine" has to reach "Brendan's
        voice on the assistant is en_GB-semaine-medium". A bare LIKE never
        will.
        """
        phrase = (phrase or "").strip()
        if not phrase:
            return None
        rows = self.store.list_memories()
        if not rows:
            return None

        if self.index is not None and self.index.embedder.available:
            hits = self.index.similar(phrase, kind="memory", limit=1,
                                      floor=0.55)
            live = {r["id"] for r in rows}
            if hits and hits[0][0] in live:
                return hits[0][0]

        needle = phrase.lower()
        for r in rows:
            if needle in r["text"].lower():
                return r["id"]

        # Word overlap, ignoring the filler that appears in every sentence.
        stop = {"the", "a", "an", "is", "was", "are", "to", "of", "on", "in",
                "for", "and", "his", "her", "their", "my", "that", "it"}
        want = {w for w in re.findall(r"[a-z0-9]+", needle) if w not in stop}
        if not want:
            return None
        best, best_score = None, 0.0
        for r in rows:
            have = {w for w in re.findall(r"[a-z0-9]+", r["text"].lower())
                    if w not in stop}
            if not have:
                continue
            overlap = len(want & have) / len(want)
            if overlap > best_score:
                best, best_score = r["id"], overlap
        # Over half the meaningful words. Below that it is a guess, and a
        # wrong guess retires a fact the user never meant to touch.
        return best if best_score >= 0.6 else None

    def _reindex(self, fact: str) -> None:
        """Embed a newly written fact straight away.

        The status of a superseded row changed, and `block()` and the index
        must agree about which memories are live. A row the store considers
        retired while the index still offers it is a confidently stated stale
        memory — the exact failure this whole change exists to prevent.
        """
        if self.index is None:
            return
        row = next((r for r in self.store.list_memories()
                    if r["text"] == fact), None)
        if row is not None:
            self.index.index("memory", [(row["id"], fact)])

    def block(self) -> str:
        """The remembered-facts block that goes into every request.

        Stale facts are INCLUDED, not hidden — a fact that has gone quiet is
        usually still true, and dropping it would lose more than it saves. It
        is marked instead, so the model hedges or asks rather than asserting.
        That marker is the entire user-visible payoff of the whole sweep.
        """
        if not self.enabled:
            return ""
        rows = self.store.list_memories()
        if not rows:
            return ""
        now = self._now()
        by_cat: dict[str, list[str]] = {}
        for r in rows:
            line = r["text"]
            if r["status"] == "stale":
                days = int(age_days(r, now))
                line += f"  (unconfirmed for {days} days — check before relying on it)"
            by_cat.setdefault(r["category"], []).append(line)
        out = []
        for cat in sorted(by_cat):
            out.append(f"{cat}:")
            out += [f"  - {t}" for t in by_cat[cat]]
        return "\n".join(out)

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [
            {
                "name": "remember",
                "description": (
                    "Store a durable fact about the user, their household, "
                    "or their preferences, so you still know it in future "
                    "conversations. Use this when they explicitly ask you to "
                    "remember something, and when they mention a stable "
                    "personal fact in passing. Do NOT use it for one-off "
                    "task details, anything that will be untrue next week, "
                    "or anything sensitive like passwords or account "
                    "numbers. Write the fact as one self-contained sentence "
                    "that will still make sense read cold in a year — no "
                    "'he' or 'that' without saying who or what."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "One self-contained sentence.",
                        },
                        "category": {
                            "type": "string",
                            "description": (
                                "Short grouping label, lowercase: people, "
                                "preferences, household, work, health, "
                                "routines, or general."
                            ),
                        },
                        "volatility": {
                            "type": "string",
                            "enum": ["stable", "slow", "fast"],
                            "description": (
                                "How fast this kind of fact goes out of date. "
                                "'stable' for things that essentially never "
                                "change — a birthday, a device ID, a "
                                "preference. 'slow' for jobs, addresses, "
                                "equipment, policies. 'fast' for a current "
                                "project or anything tied to the next few "
                                "weeks. Default slow."
                            ),
                        },
                        "replaces": {
                            "type": "string",
                            "description": (
                                "If this CORRECTS something already "
                                "remembered, give the old fact in a few words "
                                "so it can be retired. The old one is kept "
                                "with its dates closed, never deleted, so "
                                "'what did I think before' still answers. "
                                "Leave this out when the new fact simply sits "
                                "alongside the old — two things can both be "
                                "true."
                            ),
                        },
                    },
                    "required": ["fact"],
                },
            },
            {
                "name": "forget",
                "description": (
                    "Remove remembered facts matching some text. Use when "
                    "the user says to forget something, or when they "
                    "correct a fact you have stored — in that case forget "
                    "the old one and remember the new one."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "about": {
                            "type": "string",
                            "description": "Text to match against stored facts.",
                        },
                    },
                    "required": ["about"],
                },
            },
            {
                "name": "confirm_fact",
                "description": (
                    "Mark a remembered fact as still true. Use when the user "
                    "confirms something you flagged as unconfirmed, or "
                    "restates a fact you already hold. This resets its "
                    "freshness — it is how a long-standing fact stays "
                    "trusted instead of slowly going grey."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "about": {"type": "string",
                                  "description": "A few words identifying it."},
                    },
                    "required": ["about"],
                },
            },
            {
                "name": "stale_facts",
                "description": (
                    "The remembered facts that have gone unconfirmed long "
                    "enough to doubt. Use when the user asks what needs "
                    "checking, what might be out of date, or to review what "
                    "you know. Read back at most three and offer the rest."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "fact_history",
                "description": (
                    "What was previously believed about something, including "
                    "facts that have since been corrected or retired. Use for "
                    "'what did I have that set to before', 'what did you used "
                    "to think', 'when did that change'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"about": {"type": "string"}},
                    "required": ["about"],
                },
            },
            {
                "name": "recall",
                "description": (
                    "Search everything that has ever been said in past "
                    "conversations. Use when the user refers to something "
                    "discussed before that is not in the facts you already "
                    "know — 'what did I ask you about the insurance "
                    "letter', 'what was that restaurant called'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords to search for.",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def run_sync(self, name: str, args: dict) -> str | None:
        if not self.enabled:
            return None

        if name == "remember":
            fact = (args.get("fact") or "").strip()
            if not fact:
                return "Nothing to remember."
            if len(fact) > 400:
                return "That is too long for a remembered fact."
            n = self.store.count_memories()
            if n >= self.max_facts:
                return (f"Memory is full at {n} facts. Ask the user what to "
                        f"forget before storing more.")
            # A restatement is not a new fact. UNIQUE on the text column only
            # catches character-identical entries, so "his daughter plays
            # soccer" and "Brendan's daughter is on a soccer team" both land,
            # and both then ride along in every future prompt forever.
            now = self._now()
            # A near-identical fact is EVIDENCE, not a duplicate to reject.
            # Restating something you already know means it is still true, so
            # the honest response is to refresh its clock.
            if self.index is not None and not args.get("replaces"):
                dup = self.index.duplicate_of(fact)
                if dup is not None:
                    row = next((r for r in self.store.list_memories()
                                if r["id"] == dup), None)
                    if row is not None:
                        self.store.confirm_memory(dup, now)
                        log.info("memory confirmed #%d: %s", dup, fact)
                        return f"Already knew that. Confirmed: {row['text']}"

            # An explicit correction retires the old fact rather than
            # overwriting it — its window closes, and it stays answerable.
            replaces_id = None
            if args.get("replaces"):
                replaces_id = self._match_memory(str(args["replaces"]))
                if replaces_id is None:
                    # Refusing is the right failure. Storing it anyway leaves
                    # two contradictory facts live, both going into every
                    # future prompt — which is worse than the staleness this
                    # whole design is here to prevent.
                    return (f"I could not find a stored fact matching "
                            f"{args['replaces']!r}, so I have not saved this "
                            f"one — it would have contradicted whatever is "
                            f"actually there. Tell me which to replace.")

            what = self.store.add_memory(
                fact, (args.get("category") or "general").strip().lower(),
                volatility=(args.get("volatility") or "slow"),
                now=now, replaces_id=replaces_id)
            log.info("memory %s: %s", what, fact)
            if what == "replaced":
                old_row = self.store.conn.execute(
                    "SELECT text FROM memories WHERE superseded_by = "
                    "(SELECT MAX(id) FROM memories)").fetchone()
                self._reindex(fact)
                return ("Updated. I had "
                        + (f"{old_row['text']!r}" if old_row else "the old one")
                        + ", and I have kept it dated rather than deleted.")
            if self.index is not None:
                row = next((r for r in self.store.list_memories()
                            if r["text"] == fact), None)
                if row is not None:
                    # Indexed immediately rather than by the backfill loop.
                    # Memories are few and the very next thing said is often
                    # a correction of this one.
                    self.index.index("memory", [(row["id"], fact)])
            return f"{what.capitalize()}. You will know this next time."

        if name == "forget":
            about = (args.get("about") or "").strip()
            if not about:
                return "Nothing specified to forget."
            rows = self.store.find_memories(about)
            if not rows:
                return f"Nothing remembered matching {about!r}."
            now = self._now()
            # "Forget that" means "stop telling me that", not "destroy the
            # record it was ever so". The window closes; the row stays.
            gone = [r["text"] for r in rows
                    if self.store.expire_memory(r["id"], now, "user asked")]
            log.info("retired %d: %s", len(gone), "; ".join(gone))
            return ("Retired: " + "; ".join(gone)) if gone \
                else "Nothing matched."

        if name == "confirm_fact":
            about = (args.get("about") or "").strip()
            rows = self.store.find_memories(about, limit=1)
            if not rows and self.index is not None:
                hits = self.index.similar(about, kind="memory", limit=1,
                                          floor=0.5)
                rows = [r for r in self.store.list_memories()
                        if hits and r["id"] == hits[0][0]]
            if not rows:
                return f"Nothing remembered matching {about!r}."
            self.store.confirm_memory(rows[0]["id"], self._now())
            return f"Confirmed: {rows[0]['text']}"

        if name == "stale_facts":
            now = self._now()
            rows = [r for r in self.store.list_memories()
                    if r["status"] == "stale"]
            if not rows:
                return "Nothing needs checking. Everything is current."
            parts = []
            for r in rows[:3]:
                parts.append(f"{r['text']} — {int(age_days(r, now))} days")
            more = len(rows) - len(parts)
            return (f"{len(rows)} to check: " + "; ".join(parts)
                    + (f". And {more} more." if more > 0 else "."))

        if name == "fact_history":
            about = (args.get("about") or "").strip()
            rows = self.store.memory_history(about)
            if not rows:
                return f"No history for {about!r}."
            out = []
            for r in rows:
                when = _dt.datetime.fromtimestamp(
                    r["valid_from"]).strftime("%d %b")
                mark = {"active": "now", "stale": "now, unconfirmed",
                        "superseded": "until "
                        + (_dt.datetime.fromtimestamp(r["valid_to"]).strftime("%d %b")
                           if r["valid_to"] else "?"),
                        "expired": "ended"}.get(r["status"], r["status"])
                out.append(f"from {when}, {mark}: {r['text']}")
            return " | ".join(out)

        if name == "recall":
            query = (args.get("query") or "").strip()
            if not query:
                return "No search terms given."
            # Keyword first. FTS5 is the only one of the two that reliably
            # finds "INC-4412", a surname, or a part number — every ticket
            # number embeds to roughly the same place.
            try:
                kw_ids = self.store.search_ids(query, limit=20)
            except Exception:  # noqa: BLE001 — FTS syntax, e.g. a bare quote
                kw_ids = []
            rows = []
            if self.index is not None and self.index.embedder.available:
                ids = self.index.hybrid(query, kw_ids, limit=8)
                rows = self.store.messages_by_ids(ids)
            if not rows:
                # No index, or it returned nothing: the old path, unchanged.
                try:
                    rows = self.store.search(query, limit=8)
                except Exception as exc:  # noqa: BLE001
                    return f"Could not search for that: {exc}"
            if not rows:
                return f"Nothing found in past conversations about {query!r}."
            out = []
            for r in rows:
                when = _dt.datetime.fromtimestamp(r["ts"]).strftime("%d %b")
                who = "you" if r["role"] == "user" else "I"
                snippet = " ".join(r["content"].split())[:300]
                out.append(f"[{when}] {who} said: {snippet}")
            return "\n".join(out)

        return None



class TimerTools:
    """Timers, reminders, and the shopping list.

    Note what these tools do NOT do: parse dates. The model receives the
    current time in every request, so it converts "in twenty minutes" or
    "tomorrow at seven" into an absolute timestamp before calling. That
    keeps timezone and daylight-saving arithmetic out of this file entirely,
    which is the correct place for it to be absent.
    """

    def __init__(self, cfg: dict, store):
        r = cfg.get("reminders", {}) or {}
        self.enabled = bool(r.get("enabled", False))
        self.max_pending = int(r.get("max_pending", 50))
        l = cfg.get("lists", {}) or {}
        self.lists_enabled = bool(l.get("enabled", False))
        self.default_list = str(l.get("default", "shopping"))
        self.allow_new_lists = bool(l.get("allow_new", True))

        # Named lists, each with the words a person might actually say for
        # it. "Put it on the honey-do list" has to reach the same place as
        # "add it to the house list" — speech does not come with an enum.
        self.known: dict[str, dict] = {}
        for name, spec in (l.get("known") or {}).items():
            spec = spec or {}
            self.known[str(name).lower()] = {
                "say": spec.get("say") or f"{name} list",
                "aliases": [str(a).lower() for a in (spec.get("aliases") or [])],
            }
        self.store = store

    # -- list naming -------------------------------------------------

    def resolve_list(self, raw: str | None) -> str:
        """Map whatever was said to a canonical list name.

        Matching is exact-then-alias-then-word, in that order, because the
        cost of guessing wrong is silent: the item goes onto a list nobody
        looks at, and it is gone as surely as if it had never been added.
        """
        if not raw or not str(raw).strip():
            return self.default_list
        text = str(raw).lower()
        # Possessives FIRST. Stripping punctuation blindly turns
        # "children's" into "childrens", which then matches no alias and
        # quietly creates a brand new list called "childrens" — exactly the
        # silent misfile this whole function exists to prevent.
        text = re.sub(r"[\u2019']s\b", "", text)
        text = " ".join(text.translate(
            str.maketrans("", "", ".,!?;:\"'\u2019")).split())
        # Strip the words people put around a list name.
        for filler in ("the ", " list", " to do", " to-do", " todo"):
            text = text.replace(filler, " ")
        text = " ".join(text.split())

        if text in self.known:
            return text
        for name, spec in self.known.items():
            if text in spec["aliases"]:
                return name
        # Whole-word containment, longest alias first so "hardware store"
        # does not get claimed by "store".
        words = set(text.split())
        best, best_len = None, 0
        for name, spec in self.known.items():
            for alias in [name] + spec["aliases"]:
                parts = alias.split()
                if set(parts) <= words and len(alias) > best_len:
                    best, best_len = name, len(alias)
        if best:
            return best
        if self.allow_new_lists:
            return slugify(text)
        return self.default_list

    def list_menu(self) -> str:
        """The known lists, for the tool descriptions."""
        if not self.known:
            return "shopping"
        return ", ".join(f"{n} ({s['say']})" for n, s in self.known.items())

    def spoken_name(self, name: str) -> str:
        spec = self.known.get(name)
        return spec["say"] if spec else f"{name} list"

    def schemas(self) -> list[dict]:
        out = []
        if self.enabled:
            out += [
                {
                    "name": "set_reminder",
                    "description": (
                        "Set a timer or a reminder. You must convert the "
                        "user's phrasing into an absolute time yourself — "
                        "the current date and time are given to you in every "
                        "message. Use kind 'timer' for short cooking-style "
                        "countdowns and 'reminder' for anything tied to a "
                        "clock time or a future day."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "at": {
                                "type": "string",
                                "description": (
                                    "When it should fire, as ISO 8601 local "
                                    "time, e.g. 2026-08-22T19:30:00. No "
                                    "timezone suffix — local time is assumed."
                                ),
                            },
                            "text": {
                                "type": "string",
                                "description": (
                                    "What to say. For a timer this is the "
                                    "label, like 'pasta'. For a reminder it "
                                    "is the whole message, phrased for "
                                    "speaking aloud: 'take the bins out'."
                                ),
                            },
                            "kind": {
                                "type": "string",
                                "enum": ["timer", "reminder"],
                            },
                        },
                        "required": ["at", "text"],
                    },
                },
                {
                    "name": "list_reminders",
                    "description": "List timers and reminders not yet fired.",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "cancel_reminder",
                    "description": (
                        "Cancel pending timers or reminders. Give `about` to "
                        "match by text, or omit it to cancel everything "
                        "pending — only do that if the user clearly meant all."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "about": {"type": "string",
                                      "description": "Text to match."},
                        },
                    },
                },
            ]
        if self.lists_enabled:
            out += [
                {
                    "name": "list_add",
                    "description": (
                        "Add items to one of the user's lists. Pass several "
                        "at once when several are named. Duplicates are "
                        "ignored. Available lists: " + self.list_menu() +
                        ". Choose the one that fits what was said; if it is "
                        "genuinely unclear which list they mean, ask rather "
                        "than guessing — an item on the wrong list is lost."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "items": {"type": "array",
                                      "items": {"type": "string"}},
                            "list": {"type": "string",
                                     "description": (
                                         "Which list. One of: "
                                         + self.list_menu()
                                         + ". Omit for the default.")},
                        },
                        "required": ["items"],
                    },
                },
                {
                    "name": "list_remove",
                    "description": "Remove items from a list. Matches loosely.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "items": {"type": "array",
                                      "items": {"type": "string"}},
                            "list": {"type": "string"},
                        },
                        "required": ["items"],
                    },
                },
                {
                    "name": "show_board",
                    "description": (
                        "Put the lists and reminders up on the screen. Use "
                        "when the user asks to see, show, or pull up their "
                        "lists rather than hear them. Still say something "
                        "brief out loud — 'here they are' — because they may "
                        "not be looking at the screen."
                    ),
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "list_all",
                    "description": (
                        "Show every list and how many items each holds. Use "
                        "when the user asks what lists there are, or what is "
                        "outstanding generally."
                    ),
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "list_complete",
                    "description": (
                        "Tick items off as done. Use whenever the user says "
                        "they finished, did, completed, or handled something. "
                        "Items are kept, not deleted — they show as done on "
                        "the board and can be reopened."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "items": {"type": "array",
                                      "items": {"type": "string"}},
                            "list": {"type": "string"},
                        },
                        "required": ["items"],
                    },
                },
                {
                    "name": "list_reopen",
                    "description": (
                        "Put a completed item back to open — for when "
                        "something was ticked off by mistake or needs doing "
                        "again."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "items": {"type": "array",
                                      "items": {"type": "string"}},
                            "list": {"type": "string"},
                        },
                        "required": ["items"],
                    },
                },
                {
                    "name": "list_clear_done",
                    "description": (
                        "Permanently remove completed items. Omit `list` to "
                        "clear finished items everywhere. This one cannot be "
                        "undone, so confirm first."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"list": {"type": "string"}},
                    },
                },
                {
                    "name": "list_read",
                    "description": (
                        "Read a list back. Returns the items; say them "
                        "naturally rather than reciting a numbered list."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"list": {"type": "string"}},
                    },
                },
                {
                    "name": "list_clear",
                    "description": (
                        "Empty a list completely. Confirm with the user "
                        "before calling this — it cannot be undone."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"list": {"type": "string"}},
                    },
                },
            ]
        return out

    def run_sync(self, name: str, args: dict) -> str | None:
        if name in ("set_reminder", "list_reminders", "cancel_reminder"):
            if not self.enabled:
                return "Reminders are turned off in the configuration."
            return self._reminders(name, args)
        if name in ("list_add", "list_remove", "list_read", "list_clear",
                    "list_all", "show_board", "list_complete", "list_reopen",
                    "list_clear_done"):
            if not self.lists_enabled:
                return "Lists are turned off in the configuration."
            return self._lists(name, args)
        return None

    # -- reminders ---------------------------------------------------

    def _reminders(self, name: str, args: dict) -> str:
        from .scheduler import describe_when

        if name == "set_reminder":
            raw = (args.get("at") or "").strip()
            text = (args.get("text") or "").strip()
            kind = args.get("kind") or "reminder"
            if not text:
                return "I need to know what to remind you about."
            try:
                # Tolerate a trailing Z or offset by stripping it: the
                # appliance and the user share one timezone, and a model
                # guessing at UTC offsets is a reliable source of alarms
                # that fire seven hours late.
                cleaned = raw.replace("Z", "").split("+")[0].strip()
                due = _dt.datetime.fromisoformat(cleaned).timestamp()
            except ValueError:
                return (f"I could not read {raw!r} as a time. Use ISO 8601 "
                        f"local time like 2026-08-22T19:30:00.")

            if due <= _t.time():
                return "That time has already passed. Did you mean tomorrow?"
            if due > _t.time() + 366 * 86400:
                return "That is more than a year away — I think something is wrong."
            pending = len(self.store.pending_reminders())
            if pending >= self.max_pending:
                return (f"There are already {pending} pending. Cancel some "
                        f"before setting more.")

            self.store.add_reminder(due, text, kind)
            log.info("%s set for %s: %s", kind,
                     _dt.datetime.fromtimestamp(due).isoformat(timespec="minutes"),
                     text)
            return f"Set, {describe_when(due)}."

        if name == "list_reminders":
            rows = self.store.pending_reminders()
            if not rows:
                return "Nothing pending."
            return "\n".join(
                f"{r['kind']}: {r['text']} — {describe_when(r['due'])}"
                for r in rows)

        if name == "cancel_reminder":
            about = (args.get("about") or "").strip() or None
            gone = self.store.cancel_reminders(about)
            if not gone:
                return ("Nothing pending matching that."
                        if about else "Nothing was pending.")
            return f"Cancelled {len(gone)}: " + "; ".join(gone)

        return "Unknown reminder operation."

    # -- lists -------------------------------------------------------

    def _lists(self, name: str, args: dict) -> str:
        if name == "show_board":
            counts = self.store.list_counts()
            total = sum(c for _, c in counts)
            pend = len(self.store.pending_reminders())
            bits = []
            if total:
                bits.append(f"{total} item{'s' if total != 1 else ''} across "
                            f"{len(counts)} list{'s' if len(counts) != 1 else ''}")
            if pend:
                bits.append(f"{pend} reminder{'s' if pend != 1 else ''}")
            return ("On screen now: " + " and ".join(bits) + ".") if bits \
                else "Everything is empty, but it is on screen."

        if name == "list_all":
            counts = self.store.list_counts()
            if not counts:
                return "All lists are empty."
            parts = [f"{self.spoken_name(n)}: {c}" for n, c in counts]
            return "; ".join(parts)

        which = self.resolve_list(args.get("list"))
        items = args.get("items") or []
        if not isinstance(items, list):
            items = [str(items)]

        if name == "list_add":
            added = self.store.list_add([str(i) for i in items], which)
            if not added:
                return f"Already on the {self.spoken_name(which)} — nothing new added."
            total = len(self.store.list_read(which))
            return (f"Added {', '.join(added)} to the "
                    f"{self.spoken_name(which)}. {total} "
                    f"item{'s' if total != 1 else ''} now.")

        if name == "list_complete":
            done = self.store.list_complete([str(i) for i in items], which, True)
            if not done:
                return (f"Could not find that open on the "
                        f"{self.spoken_name(which)}.")
            left = len(self.store.list_read(which))
            return (f"Ticked off {', '.join(done)}. "
                    f"{left} left on the {self.spoken_name(which)}."
                    if left else
                    f"Ticked off {', '.join(done)}. That clears the "
                    f"{self.spoken_name(which)}.")

        if name == "list_reopen":
            back = self.store.list_complete([str(i) for i in items], which, False)
            if not back:
                return f"Nothing completed matching that."
            return f"Put {', '.join(back)} back."

        if name == "list_clear_done":
            target = self.resolve_list(args.get("list")) if args.get("list") else None
            n = self.store.list_clear_done(target)
            where = f"the {self.spoken_name(target)}" if target else "all lists"
            return (f"Removed {n} completed item{'s' if n != 1 else ''} from "
                    f"{where}.") if n else f"Nothing completed to clear from {where}."

        if name == "list_remove":
            gone = self.store.list_remove([str(i) for i in items], which)
            if not gone:
                return f"Could not find that on the {self.spoken_name(which)}."
            return f"Removed {', '.join(gone)}."

        if name == "list_read":
            rows = self.store.list_read(which)
            all_rows = self.store.list_read(which, include_done=True)
            done_n = len(all_rows) - len(rows)
            if not rows:
                return (f"Nothing open on the {self.spoken_name(which)}"
                        + (f", though {done_n} done." if done_n else "."))
            tail = f" And {done_n} done." if done_n else ""
            return (f"{len(rows)} on the {self.spoken_name(which)}: "
                    + ", ".join(r["text"] for r in rows) + "." + tail)

        if name == "list_clear":
            n = self.store.list_clear(which)
            return (f"Cleared {n} item{'s' if n != 1 else ''} from the "
                    f"{self.spoken_name(which)}.") if n else \
                   f"The {self.spoken_name(which)} was already empty."

        return "Unknown list operation."


from .desktop import DesktopTools
from .casework import CaseTools
from .printer import PrinterTools
from .embed import Embedder, SemanticIndex
from .workout import WorkoutTools
from .media import MediaTools

# Where the appliance lives on disk. Used for the mpv IPC socket, which has to
# sit somewhere writable and predictable — the data directory, alongside the
# database, rather than /tmp where a reboot or a cleaner can take it.
ROOT = Path(__file__).resolve().parent.parent
from .clock import days_between, make_clock
from .sweep import Sweep, age_days, score


class Toolbox:
    """Everything that runs locally, behind one schemas()/run() pair."""

    def __init__(self, cfg: dict, store):
        self.docs = DocumentTools(cfg)
        self.embedder = Embedder(cfg)
        self.index = SemanticIndex(store, self.embedder)
        # One clock, passed down. `memory.clock_at` in config freezes it,
        # which is how a year of decay gets demonstrated in a second.
        self.clock = make_clock(
            (cfg.get("memory", {}) or {}).get("clock_at") or None)
        self.mem = MemoryTools(cfg, store, index=self.index, clock=self.clock)
        self.sweep = Sweep(store, cfg, clock=self.clock)
        self.timers = TimerTools(cfg, store)
        self.desktop = DesktopTools(cfg)
        self.cases = CaseTools(cfg, store)
        # The printer borrows the list-name resolver and the notes directory
        # rather than owning its own. "Print the honey-do list" has to land on
        # the same board "add milk to the honey-do list" does — two resolvers
        # would drift, and the failure mode is a printed list that is real but
        # not the one that was asked for.
        self.printer = PrinterTools(cfg, store, timers=self.timers,
                                    doctools=self.docs)
        # Workouts borrow the case matcher: the same loose spoken-name
        # matching that ticks off 'I purged it' ticks off 'bench done'.
        self.workouts = WorkoutTools(cfg, store, cases=self.cases)
        # Media owns a child process, which nothing else here does. It needs
        # ROOT for the default IPC socket path, and app.py hands it the
        # scheduler tick so a finished video stops being "playing".
        self.media = MediaTools(cfg, ROOT)

    def schemas(self) -> list[dict]:
        return (self.docs.schemas() + self.mem.schemas()
                + self.timers.schemas() + self.desktop.schemas()
                + self.cases.schemas() + self.printer.schemas()
                + self.workouts.schemas() + self.media.schemas())

    def memory_block(self) -> str:
        return self.mem.block()

    async def run(self, name: str, args: dict) -> str:
        try:
            result = await asyncio.to_thread(self._run_sync, name, args)
            return result
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return f"That failed: {type(exc).__name__}: {exc}"

    def _run_sync(self, name: str, args: dict) -> str:
        for handler in (self.media.run_sync,
                        self.workouts.run_sync, self.printer.run_sync,
                        self.cases.run_sync,
                        self.desktop.run_sync, self.timers.run_sync,
                        self.mem.run_sync):
            out = handler(name, args)
            if out is not None:
                return out
        return self.docs._run_sync(name, args)


def build_tools(cfg: dict, store=None) -> Toolbox:
    return Toolbox(cfg, store)
