"""SQLite persistence: conversation history, usage log, wake events.

One file, no server, survives power cuts. FTS5 over message content so the
assistant can answer "what did I ask you about the insurance letter?".
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL,
    role      TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content   TEXT    NOT NULL,
    tier      TEXT,
    model     TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(content, content='messages', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;

-- Durable facts. This is the difference between an assistant that knows you
-- and one that is meeting you for the first time, every time.
--
-- Deliberately separate from `messages`. History is a transcript — long,
-- noisy, and mostly irrelevant tomorrow. This table is small, curated, and
-- goes into EVERY request, so a bad entry here costs you on every single
-- exchange until it is removed. That asymmetry is why the model is told to
-- be conservative about what lands here.
--
-- UNIQUE on text so saving the same fact twice updates rather than
-- duplicates. Nothing is more tiresome than an assistant that remembers
-- your coffee order four times.
CREATE TABLE IF NOT EXISTS memories (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created  REAL NOT NULL,
    updated  REAL NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    text     TEXT NOT NULL UNIQUE
);

-- Timers and reminders. Persisted deliberately: a kitchen timer that
-- forgets itself when the service restarts is not a kitchen timer, and
-- restarts happen (updates, crashes, the USB wedge).
--
-- `due` is an absolute unix timestamp. The MODEL does the arithmetic —
-- it is told the current time on every request, so "in twenty minutes"
-- and "tomorrow at seven" both resolve upstream and no date parser is
-- needed down here. One less dependency and one less thing to be subtly
-- wrong about daylight saving.
CREATE TABLE IF NOT EXISTS reminders (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created  REAL NOT NULL,
    due      REAL NOT NULL,
    text     TEXT NOT NULL,
    kind     TEXT NOT NULL DEFAULT 'reminder',   -- timer | reminder
    fired    INTEGER NOT NULL DEFAULT 0,
    cancelled INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_reminders_due
    ON reminders(due) WHERE fired = 0 AND cancelled = 0;

-- Named lists. Shopping is the one everybody uses; the table is general
-- because "packing list" and "hardware store" cost nothing extra.
CREATE TABLE IF NOT EXISTS list_items (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    list     TEXT NOT NULL DEFAULT 'shopping',
    text     TEXT NOT NULL,
    added    REAL NOT NULL,
    -- Checking a thing off is what separates a task board from a list. The
    -- item stays: "done" is a state, not a deletion, so the board can show
    -- what you got through today rather than only what is left.
    done     INTEGER NOT NULL DEFAULT 0,
    done_at  REAL
);

CREATE INDEX IF NOT EXISTS idx_list_items ON list_items(list, id);

-- Casework: an investigation with a clock on it.
--
-- This is deliberately NOT another named list. A list is a bag of strings you
-- add to and tick off in any order. A case is a template instantiated at a
-- moment in time, with an SLA deadline, an ordered checklist that came from a
-- playbook, and findings attached to individual steps. Those differences all
-- point the same way: a separate table.
--
-- The steps are COPIED from the playbook at open time rather than referenced.
-- That costs a few rows and buys the thing that actually matters: editing the
-- playbook tomorrow does not silently rewrite what you did today. A closed
-- case is a record, and a record that changes underneath you is not one.
CREATE TABLE IF NOT EXISTS cases (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT    NOT NULL DEFAULT 'phishing',
    title     TEXT    NOT NULL,
    ref       TEXT,                        -- ticket number, if there is one
    severity  TEXT    NOT NULL DEFAULT 'standard',
    opened    REAL    NOT NULL,
    sla_due   REAL,                        -- absolute deadline, or NULL
    closed    REAL,
    outcome   TEXT,
    notes     TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cases_open
    ON cases(opened) WHERE closed IS NULL;

CREATE TABLE IF NOT EXISTS case_steps (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id  INTEGER NOT NULL,
    side     TEXT    NOT NULL DEFAULT 'investigation',  -- investigation | admin
    phase    TEXT,                          -- grouping header on screen
    seq      INTEGER NOT NULL,
    key      TEXT    NOT NULL,
    text     TEXT    NOT NULL,
    aliases  TEXT    NOT NULL DEFAULT '',   -- newline-joined spoken variants
    done     INTEGER NOT NULL DEFAULT 0,
    done_at  REAL,
    -- What he actually found. "sender domain registered four days ago" hangs
    -- off the step it belongs to, which is what makes the write-up at the end
    -- a transcription job rather than a memory test.
    finding  TEXT,
    -- Seconds to rest after this step. Only workouts use it; an
    -- investigation step has no such thing, and NULL says so.
    rest_seconds INTEGER
);

CREATE INDEX IF NOT EXISTS idx_case_steps
    ON case_steps(case_id, side, seq);

-- One row per SLA warning already spoken. Without this the scheduler
-- re-announces "halfway through your SLA" every single second, which is a
-- uniquely bad way to spend an afternoon.
CREATE TABLE IF NOT EXISTS case_alerts (
    case_id INTEGER NOT NULL,
    marker  TEXT    NOT NULL,
    fired   REAL    NOT NULL,
    PRIMARY KEY (case_id, marker)
);

-- Vectors for semantic recall. Same file as everything else, deliberately:
-- one thing to back up, one thing to corrupt, and no second server to be
-- running or not running.
--
-- `model` is part of the row, not a global setting, because vectors from two
-- different models are not comparable and mixing them produces confident
-- nonsense. Changing the configured model orphans the old rows rather than
-- silently ranking against them.
CREATE TABLE IF NOT EXISTS embeddings (
    kind    TEXT    NOT NULL,          -- message | memory
    ref_id  INTEGER NOT NULL,
    model   TEXT    NOT NULL,
    vec     BLOB    NOT NULL,
    made    REAL    NOT NULL,
    PRIMARY KEY (kind, ref_id, model)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_lookup
    ON embeddings(kind, model);

CREATE TABLE IF NOT EXISTS usage (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL NOT NULL,
    model          TEXT NOT NULL,
    tier           TEXT,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write    INTEGER NOT NULL DEFAULT 0,
    cache_read     INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);

-- Every wake-word trigger, including ones you did not mean. This table is
-- how you tune the threshold honestly instead of guessing.
CREATE TABLE IF NOT EXISTS wake_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    score      REAL NOT NULL,
    accepted   INTEGER NOT NULL DEFAULT 1,
    transcript TEXT
);
"""


def fts_query(text: str) -> str | None:
    """Turn what a person said into something FTS5 will accept.

    FTS5 does not take a search string — it takes a QUERY EXPRESSION, with its
    own operators and keywords. So the raw text of a question is a small
    minefield:

        "INC-4412"   -> no such column: 4412     (the hyphen)
        "don't"      -> syntax error near "'"    (every contraction, ever)
        "a:b"        -> no such column: a        (the colon)
        "C++"        -> syntax error near "+"
        "AND"        -> syntax error             (a bare keyword)
        ""           -> syntax error

    Speech is transcribed, so apostrophes arrive constantly and the failure is
    a raw SQLite error read out loud. The fix is to stop passing user text as
    syntax at all: pull out the alphanumeric tokens and quote each one, which
    makes every character above a literal and every keyword an ordinary word.

    Tokens stay space-separated, which FTS5 reads as AND. That is what you
    want here: "INC 4412" should mean the message with both, not either. A
    long natural question ANDs down to nothing, which is correct — keyword
    search cannot answer those, and the semantic half is what does.

    Returns None when there is nothing searchable, so callers skip the query
    rather than handing FTS5 an empty string.
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", text or "")
    if not tokens:
        return None
    # Trim absurd input rather than building a 500-term AND that matches
    # nothing and takes a while to decide that.
    return " ".join(f'"{t}"' for t in tokens[:24])


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns to tables that already exist.

        CREATE TABLE IF NOT EXISTS silently does nothing when the table is
        already there, so a new column in SCHEMA never reaches a live
        database. Checked and added explicitly — this runs against a file
        holding real household data and must never drop anything.
        """
        cols = {r["name"] for r in
                self.conn.execute("PRAGMA table_info(list_items)").fetchall()}
        for name, ddl in (("done", "INTEGER NOT NULL DEFAULT 0"),
                          ("done_at", "REAL")):
            if name not in cols:
                self.conn.execute(
                    f"ALTER TABLE list_items ADD COLUMN {name} {ddl}")
                log_msg = f"migrated: list_items.{name} added"
                print(log_msg)

        # case_steps.rest_seconds arrived with workouts, after the table
        # already existed on a live box. CREATE TABLE IF NOT EXISTS will not
        # add it, so it is added explicitly here, same as above.
        step_cols = {r["name"] for r in
                     self.conn.execute("PRAGMA table_info(case_steps)").fetchall()}
        if step_cols and "rest_seconds" not in step_cols:
            self.conn.execute(
                "ALTER TABLE case_steps ADD COLUMN rest_seconds INTEGER")
            print("migrated: case_steps.rest_seconds added")

    # -- messages ---------------------------------------------------

    def add_message(self, role: str, content: str,
                    tier: str | None = None, model: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages (ts, role, content, tier, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), role, content, tier, model),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_turns(self, n_turns: int) -> list[dict[str, str]]:
        """Return the last n_turns exchanges as provider-shaped messages.

        A "turn" is a user message plus whatever followed it, so we pull
        2*n rows and trim to start on a user message — a history that
        begins with an assistant reply is rejected by most providers.
        """
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (n_turns * 2,),
        ).fetchall()
        msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs

    def search(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        match = fts_query(query)
        if match is None:
            return []
        return self.conn.execute(
            "SELECT m.ts, m.role, m.content FROM messages_fts f "
            "JOIN messages m ON m.id = f.rowid "
            "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()

    def turns_since(self, since_ts: float, cap: int) -> list[dict[str, str]]:
        """Recent turns bounded by TIME as well as count.

        `recent_turns` alone has an odd failure: come back after three days
        and the assistant opens mid-conversation, replying to something you
        said on Tuesday as though you had just said it. Bounding by time too
        means a long gap starts fresh, which is what a person would do.
        """
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE ts >= ? "
            "ORDER BY id DESC LIMIT ?", (since_ts, cap * 2),
        ).fetchall()
        msgs = [{"role": r["role"], "content": r["content"]}
                for r in reversed(rows)]
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs

    # -- memories ---------------------------------------------------

    def add_memory(self, text: str, category: str = "general") -> str:
        """Insert or update. Returns 'saved' or 'updated'."""
        now, text = time.time(), text.strip()
        # rowcount is 1 for both the insert and the update path, so it cannot
        # tell us which happened. Ask first — it is one indexed lookup, and
        # the answer is what the user hears.
        existed = self.conn.execute(
            "SELECT 1 FROM memories WHERE text = ?", (text,)).fetchone()
        self.conn.execute(
            "INSERT INTO memories (created, updated, category, text) "
            "VALUES (?,?,?,?) ON CONFLICT(text) DO UPDATE SET "
            "updated=excluded.updated, category=excluded.category",
            (now, now, category, text),
        )
        self.conn.commit()
        return "updated" if existed else "saved"

    def list_memories(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, category, text, updated FROM memories "
            "ORDER BY category, id"
        ).fetchall()

    def count_memories(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) AS n FROM memories").fetchone()["n"])

    def find_memories(self, needle: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, category, text FROM memories WHERE text LIKE ? "
            "ORDER BY id LIMIT ?", (f"%{needle.strip()}%", limit),
        ).fetchall()

    def delete_memory(self, mem_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- reminders --------------------------------------------------

    def add_reminder(self, due: float, text: str, kind: str = "reminder") -> int:
        cur = self.conn.execute(
            "INSERT INTO reminders (created, due, text, kind) VALUES (?,?,?,?)",
            (time.time(), due, text.strip(), kind),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def due_reminders(self, now: float) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, due, text, kind FROM reminders "
            "WHERE fired = 0 AND cancelled = 0 AND due <= ? ORDER BY due",
            (now,),
        ).fetchall()

    def pending_reminders(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, due, text, kind FROM reminders "
            "WHERE fired = 0 AND cancelled = 0 ORDER BY due"
        ).fetchall()

    def mark_fired(self, rid: int) -> None:
        self.conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (rid,))
        self.conn.commit()

    def cancel_reminders(self, needle: str | None = None) -> list[str]:
        """Cancel by substring, or everything pending if needle is None."""
        rows = self.pending_reminders()
        if needle:
            n = needle.strip().lower()
            rows = [r for r in rows if n in r["text"].lower()]
        for r in rows:
            self.conn.execute(
                "UPDATE reminders SET cancelled = 1 WHERE id = ?", (r["id"],))
        self.conn.commit()
        return [r["text"] for r in rows]

    # -- lists ------------------------------------------------------

    def list_add(self, items: list[str], list_name: str = "shopping") -> list[str]:
        added = []
        existing = {r["text"].lower() for r in self.list_read(list_name)}
        # Only OPEN items count as duplicates — re-adding something
        # you finished last month is a new task, not a mistake.
        for raw in items:
            text = " ".join(raw.split()).strip()
            if not text or text.lower() in existing:
                continue
            self.conn.execute(
                "INSERT INTO list_items (list, text, added) VALUES (?,?,?)",
                (list_name, text, time.time()),
            )
            existing.add(text.lower())
            added.append(text)
        self.conn.commit()
        return added

    def list_read(self, list_name: str = "shopping",
                  include_done: bool = False) -> list[sqlite3.Row]:
        """Open items by default. Done ones are still there — ask for them."""
        sql = ("SELECT id, text, done, done_at FROM list_items WHERE list = ?"
               + ("" if include_done else " AND done = 0")
               + " ORDER BY done, id")
        return self.conn.execute(sql, (list_name,)).fetchall()

    def list_complete(self, items: list[str], list_name: str,
                      done: bool = True) -> list[str]:
        """Tick items off, or put them back. Matches loosely, like removal."""
        changed = []
        now = time.time()
        for raw in items:
            needle = " ".join(str(raw).split()).strip().lower()
            if not needle:
                continue
            for row in self.list_read(list_name, include_done=True):
                if needle in row["text"].lower() and bool(row["done"]) != done:
                    self.conn.execute(
                        "UPDATE list_items SET done = ?, done_at = ? "
                        "WHERE id = ?",
                        (1 if done else 0, now if done else None, row["id"]))
                    changed.append(row["text"])
        self.conn.commit()
        return changed

    def list_clear_done(self, list_name: str | None = None) -> int:
        if list_name:
            cur = self.conn.execute(
                "DELETE FROM list_items WHERE done = 1 AND list = ?",
                (list_name,))
        else:
            cur = self.conn.execute("DELETE FROM list_items WHERE done = 1")
        self.conn.commit()
        return cur.rowcount

    def done_since(self, since_ts: float) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM list_items WHERE done = 1 "
            "AND done_at >= ?", (since_ts,)).fetchone()
        return int(row["n"])

    def list_remove(self, items: list[str],
                    list_name: str = "shopping") -> list[str]:
        removed = []
        for raw in items:
            needle = " ".join(raw.split()).strip().lower()
            if not needle:
                continue
            for row in self.list_read(list_name):
                if needle in row["text"].lower():
                    self.conn.execute("DELETE FROM list_items WHERE id = ?",
                                      (row["id"],))
                    removed.append(row["text"])
        self.conn.commit()
        return removed

    def list_counts(self) -> list[tuple[str, int]]:
        """Every list that has anything on it, with its size."""
        rows = self.conn.execute(
            "SELECT list, COUNT(*) AS n FROM list_items WHERE done = 0 "
            "GROUP BY list ORDER BY list"
        ).fetchall()
        return [(r["list"], int(r["n"])) for r in rows]

    def list_clear(self, list_name: str = "shopping") -> int:
        cur = self.conn.execute("DELETE FROM list_items WHERE list = ?",
                                (list_name,))
        self.conn.commit()
        return cur.rowcount

    # -- casework ---------------------------------------------------

    def case_open(self, kind: str, title: str, severity: str,
                  sla_due: float | None, ref: str | None,
                  steps: list[dict]) -> int:
        """Create a case and stamp the playbook's steps onto it."""
        cur = self.conn.execute(
            "INSERT INTO cases (kind, title, ref, severity, opened, sla_due) "
            "VALUES (?,?,?,?,?,?)",
            (kind, title.strip(), (ref or "").strip() or None,
             severity, time.time(), sla_due),
        )
        case_id = int(cur.lastrowid)
        for i, s in enumerate(steps):
            self.conn.execute(
                "INSERT INTO case_steps "
                "(case_id, side, phase, seq, key, text, aliases) "
                "VALUES (?,?,?,?,?,?,?)",
                (case_id, s.get("side", "investigation"), s.get("phase"),
                 i, s["key"], s["text"], "\n".join(s.get("aliases") or [])),
            )
        self.conn.commit()
        return case_id

    def case_active(self) -> sqlite3.Row | None:
        """The case in front of you: most recently opened and still open."""
        return self.conn.execute(
            "SELECT * FROM cases WHERE closed IS NULL "
            "ORDER BY opened DESC LIMIT 1"
        ).fetchone()

    def case_get(self, case_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()

    def cases_open(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM cases WHERE closed IS NULL ORDER BY opened"
        ).fetchall()

    def case_steps(self, case_id: int,
                   side: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM case_steps WHERE case_id = ?"
        args: list[Any] = [case_id]
        if side:
            sql += " AND side = ?"
            args.append(side)
        return self.conn.execute(sql + " ORDER BY seq", args).fetchall()

    def case_step_set(self, step_id: int, done: bool,
                      finding: str | None = None) -> None:
        if finding is not None:
            self.conn.execute(
                "UPDATE case_steps SET done=?, done_at=?, finding=? WHERE id=?",
                (1 if done else 0, time.time() if done else None,
                 finding.strip() or None, step_id))
        else:
            self.conn.execute(
                "UPDATE case_steps SET done=?, done_at=? WHERE id=?",
                (1 if done else 0, time.time() if done else None, step_id))
        self.conn.commit()

    def case_progress(self, case_id: int) -> dict[str, tuple[int, int]]:
        """{side: (done, total)} — what the status read-back is built from."""
        rows = self.conn.execute(
            "SELECT side, COUNT(*) AS n, SUM(done) AS d FROM case_steps "
            "WHERE case_id = ? GROUP BY side", (case_id,)).fetchall()
        return {r["side"]: (int(r["d"] or 0), int(r["n"])) for r in rows}

    def case_steps_set_rest(self, case_id: int, by_key: dict[str, int]) -> None:
        self.conn.executemany(
            "UPDATE case_steps SET rest_seconds = ? WHERE case_id = ? AND key = ?",
            [(sec, case_id, key) for key, sec in by_key.items()])
        self.conn.commit()

    def case_step_rest(self, step_id: int) -> int | None:
        row = self.conn.execute(
            "SELECT rest_seconds FROM case_steps WHERE id = ?",
            (step_id,)).fetchone()
        return int(row["rest_seconds"]) if row and row["rest_seconds"] else None

    def workout_history(self, exercise: str, limit: int = 5) -> list[sqlite3.Row]:
        """What was logged for this exercise in earlier sessions.

        No new table: a workout IS a case, so its history is already sitting
        in case_steps. Closed sessions only — what you are lifting right now
        is not "last time".
        """
        needle = f"%{' '.join(str(exercise).split()).strip().lower()}%"
        return self.conn.execute(
            "SELECT c.opened, s.text, s.finding FROM case_steps s "
            "JOIN cases c ON c.id = s.case_id "
            "WHERE c.kind = 'workout' AND c.closed IS NOT NULL "
            "  AND s.finding IS NOT NULL AND LOWER(s.text) LIKE ? "
            "ORDER BY c.opened DESC LIMIT ?", (needle, limit)).fetchall()

    def case_note(self, case_id: int, text: str) -> None:
        row = self.case_get(case_id)
        prior = (row["notes"] if row else "") or ""
        stamp = time.strftime("%H:%M", time.localtime())
        self.conn.execute("UPDATE cases SET notes = ? WHERE id = ?",
                          (f"{prior}{stamp}  {text.strip()}\n", case_id))
        self.conn.commit()

    def case_close(self, case_id: int, outcome: str | None) -> None:
        self.conn.execute(
            "UPDATE cases SET closed = ?, outcome = ? WHERE id = ?",
            (time.time(), (outcome or "").strip() or None, case_id))
        self.conn.commit()

    def case_set_sla(self, case_id: int, sla_due: float | None) -> None:
        self.conn.execute("UPDATE cases SET sla_due = ? WHERE id = ?",
                          (sla_due, case_id))
        self.conn.commit()

    def case_alert_once(self, case_id: int, marker: str) -> bool:
        """True the first time this marker is claimed, False ever after.

        The INSERT is the lock. Checking-then-inserting would let two
        scheduler ticks both decide they were first.
        """
        try:
            self.conn.execute(
                "INSERT INTO case_alerts (case_id, marker, fired) "
                "VALUES (?,?,?)", (case_id, marker, time.time()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # -- embeddings -------------------------------------------------

    def embeddings_put(self, kind: str, model: str,
                       rows: list[tuple[int, bytes]]) -> None:
        self.conn.executemany(
            "INSERT INTO embeddings (kind, ref_id, model, vec, made) "
            "VALUES (?,?,?,?,?) ON CONFLICT(kind, ref_id, model) "
            "DO UPDATE SET vec=excluded.vec, made=excluded.made",
            [(kind, rid, model, vec, time.time()) for rid, vec in rows],
        )
        self.conn.commit()

    def embeddings_all(self, kind: str, model: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT ref_id, vec FROM embeddings WHERE kind = ? AND model = ? "
            "ORDER BY ref_id", (kind, model),
        ).fetchall()

    def embeddings_missing(self, kind: str, model: str,
                           limit: int) -> list[sqlite3.Row]:
        """Rows of `kind` with no vector for this model yet.

        Newest first. If the backfill never finishes — a big history, a box
        that gets rebooted — the half that IS indexed should be the half you
        are most likely to ask about.
        """
        if kind == "memory":
            sql = ("SELECT m.id, m.text FROM memories m "
                   "LEFT JOIN embeddings e ON e.ref_id = m.id "
                   "  AND e.kind = 'memory' AND e.model = ? "
                   "WHERE e.ref_id IS NULL ORDER BY m.id DESC LIMIT ?")
        else:
            sql = ("SELECT m.id, m.content AS text FROM messages m "
                   "LEFT JOIN embeddings e ON e.ref_id = m.id "
                   "  AND e.kind = 'message' AND e.model = ? "
                   "WHERE e.ref_id IS NULL AND LENGTH(m.content) >= 12 "
                   "ORDER BY m.id DESC LIMIT ?")
        return self.conn.execute(sql, (model, limit)).fetchall()

    def embeddings_counts(self, model: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT kind, COUNT(*) AS n FROM embeddings WHERE model = ? "
            "GROUP BY kind", (model,)).fetchall()
        return {r["kind"]: int(r["n"]) for r in rows}

    def embeddings_drop_other_models(self, keep: str) -> int:
        """Delete vectors from any other model.

        Called when the configured model changes. Keeping them wastes space
        and risks a future bug ranking across incompatible spaces; the cost of
        being wrong is a re-index, which is cheap and automatic.
        """
        cur = self.conn.execute(
            "DELETE FROM embeddings WHERE model != ?", (keep,))
        self.conn.commit()
        return cur.rowcount

    def messages_by_ids(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, ts, role, content FROM messages WHERE id IN ({marks})",
            ids).fetchall()
        order = {rid: i for i, rid in enumerate(ids)}
        return sorted(rows, key=lambda r: order.get(r["id"], 1 << 30))

    def search_ids(self, query: str, limit: int = 20) -> list[int]:
        """Keyword hit IDs in relevance order, for the hybrid merge."""
        match = fts_query(query)
        if match is None:
            return []
        rows = self.conn.execute(
            "SELECT f.rowid AS id FROM messages_fts f "
            "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, limit)).fetchall()
        return [int(r["id"]) for r in rows]

    # -- usage ------------------------------------------------------

    def add_usage(self, model: str, tier: str, usage: dict[str, int],
                  cost_usd: float) -> None:
        self.conn.execute(
            "INSERT INTO usage (ts, model, tier, input_tokens, output_tokens,"
            " cache_write, cache_read, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
            (
                time.time(), model, tier,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
                usage.get("cache_read_input_tokens", 0),
                cost_usd,
            ),
        )
        self.conn.commit()

    def spend_since(self, since_ts: float) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM usage WHERE ts >= ?",
            (since_ts,),
        ).fetchone()
        return float(row["total"])

    # -- wake events ------------------------------------------------

    def add_wake_event(self, score: float, accepted: bool,
                       transcript: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO wake_events (ts, score, accepted, transcript) "
            "VALUES (?, ?, ?, ?)",
            (time.time(), score, 1 if accepted else 0, transcript),
        )
        self.conn.commit()

    def wake_stats(self, since_ts: float) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT score, accepted, transcript FROM wake_events WHERE ts >= ?",
            (since_ts,),
        ).fetchall()
        total = len(rows)
        empty = sum(1 for r in rows if not (r["transcript"] or "").strip())
        return {
            "triggers": total,
            "likely_false_positives": empty,
            "scores": [r["score"] for r in rows],
        }

    def close(self) -> None:
        self.conn.close()
