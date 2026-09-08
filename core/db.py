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
-- A memory is not a row. It is an interval with a belief attached.
--
-- The dangerous memory is not the one that was never true — that contradicts
-- something, retrieval scores it badly, and a user corrects it. It is the one
-- that WAS true. "The voice is lessac" was right for weeks. It embeds
-- perfectly, retrieves at the top, and gets stated with total confidence long
-- after it stopped being so.
--
-- Two columns fix the first half: valid_from/valid_to make a memory a window
-- rather than a value, so a correction CLOSES the old row instead of erasing
-- it. "What did I have the voice set to before?" stays answerable.
--
-- last_verified fixes the second half. Age is measured from EVIDENCE, not
-- from first contact: a fact learned a year ago and confirmed last week is
-- fresh, while one learned last week and never mentioned since is starting to
-- rot. Measuring from `created` punishes long-standing facts that keep being
-- re-confirmed, which is exactly backwards.
--
-- Note what is absent: nothing here is ever DELETED. That is the single
-- decision the rest of this design exists to protect.
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created       REAL NOT NULL,
    updated       REAL NOT NULL,
    category      TEXT NOT NULL DEFAULT 'general',
    text          TEXT NOT NULL,
    -- How fast this kind of fact goes bad. Drives the half-life.
    volatility    TEXT NOT NULL DEFAULT 'slow'
                  CHECK (volatility IN ('stable', 'slow', 'fast')),
    -- Valid time: when the fact was true in the world. NULL end = still open.
    valid_from    REAL NOT NULL,
    valid_to      REAL,
    -- When we last had evidence it still holds.
    last_verified REAL NOT NULL,
    --   active      believed, and fresh
    --   stale       believed, but past its half-life. Still retrieved, flagged.
    --   superseded  replaced; its successor is the live one
    --   expired     ended, and nothing succeeds it
    status        TEXT NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'stale', 'superseded', 'expired')),
    superseded_by INTEGER REFERENCES memories(id)
);

-- UNIQUE on live rows only. The old column-level UNIQUE made it impossible to
-- ever hold a fact that had been true, stopped being true, and became true
-- again — which is a thing that happens.
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_live
    ON memories(text) WHERE status IN ('active', 'stale');
CREATE INDEX IF NOT EXISTS idx_memories_status
    ON memories(status, last_verified);

-- Append-only. Rows are never rewritten, so this is the record of every
-- change a memory has been through and what decided it.
CREATE TABLE IF NOT EXISTS memory_repairs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          REAL NOT NULL,
    op          TEXT NOT NULL
                CHECK (op IN ('insert','supersede','expire','confirm','flag')),
    detector    TEXT NOT NULL
                CHECK (detector IN ('arrival','aged','scheduled','human')),
    old_id      INTEGER,
    new_id      INTEGER,
    before_text TEXT,
    after_text  TEXT,
    reason      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_repairs_at ON memory_repairs(at);

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
        # BEFORE the schema script. The new memories table carries a partial
        # unique index predicated on `status`, and running that against a
        # pre-migration table fails with "no such column: status". The
        # rebuild has to clear the way first.
        self._migrate_memories()
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

    def _migrate_memories(self) -> None:
        """Give existing memories a validity window and a freshness clock.

        This one cannot be done with ALTER TABLE alone. The old schema put
        UNIQUE directly on `text`, and superseding needs that gone — a fact
        that was true, stopped being true, and became true again is a real
        thing, and a column-level constraint makes it unstorable. SQLite
        cannot drop a constraint, so the table is rebuilt.

        Rebuilding a table of real household facts deserves care, so: one
        transaction, ids preserved (the embeddings table references them),
        row counts compared before anything is dropped, and a rollback if
        they disagree.
        """
        cols = {r["name"] for r in
                self.conn.execute("PRAGMA table_info(memories)").fetchall()}
        if not cols or "volatility" in cols:
            return                      # fresh database, or already done

        before = int(self.conn.execute(
            "SELECT COUNT(*) AS n FROM memories").fetchone()["n"])
        print(f"migrating {before} memories to windowed form...")
        now = time.time()
        try:
            self.conn.execute("BEGIN")
            self.conn.execute("ALTER TABLE memories RENAME TO memories_v1")
            # Re-run the schema to build the new table and its indexes.
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT INTO memories (id, created, updated, category, text,"
                " volatility, valid_from, valid_to, last_verified, status,"
                " superseded_by) "
                # `updated` becomes last_verified: it is the closest thing the
                # old schema has to "when we last had evidence for this".
                # valid_from is `created` — we do not know when the fact
                # became true, only when we heard it, and that is the honest
                # lower bound.
                "SELECT id, created, updated, category, text,"
                " 'slow', created, NULL, updated, 'active', NULL "
                "FROM memories_v1")
            after = int(self.conn.execute(
                "SELECT COUNT(*) AS n FROM memories").fetchone()["n"])
            if after != before:
                raise RuntimeError(
                    f"row count changed: {before} -> {after}")
            self.conn.execute("DROP TABLE memories_v1")
            self.conn.execute("COMMIT")
            print(f"migrated: {after} memories, none lost")
            self.conn.execute(
                "INSERT INTO memory_repairs (at, op, detector, reason) "
                "VALUES (?, 'confirm', 'human', ?)",
                (now, f"migrated {after} memories to windowed form"))
            self.conn.commit()
        except Exception as exc:  # noqa: BLE001
            self.conn.execute("ROLLBACK")
            print(f"memory migration FAILED and was rolled back: {exc}")
            raise

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

    LIVE = ("active", "stale")

    def _repair(self, op: str, detector: str, reason: str = "",
                old_id: int | None = None, new_id: int | None = None,
                before: str | None = None, after: str | None = None,
                now: float | None = None) -> None:
        self.conn.execute(
            "INSERT INTO memory_repairs (at, op, detector, old_id, new_id,"
            " before_text, after_text, reason) VALUES (?,?,?,?,?,?,?,?)",
            (now or time.time(), op, detector, old_id, new_id,
             before, after, reason))

    def add_memory(self, text: str, category: str = "general",
                   volatility: str = "slow", now: float | None = None,
                   replaces_id: int | None = None) -> str:
        """Store a fact. Returns 'saved', 'replaced', or 'confirmed'.

        Three outcomes, not two, and the third is the interesting one. Saying
        a fact that is already stored is EVIDENCE it still holds — so it
        refreshes the verification clock rather than being a no-op. That is
        what stops a fact you mention weekly from ever going stale, and it
        costs nothing.
        """
        now = now or time.time()
        text = text.strip()
        if volatility not in ("stable", "slow", "fast"):
            volatility = "slow"

        existing = self.conn.execute(
            f"SELECT id FROM memories WHERE text = ? AND status IN {self.LIVE}",
            (text,)).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE memories SET last_verified = ?, updated = ?, "
                "status = 'active', category = ? WHERE id = ?",
                (now, now, category, existing["id"]))
            self._repair("confirm", "arrival", "restated by the user",
                         old_id=existing["id"], before=text, now=now)
            self.conn.commit()
            return "confirmed"

        old = None
        if replaces_id is not None:
            old = self.conn.execute(
                f"SELECT id, text FROM memories WHERE id = ? "
                f"AND status IN {self.LIVE}", (replaces_id,)).fetchone()

        cur = self.conn.execute(
            "INSERT INTO memories (created, updated, category, text,"
            " volatility, valid_from, valid_to, last_verified, status) "
            "VALUES (?,?,?,?,?,?,NULL,?,'active')",
            (now, now, category, text, volatility, now, now))
        new_id = int(cur.lastrowid)

        if old is not None:
            # The successor's window opens exactly where the predecessor's
            # closes. No gap, no overlap — that is what makes "what did I
            # think in March" answer with one row rather than none or two.
            self.conn.execute(
                "UPDATE memories SET valid_to = ?, status = 'superseded', "
                "superseded_by = ?, updated = ? WHERE id = ?",
                (now, new_id, now, old["id"]))
            self._repair("supersede", "arrival", "corrected by the user",
                         old_id=old["id"], new_id=new_id,
                         before=old["text"], after=text, now=now)
            self.conn.commit()
            return "replaced"

        self._repair("insert", "arrival", "new fact", new_id=new_id,
                     after=text, now=now)
        self.conn.commit()
        return "saved"

    def list_memories(self, include_retired: bool = False) -> list[sqlite3.Row]:
        sql = ("SELECT id, category, text, updated, volatility, status,"
               " last_verified, valid_from, valid_to, superseded_by"
               " FROM memories")
        if not include_retired:
            sql += f" WHERE status IN {self.LIVE}"
        return self.conn.execute(sql + " ORDER BY category, id").fetchall()

    def count_memories(self) -> int:
        return int(self.conn.execute(
            f"SELECT COUNT(*) AS n FROM memories WHERE status IN {self.LIVE}"
        ).fetchone()["n"])

    def find_memories(self, needle: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute(
            f"SELECT id, category, text, status FROM memories "
            f"WHERE text LIKE ? AND status IN {self.LIVE} "
            f"ORDER BY id LIMIT ?", (f"%{needle.strip()}%", limit),
        ).fetchall()

    def memory_history(self, needle: str, limit: int = 6) -> list[sqlite3.Row]:
        """Everything ever believed on a subject, newest first.

        This is the whole point of never deleting. "What did I have the voice
        set to before?" has an answer because the old row is still there with
        its window closed, rather than having been overwritten.
        """
        return self.conn.execute(
            "SELECT id, text, status, valid_from, valid_to, category "
            "FROM memories WHERE text LIKE ? ORDER BY valid_from DESC LIMIT ?",
            (f"%{needle.strip()}%", limit)).fetchall()

    def expire_memory(self, mem_id: int, now: float | None = None,
                      reason: str = "no longer true") -> bool:
        """Close a fact's window. It stops being retrieved; it is not deleted.

        This replaces the old delete_memory. "Forget that" should mean "stop
        telling me that", not "destroy the record that it was ever so" — the
        second makes the assistant unable to explain itself later.
        """
        now = now or time.time()
        row = self.conn.execute(
            f"SELECT id, text FROM memories WHERE id = ? AND status IN {self.LIVE}",
            (mem_id,)).fetchone()
        if row is None:
            return False
        self.conn.execute(
            "UPDATE memories SET valid_to = ?, status = 'expired', updated = ? "
            "WHERE id = ?", (now, now, mem_id))
        self._repair("expire", "human", reason, old_id=mem_id,
                     before=row["text"], now=now)
        self.conn.commit()
        return True

    def confirm_memory(self, mem_id: int, now: float | None = None) -> bool:
        """Fresh evidence. Resets the decay clock and clears any flag."""
        now = now or time.time()
        cur = self.conn.execute(
            f"UPDATE memories SET last_verified = ?, updated = ?, "
            f"status = 'active' WHERE id = ? AND status IN {self.LIVE}",
            (now, now, mem_id))
        if cur.rowcount:
            self._repair("confirm", "human", "confirmed still true",
                         old_id=mem_id, now=now)
            self.conn.commit()
        return cur.rowcount > 0

    def flag_stale(self, ids: list[int], now: float | None = None) -> int:
        now = now or time.time()
        n = 0
        for mid in ids:
            cur = self.conn.execute(
                "UPDATE memories SET status = 'stale' WHERE id = ? "
                "AND status = 'active'", (mid,))
            if cur.rowcount:
                self._repair("flag", "aged", "past its half-life",
                             old_id=mid, now=now)
                n += cur.rowcount
        self.conn.commit()
        return n

    def memory_repairs(self, limit: int = 30) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM memory_repairs ORDER BY at DESC LIMIT ?",
            (limit,)).fetchall()

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
        # `is not None`, not a truth test. Zero is a MEANING here — "no rest
        # after this one" — and a falsy check turns it back into NULL, which
        # then falls through to the default. That is the difference between a
        # silent warm-up and ninety seconds of countdown after every ankle
        # rock.
        if row is None or row["rest_seconds"] is None:
            return None
        return int(row["rest_seconds"])

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

    def workout_ran_today(self, filename: str) -> bool:
        """Has this workout file already been started today?

        Only meaningful for a dated programme with two sessions on one date:
        it is what makes the second "start today's workout" open the evening
        session instead of repeating the morning.
        """
        midnight = time.time() - (time.time() % 86400)
        row = self.conn.execute(
            "SELECT 1 FROM cases WHERE kind = 'workout' AND ref = ? "
            "AND opened >= ? LIMIT 1", (filename, midnight)).fetchone()
        return row is not None

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
