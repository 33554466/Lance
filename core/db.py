"""SQLite persistence: conversation history, usage log, wake events.

One file, no server, survives power cuts. FTS5 over message content so the
assistant can answer "what did I ask you about the insurance letter?".
"""
from __future__ import annotations

import json
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
    added    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_list_items ON list_items(list, id);

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


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

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
        return self.conn.execute(
            "SELECT m.ts, m.role, m.content FROM messages_fts f "
            "JOIN messages m ON m.id = f.rowid "
            "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
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

    def list_read(self, list_name: str = "shopping") -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, text FROM list_items WHERE list = ? ORDER BY id",
            (list_name,),
        ).fetchall()

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
            "SELECT list, COUNT(*) AS n FROM list_items "
            "GROUP BY list ORDER BY list"
        ).fetchall()
        return [(r["list"], int(r["n"])) for r in rows]

    def list_clear(self, list_name: str = "shopping") -> int:
        cur = self.conn.execute("DELETE FROM list_items WHERE list = ?",
                                (list_name,))
        self.conn.commit()
        return cur.rowcount

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
