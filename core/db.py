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
