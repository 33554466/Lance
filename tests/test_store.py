"""SQLite behaviour: the migration, list removal, and what "today" means.

These three cover the paths where the appliance could lose something and say
nothing. Run it:

    python -m tests.test_store
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.clock import local_midnight            # noqa: E402
from core.db import SCHEMA, Store                # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def _store(tmp: Path, name: str = "t.db") -> Store:
    return Store(tmp / name)


# ------------------------------------------------------------ list_remove

def test_list_remove(tmp: Path) -> None:
    print("\nTaking things off a list")
    st = _store(tmp, "lists.db")
    st.list_add(["milk", "almond milk", "milk chocolate", "bread"], "shopping")

    def texts():
        return [r["text"] for r in st.list_read("shopping")]

    # An exact match wins outright, even though two other rows contain it.
    check("an exact match removes exactly one",
          st.list_remove(["milk"], "shopping") == ["milk"])
    check("...and leaves the other two alone",
          texts() == ["almond milk", "milk chocolate", "bread"], str(texts()))

    # The bug this replaced: a single character emptied the board.
    check("a one-letter mishearing removes nothing",
          st.list_remove(["a"], "shopping") == [])
    check("...and the list is untouched", len(texts()) == 3)

    # Ambiguity refuses rather than guessing.
    check("an ambiguous substring removes nothing",
          st.list_remove(["milk"], "shopping") == [])
    check("...because it matched two items", len(texts()) == 3)

    # An unambiguous substring still works — you should not have to be exact.
    check("an unambiguous substring works",
          st.list_remove(["chocolate"], "shopping") == ["milk chocolate"])

    # Case and spacing do not matter.
    check("case-insensitive", st.list_remove(["BREAD"], "shopping") == ["bread"])
    check("extra spaces collapse",
          st.list_remove(["  almond   milk "], "shopping") == ["almond milk"])
    check("the list is now empty", texts() == [], str(texts()))

    # Junk arguments from a model must not raise.
    check("empty and None arguments are ignored",
          st.list_remove(["", None, "   "], "shopping") == [])


# --------------------------------------------------------- local midnight

def test_local_midnight() -> None:
    print("\nWhat 'today' means")
    utc_style = time.time() - (time.time() % 86400)
    local = local_midnight()

    now = dt.datetime.now()
    expected = now.replace(hour=0, minute=0, second=0,
                           microsecond=0).timestamp()
    check("it is actually local midnight", abs(local - expected) < 1,
          f"off by {local - expected:.0f}s")
    check("it is never in the future", local <= time.time())
    check("it is within the last 24 hours", time.time() - local < 86400)

    off = dt.datetime.now().astimezone().utcoffset()
    if off and off.total_seconds() != 0:
        check("it differs from the UTC-modulus version this test replaced",
              abs(local - utc_style) > 60,
              f"identical — is this box on UTC? offset {off}")
        print(f"     (this machine is UTC{off.total_seconds() / 3600:+.0f}; "
              f"the old code was "
              f"{abs(local - utc_style) / 3600:.0f}h out)")
    else:
        print("     (this machine is on UTC, so both agree here — the fix "
              "still matters in Phoenix)")

    # A workout started at 11pm last night is NOT today's.
    st_now = dt.datetime.now()
    if st_now.hour >= 1:
        yesterday_late = (st_now - dt.timedelta(days=1)).replace(
            hour=23, minute=0).timestamp()
        check("last night at 11pm is before today's midnight",
              yesterday_late < local)


def test_workout_ran_today(tmp: Path) -> None:
    print("\nDouble-session days")
    st = _store(tmp, "workouts.db")
    steps = [{"side": "main", "phase": "Main", "key": "bench",
              "text": "Bench press, 5 x 5", "aliases": ["bench"]}]

    check("nothing run yet", st.workout_ran_today("day-1.txt") is False)
    st.case_open("workout", "Day 1", "standard", None, "day-1.txt", steps)
    check("now it has", st.workout_ran_today("day-1.txt") is True)
    check("a different file has not",
          st.workout_ran_today("day-2.txt") is False)

    # A session opened before local midnight must not count as today's.
    before = local_midnight() - 3600
    cid = st.case_open("workout", "Yesterday", "standard", None,
                       "old.txt", steps)
    st.conn.execute("UPDATE cases SET opened = ? WHERE id = ?", (before, cid))
    st.conn.commit()
    check("yesterday evening does not count as today",
          st.workout_ran_today("old.txt") is False)


# ------------------------------------------------------------- migration

def _legacy_db(path: Path, rows: int = 3) -> None:
    """A database in the pre-window memories shape."""
    c = sqlite3.connect(str(path))
    c.executescript("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created REAL NOT NULL,
            updated REAL NOT NULL,
            category TEXT NOT NULL DEFAULT 'fact',
            text TEXT NOT NULL UNIQUE
        );
    """)
    now = time.time()
    for i in range(rows):
        c.execute("INSERT INTO memories (created, updated, category, text) "
                  "VALUES (?, ?, 'fact', ?)",
                  (now - i * 86400, now, f"remembered fact number {i}"))
    c.commit()
    c.close()


def test_migration(tmp: Path) -> None:
    print("\nMigrating memories to windowed form")
    path = tmp / "legacy.db"
    _legacy_db(path, rows=4)

    st = Store(path)
    rows = st.list_memories()
    check("all four facts survived", len(rows) == 4, f"got {len(rows)}")
    check("ids were preserved (embeddings reference them)",
          sorted(r["id"] for r in rows) == [1, 2, 3, 4])
    check("every fact got a volatility",
          all(r["volatility"] for r in rows))
    check("every fact is active", all(r["status"] == "active" for r in rows))
    check("the old table is gone",
          st.conn.execute("SELECT COUNT(*) AS n FROM sqlite_master "
                          "WHERE name = 'memories_v1'").fetchone()["n"] == 0)

    backups = list(tmp.glob("legacy.db.*pre-memory-migration.bak"))
    check("a file backup was taken before touching anything",
          len(backups) == 1, f"found {len(backups)}")
    if backups:
        c = sqlite3.connect(str(backups[0]))
        n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        c.close()
        check("...and the backup holds the original rows", n == 4, f"{n} rows")

    # Re-opening must not migrate twice.
    st2 = Store(path)
    check("re-opening is a no-op", len(st2.list_memories()) == 4)
    check("no second backup was taken",
          len(list(tmp.glob("legacy.db.*pre-memory-migration.bak"))) == 1)


def test_migration_failure_keeps_a_copy(tmp: Path) -> None:
    """The failure mode that made this a Tier 1 finding.

    executescript commits, so the ALTER TABLE cannot be rolled back. What has
    to survive instead is the file copy — and the log has to say where it is
    rather than claiming a rollback that did not happen.
    """
    print("\nWhen the migration fails")
    path = tmp / "doomed.db"
    _legacy_db(path, rows=3)

    import core.db as dbmod
    original = dbmod.SCHEMA
    # Break the copy step: a NOT NULL column the INSERT ... SELECT cannot fill.
    dbmod.SCHEMA = original + "\nCREATE TABLE IF NOT EXISTS _x (a NOT NULL);" \
                              "\nINSERT INTO _x (a) VALUES (NULL);"
    raised = None
    try:
        Store(path)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    finally:
        dbmod.SCHEMA = original

    check("it fails loudly rather than continuing", raised is not None,
          "the Store was built despite a broken schema")
    backups = list(tmp.glob("doomed.db.*pre-memory-migration.bak"))
    check("the pre-migration copy exists", len(backups) == 1,
          f"found {len(backups)}")
    if backups:
        c = sqlite3.connect(str(backups[0]))
        n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        c.close()
        check("...with all three facts in it", n == 3, f"{n} rows")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_list_remove(tmp)
        test_local_midnight()
        test_workout_ran_today(tmp)
        test_migration(tmp)
        test_migration_failure_keeps_a_copy(tmp)
    print()
    if FAILURES:
        print(f"\033[1;31m{len(FAILURES)} failed\033[0m")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\033[1;32mall passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
