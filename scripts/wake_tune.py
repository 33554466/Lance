#!/usr/bin/env python3
"""What would a different wake-word threshold actually have cost you?

Every trigger is already logged with its score and whatever you said next, so
this does not have to guess: it replays your own history against candidate
thresholds and prints the trade in the only two units that matter — false
wakes avoided, and real wakes lost.

    python3 scripts/wake_tune.py             # last 7 days
    python3 scripts/wake_tune.py --days 30
    python3 scripts/wake_tune.py --days 30 --show 20

One honest limitation, stated up front. The log only contains triggers that
FIRED, so this can tell you what a HIGHER threshold would have done and can
say nothing at all about a lower one. It also cannot model confirm_frames or
vad_threshold, which suppress candidates before they are ever recorded. Change
those, then let it run for a few days, then come back here.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "assistant.db"

# A wake that produced no transcript is one you did not mean. The device woke,
# chimed, listened, and nobody was talking to it.
CANDIDATES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--show", type=int, default=10,
                    help="list this many recent false wakes")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"No database at {args.db}")
        return 1

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    since = time.time() - args.days * 86400
    rows = conn.execute(
        "SELECT ts, score, transcript FROM wake_events "
        "WHERE ts >= ? ORDER BY ts", (since,)
    ).fetchall()

    if not rows:
        print(f"No wake events in the last {args.days:g} days.\n"
              f"Check logging.log_wake_events is true in config.yaml.")
        return 1

    real = [r for r in rows if (r["transcript"] or "").strip()]
    false = [r for r in rows if not (r["transcript"] or "").strip()]

    print(f"\n  Wake events, last {args.days:g} days")
    print(f"  {'-' * 58}")
    print(f"  {len(rows):4d} triggers   "
          f"{len(rows) / max(args.days, 0.01):.1f} per day")
    print(f"  {len(real):4d} you meant")
    print(f"  {len(false):4d} you did not  "
          f"({100 * len(false) / len(rows):.0f}% of all triggers)")

    if real:
        rs = sorted(r["score"] for r in real)
        print(f"\n  Scores when you MEANT it:      "
              f"min {rs[0]:.3f}   median {rs[len(rs) // 2]:.3f}   "
              f"max {rs[-1]:.3f}")
    if false:
        fs = sorted(r["score"] for r in false)
        print(f"  Scores when you did NOT:       "
              f"min {fs[0]:.3f}   median {fs[len(fs) // 2]:.3f}   "
              f"max {fs[-1]:.3f}")

    # The replay. A trigger survives a candidate threshold if its score was at
    # or above it; everything below simply would not have fired.
    print(f"\n  If the threshold had been...")
    print(f"  {'-' * 58}")
    print(f"  {'threshold':>10}  {'false wakes':>12}  {'real wakes lost':>16}")
    best = None
    for t in CANDIDATES:
        f_kept = sum(1 for r in false if r["score"] >= t)
        r_lost = sum(1 for r in real if r["score"] < t)
        flag = ""
        # The recommendation: the highest threshold that still costs you
        # nothing. Missing a wake word is a much worse experience than an
        # occasional spurious one, so this will never trade a real wake away.
        if r_lost == 0:
            best = t
            flag = "  <- free"
        print(f"  {t:>10.2f}  {f_kept:>12d}  {r_lost:>16d}{flag}")

    if best is not None:
        now_false = len(false)
        would = sum(1 for r in false if r["score"] >= best)
        print(f"\n  Recommended: threshold {best:.2f}")
        if would < now_false:
            print(f"  Removes {now_false - would} of {now_false} false wakes "
                  f"and loses none of the real ones.")
        else:
            print(f"  Your false wakes score as high as the real ones, so the "
                  f"threshold\n  cannot separate them. Raise confirm_frames "
                  f"instead, or train a\n  custom wake phrase — that is what "
                  f"the ceiling here is telling you.")

    if false and args.show:
        print(f"\n  Recent false wakes")
        print(f"  {'-' * 58}")
        for r in false[-args.show:]:
            when = time.strftime("%a %d %b %H:%M", time.localtime(r["ts"]))
            print(f"  {when}   score {r['score']:.3f}")
        print("\n  Look at the times. A cluster at dinner or during a film is "
              "the room,\n  not the threshold — and the fix for the room is "
              "confirm_frames.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
