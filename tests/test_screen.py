"""The dashboard's panel logic, driven in a real browser.

This suite exists because of a specific bug and the shape of it is worth
remembering: the orchestrator redraws the interval timer once a second, and
the renderer treated every one of those frames as "take the screen". Pressing
b during an AMRAP put the board up for a fraction of a second and then the
next frame took it straight back — which looks exactly like a key that does
nothing, and is invisible in any test of the Python side, because the Python
side was behaving perfectly.

So: taking the screen is an EVENT (a new session), not a property of every
frame. That is what is asserted here.

Skips rather than fails when Playwright or a browser is not installed — the
appliance does not need a browser engine to run, only to be tested.

    python -m tests.test_screen
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = "file://" + str(ROOT / "ui" / "index.html")

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def skip(why: str) -> int:
    print(f"  \033[1;33m·\033[0m skipped: {why}")
    return 0


# A frame as the orchestrator sends it.
FRAME = dict(running=True, session=1.0, kind="run", label="AMRAP",
             name="20 minute AMRAP", remaining=1200, phase_seconds=1200,
             left=1200, total=1200, round=0, rounds=0, paused=False,
             held=False)


def _launch(pw):
    """Whatever browser this machine has. Nothing is installed for this."""
    import glob
    tries = [None]
    tries += sorted(glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome"))
    for path in tries:
        try:
            if path is None:
                return pw.chromium.launch()
            return pw.chromium.launch(executable_path=path)
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    print("\nThe screen")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return skip("playwright is not installed")

    with sync_playwright() as pw:
        browser = _launch(pw)
        if browser is None:
            return skip("no chromium for playwright to drive")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(PAGE)
        page.wait_for_timeout(400)

        def draw(**over) -> None:
            frame = dict(FRAME)
            frame.update(over)
            page.evaluate(f"window.lance.renderTimer({json.dumps(frame)})")

        def on() -> bool:
            return page.evaluate("window.lance.timerOn()")

        draw()
        check("a new timer takes the screen", on())
        check("...and says what it is",
              page.inner_text("#timer-label") == "AMRAP",
              page.inner_text("#timer-label"))

        page.evaluate("window.lance.timerAsideSet(true)")
        check("stepping it aside clears the screen", not on())

        # The regression. Five seconds of redraws, and the screen stays his.
        for i in range(5):
            draw(remaining=1200 - i)
        check("redraws do not take the screen back", not on())

        page.evaluate("window.lance.toggleTimer()")
        check("asking for it brings it back", on())
        draw(remaining=900)
        check("...showing the real clock, not a stale one",
              page.inner_text("#timer-clock") == "15:00",
              page.inner_text("#timer-clock"))

        page.evaluate("window.lance.timerAsideSet(true)")
        draw(session=2.0, remaining=40, phase_seconds=40, kind="work",
             label="Work", round=1, rounds=8, name="40/20 x8")
        check("a NEW timer does take the screen", on())
        check("...with the round on it",
              page.inner_text("#timer-round") == "round 1 of 8",
              page.inner_text("#timer-round"))

        draw(running=False)
        check("finishing clears it", not on())
        page.evaluate("window.lance.toggleTimer()")
        check("asking for a timer that is not running does nothing", not on())

        draw(session=3.0, kind="ready", label="Ready", remaining=5,
             phase_seconds=5, held=True)
        check("a held timer shows", on())
        check("...and says what it is waiting for",
              "go" in page.inner_text("#timer-round"),
              page.inner_text("#timer-round"))
        check("...and is not urgent, because its clock is not moving",
              "urgent" not in (page.get_attribute("#timer", "class") or ""),
              page.get_attribute("#timer", "class"))

        draw(session=3.0, kind="work", label="Work", remaining=3,
             phase_seconds=40, round=2, rounds=8, held=False)
        check("three seconds of a real phase is urgent",
              "urgent" in (page.get_attribute("#timer", "class") or ""),
              page.get_attribute("#timer", "class"))

        # ---- the status pill --------------------------------------
        # It has to be readable from across the room at every moment,
        # including the ones where a full-bleed panel owns the screen. That
        # is the whole reason it exists — the bar at the bottom says the
        # same words and is covered exactly when you need them.
        print("\n  the status pill")

        # Two things would make this flaky, and both are the page behaving
        # correctly rather than badly. The colours cross-fade over 250ms, so
        # anything read immediately is a blend; and with no orchestrator to
        # talk to, the socket keeps failing and setting "offline" between a
        # call and a read. Kill the transitions and take each reading in a
        # single evaluate, so nothing can land in the gap.
        page.add_style_tag(content="*{transition:none!important;"
                                   "animation:none!important}")

        def pill(prop: str) -> str:
            return page.evaluate(
                f"getComputedStyle(document.getElementById('pill')).{prop}")

        def set_and_read(state: str) -> dict:
            return page.evaluate(
                "(s) => { window.lance.setState(s, '');"
                " const p = document.getElementById('pill');"
                " const d = document.getElementById('pill-dot');"
                " return { text: document.getElementById('pill-text')"
                "            .textContent,"
                "          dot: getComputedStyle(d).backgroundColor,"
                "          opacity: getComputedStyle(p).opacity }; }",
                state)

        rect = json.loads(page.evaluate(
            "JSON.stringify(document.getElementById('pill')"
            ".getBoundingClientRect())"))
        check("it sits in the top right",
              rect["top"] < 60 and rect["right"] > 1860, str(rect))
        check("it is above the timer and the check",
              int(pill("zIndex")) > 60, pill("zIndex"))
        check("it never eats a click", pill("pointerEvents") == "none",
              pill("pointerEvents"))

        seen = {}
        for state, word in (("idle", "Ready"),
                            ("listening", "Listening"),
                            ("thinking", "Thinking"),
                            ("speaking", "Speaking"),
                            ("offline", "offline")):
            got = set_and_read(state)
            check(f"{state} says so", word.lower() in got["text"].lower(),
                  got["text"])
            seen[state] = got

        # One colour per state. A status light that looks the same in two
        # states is decoration, not information.
        colours = {k: v["dot"] for k, v in seen.items()}
        check("every state has its own colour",
              len(set(colours.values())) == len(colours), str(colours))
        check("listening is green",
              colours["listening"] == "rgb(63, 185, 80)",
              colours["listening"])
        check("offline is red", colours["offline"] == "rgb(248, 81, 73)",
              colours["offline"])

        # Idle recedes, anything else comes forward. This sits in a room all
        # night; a status light at full brightness is one people cover up.
        check("idle is dimmer than working",
              float(seen["idle"]["opacity"])
              < float(seen["listening"]["opacity"]),
              f'{seen["idle"]["opacity"]} vs {seen["listening"]["opacity"]}')

        # And none of it is any use if a panel can cover it.
        set_and_read("listening")
        draw(session=9.0, kind="work", label="Work", remaining=12,
             phase_seconds=40, round=3, rounds=8)
        check("still visible with the timer full screen",
              page.is_visible("#pill"))
        check("...still saying what she is doing",
              page.inner_text("#pill-text") == "Listening…",
              page.inner_text("#pill-text"))

        # ---- the resting wall --------------------------------------
        # The appliance has no keyboard, so the status wall cannot live
        # behind a shortcut: it is the screen's resting state. What has to
        # hold is the handoff — anything that wants the monitor takes it,
        # and the wall comes back when that thing is done. Get this wrong
        # and you either lose the wall forever or it covers a running
        # workout.
        print("\n  the resting wall")
        SNAP = {
            "at": 1_760_000_000, "verdict": "warn", "uptime": 900,
            "resting": True,
            "counts": {"ok": 3, "warn": 1, "fail": 0, "skip": 0},
            "groups": ["Hardware", "Services", "Access", "Models",
                       "Storage", "State"],
            "checks": [
                {"name": "printer", "group": "Hardware", "status": "warn",
                 "detail": "low on paper", "fix": "Load a roll.",
                 "tier": "slow", "every": 900, "age": 12, "seconds": 0.1,
                 "runs": 2, "recovered": False, "last_bad": None},
                {"name": "orchestrator", "group": "Services", "status": "ok",
                 "detail": "answering", "fix": "", "tier": "fast",
                 "every": 5, "age": 1, "seconds": 0.01, "runs": 9,
                 "recovered": False, "last_bad": None},
            ],
        }
        SNAP["problems"] = [c for c in SNAP["checks"]
                            if c["status"] in ("warn", "fail")]
        page.evaluate("(d) => { window.lance.renderCheck(d, true); }", SNAP)

        def cls(sel: str) -> str:
            return page.get_attribute(sel, "class") or ""

        check("the wall is on screen", "on" in cls("#check"))
        check("...and knows it is the resting view",
              "resting" in cls("#check"))
        check("...titled as status, not as a one-off check",
              page.inner_text("#check-title") == "Status",
              page.inner_text("#check-title"))
        check("...with the fix under the problem",
              "Load a roll." in page.inner_text("#check-groups"))

        # A resting panel must never time out — there is nothing behind it,
        # and a blank appliance screen is indistinguishable from a dead one.
        page.evaluate("window.lance.hideCheck()")
        check("hideCheck cannot blank the resting wall",
              "on" in cls("#check"))

        # A one-off report is a different thing and must still stand down.
        page.evaluate("(d) => { window.lance.renderCheck(d, false); }", SNAP)
        check("a one-off report is not marked resting",
              "resting" not in cls("#check"))
        check("...and is titled as a check",
              page.inner_text("#check-title") == "Systems check")
        page.evaluate("window.lance.hideCheck()")
        check("...and can be dismissed", "on" not in cls("#check"))

        check("the page threw nothing", not errors, "; ".join(errors))
        browser.close()

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
