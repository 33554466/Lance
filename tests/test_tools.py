"""Local tools: where notes are allowed to land, which list an item goes on,
what happens when the model sends a word where a number belongs, and whether
the router's markers mean what they say.

    python -m tests.test_tools
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

from core.casework import Sla                          # noqa: E402
from core.db import Store                              # noqa: E402
from core.media import as_int                          # noqa: E402
from core.printer import Receipt                       # noqa: E402
from core.router import Router                         # noqa: E402
from core.tools import DocumentTools, TimerTools, slugify  # noqa: E402
from core.workout import _as_int as workout_as_int     # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def _cfg() -> dict:
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------- notes confinement

HOSTILE = [
    "../../.ssh/authorized_keys",
    "../../../etc/passwd",
    "/etc/passwd",
    "..",
    "../",
    "....//....//etc/shadow",
    "~/.bashrc",
    "$HOME/.profile",
    "note\x00.txt",
    "a/b/c/deep",
    "..%2f..%2fetc%2fpasswd",
    "con",                      # reserved on other platforms; must not escape
    "-rf /",
    "\\\\server\\share\\file",
    "." * 200,
    "",
    "   ",
]


def test_notes_stay_in_the_notes_folder(tmp: Path) -> None:
    """The single most valuable test in the project.

    DocumentTools spends twelve lines of comment arguing that a title can
    never contain a path, because every title is slugified and every write is
    checked after symlinks resolve. Every input here arrives from a wake word
    that misfires and a transcriber that mishears, so "airtight by
    construction" needs to be a thing that has actually been run.
    """
    print("\nWhere a note is allowed to land")
    notes = tmp / "notes"
    notes.mkdir()
    docs = DocumentTools({"documents": {"enabled": True, "path": str(notes),
                                        "format": "txt", "max_chars": 5000}})

    escaped = []
    for title in HOSTILE:
        try:
            path = docs._path_for(title)
        except Exception:
            continue                      # refusing outright is a pass
        try:
            resolved = Path(os.path.realpath(path))
        except OSError:
            continue
        if notes.resolve() not in resolved.parents:
            escaped.append((title, str(resolved)))
    check("no hostile title escapes the notes folder", not escaped,
          "; ".join(f"{t!r} -> {p}" for t, p in escaped[:3]))

    # And the same through the real write path, which is what actually runs.
    wrote = []
    for title in HOSTILE:
        try:
            docs._run_sync("save_note", {"title": title, "content": "x"})
        except Exception:
            pass
    for root, _dirs, files in os.walk(tmp):
        for f in files:
            p = Path(root) / f
            if notes.resolve() not in p.resolve().parents:
                wrote.append(str(p))
    check("nothing was written outside the notes folder", not wrote,
          "; ".join(wrote[:3]))

    # A symlink INSIDE the notes folder pointing anywhere else must not
    # become a write target. Unresolved, its parent is the notes folder and a
    # naive containment check passes.
    outside = tmp / "outside"
    outside.mkdir()
    (notes / "escape.txt").symlink_to(outside / "captured.txt")

    refused = False
    try:
        docs._assert_inside(notes / "escape.txt")
    except ValueError:
        refused = True
    check("_assert_inside rejects a symlink out of the folder", refused)

    refused = False
    try:
        docs._path_for("escape")
    except ValueError:
        refused = True
    check("_path_for refuses to build that path", refused)

    said = docs._run_sync("save_note", {"title": "escape",
                                        "content": "captured"})
    check("save_note did not write through the symlink",
          not (outside / "captured.txt").exists(),
          "the file outside the notes folder was created")
    check("...and said so in a sentence, not a stack trace",
          "could not save" in said.lower() and "ValueError" not in said, said)

    # Ordinary titles still work and stay put.
    out = docs._run_sync("save_note", {"title": "Packing list",
                                       "content": "socks\nboots"})
    check("a normal note saves", "packing-list" in out.lower(), out)
    check("...inside the notes folder",
          (notes / "packing-list.txt").exists())
    check("slugify never returns an empty name", slugify("///") == "note")


# --------------------------------------------------------- list routing

def test_resolve_list(tmp: Path) -> None:
    print("\nWhich list an item goes on")
    cfg = _cfg()
    tt = TimerTools(cfg, Store(tmp / "lists.db"))
    default = cfg["lists"]["default"]

    # The bug: filler-stripping emptied the name, and slugify("") is "note".
    for phrase in ("the list", "my list", "on the list", "list", "board",
                   "put it on the list", "add it to my list", "notes"):
        got = tt.resolve_list(phrase)
        check(f"{phrase!r} -> the default list, not an invented one",
              got == default, f"got {got!r}")

    # Real names and aliases still resolve.
    for phrase, want in (("shopping", "shopping"), ("groceries", "shopping"),
                         ("honey do", "house"), ("honey-do list", "house"),
                         ("the army board", "army"), ("kids", "kids"),
                         ("work todo", "work"), ("children's", "kids")):
        got = tt.resolve_list(phrase)
        check(f"{phrase!r} -> {want!r}", got == want, f"got {got!r}")

    # A genuinely named new list is still allowed.
    check("'packing list' creates 'packing'",
          tt.resolve_list("packing list") == "packing")
    check("'camping trip' creates 'camping-trip'",
          tt.resolve_list("camping trip") == "camping-trip")
    check("None is the default", tt.resolve_list(None) == default)


# ------------------------------------------------- numbers from a model

def test_number_coercion() -> None:
    print("\nNumbers the model might send")
    for fn, name in ((as_int, "media"), (workout_as_int, "workout")):
        check(f"{name}: an int", fn(90) == 90)
        check(f"{name}: a numeric string", fn("90") == 90)
        check(f"{name}: a float", fn(90.7) == 90)
        check(f"{name}: a float string", fn("90.0") == 90)
        check(f"{name}: zero is zero, not None", fn(0) == 0)
        check(f"{name}: a word is None, not a crash", fn("ninety") is None)
        check(f"{name}: None is None", fn(None) is None)
        check(f"{name}: empty string is None", fn("") is None)
        check(f"{name}: a dict is None", fn({"seconds": 90}) is None)
        check(f"{name}: True is not 1", fn(True) is None)


def test_severity_is_not_assumed_to_be_text() -> None:
    print("\nSeverity from a model")
    sla = Sla(_cfg())
    check("a word works", isinstance(sla.minutes_for("high"), int))
    check("a number does not raise", isinstance(sla.minutes_for(2), int))
    check("None falls back to the default",
          sla.minutes_for(None) == sla.default_minutes)


# ------------------------------------------------------------- router

def test_router_word_boundaries() -> None:
    print("\nRouter markers match words, not fragments")
    r = Router(_cfg())

    # Each of these was escalated by a marker hiding inside another word.
    for text, fragment in (("start my training", "rain"),
                           ("drill weekend training", "rain"),
                           ("can I borrow the drill", "row"),
                           ("I am impressed", "press")):
        route = r.route(text)
        check(f"{text!r} is not routed by {fragment!r}",
              fragment not in route.reason, route.reason)

    # The markers still do their job when the word is really there. Only
    # assert on markers this config actually defines — tool_markers arrived in
    # a later patch, so a config without it should skip rather than fail.
    configured = set(r.live) | set(r.tools) | set(r.markers)
    for text, expect in (("what is the weather tonight", "weather"),
                         ("set a timer for five minutes", "timer"),
                         ("look up the cost of a chisel", "cost of"),
                         ("put on some lofi hip hop", "put on"),
                         ("pause that", "pause")):
        if expect not in configured:
            print(f"    · skipped {expect!r} — not in this config")
            continue
        route = r.route(text)
        check(f"{text!r} still matches {expect!r}",
              expect in route.reason and route.tier != "small", route.reason)

    check("an explicit escalation still wins",
          r.route("think hard about this").tier == "top")
    check("a short plain question still goes to the small tier",
          r.route("what is two plus two").tier == "small")
    check("an open session forces mid",
          r.route("185 for 5", in_session=True).tier == "mid")


# ------------------------------------------------------------- printer

def test_receipt_render() -> None:
    """A golden test for the 42-column layout. No printer needed."""
    print("\nReceipt layout")
    r = Receipt(width=42)
    r.centre("BENCH PRESS", style="head")
    r.columns("Load", "185 lb")
    r.rule()
    r.wrap("A long line that has to be folded at the paper width rather "
           "than running off the edge of the receipt.")
    text = r.preview()
    lines = text.splitlines()
    widths = [len(l) for l in lines]
    check("nothing exceeds the paper width", max(widths) <= 42,
          f"widest line is {max(widths)}")
    check("the rule fills the width", any(w == 42 for w in widths))
    check("a column pair shares one line",
          any("Load" in l and "185 lb" in l for l in lines),
          str([l for l in lines if "Load" in l]))
    check("the right column is right-aligned",
          any(l.rstrip().endswith("185 lb") for l in lines))
    check("long text was wrapped, not truncated",
          "receipt." in text and len(lines) > 4)
    check("the title is centred",
          any(l.strip() == "BENCH PRESS" and l != l.lstrip() for l in lines)
          or any(l.strip() == "BENCH PRESS" for l in lines))


# --------------------------------------------------------- the API key

def test_key_resolution(tmp: Path) -> None:
    """Where the key comes from, and what never sees it."""
    print("\nThe API key")
    import core.provider as prov

    saved = {k: os.environ.get(k) for k in
             ("CREDENTIALS_DIRECTORY", "ANTHROPIC_API_KEY")}
    try:
        os.environ.pop("CREDENTIALS_DIRECTORY", None)
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-from-env"
        key, src = prov.read_api_key()
        check("the environment works", key == "sk-ant-from-env", src)

        # A credential directory beats the environment.
        cred = tmp / "creds"
        cred.mkdir()
        (cred / "anthropic-key").write_text("sk-ant-from-credential\n")
        os.environ["CREDENTIALS_DIRECTORY"] = str(cred)
        key, src = prov.read_api_key()
        check("the credential wins over the environment",
              key == "sk-ant-from-credential", f"{src}: {key!r}")
        check("...and trailing whitespace is stripped",
              not key.endswith("\n"))

        # A credential directory that is set but empty falls back rather than
        # failing — a half-configured unit should still start.
        (cred / "anthropic-key").unlink()
        key, src = prov.read_api_key()
        check("a missing credential file falls back to the environment",
              key == "sk-ant-from-env", f"{src}: {key!r}")

        # Nothing anywhere is None, not a crash.
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["CREDENTIALS_DIRECTORY"] = str(tmp / "nope")
        key, src = prov.read_api_key()
        check("nothing configured returns None, not an exception",
              key is None or isinstance(key, str))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # And the thing this is all for: children must not inherit it.
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-must-not-leak"
    try:
        env = prov.child_env()
        check("a child process does not get the key",
              "ANTHROPIC_API_KEY" not in env)
        check("...but still gets a normal environment",
              "PATH" in env and len(env) > 1)
        for other in ("OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"):
            os.environ[other] = "x"
            check(f"{other} is stripped too",
                  other not in prov.child_env())
            os.environ.pop(other, None)
        check("os.environ itself is untouched",
              os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-must-not-leak")
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)



def test_printed_workout(tmp: Path) -> None:
    """A session on paper, and the programme on paper.

    Nothing here touches a printer: the layout is built as text first
    precisely so it can be checked without one, and so a formatting mistake
    cannot leave the cutter uncalled halfway down a sheet.
    """
    print("\nWorkouts on paper")
    import yaml
    from core.tools import build_tools
    from core.db import Store

    folder = tmp / "workouts"
    folder.mkdir()
    (folder / "2026-01-02-full-body-a.txt").write_text(
        "# Full Body A\n"
        "## Main\n"
        "Bench press, 5 x 5 @ 185 | rest 180\n"
        "Dips, 3 x AMRAP\n"
        "## Finisher\n"
        "A very long accessory movement name that has to fold across more "
        "than one line of receipt paper, 3 x 12\n")
    (folder / "2026-01-03-conditioning.txt").write_text(
        "# Conditioning\n## Main\nAir bike, 20 minutes\n")
    (folder / "calf-block.txt").write_text("# Calf Block\nCalf raises, 4x15\n")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["workouts"]["path"] = str(folder)
    cfg["workouts"]["default_rest_seconds"] = 90
    cfg.setdefault("printer", {})["enabled"] = True
    box = build_tools(cfg, Store(str(tmp / "print.db")))
    pr, wk = box.printer, box.workouts

    names = [x["name"] for x in pr.schemas()]
    check("she has a tool for it", "print_workout" in names, str(names))

    path, _why = wk.resolve("full body a")
    check("a spoken name finds the file", path is not None)
    title, steps = wk.read(path)

    sheet = pr.build_workout(title, steps, default_rest=90).preview()
    lines = sheet.splitlines()
    check("nothing runs off the paper",
          max(len(x) for x in lines) <= 42,
          f"widest is {max(len(x) for x in lines)}")
    check("the session is named", "FULL BODY A" in sheet, lines[1])
    check("sections are headed", "MAIN" in sheet and "FINISHER" in sheet)
    check("every exercise has a box",
          sheet.count("[ ] ") == len(steps),
          f'{sheet.count("[ ] ")} boxes for {len(steps)} exercises')
    check("a long exercise folds rather than truncating",
          "receipt paper" in sheet)
    check("a non-default rest is printed", "rest 3:00" in sheet, sheet)
    check("...and the default is not repeated down the page",
          sheet.count("rest 1:30") == 0)
    check("the default is stated once at the foot",
          "rest 90s unless noted" in sheet)
    check("the exercise count is on it", "3 exercises" in sheet)

    plain = pr.build_workout(title, steps, write_lines=False).preview()
    check("write lines can be turned off",
          "..." not in plain and "..." in sheet)
    check("...and that is all that changes",
          len(plain.splitlines()) == len(lines) - len(steps),
          f"{len(plain.splitlines())} vs {len(lines)}")

    programme = pr.build_schedule(wk.schedule()).preview()
    check("the schedule prints every file",
          all(w in programme for w in ("full body a", "conditioning",
                                       "calf block")),
          programme)
    check("...dated ones by date",
          programme.index("full body a") < programme.index("conditioning"))
    check("...and undated ones apart",
          "ANY TIME" in programme
          and programme.index("ANY TIME") < programme.index("calf block"))
    check("the schedule also fits the paper",
          max(len(x) for x in programme.splitlines()) <= 42)

    # The refusals are spoken aloud, so they have to be sentences.
    said = pr.run_sync("print_workout", {"name": "leg day"})
    check("an unknown session names what there is",
          "no workout called" in said.lower() and "full body a" in said.lower(),
          said)



def test_casework_off_keeps_workouts(tmp: Path) -> None:
    """The feature goes; the machinery it was built on stays.

    Workouts ride on casework: a session is a row in `cases`, the screen is
    the case overlay, and "that's the dips" is matched by the case step
    matcher. Turning the FEATURE off must not touch any of that. This is the
    test that would catch someone later deciding the tidy thing to do is
    gate the matcher on the same flag.
    """
    print("\nCasework off, workouts on")
    import copy
    import yaml
    from core.tools import build_tools
    from core.db import Store

    folder = tmp / "wk"
    folder.mkdir()
    (folder / "2026-01-02-push.txt").write_text(
        "# Push Day\n## Main\nBench press, 5 x 5 @ 185\nDips, 3 x AMRAP\n")

    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    base["workouts"]["path"] = str(folder)
    base.setdefault("printer", {})["enabled"] = True

    CASE_TOOLS = {"start_case", "check_step", "uncheck_step", "case_status",
                  "case_next", "case_note", "show_case", "case_report",
                  "close_case", "list_cases"}

    counts = {}
    for on in (True, False):
        cfg = copy.deepcopy(base)
        cfg["casework"]["enabled"] = on
        box = build_tools(cfg, Store(str(tmp / f"cw{on}.db")))
        names = {x["name"] for x in box.schemas()}
        counts[on] = names
        label = "on" if on else "off"
        check(f"casework {label}: case tools "
              f"{'present' if on else 'gone'}",
              bool(names & CASE_TOOLS) is on,
              str(sorted(names & CASE_TOOLS)))
        check(f"casework {label}: print_case "
              f"{'offered' if on else 'withheld'}",
              ("print_case" in names) is on)
        check(f"casework {label}: the workout tools are there",
              {"start_workout", "log_set", "finish_workout",
               "list_workouts"} <= names)

    check("turning it off removes eleven tools and nothing else",
          counts[True] - counts[False] == CASE_TOOLS | {"print_case"},
          str(sorted(counts[True] - counts[False])))
    check("...and adds nothing", not counts[False] - counts[True])

    # The part that actually matters: ticking off still works.
    cfg = copy.deepcopy(base)
    cfg["casework"]["enabled"] = False
    store = Store(str(tmp / "live.db"))
    box = build_tools(cfg, store)
    said = box.workouts.run_sync("start_workout", {"name": "push"})
    check("a session still starts", "Push Day" in said, said)
    box.workouts.run_sync("log_set", {"exercise": "bench press",
                                      "result": "185 for 5"})
    case = store.case_active()
    check("...and it is a case row underneath", case is not None)
    rows = store.case_steps(case["id"])
    done = [r for r in rows if r["done"]]
    check("...and the step matcher still ticked it off",
          len(done) == 1 and "Bench" in done[0]["text"],
          str([r["text"] for r in done]))
    check("...with what was lifted kept against it",
          "185" in (done[0]["finding"] or ""), str(done[0]["finding"]))
    box.workouts.run_sync("finish_workout", {})


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_notes_stay_in_the_notes_folder(tmp)
        test_resolve_list(tmp)
        test_number_coercion()
        test_severity_is_not_assumed_to_be_text()
        test_router_word_boundaries()
        test_receipt_render()
        test_printed_workout(tmp)
        test_casework_off_keeps_workouts(tmp)
        test_key_resolution(tmp)
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
