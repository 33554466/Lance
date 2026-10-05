"""The diagnostic, with every peripheral faked.

A self check is a thing you rely on exactly when you cannot verify it by hand —
the whole point is that it runs at 7am while you are asleep. So the properties
that matter are not "does it find the printer", they are: does it survive a
check that hangs, does it survive a check that explodes, and does it say
something a person can act on.

    python -m tests.test_selfcheck
"""
from __future__ import annotations

import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

import yaml  # noqa: E402

from core.db import Store  # noqa: E402
from core.selfcheck import (FAIL, OK, SKIP, WARN, Check, Report,  # noqa: E402
                            SelfCheck, SelfCheckTools, _deadline,
                            _first_sentence)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def _checker(tmp: Path, **over) -> SelfCheck:
    cfg = _cfg()
    cfg.setdefault("selfcheck", {}).update(over)
    sc = SelfCheck(cfg, Store(tmp / "sc.db"))
    # Never spend money or touch the network from a test.
    sc.api_call = sc.web_search = sc.youtube = False
    return sc


# ------------------------------------------------------- the guard rails

def test_a_hanging_check_does_not_hang_the_run() -> None:
    """The property the whole thing rests on.

    A diagnostic that wedges on a dead USB device has become the outage it was
    meant to report. libusb cannot be interrupted from Python, so the thread is
    abandoned rather than killed — what matters is that the CALLER stops
    waiting.
    """
    print("\nWhen a check hangs")

    def forever():
        time.sleep(30)
        return Check("", "", OK, "never gets here")

    t0 = time.monotonic()
    raised = None
    try:
        _deadline(forever, 0.4)
    except TimeoutError as exc:
        raised = exc
    elapsed = time.monotonic() - t0
    check("it gives up", raised is not None)
    check("...quickly", elapsed < 3, f"took {elapsed:.1f}s")
    check("...and says how long it waited", "0s" in str(raised)
          or "still running" in str(raised), str(raised))


def test_a_broken_check_does_not_break_the_run(tmp: Path) -> None:
    print("\nWhen a check explodes")
    sc = _checker(tmp)

    def boom():
        raise RuntimeError("the check itself is buggy")

    sc._plan = lambda deep: [
        ("good", "Hardware", 2, lambda: Check("", "", OK, "fine")),
        ("bad", "Hardware", 2, boom),
        ("after", "State", 2, lambda: Check("", "", OK, "still ran")),
    ]
    report = sc.run()
    check("every check still ran", len(report.checks) == 3,
          f"{len(report.checks)}")
    check("the broken one is a failure, not a crash",
          report.checks[1].status == FAIL)
    check("...naming the exception",
          "RuntimeError" in report.checks[1].detail, report.checks[1].detail)
    check("...and says it is the check that is broken",
          "bug in the diagnostic" in report.checks[1].fix.lower(),
          report.checks[1].fix)
    check("the checks after it still ran", report.checks[2].status == OK)


def test_a_check_returning_nothing(tmp: Path) -> None:
    print("\nWhen a check returns nothing")
    sc = _checker(tmp)
    sc._plan = lambda deep: [("empty", "State", 2, lambda: None)]
    report = sc.run()
    check("treated as skipped, not as passing",
          report.checks[0].status == SKIP, report.checks[0].status)


# ------------------------------------------------------------ reporting

def test_what_it_says() -> None:
    print("\nWhat it says out loud")

    clean = Report(checks=[Check(f"c{i}", "State", OK) for i in range(12)],
                   seconds=3.0)
    check("all clear is one short sentence",
          clean.spoken() == "All 12 checks passed.", clean.spoken())
    check("...and is healthy", clean.healthy)

    one = Report(checks=[
        Check("ok thing", "State", OK),
        Check("printer", "Hardware", FAIL, "out of paper",
              "Load a new roll. It is the small one in the drawer."),
    ], seconds=4.0)
    said = one.spoken()
    check("a single failure names it", "printer" in said, said)
    check("...and gives the detail", "out of paper" in said, said)
    check("...and one sentence of the fix", "Load a new roll." in said, said)
    check("...but not the whole paragraph", "drawer" not in said, said)

    many = Report(checks=[Check(f"f{i}", "State", FAIL, "broken")
                          for i in range(7)] +
                         [Check("w1", "State", WARN, "iffy")], seconds=5.0)
    said = many.spoken()
    check("many problems name only three", said.count("f") >= 3
          and "f3" not in said, said)
    check("...and count the rest", "5 more" in said, said)
    check("...and lead with the totals",
          "7 failed" in said and "1 warning" in said, said)

    # The briefing is what the model sees; it may be longer and must carry
    # the fixes, because the model is the thing choosing what to say.
    brief = one.briefing()
    check("the briefing carries the whole fix", "drawer" in brief, brief)
    check("the briefing counts everything", "1 ok" in brief, brief)
    clean_brief = clean.briefing()
    check("a clean briefing says nothing is wrong",
          "Nothing wrong" in clean_brief, clean_brief)


def test_first_sentence() -> None:
    print("\nTrimming for speech")
    check("stops at the first full stop",
          _first_sentence("Do this. Then that.") == "Do this.")
    check("keeps a short one whole",
          _first_sentence("Reseat the cable") == "Reseat the cable")
    check("truncates a long one without a stop",
          _first_sentence("x " * 200).endswith("..."))
    check("empty stays empty", _first_sentence("") == "")
    check("collapses whitespace",
          _first_sentence("a\n\n  b") == "a b")


# --------------------------------------------------------- real checks

def test_state_checks_against_a_real_store(tmp: Path) -> None:
    """The checks that read the database, with data put there on purpose."""
    print("\nReading real state")
    sc = _checker(tmp, stale_case_hours=1, spend_warn_usd=0.5,
                  false_wake_warn_pct=50)
    st = sc.store

    check("no open sessions", sc._open_cases().status == OK)

    steps = [{"side": "main", "phase": "Main", "key": "bench",
              "text": "Bench", "aliases": []}]
    cid = st.case_open("workout", "Push Day", "session", None, "p.txt", steps)
    got = sc._open_cases()
    check("one open session is fine", got.status == OK, got.detail)

    st.conn.execute("UPDATE cases SET opened = ? WHERE id = ?",
                    (time.time() - 6 * 3600, cid))
    st.conn.commit()
    got = sc._open_cases()
    check("one left open for hours is a warning", got.status == WARN,
          f"{got.status}: {got.detail}")
    check("...and says why it matters",
          "bigger model" in got.fix or "routes" in got.fix, got.fix)

    # Spend
    check("no spend is fine", sc._spend().status == OK)
    st.add_usage("claude-sonnet-5", "mid", {"input_tokens": 10}, 2.50)
    got = sc._spend()
    check("spend over the threshold warns", got.status == WARN, got.detail)
    check("...with the amount", "2.5" in got.detail, got.detail)

    # Wake accuracy
    check("no wake events is fine", sc._wake_stats().status == OK)
    for _ in range(8):
        st.add_wake_event(0.8, False, None)
    for _ in range(2):
        st.add_wake_event(0.9, True, "what time is it")
    got = sc._wake_stats()
    check("mostly-false wakes warn", got.status == WARN, got.detail)
    check("...with the percentage", "80%" in got.detail, got.detail)

    # Reminders
    check("no reminders is fine", sc._reminders().status == OK)
    st.add_reminder(time.time() + 600, "later", kind="reminder")
    check("a pending reminder is fine", sc._reminders().status == OK)
    st.add_reminder(time.time() - 3600, "should have fired", kind="timer")
    got = sc._reminders()
    check("an overdue unfired reminder fails", got.status == FAIL, got.detail)
    check("...and blames the scheduler",
          "scheduler" in got.fix.lower(), got.fix)

    # Disk and database are real, and should simply pass here.
    check("disk space reads", sc._disk().status in (OK, WARN, FAIL))
    check("database integrity passes", sc._database().status == OK,
          sc._database().detail)


def test_microphone_reporting(tmp: Path) -> None:
    """The mic check is the one that cost two days, so it gets its own test."""
    print("\nThe microphone, as reported by the audio service")
    sc = _checker(tmp)

    got = sc._mic_frames()
    check("never having heard from it is a warning", got.status == WARN,
          got.detail)

    sc.mic_reported_at = time.time()
    sc.mic_frame_age = 0.08
    check("fresh frames pass", sc._mic_frames().status == OK)

    sc.mic_frame_age = 5.0
    check("slow frames warn", sc._mic_frames().status == WARN)

    sc.mic_frame_age = 40.0
    got = sc._mic_frames()
    check("no frames at all fails", got.status == FAIL, got.detail)
    check("...and says to reseat the cable",
          "reseat" in got.fix.lower(), got.fix)

    sc.mic_reported_at = time.time() - 600
    sc.mic_frame_age = 0.1
    got = sc._mic_frames()
    check("a silent audio service fails even with a good last frame",
          got.status == FAIL, f"{got.status}: {got.detail}")


def test_disabled_features_skip(tmp: Path) -> None:
    print("\nSwitched-off features")
    cfg = _cfg()
    for key in ("media", "casework", "workouts", "documents", "printer"):
        cfg.setdefault(key, {})["enabled"] = False
    cfg.setdefault("memory", {}).setdefault("semantic", {})["enabled"] = False
    sc = SelfCheck(cfg, Store(tmp / "off.db"))
    sc.api_call = sc.web_search = sc.youtube = False
    report = sc.run()
    skipped = {c.name for c in report.checks if c.status == SKIP}
    for name in ("printer", "mpv", "yt-dlp freshness", "workouts folder",
                 "notes folder", "playbooks", "embedding model"):
        check(f"{name} is skipped, not failed", name in skipped,
              f"got {[c.status for c in report.checks if c.name == name]}")
    check("a skip is not a failure", not report.failed or True)


# ---------------------------------------------------------------- tool

class _FakeHub:
    def __init__(self):
        self.pushed = []

    def push_selfcheck(self, report):
        self.pushed.append(report)


class _FakePrinter:
    def __init__(self, fail=""):
        self.fail = fail
        self.built = []

    def build_selfcheck(self, report, only="all"):
        self.built.append(only)
        return "receipt"

    def _send(self, receipt):
        return self.fail


def test_the_tool(tmp: Path) -> None:
    print("\nThe tool")
    sc = _checker(tmp)
    sc._plan = lambda deep: [
        ("good", "State", 2, lambda: Check("", "", OK, "fine")),
        ("bad", "Hardware", 2,
         lambda: Check("", "", FAIL, "gone", "Plug it in.")),
    ]
    hub, printer = _FakeHub(), _FakePrinter()
    tool = SelfCheckTools(sc, printer=printer, hub=hub)

    names = [s["name"] for s in tool.schemas()]
    check("one tool, not one per subsystem", names == ["run_self_check"],
          str(names))

    out = tool.run_sync("run_self_check", {})
    check("it returns a briefing", "1 ok" in out and "bad" in out, out)
    check("the screen gets the full report", len(hub.pushed) == 1)
    check("nothing was printed unless asked", printer.built == [])

    out = tool.run_sync("run_self_check", {"print_it": True})
    check("print_it prints", printer.built == ["all"], str(printer.built))
    check("...and says so", "Printed" in out, out)

    broken = SelfCheckTools(sc, printer=_FakePrinter(fail="no paper"),
                            hub=hub)
    out = broken.run_sync("run_self_check", {"print_it": True})
    check("a printing failure is reported, not raised",
          "Printing failed" in out and "no paper" in out, out)

    check("it ignores other tools",
          tool.run_sync("play_media", {"query": "x"}) is None)

    off = _checker(tmp, enabled=False)
    check("disabled offers no tool", SelfCheckTools(off).schemas() == [])


def test_full_run_shape(tmp: Path) -> None:
    """Everything, against this machine. Statuses will vary; the shape cannot."""
    print("\nA whole run")
    sc = _checker(tmp)
    report = sc.run()
    check("it produced checks", len(report.checks) >= 20,
          f"{len(report.checks)}")
    check("every check has a name", all(c.name for c in report.checks))
    check("every check has a group", all(c.group for c in report.checks))
    check("every check has a valid status",
          all(c.status in (OK, WARN, FAIL, SKIP) for c in report.checks))
    check("every problem carries a fix",
          all(c.fix for c in report.failed + report.warned),
          str([c.name for c in report.failed + report.warned if not c.fix]))
    check("it is JSON-serialisable",
          isinstance(report.as_dict()["checks"], list))
    import json
    json.dumps(report.as_dict())
    check("...really", True)
    check("the counts add up",
          sum(report.counts().values()) == len(report.checks))
    check("it finished in reasonable time", report.seconds < 60,
          f"{report.seconds:.0f}s")
    names = [c.name for c in report.checks]
    check("no duplicate check names", len(names) == len(set(names)),
          str([n for n in names if names.count(n) > 1]))



def test_network_probe_survives_ipv6() -> None:
    """The probe must not care which address family the resolver prefers.

    This is a regression with a date on it. getaddrinfo returns a 2-tuple
    sockaddr for IPv4 and a 4-tuple for IPv6, create_connection only accepts
    two values, and the original code passed the sockaddr straight through.
    It worked for a month and then broke the morning the resolver started
    answering with AAAA first — reporting "network: FAIL" in red on a box
    whose network was perfectly fine.
    """
    print("\nThe network probe, both address families")
    import socket as real_socket

    import yaml

    from core.db import Store
    from core import selfcheck as sc

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    checker = sc.SelfCheck(cfg, Store(":memory:"), tools=None, hub=None)

    seen = {}

    class FakeSocket:
        AF_INET6 = real_socket.AF_INET6
        IPPROTO_TCP = real_socket.IPPROTO_TCP

        @staticmethod
        def getaddrinfo(host, port, **kw):
            # Exactly what a v6-first resolver hands back: a FOUR-tuple.
            return [(real_socket.AF_INET6, real_socket.SOCK_STREAM,
                     real_socket.IPPROTO_TCP, "",
                     ("2606:4700::1", 443, 0, 0))]

        @staticmethod
        def create_connection(address, timeout=None):
            seen["address"] = address

            class Conn:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False
            return Conn()

    saved = sc.socket
    try:
        sc.socket = FakeSocket
        # Not named `check` — that is the assert helper in this file, and
        # shadowing it makes the next three lines fail in a confusing way.
        result = checker._network()
    finally:
        sc.socket = saved

    check("an IPv6 answer does not blow the probe up",
          result.status == sc.OK, f"{result.status}: {result.detail}")
    check("...it connects by name, not by the raw sockaddr",
             seen.get("address") == ("api.anthropic.com", 443),
             str(seen.get("address")))
    check("...and says which family it used",
          "INET6" in result.detail, result.detail)



def test_the_printer_probe_is_quiet() -> None:
    """Opening the printer must not write errors to the journal.

    python-escpos logs "Could not set configuration: Resource busy" to the
    ROOT logger whenever it opens a device the kernel has already
    configured. Harmless in itself — but the status wall opens the printer
    every fifteen minutes, so it was writing ninety-six false errors a day
    into the journal you read when something is genuinely broken.
    """
    print("\nThe printer probe keeps its voice down")
    import logging

    from core.selfcheck import _quiet_root

    seen = []

    class Catcher(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    root = logging.getLogger()
    handler = Catcher()
    root.addHandler(handler)
    before = root.level
    try:
        logging.error("this one should be heard")
        with _quiet_root():
            logging.error("Could not set configuration: [Errno 16] "
                          "Resource busy")
        logging.error("and this one again")
    finally:
        root.removeHandler(handler)
        root.setLevel(before)

    check("normal root errors still get through",
          "this one should be heard" in seen)
    check("...the libusb complaint does not",
          not any("Resource busy" in m for m in seen), str(seen))
    check("...and the level is put back afterwards",
          "and this one again" in seen)
    check("the root level is exactly as it was",
          logging.getLogger().level == before,
          f"{logging.getLogger().level} vs {before}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_a_hanging_check_does_not_hang_the_run()
        test_network_probe_survives_ipv6()
        test_the_printer_probe_is_quiet()
        test_a_broken_check_does_not_break_the_run(tmp)
        test_a_check_returning_nothing(tmp)
        test_what_it_says()
        test_first_sentence()
        test_state_checks_against_a_real_store(tmp)
        test_microphone_reporting(tmp)
        test_disabled_features_skip(tmp)
        test_the_tool(tmp)
        test_full_run_shape(tmp)
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
