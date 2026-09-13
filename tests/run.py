#!/usr/bin/env python3
"""Run every suite and sum the failures.

There was no runner, so the README named only test_pipeline and the three
newer suites were invisible — which is how a test file rots. Anything matching
tests/test_*.py is picked up automatically, so a new suite needs no wiring.

    .venv/bin/python -m tests.run
    .venv/bin/python -m tests.run --quiet      # one line per suite

Exit code is the number of suites that failed, so it works in a shell guard:

    .venv/bin/python -m tests.run && git commit ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

BOLD, GREEN, RED, DIM, OFF = ("\033[1m", "\033[1;32m", "\033[1;31m",
                              "\033[2m", "\033[0m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="one line per suite instead of full output")
    ap.add_argument("only", nargs="*",
                    help="suite names to run, e.g. store tools")
    args = ap.parse_args()

    suites = sorted(p.stem for p in HERE.glob("test_*.py"))
    if args.only:
        wanted = {f"test_{n}" if not n.startswith("test_") else n
                  for n in args.only}
        missing = wanted - set(suites)
        if missing:
            print(f"No such suite: {', '.join(sorted(missing))}")
            print(f"Available: {', '.join(s[5:] for s in suites)}")
            return 1
        suites = [s for s in suites if s in wanted]

    print(f"\n{BOLD}{len(suites)} suite(s){OFF}  "
          f"{DIM}{sys.executable}{OFF}")

    failed, results = [], []
    for suite in suites:
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", f"tests.{suite}"],
                              cwd=ROOT,
                              capture_output=args.quiet, text=True)
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        if not ok:
            failed.append(suite)
        results.append((suite, ok, elapsed))
        if args.quiet:
            mark = f"{GREEN}pass{OFF}" if ok else f"{RED}FAIL{OFF}"
            print(f"  {mark}  {suite:22} {elapsed:5.1f}s")
            if not ok:
                # Only the failures are worth reading when running quiet.
                tail = (proc.stdout or "").strip().splitlines()[-12:]
                for line in tail:
                    print(f"        {line}")
                if proc.stderr:
                    for line in proc.stderr.strip().splitlines()[-6:]:
                        print(f"        {line}")

    print(f"\n{BOLD}{'-' * 46}{OFF}")
    for suite, ok, elapsed in results:
        mark = f"{GREEN}✓{OFF}" if ok else f"{RED}✗{OFF}"
        print(f"  {mark} {suite:24} {elapsed:5.1f}s")
    total = sum(e for _, _, e in results)
    if failed:
        print(f"\n{RED}{len(failed)} of {len(suites)} suites failed{OFF}: "
              f"{', '.join(failed)}   {DIM}({total:.1f}s){OFF}\n")
    else:
        print(f"\n{GREEN}all {len(suites)} suites passed{OFF}   "
              f"{DIM}({total:.1f}s){OFF}\n")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
