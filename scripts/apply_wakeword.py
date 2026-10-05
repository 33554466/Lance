#!/usr/bin/env python3
"""Switch the wake phrase to a model you trained.

Run this AFTER the .onnx from the training notebook is on the box. It does
four things, in this order, and stops at the first one that fails:

  1. finds the model file
  2. LOADS it with openWakeWord and runs a frame of silence through it
  3. only then edits config.yaml
  4. tells you how to check it before you trust it

Step 2 is the point of the script. A wake word that will not load takes the
whole appliance deaf — the audio service raises on startup and nothing is
listening — and you would find that out by talking to a device that does not
answer. Better to find out here, with the old model still in place.

    .venv/bin/python scripts/apply_wakeword.py
    .venv/bin/python scripts/apply_wakeword.py --model models/wake/other.onnx
    .venv/bin/python scripts/apply_wakeword.py --revert

Use the VENV python. The system one does not have openwakeword, and without
it the load test is skipped, which is the only reason this script exists.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path


def _root() -> Path:
    here = Path(__file__).resolve().parent
    for guess in (here, here.parent, Path.cwd(), Path.home() / "assistant"):
        if (guess / "config.yaml").exists() and (guess / "core").is_dir():
            return guess
    return Path.home() / "assistant"


ROOT = _root()
CFG = ROOT / "config.yaml"
DEFAULT = "models/wake/lance_are_you_there.onnx"
GREEN, YELLOW, RED, BOLD, OFF = (
    "\033[1;32m", "\033[1;33m", "\033[1;31m", "\033[1m", "\033[0m")


def ok(m): print(f"  {GREEN}✓{OFF} {m}")
def skip(m): print(f"  {YELLOW}·{OFF} {m}")


def die(m):
    print(f"  {RED}✗{OFF} {m}", file=sys.stderr)
    raise SystemExit(1)


def current(text: str) -> str | None:
    m = re.search(r'(?m)^(  model:\s*)"([^"]*)"', text)
    return m.group(2) if m else None


def set_model(text: str, value: str) -> str:
    new, n = re.subn(r'(?m)^(  model:\s*)"([^"]*)"',
                     lambda m: f'{m.group(1)}"{value}"', text, count=1)
    if n != 1:
        die("could not find the wake_word model line in config.yaml")
    return new


def load_test(path: Path) -> None:
    """Prove the file is a wake model before anything depends on it."""
    try:
        import numpy as np
        from openwakeword.model import Model as OWWModel
    except ImportError as exc:
        skip(f"openwakeword is not importable here ({exc.name}) — "
             f"SKIPPED THE LOAD TEST. Re-run with .venv/bin/python.")
        return
    try:
        model = OWWModel(wakeword_models=[str(path)],
                         inference_framework="onnx")
    except Exception as exc:  # noqa: BLE001
        die(f"that file will not load as a wake model: "
            f"{type(exc).__name__}: {exc}")
    # 80ms of silence, the frame size the listener feeds it.
    try:
        scores = model.predict(np.zeros(1280, dtype=np.int16))
    except Exception as exc:  # noqa: BLE001
        die(f"the model loaded but will not run: {type(exc).__name__}: {exc}")
    keys = list(scores)
    ok(f"loads and runs (scores keyed as {keys!r})")
    if path.stem not in keys:
        skip(f"openWakeWord keys this model {keys!r}, not by its filename "
             f"{path.stem!r}. The listener looks it up by filename stem, so "
             f"rename the file to {keys[0]}.onnx or wakes will never fire.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--revert", action="store_true",
                    help="put hey_jarvis back")
    args = ap.parse_args()

    print(f"\n{BOLD}The wake phrase{OFF}")
    if not CFG.exists():
        die(f"no config.yaml at {CFG}")
    text = CFG.read_text()
    now = current(text)
    ok(f"currently: {now!r}")

    if args.revert:
        if now == "hey_jarvis":
            skip("already back on hey_jarvis")
            return 0
        wanted = "hey_jarvis"
    else:
        path = Path(args.model)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            die(f"no model at {path}\n"
                f"      Train one first — see the notes in the wake_word "
                f"block of config.yaml — then put the .onnx there.")
        size = path.stat().st_size
        ok(f"found {path.relative_to(ROOT)} ({size / 1024:.0f} KB)")
        if size < 10_000:
            skip("that is very small for a wake model — check it downloaded "
                 "completely")
        load_test(path)
        wanted = str(Path(args.model).as_posix())

    if now == wanted:
        skip("config already points at it")
        print()
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = CFG.with_suffix(f".yaml.{stamp}.bak")
    shutil.copy2(CFG, backup)
    ok(f"backup: {backup.name}")
    CFG.write_text(set_model(text, wanted))
    ok(f"config.yaml: model -> {wanted!r}")

    print(f"""
  Now, and in this order — the middle step is the one people skip:

      systemctl --user restart assistant-audio
      journalctl --user -u assistant-audio -n 15 --no-pager

  You want a line saying the custom wake word loaded. If it raises instead,
  nothing is listening: run this script with --revert and tell me the error.

  Then WATCH THE SCORES before trusting it:

      systemctl --user stop assistant-audio
      .venv/bin/python scripts/wake_live.py
      # say it ten times, from where you normally stand
      systemctl --user start assistant-audio

  A phrase that peaks around 0.7-0.9 when you say it and stays near zero
  otherwise is working. If it barely clears the threshold, lower
  wake_word.threshold a little rather than shouting at her.

  After a week of real use:

      python3 scripts/wake_tune.py --days 7

  To go back at any point:

      .venv/bin/python scripts/apply_wakeword.py --revert
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
