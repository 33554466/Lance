#!/usr/bin/env python3
"""Watch the wake-word score in real time, with and without the VAD gate.

Run this when the wake word has stopped working. It answers, in about ten
seconds, the question every other check dances around: is the model scoring
your voice at all?

It loads TWO copies of the wake model — one exactly as config.yaml asks for it,
one with the VAD gate switched off — and prints both scores side by side. If
the ungated column lights up when you speak and the configured column stays at
zero, the VAD gate is what broke it, and nothing else needs investigating.

The audio service holds the microphone, so stop it first:

    systemctl --user stop assistant-audio
    python3 scripts/wake_live.py
    systemctl --user start assistant-audio      # when you are done
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WAKE_FRAME = 1280        # 80 ms at 16 kHz, what openWakeWord expects
BAR = 28


def load_cfg() -> dict:
    import yaml
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def build(name: str, vad_threshold: float, root: Path):
    """One openWakeWord model. Returns (model, score_key) or (None, reason)."""
    from openwakeword.model import Model
    from openwakeword.utils import download_models

    if str(name).endswith(".onnx"):
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            return None, f"custom model not found at {path}"
        target, key = str(path), path.stem
    else:
        download_models(model_names=[name])
        target, key = name, name

    try:
        model = Model(wakeword_models=[target], vad_threshold=vad_threshold,
                      inference_framework="onnx")
    except TypeError as exc:
        # An older openWakeWord that does not know the argument. Worth saying
        # plainly, because the symptom of guessing is a service that will not
        # start at all.
        return None, (f"this openWakeWord does not accept vad_threshold "
                      f"({exc}). Set wake_word.vad_threshold to 0 in "
                      f"config.yaml.")
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    return (model, key), None


def meter(score: float, threshold: float) -> str:
    filled = int(round(min(1.0, max(0.0, score)) * BAR))
    mark = int(round(threshold * BAR))
    cells = []
    for i in range(BAR):
        if i == mark:
            cells.append("|")
        elif i < filled:
            cells.append("#")
        else:
            cells.append(".")
    return "".join(cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0,
                    help="stop after this long (default: until Ctrl-C)")
    args = ap.parse_args()

    cfg = load_cfg()
    w = cfg["wake_word"]
    name = w["model"]
    threshold = float(w["threshold"])
    configured_vad = float(w.get("vad_threshold", 0) or 0)
    confirm = int(w.get("confirm_frames", 1) or 1)

    print(f"\n  model            {name}")
    print(f"  threshold        {threshold}")
    print(f"  vad_threshold    {configured_vad}")
    print(f"  confirm_frames   {confirm}")

    print("\n  loading two copies of the model...")
    gated, err = build(name, configured_vad, ROOT)
    if err:
        print(f"  configured copy FAILED: {err}")
    plain, err2 = build(name, 0.0, ROOT)
    if err2:
        print(f"  ungated copy FAILED: {err2}")
        return 1
    if gated is None and plain is None:
        return 1

    import numpy as np
    import sounddevice as sd

    q: queue.Queue = queue.Queue(maxsize=64)

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    device = cfg["audio"].get("input_device")
    try:
        stream = sd.InputStream(samplerate=cfg["audio"]["sample_rate"],
                                channels=1, dtype="float32",
                                blocksize=WAKE_FRAME, device=device, callback=cb)
        stream.start()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Could not open {device!r}: {exc}")
        print("  If this says the device is busy, the audio service still has "
              "it:\n      systemctl --user stop assistant-audio")
        return 1

    print(f"\n  Listening on {device!r}. Say the wake word a few times.")
    print(f"  '|' marks the threshold. LEVEL is how loud the room is.\n")
    print(f"  {'configured':^30}  {'vad gate off':^30}  level")

    started = time.time()
    peak_gated = peak_plain = 0.0
    fires = 0
    above = 0
    try:
        while True:
            if args.seconds and time.time() - started > args.seconds:
                break
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                print("  no audio arriving — the microphone is not streaming")
                continue

            rms = float(np.sqrt(np.mean(frame ** 2)))
            pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)

            g = 0.0
            if gated is not None:
                g = float(gated[0].predict(pcm).get(gated[1], 0.0))
            p = float(plain[0].predict(pcm).get(plain[1], 0.0))
            peak_gated = max(peak_gated, g)
            peak_plain = max(peak_plain, p)

            # Replicate the real confirmation rule, so what prints is what the
            # service would actually have done with the same audio.
            if g >= threshold:
                above += 1
                if above == confirm:
                    fires += 1
                    print(f"  >>> WOULD FIRE  score {g:.3f}"
                          f"   (fire #{fires})")
                    if gated is not None:
                        gated[0].reset()
                    above = 0
            else:
                above = 0

            if rms > 0.004 or g > 0.05 or p > 0.05:
                print(f"  {meter(g, threshold)} {g:5.3f}  "
                      f"{meter(p, threshold)} {p:5.3f}  {rms:.3f}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

    print(f"\n  peak with the config as written: {peak_gated:.3f}")
    print(f"  peak with the VAD gate off:      {peak_plain:.3f}")
    print(f"  times it would have fired:       {fires}")

    print()
    if peak_plain < 0.2:
        print("  VERDICT: the model never scored your voice, gate or no gate.\n"
              "  This is not the threshold and not the VAD — the audio itself\n"
              "  is wrong. Check the level column above: if it stayed near\n"
              "  zero while you were talking, the array is muted (its own\n"
              "  button) or wedged. If the level moved but the score did not,\n"
              "  the wrong device is being captured.")
    elif peak_gated < 0.2 <= peak_plain:
        print("  VERDICT: the VAD gate is suppressing everything.\n"
              "  Set wake_word.vad_threshold to 0 in config.yaml and restart:\n"
              "      systemctl --user restart assistant-audio\n"
              "  That is my bug, not your setup.")
    elif fires == 0 and peak_gated >= threshold:
        print(f"  VERDICT: it scored above {threshold} but never held for\n"
              f"  {confirm} frames in a row. Set wake_word.confirm_frames to 1.")
    elif fires == 0:
        print(f"  VERDICT: your best score was {peak_gated:.3f}, under the\n"
              f"  {threshold} threshold. Lower it to about "
              f"{max(0.3, round(peak_gated - 0.1, 2))} and try again —\n"
              f"  or move closer and re-run, to see whether it is distance.")
    else:
        print("  VERDICT: detection is working. It fired "
              f"{fires} time(s) with these exact settings, so if the\n"
              "  assistant still does nothing, the problem is downstream —\n"
              "  transcription or the orchestrator, not the wake word.\n"
              "      journalctl --user -u assistant -n 50 --no-pager")
    return 0


if __name__ == "__main__":
    sys.exit(main())
