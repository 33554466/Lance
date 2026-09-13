"""Wake-word confirmation and sleep-phrase matching.

Both of these are decided in the audio process with no network and no model,
which is what makes them instant — and also what makes them worth testing,
because there is nothing downstream to catch a mistake. Run it:

    python -m tests.test_wake_sleep
"""
from __future__ import annotations

import queue
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# sounddevice needs a sound card present just to import. Nothing under test
# touches it, so stub it rather than requiring hardware to run the tests.
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))
sys.modules.setdefault("websockets", types.ModuleType("websockets"))

from audio.listener import Listener, VAD_FRAME, WAKE_FRAME  # noqa: E402
from audio.service import AudioService, normalise, strip_fillers  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


# ------------------------------------------------------------- dismissal

PHRASES = [
    "stop", "stop it", "cancel", "quiet", "be quiet", "never mind",
    "forget it", "thats all", "that is all", "thats enough", "im done",
    "were done", "go to sleep", "go back to sleep", "goodnight",
    "nothing else", "no thanks", "no more", "stop talking", "thank you thats all",
]


def _svc() -> AudioService:
    """An AudioService with only the fields the matcher touches."""
    svc = AudioService.__new__(AudioService)
    svc.stop_phrases = {normalise(p) for p in PHRASES}
    svc.stop_fillers = {"lance"}
    return svc


def test_dismissal() -> None:
    print("\nSleep phrases")
    svc = _svc()

    def says(text: str) -> bool:
        return svc.is_dismissal(normalise(text))

    # The bare phrases, as transcribed by Whisper — punctuated and capitalised.
    check("'Go to sleep.'", says("Go to sleep."))
    check("'Stop!'", says("Stop!"))
    check("'That's all.'", says("That's all."))

    # What people ACTUALLY say. Every one of these failed before.
    check("'Okay Lance, go to sleep'", says("Okay Lance, go to sleep"))
    check("'Alright, go to sleep now please'",
          says("Alright, go to sleep now please"))
    check("'Lance, that's enough.'", says("Lance, that's enough."))
    check("'Yeah, that's all thanks'", says("Yeah, that's all thanks"))
    check("'Go to sleep, Lance.'", says("Go to sleep, Lance."))
    check("'Um, stop.'", says("Um, stop."))
    check("'No thanks.'", says("No thanks."))

    # And the whole reason matching stays exact at the core.
    check("'Stop by the store on the way home' does NOT dismiss",
          not says("Stop by the store on the way home"))
    check("'Remind me to go to sleep at ten' does NOT dismiss",
          not says("Remind me to go to sleep at ten"))
    check("'Cancel my three o'clock' does NOT dismiss",
          not says("Cancel my three o'clock"))
    check("'Is it quiet in there' does NOT dismiss",
          not says("Is it quiet in there"))
    check("empty transcript does NOT dismiss", not says(""))

    # Stripping must never leave nothing and call that a match.
    check("'okay thanks' alone does NOT dismiss", not says("okay thanks"))
    check("'Lance' alone does NOT dismiss", not says("Lance"))
    check("'um, uh, yeah' does NOT dismiss", not says("um, uh, yeah"))

    print("\nFiller stripping")
    check("both ends trimmed",
          strip_fillers("okay go to sleep please") == "go to sleep")
    check("interior words untouched",
          strip_fillers("stop by the store") == "stop by the store")
    check("all-filler collapses to empty", strip_fillers("okay um yeah") == "")


# --------------------------------------------------------- wake confirm

class _FakeOWW:
    def __init__(self, scores: list[float]):
        self.scores = list(scores)
        self.resets = 0

    def predict(self, pcm):  # noqa: ARG002
        return {"hey_jarvis": self.scores.pop(0) if self.scores else 0.0}

    def reset(self):
        self.resets += 1


def _listener(scores: list[float], confirm: int) -> Listener:
    lis = Listener.__new__(Listener)
    lis.threshold = 0.6
    lis.refractory = 0.0
    lis.confirm_frames = confirm
    lis.wake_key = "hey_jarvis"
    lis.oww = _FakeOWW(scores)
    lis._above = 0
    lis._peak = 0.0
    lis._last_wake = 0.0
    lis._audio_q = queue.Queue()
    for _ in scores:
        lis._audio_q.put(np.zeros(WAKE_FRAME, dtype=np.float32))
    return lis


def _fire(lis: Listener, n: int) -> list[float]:
    return [s for s in (lis.poll_wake(timeout=0.01) for _ in range(n))
            if s is not None]


def test_confirmation() -> None:
    print("\nWake confirmation")

    # A single frame spiking over the line — a clatter, a syllable off the
    # television. This is the false wake we are trying to kill.
    spike = [0.2, 0.9, 0.1, 0.1]
    check("one loud frame fires with confirm_frames=1",
          _fire(_listener(spike, 1), 4) == [0.9])
    check("one loud frame is ignored with confirm_frames=2",
          _fire(_listener(spike, 2), 4) == [])

    # Actually saying it: the score holds across several frames.
    said = [0.2, 0.72, 0.88, 0.81, 0.1]
    fired = _fire(_listener(said, 2), 5)
    check("a held wake word still fires", len(fired) == 1)
    check("it reports the peak, not the first frame",
          fired == [0.88], str(fired))

    # A run that lapses and restarts must not accumulate across the gap.
    stutter = [0.7, 0.1, 0.7, 0.1]
    check("a broken run does not add up", _fire(_listener(stutter, 2), 4) == [])

    # Without the reset the model re-fires on the decaying tail of its own
    # detection, which reads as the wake word triggering twice in a row.
    lis = _listener(said, 2)
    _fire(lis, 5)
    check("openWakeWord state is reset on a real fire", lis.oww.resets == 1)


# --------------------------------------------------------------- snippet

class _FakeVAD:
    """Speech wherever the sample value is 1.0."""

    def predict(self, pcm):
        return 1.0 if float(np.max(np.abs(pcm))) > 0.5 else 0.0

    def reset_states(self):
        pass


def _snip_listener(pattern: list[bool]) -> Listener:
    """pattern: one bool per 30 ms VAD frame, True meaning speech."""
    lis = Listener.__new__(Listener)
    lis.sr = 16000
    lis.vad = _FakeVAD()
    lis._snip_buf = np.zeros(0, dtype=np.float32)
    lis._snip_audio = []
    lis._snip_speech = 0
    lis._snip_silence = 0
    lis._audio_q = queue.Queue()
    samples = np.concatenate([
        np.full(VAD_FRAME, 32000.0 if on else 0.0, dtype=np.float32)
        for on in pattern
    ])
    lis._audio_q.put(samples)
    return lis


def test_snippet() -> None:
    print("\nInterrupt listening")

    # Silence only. The common case while a reply plays, and it has to cost
    # nothing — no clip, so no transcription.
    lis = _snip_listener([False] * 40)
    check("silence returns nothing", lis.snippet().size == 0)

    # A real short phrase: ~600 ms of speech, then a pause.
    lis = _snip_listener([True] * 20 + [False] * 20)
    clip = lis.snippet()
    check("a spoken phrase comes back", clip.size > 0, f"size {clip.size}")
    check("it is long enough for Whisper", clip.size >= 16000 * 0.3,
          f"{clip.size / 16000:.2f}s")

    # A cough: one frame of noise. Must not be transcribed.
    lis = _snip_listener([True] * 2 + [False] * 30)
    check("a cough is discarded", lis.snippet().size == 0)

    # Continuous speech longer than the cap gets cut at the cap rather than
    # growing forever — a radio left on must not build an unbounded buffer.
    lis = _snip_listener([True] * 200)
    clip = lis.snippet(max_seconds=2.5)
    check("long speech is capped", 0 < clip.size <= 16000 * 2.6,
          f"{clip.size / 16000:.2f}s")

    # State survives across calls, since a phrase rarely lands in one block.
    lis = Listener.__new__(Listener)
    lis.sr, lis.vad = 16000, _FakeVAD()
    lis._snip_buf = np.zeros(0, dtype=np.float32)
    lis._snip_audio, lis._snip_speech, lis._snip_silence = [], 0, 0
    lis._audio_q = queue.Queue()
    speech = np.full(VAD_FRAME * 12, 32000.0, dtype=np.float32)
    quiet = np.zeros(VAD_FRAME * 20, dtype=np.float32)
    lis._audio_q.put(speech)
    first = lis.snippet()
    lis._audio_q.put(quiet)
    second = lis.snippet()
    check("a phrase split across calls is assembled",
          first.size == 0 and second.size > 0,
          f"{first.size} then {second.size}")


class _BrokenListener:
    """A listener whose snippet() explodes, as the real one did."""

    def __init__(self):
        self.calls = 0

    def snippet(self, **kw):
        self.calls += 1
        raise AttributeError("'Listener' object has no attribute '_snip_buf'")


def test_dismissal_failure_is_contained() -> None:
    """A broken interrupt-listener must not end the conversation.

    This is the bug that made follow-up questions stop working: snippet()
    raised, the listener loop's catch-all caught it and restarted the outer
    loop, and the whole exchange was abandoned. Nothing in the symptom pointed
    at the cause.
    """
    print("\nWhen the interrupt listener is broken")
    svc = _svc()
    svc.dismiss_while_speaking = True
    svc.dismiss_max_seconds = 2.5
    svc.listener = _BrokenListener()

    raised = None
    try:
        heard = svc._heard_dismissal()
    except Exception as exc:  # noqa: BLE001
        raised = exc
        heard = None

    check("the failure does not escape", raised is None, repr(raised))
    check("it reports 'no dismissal' rather than blowing up", heard is False)
    check("it switches itself off after one failure",
          svc.dismiss_while_speaking is False)

    # Second call must not even try — one fault should not cost a call per
    # reply for the rest of the day.
    svc._heard_dismissal()
    check("it does not try again", svc.listener.calls == 1,
          f"called {svc.listener.calls} times")


class _OldListener:
    """A listener from before snippet() existed."""


class _OldSpeaker:
    is_speaking = False


class _NewListener:
    def snippet(self, **kw):
        return np.zeros(0, dtype=np.float32)


class _NewSpeaker:
    is_speaking = False
    is_busy = False


def _service_with(listener, speaker, cfg_on=True) -> AudioService:
    """Run just the compatibility check from __init__, nothing else."""
    svc = AudioService.__new__(AudioService)
    svc.listener, svc.speaker = listener, speaker
    svc.dismiss_while_speaking = cfg_on
    missing = [name for obj, name in ((svc.listener, "snippet"),
                                      (svc.speaker, "is_busy"))
               if not hasattr(obj, name)]
    if missing:
        svc.dismiss_while_speaking = False
    svc._missing = missing
    return svc


def test_version_skew() -> None:
    """A new service.py beside an old listener.py must degrade, not crash.

    This is the bug that ate an afternoon: audio_recovery_v1 shipped a
    service.py that called listener.snippet(), while the archive carrying that
    method had never been extracted. Every reply raised, the listener loop
    caught it and started over, and the conversation was abandoned — so it
    looked like follow-up questions were broken.
    """
    print("\nWhen the audio files are from different archives")

    old = _service_with(_OldListener(), _OldSpeaker())
    check("an old listener is noticed", "snippet" in old._missing)
    check("an old speaker is noticed", "is_busy" in old._missing)
    check("interrupting is switched off rather than attempted",
          old.dismiss_while_speaking is False)

    new = _service_with(_NewListener(), _NewSpeaker())
    check("a matched set is left alone", new._missing == []
          and new.dismiss_while_speaking is True)

    # And the playback check has to work either way.
    svc = AudioService.__new__(AudioService)
    svc.speaker = _OldSpeaker()
    check("_still_playing works without is_busy",
          svc._still_playing() is False)
    svc.speaker = _NewSpeaker()
    svc.speaker.is_busy = True
    check("_still_playing prefers is_busy", svc._still_playing() is True)
    svc.speaker.is_busy = False
    svc.speaker.is_speaking = True
    check("...and is_busy False means finished even while is_speaking lags",
          svc._still_playing() is False)


def main() -> int:
    test_dismissal()
    test_dismissal_failure_is_contained()
    test_version_skew()
    test_confirmation()
    test_snippet()
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
