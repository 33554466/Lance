"""Text to speech with barge-in.

Piper is invoked as a subprocess writing raw PCM to stdout. One sentence is
synthesised, resampled if the output device demands it, then played in small
blocks so barge-in can cut it mid-word.

The perceived-latency win comes from core/app.py chunking the reply into
sentences as it streams, not from streaming within a sentence — Piper
synthesises a sentence in a couple of hundred milliseconds, so buffering one
costs nothing and avoids carrying resampler state across block boundaries.
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

log = logging.getLogger("assistant.speaker")


def _find_piper() -> str:
    """Locate the piper binary without depending on PATH.

    Under systemd the virtualenv is never "activated" — the unit runs
    .venv/bin/python directly, which is correct, but means .venv/bin is not
    on PATH. A bare subprocess call to "piper" then fails with ENOENT even
    though it is installed. Look beside the interpreter that is running us,
    which is right whether or not anyone activated anything.
    """
    beside_interpreter = Path(sys.executable).parent / "piper"
    if beside_interpreter.exists():
        return str(beside_interpreter)
    on_path = shutil.which("piper")
    if on_path:
        return on_path
    raise FileNotFoundError(
        f"piper not found next to {sys.executable} or on PATH. "
        f"Install it with: pip install piper-tts"
    )


class Speaker:
    def __init__(self, cfg: dict, root: Path):
        t = cfg["tts"]
        self.voice = (root / t["voice"]).resolve()
        self.length_scale = t.get("length_scale", 1.0)
        self.output_device = cfg["audio"].get("output_device")

        if not self.voice.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {self.voice}. "
                f"Run scripts/fetch_models.sh first."
            )

        # Piper ships a sidecar JSON with the model's native sample rate.
        # Guessing this wrong makes the voice sound like a chipmunk or a
        # ghost, which is a memorable way to lose an evening.
        cfg_path = Path(str(self.voice) + ".json")
        if cfg_path.exists():
            with open(cfg_path) as fh:
                self.sample_rate = json.load(fh)["audio"]["sample_rate"]
        else:
            log.warning("no %s — assuming 22050 Hz", cfg_path.name)
            self.sample_rate = 22050

        # The ReSpeaker XVF3800 is a fixed 16 kHz device — its entire DSP
        # pipeline runs at 16 k, so its USB playback endpoint rejects
        # anything else. Piper's medium-quality voices are 22,050 Hz. Rather
        # than force a lower-quality 16 kHz voice, work out what the device
        # will actually accept and resample into it.
        # Resolve now, so a missing binary is a loud startup failure rather
        # than a traceback after every single reply.
        self.piper_bin = _find_piper()
        log.info("piper: %s", self.piper_bin)

        self.out_rate = self._pick_output_rate()
        if self.out_rate != self.sample_rate:
            log.info("output device wants %d Hz, voice is %d Hz — resampling",
                     self.out_rate, self.sample_rate)

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- rate negotiation --------------------------------------------

    def _pick_output_rate(self) -> int:
        """Return a sample rate this output device will actually open at."""
        try:
            sd.check_output_settings(device=self.output_device,
                                     samplerate=self.sample_rate,
                                     channels=1, dtype="int16")
            return self.sample_rate
        except Exception:
            pass
        try:
            info = sd.query_devices(self.output_device, "output")
            rate = int(info["default_samplerate"])
            sd.check_output_settings(device=self.output_device, samplerate=rate,
                                     channels=1, dtype="int16")
            return rate
        except Exception:
            log.warning("could not negotiate an output rate; assuming 16 kHz")
            return 16000

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        if self.out_rate == self.sample_rate:
            return audio
        out = resample_poly(audio.astype(np.float32),
                            self.out_rate, self.sample_rate)
        return np.clip(out, -32768, 32767).astype(np.int16)

    # -- public ------------------------------------------------------

    def say(self, text: str) -> None:
        """Queue a sentence. Returns immediately."""
        if text.strip():
            self._queue.put(text.strip())

    def stop(self) -> None:
        """Barge-in: drop everything queued and cut the current utterance."""
        self._stop.set()
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            log.info("barge-in dropped %d queued sentence(s)", drained)

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def wait_until_idle(self, timeout: float = 90.0) -> bool:
        """Block until nothing is queued and nothing is playing.

        Needed for the follow-up window: we must not reopen the microphone
        while the assistant is still talking, or it transcribes itself.
        """
        import time as _t
        deadline = _t.time() + timeout
        while _t.time() < deadline:
            if self._queue.empty() and not self._speaking.is_set():
                return True
            _t.sleep(0.05)
        return False

    def close(self) -> None:
        self._queue.put(None)

    # -- worker ------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self._stop.clear()
            try:
                self._speak_one(item)
            except Exception:  # noqa: BLE001
                log.exception("TTS failure on: %.60s", item)
            finally:
                self._speaking.clear()

    def _speak_one(self, text: str) -> None:
        self._speaking.set()
        proc = subprocess.Popen(
            [
                self.piper_bin,
                "--model", str(self.voice),
                "--length_scale", str(self.length_scale),
                "--output-raw",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(text.encode("utf-8") + b"\n")
        proc.stdin.close()

        # Read the whole sentence, then resample once. Streaming a resampler
        # across block boundaries needs careful state handling for no real
        # gain: Piper synthesises a sentence in a couple of hundred
        # milliseconds, and core/app.py already chunks the reply per
        # sentence, which is where the perceived-latency win actually comes
        # from.
        raw = proc.stdout.read()
        proc.wait(timeout=5)
        if not raw:
            log.warning("piper produced no audio for: %.50s", text)
            return

        audio = self._resample(np.frombuffer(raw, dtype=np.int16))

        stream = sd.RawOutputStream(
            samplerate=self.out_rate,
            channels=1,
            dtype="int16",
            device=self.output_device,
            blocksize=1024,
        )
        stream.start()
        try:
            for i in range(0, len(audio), 1024):
                if self._stop.is_set():
                    stream.abort()
                    break
                stream.write(audio[i:i + 1024].tobytes())
        finally:
            stream.stop()
            stream.close()


def play_chime(sample_rate: int = 16000, device=None) -> None:
    """A short two-tone acknowledgement, synthesised rather than shipped as
    a file. It fires the instant the wake word is detected, before
    transcription begins — which is the point. It tells you the device heard
    you, so you do not repeat yourself into a system that was already
    listening."""
    try:
        def tone(freq: float, ms: int) -> np.ndarray:
            n = int(sample_rate * ms / 1000)
            t = np.linspace(0, ms / 1000, n, endpoint=False)
            wave = np.sin(2 * np.pi * freq * t)
            # Short fades stop the click you would otherwise get at the
            # discontinuity on each end.
            fade = max(1, n // 12)
            env = np.ones(n)
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
            return (wave * env * 0.25).astype(np.float32)

        sig = np.concatenate([tone(880, 60), tone(1320, 70)])
        sd.play(sig, samplerate=sample_rate, device=device, blocking=True)
    except Exception:  # noqa: BLE001
        log.debug("chime failed", exc_info=True)
