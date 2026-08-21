"""Wake word, endpointing, and transcription.

The chain is: continuous 16 kHz capture -> openWakeWord -> Silero VAD ->
faster-whisper. Everything here runs on the appliance and nothing leaves it.
Only the resulting transcript is ever sent to the cloud, which is the single
best property of this architecture and it costs nothing.
"""
from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger("assistant.listener")

WAKE_FRAME = 1280        # openWakeWord expects 80 ms at 16 kHz
VAD_FRAME = 480          # 30 ms at 16 kHz — openWakeWord's VAD frame size


class Listener:
    """Blocking, thread-friendly. Drive it by calling `poll()` in a loop."""

    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.sr = cfg["audio"]["sample_rate"]
        self.input_device = cfg["audio"].get("input_device")

        w = cfg["wake_word"]
        self.threshold = w["threshold"]
        self.refractory = w["refractory_seconds"]
        self.wake_name = w["model"]

        v = cfg["vad"]
        self.silence_frames = int(v["silence_ms"] / (VAD_FRAME / self.sr * 1000))
        self.max_frames = int(v["max_utterance_seconds"] * self.sr / VAD_FRAME)
        self.min_speech_frames = int(
            v["min_speech_ms"] / (VAD_FRAME / self.sr * 1000)
        )

        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._last_wake = 0.0
        self._stream: sd.InputStream | None = None

        self._load_models(cfg, root)

    # -- model loading ----------------------------------------------

    def _load_models(self, cfg: dict, root: Path) -> None:
        log.info("loading wake word model '%s'", self.wake_name)
        from openwakeword.model import Model as OWWModel
        from openwakeword.utils import download_models

        download_models(model_names=[self.wake_name])
        self.oww = OWWModel(wakeword_models=[self.wake_name],
                            inference_framework="onnx")

        # Silero VAD, via openWakeWord's ONNX wrapper rather than the
        # silero-vad package. That package imports torchaudio at module
        # level, which drags in PyTorch — and pip resolves torchaudio from
        # PyPI (a CUDA build) even when torch came from the CPU index, so
        # you end up with a CPU torch and a CUDA torchaudio that cannot
        # load libcudart. Using the ONNX model directly removes PyTorch,
        # torchaudio and roughly 200 MB from this install, and removes that
        # entire class of bug permanently.
        log.info("loading Silero VAD (ONNX, no PyTorch)")
        from openwakeword.vad import VAD
        self.vad = VAD()

        s = cfg["stt"]
        log.info("loading faster-whisper '%s' (%s/%s)",
                 s["model"], s["device"], s["compute_type"])
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(
            s["model"], device=s["device"], compute_type=s["compute_type"],
            download_root=str(root / "models" / "whisper"),
        )
        self.beam_size = s.get("beam_size", 1)
        self.language = s.get("language", "en")
        self.min_logprob = s.get("min_logprob", -1.0)
        log.info("models ready")

    # -- stream -----------------------------------------------------

    def start(self) -> None:
        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                log.debug("input status: %s", status)
            try:
                self._audio_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                # Dropping a frame is strictly better than growing an
                # unbounded backlog and drifting further behind real time.
                pass

        self._stream = sd.InputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            blocksize=WAKE_FRAME, device=self.input_device,
            callback=callback,
        )
        self._stream.start()
        log.info("input stream open at %d Hz on device %s",
                 self.sr, self.input_device if self.input_device is not None else "<default>")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def drain(self) -> None:
        """Throw away buffered audio. Called after speaking so the assistant
        does not transcribe the tail of its own reply."""
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                return

    # -- wake word ---------------------------------------------------

    def poll_wake(self, timeout: float = 0.5) -> float | None:
        """Consume one frame. Returns the score if the wake word fired."""
        try:
            frame = self._audio_q.get(timeout=timeout)
        except queue.Empty:
            return None

        # openWakeWord wants int16
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        scores = self.oww.predict(pcm)
        score = float(scores.get(self.wake_name, 0.0))

        if score < self.threshold:
            return None

        now = time.time()
        if now - self._last_wake < self.refractory:
            return None
        self._last_wake = now

        # Reset internal state so the next detection starts clean, otherwise
        # the model can re-fire on the decaying tail of this one.
        self.oww.reset()
        log.info("wake word fired, score=%.3f", score)
        return score

    # -- capture -----------------------------------------------------

    def capture_utterance(self, start_timeout: float | None = None) -> np.ndarray:
        """Record until the speaker stops. Returns float32 mono at 16 kHz.

        start_timeout bounds how long we wait for speech to BEGIN. None means
        wait up to max_utterance_seconds, which is right after a wake word —
        you asked for its attention, so it should be patient. A few seconds is
        right for a follow-up window, where silence means "I'm done".
        """
        collected: list[np.ndarray] = []
        speech_frames = 0
        silence_run = 0
        total_frames = 0
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self.vad.reset_states()
        started = time.time()

        while total_frames < self.max_frames:
            try:
                block = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                if time.time() - started > 3:
                    break
                continue

            collected.append(block)
            self._vad_buf = np.concatenate([self._vad_buf, block])

            while len(self._vad_buf) >= VAD_FRAME:
                chunk = self._vad_buf[:VAD_FRAME]
                self._vad_buf = self._vad_buf[VAD_FRAME:]
                total_frames += 1

                # The VAD wants int16 PCM; sounddevice hands us float32.
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                prob = float(self.vad.predict(pcm))

                if prob >= 0.5:
                    speech_frames += 1
                    silence_run = 0
                else:
                    silence_run += 1

                # Nobody started talking within the window — bail quietly.
                if (start_timeout is not None and speech_frames == 0
                        and time.time() - started > start_timeout):
                    return np.zeros(0, dtype=np.float32)

                # Only start counting silence once we have heard something,
                # so a slow start does not end the utterance before it begins.
                if speech_frames >= self.min_speech_frames and \
                        silence_run >= self.silence_frames:
                    log.info("endpoint after %.1fs (%d speech frames)",
                             time.time() - started, speech_frames)
                    return np.concatenate(collected) if collected else np.zeros(0, np.float32)

        if speech_frames < self.min_speech_frames:
            log.info("no meaningful speech after wake (%d frames)", speech_frames)
            return np.zeros(0, dtype=np.float32)

        log.info("hit max utterance length")
        return np.concatenate(collected) if collected else np.zeros(0, np.float32)

    # -- transcription -----------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < self.sr * 0.3:
            return ""

        t0 = time.time()
        segments, info = self.whisper.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=False,          # we already endpointed with Silero
            condition_on_previous_text=False,
        )

        parts, logprobs = [], []
        for seg in segments:
            parts.append(seg.text)
            logprobs.append(seg.avg_logprob)

        text = " ".join(p.strip() for p in parts).strip()
        if not text:
            return ""

        # Whisper confabulates confident-sounding text out of silence and
        # noise — "Thank you." and "Thanks for watching!" are the classic
        # artefacts. Dropping low-confidence output is what stops the
        # assistant answering questions nobody asked.
        mean_lp = sum(logprobs) / len(logprobs) if logprobs else -99.0
        if mean_lp < self.min_logprob:
            log.info("discarded low-confidence transcript (%.2f): %r",
                     mean_lp, text[:60])
            return ""

        log.info("transcribed in %.2fs (logprob %.2f): %s",
                 time.time() - t0, mean_lp, text)
        return text
