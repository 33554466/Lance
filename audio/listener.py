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

        # How many 80 ms frames in a row must clear the threshold before we
        # believe it. The model already integrates over roughly a second and a
        # half, so a genuine wake word holds its score for several frames
        # while a clatter, a consonant off the television, or a syllable that
        # happens to rhyme spikes for one and falls away.
        #
        # This is a much better lever than the threshold alone: raising the
        # threshold costs you real wakes from across the room, where the score
        # is honestly lower. Requiring the score to PERSIST costs almost
        # nothing, because when you actually say it, it does.
        self.confirm_frames = max(1, int(w.get("confirm_frames", 1)))

        # Hand openWakeWord its own Silero gate. With this set it zeroes any
        # prediction made on audio that is not speech at all, which removes
        # the whole family of false wakes that come from noise rather than
        # from words — a dropped pan, a chair, the extractor fan.
        self.wake_vad = float(w.get("vad_threshold", 0.0) or 0.0)

        self._above = 0          # consecutive frames over the threshold
        self._peak = 0.0         # best score seen during those frames

        v = cfg["vad"]
        self.silence_frames = int(v["silence_ms"] / (VAD_FRAME / self.sr * 1000))
        self.max_frames = int(v["max_utterance_seconds"] * self.sr / VAD_FRAME)
        self.min_speech_frames = int(
            v["min_speech_ms"] / (VAD_FRAME / self.sr * 1000)
        )

        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._snip_buf = np.zeros(0, dtype=np.float32)
        self._snip_audio: list[np.ndarray] = []
        self._snip_speech = 0
        self._snip_silence = 0
        self._last_wake = 0.0
        # Updated on every audio callback. The service watches this to
        # notice a microphone that has stopped delivering audio — muted
        # is fine (silence still streams), unplugged is not.
        self.last_frame_ts = time.time()
        self._stream: sd.InputStream | None = None

        self._load_models(cfg, root)

    # -- model loading ----------------------------------------------

    def _load_models(self, cfg: dict, root: Path) -> None:
        log.info("loading wake word model '%s'", self.wake_name)
        from openwakeword.model import Model as OWWModel
        from openwakeword.utils import download_models

        # Two kinds of value are accepted here. A bare name like
        # "hey_jarvis" is one of the bundled models and gets downloaded on
        # first use. Anything ending in .onnx is a model YOU trained, loaded
        # from disk — which is the only way to get a wake phrase that is not
        # on openWakeWord's short list of pretrained options.
        if str(self.wake_name).endswith(".onnx"):
            path = Path(self.wake_name)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                raise FileNotFoundError(
                    f"Custom wake word model not found at {path}. Train one "
                    f"with openWakeWord's automatic_model_training notebook "
                    f"and drop the .onnx file there."
                )
            self.oww = OWWModel(wakeword_models=[str(path)],
                                vad_threshold=self.wake_vad,
                                inference_framework="onnx")
            # openWakeWord keys its scores by the model's filename stem, not
            # by whatever you called it in config.
            self.wake_key = path.stem
            log.info("custom wake word loaded from %s (key '%s')",
                     path, self.wake_key)
        else:
            download_models(model_names=[self.wake_name])
            self.oww = OWWModel(wakeword_models=[self.wake_name],
                                vad_threshold=self.wake_vad,
                                inference_framework="onnx")
            self.wake_key = self.wake_name

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
            self.last_frame_ts = time.time()
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
        self._snip_buf = np.zeros(0, dtype=np.float32)
        self._reset_snippet()
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                return

    def _reset_snippet(self) -> None:
        self._snip_audio = []
        self._snip_speech = 0
        self._snip_silence = 0

    def snippet(self, max_seconds: float = 2.5, silence_ms: float = 350,
                min_speech_ms: float = 250) -> np.ndarray:
        """Return one short island of speech, or nothing, without waiting.

        This exists so the assistant can be told to shut up WHILE it is
        talking. The obvious implementation — stop playback, listen, decide —
        is unacceptable: it cuts him off every time a chair scrapes. So
        instead we listen underneath our own voice and only stop when a
        dismissal has actually been transcribed. Worst case that is about a
        second of extra talking; the trade is that it never cuts off for
        nothing, which is the failure mode people cannot forgive.

        It is safe to call in a tight loop. Speech state carries across calls,
        so an utterance spanning several calls is assembled properly, and a
        call that finds only silence costs one VAD pass over whatever has
        arrived and returns an empty array.

        Depends entirely on the microphone array's hardware echo
        cancellation, which is why the reference signal has to come back
        through the array's own USB playback endpoint. Everything about this
        function is wrong on a speaker wired to the mini PC's jack.
        """
        sil_frames = max(1, int(silence_ms / (VAD_FRAME / self.sr * 1000)))
        min_frames = max(1, int(min_speech_ms / (VAD_FRAME / self.sr * 1000)))
        max_frames = max(1, int(max_seconds * self.sr / VAD_FRAME))

        # One short blocking read paces the caller's loop; everything else
        # already queued is taken without waiting, so we never fall behind.
        try:
            blocks = [self._audio_q.get(timeout=0.2)]
        except queue.Empty:
            return np.zeros(0, dtype=np.float32)
        while True:
            try:
                blocks.append(self._audio_q.get_nowait())
            except queue.Empty:
                break
        self._snip_buf = np.concatenate([self._snip_buf, *blocks])

        while len(self._snip_buf) >= VAD_FRAME:
            chunk = self._snip_buf[:VAD_FRAME]
            self._snip_buf = self._snip_buf[VAD_FRAME:]

            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            speech = float(self.vad.predict(pcm)) >= 0.5

            if speech:
                self._snip_audio.append(chunk)
                self._snip_speech += 1
                self._snip_silence = 0
            elif self._snip_audio:
                # Keep trailing silence in the clip. Whisper transcribes a
                # word with a little room after it far better than one that
                # ends the instant the speaker does.
                self._snip_audio.append(chunk)
                self._snip_silence += 1
            else:
                continue

            done = (self._snip_silence >= sil_frames
                    or len(self._snip_audio) >= max_frames)
            if not done:
                continue

            clip = (np.concatenate(self._snip_audio) if self._snip_audio
                    else np.zeros(0, dtype=np.float32))
            heard = self._snip_speech
            self._reset_snippet()
            # Too short to be a phrase — a cough, a door, one syllable of
            # our own voice getting past the canceller.
            if heard >= min_frames:
                return clip
            return np.zeros(0, dtype=np.float32)

        return np.zeros(0, dtype=np.float32)

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
        score = float(scores.get(self.wake_key, 0.0))

        if score < self.threshold:
            # One frame under the line ends the run. A wake word that is
            # really being said does not flicker.
            if self._above:
                log.debug("wake candidate lapsed after %d frame(s), peak %.3f",
                          self._above, self._peak)
            self._above = 0
            self._peak = 0.0
            return None

        self._above += 1
        self._peak = max(self._peak, score)
        if self._above < self.confirm_frames:
            return None

        peak, self._above, self._peak = self._peak, 0, 0.0

        now = time.time()
        if now - self._last_wake < self.refractory:
            return None
        self._last_wake = now

        # Reset internal state so the next detection starts clean, otherwise
        # the model can re-fire on the decaying tail of this one.
        self.oww.reset()
        log.info("wake word fired, score=%.3f (held %d frames)",
                 peak, self.confirm_frames)
        return peak

    # -- capture -----------------------------------------------------

    def capture_utterance(self, start_timeout: float | None = None,
                          silence_ms: float | None = None,
                          max_seconds: float | None = None) -> np.ndarray:
        """Record until the speaker stops. Returns float32 mono at 16 kHz.

        start_timeout bounds how long we wait for speech to BEGIN. None means
        wait up to max_utterance_seconds, which is right after a wake word —
        you asked for its attention, so it should be patient. A few seconds is
        right for a follow-up window, where silence means "I'm done".
        """
        # Per-capture overrides exist for dictation: a person composing a
        # sentence out loud pauses far longer than one asking a question, and
        # a single global setting cannot serve both.
        silence_frames = self.silence_frames if silence_ms is None else \
            int(silence_ms / (VAD_FRAME / self.sr * 1000))
        max_frames = self.max_frames if max_seconds is None else \
            int(max_seconds * self.sr / VAD_FRAME)

        collected: list[np.ndarray] = []
        speech_frames = 0
        silence_run = 0
        total_frames = 0
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self.vad.reset_states()
        started = time.time()

        while total_frames < max_frames:
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
                        silence_run >= silence_frames:
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
