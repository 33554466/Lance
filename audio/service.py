"""The audio service — owns the sound devices, talks to the orchestrator.

This process never touches the network beyond a localhost WebSocket. Keeping
all audio in Python and none of it in the kiosk browser is the single most
valuable structural decision in the build: no getUserMedia permission flags,
no autoplay policy to fight, no browser holding the sound device hostage, and
every audio bug is a Python bug rather than a Chromium bug.

Run it directly to test:  python -m audio.service
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path

import yaml
import websockets

from .listener import Listener
from .speaker import Speaker, play_chime
from .usbpower import power_cycle as usb_power_cycle
from .usbreset import reset as usb_reset

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("assistant.audio")


def resolve_devices(cfg: dict) -> None:
    """Turn device NAMES from config.yaml into PortAudio indices, once, before
    anything opens a stream. Rewrites cfg in place.

    Two reasons this is a separate up-front step rather than letting each
    component match by name when it needs to:

      * PortAudio enumerates devices when it initialises and interleaving
        name lookups with stream opens has proven fragile — a failed
        check_output_settings() early on can leave the input list unmatched
        later, which surfaces as a baffling "No input device matching X"
        for a device that plainly exists.
      * Resolving once means one clear error at startup listing every device
        it CAN see, instead of a stack trace from inside a worker thread.

    Indices are stable for the life of the process. Names are still what you
    put in config.yaml, because indices shift on re-enumeration.
    """
    import sounddevice as sd

    # Force a clean enumeration before we look at anything.
    try:
        sd._terminate()
    except Exception:  # noqa: BLE001
        pass
    sd._initialize()

    devices = sd.query_devices()

    def find(name, kind: str):
        if name is None or isinstance(name, int):
            return name
        key = "max_input_channels" if kind == "input" else "max_output_channels"
        candidates = [(i, d) for i, d in enumerate(devices) if d[key] > 0]
        for i, d in candidates:                      # exact
            if d["name"] == name:
                return i
        for i, d in candidates:                      # substring, case-insensitive
            if name.lower() in d["name"].lower():
                return i
        listing = "\n".join(f"    [{i}] {d['name']}" for i, d in candidates)

        # PortAudio does not enumerate a device that it cannot open, so a
        # BUSY device looks exactly like a MISSING one. Cross-check against
        # ALSA, which lists cards regardless of who holds them, and say which
        # of the two it actually is — those need completely different fixes.
        alsa_has_it = False
        try:
            with open("/proc/asound/cards") as fh:
                alsa_has_it = name.lower() in fh.read().lower()
        except OSError:
            pass

        if alsa_has_it:
            raise SystemExit(
                f"\n{name!r} exists in ALSA but something already has it open,"
                f"\nso PortAudio cannot see it. This is almost always another"
                f"\ncopy of the audio service — a manual run alongside the"
                f"\nsystemd unit, or one left behind by a crash.\n\n"
                f"    systemctl --user stop assistant-audio\n"
                f"    pkill -f audio.service\n"
                f"    arecord -l | grep -A1 {name}   # want Subdevices: 1/1\n"
                f"    systemctl --user start assistant-audio\n\n"
                f"If that does not clear it:  sudo fuser -v /dev/snd/*\n"
            )

        raise SystemExit(
            f"\nNo {kind} device matching {name!r}, and ALSA does not list it"
            f"\neither — so it is genuinely absent, not busy. Check the USB"
            f"\nconnection (lsusb) and `sudo dmesg | tail` for disconnects.\n\n"
            f"Available {kind} devices:\n{listing}\n"
            f"Set audio.{kind}_device to one of those names."
        )

    a = cfg["audio"]

    # Remember the NAMES on first call, and resolve from them every time after.
    #
    # This function runs again on every recovery attempt. The old version
    # finished by writing the resolved INDEX back over the name — and `find()`
    # returns an int unchanged — so every later call skipped enumeration
    # entirely and reused a number. That is exactly backwards from what the
    # docstring above promises, and it failed in the worst possible way:
    # when the array dropped off the USB bus, index 4 had become the mini PC's
    # onboard analog codec, so the service bound to the wrong sound card,
    # failed forever on sample rate, and could not recover even after the
    # microphone was plugged back in.
    a.setdefault("input_name", a.get("input_device"))
    a.setdefault("output_name", a.get("output_device"))
    in_idx = find(a["input_name"], "input")
    out_idx = find(a["output_name"], "output")
    a["input_device"], a["output_device"] = in_idx, out_idx

    def _describe(idx) -> str:
        return devices[idx]["name"] if idx is not None else "<default>"

    log.info("audio in  -> [%s] %s", in_idx, _describe(in_idx))
    log.info("audio out -> [%s] %s", out_idx, _describe(out_idx))

    # Belt and braces on top of the fix. If what we resolved does not look
    # like what was asked for, say so loudly rather than quietly recording
    # somebody's onboard line-in for the rest of the day.
    for label, want, idx in (("input", a["input_name"], in_idx),
                             ("output", a["output_name"], out_idx)):
        if isinstance(want, str) and idx is not None \
                and want.lower() not in _describe(idx).lower():
            log.error("%s device %r resolved to %r, which does not match. "
                      "Refusing to use it.", label, want, _describe(idx))
            raise SystemExit(
                f"\nconfig.yaml asks for the {label} device {want!r} but the "
                f"closest match is\n  {_describe(idx)!r}, which is a "
                f"different piece of hardware.\n\nThis usually means the "
                f"microphone array is not on the USB bus:\n"
                f"    lsusb | grep 2886:001a\n\n"
                f"Unplug it, wait fifteen seconds, plug it back in.\n")


def probe_capture(cfg: dict, timeout: float = 4.0) -> bool:
    """Can we actually read audio, right now?

    Enumeration is not health. After a warm reboot the array enumerates,
    snd-usb-audio binds, `arecord -l` lists the card — and every read either
    returns EIO or, worse, simply never delivers a frame. Nothing short of
    pulling audio off the device distinguishes the two.

    This must never block. The first version of this function called
    stream.read(), which waits for frames that a wedged device will never
    send: the service sat in it for five minutes holding /dev/snd/pcmC0D0c
    open, so it never reached the recovery code AND made every other tool
    report the device as busy rather than broken. A health check that can
    hang is worse than no health check at all.

    So: open with a callback, wait on an Event with a deadline, and run the
    whole thing on a daemon thread so that even a hang inside PortAudio's
    own close path cannot stop the service from moving on.
    """
    import sounddevice as sd

    got_audio = threading.Event()
    finished = threading.Event()
    outcome: dict = {}

    def worker() -> None:
        def callback(indata, frames, time_info, status):  # noqa: ARG001
            # Any frame at all is proof of life. Silence is fine; we are
            # testing the transport, not whether anyone is talking.
            got_audio.set()

        try:
            with sd.InputStream(samplerate=cfg["audio"]["sample_rate"],
                                channels=1, dtype="float32", blocksize=1024,
                                device=cfg["audio"].get("input_device"),
                                callback=callback):
                got_audio.wait(timeout)
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            finished.set()

    threading.Thread(target=worker, daemon=True).start()

    # Give the worker the probe window plus a little room to tear down.
    if not finished.wait(timeout + 2.0):
        log.warning("capture probe hung — the device is enumerated but is "
                    "not delivering audio, and will not close cleanly")
        return False
    if "error" in outcome:
        log.warning("capture probe failed (%s)", outcome["error"])
        return False
    if not got_audio.is_set():
        log.warning("capture probe opened the device but no audio arrived "
                    "in %.1fs", timeout)
        return False
    return True


# How many times to run the reset ladder before giving up on software and
# simply waiting. See the note inside ensure_capture_works().
MAX_RESET_ATTEMPTS = 3
_recovery_attempts = 0


def _try_resolve(cfg: dict) -> bool:
    """resolve_devices(), but a missing device is False rather than fatal.

    resolve_devices raises SystemExit when it cannot find what config.yaml
    asks for, which is right when the service is starting up and wrong once it
    is running: an unplugged microphone is a WAITING problem. Exiting hands it
    to systemd, which restarts every few seconds, and every restart reloads
    Whisper — a CPU-burning loop that fixes nothing and stops the service
    being there when somebody finally plugs the cable back in.
    """
    try:
        resolve_devices(cfg)
        return True
    except SystemExit as exc:
        log.error("%s", exc)
        return False


def ensure_capture_works(cfg: dict) -> bool:
    """Resolve devices, check we can capture, and if not, escalate.

    The escalation ladder, cheapest first:

        probe -> power cycle the port (VBUS actually drops) -> probe
              -> USBDEVFS_RESET (protocol reset only)       -> probe

    Note the order. The old code reset unconditionally at startup, which cost
    every healthy boot two seconds and re-enumeration for nothing. Probing
    first means the recovery path only runs when there is something to
    recover from — and it tells the log which boots were wedged, which is the
    number worth watching.
    """
    if not _try_resolve(cfg):
        return False
    if probe_capture(cfg):
        log.info("capture healthy")
        return True

    vid_pid = cfg["audio"].get("usb_reset")
    if not vid_pid:
        return False

    # Stop hammering after a few goes. Measured on this build: repeated
    # USBDEVFS_RESETs eventually knock the array off the bus ALTOGETHER —
    # the log shows a reset at 13:38:57, a re-enumeration two seconds later,
    # and by 13:39:35 no such device to reset at all. Past a few attempts the
    # resets are not recovery, they are the problem, and the right behaviour
    # is to keep probing patiently until somebody reseats the cable.
    global _recovery_attempts
    _recovery_attempts += 1
    if _recovery_attempts > MAX_RESET_ATTEMPTS:
        log.warning("not resetting again (tried %d times) — still waiting for "
                    "the microphone to be reseated by hand",
                    _recovery_attempts - 1)
        return False

    log.warning("microphone is enumerated but will not stream — recovering "
                "(attempt %d of %d)", _recovery_attempts, MAX_RESET_ATTEMPTS)

    if usb_power_cycle(vid_pid,
                       off_seconds=cfg["audio"].get("usb_power_off_seconds", 3)):
        if _try_resolve(cfg) and probe_capture(cfg):
            log.info("recovered by power cycling the port")
            _recovery_attempts = 0
            return True

    if usb_reset(vid_pid):
        if _try_resolve(cfg) and probe_capture(cfg):
            log.info("recovered by USB reset")
            _recovery_attempts = 0
            return True

    log.error(
        "could not revive the microphone in software.\n"
        "    Measured on this build: USBDEVFS_RESET, authorized-toggling, "
        "driver unbind/rebind and uhubctl power cycling ALL fail. Only "
        "physically unplugging it for ~10s works, because the XMOS "
        "processor only restarts when VBUS actually goes away, and no reset "
        "the host can issue takes VBUS away.\n"
        "    UNPLUG THE MICROPHONE'S USB CABLE, WAIT TEN SECONDS, PLUG IT "
        "BACK IN. This service is still running and will pick it up on its "
        "own within a minute — you do not need to restart anything.")
    return False


# How often to tell the orchestrator the microphone is still delivering
# frames. Ten seconds is frequent enough that a self check always has a fresh
# number, and rare enough to be invisible.
HEARTBEAT_SECONDS = 10.0

_PUNCT = str.maketrans("", "", ".,!?;:\"'’")


def normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Whisper punctuates and capitalises freely — "Stop." and "stop" and
    "Stop!" are all the same intent and must all match.
    """
    return " ".join(text.lower().translate(_PUNCT).split())


# Words that carry no meaning at the edges of a dismissal. "Okay Lance, go
# to sleep now please" and "go to sleep" are the same instruction, and a
# person who has just been told the first one worked will say the second one
# next time — which is how you end up with a device that obeys sometimes.
#
# These are stripped from the ENDS only, and only ever leaving a phrase that
# is on the list in full. "Stop by the store on the way home" still survives
# untouched, because "by the store on the way home" is not a dismissal.
_FILLERS = frozenset("""
ok okay okey alright allright right yeah yep yup yes no nope
um uh er erm hmm mm well so just then now please thanks thank
hey hi hello mate sir buddy man dude actually anyway alright
""".split())


def strip_fillers(norm: str, extra: frozenset[str] | set[str] = frozenset()) -> str:
    """Trim padding words from both ends of an already-normalised phrase."""
    fillers = _FILLERS | set(extra)
    words = norm.split()
    while words and words[0] in fillers:
        words.pop(0)
    while words and words[-1] in fillers:
        words.pop()
    return " ".join(words)


class AudioService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = (f"ws://{cfg['server']['host']}:{cfg['server']['port']}"
                    f"/ws/audio")

        # Devices are resolved and verified before anything else is built,
        # so the rest of the process can assume a working sound card.
        #
        # If recovery fails we wait and try again rather than exiting. Exiting
        # would hand the problem to systemd, which restarts us every three
        # seconds, and each restart reloads Whisper — a CPU-burning loop that
        # fixes nothing. Waiting here also means that when somebody does walk
        # over and reseat the cable, the assistant comes back by itself
        # instead of needing to be told to.
        attempt = 0
        while not ensure_capture_works(cfg):
            attempt += 1
            wait = min(30 * attempt, 300)
            log.warning("no working microphone (attempt %d) — retrying in %ds",
                        attempt, wait)
            threading.Event().wait(wait)
        self.listener = Listener(cfg, ROOT)
        self.speaker = Speaker(cfg, ROOT)
        b = cfg["behaviour"]
        self.ack_chime = b.get("ack_chime", True)
        self.barge_in = b.get("barge_in", True)
        self.mute_while_speaking = b.get("mute_while_speaking", True)
        self.follow_up_seconds = float(b.get("follow_up_seconds", 0))

        # Dismissal is handled here, in the audio process, and never reaches
        # the network. "Stop" that takes two seconds and a round trip to a
        # data centre is not stopping. This costs nothing and always works,
        # including when the API is down or the reply is halfway through.
        #
        # Matching is EXACT after normalisation, deliberately. Substring
        # matching would make "stop by the store on the way home" end the
        # conversation, and a dismissal that fires when you did not mean it
        # is far more annoying than one you have to repeat.
        self.stop_phrases = {
            normalise(p) for p in b.get("stop_phrases", []) if p.strip()
        }
        self.stop_chime = b.get("stop_chime", True)

        # The assistant's own name is padding inside a dismissal — "Lance,
        # that's enough" is not a different instruction from "that's enough" —
        # even though it is anything but padding everywhere else.
        self.stop_fillers = {normalise(w) for w in b.get("stop_fillers", [])}
        self.stop_fillers.add(normalise(cfg["identity"]["name"]))
        self.stop_fillers.discard("")

        # Listen for a dismissal while a reply is playing. Without this,
        # "go to sleep" works in the follow-up window and does nothing at all
        # mid-sentence — which reads as unreliable rather than as a gap, and
        # mid-sentence is exactly when you want it most.
        self.dismiss_while_speaking = bool(b.get("dismiss_while_speaking", True))

        # Check, ONCE and at startup, that the other files in this package are
        # the versions this one expects.
        #
        # This is not paranoia. The appliance is assembled from a dozen
        # archives applied over months, and it is entirely possible to end up
        # with a new service.py beside an old listener.py. When that happened
        # the symptom was an AttributeError raised on every reply, caught by
        # the listener loop's catch-all, which abandoned the conversation — so
        # it presented as "follow-up questions do not work" and named nothing
        # about the real problem. A missing method is knowable at startup;
        # discovering it once per utterance is a choice, and the wrong one.
        missing = [name for obj, name in ((self.listener, "snippet"),
                                          (self.speaker, "is_busy"))
                   if not hasattr(obj, name)]
        if missing:
            log.error(
                "this audio/service.py expects %s, which this build of the "
                "other audio files does not have. Interrupting a reply by "
                "voice is switched off; everything else — wake word, "
                "conversation, follow-up questions, saying 'stop' in the gap "
                "after a reply — works normally. Install the archive that "
                "ships audio/listener.py and audio/speaker.py alongside this "
                "file to get it back.", " and ".join(missing))
            self.dismiss_while_speaking = False
        self.dismiss_max_seconds = float(b.get("dismiss_max_seconds", 2.5))

        # How long to wait, after the chime, for you to start saying
        # something. Nothing by then means the wake word fired on its own —
        # and sitting there with the microphone open for twenty seconds is
        # what makes an occasional false trigger feel constant.
        self.wake_speech_timeout = float(
            cfg["wake_word"].get("speech_timeout_seconds", 0) or 0) or None

        # Dictation. A person composing a sentence out loud pauses far
        # longer than one asking a question, so a single endpoint setting
        # cannot serve both. These phrases switch to patient endpointing for
        # exactly one utterance, then it reverts.
        self.dictation_phrases = {
            normalise(p) for p in b.get("dictation_phrases", []) if p.strip()
        }
        self.dictation_silence_ms = float(b.get("dictation_silence_ms", 1800))
        self.dictation_max_seconds = float(b.get("dictation_max_seconds", 120))

        # Repeat. Answered from the last reply already in memory: no API
        # call, no latency, no cost, and it works with the network down.
        self.repeat_phrases = {
            normalise(p) for p in b.get("repeat_phrases", []) if p.strip()
        }
        self.repeat_slow_phrases = {
            normalise(p) for p in b.get("repeat_slow_phrases", []) if p.strip()
        }
        self.slow_length_scale = float(b.get("slow_length_scale", 1.4))
        self._last_reply: list[str] = []
        self._reply_building: list[str] = []

        # A microphone that stops delivering frames is unplugged, wedged, or
        # gone. Muting does NOT trigger this — a muted array still streams
        # silence, which is exactly the distinction we want.
        self.stall_timeout = float(
            cfg["audio"].get("stall_timeout_seconds", 15))
        self._reply_done = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self._outbox: asyncio.Queue = asyncio.Queue()

    # -- dismissal ---------------------------------------------------

    def is_dismissal(self, norm: str) -> bool:
        """Does this utterance mean 'that's enough, go away'?

        Two passes. The whole phrase first, then the phrase with padding
        trimmed off both ends. Both are EXACT matches against the configured
        list, which is what keeps "stop by the store on the way home" from
        ending the conversation — the core of that sentence is not a
        dismissal, so nothing matches and it goes to the model like anything
        else.

        What this fixes is the other half of the problem. Whisper writes down
        what you actually said, and what people actually say is "okay Lance,
        go to sleep" or "go to sleep now please" — never the bare phrase the
        list contains. Exact matching alone meant that worked about half the
        time, and a dismissal that works half the time is worse than one that
        does not exist, because you keep trying it.
        """
        if not norm:
            return False
        if norm in self.stop_phrases:
            return True
        core = strip_fillers(norm, self.stop_fillers)
        return bool(core) and core in self.stop_phrases

    def _dismiss(self, why: str) -> None:
        """Shut up, forget the reply, go back to sleep. Never touches the
        network — that is the whole point. "Stop" that takes two seconds and
        a round trip to a data centre is not stopping."""
        log.info("dismissed: %s", why)
        self.speaker.stop()                 # cut playback now
        self.send({"type": "cancel"})       # abandon any generation
        if self.stop_chime:
            play_chime(device=self.cfg["audio"].get("output_device"),
                       kind="sleep")
        self.listener.drain()
        self.send({"type": "state", "state": "idle"})

    def _heard_dismissal(self) -> bool:
        """One pass of listening underneath our own voice. Cheap when quiet.

        Returns True only when a complete short utterance transcribed to a
        dismissal. Anything else — a cough, the radio, a sentence that is not
        on the list — is ignored and playback continues, which is the
        behaviour that makes this safe to leave switched on.

        Every failure in here is swallowed, and that is deliberate. This is a
        CONVENIENCE: being able to say "that's enough" without waiting for the
        reply to finish. The conversation it sits inside is the actual
        feature. The first version let an exception escape, the listener loop's
        catch-all caught it, logged it, and started over — which abandoned the
        whole exchange and went back to waiting for the wake word. So a broken
        interrupt-listener read as "follow-up questions do not work", and the
        symptom named nothing about the cause.

        After one failure it switches itself off for the rest of the process.
        A fault here is almost never transient — a missing attribute, a model
        that will not accept a frame — and retrying it on every reply turns
        one bug into a conversation that ends early every single time.
        """
        if not self.dismiss_while_speaking:
            threading.Event().wait(0.1)
            return False
        try:
            clip = self.listener.snippet(max_seconds=self.dismiss_max_seconds)
            if clip.size == 0:
                return False
            text = self.listener.transcribe(clip)
            if not text:
                return False
            if self.is_dismissal(normalise(text)):
                return True
            log.debug("heard %r while speaking — not a dismissal", text[:60])
            return False
        except Exception:  # noqa: BLE001
            self.dismiss_while_speaking = False
            log.exception(
                "listening for a dismissal mid-reply failed — switching it "
                "off for the rest of this run. Conversations and follow-ups "
                "are unaffected; you just cannot interrupt a reply by voice "
                "until this is fixed. Set behaviour.dismiss_while_speaking "
                "to false in config.yaml to stop this being attempted at all.")
            threading.Event().wait(0.1)
            return False

    def _still_playing(self) -> bool:
        """Playing or about to. Falls back on an older speaker.py.

        `is_busy` covers the gap between two sentences, where `is_speaking` is
        briefly false while the next one is still queued. If it is not there,
        use what is — a slightly early follow-up window is a much smaller
        problem than a crash.
        """
        busy = getattr(self.speaker, "is_busy", None)
        return bool(busy) if busy is not None else self.speaker.is_speaking

    def _await_reply(self, timeout: float, generating: bool = True) -> str:
        """Wait for the reply to finish, staying interruptible throughout.

        Returns "ok", "dismissed", or "timeout". Previously this was two
        blocking waits, which meant the microphone was effectively off for the
        entire length of a reply: telling it to go to sleep mid-sentence did
        nothing whatsoever, and the same words worked fine two seconds later
        once the follow-up window opened. That is the whole of "sometimes it
        works and sometimes it doesn't".
        """
        end = time.time() + timeout

        # Phase one: the orchestrator is still generating and queueing.
        while generating and not self._reply_done.is_set():
            if time.time() > end:
                return "timeout"
            if self._heard_dismissal():
                return "dismissed"

        # Phase two: the last sentences are still draining out of the
        # speaker. `is_busy` rather than `is_speaking` — between two sentences
        # the second is briefly false while the next is still queued, and
        # coming out of this loop there would reopen the microphone onto the
        # assistant's own voice.
        while self._still_playing():
            if time.time() > end:
                return "timeout"
            if self._heard_dismissal():
                return "dismissed"

        return "ok"

    # -- outbound ----------------------------------------------------

    def send(self, payload: dict) -> None:
        """Callable from the listener thread."""
        if self.loop:
            self.loop.call_soon_threadsafe(self._outbox.put_nowait, payload)

    # -- listener thread ---------------------------------------------

    def listen_forever(self) -> None:
        try:
            self.listener.start()
        except Exception:  # noqa: BLE001
            # If we cannot open the microphone there is no point continuing,
            # and lingering is actively harmful: the process keeps its grip
            # on whatever ALSA handles it did acquire, so the next attempt
            # finds the device busy (arecord -l shows "Subdevices: 0/1") and
            # reports it as missing. Take the whole process down instead.
            log.exception("could not open the input stream — exiting")
            import os
            os._exit(1)

        log.info("listening for '%s' (threshold %.2f)",
                 self.cfg["wake_word"]["model"],
                 self.cfg["wake_word"]["threshold"])
        self.send({"type": "state", "state": "idle"})
        next_beat = 0.0

        while True:
            try:
                # --- heartbeat ------------------------------------------
                # Tell the orchestrator how long ago the last audio frame
                # arrived. This process holds the sound device exclusively, so
                # it is the ONLY thing that can answer "is the microphone
                # actually delivering audio" — nothing else can open it to
                # look. Without this, a self check can confirm the array is on
                # the USB bus and still not know whether it is hearing
                # anything, which is exactly the wedge that cost two days.
                now = time.time()
                if now >= next_beat:
                    next_beat = now + HEARTBEAT_SECONDS
                    self.send({
                        "type": "heartbeat",
                        "frame_age": now - self.listener.last_frame_ts,
                        "wake_model": self.cfg["wake_word"]["model"],
                        "threshold": self.cfg["wake_word"]["threshold"],
                        "speaking": self.speaker.is_speaking,
                    })

                # --- microphone stall watchdog ---------------------------
                # Exit rather than sit here deaf. systemd restarts us, and
                # ensure_capture_works() then waits patiently for the device
                # to come back — so an unplug, a switch, or a wedged array
                # all recover on their own without anyone typing anything.
                since = time.time() - self.listener.last_frame_ts
                if self.stall_timeout > 0 and since > self.stall_timeout:
                    log.error(
                        "no audio for %.0fs — the microphone has stopped "
                        "delivering frames. Exiting so systemd can restart "
                        "into device recovery.", since)
                    import os
                    os._exit(1)

                speaking = self.speaker.is_speaking

                # While the assistant is talking we either ignore the mic
                # entirely, or keep watching for the wake word so you can
                # interrupt. The second is much nicer to live with, and it
                # is only possible because the mic array does echo
                # cancellation in hardware.
                if speaking and self.mute_while_speaking and not self.barge_in:
                    self.listener.drain()
                    threading.Event().wait(0.1)
                    continue

                score = self.listener.poll_wake(timeout=0.5)
                if score is None:
                    continue

                if speaking:
                    log.info("barge-in")
                    self.speaker.stop()
                    self.send({"type": "barge_in"})

                if self.ack_chime:
                    play_chime(device=self.cfg["audio"].get("output_device"))
                    # The chime leaks into the mic; drop it rather than
                    # transcribing our own beep.
                    self.listener.drain()

                # One wake word can carry a whole conversation. After each
                # reply finishes playing we reopen the microphone briefly —
                # long enough to answer "anything else?" without making you
                # say the wake word again, short enough that the device is
                # not sitting there listening indefinitely.
                follow_up = False
                while True:
                    self._reply_done.clear()
                    self.send({"type": "state", "state": "listening"})
                    audio = self.listener.capture_utterance(
                        start_timeout=(self.follow_up_seconds if follow_up
                                       else self.wake_speech_timeout)
                    )

                    self.send({"type": "state", "state": "transcribing"})
                    text = self.listener.transcribe(audio)

                    norm = normalise(text) if text else ""

                    # "Say that again" — answered here, from memory. Never
                    # reaches the network, so it is instant and free, and it
                    # still works when the API is down.
                    if norm and (norm in self.repeat_phrases
                                 or norm in self.repeat_slow_phrases):
                        slow = norm in self.repeat_slow_phrases
                        if self._last_reply:
                            log.info("repeating last reply%s",
                                     " slowly" if slow else "")
                            for sentence in self._last_reply:
                                self.speaker.say(
                                    sentence,
                                    length_scale=(self.slow_length_scale
                                                  if slow else None))
                        else:
                            self.speaker.say("I have not said anything yet.")
                        if self._await_reply(90, generating=False) == "dismissed":
                            self._dismiss("mid-repeat")
                            break
                        self.listener.drain()
                        follow_up = True
                        continue

                    # "Take this down" — switch to patient endpointing for
                    # the next utterance only.
                    if norm and norm in self.dictation_phrases:
                        log.info("dictation mode for one utterance")
                        self.speaker.say("Go ahead.")
                        self.speaker.wait_until_idle(timeout=30)
                        self.listener.drain()
                        self.send({"type": "state", "state": "listening"})
                        audio = self.listener.capture_utterance(
                            silence_ms=self.dictation_silence_ms,
                            max_seconds=self.dictation_max_seconds,
                        )
                        self.send({"type": "state", "state": "transcribing"})
                        dictated = self.listener.transcribe(audio)
                        if not dictated:
                            self.speaker.say("I did not catch anything.")
                            self.send({"type": "state", "state": "idle"})
                            break
                        self.send({"type": "transcript", "text": dictated,
                                   "wake_score": -1.0, "mode": "dictation"})
                        outcome = self._await_reply(180)
                        if outcome == "dismissed":
                            self._dismiss("mid-reply, after dictation")
                            break
                        if outcome != "ok":
                            break
                        self.listener.drain()
                        follow_up = True
                        continue

                    # Check for dismissal before anything else looks at this.
                    if self.is_dismissal(norm):
                        self._dismiss(f"heard {text!r}")
                        break

                    if not text:
                        if follow_up:
                            log.info("follow-up window closed, back to sleep")
                        else:
                            # Nobody said anything after the chime. That is a
                            # false trigger, and counting them in the log is
                            # how the threshold gets tuned against this house
                            # rather than against a benchmark.
                            log.info("FALSE WAKE — score %.3f, nothing said",
                                     score)
                        self.send({"type": "transcript", "text": "",
                                   "wake_score": score if not follow_up else -1.0})
                        self.send({"type": "state", "state": "idle"})
                        break

                    self.send({
                        "type": "transcript",
                        "text": text,
                        # -1 marks a follow-up, so wake-word statistics stay
                        # honest — these were not wake-word triggers.
                        "wake_score": score if not follow_up else -1.0,
                    })

                    if self.follow_up_seconds <= 0:
                        break

                    # Wait for the whole reply to finish before listening
                    # again, or we transcribe our own voice — but stay
                    # interruptible while we do, so "that's enough" works
                    # mid-sentence and not only in the gap afterwards.
                    outcome = self._await_reply(90)
                    if outcome == "dismissed":
                        self._dismiss("mid-reply")
                        break
                    if outcome != "ok":
                        break

                    self.listener.drain()
                    follow_up = True
                    log.info("follow-up window open for %.0fs",
                             self.follow_up_seconds)

            except Exception:  # noqa: BLE001
                # Degrade, never die. A failure in one utterance must not
                # take the wake-word loop down with it.
                log.exception("listener loop error — continuing")
                threading.Event().wait(0.5)

    # -- websocket ---------------------------------------------------

    async def _pump_outbox(self, ws) -> None:
        while True:
            payload = await self._outbox.get()
            try:
                await ws.send(json.dumps(payload))
            except Exception:  # noqa: BLE001
                return

    async def _handle_inbound(self, ws) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "speak":
                text = msg.get("text", "")
                self._reply_building.append(text)
                self.speaker.say(text)
            elif kind == "chime":
                play_chime(device=self.cfg["audio"].get("output_device"),
                           kind=msg.get("kind", "wake"))
            elif kind == "stop_speaking":
                self.speaker.stop()
            elif kind == "speak_done":
                # Snapshot what was just said, so "say that again" has
                # something to repeat.
                if self._reply_building:
                    self._last_reply = self._reply_building
                    self._reply_building = []
                # The orchestrator has queued the last sentence. Playback may
                # still be draining — the follow-up window waits for that too.
                self._reply_done.set()

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        threading.Thread(target=self.listen_forever, daemon=True).start()

        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    log.info("connected to orchestrator at %s", self.url)
                    backoff = 1
                    self.ws = ws
                    await asyncio.gather(
                        self._pump_outbox(ws),
                        self._handle_inbound(ws),
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("orchestrator unreachable (%s) — retry in %ds",
                            type(exc).__name__, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)


def main() -> int:
    with open(ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"].upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    svc = AudioService(cfg)
    try:
        asyncio.run(svc.run())
    except KeyboardInterrupt:
        pass
    finally:
        svc.speaker.close()
        svc.listener.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
