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
    in_idx = find(a.get("input_device"), "input")
    out_idx = find(a.get("output_device"), "output")
    a["input_device"], a["output_device"] = in_idx, out_idx

    log.info("audio in  -> [%s] %s", in_idx,
             devices[in_idx]["name"] if in_idx is not None else "<default>")
    log.info("audio out -> [%s] %s", out_idx,
             devices[out_idx]["name"] if out_idx is not None else "<default>")


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
    resolve_devices(cfg)
    if probe_capture(cfg):
        log.info("capture healthy")
        return True

    vid_pid = cfg["audio"].get("usb_reset")
    if not vid_pid:
        return False

    log.warning("microphone is enumerated but will not stream — recovering")

    if usb_power_cycle(vid_pid,
                       off_seconds=cfg["audio"].get("usb_power_off_seconds", 3)):
        resolve_devices(cfg)
        if probe_capture(cfg):
            log.info("recovered by power cycling the port")
            return True

    if usb_reset(vid_pid):
        resolve_devices(cfg)
        if probe_capture(cfg):
            log.info("recovered by USB reset")
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
        self._reply_done = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self._outbox: asyncio.Queue = asyncio.Queue()

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

        while True:
            try:
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
                        start_timeout=self.follow_up_seconds if follow_up else None
                    )

                    self.send({"type": "state", "state": "transcribing"})
                    text = self.listener.transcribe(audio)

                    if not text:
                        if follow_up:
                            log.info("follow-up window closed, back to sleep")
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
                    # again, or we transcribe our own voice.
                    if not self._reply_done.wait(timeout=90):
                        break
                    if not self.speaker.wait_until_idle(timeout=90):
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
                self.speaker.say(msg.get("text", ""))
            elif kind == "stop_speaking":
                self.speaker.stop()
            elif kind == "speak_done":
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
