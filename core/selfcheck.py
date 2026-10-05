"""Ask the appliance whether it is actually well.

The failure this exists for is not a crash. It is the Tuesday you say "print
the shopping list" and nothing happens, and the printer has in fact been
unplugged since Saturday. Nothing was broken in a way anybody could see —
you just had no reason to look, and the appliance had no way to tell you.

So: check everything, once a day, and say nothing unless something is wrong.

Three rules shape this file.

**Every check answers with a fix, not just a verdict.** "printer: FAIL" is a
worse output than no output, because it converts a known problem into a
mystery. Each check carries the sentence you would want at 7am.

**Nothing here may hang.** A diagnostic that wedges on a dead USB device has
become the outage. Every check runs under a deadline, and a check that blows
its deadline reports that as its result rather than blocking the next one.

**Nothing here may fix anything.** It looks and it reports. A diagnostic that
restarts services turns "something was briefly wrong" into "something was
briefly wrong and then several things changed underneath me". The watchdog
restarts things; this explains things.
"""
from __future__ import annotations

import logging
import shutil
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("assistant.selfcheck")

ROOT = Path(__file__).resolve().parent.parent

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

# Order matters: this is the order they are spoken, printed and drawn, and it
# runs outward from the physical to the abstract. If the microphone is dead,
# that is the headline; nobody cares about the embedding backlog.
GROUPS = ("Hardware", "Services", "Access", "Models", "Storage", "State")


@dataclass
class Check:
    name: str
    group: str
    status: str = OK
    detail: str = ""
    # What to do about it. Only meaningful for warn and fail, and the single
    # most useful field in the whole structure.
    fix: str = ""
    seconds: float = 0.0

    @property
    def bad(self) -> bool:
        return self.status in (FAIL, WARN)

    def as_dict(self) -> dict:
        return {"name": self.name, "group": self.group, "status": self.status,
                "detail": self.detail, "fix": self.fix,
                "seconds": round(self.seconds, 3)}


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    started: float = 0.0
    seconds: float = 0.0

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def healthy(self) -> bool:
        return not self.failed and not self.warned

    def counts(self) -> dict[str, int]:
        out = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def as_dict(self) -> dict:
        return {
            "at": self.started,
            "seconds": round(self.seconds, 2),
            "healthy": self.healthy,
            "counts": self.counts(),
            "checks": [c.as_dict() for c in self.checks],
        }

    # -- what it says out loud ---------------------------------------

    def spoken(self) -> str:
        """One or two sentences. Names problems; never reads a list of passes.

        The hard part of a diagnostic is not gathering the data, it is not
        drowning the person in it. Thirty lines of "ok" read aloud is how a
        daily check becomes something you switch off in a week.
        """
        bad = self.failed + self.warned
        if not bad:
            n = self.counts()[OK]
            return f"All {n} checks passed."
        # Name at most three, then count the rest. Three is about as many
        # problems as anybody can hold while walking to the other room.
        names = [c.name for c in bad[:3]]
        listed = (", ".join(names[:-1]) + " and " + names[-1]
                  if len(names) > 1 else names[0])
        more = len(bad) - len(names)
        head = (f"{len(self.failed)} failed"
                if self.failed and not self.warned else
                f"{len(self.warned)} warning{'s' if len(self.warned) != 1 else ''}"
                if not self.failed else
                f"{len(self.failed)} failed and {len(self.warned)} warning"
                f"{'s' if len(self.warned) != 1 else ''}")
        tail = f", and {more} more" if more else ""
        first = bad[0]
        # One sentence of the worst one, and no more. This string is read out
        # loud by the daily run, and a paragraph of shell commands spoken at
        # 7am is how a useful feature becomes one you switch off.
        extra = _first_sentence(first.detail) or ""
        if first.fix:
            extra = (extra + " " + _first_sentence(first.fix)).strip()
        return f"{head}: {listed}{tail}. {extra}".strip()

    def briefing(self, limit: int = 8) -> str:
        """The whole thing as text, for the model to summarise or read.

        Longer than `spoken()` on purpose: this goes to the language model as
        a tool result, and the model is better at choosing what to say than a
        format string is. It gets the counts, then every problem with its fix,
        and is told in the prompt to be brief about it.
        """
        c = self.counts()
        lines = [f"Self check: {c[OK]} ok, {c[WARN]} warnings, {c[FAIL]} "
                 f"failed, {c[SKIP]} skipped, in "
                 f"{self.seconds:.0f} seconds."]
        bad = self.failed + self.warned
        if not bad:
            lines.append("Nothing wrong. Every peripheral, service, model and "
                         "credential checked out.")
            return "\n".join(lines)
        for check in bad[:limit]:
            mark = "FAILED" if check.status == FAIL else "warning"
            lines.append(f"- {check.name} ({mark}): {check.detail}"
                         + (f" Fix: {check.fix}" if check.fix else ""))
        if len(bad) > limit:
            lines.append(f"- and {len(bad) - limit} more, on the screen.")
        return "\n".join(lines)


def _first_sentence(text: str, limit: int = 110) -> str:
    """The first sentence, for anything that gets read out loud."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    for end in (". ", "? ", "! "):
        i = text.find(end)
        if 0 < i <= limit:
            return text[:i + 1]
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "..."


class _quiet_root:
    """Silence the root logger for the length of a block.

    For one specific nuisance: python-escpos calls set_configuration() when
    it opens the printer, and on a device the kernel has already configured
    that raises EBUSY, which the library catches and logs to the ROOT logger
    at ERROR — "Could not set configuration: [Errno 16] Resource busy".

    It is harmless; the device is already configured, which is why the probe
    goes on to read its status successfully. But the status wall now opens
    the printer every fifteen minutes, so it writes a red herring into the
    journal ninety-six times a day, and the journal is where you look when
    something is actually wrong. A diagnostic that fills the log with false
    errors has made the next real outage harder to find.
    """

    def __init__(self, level: int = logging.CRITICAL):
        self.level = level
        self.was = None

    def __enter__(self):
        root = logging.getLogger()
        self.was = root.level
        root.setLevel(self.level)
        return self

    def __exit__(self, *exc):
        logging.getLogger().setLevel(self.was)
        return False


def _deadline(fn, seconds: float, *args, **kwargs):
    """Run fn on a thread and give up on it after `seconds`.

    Not cancellation — Python cannot interrupt a blocking libusb call — but the
    CALLER stops waiting, which is the part that matters. The thread is a
    daemon, so a wedged USB read cannot keep the process alive either.
    """
    import threading
    box: dict = {}

    def go():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"still running after {seconds:.0f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class SelfCheck:
    """Every check the appliance can run on itself."""

    def __init__(self, cfg: dict, store, tools=None, hub=None):
        self.cfg = cfg
        self.store = store
        self.tools = tools          # the Toolbox, for printer/embedder/etc
        self.hub = hub              # to see who is connected
        s = cfg.get("selfcheck", {}) or {}
        self.enabled = bool(s.get("enabled", True))
        self.api_call = bool(s.get("test_api", True))
        self.web_search = bool(s.get("test_web_search", True))
        self.youtube = bool(s.get("test_youtube", True))
        self.disk_warn_gb = float(s.get("disk_warn_gb", 5))
        self.disk_fail_gb = float(s.get("disk_fail_gb", 1))
        self.spend_warn = float(s.get("spend_warn_usd", 3.0))
        self.false_wake_warn = float(s.get("false_wake_warn_pct", 50))
        self.stale_case_hours = float(s.get("stale_case_hours", 18))
        self.budget = float(s.get("check_timeout_seconds", 12))
        # Set by app.py from the audio service's heartbeat. None means the
        # audio service has never reported, which is itself a finding.
        self.mic_frame_age: float | None = None
        self.mic_reported_at: float = 0.0
        self.last: Report | None = None

    # -- the runner ---------------------------------------------------

    def run(self, deep: bool = True) -> Report:
        """Every check, in group order. Never raises."""
        report = Report(started=time.time())
        t0 = time.monotonic()
        for name, group, budget, fn in self._plan(deep):
            started = time.monotonic()
            try:
                check = _deadline(fn, budget)
            except TimeoutError as exc:
                check = Check(name, group, FAIL, str(exc),
                              "Something is not answering. Look at the "
                              "journal for this subsystem.")
            except Exception as exc:  # noqa: BLE001
                log.exception("check %r blew up", name)
                check = Check(name, group, FAIL,
                              f"the check itself failed: "
                              f"{type(exc).__name__}: {exc}",
                              "This is a bug in the diagnostic, not "
                              "necessarily in the thing it checks.")
            if check is None:
                check = Check(name, group, SKIP, "returned nothing")
            check.name, check.group = name, group
            check.seconds = time.monotonic() - started
            report.checks.append(check)
        report.seconds = time.monotonic() - t0
        self.last = report
        counts = report.counts()
        log.info("self check in %.1fs — %d ok, %d warn, %d fail, %d skipped",
                 report.seconds, counts[OK], counts[WARN], counts[FAIL],
                 counts[SKIP])
        for c in report.failed + report.warned:
            log.warning("  %s: %s — %s", c.name, c.status.upper(), c.detail)
        return report

    def _plan(self, deep: bool):
        """(name, group, seconds allowed, function) in the order they run."""
        p = [
            ("microphone on the bus", "Hardware", 6, self._mic_usb),
            ("microphone hearing", "Hardware", 2, self._mic_frames),
            ("speaker output", "Hardware", 4, self._speaker),
            ("printer", "Hardware", 10, self._printer),

            ("orchestrator", "Services", 2, self._core),
            ("audio service", "Services", 2, self._audio_service),
            ("display", "Services", 2, self._display),
            ("watchdog", "Services", 6, self._watchdog),

            ("network", "Access", 8, self._network),
        ]
        if deep and self.api_call:
            p.append(("model API", "Access", 25, self._api))
        if deep and self.web_search:
            p.append(("web search", "Access", 30, self._websearch))
        if deep and self.youtube:
            p.append(("youtube", "Access", 25, self._youtube))
        p += [
            ("yt-dlp freshness", "Access", 10, self._ytdlp_age),
            ("mpv", "Access", 6, self._mpv),

            ("transcription model", "Models", 4, self._whisper),
            ("voice", "Models", 4, self._piper),
            ("wake word model", "Models", 4, self._wakeword),
            ("embedding model", "Models", 20, self._embedder),

            ("disk space", "Storage", 4, self._disk),
            ("database", "Storage", 10, self._database),
            ("notes folder", "Storage", 4, self._notes),
            ("workouts folder", "Storage", 4, self._workouts),
            ("playbooks", "Storage", 4, self._playbooks),

            ("open sessions", "State", 4, self._open_cases),
            ("memory index", "State", 6, self._index_backlog),
            ("stale facts", "State", 6, self._stale_facts),
            ("wake word accuracy", "State", 4, self._wake_stats),
            ("spend", "State", 4, self._spend),
            ("reminders", "State", 4, self._reminders),
        ]
        return p

    # ================================================== Hardware

    def _usb_present(self, vid_pid: str) -> bool:
        """Is this vendor:product on the bus? pyusb first, lsusb as fallback."""
        vid, pid = (int(x, 16) for x in vid_pid.split(":"))
        try:
            import usb.core
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                import usb.util
                # Release it immediately. A held handle is exactly how the
                # printer probe once made the printer look busy to itself.
                usb.util.dispose_resources(dev)
                return True
            return False
        except Exception:  # noqa: BLE001
            pass
        try:
            out = subprocess.run(["lsusb"], capture_output=True, text=True,
                                 timeout=5)
            return vid_pid.lower() in out.stdout.lower()
        except Exception:  # noqa: BLE001
            return False

    def _mic_usb(self) -> Check:
        vid_pid = (self.cfg.get("audio", {}) or {}).get("usb_reset")
        if not vid_pid:
            return Check("", "", SKIP, "no audio.usb_reset configured")
        if self._usb_present(str(vid_pid)):
            return Check("", "", OK, f"{vid_pid} present")
        return Check("", "", FAIL, f"{vid_pid} is not on the USB bus",
                     "Reseat the microphone's USB cable. Nothing in software "
                     "can bring it back — the processor inside it only "
                     "restarts when the power actually goes away.")

    def _mic_frames(self) -> Check:
        """Is the microphone delivering audio, as reported by the one process
        that can know?

        The audio service holds the sound device exclusively, so nothing else
        can open it to find out. It sends its frame age in a heartbeat instead.
        A microphone that is muted at the array's own button still streams
        silence, so this tests the transport, not whether anyone is talking.
        """
        if self.mic_reported_at == 0:
            return Check("", "", WARN, "the audio service has never reported",
                         "It may be starting up, or running an older build. "
                         "Check: systemctl --user status assistant-audio")
        stale = time.time() - self.mic_reported_at
        if stale > 120:
            return Check("", "", FAIL,
                         f"the last report was {stale / 60:.0f} minutes ago",
                         "The audio service has stopped talking to the "
                         "orchestrator. systemctl --user restart "
                         "assistant-audio")
        age = self.mic_frame_age
        if age is None:
            return Check("", "", WARN, "frame age not reported")
        if age > 15:
            return Check("", "", FAIL,
                         f"no audio for {age:.0f} seconds",
                         "The array is wedged or unplugged. Reseat the cable.")
        if age > 2:
            return Check("", "", WARN, f"last frame {age:.1f}s ago",
                         "Slower than expected but not dead.")
        return Check("", "", OK, f"frames arriving ({age:.2f}s ago)")

    def _speaker(self) -> Check:
        """Does the configured output device exist?

        Note what proves this better than any check: you are about to hear the
        result read out loud. If the summary is audible, the speaker works.
        This catches the case where it is NOT — a device that has vanished
        from the sound system while the appliance was idle.
        """
        want = (self.cfg.get("audio", {}) or {}).get("output_name") \
            or (self.cfg.get("audio", {}) or {}).get("output_device")
        if isinstance(want, int) or not want:
            return Check("", "", SKIP, "output device configured by index")
        try:
            with open("/proc/asound/cards") as fh:
                cards = fh.read()
        except OSError:
            return Check("", "", SKIP, "cannot read /proc/asound/cards")
        if str(want).lower() in cards.lower():
            return Check("", "", OK, f"{want!r} present in ALSA")
        return Check("", "", FAIL, f"no sound card matching {want!r}",
                     "The output device is gone. If the microphone array is "
                     "also failing above, this is the same cable.")

    def _printer(self) -> Check:
        """Is the printer ready — without printing anything.

        ESC/POS has a real-time status request: the printer answers while busy,
        which is the whole point of it. That gives paper-out, cover-open and
        error state for the cost of four bytes and no paper. Falling back to
        "can I claim the interface" still catches unplugged, no permission, and
        another process holding it, which is most of what goes wrong.
        """
        p = self.cfg.get("printer", {}) or {}
        if not p.get("enabled"):
            return Check("", "", SKIP, "printer disabled in config")
        vid_pid = f"{str(p.get('vendor_id', '0x04b8'))[2:]}:" \
                  f"{str(p.get('product_id', '0x0e20'))[2:]}"
        if not self._usb_present(vid_pid):
            return Check("", "", FAIL, f"{vid_pid} is not on the USB bus",
                         "The printer is unplugged or switched off.")
        try:
            from escpos.printer import Usb
        except ImportError:
            return Check("", "", WARN, f"{vid_pid} present, python-escpos is "
                                       f"not installed",
                         "pip install python-escpos")
        vid = int(str(p.get("vendor_id", "0x04b8")), 16)
        pid = int(str(p.get("product_id", "0x0e20")), 16)
        out_ep = int(str(p.get("out_ep", "0x01")), 16)
        raw_in = p.get("in_ep", "0x81")
        kwargs = {}
        if raw_in not in (None, "", "null"):
            kwargs["in_ep"] = int(str(raw_in), 16)
        dev = None
        try:
            with _quiet_root():
                dev = Usb(vid, pid, timeout=3000, out_ep=out_ep, **kwargs)
            notes = []
            status = None
            for attr in ("paper_status", "is_online"):
                fn = getattr(dev, attr, None)
                if not callable(fn):
                    continue
                try:
                    status = fn()
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"{attr} did not answer ({type(exc).__name__})")
                    continue
                if attr == "paper_status":
                    # python-escpos: 2 = plenty, 1 = low, 0 = out.
                    if status == 0:
                        return Check("", "", FAIL, "out of paper",
                                     "Load a new roll.")
                    if status == 1:
                        return Check("", "", WARN, "paper is low",
                                     "Worth a new roll soon.")
                    notes.append("paper ok")
                elif attr == "is_online" and status is False:
                    return Check("", "", FAIL, "reports itself offline",
                                 "Check the cover is shut and there is no "
                                 "paper jam.")
            detail = f"{vid_pid} claimed" + (f", {', '.join(notes)}"
                                             if notes else "")
            return Check("", "", OK, detail)
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            if "permission" in text or "access" in text or "errno 13" in text:
                return Check("", "", FAIL, "present but refuses access",
                             "The udev rule is missing. "
                             "sudo install -m 644 systemd/99-respeaker.rules "
                             "/etc/udev/rules.d/ and replug it.")
            if "busy" in text or "resource" in text:
                return Check("", "", FAIL, "present but busy",
                             "Something else is holding it. CUPS or ipp-usb "
                             "will do this — they claim USB printers.")
            return Check("", "", FAIL, f"{type(exc).__name__}: {exc}",
                         "Run scripts/printer_probe.py for detail.")
        finally:
            if dev is not None:
                try:
                    dev.close()
                except Exception:  # noqa: BLE001
                    pass

    # ================================================== Services

    def _core(self) -> Check:
        # Trivially true — this code is running inside it. Worth reporting
        # anyway: the receipt is read by someone who was not here, and "the
        # orchestrator is up" is the premise everything else rests on.
        return Check("", "", OK, "answering (this check ran inside it)")

    def _audio_service(self) -> Check:
        if self.hub is None:
            return Check("", "", SKIP, "no hub")
        if self.hub.audio:
            return Check("", "", OK, f"{len(self.hub.audio)} connection")
        return Check("", "", FAIL, "not connected",
                     "Nothing is listening for the wake word. "
                     "systemctl --user restart assistant-audio")

    def _display(self) -> Check:
        if self.hub is None:
            return Check("", "", SKIP, "no hub")
        if self.hub.display:
            return Check("", "", OK, f"{len(self.hub.display)} connection")
        return Check("", "", WARN, "no browser attached",
                     "The screen shows nothing. Harmless if you are not "
                     "using it. systemctl --user restart assistant-ui")

    def _watchdog(self) -> Check:
        try:
            out = subprocess.run(
                ["systemctl", "--user", "is-active",
                 "assistant-watchdog.timer"],
                capture_output=True, text=True, timeout=5)
            state = (out.stdout or "").strip()
        except Exception as exc:  # noqa: BLE001
            return Check("", "", SKIP, f"cannot ask systemd ({type(exc).__name__})")
        if state == "active":
            return Check("", "", OK, "timer active")
        return Check("", "", WARN, f"timer is {state or 'not installed'}",
                     "Nothing is watching for a deaf microphone. "
                     "systemctl --user enable --now assistant-watchdog.timer")

    # ================================================== Access

    def _network(self) -> Check:
        host = "api.anthropic.com"
        t0 = time.monotonic()
        try:
            addr = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)[0]
        except OSError as exc:
            return Check("", "", FAIL, f"cannot resolve {host} ({exc})",
                         "DNS is broken or there is no network at all.")
        dns = time.monotonic() - t0
        family = getattr(addr[0], "name", str(addr[0])).replace("AF_", "")
        try:
            # Connect by NAME, not by the sockaddr getaddrinfo handed back.
            #
            # addr[4] is a 2-tuple for IPv4 and a 4-TUPLE for IPv6 (it carries
            # flowinfo and scope_id), and create_connection only ever accepts
            # two values. Passing it through worked for a month and then broke
            # the morning the resolver started answering with AAAA first:
            #   ValueError: too many values to unpack (expected 2)
            # which is not a network fault at all, and said so on the wall in
            # red. Handing it the name lets it resolve and try each address
            # itself, which is also a truer test of "can I reach this".
            with socket.create_connection((host, 443), timeout=5):
                pass
        except OSError as exc:
            return Check("", "", FAIL, f"cannot reach {host} ({exc})",
                         "DNS works but the connection does not — a firewall, "
                         "or the router is up without internet behind it.")
        return Check("", "", OK,
                     f"{host} reachable over {family}, DNS in {dns * 1000:.0f}ms")

    def _api(self) -> Check:
        """One real call to the cheapest model.

        Nothing else proves the key is valid. Reaching the host proves the
        network; only a call that comes back proves the credential, the
        account and the model. It costs a hundredth of a cent.
        """
        try:
            from .provider import read_api_key
            key, source = read_api_key()
        except Exception as exc:  # noqa: BLE001
            return Check("", "", FAIL, f"cannot read a key ({exc})", "")
        if not key:
            return Check("", "", FAIL, "no API key anywhere",
                         "./scripts/setup_credential.sh")
        try:
            from anthropic import Anthropic
        except ImportError:
            return Check("", "", SKIP, "the anthropic SDK is not installed")
        model = ((self.cfg.get("provider", {}) or {})
                 .get("models", {}) or {}).get("small", "claude-haiku-4-5")
        try:
            client = Anthropic(api_key=key, timeout=20.0, max_retries=0)
            msg = client.messages.create(
                model=model, max_tokens=4,
                messages=[{"role": "user", "content": "Reply with: ok"}])
            said = "".join(getattr(b, "text", "") for b in msg.content).strip()
            return Check("", "", OK,
                         f"{model} answered {said[:12]!r}, key from {source}")
        except Exception as exc:  # noqa: BLE001
            from .provider import explain_failure
            spoken = explain_failure(exc)
            status = getattr(exc, "status_code", None)
            fix = ("./scripts/setup_credential.sh, then restart "
                   "assistant-core" if status in (401, 403) else "")
            return Check("", "", FAIL, spoken, fix)

    def _websearch(self) -> Check:
        """Does the search tool itself work? Costs about a cent."""
        ws = (self.cfg.get("tools", {}) or {}).get("web_search", {}) or {}
        if not ws.get("enabled"):
            return Check("", "", SKIP, "web search disabled in config")
        try:
            from anthropic import Anthropic
            from .provider import read_api_key, web_search_tool
        except ImportError:
            return Check("", "", SKIP, "the anthropic SDK is not installed")
        key, _ = read_api_key()
        if not key:
            return Check("", "", SKIP, "no API key")
        tiers = ws.get("tiers", ["mid", "top"])
        models = (self.cfg.get("provider", {}) or {}).get("models", {}) or {}
        model = models.get(tiers[0] if tiers else "mid", "claude-sonnet-5")
        tool = dict(web_search_tool(self.cfg))
        tool["max_uses"] = 1
        try:
            client = Anthropic(api_key=key, timeout=25.0, max_retries=0)
            msg = client.messages.create(
                model=model, max_tokens=256, tools=[tool],
                messages=[{"role": "user", "content":
                           "Search the web for today's date and say it."}])
            used = getattr(getattr(msg, "usage", None),
                           "server_tool_use", None)
            n = getattr(used, "web_search_requests", 0) or 0
            if n:
                return Check("", "", OK, f"{n} search performed on {model}")
            return Check("", "", WARN,
                         "the model answered without searching",
                         "Not necessarily broken — it may have decided it "
                         "already knew. Worth re-running.")
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "web_search" in text or "tool" in text.lower():
                return Check("", "", FAIL,
                             f"the search tool was rejected: {text[:120]}",
                             "WEB_SEARCH_TOOL_TYPE in core/provider.py is "
                             "versioned by date and may need updating.")
            from .provider import explain_failure
            return Check("", "", FAIL, explain_failure(exc), "")

    def _youtube(self) -> Check:
        """A real search, because this is the thing that rots silently."""
        m = self.cfg.get("media", {}) or {}
        if not m.get("enabled"):
            return Check("", "", SKIP, "media disabled in config")
        from .media import YouTube
        yt = YouTube(self.cfg)
        missing = yt.available()
        if missing:
            return Check("", "", FAIL, missing,
                         "~/assistant/.venv/bin/pip install -U yt-dlp")
        hits = yt.search("test", limit=1)
        if hits:
            return Check("", "", OK, f"search returned {hits[0].title[:28]!r}")
        return Check("", "", FAIL, "a search returned nothing",
                     "Almost always an out-of-date yt-dlp. "
                     "~/assistant/.venv/bin/pip install -U yt-dlp")

    def _ytdlp_age(self) -> Check:
        m = self.cfg.get("media", {}) or {}
        if not m.get("enabled"):
            return Check("", "", SKIP, "media disabled in config")
        from .media import YouTube
        path = YouTube(self.cfg).path()
        if not path:
            return Check("", "", FAIL, "yt-dlp not found",
                         "~/assistant/.venv/bin/pip install -U yt-dlp")
        try:
            out = subprocess.run([path, "--version"], capture_output=True,
                                 text=True, timeout=8)
            stamp = (out.stdout or "").strip().split()[0]
            import datetime as dt
            built = dt.date(*[int(x) for x in stamp.split(".")[:3]])
            days = (dt.date.today() - built).days
        except Exception as exc:  # noqa: BLE001
            return Check("", "", WARN,
                         f"cannot read the version ({type(exc).__name__})")
        if days > 365:
            return Check("", "", FAIL, f"{days} days old ({stamp})",
                         "This will fail against YouTube. "
                         "~/assistant/.venv/bin/pip install -U yt-dlp")
        if days > 180:
            return Check("", "", WARN, f"{days} days old ({stamp})",
                         "Worth updating before it starts failing.")
        return Check("", "", OK, f"{days} days old ({stamp})")

    def _mpv(self) -> Check:
        m = self.cfg.get("media", {}) or {}
        if not m.get("enabled"):
            return Check("", "", SKIP, "media disabled in config")
        binary = m.get("mpv_bin", "mpv")
        if shutil.which(binary):
            return Check("", "", OK, f"{shutil.which(binary)}")
        return Check("", "", FAIL, f"{binary} not found",
                     "sudo apt install -y mpv")

    # ================================================== Models

    def _whisper(self) -> Check:
        name = (self.cfg.get("stt", {}) or {}).get("model", "small.en")
        cache = ROOT / "models" / "whisper"
        if not cache.exists():
            return Check("", "", WARN, f"{cache.name}/ is missing",
                         "It downloads on first use, so this is only a "
                         "problem with no network.")
        bins = list(cache.rglob("model.bin")) + list(cache.rglob("*.bin"))
        if bins:
            mb = sum(b.stat().st_size for b in bins) / 1e6
            return Check("", "", OK, f"{name}, {mb:.0f} MB cached")
        return Check("", "", WARN, f"no model files under {cache.name}/",
                     "The first transcription will download it.")

    def _piper(self) -> Check:
        voice = (self.cfg.get("tts", {}) or {}).get("voice", "")
        path = Path(voice)
        if not path.is_absolute():
            path = ROOT / path
        problems = []
        if not path.exists():
            problems.append(f"voice file missing: {path.name}")
        cfgfile = Path(str(path) + ".json")
        if path.exists() and not cfgfile.exists():
            problems.append(f"{cfgfile.name} missing (piper needs both)")
        if not shutil.which("piper") and not (ROOT / ".venv/bin/piper").exists():
            problems.append("the piper binary is not on PATH")
        if problems:
            return Check("", "", FAIL, "; ".join(problems),
                         "./scripts/set_voice.sh  (or scripts/fetch_models.sh)")
        return Check("", "", OK, f"{path.name}")

    def _wakeword(self) -> Check:
        w = self.cfg.get("wake_word", {}) or {}
        name = str(w.get("model", "hey_jarvis"))
        if name.endswith(".onnx"):
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                return Check("", "", OK, f"custom model {path.name}")
            return Check("", "", FAIL, f"custom model missing: {path}",
                         "Train one with openWakeWord's notebook, or set "
                         "wake_word.model back to a bundled name.")
        try:
            import openwakeword
            base = Path(openwakeword.__file__).parent / "resources" / "models"
            hits = list(base.glob(f"{name}*")) if base.exists() else []
            if hits:
                return Check("", "", OK,
                             f"{name} (threshold {w.get('threshold')}, "
                             f"confirm {w.get('confirm_frames', 1)})")
            return Check("", "", WARN, f"{name} not in the bundled cache",
                         "It downloads on first use.")
        except ImportError:
            return Check("", "", FAIL, "openwakeword is not installed",
                         "pip install --no-deps 'openwakeword>=0.6.0'")

    def _embedder(self) -> Check:
        sem = ((self.cfg.get("memory", {}) or {}).get("semantic", {}) or {})
        if not sem.get("enabled"):
            return Check("", "", SKIP, "semantic recall disabled in config")
        if self.tools is None or getattr(self.tools, "embedder", None) is None:
            return Check("", "", SKIP, "no toolbox")
        emb = self.tools.embedder
        try:
            vec = emb.query("does this still work")
        except Exception as exc:  # noqa: BLE001
            return Check("", "", FAIL,
                         f"the model would not run: {type(exc).__name__}",
                         "./scripts/fetch_embed_model.sh")
        if vec is None or len(vec) == 0:
            return Check("", "", FAIL, "the model returned nothing",
                         "./scripts/fetch_embed_model.sh")
        return Check("", "", OK, f"{emb.model_name}, {len(vec)} dimensions")

    # ================================================== Storage

    def _disk(self) -> Check:
        usage = shutil.disk_usage(ROOT)
        free_gb = usage.free / 1e9
        pct = 100 * usage.free / usage.total
        detail = f"{free_gb:.1f} GB free ({pct:.0f}%)"
        if free_gb < self.disk_fail_gb:
            return Check("", "", FAIL, detail,
                         "A full disk is how an appliance dies quietly. "
                         "Check data/ for old database backups.")
        if free_gb < self.disk_warn_gb:
            return Check("", "", WARN, detail, "Worth a look before it bites.")
        return Check("", "", OK, detail)

    def _database(self) -> Check:
        path = Path(self.store.path) if hasattr(self.store, "path") else None
        if path is None or not path.exists():
            return Check("", "", FAIL, "no database file", "")
        mb = path.stat().st_size / 1e6
        try:
            conn = sqlite3.connect(str(path))
            row = conn.execute("PRAGMA quick_check").fetchone()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            return Check("", "", FAIL, f"cannot check: {exc}", "")
        if row and row[0] == "ok":
            return Check("", "", OK, f"{mb:.1f} MB, integrity ok")
        return Check("", "", FAIL, f"integrity check said {row[0] if row else '?'}",
                     "Stop the services and restore the most recent "
                     "data/*.bak before writing anything else.")

    def _writable(self, path: Path) -> str:
        probe = path / ".selfcheck"
        try:
            probe.write_text("x")
            probe.unlink()
            return ""
        except OSError as exc:
            return str(exc)

    def _notes(self) -> Check:
        d = self.cfg.get("documents", {}) or {}
        if not d.get("enabled"):
            return Check("", "", SKIP, "notes disabled in config")
        path = Path(str(d.get("path", "~/Documents/Assistant"))).expanduser()
        if not path.exists():
            return Check("", "", FAIL, f"{path} does not exist",
                         f"mkdir -p {path}")
        err = self._writable(path)
        if err:
            return Check("", "", FAIL, f"not writable: {err}",
                         f"chmod u+w {path}")
        n = len(list(path.glob(f"*.{str(d.get('format', 'txt')).lstrip('.')}")))
        return Check("", "", OK, f"{n} notes, writable")

    def _workouts(self) -> Check:
        w = self.cfg.get("workouts", {}) or {}
        if not w.get("enabled"):
            return Check("", "", SKIP, "workouts disabled in config")
        path = Path(str(w.get("path", "workouts"))).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            return Check("", "", FAIL, f"{path} does not exist",
                         f"mkdir -p {path}")
        files = sorted(path.glob("*.txt"))
        if not files:
            return Check("", "", WARN, "no workout files",
                         f"Drop one in {path.name} and ask her to list them.")
        return Check("", "", OK, f"{len(files)} files, newest "
                                f"{files[-1].name}")

    def _playbooks(self) -> Check:
        c = self.cfg.get("casework", {}) or {}
        if not c.get("enabled"):
            return Check("", "", SKIP, "casework disabled in config")
        if self.tools is None or getattr(self.tools, "cases", None) is None:
            return Check("", "", SKIP, "no toolbox")
        try:
            books = self.tools.cases.books.all()
        except Exception as exc:  # noqa: BLE001
            return Check("", "", FAIL, f"cannot load: {type(exc).__name__}: {exc}",
                         "A playbook YAML file is malformed.")
        if not books:
            return Check("", "", WARN, "no playbooks found",
                         "Nothing to start an investigation from.")
        holes = []
        for name, spec in books.items():
            body = str(spec)
            if "placeholder" in body.lower():
                holes.append(name)
        if holes:
            return Check("", "", WARN,
                         f"{len(books)} playbook(s); {', '.join(holes)} still "
                         f"contains placeholder steps",
                         "Fill in the real steps before relying on it.")
        return Check("", "", OK, f"{len(books)} playbook(s): "
                                f"{', '.join(books)}")

    # ================================================== State

    def _open_cases(self) -> Check:
        try:
            rows = self.store.cases_open()
        except Exception as exc:  # noqa: BLE001
            return Check("", "", FAIL, f"cannot read: {exc}", "")
        if not rows:
            return Check("", "", OK, "none open")
        now = time.time()
        stale = [r for r in rows
                 if now - r["opened"] > self.stale_case_hours * 3600]
        names = ", ".join(f"{r['title']}" for r in rows[:3])
        if stale:
            return Check("", "", WARN,
                         f"{len(rows)} open, {len(stale)} older than "
                         f"{self.stale_case_hours:.0f}h ({names})",
                         "An open session routes every request to the bigger "
                         "model. Say 'close the case' or 'finish the "
                         "workout'.")
        return Check("", "", OK, f"{len(rows)} open ({names})")

    def _index_backlog(self) -> Check:
        sem = ((self.cfg.get("memory", {}) or {}).get("semantic", {}) or {})
        if not sem.get("enabled"):
            return Check("", "", SKIP, "semantic recall disabled in config")
        try:
            stats = self.tools.index.stats() if self.tools else {}
        except Exception as exc:  # noqa: BLE001
            return Check("", "", WARN, f"cannot read: {type(exc).__name__}")
        # Both come back as {kind: count} — messages and memories are indexed
        # separately, and the interesting number is the total either way.
        def total(value) -> int:
            if isinstance(value, dict):
                return sum(int(v or 0) for v in value.values())
            return int(value or 0)

        pending = total(stats.get("pending"))
        done = total(stats.get("indexed"))
        if not stats.get("available", True):
            return Check("", "", FAIL,
                         "the embedding model would not load",
                         "Recall falls back to keyword search until this is "
                         "fixed. ./scripts/fetch_embed_model.sh")
        if pending > 500:
            return Check("", "", WARN,
                         f"{done} indexed, {pending} still waiting",
                         "It catches up in the background. A big backlog "
                         "means recall is missing recent conversations.")
        return Check("", "", OK, f"{done} indexed, {pending} pending")

    def _stale_facts(self) -> Check:
        if self.tools is None or getattr(self.tools, "sweep", None) is None:
            return Check("", "", SKIP, "no toolbox")
        try:
            rows = self.tools.sweep.review()
        except Exception as exc:  # noqa: BLE001
            return Check("", "", WARN, f"cannot read: {type(exc).__name__}")
        n = len(rows or [])
        if n > 10:
            return Check("", "", WARN, f"{n} facts flagged as possibly stale",
                         "Ask her what she is unsure about.")
        return Check("", "", OK, f"{n} flagged")

    def _wake_stats(self) -> Check:
        try:
            s = self.store.wake_stats(time.time() - 86400)
        except Exception as exc:  # noqa: BLE001
            return Check("", "", WARN, f"cannot read: {type(exc).__name__}")
        total = int(s.get("triggers", 0))
        false = int(s.get("likely_false_positives", 0))
        if total == 0:
            return Check("", "", OK, "no wake events in 24 hours")
        pct = 100.0 * false / total
        detail = f"{total} triggers, {false} with nothing said ({pct:.0f}%)"
        if pct >= self.false_wake_warn and total >= 5:
            return Check("", "", WARN, detail,
                         "Raise wake_word.confirm_frames to 2, or run "
                         "scripts/wake_tune.py to pick a threshold from this "
                         "log.")
        return Check("", "", OK, detail)

    def _spend(self) -> Check:
        try:
            spent = self.store.spend_since(time.time() - 86400)
        except Exception as exc:  # noqa: BLE001
            return Check("", "", WARN, f"cannot read: {type(exc).__name__}")
        detail = f"${spent:.2f} in 24 hours"
        if spent > self.spend_warn:
            return Check("", "", WARN, detail,
                         "Higher than a normal day. Check the journal for a "
                         "loop: journalctl --user -u assistant-core | grep "
                         "'tools used'")
        return Check("", "", OK, detail)

    def _reminders(self) -> Check:
        try:
            rows = self.store.pending_reminders()
        except Exception as exc:  # noqa: BLE001
            return Check("", "", WARN, f"cannot read: {type(exc).__name__}")
        if not rows:
            return Check("", "", OK, "none pending")
        overdue = [r for r in rows if r["due"] < time.time() - 60]
        if overdue:
            return Check("", "", FAIL,
                         f"{len(overdue)} overdue and unfired",
                         "The scheduler has stopped. systemctl --user restart "
                         "assistant-core")
        return Check("", "", OK, f"{len(rows)} pending")


# =====================================================================
# The tool
# =====================================================================

class SelfCheckTools:
    """`run_self_check` — one tool, because it is one question.

    Deliberately not "check the printer" and "check the microphone" as separate
    tools. The whole value is that it checks the things you did NOT think to
    ask about; a tool per subsystem would put the person back in the position
    of having to guess which one is broken.
    """

    def __init__(self, checker: SelfCheck, printer=None, hub=None):
        self.checker = checker
        self.printer = printer
        self.hub = hub
        self.enabled = checker.enabled

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [{
            "name": "run_self_check",
            "description": (
                "Check the whole appliance and report what is wrong: "
                "microphone, speaker, printer, services, network, API key, "
                "models, disk, database and internal state. Use whenever "
                "asked to run a check, a diagnostic, a health check, to test "
                "herself, or whether everything is working. Also use it when "
                "asked whether one particular thing works — the printer, the "
                "microphone, the internet — because the answer for one is a "
                "subset of this and a failure elsewhere is worth knowing. "
                "Takes up to a minute."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "print_it": {
                        "type": "boolean",
                        "description": (
                            "Print the results on the receipt printer. "
                            "Default false. Set true when asked to print the "
                            "check, or for a record of it."
                        ),
                    },
                    "quick": {
                        "type": "boolean",
                        "description": (
                            "Skip the checks that cost money or take time — "
                            "the API call, the web search, the YouTube "
                            "search. Default false."
                        ),
                    },
                },
            },
        }]

    def run_sync(self, name: str, args: dict) -> str | None:
        if not self.enabled or name != "run_self_check":
            return None
        report = self.checker.run(deep=not bool(args.get("quick")))
        out = [report.briefing()]

        # The screen gets the whole thing whether or not it was asked for: the
        # spoken summary names three problems at most, and the rest have to be
        # somewhere.
        if self.hub is not None:
            try:
                self.hub.push_selfcheck(report)
            except Exception:  # noqa: BLE001
                log.debug("could not push to the display", exc_info=True)

        if args.get("print_it") and self.printer is not None:
            try:
                receipt = self.printer.build_selfcheck(report, only="all")
                err = self.printer._send(receipt)
                out.append(f"Printing failed: {err}" if err
                           else "Printed the full check.")
            except Exception as exc:  # noqa: BLE001
                out.append(f"Printing failed: {type(exc).__name__}.")
        return "\n".join(out)
