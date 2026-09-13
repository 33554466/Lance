"""Play something from YouTube, out loud or on the screen.

Two decisions shape this whole file.

The first is that there is no browser in it. The obvious build — open Chromium
at a watch page — gets you a cookie wall, an ad, a player you cannot control,
and a window that has to be closed by hand. mpv plus yt-dlp gets you the video
with no ads at all, starting at the first frame, and an IPC socket that turns
"pause", "back thirty seconds" and "turn it down" into one-line commands. The
appliance already owns its audio; it should own its video too.

The second is that nothing here blocks. Resolving a YouTube URL takes a couple
of seconds, and a tool call that waits for it is two seconds of silence in a
conversation. So the search runs (fast, one page), the title comes back, mpv is
launched and left to catch up on its own. What you hear is "Playing Ocean's
Eleven, the poker scene" and then the video starts, which is the right order.

What this deliberately does NOT do is download anything. Everything is
streamed, nothing is written to disk, and the only thing that persists between
requests is which search results we are part-way through.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("assistant.media")

# yt-dlp's --print joins fields with whatever you put between them. Titles
# contain every printable character including tabs and pipes, so use a control
# character that cannot appear in one.
SEP = "\x1f"

# Recognise a spoken or pasted link, so "play this" with a URL in it skips the
# search entirely.
_URL = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/watch\?v=|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})")

# Words people put around a request that are not part of what they want to
# watch. Left in, they poison the search: "play me the video of the moon
# landing" returns videos about playing, and "pull up a clip about the battle
# of midway" is a worse query than "battle of midway".
#
# Done by walking words rather than with one large regular expression. The
# regex version was written first and was wrong in ways that were hard to see:
# optional groups next to a mandatory space meant "show me how to..." matched
# and "put on some..." did not, for reasons nobody could read off the pattern.
# A list you can print is a list you can fix.
#
# Longest first, so "pull up" wins over a bare verb and "play me" over "play".
_VERBS = (
    "can you pull up", "can you play", "can you put on", "can you show me",
    "pull up", "put on", "throw on", "queue up", "bring up", "fire up",
    "search for", "look up", "show me", "find me", "play me", "give me",
    "play", "show", "find", "search", "open", "watch",
)

# Politeness and pointing, stripped before the verb is looked for.
_PREFIX = ("hey", "ok", "okay", "lance", "please", "could you", "can you",
           "would you", "go ahead and", "i want to", "i'd like to",
           "lets", "let's")

# Stripped after the verb, while they keep coming. "of the" and "about the"
# both disappear this way without needing entries of their own.
_AFTER = frozenset("""
me us a an the some that this any
youtube yt video videos vid clip clips footage song songs track tracks
music movie film channel episode
of about for on by from called titled with something
""".split())

# Never strip more than this many. A run longer than it means the sentence is
# not shaped the way we assumed, and guessing further does more harm than
# leaving it alone.
_MAX_STRIPPED = 6


@dataclass(frozen=True)
class Hit:
    video_id: str
    title: str
    channel: str
    seconds: int | None
    live: bool = False

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def spoken_length(self) -> str:
        """How long it is, the way a person would say it."""
        if self.live:
            return "live"
        if not self.seconds:
            return ""
        m, s = divmod(int(self.seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h} hour{'s' if h != 1 else ''} {m} minutes" if m \
                else f"{h} hour{'s' if h != 1 else ''}"
        if m:
            return f"{m} minute{'s' if m != 1 else ''}"
        return f"{s} seconds"

    def describe(self) -> str:
        bits = [f"{self.title}"]
        if self.channel:
            bits.append(f"by {self.channel}")
        length = self.spoken_length()
        if length:
            bits.append(f"— {length}")
        return " ".join(bits)


def clean_query(raw: str) -> str:
    """Strip the asking off the front of what was asked for.

    Always returns something. Stripping a request down to nothing — "play" on
    its own, or "put on some music" if the word list were greedier — would
    search YouTube for an empty string, which quietly returns whatever it
    feels like. Falling back to the original text is worse in theory and far
    better in practice: you get an odd result instead of a random one.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    words = text.split()
    lowered = [w.lower().strip(",.!?") for w in words]

    # 1. Politeness, repeatedly — "hey Lance, could you please..."
    changed = True
    while changed and lowered:
        changed = False
        for pre in _PREFIX:
            n = len(pre.split())
            if lowered[:n] == pre.split():
                words, lowered = words[n:], lowered[n:]
                changed = True
                break

    # 2. Exactly one verb phrase, longest match first.
    for verb in sorted(_VERBS, key=lambda v: -len(v.split())):
        n = len(verb.split())
        if lowered[:n] == verb.split():
            words, lowered = words[n:], lowered[n:]
            break
    else:
        # No verb at the front means this is already the thing itself —
        # "stairway to heaven live 1973" — and must not be touched.
        return text

    # 3. Filler, while it keeps coming and while something is left.
    dropped = 0
    while (lowered and dropped < _MAX_STRIPPED
           and lowered[0] in _AFTER and len(lowered) > 1):
        words, lowered = words[1:], lowered[1:]
        dropped += 1

    return " ".join(words).strip() or text


class YouTube:
    """Search, via yt-dlp. No API key, no quota, no Google account."""

    def __init__(self, cfg: dict):
        m = cfg.get("media", {}) or {}
        self.bin = m.get("ytdlp_bin", "yt-dlp")
        self.timeout = float(m.get("search_timeout_seconds", 20))
        self.results = int(m.get("results", 5))

    def available(self) -> str | None:
        """None if usable, otherwise why not.

        Deliberately does NOT suggest apt. Ubuntu 24.04 ships yt-dlp
        2023.11.16, which is years behind YouTube and fails extraction on
        everything — and it fails with a confusing error rather than saying it
        is out of date, which is the worst possible way to be broken. The
        appliance already has a virtualenv; putting it there means one command
        to update it and an absolute path that does not depend on how the
        service was started.
        """
        if self.path() is None:
            return (f"{self.bin} is not installed, or is not on the service's "
                    f"PATH. Install it into the appliance's own virtualenv: "
                    f"~/assistant/.venv/bin/pip install -U yt-dlp . "
                    f"Do not use the apt package, it is from 2023 and no "
                    f"longer works.")
        return None

    def path(self) -> str | None:
        """Absolute path to yt-dlp, or None.

        Resolved rather than assumed because a systemd user service gets a
        minimal PATH — /usr/local/bin:/usr/bin:/bin — so anything in
        ~/.local/bin or a virtualenv is invisible to it even though it works
        perfectly in a login shell. That failure looks exactly like "not
        installed" from the outside.
        """
        if Path(self.bin).is_absolute():
            return self.bin if Path(self.bin).exists() else None
        found = shutil.which(self.bin)
        if found:
            return found
        # The obvious places, in the order they are worth trying.
        for candidate in (Path.home() / "assistant" / ".venv" / "bin" / self.bin,
                          Path.home() / ".local" / "bin" / self.bin,
                          Path("/snap/bin") / self.bin):
            if candidate.exists():
                return str(candidate)
        return None

    def search(self, query: str, limit: int | None = None) -> list[Hit]:
        limit = limit or self.results
        direct = _URL.search(query)
        if direct:
            # A link was given. Ask yt-dlp about that one video rather than
            # searching for its id as though it were words.
            return self._run([f"https://www.youtube.com/watch?v={direct.group(1)}"],
                             flat=False)
        q = clean_query(query)
        if not q:
            return []
        return self._run([f"ytsearch{limit}:{q}"], flat=True)

    def _run(self, targets: list[str], flat: bool) -> list[Hit]:
        fmt = SEP.join(["%(id)s", "%(title)s", "%(channel)s",
                        "%(duration)s", "%(live_status)s"])
        cmd = [self.path() or self.bin, "--ignore-config", "--no-warnings",
               "--skip-download", "--no-playlist", "--print", fmt]
        if flat:
            # One page of search results, no per-video resolution. This is the
            # difference between a second and fifteen.
            cmd.append("--flat-playlist")
        cmd += targets

        t0 = time.time()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=self.timeout)
        except subprocess.TimeoutExpired:
            log.warning("yt-dlp search timed out after %.0fs", self.timeout)
            return []
        if out.returncode != 0:
            log.warning("yt-dlp failed (%d): %s", out.returncode,
                        (out.stderr or "").strip()[:300])
            return []

        hits: list[Hit] = []
        for line in out.stdout.splitlines():
            parts = line.split(SEP)
            if len(parts) < 5 or not parts[0] or parts[0] == "NA":
                continue
            vid, title, channel, dur, live = parts[:5]
            try:
                seconds = int(float(dur))
            except (TypeError, ValueError):
                seconds = None
            hits.append(Hit(
                video_id=vid.strip(),
                title=(title or "").strip() or "Untitled",
                channel="" if channel in ("NA", "") else channel.strip(),
                seconds=seconds,
                live=live in ("is_live", "is_upcoming"),
            ))
        log.info("search returned %d hit(s) in %.1fs", len(hits),
                 time.time() - t0)
        return hits


class Player:
    """One mpv process, driven over a unix socket.

    mpv is started and NOT waited for. It resolves the stream itself through
    its ytdl hook, which takes a second or two; by then the assistant has
    already said what it is playing.
    """

    def __init__(self, cfg: dict, root: Path):
        m = cfg.get("media", {}) or {}
        self.bin = m.get("mpv_bin", "mpv")
        # Relative paths in config are relative to the appliance, never to
        # whatever directory the service happened to start in. A socket that
        # moves with the working directory is a control channel that works
        # when you test it by hand and silently does not under systemd.
        sock = Path(m.get("ipc_socket") or "data/mpv.sock").expanduser()
        self.sock = str(sock if sock.is_absolute() else root / sock)
        # mpv's own complaints, kept on disk. The first version of this file
        # sent stderr to /dev/null, and it passed --no-playlist — a yt-dlp
        # option that mpv does not have. mpv died on the spot every single
        # time, printed exactly why, and nobody could see it. All that reached
        # the surface was "mpv never answered", which is true and useless.
        self.log = str(Path(self.sock).with_name("mpv.log"))
        self.display = m.get("display", (cfg.get("desktop", {}) or {})
                             .get("display", ":0"))
        self.audio_device = m.get("audio_device", "auto")
        self.volume = int(m.get("volume", 85))
        self.duck_volume = int(m.get("duck_volume", 25))
        self.max_height = int(m.get("max_height", 1080))
        self.seek_seconds = int(m.get("seek_seconds", 30))
        self.duck_timeout = float(m.get("duck_timeout_seconds", 180))
        # mpv resolves the stream itself, through its own ytdl hook, which
        # looks for yt-dlp on PATH. A yt-dlp living in the appliance's
        # virtualenv is not on the service's PATH, so search would work and
        # PLAYBACK would fail — the single most confusing way for this to
        # break. Tell mpv exactly where it is instead of hoping.
        self.ytdlp_bin = m.get("ytdlp_bin", "yt-dlp")

        # Draw the assistant's state on top of the video.
        #
        # Fullscreen video covers the dashboard, and the dashboard is the only
        # thing that says whether you have been heard. Without this you are
        # talking to a screen showing somebody else's video with no idea
        # whether anything is listening — which is the moment an appliance
        # stops feeling like an appliance.
        #
        # It goes through the same IPC socket as every other control, so there
        # is no second window to keep on top of the first.
        self.osd_on = bool(m.get("osd", True))
        self.osd_size = int(m.get("osd_font_size", 34))

        self.proc: subprocess.Popen | None = None
        self.now: Hit | None = None
        self.audio_only = False
        self._ducked_at: float | None = None
        self._req = 0
        self._lock = threading.Lock()

    # -- availability -------------------------------------------------

    def available(self) -> str | None:
        if shutil.which(self.bin) is None:
            return f"{self.bin} is not installed. sudo apt install -y mpv"
        return None

    # -- process ------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tick(self) -> None:
        """Housekeeping, safe to call on a one-second timer.

        Two jobs. Notice that a video ended on its own, so `media_status` does
        not claim something is playing when the screen went back to the
        dashboard five minutes ago. And lift a duck that was never lifted —
        without this, one dropped websocket message leaves every video at a
        murmur until the next restart, and nothing about that failure tells
        you what happened.
        """
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            if code:
                # A non-zero exit is mpv refusing, not a video ending. Put its
                # own words in the log rather than leaving "it did not work".
                log.warning("mpv exited %d — %s", code, self.why_it_died()
                            or "no output captured")
            else:
                log.info("playback finished: %s",
                         self.now.title if self.now else "?")
            self.proc, self.now = None, None
            self._ducked_at = None
            return
        if (self._ducked_at is not None
                and time.time() - self._ducked_at > self.duck_timeout):
            log.warning("duck outlived its welcome — restoring volume")
            self.unduck()

    def command(self, hit: Hit, audio_only: bool) -> list[str]:
        """Build the mpv command line. Separate from start() so it can be
        tested without launching anything — see tests/test_media.py, which
        checks every flag against the list of options that belong to yt-dlp
        rather than to mpv. That is the mistake this function already made
        once, and it cost an afternoon."""
        cmd = [
            self.bin, "--no-terminal", "--msg-level=all=warn",
            f"--input-ipc-server={self.sock}",
            "--idle=no", "--keep-open=no",
            f"--volume={self.volume}",
        ]
        if self.audio_device and self.audio_device != "auto":
            cmd.append(f"--audio-device={self.audio_device}")

        ytdl = YouTube({"media": {"ytdlp_bin": self.ytdlp_bin}}).path()
        if ytdl:
            cmd.append(f"--script-opts=ytdl_hook-ytdl_path={ytdl}")
        else:
            log.warning("cannot find %s to hand to mpv — playback will "
                        "probably fail even though search worked",
                        self.ytdlp_bin)

        if audio_only:
            # Nothing on screen, nothing decoded. A four-hour mix costs
            # almost no CPU and leaves the dashboard where it was.
            cmd += ["--no-video", "--ytdl-format=bestaudio/best"]
        else:
            cmd += [
                "--fullscreen", "--ontop", "--no-osc",
                "--cursor-autohide=always", "--hwdec=auto-safe",
                # Top-right, out of the way of subtitles and of whatever is
                # happening in the middle of the frame.
                f"--osd-font-size={self.osd_size}",
                "--osd-align-x=right", "--osd-align-y=top",
                "--osd-margin-x=40", "--osd-margin-y=30",
                "--osd-color=#FFFFFFFF", "--osd-border-color=#C0000000",
                "--osd-border-size=3",
                f"--ytdl-format=bestvideo[height<=?{self.max_height}]"
                f"+bestaudio/best[height<=?{self.max_height}]/best",
            ]

        # "Do not expand a playlist" is a YT-DLP instruction, so it travels
        # inside --ytdl-raw-options. Passed to mpv directly as --no-playlist it
        # is a fatal unknown option, which is exactly what went wrong.
        cmd.append("--ytdl-raw-options=no-playlist=")
        cmd.append(hit.url)
        return cmd

    def why_it_died(self, lines: int = 6) -> str:
        """The tail of mpv's stderr, for when it will not start."""
        try:
            text = Path(self.log).read_text(errors="replace").strip()
        except OSError:
            return ""
        if not text:
            return ""
        return " / ".join(text.splitlines()[-lines:])

    def start(self, hit: Hit, audio_only: bool) -> None:
        self.stop()
        cmd = self.command(hit, audio_only)

        env = dict(os.environ)
        if not audio_only and self.display:
            env["DISPLAY"] = self.display

        # Remove a stale socket from a process that was killed rather than
        # asked to quit, or the new mpv refuses to bind and every control
        # silently does nothing.
        try:
            os.unlink(self.sock)
        except OSError:
            pass
        Path(self.sock).parent.mkdir(parents=True, exist_ok=True)

        log.info("mpv %s: %s", "audio" if audio_only else "video", hit.title)
        log.debug("mpv command: %s", " ".join(cmd))
        try:
            errlog = open(self.log, "w")
        except OSError:
            errlog = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL, stderr=errlog,
            start_new_session=True,
        )
        if errlog is not subprocess.DEVNULL:
            errlog.close()          # the child holds its own handle now
        self.now = hit
        self.audio_only = audio_only
        self._ducked_at = None

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            if not self._ipc(["quit"], wait=1.5)[0]:
                # The socket was not answering. Ask the process directly.
                self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            os.unlink(self.sock)
        except OSError:
            pass
        self.proc, self.now = None, None
        self._ducked_at = None

    # -- ipc ----------------------------------------------------------

    def _ipc(self, command: list, wait: float = 3.0):
        """Send one command and return (ok, data). Never raises.

        Everything on the far side is best-effort by design. mpv may still be
        resolving the URL and not listening on the socket yet, or it may have
        exited because the video ended. Neither is worth an exception — a
        control that arrives a moment too early should be a quiet no-op, not a
        stack trace read out loud.

        Replies are matched by request_id. mpv interleaves unsolicited events
        on the same socket, so reading "the next line" gets you a property
        change notification about half the time.
        """
        if not self.playing:
            return False, None

        self._req += 1
        rid = self._req
        line = json.dumps({"command": command, "request_id": rid}) + "\n"
        deadline = time.time() + wait

        with self._lock:
            while time.time() < deadline:
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.settimeout(max(0.3, deadline - time.time()))
                        s.connect(self.sock)
                        s.sendall(line.encode())
                        with s.makefile("r", encoding="utf-8") as fh:
                            for raw in fh:
                                try:
                                    msg = json.loads(raw)
                                except ValueError:
                                    continue
                                if msg.get("request_id") != rid:
                                    continue          # an event, not our reply
                                ok = msg.get("error") == "success"
                                if not ok:
                                    log.debug("mpv refused %s: %s",
                                              command, msg.get("error"))
                                return ok, msg.get("data")
                    return False, None
                except (FileNotFoundError, ConnectionRefusedError, OSError):
                    # mpv has not created the socket yet. It is still starting.
                    time.sleep(0.15)
        log.debug("mpv did not answer %s within %.1fs", command, wait)
        return False, None

    def get(self, prop: str):
        ok, data = self._ipc(["get_property", prop])
        return data if ok else None

    def set(self, prop: str, value) -> bool:
        return self._ipc(["set_property", prop, value])[0]

    # -- controls -----------------------------------------------------

    def pause(self, on: bool) -> bool:
        return self.set("pause", bool(on))

    def is_paused(self) -> bool:
        return self.get("pause") is True

    def seek(self, seconds: float) -> bool:
        return self._ipc(["seek", seconds, "relative"])[0]

    def restart(self) -> bool:
        return self._ipc(["seek", 0, "absolute"])[0]

    def set_volume(self, level: int) -> bool:
        self.volume = max(0, min(130, int(level)))
        return self.set("volume", self.volume)

    def nudge_volume(self, delta: int) -> bool:
        return self.set_volume(self.volume + delta)

    def duck(self, on: bool) -> None:
        """Drop to a murmur while the assistant talks, then come back.

        Called from the websocket handler on every state change, so it must be
        cheap and idempotent. It also must never raise: a video is not
        important enough to take a conversation down with it.
        """
        try:
            if on:
                if not self.playing or self._ducked_at is not None:
                    return
                self._ducked_at = time.time()
                self.set("volume", self.duck_volume)
            else:
                if self._ducked_at is None:
                    return
                self._ducked_at = None
                if self.playing:
                    self.set("volume", self.volume)
        except Exception:  # noqa: BLE001
            log.debug("duck failed — ignoring", exc_info=True)

    def unduck(self) -> None:
        self.duck(False)

    def osd(self, text: str, seconds: float = 3.0) -> bool:
        """Put a line on the video. Silent no-op when nothing is playing."""
        if not self.osd_on or not self.playing or self.audio_only:
            return False
        return self._ipc(["show-text", text, int(max(0.001, seconds) * 1000)],
                         wait=1.0)[0]

    def position(self) -> tuple[float | None, float | None]:
        return self.get("time-pos"), self.get("duration")


def spoken_clock(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"
    if m:
        return f"{m} minute{'s' if m != 1 else ''} {s} second{'s' if s != 1 else ''}"
    return f"{s} second{'s' if s != 1 else ''}"


# The whole surface, in three tools rather than nine. Every tool schema is
# sent on every request, so a verb per action would cost real money on a
# feature that is used a few times a day — and a model that can pick
# "media_control" can pick an action out of a list just as reliably.
ACTIONS = ("pause", "resume", "stop", "restart", "forward", "back",
           "next", "louder", "quieter", "volume")

# What to draw over a playing video for each state the audio service reports.
# Durations are generous on purpose: the message must outlast the state, or it
# vanishes while you are still mid-sentence and you are back to guessing. Each
# new state overwrites the last, and idle clears.
OSD_STATES = {
    "listening":    ("●  Listening", 25.0),
    "transcribing": ("●  Thinking",  20.0),
    "idle":         ("",              0.001),
}


class MediaTools:
    """Play things from YouTube, and control what is playing."""

    def __init__(self, cfg: dict, root: Path):
        m = cfg.get("media", {}) or {}
        self.enabled = bool(m.get("enabled", False))
        self.yt = YouTube(cfg)
        self.player = Player(cfg, root)
        self.step = int(m.get("volume_step", 15))
        # The search results for whatever was last asked for, so "not that
        # one" costs nothing — the alternatives are already in hand.
        self._results: list[Hit] = []
        self._idx = 0
        self._audio_only = False

    # -- plumbing -----------------------------------------------------

    def tick(self) -> None:
        if self.enabled:
            self.player.tick()

    def duck(self, on: bool) -> None:
        if self.enabled:
            self.player.duck(on)

    def show_state(self, state: str) -> None:
        """Mirror the assistant's state onto the video.

        Called from the websocket handler on every state change, so — like
        duck() — it must be cheap and must never raise. A status light is not
        worth taking a conversation down for.
        """
        if not self.enabled:
            return
        try:
            line = OSD_STATES.get(state)
            if line is not None:
                self.player.osd(*line)
        except Exception:  # noqa: BLE001
            log.debug("osd state failed — ignoring", exc_info=True)

    def show_heard(self, text: str) -> None:
        """Echo what was understood, briefly.

        Over fullscreen video this is doing the job the transcript line on the
        dashboard normally does: confirming she heard the words you actually
        said, not ones that merely sound like them.
        """
        if not self.enabled or not text:
            return
        try:
            trimmed = text if len(text) <= 70 else text[:67].rstrip() + "..."
            self.player.osd(f"\u201c{trimmed}\u201d", 4.0)
        except Exception:  # noqa: BLE001
            log.debug("osd transcript failed — ignoring", exc_info=True)

    def close(self) -> None:
        if self.enabled:
            self.player.stop()

    def _missing(self) -> str | None:
        return self.yt.available() or self.player.available()

    # -- schemas ------------------------------------------------------

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [
            {
                "name": "play_media",
                "description": (
                    "Find something on YouTube and play it. Use for any "
                    "request to play, put on, pull up or show a video, song, "
                    "clip or channel. Pass what they actually want to watch, "
                    "not their whole sentence: 'pull up a video on how to "
                    "sharpen a chisel' is the query 'how to sharpen a "
                    "chisel'. Set audio_only when they ask for music, a "
                    "podcast, or say 'put on' rather than 'show me' — it "
                    "leaves the dashboard on screen."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search YouTube for.",
                        },
                        "audio_only": {
                            "type": "boolean",
                            "description": (
                                "Sound only, screen untouched. Default false."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "media_control",
                "description": (
                    "Control whatever is already playing. 'next' plays the "
                    "next search result, for when the wrong video started. "
                    "'forward' and 'back' take seconds; 'volume' takes a "
                    "level from 0 to 100."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "seconds": {
                            "type": "integer",
                            "description": (
                                "How far to skip, for forward and back. "
                                "Defaults to the configured step."
                            ),
                        },
                        "level": {
                            "type": "integer",
                            "description": "Volume 0-100, for 'volume'.",
                        },
                    },
                    "required": ["action"],
                },
            },
            {
                "name": "media_status",
                "description": (
                    "What is playing, and how far through it is. Use when "
                    "asked what this is, what's on, or how long is left."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    # -- dispatch -----------------------------------------------------

    def run_sync(self, name: str, args: dict) -> str | None:
        if not self.enabled or name not in (
                "play_media", "media_control", "media_status"):
            return None
        missing = self._missing()
        if missing:
            return f"I cannot play anything: {missing}"
        if name == "play_media":
            return self._play(args)
        if name == "media_control":
            return self._control(args)
        return self._status()

    # -- handlers -----------------------------------------------------

    def _play(self, args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Nothing to search for."
        audio_only = bool(args.get("audio_only"))

        hits = self.yt.search(query)
        if not hits:
            return (f"I could not find anything on YouTube for {query!r}. "
                    f"Say it a different way and I will try again.")

        self._results, self._idx, self._audio_only = hits, 0, audio_only
        return self._start(hits[0], audio_only)

    def _start(self, hit: Hit, audio_only: bool) -> str:
        self.player.start(hit, audio_only)
        where = "Playing" if audio_only else "On screen"
        return (f"{where}: {hit.describe()}. "
                f"Say 'not that one' for the next result.")

    def _control(self, args: dict) -> str:
        action = (args.get("action") or "").strip().lower()
        p = self.player

        # 'next' is the only control that is meaningful with nothing playing —
        # the video may have ended while he was deciding it was wrong.
        if action == "next":
            if self._idx + 1 >= len(self._results):
                return ("That was the last result I had. Ask for it a "
                        "different way and I will search again.")
            self._idx += 1
            return self._start(self._results[self._idx], self._audio_only)

        if not p.playing:
            return "Nothing is playing."

        if action == "stop":
            title = p.now.title if p.now else "it"
            p.stop()
            return f"Stopped {title}."
        if action == "pause":
            return "Paused." if p.pause(True) else "I could not pause it."
        if action == "resume":
            return "Carrying on." if p.pause(False) else "I could not resume it."
        if action == "restart":
            return "Back to the start." if p.restart() \
                else "I could not seek it."
        if action in ("forward", "back"):
            step = int(args.get("seconds") or p.seek_seconds)
            delta = step if action == "forward" else -step
            if not p.seek(delta):
                return "I could not seek it."
            return (f"{'Forward' if delta > 0 else 'Back'} "
                    f"{spoken_clock(abs(delta))}.")
        if action in ("louder", "quieter"):
            delta = self.step if action == "louder" else -self.step
            if not p.nudge_volume(delta):
                return "I could not change the volume."
            return f"Volume {p.volume}."
        if action == "volume":
            level = args.get("level")
            if level is None:
                return "What level?"
            if not p.set_volume(int(level)):
                return "I could not change the volume."
            return f"Volume {p.volume}."
        return f"I do not know how to {action}."

    def _status(self) -> str:
        p = self.player
        if not p.playing or p.now is None:
            return "Nothing is playing."
        pos, total = p.position()
        line = p.now.describe()
        if p.is_paused():
            line += ", paused"
        if pos is not None and total:
            left = max(0.0, total - pos)
            line += f". {spoken_clock(left)} left"
        elif pos is not None:
            line += f". {spoken_clock(pos)} in"
        return line + "."
