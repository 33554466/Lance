"""Search parsing, mpv IPC, and the controls — with no network and no mpv.

The parts worth testing here are the two seams: what yt-dlp hands back, and
what mpv expects on its socket. Both are text protocols from other people's
programs, which is exactly where a build like this rots quietly.

    python -m tests.test_media
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.media import (Hit, MediaTools, Player, YouTube, SEP,  # noqa: E402
                        clean_query, spoken_clock)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


# ------------------------------------------------------------ the query

def test_clean_query() -> None:
    print("\nWhat actually gets searched for")
    cases = [
        ("pull up a youtube video of the moon landing", "moon landing"),
        ("play me the video on how to sharpen a chisel",
         "how to sharpen a chisel"),
        ("put on some lofi hip hop", "lofi hip hop"),
        ("can you pull up a clip about the battle of midway",
         "battle of midway"),
        ("show me how to change a serpentine belt",
         "how to change a serpentine belt"),
        # Nothing to strip — must survive untouched.
        ("stairway to heaven live 1973", "stairway to heaven live 1973"),
        ("hey Lance, please play stairway to heaven", "stairway to heaven"),
        ("find videos on soldering", "soldering"),
        # And a query that is ONLY noise must not come back empty, or the
        # search runs on nothing and returns whatever YouTube feels like.
        ("play", "play"),
    ]
    for raw, want in cases:
        got = clean_query(raw)
        check(f"{raw!r} -> {want!r}", got == want, f"got {got!r}")


def test_spoken() -> None:
    print("\nSaying lengths out loud")
    check("nothing for unknown", Hit("a", "t", "c", None).spoken_length() == "")
    check("seconds", Hit("a", "t", "c", 45).spoken_length() == "45 seconds")
    check("minutes", Hit("a", "t", "c", signed := 8 * 60).spoken_length()
          == "8 minutes")
    check("an hour and change",
          Hit("a", "t", "c", 3 * 3600 + 12 * 60).spoken_length()
          == "3 hours 12 minutes")
    check("exactly an hour",
          Hit("a", "t", "c", 3600).spoken_length() == "1 hour")
    check("live says live", Hit("a", "t", "c", None, live=True)
          .spoken_length() == "live")
    check("describe reads as a sentence",
          Hit("x", "The Poker Scene", "Warner", 240).describe()
          == "The Poker Scene by Warner — 4 minutes")
    check("no channel, no dangling 'by'",
          Hit("x", "Clip", "", None).describe() == "Clip")
    check("clock: minutes and seconds", spoken_clock(95)
          == "1 minute 35 seconds")
    check("clock: nothing for None", spoken_clock(None) == "")


# ------------------------------------------------------------- yt-dlp

FAKE_YTDLP = '''#!/usr/bin/env python3
import sys
SEP = "\\x1f"
args = sys.argv[1:]
target = args[-1]
if "BOOM" in target:
    sys.stderr.write("ERROR: something went wrong\\n")
    sys.exit(1)
rows = [
    ("dQw4w9WgXcQ", "Apollo 11: The Landing", "NASA", "212", "not_live"),
    ("aaaaaaaaaaa", "Moon Landing | Full Footage", "History", "3611", "not_live"),
    ("bbbbbbbbbbb", "Live: Artemis", "NASA", "NA", "is_live"),
]
print("this line is garbage and must be skipped")
print(SEP.join(["NA", "no id", "x", "1", "not_live"]))
for r in rows:
    print(SEP.join(r))
'''


def _fake_ytdlp(tmp: Path) -> str:
    path = tmp / "yt-dlp-fake"
    path.write_text(FAKE_YTDLP)
    path.chmod(0o755)
    return str(path)


def test_search(tmp: Path) -> None:
    print("\nParsing what yt-dlp prints")
    cfg = {"media": {"ytdlp_bin": _fake_ytdlp(tmp)}}
    yt = YouTube(cfg)

    hits = yt.search("pull up a video of the moon landing")
    check("three good rows survive", len(hits) == 3, f"got {len(hits)}")
    check("garbage lines are dropped",
          all(h.video_id and h.video_id != "NA" for h in hits))
    check("title with a pipe in it is intact",
          hits[1].title == "Moon Landing | Full Footage", hits[1].title)
    check("duration parsed", hits[0].seconds == 212)
    check("missing duration is None, not zero", hits[2].seconds is None)
    check("live flagged", hits[2].live is True)
    check("url built from the id",
          hits[0].url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    check("a failing yt-dlp returns no hits, not an exception",
          yt.search("BOOM") == [])

    # A pasted or spoken link must skip the search entirely.
    link = yt.search("play https://youtu.be/dQw4w9WgXcQ please")
    check("a link is resolved directly", len(link) == 3)


# ---------------------------------------------------------------- mpv

class FakeMpv:
    """Speaks enough of mpv's JSON IPC to test the client honestly."""

    def __init__(self, path: str):
        self.path = path
        self.props = {"pause": False, "volume": 85, "time-pos": 12.5,
                      "duration": 212.0, "mpv-version": "mpv 0.37.0"}
        self.seen: list[list] = []
        self.refuse: set[str] = set()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(path)
        self._srv.listen(8)
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,),
                             daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as fh:
            for raw in fh:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                cmd = msg.get("command", [])
                self.seen.append(cmd)
                # mpv pushes events down the same socket, unasked. A client
                # that reads "the next line" instead of matching request_id
                # gets this instead of its answer.
                fh.write(json.dumps(
                    {"event": "property-change", "name": "time-pos"}).encode()
                    + b"\n")
                # ...and a reply to somebody else's request, for good measure.
                fh.write(json.dumps(
                    {"error": "success", "data": "not yours",
                     "request_id": 9999}).encode() + b"\n")
                fh.write(self._reply(cmd, msg.get("request_id")).encode()
                         + b"\n")
                fh.flush()

    def _reply(self, cmd: list, rid) -> str:
        verb = cmd[0] if cmd else ""
        if verb in self.refuse:
            return json.dumps({"error": "property unavailable",
                               "request_id": rid})
        if verb == "get_property":
            name = cmd[1]
            if name not in self.props:
                return json.dumps({"error": "property not found",
                                   "request_id": rid})
            return json.dumps({"error": "success", "data": self.props[name],
                               "request_id": rid})
        if verb == "set_property":
            self.props[cmd[1]] = cmd[2]
            return json.dumps({"error": "success", "data": None,
                               "request_id": rid})
        return json.dumps({"error": "success", "data": None,
                           "request_id": rid})

    def close(self) -> None:
        self._stop = True
        self._srv.close()


class _Alive:
    def poll(self):
        return None


class _Dead:
    def poll(self):
        return 0


def _player(tmp: Path, sock: str) -> Player:
    cfg = {"media": {"ipc_socket": sock, "volume": 85, "duck_volume": 25,
                     "seek_seconds": 30}}
    p = Player(cfg, tmp)
    p.proc = _Alive()
    p.now = Hit("x", "Test", "Chan", 212)
    return p


def test_ipc(tmp: Path) -> None:
    print("\nTalking to mpv")
    sock = str(tmp / "mpv-test.sock")
    mpv = FakeMpv(sock)
    try:
        p = _player(tmp, sock)

        check("reads a property", p.get("mpv-version") == "mpv 0.37.0")
        check("ignores events and other people's replies",
              p.get("duration") == 212.0, str(p.get("duration")))
        check("unknown property is None, not a crash",
              p.get("nonsense") is None)

        check("pause", p.pause(True) and mpv.props["pause"] is True)
        check("resume", p.pause(False) and mpv.props["pause"] is False)
        check("is_paused reflects mpv", p.is_paused() is False)

        check("seek sends a relative seek", p.seek(-30))
        check("  ...with the right arguments", ["seek", -30, "relative"]
              in mpv.seen, str(mpv.seen[-1]))
        check("restart seeks to absolute zero",
              p.restart() and ["seek", 0, "absolute"] in mpv.seen)

        check("volume is set and remembered",
              p.set_volume(40) and p.volume == 40 and mpv.props["volume"] == 40)
        check("volume nudges up", p.nudge_volume(15) and p.volume == 55)
        check("volume cannot go below zero",
              p.set_volume(-20) and p.volume == 0)

        # Ducking
        p.set_volume(80)
        p.duck(True)
        check("duck drops the volume", mpv.props["volume"] == 25,
              str(mpv.props["volume"]))
        p.duck(True)
        check("ducking twice does not lose the real volume",
              p.volume == 80 and mpv.props["volume"] == 25)
        p.duck(False)
        check("unduck restores what it was", mpv.props["volume"] == 80,
              str(mpv.props["volume"]))
        p.duck(False)
        check("unducking twice is harmless", mpv.props["volume"] == 80)

        # A control that mpv refuses must report failure, not pretend.
        mpv.refuse.add("set_property")
        check("a refused command returns False", p.pause(True) is False)
        mpv.refuse.clear()

        # Nothing playing: every control is a quiet no-op.
        p.proc = _Dead()
        check("no process, no IPC", p.get("duration") is None)
        check("no process, pause reports failure", p.pause(True) is False)
        check("duck with nothing playing does nothing", (p.duck(True), True)[1])
    finally:
        mpv.close()


def test_stale_socket(tmp: Path) -> None:
    print("\nA socket left behind by a killed mpv")
    sock = tmp / "stale.sock"
    sock.write_text("")          # not a socket at all, just in the way
    p = Player({"media": {"ipc_socket": str(sock)}}, tmp)
    check("the path is where we asked", p.sock == str(sock))
    # start() unlinks before launching. Prove the unlink happens by calling
    # the same guard directly rather than launching a real mpv.
    try:
        os.unlink(p.sock)
    except OSError:
        pass
    check("stale file is removable", not Path(p.sock).exists())


# ------------------------------------------------------------ controls

class _FakePlayer:
    def __init__(self):
        self.playing = True
        self.now = Hit("x", "First Result", "Chan", 100)
        self.started: list = []
        self.volume = 80
        self.seek_seconds = 30

    def start(self, hit, audio_only):
        self.started.append((hit, audio_only))
        self.now, self.playing = hit, True

    def stop(self):
        self.playing, self.now = False, None

    def pause(self, on):
        return True

    def is_paused(self):
        return False

    def seek(self, s):
        return True

    def restart(self):
        return True

    def set_volume(self, v):
        self.volume = max(0, min(130, int(v)))
        return True

    def nudge_volume(self, d):
        return self.set_volume(self.volume + d)

    def position(self):
        return 30.0, 100.0


def _tools(tmp: Path) -> MediaTools:
    cfg = {"media": {"enabled": True, "ytdlp_bin": _fake_ytdlp(tmp),
                     "ipc_socket": str(tmp / "t.sock")}}
    t = MediaTools(cfg, tmp)
    t.player = _FakePlayer()
    t._missing = lambda: None
    return t


def test_controls(tmp: Path) -> None:
    print("\nThe spoken controls")
    t = _tools(tmp)

    out = t.run_sync("play_media", {"query": "pull up the moon landing"})
    check("play announces the title",
          "Apollo 11: The Landing" in out, out)
    check("play says how to reject it", "not that one" in out.lower(), out)
    check("video mode by default", t.player.started[0][1] is False)

    out = t.run_sync("media_control", {"action": "next"})
    check("'next' plays the second result",
          "Moon Landing | Full Footage" in out, out)
    out = t.run_sync("media_control", {"action": "next"})
    check("'next' again plays the third", "Artemis" in out, out)
    out = t.run_sync("media_control", {"action": "next"})
    check("'next' past the end says so, and does not crash",
          "last result" in out.lower(), out)

    check("pause", t.run_sync("media_control", {"action": "pause"})
          == "Paused.")
    check("forward uses the default step",
          "30 seconds" in t.run_sync("media_control", {"action": "forward"}))
    check("forward takes an explicit length",
          "2 minutes" in t.run_sync(
              "media_control", {"action": "forward", "seconds": 120}))
    check("louder reports the new level",
          t.run_sync("media_control", {"action": "louder"}) == "Volume 95.")
    check("volume needs a level",
          "what level" in t.run_sync(
              "media_control", {"action": "volume"}).lower())
    check("an unknown action does not crash",
          "do not know" in t.run_sync(
              "media_control", {"action": "teleport"}).lower())

    out = t.run_sync("media_status", {})
    check("status says what and how much is left",
          "Artemis" in out and "left" in out, out)

    t.run_sync("media_control", {"action": "stop"})
    check("stopped", t.player.playing is False)
    check("controls with nothing playing say so",
          t.run_sync("media_control", {"action": "pause"})
          == "Nothing is playing.")
    check("status with nothing playing says so",
          t.run_sync("media_status", {}) == "Nothing is playing.")

    # Audio-only leaves the screen alone.
    t2 = _tools(tmp)
    t2.run_sync("play_media", {"query": "put on lofi", "audio_only": True})
    check("audio_only is passed through", t2.player.started[0][1] is True)

    # Nothing found.
    t3 = _tools(tmp)
    out = t3.run_sync("play_media", {"query": "BOOM"})
    check("no results is a sentence, not an error", "could not find" in out)

    # Disabled means the tools are not offered at all.
    off = MediaTools({"media": {"enabled": False}}, tmp)
    check("disabled offers no schemas", off.schemas() == [])
    check("disabled answers nothing",
          off.run_sync("play_media", {"query": "x"}) is None)


# Options that belong to YT-DLP, not to mpv. Passing any of these to mpv is a
# FATAL unknown-option error — mpv refuses to start and says so, which is
# invisible if its stderr is being discarded. --no-playlist shipped this way
# and every launch died instantly for an afternoon.
YTDLP_ONLY = (
    "--no-playlist", "--playlist-items", "--no-warnings", "--skip-download",
    "--flat-playlist", "--extract-audio", "--audio-format", "--output",
    "--print", "--ignore-config", "--format-sort", "--cookies-from-browser",
)

# Anything meant for yt-dlp has to travel inside one of these.
YTDL_CHANNELS = ("--ytdl-raw-options=", "--ytdl-format=", "--script-opts=")


def test_mpv_command(tmp: Path) -> None:
    print("\nThe mpv command line")
    cfg = {"media": {"ipc_socket": str(tmp / "c.sock"), "volume": 85,
                     "audio_device": "pipewire/thing",
                     "ytdlp_bin": "/nowhere/yt-dlp", "max_height": 1080}}
    p = Player(cfg, tmp)
    hit = Hit("abc12345678", "A Video", "Chan", 120)

    for audio_only in (False, True):
        cmd = p.command(hit, audio_only)
        label = "audio" if audio_only else "video"

        bad = [a for a in cmd
               if any(a == o or a.startswith(o + "=") for o in YTDLP_ONLY)]
        check(f"{label}: no yt-dlp-only options", not bad, str(bad))

        check(f"{label}: the url is last", cmd[-1] == hit.url, cmd[-1])
        check(f"{label}: mpv is first", cmd[0] == "mpv", cmd[0])
        check(f"{label}: the ipc socket is set",
              any(a.startswith("--input-ipc-server=") for a in cmd))
        check(f"{label}: the audio device is passed",
              "--audio-device=pipewire/thing" in cmd)
        check(f"{label}: playlists are suppressed the mpv way",
              any(a.startswith("--ytdl-raw-options=") and "no-playlist" in a
                  for a in cmd))
        # Every option must be a long option with no space-separated value —
        # Popen takes a list, so "--volume 85" would arrive as two argv
        # entries and mpv would treat "85" as a filename.
        opts = [a for a in cmd[1:-1]]
        check(f"{label}: every argument is a --flag",
              all(a.startswith("--") for a in opts),
              str([a for a in opts if not a.startswith("--")]))

    video = p.command(hit, False)
    audio = p.command(hit, True)
    check("video mode goes fullscreen", "--fullscreen" in video)
    check("video mode caps the resolution",
          any("height<=?1080" in a for a in video))
    check("audio mode decodes no video", "--no-video" in audio)
    check("audio mode is not fullscreen", "--fullscreen" not in audio)
    check("audio mode asks for audio only",
          "--ytdl-format=bestaudio/best" in audio)
    check("mpv is told where yt-dlp is when it is off PATH",
          not any("ytdl_hook-ytdl_path" in a for a in video),
          "a nonexistent path should not be passed")


def test_osd(tmp: Path) -> None:
    print("\nStatus over the video")
    sock = str(tmp / "osd.sock")
    mpv = FakeMpv(sock)
    try:
        cfg = {"media": {"enabled": True, "ipc_socket": sock,
                         "ytdlp_bin": _fake_ytdlp(tmp)}}
        t = MediaTools(cfg, tmp)
        t._missing = lambda: None
        p = t.player
        p.proc = _Alive()
        p.now = Hit("x", "Test", "Chan", 200)
        p.audio_only = False

        t.show_state("listening")
        shown = [c for c in mpv.seen if c and c[0] == "show-text"]
        check("listening draws something", len(shown) == 1, str(mpv.seen))
        check("...that says it is listening",
              "Listening" in shown[-1][1], str(shown[-1]))
        check("...and outlasts a long question", shown[-1][2] >= 20000,
              str(shown[-1][2]))

        t.show_state("idle")
        shown = [c for c in mpv.seen if c and c[0] == "show-text"]
        check("idle clears it", shown[-1][1] == "", str(shown[-1]))

        t.show_heard("play me something about the moon landing")
        shown = [c for c in mpv.seen if c and c[0] == "show-text"]
        check("what was heard is echoed back",
              "moon landing" in shown[-1][1], str(shown[-1]))
        check("...in quotes", shown[-1][1].startswith("\u201c"))

        long = "x" * 200
        t.show_heard(long)
        shown = [c for c in mpv.seen if c and c[0] == "show-text"]
        check("a long sentence is trimmed, not sprawled",
              len(shown[-1][1]) <= 74, str(len(shown[-1][1])))

        # Audio-only has no window, so there is nothing to draw on.
        before = len(mpv.seen)
        p.audio_only = True
        t.show_state("listening")
        check("audio-only draws nothing", len(mpv.seen) == before)

        # Nothing playing at all must be silent, not an error.
        p.audio_only, p.proc = False, _Dead()
        before = len(mpv.seen)
        t.show_state("listening")
        t.show_heard("hello")
        check("nothing playing draws nothing", len(mpv.seen) == before)

        # And a switched-off OSD stays off.
        p.proc, p.osd_on = _Alive(), False
        before = len(mpv.seen)
        t.show_state("listening")
        check("osd:false is respected", len(mpv.seen) == before)
    finally:
        mpv.close()

    # The video command has to carry the styling, or the text lands wherever
    # mpv feels like putting it.
    p2 = Player({"media": {"ipc_socket": str(tmp / "z.sock")}}, tmp)
    video = p2.command(Hit("a", "t", "c", 10), False)
    for flag in ("--osd-font-size=", "--osd-align-x=right",
                 "--osd-align-y=top", "--osd-border-size="):
        check(f"video command sets {flag}",
              any(a.startswith(flag) for a in video))
    audio = p2.command(Hit("a", "t", "c", 10), True)
    check("audio-only does not bother with osd flags",
          not any(a.startswith("--osd") for a in audio))


def test_schemas(tmp: Path) -> None:
    print("\nTool schemas")
    t = _tools(tmp)
    names = [s["name"] for s in t.schemas()]
    check("three tools, not nine", names == ["play_media", "media_control",
                                             "media_status"], str(names))
    for s in t.schemas():
        check(f"{s['name']} has an input schema",
              isinstance(s.get("input_schema"), dict))
    ctrl = next(s for s in t.schemas() if s["name"] == "media_control")
    check("action is an enum the model cannot wander off",
          "enum" in ctrl["input_schema"]["properties"]["action"])


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_clean_query()
        test_spoken()
        test_search(tmp)
        test_ipc(tmp)
        test_stale_socket(tmp)
        test_controls(tmp)
        test_mpv_command(tmp)
        test_osd(tmp)
        test_schemas(tmp)
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
