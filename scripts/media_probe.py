#!/usr/bin/env python3
"""Prove the video path works, one layer at a time.

Same idea as the printer probe: find out WHICH part is broken rather than
being told "it didn't play". Run it on the appliance, over SSH is fine for
everything except the last step.

    python3 scripts/media_probe.py                 # check everything
    python3 scripts/media_probe.py --devices       # just list audio outputs
    python3 scripts/media_probe.py --play "query"  # search and actually play
    python3 scripts/media_probe.py --play "x" --audio-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK = "\033[1;32m✓\033[0m"
NO = "\033[1;31m✗\033[0m"
HM = "\033[1;33m!\033[0m"


def load_cfg() -> dict:
    import yaml
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def check_binaries(cfg: dict) -> bool:
    print("\nPrograms")
    m = cfg.get("media", {}) or {}
    ok = True
    from core.media import YouTube
    yt_path = YouTube(cfg).path()
    for label, binary, fix in (
        ("yt-dlp", m.get("ytdlp_bin", "yt-dlp"),
         "~/assistant/.venv/bin/pip install -U yt-dlp\n"
         "      NOT apt — Ubuntu 24.04 ships the 2023 build, which no longer "
         "works."),
        ("mpv", m.get("mpv_bin", "mpv"), "sudo apt install -y mpv"),
    ):
        path = yt_path if label == "yt-dlp" else shutil.which(binary)
        if path:
            try:
                ver = subprocess.run([binary, "--version"], capture_output=True,
                                     text=True, timeout=15).stdout.splitlines()
                ver = ver[0].strip() if ver else ""
            except Exception:  # noqa: BLE001
                ver = ""
            print(f"  {OK} {label}: {path}  {ver}")
        else:
            ok = False
            print(f"  {NO} {label} not found\n      {fix}")

    # yt-dlp goes stale and YouTube changes underneath it. An old copy fails
    # with a confusing extraction error rather than saying it is out of date,
    # so name the age here where it can be seen.
    if yt_path:
        try:
            import datetime as _dt
            out = subprocess.run([yt_path, "--version"], capture_output=True,
                                 text=True, timeout=20)
            stamp = out.stdout.strip().split()[0]
            built = _dt.date(*[int(x) for x in stamp.split(".")[:3]])
            age = (_dt.date.today() - built).days
            if age > 365:
                print(f"  {NO} yt-dlp is {age} days old ({stamp}). This "
                      f"WILL fail on YouTube.\n      "
                      f"~/assistant/.venv/bin/pip install -U yt-dlp")
                ok = False
            elif age > 180:
                print(f"  {HM} yt-dlp is {age} days old ({stamp}) — worth "
                      f"updating before you\n      debug anything else.")
            else:
                print(f"  {OK} yt-dlp is {age} days old ({stamp}), recent "
                      f"enough.")
        except Exception:  # noqa: BLE001
            print(f"  {HM} could not read the yt-dlp version")

    # mpv resolves the stream through its own ytdl hook, which searches PATH.
    # Search working proves nothing about playback if they disagree.
    if yt_path:
        on_path = shutil.which(m.get("ytdlp_bin", "yt-dlp"))
        if on_path:
            print(f"  {OK} mpv will find yt-dlp on PATH ({on_path})")
        else:
            print(f"  {OK} yt-dlp is off PATH, so mpv is told where it is\n"
                  f"      --script-opts=ytdl_hook-ytdl_path={yt_path}")
    return ok


def check_devices(cfg: dict) -> None:
    """Which audio outputs mpv can see, and which one is the array.

    This is the setting people get wrong. The assistant's voice goes out
    through the ReSpeaker so its echo canceller has a reference; if the video
    goes out somewhere else, the canceller never hears it and the wake word
    stops working the moment anything is playing.
    """
    print("\nAudio outputs mpv can see")
    binary = (cfg.get("media", {}) or {}).get("mpv_bin", "mpv")
    if not shutil.which(binary):
        print(f"  {NO} mpv is not installed")
        return
    out = subprocess.run([binary, "--audio-device=help"],
                         capture_output=True, text=True, timeout=20)
    want = str((cfg.get("media", {}) or {}).get("audio_device", "auto"))
    found = False
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.startswith("'"):
            continue
        name = line.split("'")[1]
        array = "array" in line.lower() or "xvf" in line.lower() \
            or "respeaker" in line.lower() or "seeed" in line.lower()
        mark = OK if name == want else ("  " if not array else HM)
        print(f"  {mark} {line}")
        if name == want:
            found = True
    print()
    if want == "auto":
        print(f"  {HM} media.audio_device is 'auto'. Works, but it follows the "
              f"system\n      default — which is not the array. Copy the name "
              f"marked above\n      into config.yaml so video audio goes "
              f"through the microphone array\n      and echo cancellation "
              f"keeps working while something is playing.")
    elif found:
        print(f"  {OK} media.audio_device = {want!r} exists.")
    else:
        print(f"  {NO} media.audio_device = {want!r} is NOT in that list. "
              f"mpv will fall\n      back to the default and the wake word "
              f"will struggle over audio.")


def check_search(cfg: dict, query: str) -> list:
    print(f"\nSearch: {query!r}")
    from core.media import YouTube, clean_query
    yt = YouTube(cfg)
    print(f"  query after trimming: {clean_query(query)!r}")
    t0 = time.time()
    hits = yt.search(query)
    print(f"  {OK if hits else NO} {len(hits)} result(s) in "
          f"{time.time() - t0:.1f}s")
    for i, h in enumerate(hits, 1):
        print(f"     {i}. {h.describe()}")
        print(f"        {h.url}")
    if not hits:
        print(f"  {NO} No results. Try it by hand to see the real error:\n"
              f"      yt-dlp --flat-playlist --print \"%(title)s\" "
              f"\"ytsearch3:{clean_query(query)}\"")
    return hits


def do_play(cfg: dict, query: str, audio_only: bool) -> int:
    hits = check_search(cfg, query)
    if not hits:
        return 1
    from core.media import Player
    player = Player(cfg, ROOT)
    hit = hits[0]

    print(f"\nPlaying {'audio only' if audio_only else 'on screen'}: "
          f"{hit.title}")
    print(f"  socket: {player.sock}")
    if not audio_only:
        print(f"  display: {player.display}   "
              f"(if you are over SSH, the video appears on the appliance's "
              f"own screen)")
    player.start(hit, audio_only)

    print("  waiting for mpv to answer on the socket...")
    for _ in range(40):
        time.sleep(0.5)
        if player.get("mpv-version") is not None:
            break
    version = player.get("mpv-version")
    if version is None:
        print(f"  {NO} mpv never answered on the socket.")
        why = player.why_it_died()
        if why:
            print(f"\n      mpv said:\n        {why}\n")
        else:
            print(f"      and left nothing in {player.log}\n")
        print(f"      The exact command it ran:\n")
        for i, part in enumerate(player.command(hit, audio_only)):
            print(f"        {part}" + (" \\" if i else " \\"))
        print(f"\n      Run that yourself without --no-terminal and "
              f"--msg-level to see it live.")
        player.stop()
        return 1
    print(f"  {OK} IPC is up ({version})")

    print("  letting it run for eight seconds, then exercising the controls")
    time.sleep(8)
    pos, total = player.position()
    print(f"     position {pos and round(pos, 1)} of {total and round(total)}")
    if not pos:
        print(f"  {HM} no playback position — it is still resolving, or the "
              f"stream failed")

    for label, fn in (("pause", lambda: player.pause(True)),
                      ("resume", lambda: player.pause(False)),
                      ("skip forward 15s", lambda: player.seek(15)),
                      ("volume 40", lambda: player.set_volume(40)),
                      ("duck", lambda: (player.duck(True), True)[1]),
                      ("unduck", lambda: (player.duck(False), True)[1])):
        time.sleep(1)
        print(f"     {OK if fn() else NO} {label}")

    print("\n  stopping")
    player.stop()
    print(f"  {OK} done. If you saw and heard it, the whole path works.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", action="store_true")
    ap.add_argument("--play", metavar="QUERY")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--search", metavar="QUERY",
                    default="how to sharpen a chisel")
    args = ap.parse_args()

    cfg = load_cfg()
    if not (cfg.get("media", {}) or {}).get("enabled"):
        print(f"{HM} media.enabled is false in config.yaml — the tools are "
              f"not loaded.\n  Probing anyway.")

    if args.devices:
        check_devices(cfg)
        return 0
    if args.play:
        if not check_binaries(cfg):
            return 1
        return do_play(cfg, args.play, args.audio_only)

    ok = check_binaries(cfg)
    check_devices(cfg)
    if ok:
        check_search(cfg, args.search)
    print(f"\nWhen that all looks right:\n"
          f"  python3 scripts/media_probe.py --play \"lofi hip hop\" "
          f"--audio-only\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
