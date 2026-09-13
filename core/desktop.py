"""Window management, as a fixed vocabulary.

The tempting design is one `run_command` tool that hands the model a shell.
Do not build that. Every input to this system arrives through a wake word
that occasionally misfires and a transcriber that occasionally mishears, and
`xdotool type` can drive any application on the desktop — including a
terminal. A generic "send keystrokes" tool is root access wearing a hat.

So this module exposes verbs, not power:

    list_windows      what is open
    focus_window      bring one forward
    arrange_window    maximise / restore / fullscreen / minimise / left / right
    switch_workspace  move between desktops
    set_volume        the assistant's own output level

Deliberately absent: closing windows, typing text, clicking coordinates,
running commands. Closing loses unsaved work on a single mishearing; the
other three are indistinguishable from full control of the machine.

X11 only. Wayland compositors isolate applications from each other on
purpose, and GNOME exposes no general way around it — so on Wayland these
tools report that they cannot help rather than failing obscurely.

Everything shells out with argument LISTS, never a string, and never with
shell=True. A window title is attacker-adjacent input — it is whatever some
web page decided to call itself — and it must never reach a shell.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess

log = logging.getLogger("assistant.desktop")

TIMEOUT = 5

# States the model may ask for, mapped to what actually does them. Keeping
# this an explicit table means an unexpected string is a clean "I cannot do
# that" rather than an argument smuggled into a command line.
ARRANGEMENTS = ("maximize", "restore", "fullscreen", "minimize",
                "left", "right", "center")


def _run(args: list[str]) -> tuple[int, str]:
    # Secrets stripped from the child environment. wmctrl and xdotool have no
    # business being able to read an API key, and a subprocess that inherits
    # one is a subprocess that can leak it.
    from .provider import child_env
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           timeout=TIMEOUT, env=child_env())
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{args[0]} is not installed"
    except subprocess.TimeoutExpired:
        return 124, f"{args[0]} timed out"
    except OSError as exc:
        return 1, str(exc)


class DesktopTools:
    def __init__(self, cfg: dict):
        d = cfg.get("desktop", {}) or {}
        self.enabled = bool(d.get("enabled", False))
        # Windows the assistant must never hide. Its own kiosk display is on
        # this list because "minimise everything" would otherwise blank the
        # screen you use to see what it is doing.
        self.protect = [str(p).lower() for p in
                        (d.get("protect") or ["assistant"])]
        self.volume_card = str(d.get("volume_card", "")).strip()
        self.display = str(d.get("display", ":0"))

    # -- environment -------------------------------------------------

    def _unavailable(self) -> str | None:
        """Return a sentence explaining why this cannot work, or None."""
        if not self.enabled:
            return "Desktop control is turned off in the configuration."
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            return ("This desktop is running Wayland, which does not let one "
                    "application move another's windows. Nothing I can do "
                    "from here.")
        if not shutil.which("wmctrl"):
            return ("wmctrl is not installed. Run: sudo apt install -y "
                    "wmctrl xdotool")
        os.environ.setdefault("DISPLAY", self.display)
        return None

    # -- windows -----------------------------------------------------

    def _windows(self) -> list[dict]:
        """Every real application window: id, class, title."""
        rc, out = _run(["wmctrl", "-l", "-x"])
        if rc != 0:
            return []
        found = []
        for line in out.splitlines():
            # 0x03c00007  0 chromium.Chromium  host  Title with spaces
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            wid, desktop, wclass, _host, title = parts
            if desktop == "-1":
                continue          # panels, docks, the desktop itself
            found.append({"id": wid, "class": wclass.split(".")[-1],
                          "title": title.strip()})
        return found

    def _match(self, needle: str) -> tuple[list[dict], str | None]:
        """Find windows by title or application name.

        Returns (matches, error). Ambiguity is returned to the model as a
        question rather than resolved by guessing — picking the wrong window
        and maximising it is a small thing, but doing it silently is not.
        """
        wins = self._windows()
        if not wins:
            return [], "There are no windows open that I can see."
        n = " ".join(str(needle).lower().split())
        if not n:
            return [], "Which window?"

        exact = [w for w in wins if w["title"].lower() == n
                 or w["class"].lower() == n]
        if exact:
            return exact[:1], None
        partial = [w for w in wins
                   if n in w["title"].lower() or n in w["class"].lower()]
        if not partial:
            names = ", ".join(sorted({w["class"] for w in wins}))
            return [], f"Nothing open matching {needle!r}. Open windows: {names}."
        return partial, None

    def _is_protected(self, win: dict) -> bool:
        hay = (win["title"] + " " + win["class"]).lower()
        return any(p in hay for p in self.protect)

    # -- schemas -----------------------------------------------------

    def schemas(self) -> list[dict]:
        if not self.enabled:
            return []
        return [
            {
                "name": "list_windows",
                "description": (
                    "List the application windows currently open on the "
                    "screen. Use before arranging anything if it is not "
                    "obvious which window the user means."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "focus_window",
                "description": (
                    "Bring a window to the front and give it focus. Match by "
                    "part of its title or the application name."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "window": {"type": "string",
                                   "description": "Part of the title, or the app name."},
                    },
                    "required": ["window"],
                },
            },
            {
                "name": "arrange_window",
                "description": (
                    "Change a window's size or position: maximize, restore, "
                    "fullscreen, minimize, or snap it to the left or right "
                    "half of the screen, or center it. Omit `window` to act "
                    "on whatever is currently in front."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "how": {"type": "string", "enum": list(ARRANGEMENTS)},
                        "window": {"type": "string",
                                   "description": "Part of the title. Omit for the active window."},
                    },
                    "required": ["how"],
                },
            },
            {
                "name": "switch_workspace",
                "description": "Switch to another virtual desktop, numbered from 1.",
                "input_schema": {
                    "type": "object",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            },
            {
                "name": "set_volume",
                "description": (
                    "Set the speaker volume as a percentage, 0 to 100. This "
                    "is the assistant's own output level."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"percent": {"type": "integer"}},
                    "required": ["percent"],
                },
            },
        ]

    # -- execution ---------------------------------------------------

    def run_sync(self, name: str, args: dict) -> str | None:
        if name not in ("list_windows", "focus_window", "arrange_window",
                        "switch_workspace", "set_volume"):
            return None

        problem = self._unavailable()
        if problem:
            return problem

        if name == "list_windows":
            wins = self._windows()
            if not wins:
                return "Nothing is open."
            return f"{len(wins)} open: " + "; ".join(
                f"{w['class']} — {w['title'][:60]}" for w in wins)

        if name == "focus_window":
            matches, err = self._match(args.get("window", ""))
            if err:
                return err
            if len(matches) > 1:
                return ("Several windows match: "
                        + "; ".join(m["title"][:50] for m in matches[:5])
                        + ". Which one?")
            w = matches[0]
            rc, out = _run(["wmctrl", "-i", "-a", w["id"]])
            if rc != 0:
                return f"Could not focus that: {out.strip()[:120]}"
            log.info("focused %s", w["title"][:60])
            return f"Brought {w['class']} to the front."

        if name == "arrange_window":
            how = str(args.get("how", "")).lower().strip()
            if how not in ARRANGEMENTS:
                return (f"I cannot do {how!r}. I can " + ", ".join(ARRANGEMENTS)
                        + ".")
            target = args.get("window")
            if target:
                matches, err = self._match(target)
                if err:
                    return err
                if len(matches) > 1:
                    return ("Several windows match: "
                            + "; ".join(m["title"][:50] for m in matches[:5])
                            + ". Which one?")
                w = matches[0]
            else:
                w = self._active()
                if not w:
                    return "I could not tell which window is in front."

            if how == "minimize" and self._is_protected(w):
                return ("That is my own display — I will not hide it. Ask "
                        "again naming a different window.")
            return self._arrange(w, how)

        if name == "switch_workspace":
            try:
                n = int(args.get("number", 0))
            except (TypeError, ValueError):
                return "Which workspace number?"
            if n < 1:
                return "Workspaces are numbered from one."
            rc, out = _run(["wmctrl", "-s", str(n - 1)])
            if rc != 0:
                return f"Could not switch: {out.strip()[:120]}"
            return f"Workspace {n}."

        if name == "set_volume":
            return self._volume(args.get("percent"))

        return None

    # -- helpers -----------------------------------------------------

    def _active(self) -> dict | None:
        rc, out = _run(["xdotool", "getactivewindow"])
        if rc != 0:
            return None
        try:
            wid = int(out.strip())
        except ValueError:
            return None
        hexid = f"0x{wid:08x}"
        for w in self._windows():
            if int(w["id"], 16) == wid:
                return w
        return {"id": hexid, "class": "window", "title": ""}

    def _geometry(self) -> tuple[int, int] | None:
        rc, out = _run(["xdotool", "getdisplaygeometry"])
        if rc != 0:
            return None
        m = re.match(r"\s*(\d+)\s+(\d+)", out)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _arrange(self, w: dict, how: str) -> str:
        wid = w["id"]
        MAX = "maximized_vert,maximized_horz"

        if how == "maximize":
            rc, out = _run(["wmctrl", "-i", "-r", wid, "-b", "add", MAX])
            return (f"Maximised {w['class']}." if rc == 0
                    else f"Could not: {out.strip()[:120]}")

        if how == "restore":
            _run(["wmctrl", "-i", "-r", wid, "-b", "remove", "fullscreen"])
            rc, out = _run(["wmctrl", "-i", "-r", wid, "-b", "remove", MAX])
            return (f"Restored {w['class']}." if rc == 0
                    else f"Could not: {out.strip()[:120]}")

        if how == "fullscreen":
            rc, out = _run(["wmctrl", "-i", "-r", wid, "-b", "add",
                            "fullscreen"])
            return (f"{w['class']} is fullscreen. Say restore to bring it back."
                    if rc == 0 else f"Could not: {out.strip()[:120]}")

        if how == "minimize":
            rc, out = _run(["xdotool", "windowminimize", str(int(wid, 16))])
            return (f"Minimised {w['class']}." if rc == 0
                    else f"Could not: {out.strip()[:120]}")

        # Half-screen snapping and centring need real pixels, and a window
        # that is still maximised ignores a move — so clear that first.
        geo = self._geometry()
        if not geo:
            return "I could not work out the screen size."
        sw, sh = geo
        _run(["wmctrl", "-i", "-r", wid, "-b", "remove", MAX])
        if how == "left":
            box, said = (0, 0, sw // 2, sh), "left half"
        elif how == "right":
            box, said = (sw // 2, 0, sw // 2, sh), "right half"
        else:
            box, said = (sw // 6, sh // 8, (sw * 2) // 3, (sh * 3) // 4), "centred"
        rc, out = _run(["wmctrl", "-i", "-r", wid, "-e",
                        "0,{},{},{},{}".format(*box)])
        return (f"{w['class']} to the {said}." if rc == 0
                else f"Could not move it: {out.strip()[:120]}")

    def _volume(self, percent) -> str:
        try:
            pct = max(0, min(100, int(percent)))
        except (TypeError, ValueError):
            return "What volume, as a percentage?"

        # The assistant plays through the mic array, so its ALSA mixer is the
        # one that matters — the system default sink is a different device
        # entirely. Fall back to the default sink if the array exposes no
        # playback control, which some firmware revisions do not.
        if self.volume_card:
            for control in ("PCM", "Speaker", "Master"):
                rc, _ = _run(["amixer", "-c", self.volume_card, "sset",
                              control, f"{pct}%", "unmute"])
                if rc == 0:
                    log.info("volume %d%% via %s/%s", pct, self.volume_card,
                             control)
                    return f"Volume {pct} percent."
        rc, out = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                        f"{pct}%"])
        if rc == 0:
            return f"Volume {pct} percent."
        return ("I could not find a volume control I can change. The speaker "
                "has its own knob.")
