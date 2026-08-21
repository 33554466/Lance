"""Cut and restore USB bus power to the mic array, via uhubctl.

Why this exists, when usbreset.py already exists
------------------------------------------------
usbreset.py issues USBDEVFS_RESET, which asks the host controller to send a
bus reset down the wire. That is a *protocol* reset. It is enough for most
misbehaving USB devices, and it is what the `usbreset` utility does.

It is not enough for the XVF3800. Measured on this build, after a warm
reboot, all three software-only reset paths fail to revive it:

    1. USBDEVFS_RESET                       -> capture still returns EIO
    2. authorized 0 / authorized 1          -> capture still returns EIO
    3. driver unbind / rebind               -> capture still returns EIO

Only pulling the plug works. The reason is that none of those three interrupt
VBUS. The XMOS processor is never de-powered, so its firmware never restarts,
and the wedged audio endpoint is a firmware state that survives every reset
the host is able to ask for.

Which leaves exactly one software equivalent of pulling the plug: per-port
power switching (ppps). A hub that implements it can be told to remove VBUS
from one downstream port. To the array that is indistinguishable from being
unplugged, because it is being unplugged — the electrons stop.

The catch is that most root hubs on modern boards do NOT implement ppps; the
port power bit is hardwired. Cheap external hubs frequently do. On this
build the Sabrent's three internal Realtek controllers (0bda:5411) all
advertise ppps and the SER8's root hubs do not, which is why the array has to
live on the hub rather than on a rear port — the opposite of the usual advice
about keeping audio devices off shared hubs, and worth the trade.

Permissions
-----------
uhubctl talks to the HUB, not to the array, so the udev rule that grants
access to 2886:001a is not enough. systemd/99-respeaker.rules grants the
hub's vid:pid too. Without it this falls back to `sudo -n`, which only works
if a sudoers rule exists, and otherwise degrades quietly to usbreset.py.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from .usbreset import _norm_id

log = logging.getLogger("assistant.usbpower")

SYS_USB = Path("/sys/bus/usb/devices")


def available() -> bool:
    return shutil.which("uhubctl") is not None


def find_location(vid_pid: str) -> tuple[str, str] | None:
    """Return (hub_location, port) for a device, e.g. ("1-3.4.4", "2").

    USB sysfs names encode the whole topology: "1-3.4.4.2" is bus 1, root
    port 3, then port 4, then port 4, then port 2. Chop the last hop off and
    you have the hub the device is plugged into, in exactly the notation
    uhubctl's -l flag wants.
    """
    try:
        vid, pid = (_norm_id(x) for x in vid_pid.split(":"))
    except ValueError:
        return None

    for entry in sorted(SYS_USB.iterdir()):
        try:
            if (entry / "idVendor").read_text().strip().lower() != vid:
                continue
            if (entry / "idProduct").read_text().strip().lower() != pid:
                continue
        except OSError:
            continue

        name = entry.name
        if ":" in name:                 # an interface, not a device
            continue
        if "." in name:                 # behind an external hub
            hub, _, port = name.rpartition(".")
            return hub, port
        if "-" in name:                 # directly on a root hub port
            bus, _, port = name.partition("-")
            return f"{bus}-0", port
    return None


def _run(args: list[str]) -> tuple[int, str]:
    """Run uhubctl, escalating to sudo -n only if we are refused."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0 and "permission" in out.lower():
        try:
            p = subprocess.run(["sudo", "-n", *args], capture_output=True,
                               text=True, timeout=30)
            out = (p.stdout or "") + (p.stderr or "")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 1, str(exc)
    return p.returncode, out


def supports_ppps(hub: str) -> bool:
    rc, out = _run(["uhubctl", "-l", hub])
    return rc == 0 and "ppps" in out


def ppps_hubs() -> list[str]:
    """Every hub on the machine that can actually switch port power."""
    rc, out = _run(["uhubctl"])
    if rc != 0:
        return []
    return re.findall(r"status for hub (\S+).*?ppps", out)


def power_cycle(vid_pid: str, off_seconds: float = 3.0,
                settle: float = 5.0) -> bool:
    """Remove VBUS from the array's port, restore it, wait for it to come
    back. Returns True only if we are confident power was actually cut.

    Never raises. A failure here falls through to USBDEVFS_RESET, which is
    strictly better than nothing even though it is not sufficient on its own.
    """
    if not available():
        log.info("uhubctl not installed — skipping power cycle "
                 "(sudo apt install -y uhubctl)")
        return False

    loc = find_location(vid_pid)
    if not loc:
        log.warning("no USB device %s to power cycle", vid_pid)
        return False
    hub, port = loc

    if not supports_ppps(hub):
        others = [h for h in ppps_hubs() if h != hub]
        log.warning(
            "the array is on hub %s port %s, which does not support per-port "
            "power switching, so its power cannot be cut in software.%s",
            hub, port,
            ("\n    Hubs on this machine that CAN: " + ", ".join(others) +
             "\n    Move the array to one of those and this becomes "
             "self-healing.") if others else
            "\n    No hub on this machine supports it. A warm reboot will "
            "need a physical replug.")
        return False

    rc, out = _run(["uhubctl", "-l", hub, "-p", port, "-a", "cycle",
                    "-d", str(int(off_seconds))])
    if rc != 0:
        log.warning("uhubctl failed on %s port %s: %s", hub, port,
                    out.strip().splitlines()[-1] if out.strip() else rc)
        return False

    log.info("cut USB power to %s on hub %s port %s for %ss",
             vid_pid, hub, port, int(off_seconds))

    # The array's firmware boots from cold, then the host enumerates it and
    # snd-usb-audio binds. Poll rather than sleeping blind, so a fast device
    # does not cost us five seconds on every start.
    deadline = time.time() + settle
    while time.time() < deadline:
        if find_location(vid_pid):
            # Present on the bus. Give the ALSA side a moment to attach —
            # opening a stream the instant the device node appears is a
            # reliable way to get EIO from a card that is not ready yet.
            time.sleep(1.5)
            log.info("array re-enumerated after power cycle")
            return True
        time.sleep(0.25)

    log.warning("array did not come back within %.0fs of the power cycle",
                settle)
    return False
