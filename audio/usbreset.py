"""Reset a USB device from userspace, by vendor:product id.

Why this exists
---------------
A warm reboot does not cut USB bus power on most boards. The host
reinitialises; the device does not. The ReSpeaker's XMOS processor therefore
comes back still holding whatever state it was in, its audio endpoint
half-open, and every capture attempt returns EIO:

    arecord: pcm_read:2240: read error: Input/output error

It enumerates perfectly, snd-usb-audio binds, ALSA lists the card — it simply
will not stream. The only cure is a genuine device reset, which until now
meant physically unplugging it.

An appliance that needs a human to reseat a cable after every power cut is
not an appliance. So we issue USBDEVFS_RESET ourselves at startup, which is
exactly what the `usbreset` utility does.

Permissions
-----------
Writing to /dev/bus/usb/BBB/DDD normally needs root. Rather than run the
service as root, ship a udev rule granting access to this one device — see
systemd/99-respeaker.rules.
"""
from __future__ import annotations

import fcntl
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("assistant.usbreset")

# _IO('U', 20) — the ioctl the usbreset utility uses.
USBDEVFS_RESET = ord("U") << 8 | 20

SYS_USB = Path("/sys/bus/usb/devices")


def _norm_id(value: str) -> str:
    """Normalise a hex id for comparison with sysfs.

    NOT str.lstrip("0x") — that strips a character SET, so "001a" would
    become "1a" and never match sysfs's "001a". A classic and quiet bug.
    """
    value = value.strip().lower()
    return value[2:] if value.startswith("0x") else value


def find_device_node(vid: str, pid: str) -> tuple[str, str] | None:
    """Return (/dev/bus/usb/BBB/DDD, sysfs name) for a vid:pid, or None."""
    vid, pid = _norm_id(vid), _norm_id(pid)
    if not SYS_USB.is_dir():
        return None
    for entry in sorted(SYS_USB.iterdir()):
        try:
            if (entry / "idVendor").read_text().strip().lower() != vid:
                continue
            if (entry / "idProduct").read_text().strip().lower() != pid:
                continue
            bus = int((entry / "busnum").read_text())
            dev = int((entry / "devnum").read_text())
        except (OSError, ValueError):
            continue
        return f"/dev/bus/usb/{bus:03d}/{dev:03d}", entry.name
    return None


def reset(vid_pid: str, settle: float = 2.0) -> bool:
    """Reset the device. Returns True if the ioctl succeeded.

    Never raises: a failed reset should degrade to "try anyway", not stop the
    appliance from starting.
    """
    try:
        vid, pid = vid_pid.split(":")
    except ValueError:
        log.warning("audio.usb_reset should look like '2886:001a', got %r", vid_pid)
        return False

    found = find_device_node(vid, pid)
    if not found:
        log.warning("no USB device %s to reset — is it plugged in?", vid_pid)
        return False
    node, sysname = found

    try:
        fd = os.open(node, os.O_WRONLY)
    except PermissionError:
        log.warning(
            "cannot reset %s (%s): permission denied.\n"
            "    Install the udev rule so this works without root:\n"
            "      sudo cp ~/assistant/systemd/99-respeaker.rules "
            "/etc/udev/rules.d/\n"
            "      sudo udevadm control --reload-rules && sudo udevadm trigger\n"
            "    Without it the device stays deaf after a warm reboot until "
            "you unplug it by hand.", vid_pid, node)
        return False
    except OSError as exc:
        log.warning("cannot open %s: %s", node, exc)
        return False

    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
        log.info("USB reset %s at %s (%s)", vid_pid, node, sysname)
    except OSError as exc:
        log.warning("USBDEVFS_RESET on %s failed: %s", node, exc)
        return False
    finally:
        os.close(fd)

    # Re-enumeration takes a moment, and the device number changes. Anything
    # that resolves audio devices must run AFTER this settles.
    time.sleep(settle)
    return True
