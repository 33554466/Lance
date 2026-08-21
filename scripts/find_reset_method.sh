#!/usr/bin/env bash
# ============================================================
# Which USB reset actually revives the ReSpeaker on THIS machine?
#
#   sudo ./scripts/find_reset_method.sh
#
# USBDEVFS_RESET re-enumerates the device but never removes bus power, and
# the XVF3800's processor can sit right through it still wedged. There are
# progressively heavier hammers; this tries each and tells you the lightest
# one that works, so the service can use exactly that and no more.
# ============================================================
set -uo pipefail

VID=2886; PID=001a; CARD=Array
PROBE=/tmp/reset_probe.wav

if [[ $EUID -ne 0 ]]; then echo "Run with sudo."; exit 1; fi

hdr()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
pass() { printf '  \033[1;32m✓ %s\033[0m\n' "$*"; }
fail() { printf '  \033[1;31m✗ %s\033[0m\n' "$*"; }

find_sysfs() {
    for d in /sys/bus/usb/devices/*; do
        [[ -f "$d/idVendor" ]] || continue
        if [[ "$(cat "$d/idVendor")" == "$VID" && "$(cat "$d/idProduct")" == "$PID" ]]; then
            basename "$d"; return 0
        fi
    done
    return 1
}

capture_works() {
    rm -f "$PROBE"
    timeout 8 arecord -D "plughw:${CARD},0" -f S16_LE -r 16000 -c 2 \
        -d 1 "$PROBE" >/dev/null 2>&1 || return 1
    [[ -s "$PROBE" ]] && (( $(stat -c%s "$PROBE") > 20000 ))
}

settle() {
    for _ in $(seq 1 20); do
        sleep 0.5
        arecord -l 2>/dev/null | grep -q "$CARD" && return 0
    done
    return 1
}

hdr "Stopping the service so it is not holding the device"
systemctl --user --machine="${SUDO_USER:-lance}@" stop assistant-audio 2>/dev/null \
    || sudo -u "${SUDO_USER:-lance}" XDG_RUNTIME_DIR=/run/user/$(id -u "${SUDO_USER:-lance}") \
       systemctl --user stop assistant-audio 2>/dev/null || true
pkill -f audio.service 2>/dev/null || true
sleep 2

DEV=$(find_sysfs) || { fail "device $VID:$PID not present"; exit 1; }
echo "  device is $DEV"

hdr "Baseline — does capture work right now?"
if capture_works; then
    pass "capture already works. Nothing to fix; start the service."
    exit 0
fi
fail "capture fails (this is the state we need to clear)"

# ---------------------------------------------------------------
hdr "Method 1 — USBDEVFS_RESET (what the service does today)"
python3 - "$DEV" <<'PY'
import fcntl, os, sys, pathlib
d = pathlib.Path('/sys/bus/usb/devices')/sys.argv[1]
node = "/dev/bus/usb/%03d/%03d" % (int((d/'busnum').read_text()), int((d/'devnum').read_text()))
fd = os.open(node, os.O_WRONLY)
try: fcntl.ioctl(fd, ord('U') << 8 | 20, 0); print(f"    ioctl sent to {node}")
finally: os.close(fd)
PY
settle
if capture_works; then pass "METHOD 1 (usbdevfs_reset) WORKS"; exit 0; fi
fail "still failing"

# ---------------------------------------------------------------
hdr "Method 2 — deauthorize/reauthorize"
echo 0 > "/sys/bus/usb/devices/$DEV/authorized"; sleep 2
echo 1 > "/sys/bus/usb/devices/$DEV/authorized"
settle
if capture_works; then pass "METHOD 2 (authorized toggle) WORKS"; exit 0; fi
fail "still failing"

# ---------------------------------------------------------------
hdr "Method 3 — unbind/rebind the usb driver"
DEV=$(find_sysfs) || true
echo "$DEV" > /sys/bus/usb/drivers/usb/unbind 2>/dev/null; sleep 2
echo "$DEV" > /sys/bus/usb/drivers/usb/bind   2>/dev/null
settle
if capture_works; then pass "METHOD 3 (driver unbind/rebind) WORKS"; exit 0; fi
fail "still failing"

# ---------------------------------------------------------------
hdr "Method 4 — cut port power with uhubctl"
if ! command -v uhubctl >/dev/null; then
    echo "  uhubctl not installed:  sudo apt install -y uhubctl"
else
    uhubctl -a cycle -s "$VID:$PID" 2>&1 | sed 's/^/    /'
    settle
    if capture_works; then pass "METHOD 4 (uhubctl power cycle) WORKS"; exit 0; fi
    fail "still failing"
fi

hdr "Result"
echo "  None of the software resets cleared it — this port cannot cut VBUS,"
echo "  so only a physical unplug or a full power-off will. Two real options:"
echo ""
echo "    1. Move the array to the Sabrent hub. Hubs with per-port switching"
echo "       often DO support software power control, which makes method 4"
echo "       work. Re-run this script afterwards to check."
echo "    2. Use 'systemctl poweroff' instead of 'reboot', so the machine"
echo "       actually drops power to the bus."
