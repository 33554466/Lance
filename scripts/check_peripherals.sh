#!/usr/bin/env bash
# ============================================================
# Phase 2 gate: prove every peripheral works BEFORE any application code
# touches it.
#
# This script does not fix anything. It tells you what is and is not
# working, so that when something breaks in phase 5 you already know it
# was not the microphone.
# ============================================================
set -uo pipefail

pass() { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[1;31m✗\033[0m %s\n' "$*"; FAILED=1; }
info() { printf '  \033[1;34mi\033[0m %s\n' "$*"; }
head() { printf '\n\033[1m%s\033[0m\n' "$*"; }

FAILED=0

head "Network"
if ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then pass "internet reachable"
else fail "no internet"; fi
LINK=$(ip -o link show | grep -E 'state UP' | grep -v ' lo:' | awk -F': ' '{print $2}' | head -1)
if [[ -n "${LINK:-}" ]]; then
    if [[ "$LINK" == e* ]]; then pass "wired link up ($LINK)"
    else fail "on wireless ($LINK) — use Ethernet, Wi-Fi adds jitter to streaming"; fi
fi

head "Audio devices"
if command -v python3 >/dev/null && python3 -c "import sounddevice" 2>/dev/null; then
    python3 - <<'PY'
import sounddevice as sd
ins = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
outs = [d for d in sd.query_devices() if d['max_output_channels'] > 0]
print("  inputs:")
for d in ins:  print(f"    [{d['index']}] {d['name']}  ({d['max_input_channels']}ch @ {int(d['default_samplerate'])}Hz)")
print("  outputs:")
for d in outs: print(f"    [{d['index']}] {d['name']}")
names = " ".join(d['name'] for d in ins).lower()
if 'respeaker' in names or 'xvf' in names or 'xmos' in names:
    print("  \033[1;32m✓\033[0m ReSpeaker array detected")
else:
    print("  \033[1;33m!\033[0m No ReSpeaker in the input list — check the USB cable and `lsusb`")
PY
else
    fail "sounddevice not importable — activate the venv first"
fi

head "Microphone capture (5 seconds — say something)"
read -rp "  Press Enter to record, or s to skip: " k
if [[ "$k" != "s" ]]; then
    if arecord -f S16_LE -r 16000 -c 1 -d 5 /tmp/mic_test.wav 2>/dev/null; then
        SIZE=$(stat -c%s /tmp/mic_test.wav)
        if (( SIZE > 40000 )); then pass "captured ${SIZE} bytes"
        else fail "file suspiciously small (${SIZE}B) — wrong device?"; fi
        info "playing it back…"
        aplay /tmp/mic_test.wav 2>/dev/null || fail "playback failed"
        info "Did you hear yourself clearly? If not, fix that before going further."
    else
        fail "arecord failed"
    fi
fi

head "Printer (CUPS)"
if lpstat -p 2>/dev/null | grep -q printer; then
    lpstat -p 2>/dev/null | sed 's/^/    /'
    pass "printer(s) known to CUPS"
    info "print a test page with:  echo 'Appliance test' | lp"
else
    fail "no printers — check http://localhost:631 or that ipp-usb is running"
fi

head "Scanner (SANE)"
SCAN=$(scanimage -L 2>/dev/null)
if echo "$SCAN" | grep -q 'device'; then
    echo "$SCAN" | sed 's/^/    /'
    pass "scanner detected"
    info "test a scan with:  scanimage --format=png > /tmp/scan.png"
else
    fail "no scanner — for network units confirm sane-airscan is installed"
fi

head "Camera (V4L2)"
if command -v v4l2-ctl >/dev/null && v4l2-ctl --list-devices 2>/dev/null | grep -q .; then
    v4l2-ctl --list-devices 2>/dev/null | sed 's/^/    /'
    pass "camera detected"
else
    fail "no camera found"
fi

head "Displays"
if command -v xrandr >/dev/null; then
    xrandr --listmonitors 2>/dev/null | sed 's/^/    /'
fi

head "Power"
if command -v upsc >/dev/null && upsc "$(upsc -l 2>/dev/null | head -1)" 2>/dev/null | grep -q battery; then
    pass "UPS visible to NUT"
else
    info "no UPS configured yet — that is phase 9, not a problem now"
fi

echo
if (( FAILED )); then
    printf '\033[1;31mSome checks failed.\033[0m Fix these before writing application code.\n'
    printf 'Debugging a printer through your own abstraction layer later is misery.\n'
    exit 1
else
    printf '\033[1;32mAll peripheral checks passed.\033[0m Proceed to phase 3.\n'
fi
