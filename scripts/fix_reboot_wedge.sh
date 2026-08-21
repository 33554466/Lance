#!/usr/bin/env bash
# ============================================================================
# fix_reboot_wedge.sh — close the warm-reboot hole for good.
#
# Run this once, after a reboot, with the assistant stopped. It works out
# where the microphone is plugged in, whether that port can have its power
# cut in software, and — if it can — proves the recovery works by wedging
# nothing and reviving everything on the spot.
#
#   ./scripts/fix_reboot_wedge.sh
#
# It changes no configuration. Its only side effect is briefly removing power
# from one USB port.
# ============================================================================
set -uo pipefail

ARRAY_ID="2886:001a"
BOLD=$'\e[1m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; OFF=$'\e[0m'

say()  { printf '\n%s%s%s\n' "$BOLD" "$*" "$OFF"; }
ok()   { printf '  %sOK%s   %s\n'   "$GRN" "$OFF" "$*"; }
bad()  { printf '  %sBAD%s  %s\n'   "$RED" "$OFF" "$*"; }
warn() { printf '  %s--%s   %s\n'   "$YEL" "$OFF" "$*"; }

# --------------------------------------------------------------- prerequisites
say "1. Prerequisites"

if ! command -v uhubctl >/dev/null 2>&1; then
  warn "uhubctl is not installed — installing it now"
  sudo apt-get install -y uhubctl >/dev/null 2>&1 \
    && ok "installed uhubctl" \
    || { bad "could not install uhubctl; run: sudo apt install uhubctl"; exit 1; }
else
  ok "uhubctl present"
fi

# uhubctl talks to the hub, so the hub needs a permissive node too. Without
# this everything below still works via sudo, but the SERVICE cannot do it
# unattended at boot, which is the entire point.
RULES=/etc/udev/rules.d/99-respeaker.rules
SRC="$(cd "$(dirname "$0")/.." && pwd)/systemd/99-respeaker.rules"
if ! cmp -s "$SRC" "$RULES" 2>/dev/null; then
  sudo cp "$SRC" "$RULES" \
    && sudo udevadm control --reload-rules \
    && sudo udevadm trigger \
    && ok "installed the updated udev rules (microphone + hub)" \
    || bad "could not install udev rules"
  sleep 2
else
  ok "udev rules already current"
fi

# ------------------------------------------------------------------- topology
say "2. Where is the microphone?"

SYSNAME=""
for d in /sys/bus/usb/devices/*; do
  [[ -r "$d/idVendor" && -r "$d/idProduct" ]] || continue
  if [[ "$(<"$d/idVendor"):$(<"$d/idProduct")" == "$ARRAY_ID" ]]; then
    SYSNAME="$(basename "$d")"; break
  fi
done

if [[ -z "$SYSNAME" ]]; then
  bad "no device $ARRAY_ID on the bus at all. Check the cable, then lsusb."
  exit 1
fi

if [[ "$SYSNAME" == *.* ]]; then
  HUB="${SYSNAME%.*}"; PORT="${SYSNAME##*.}"
  ok "array is at $SYSNAME — hub $HUB, port $PORT (behind an external hub)"
else
  HUB="${SYSNAME%%-*}-0"; PORT="${SYSNAME##*-}"
  warn "array is at $SYSNAME — root hub $HUB, port $PORT (straight into the mini PC)"
fi

# ----------------------------------------------------------------- capability
say "3. Can that port's power be cut?"

UH="uhubctl"
$UH >/dev/null 2>&1 || UH="sudo uhubctl"

if $UH -l "$HUB" 2>/dev/null | grep -q ppps; then
  ok "hub $HUB supports per-port power switching"
else
  bad "hub $HUB does NOT support per-port power switching."
  echo
  echo "  Hubs on this machine that DO:"
  $UH 2>/dev/null | grep ppps | sed 's/^/    /' || echo "    (none)"
  echo
  echo "  ${BOLD}What to do:${OFF} unplug the ReSpeaker from where it is now and"
  echo "  plug it into any free port on the Sabrent hub, then run this script"
  echo "  again. The array must be downstream of a hub that can switch power,"
  echo "  because nothing else revives it after a warm reboot."
  exit 1
fi

# --------------------------------------------------------------------- proof
say "4. Proving the recovery"

if pgrep -f "audio.service" >/dev/null; then
  warn "the audio service is running and holding the microphone — stopping it"
  systemctl --user stop assistant-audio 2>/dev/null
  sleep 1
fi

echo "  cutting power to $HUB port $PORT for 3 seconds..."
$UH -l "$HUB" -p "$PORT" -a cycle -d 3 >/dev/null 2>&1 \
  && ok "power cycled" \
  || { bad "uhubctl refused to cycle the port"; exit 1; }

echo "  waiting for it to come back..."
for _ in {1..20}; do
  lsusb -d "$ARRAY_ID" >/dev/null 2>&1 && break
  sleep 0.5
done
sleep 2

lsusb -d "$ARRAY_ID" >/dev/null 2>&1 \
  && ok "array re-enumerated" \
  || { bad "array did not come back — power it may not have restored"; exit 1; }

echo "  recording three seconds..."
if arecord -D plughw:Array,0 -f S16_LE -r 16000 -c 2 -d 3 /tmp/wedge_test.wav \
     >/dev/null 2>&1; then
  ok "capture works"
else
  bad "capture still returns an error after the power cycle."
  echo "     That would mean even cutting VBUS is not clearing it, which"
  echo "     would be new — send me the output of: arecord -D plughw:Array,0 \\"
  echo "       -f S16_LE -r 16000 -c 2 -d 3 /tmp/t.wav"
  exit 1
fi

# --------------------------------------------------------------------- finish
say "5. Done"
echo "  The service will now do all of that by itself: it reads a fraction of"
echo "  a second of audio at startup, and only if that fails does it cut the"
echo "  port's power and try again. Healthy boots pay nothing for it."
echo
echo "  Start it back up and reboot to confirm:"
echo "    systemctl --user start assistant-audio"
echo "    sudo reboot"
