#!/usr/bin/env bash
# Every 60 seconds: is the orchestrator answering, is the audio service
# attached, is the mic still enumerated? Restart what is broken.
#
# The failure this exists to catch is not a crash — systemd already handles
# crashes. It is the quiet one: a USB re-enumeration after a power blip
# leaves the audio service running but deaf, and nothing tells you until
# you try to talk to it three days later.
set -uo pipefail

HEALTH="http://127.0.0.1:8760/healthz"
log() { printf '%s watchdog: %s\n' "$(date -Is)" "$*"; }

# --- orchestrator answering? ---
if ! RESP=$(curl -fsS --max-time 5 "$HEALTH" 2>/dev/null); then
    log "core not answering — restarting"
    systemctl --user restart assistant-core
    exit 0
fi

# --- audio service attached? ---
if ! echo "$RESP" | grep -q '"audio_connected": *true'; then
    log "audio service not attached — restarting"
    systemctl --user restart assistant-audio
    exit 0
fi

# --- microphone still present? ---
# Adjust the pattern if you use a different array.
if command -v arecord >/dev/null; then
    if ! arecord -l 2>/dev/null | grep -qiE 'respeaker|xvf|xmos|usb'; then
        log "no USB capture device enumerated — restarting audio service"
        systemctl --user restart assistant-audio
        exit 0
    fi
fi

# --- spend sanity check ---
# Not a kill switch — the provider console cap is your real backstop. This
# just puts a loud line in the journal so a runaway shows up in `journalctl`
# rather than on a statement at the end of the month.
SPEND=$(echo "$RESP" | sed -n 's/.*"spend_24h_usd": *\([0-9.]*\).*/\1/p')
if [[ -n "$SPEND" ]] && awk "BEGIN{exit !($SPEND > 3.0)}"; then
    log "WARNING: \$$SPEND spent in the last 24h — expected well under \$1"
fi
