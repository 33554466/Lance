#!/usr/bin/env bash
# Once a minute: is the orchestrator answering, and is the audio service
# actually attached to it? Restart what is genuinely broken, and only that.
#
# The failure this exists to catch is not a crash — systemd already handles
# crashes, and handles them better than a shell script would. It is the quiet
# one: a process that is running, that systemd is perfectly happy with, and
# that has silently stopped doing its job.
#
# What this deliberately does NOT do is restart things it cannot fix. An
# unplugged microphone is not a software problem, and restarting the audio
# service every sixty seconds while the cable is out reloads Whisper every
# sixty seconds, burns a core, and fixes nothing. The service's own recovery
# ladder already waits patiently for the cable to come back; the watchdog's
# job there is to say so in the journal, loudly, once.
set -uo pipefail

HEALTH="http://127.0.0.1:8760/healthz"
STATE="${XDG_RUNTIME_DIR:-/tmp}/assistant-watchdog"
# Do not restart the same unit more often than this. A restart that did not
# help will not help on the next tick either, and a restart loop destroys the
# evidence you need to work out why.
COOLDOWN=600
SPEND_ALERT=3.0

mkdir -p "$STATE"
log() { printf '%s watchdog: %s\n' "$(date -Is)" "$*"; }

# Returns 0 if we are allowed to restart $1 right now.
may_restart() {
    local unit="$1" stamp="$STATE/$1.last" now last
    now=$(date +%s)
    if [[ -f "$stamp" ]]; then
        last=$(cat "$stamp" 2>/dev/null || echo 0)
        if (( now - last < COOLDOWN )); then
            log "$unit is still broken, but it was restarted $(( now - last ))s ago — leaving it alone"
            return 1
        fi
    fi
    echo "$now" > "$stamp"
    return 0
}

restart() {
    local unit="$1" why="$2"
    if may_restart "$unit"; then
        log "$why — restarting $unit"
        systemctl --user restart "$unit"
    fi
}

# --- is the orchestrator answering at all? ---------------------------------
if ! RESP=$(curl -fsS --max-time 8 "$HEALTH" 2>/dev/null); then
    if ! systemctl --user is-active --quiet assistant-core; then
        restart assistant-core "core is not running"
    else
        restart assistant-core "core is running but not answering /healthz"
    fi
    exit 0
fi

# --- is the audio service attached? ----------------------------------------
# Starlette emits JSON with no spaces; the pattern tolerates both.
if ! grep -q '"audio_connected": *true' <<<"$RESP"; then
    if ! systemctl --user is-active --quiet assistant-audio; then
        restart assistant-audio "the audio service is not running"
    else
        # Alive but not connected: its websocket died and did not come back.
        # This is the one the watchdog exists for.
        restart assistant-audio "the audio service is running but not attached to core"
    fi
    exit 0
fi

# --- is the microphone still on the bus? -----------------------------------
# Reported, never acted on. Nothing a restart can do will plug a cable in, and
# audio/service.py already waits for it without help.
if command -v lsusb >/dev/null 2>&1; then
    if ! lsusb 2>/dev/null | grep -qi '2886:001a'; then
        log "WARNING: the microphone array is not on the USB bus. Nothing in software can fix that — reseat the cable."
    fi
fi

# --- spend sanity check ----------------------------------------------------
# Not a kill switch — the provider console cap is the real backstop. This puts
# a loud line in the journal so a runaway shows up in `journalctl` rather than
# on a statement at the end of the month.
SPEND=$(sed -n 's/.*"spend_24h_usd": *\([0-9.]*\).*/\1/p' <<<"$RESP")
if [[ -n "$SPEND" ]] && awk "BEGIN{exit !($SPEND > $SPEND_ALERT)}"; then
    log "WARNING: \$$SPEND spent in the last 24h — expected well under \$1"
fi

# Everything healthy. Clear the cooldowns so a future failure gets acted on
# immediately rather than waiting out a stamp from an unrelated incident.
rm -f "$STATE"/*.last 2>/dev/null || true
