#!/usr/bin/env bash
# Move the API key out of the environment and into a systemd credential.
#
# Today the key lives in .env, which assistant-core.service loads with
# EnvironmentFile=. That puts it in the process environment, which means:
#
#   * it is in /proc/<pid>/environ
#   * every child process inherits it — mpv, yt-dlp, wmctrl, xdotool. yt-dlp
#     is network-facing and runs per-site extractor code.
#   * a `tar` of the project directory picks it up, which is how a key got
#     shared once.
#
# LoadCredential fixes all three: systemd mounts a 0700 tmpfs, writes the file
# 0400, hands the path to the service as $CREDENTIALS_DIRECTORY, and unmounts
# it on stop. Nothing else on the box can read it, and no child inherits it.
#
# Safe to re-run. Changes nothing until it has a key in hand.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITS="$HOME/.config/systemd/user"
SECRETS="$ROOT/secrets"
KEYFILE="$SECRETS/anthropic-key"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. find a key -----------------------------------------------------
say "Finding the key"
KEY=""
if [[ -f "$KEYFILE" ]]; then
    KEY="$(tr -d '[:space:]' < "$KEYFILE")"
    [[ -n "$KEY" ]] && ok "already have one in secrets/anthropic-key"
fi
if [[ -z "$KEY" && -f "$ROOT/.env" ]]; then
    KEY="$(sed -n 's/^ANTHROPIC_API_KEY=//p' "$ROOT/.env" | head -1 \
           | tr -d '[:space:]' | tr -d "\"'")"
    [[ -n "$KEY" ]] && ok "found it in .env (${#KEY} characters)"
fi
if [[ -z "$KEY" ]]; then
    read -rsp "  Paste the key, then Enter: " KEY && echo
    KEY="$(printf '%s' "$KEY" | tr -d '[:space:]' \
           | sed -E 's/.*(sk-ant-[A-Za-z0-9_-]+).*/\1/')"
fi
[[ -n "$KEY" ]] || die "no key given, nothing changed"
[[ "$KEY" == sk-ant-* ]] || warn "that does not start with sk-ant- — continuing anyway"
(( ${#KEY} > 200 )) && warn "that is ${#KEY} characters, which is longer than an Anthropic key usually is"

# --- 2. store it -------------------------------------------------------
say "Storing it"
mkdir -p "$SECRETS"
chmod 700 "$SECRETS"
umask 077
printf '%s' "$KEY" > "$KEYFILE"
chmod 600 "$KEYFILE"
ok "$KEYFILE  (mode $(stat -c %a "$KEYFILE"), ${#KEY} characters)"

if ! grep -qxF 'secrets/' "$ROOT/.gitignore" 2>/dev/null; then
    printf 'secrets/\n' >> "$ROOT/.gitignore"
    ok "added secrets/ to .gitignore"
else
    ok "secrets/ is already gitignored"
fi

# --- 3. install the units ----------------------------------------------
say "Installing units"
mkdir -p "$UNITS"
for unit in assistant-core.service assistant-watchdog.service \
            assistant-watchdog.timer; do
    install -m 644 "$ROOT/systemd/$unit" "$UNITS/$unit"
    ok "$unit"
done
warn "assistant-audio.service and assistant-ui.service were NOT touched — they are working, leave them alone"

systemctl --user daemon-reload
ok "daemon-reload"

# --- 4. restart and verify ---------------------------------------------
say "Restarting core"
systemctl --user restart assistant-core
sleep 6

if ! curl -fsS --max-time 8 http://127.0.0.1:8760/healthz >/dev/null 2>&1; then
    die "core is not answering. Look at:  journalctl --user -u assistant-core -n 40"
fi
ok "core is answering"

if journalctl --user -u assistant-core --since "1 min ago" --no-pager 2>/dev/null \
        | grep -q "api key loaded from the systemd credential"; then
    ok "the key is coming from the credential"
    FROM_CRED=1
else
    warn "no 'loaded from the systemd credential' line yet — that message appears on the first API call, not at startup"
    FROM_CRED=0
fi

# --- 5. the watchdog ---------------------------------------------------
say "Enabling the watchdog"
systemctl --user enable --now assistant-watchdog.timer >/dev/null 2>&1
systemctl --user list-timers 'assistant*' --no-pager | sed -n '1,4p'
ok "it runs every minute; see it with:  journalctl --user -u assistant-watchdog -f"

# --- 6. what is left ---------------------------------------------------
say "Last step, by hand"
if [[ -f "$ROOT/.env" ]] && grep -q '^ANTHROPIC_API_KEY=.\+' "$ROOT/.env"; then
    cat <<TXT
  The key is now in two places. Once you have asked her something out loud and
  she answered, empty the copy in .env:

      sed -i 's/^ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=/' $ROOT/.env

  Leave the EnvironmentFile= line out of the unit (it already is) and the
  credential becomes the only source. Keeping the .env line empty rather than
  deleting the file means a manual run still has somewhere obvious to put one.
TXT
else
    ok ".env carries no key — the credential is the only source"
fi
echo
