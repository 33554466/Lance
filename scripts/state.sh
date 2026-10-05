#!/usr/bin/env bash
# One command, one paste: everything anybody needs to know what this box is
# actually doing right now.
#
# This exists because of the shape of every debugging session on this
# appliance so far: a dozen round trips of "run this, paste the output", when
# the answer was always in the same eight or nine places. So: look in all of
# them, once, and bound the output so it fits in a message.
#
#     bash scripts/state.sh            # to the screen
#     bash scripts/state.sh > /tmp/s   # to a file
#
# It reads and never changes anything. It prints no secrets: the API key is
# never read, and the one place a key could appear in config is masked.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT="$PWD"
VENV="$ROOT/.venv/bin/python"
[[ -x "$VENV" ]] || VENV="python3"

hr()  { printf '\n== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "LANCE STATE  $(date '+%Y-%m-%d %H:%M:%S %Z')  on $(hostname)"
echo "uptime:$(uptime -p 2>/dev/null | sed 's/^up//')"

# ---------------------------------------------------------------- services
hr services
# `is-active` exits non-zero for a stopped unit, so the `|| echo "?"` that
# used to be here appended a second line and the output came out mangled.
# `show --value` always exits 0 and says the same thing.
for unit in assistant-core assistant-audio assistant-ui; do
    state=$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)
    since=$(systemctl --user show "$unit" -p ActiveEnterTimestamp --value 2>/dev/null)
    printf '  %-22s %-10s %s\n' "$unit" "${state:-unknown}" "${since:-—}"
done
# The watchdog is a oneshot fired by a timer: "inactive" is what a HEALTHY
# one looks like between runs, and reporting that next to three services
# that must stay active reads as a fault. Report the timer instead.
tstate=$(systemctl --user show assistant-watchdog.timer -p ActiveState --value 2>/dev/null)
last=$(systemctl --user show assistant-watchdog.service -p ExecMainExitTimestamp --value 2>/dev/null)
code=$(systemctl --user show assistant-watchdog.service -p ExecMainStatus --value 2>/dev/null)
printf '  %-22s %-10s last run %s%s\n' "assistant-watchdog" \
    "${tstate:-no timer}" "${last:-never}" \
    "$([[ -n "${code:-}" && "$code" != "0" ]] && echo "  EXIT $code")"

# ---------------------------------------------------------------- health
hr health
curl -s --max-time 5 localhost:8760/healthz || echo "  (no answer from the orchestrator)"
echo

# The status wall knows about twenty-nine things; print only what is wrong.
curl -s --max-time 5 localhost:8760/status.json 2>/dev/null | "$VENV" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("  (no status wall on this build)"); raise SystemExit
c = d.get("counts", {})
print(f'"'"'  verdict {d.get("verdict")}  —  {c.get("ok",0)} ok, '"'"'
      f'"'"'{c.get("warn",0)} warn, {c.get("fail",0)} fail, {c.get("skip",0)} skip'"'"')
for p in d.get("problems", []):
    print(f'"'"'  {p["status"].upper():5} {p["name"]}: {p["detail"]}'"'"')
    if p.get("fix"):
        print(f'"'"'        fix: {p["fix"]}'"'"')
' 2>/dev/null

# ---------------------------------------------------------------- code
hr git
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "  branch:   $(git -C "$ROOT" branch --show-current)"
    echo "  head:     $(git -C "$ROOT" log -1 --format='%h %ad %s' --date=short)"
    remote=$(git -C "$ROOT" remote -v | head -1)
    echo "  remote:   ${remote:-none}"
    dirty=$(git -C "$ROOT" status --porcelain | wc -l)
    echo "  uncommitted: $dirty file(s)"
    git -C "$ROOT" status --short | head -25 | sed 's/^/    /'
else
    echo "  not a git repository"
fi

hr "file fingerprints"
# The drift that has actually bitten: a file quietly older than the running
# process, because an old archive overwrote it.
for f in core/app.py core/tools.py core/intervals.py core/monitor.py \
         core/selfcheck.py core/printer.py core/workout.py \
         ui/index.html ui/status.html; do
    if [[ -f "$ROOT/$f" ]]; then
        printf '  %-22s %s  %s\n' "$(basename "$f")" \
            "$(md5sum "$ROOT/$f" | cut -c1-12)" \
            "$(date -r "$ROOT/$f" '+%m-%d %H:%M')"
    else
        printf '  %-22s MISSING\n' "$(basename "$f")"
    fi
done

# ---------------------------------------------------------------- tools
hr "tools she has"
"$VENV" - <<'PY' 2>/dev/null | sed 's/^/  /' || echo "  (could not load the app)"
import sys, types
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))
try:
    from core.app import local_tools
    names = sorted(t["name"] for t in local_tools.schemas())
except Exception as exc:
    print(f"FAILED TO LOAD: {type(exc).__name__}: {exc}")
    raise SystemExit
print(f"{len(names)} registered")
import textwrap
print(textwrap.fill(", ".join(names), 72))
PY

# ---------------------------------------------------------------- config
hr "config, the parts that change behaviour"
"$VENV" - <<'PY' 2>/dev/null | sed 's/^/  /' || echo "  (config unreadable)"
import yaml
c = yaml.safe_load(open("config.yaml"))
def g(path, default="—"):
    cur = c
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    # Never print anything that smells like a credential.
    s = str(cur)
    if "sk-" in s or "key" in path.lower():
        return "***"
    return s
for path in ("identity.name", "wake_word.model", "wake_word.threshold",
             "stt.model", "tts.voice", "printer.enabled",
             "workouts.enabled", "casework.enabled", "intervals.enabled",
             "status.enabled", "selfcheck.enabled", "media.enabled",
             "location.city"):
    print(f"{path:26} {g(path)}")
PY

# ---------------------------------------------------------------- journal
hr "core — last errors"
journalctl --user -u assistant-core --since "2 hours ago" --no-pager 2>/dev/null \
  | grep -iE "error|traceback|exception|failed" | tail -12 | cut -c1-160 \
  | sed 's/^/  /' || echo "  (none)"

hr "core — last activity"
out=$(journalctl --user -u assistant-core --since "2 hours ago" --no-pager 2>/dev/null \
      | grep -iE "routed to|tools used" | tail -8 | cut -c1-160)
[[ -n "$out" ]] && sed 's/^/  /' <<<"$out" || echo "  (nobody has spoken to her in two hours)"

hr "audio — last errors"
journalctl --user -u assistant-audio --since "2 hours ago" --no-pager 2>/dev/null \
  | grep -iE "error|warning|recover|wedge|probe hung" | tail -10 | cut -c1-160 \
  | sed 's/^/  /' || echo "  (none)"

# ---------------------------------------------------------------- machine
hr machine
echo "  python:  $("$VENV" -V 2>&1)"
echo "  disk:    $(df -h "$ROOT" | awk 'NR==2 {print $4" free of "$2}')"
have lsusb && echo "  usb:     $(lsusb | grep -ciE '2886:001a|04b8:0e20') of 2 peripherals on the bus"
"$VENV" -m pip list 2>/dev/null \
  | grep -iE "^(openwakeword|faster-whisper|piper-tts|python-escpos|fastembed|yt-dlp|anthropic|onnxruntime) " \
  | sed 's/^/  /'

echo
echo "== end =================================================================="
