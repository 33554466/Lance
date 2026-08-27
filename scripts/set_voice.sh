#!/usr/bin/env bash
# ============================================================================
# set_voice.sh — swap the assistant's voice.
#
#   ./scripts/set_voice.sh en_US-ryan-medium
#   ./scripts/set_voice.sh en_GB-alba-medium
#
# Listen first: https://rhasspy.github.io/piper-samples/
# The name under each sample is exactly what you pass here.
#
# Downloads the two files a Piper voice consists of, sanity-checks them,
# updates config.yaml, and restarts the audio service. The old voice is kept,
# so switching back is another one-liner.
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

BOLD=$'\e[1m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; OFF=$'\e[0m'
ok()   { printf '  %sOK%s   %s\n' "$GRN" "$OFF" "$*"; }
bad()  { printf '  %sBAD%s  %s\n' "$RED" "$OFF" "$*"; }
warn() { printf '  %s--%s   %s\n' "$YEL" "$OFF" "$*"; }

VOICE="${1:-}"
if [[ -z "$VOICE" ]]; then
  echo "Usage: $0 <voice-name>"
  echo
  echo "  Browse and listen:  https://rhasspy.github.io/piper-samples/"
  echo "  Then pass the name shown, e.g. en_US-ryan-medium"
  echo
  echo "Currently installed:"
  ls -1 models/piper/*.onnx 2>/dev/null | xargs -n1 basename 2>/dev/null \
    | sed 's/\.onnx$//;s/^/    /' || echo "    (none)"
  echo
  echo "In use:"
  grep -E '^\s+voice:' config.yaml | sed 's/^/  /'
  exit 1
fi

# en_US-ryan-medium  ->  en / en_US / ryan / medium
LOCALE="${VOICE%%-*}"
REST="${VOICE#*-}"
NAME="${REST%%-*}"
QUALITY="${REST##*-}"
LANG="${LOCALE%%_*}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG}/${LOCALE}/${NAME}/${QUALITY}"

echo
echo "${BOLD}Fetching ${VOICE}${OFF}"
echo "  from ${BASE}"

mkdir -p models/piper
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for EXT in onnx onnx.json; do
  if ! curl -fL --progress-bar -o "$TMP/${VOICE}.${EXT}" "${BASE}/${VOICE}.${EXT}"; then
    bad "could not download ${VOICE}.${EXT}"
    echo
    echo "  Tried: ${BASE}/${VOICE}.${EXT}"
    echo
    echo "  That usually means the voice name is wrong. Names are exact,"
    echo "  including the quality suffix — check the spelling against"
    echo "  https://rhasspy.github.io/piper-samples/ and note that not every"
    echo "  voice exists at every quality level."
    exit 1
  fi
done
ok "downloaded both files"

# The sidecar JSON tells us the sample rate and whether this is a
# multi-speaker model. Both matter, so read them rather than assume.
INFO="$(python3 - "$TMP/${VOICE}.onnx.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"BAD|{exc}")
    raise SystemExit
print("OK|%s|%s" % (d.get("audio", {}).get("sample_rate", "?"),
                    d.get("num_speakers", 1)))
PY
)"
if [[ "$INFO" == BAD* ]]; then
  bad "the voice config did not parse: ${INFO#BAD|}"
  exit 1
fi
RATE="$(echo "$INFO" | cut -d'|' -f2)"
SPEAKERS="$(echo "$INFO" | cut -d'|' -f3)"
ok "sample rate ${RATE} Hz (the speaker resamples to the array automatically)"

if [[ "$SPEAKERS" != "1" ]]; then
  warn "this is a MULTI-SPEAKER model (${SPEAKERS} voices)."
  warn "The assistant does not pass a speaker id, so you will get voice 0"
  warn "and it may not be the one you heard in the sample. A single-speaker"
  warn "voice is the safer choice unless you want me to add speaker support."
fi

mv "$TMP/${VOICE}.onnx" "$TMP/${VOICE}.onnx.json" models/piper/
ok "installed to models/piper/"

# --- update config -------------------------------------------------------
OLD="$(grep -oP '(?<=^  voice: ").*(?=")' config.yaml || true)"
sed -i "s|^  voice: \".*\"|  voice: \"models/piper/${VOICE}.onnx\"|" config.yaml
NEW="$(grep -oP '(?<=^  voice: ").*(?=")' config.yaml || true)"
if [[ "$NEW" != "models/piper/${VOICE}.onnx" ]]; then
  bad "could not update config.yaml — set tts.voice by hand:"
  echo "      voice: \"models/piper/${VOICE}.onnx\""
  exit 1
fi
ok "config updated (was: ${OLD:-unset})"

python3 -c "import yaml;yaml.safe_load(open('config.yaml'))" \
  && ok "config.yaml still parses" || { bad "config.yaml is now invalid"; exit 1; }

# --- restart and prove it ------------------------------------------------
systemctl --user restart assistant-audio 2>/dev/null && ok "audio service restarted" \
  || warn "could not restart the service — do it by hand"

echo
echo "${BOLD}Hearing it${OFF}"
echo "  Say the wake word and ask it something, or test directly:"
echo
echo "    source .venv/bin/activate"
echo "    echo \"Hello Brendan, this is my new voice.\" | \\"
echo "      piper --model models/piper/${VOICE}.onnx --output-raw | \\"
echo "      aplay -r ${RATE} -f S16_LE -t raw -D plughw:Array,0"
echo
echo "  To go back:  ./scripts/set_voice.sh ${OLD##*/}"
echo "               (or just: sed -i 's|^  voice:.*|  voice: \"${OLD}\"|' config.yaml)"
