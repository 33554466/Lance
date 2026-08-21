#!/usr/bin/env bash
# ============================================================
# Home AI Appliance — installer
#
# Idempotent: safe to re-run. Does NOT install systemd units or start
# anything; that is phase 9. This gets you to the point where you can run
# each service by hand and watch it work.
#
#   ./install.sh
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }

# ------------------------------------------------------------
say "System packages"
# libportaudio2  - sounddevice needs it to reach ALSA/PipeWire
# ffmpeg         - faster-whisper uses it for resampling
# python3-venv   - Ubuntu splits this out from python3
# cups / sane    - not needed until phase 7, but installing now means the
#                  peripheral checks in phase 2 can actually run
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-dev build-essential \
    libportaudio2 portaudio19-dev \
    ffmpeg \
    chromium-browser \
    cups sane-utils sane-airscan ipp-usb \
    v4l-utils \
    iw ethtool wireless-tools \
    uhubctl \
    curl jq sqlite3

# ------------------------------------------------------------
say "Python virtual environment"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# No PyTorch. Silero VAD runs from openWakeWord's bundled ONNX model —
# see the comment block in requirements.txt for why that matters.
say "Python packages"
pip install -r requirements.txt

# openWakeWord separately, without its dependency list. It insists on
# tflite-runtime, which has no Python 3.12 wheels; we use the ONNX backend
# and never touch tflite. See the comment block in requirements.txt.
say "openWakeWord (--no-deps, on purpose)"
pip install --no-deps 'openwakeword>=0.6.0'
python -c "import openwakeword, onnxruntime; print('    openWakeWord + ONNX runtime OK')"

# ------------------------------------------------------------
say "Directories"
mkdir -p data/vault models/piper models/whisper logs

# ------------------------------------------------------------
say "Models"
./scripts/fetch_models.sh

# ------------------------------------------------------------
say "Secrets"
if [[ ! -f .env ]]; then
    cp .env.example .env
    chmod 600 .env
    warn "Created .env — put your ANTHROPIC_API_KEY in it before starting."
else
    chmod 600 .env
    echo ".env already exists, left alone."
fi

# ------------------------------------------------------------
say "Freezing dependency versions"
pip freeze > requirements.lock.txt
echo "Wrote requirements.lock.txt — install from this from now on."

cat <<'EOF'

------------------------------------------------------------
Install complete.

Next:
  1. Put your API key in .env
  2. Terminal 1:  source .venv/bin/activate && uvicorn core.app:app --host 127.0.0.1 --port 8760
  3. Browser:     http://127.0.0.1:8760   (type a question — no voice yet)
  4. Terminal 2:  source .venv/bin/activate && python -m audio.service

If step 3 works and step 4 does not, the problem is audio, not the
assistant. That separation is the whole point of running them apart.
------------------------------------------------------------
EOF
