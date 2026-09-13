#!/usr/bin/env bash
# ============================================================
# Home AI Appliance — installer
#
# Idempotent: safe to re-run. Does NOT install systemd units or start
# anything. That is the "As an appliance" section of README.md — this gets you
# to the point where you can run each service by hand and watch it work.
#
#   ./install.sh
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
ok()   { printf '    \033[1;32m✓\033[0m %s\n' "$*"; }

# ------------------------------------------------------------
say "System packages"
# libportaudio2   - sounddevice needs it to reach ALSA/PipeWire
# ffmpeg          - faster-whisper uses it for resampling
# python3-venv    - Ubuntu splits this out from python3
# mpv             - video and audio playback (core/media.py drives it over an
#                   IPC socket; there is deliberately no browser in that path)
# wmctrl, xdotool - core/desktop.py's window verbs. Without these the whole
#                   desktop feature is a runtime error on a fresh box.
# sane-*          - so scripts/check_peripherals.sh can probe the scanner
# uhubctl         - one rung of the microphone recovery ladder
#
# NOT installed: cups and ipp-usb. core/printer.py talks to the receipt
# printer through libusb on purpose — no queue, no spooler, no job stuck in
# "processing" for an hour. ipp-usb and CUPS' usb backend CLAIM USB printer
# interfaces, which is a live conflict with python-escpos: the printer is
# then busy for reasons nothing explains. They were in this list for a
# peripheral check that does not need them.
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-dev build-essential \
    libportaudio2 portaudio19-dev \
    ffmpeg \
    mpv \
    wmctrl xdotool \
    sane-utils sane-airscan \
    v4l-utils \
    iw ethtool wireless-tools \
    uhubctl \
    curl jq sqlite3

# Chromium for the kiosk. On 24.04 the `chromium-browser` deb is a
# transitional package for the snap, and the binary lands in /snap/bin —
# which is why systemd/assistant-ui.service points there and not at
# /usr/bin/chromium-browser.
say "Chromium (kiosk display)"
if command -v chromium >/dev/null 2>&1 || [[ -x /snap/bin/chromium ]]; then
    ok "already present"
else
    sudo apt-get install -y chromium-browser || sudo snap install chromium
fi
for c in /snap/bin/chromium /usr/bin/chromium /usr/bin/chromium-browser; do
    if [[ -x "$c" ]]; then ok "binary: $c"; break; fi
done

# ------------------------------------------------------------
say "Python virtual environment"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel
ok "Python $(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

# No PyTorch. Silero VAD runs from openWakeWord's bundled ONNX model —
# see the comment block in requirements.txt for why that matters.
say "Python packages"
if [[ -f requirements.lock.txt ]]; then
    echo "    installing pinned versions from requirements.lock.txt"
    pip install -r requirements.lock.txt
else
    pip install -r requirements.txt
fi

# openWakeWord separately, without its dependency list. It insists on
# tflite-runtime; we use the ONNX backend and never touch tflite. On Python
# 3.12 there is no tflite wheel at all, so a normal install fails outright.
say "openWakeWord (--no-deps, on purpose)"
pip install --no-deps 'openwakeword>=0.6.0'
python -c "import openwakeword, onnxruntime; print('    openWakeWord + ONNX runtime OK')"

# ------------------------------------------------------------
say "Directories"
mkdir -p data models/piper models/whisper models/embed workouts playbooks
chmod 700 data

# ------------------------------------------------------------
say "Models"
./scripts/fetch_models.sh
# Embeddings for semantic recall — about 69 MB. Skipped entirely before, which
# left a fresh install with recall switched on and no model to do it with.
if [[ -x ./scripts/fetch_embed_model.sh ]]; then
    ./scripts/fetch_embed_model.sh
fi

# ------------------------------------------------------------
say "udev rules"
# Without these the microphone recovery ladder cannot work: uhubctl needs to
# reach the hub, and a wedged array needs the reset path. This was only ever
# installed by scripts/fix_reboot_wedge.sh, which nothing pointed you at.
if [[ -f systemd/99-respeaker.rules ]]; then
    if sudo cmp -s systemd/99-respeaker.rules \
            /etc/udev/rules.d/99-respeaker.rules 2>/dev/null; then
        ok "already installed and unchanged"
    else
        sudo install -m 644 systemd/99-respeaker.rules /etc/udev/rules.d/
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        ok "installed /etc/udev/rules.d/99-respeaker.rules"
        warn "unplug and replug the microphone once so the new rules apply"
    fi
fi

# uhubctl needs root to cut port power, and audio/usbpower.py tries
# `sudo -n uhubctl` — which fails silently under a systemd user service with
# no tty unless a NOPASSWD rule exists. Say so rather than leaving a dead rung
# in the recovery ladder.
if ! sudo -n true 2>/dev/null; then
    warn "for the power-cycle rung of microphone recovery to work unattended:"
    echo "        echo \"$USER ALL=(root) NOPASSWD: $(command -v uhubctl)\" | sudo tee /etc/sudoers.d/uhubctl"
    echo "        sudo chmod 440 /etc/sudoers.d/uhubctl"
fi

# ------------------------------------------------------------
say "Secrets"
# The credential is the real answer; .env is only for a manual run. See
# scripts/setup_credential.sh and the comments in .env.example.
if [[ ! -f .env ]]; then
    cp .env.example .env
    chmod 600 .env
    ok "created .env (empty key, on purpose — a placeholder gets you a 401)"
else
    chmod 600 .env
    ok ".env already exists, left alone"
fi
warn "put the key where systemd can hand it over:  ./scripts/setup_credential.sh"

# ------------------------------------------------------------
say "Dependency versions"
# Write a candidate and show the difference, rather than overwriting a tracked
# file behind your back. The old version of this line ran
# `pip freeze > requirements.lock.txt` unconditionally, which bakes in
# whatever you happen to have installed — including openwakeword, whose
# presence in the lock is what makes the lock uninstallable.
pip freeze | grep -v '^openwakeword==' > requirements.lock.txt.new
if [[ -f requirements.lock.txt ]] && diff -q \
        <(grep -v '^#' requirements.lock.txt | grep . || true) \
        requirements.lock.txt.new >/dev/null 2>&1; then
    rm -f requirements.lock.txt.new
    ok "requirements.lock.txt matches what is installed"
else
    warn "installed packages differ from requirements.lock.txt:"
    diff <(grep -v '^#' requirements.lock.txt 2>/dev/null | grep . || true) \
         requirements.lock.txt.new | sed 's/^/        /' || true
    echo "    Wrote requirements.lock.txt.new. If that looks right:"
    echo "        mv requirements.lock.txt.new requirements.lock.txt"
    echo "    Keep the header comments from the old file — they explain the"
    echo "    openwakeword --no-deps step, which a lock file cannot express."
fi

cat <<'TXT'

------------------------------------------------------------
Install complete.

Next:
  1. ./scripts/setup_credential.sh
  2. .venv/bin/python -m tests.run
  3. Terminal 1:  .venv/bin/uvicorn core.app:app --host 127.0.0.1 --port 8760
  4. Browser:     http://127.0.0.1:8760/dashboard  (type a question, no voice yet)
  5. Terminal 2:  .venv/bin/python -m audio.service

If step 4 works and step 5 does not, the problem is audio, not the
assistant. That separation is the whole point of running them apart.

Then see "As an appliance" in README.md for the systemd units.
------------------------------------------------------------
TXT
