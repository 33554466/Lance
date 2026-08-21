#!/usr/bin/env bash
# Download the local models: Piper voice, openWakeWord, Whisper.
# Idempotent — skips anything already present.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VOICE_DIR="models/piper"
VOICE="en_US-lessac-medium"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

mkdir -p "$VOICE_DIR" models/whisper

# ---- Piper voice ----
# lessac-medium is a good default: clear, neutral, unhurried, and small
# enough that synthesis starts almost immediately on CPU. Browse
# https://rhasspy.github.io/piper-samples/ if you want a different voice —
# swap both files and update tts.voice in config.yaml.
if [[ ! -f "$VOICE_DIR/$VOICE.onnx" ]]; then
    echo "==> Piper voice: $VOICE"
    curl -fL --progress-bar -o "$VOICE_DIR/$VOICE.onnx"      "$BASE/$VOICE.onnx"
    curl -fL --progress-bar -o "$VOICE_DIR/$VOICE.onnx.json" "$BASE/$VOICE.onnx.json"
else
    echo "Piper voice present, skipping."
fi

# ---- openWakeWord ----
# Pulls the bundled models plus the shared melspectrogram/embedding
# frontends. Doing it here rather than on first run means the first wake
# word does not stall for thirty seconds.
echo "==> openWakeWord models"
python - <<'PY'
import yaml, pathlib
from openwakeword.utils import download_models
cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parent.parent / "config.yaml")) \
      if False else yaml.safe_load(open("config.yaml"))
name = cfg["wake_word"]["model"]
download_models(model_names=[name])
print(f"    ok: {name}")
PY

# ---- Whisper ----
# Downloaded and converted on first load; do it now so phase 5 is not
# waiting on a 500 MB fetch.
echo "==> faster-whisper model (this one takes a minute)"
python - <<'PY'
import yaml
from faster_whisper import WhisperModel
cfg = yaml.safe_load(open("config.yaml"))
s = cfg["stt"]
WhisperModel(s["model"], device=s["device"], compute_type=s["compute_type"],
             download_root="models/whisper")
print(f"    ok: {s['model']} ({s['compute_type']})")
PY

echo
echo "All models ready."
du -sh models/* 2>/dev/null || true
