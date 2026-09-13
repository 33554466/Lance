# Home AI Appliance

A dedicated voice assistant for a Beelink SER8 running Ubuntu 24.04. Wake
word, transcription and speech all happen locally; only the text transcript
goes to Anthropic.

The step-by-step build guide is the accompanying Word document. This file is
the short version for when you are sitting in front of the machine.

## Shape

Three processes under systemd, each independently restartable.

```
 [ mic array ] -> openWakeWord -> Silero VAD -> faster-whisper
                                                     |
                                              audio.service
                                                     | ws://127.0.0.1:8760
                                                     v
                                              core.app  (FastAPI)
                                             /         \
                                    Anthropic API    ui/index.html
                                                     (Chromium kiosk)
```

The one rule worth internalising: **Python owns the sound card, the browser
only draws pixels.** No `getUserMedia`, no autoplay policy, no browser audio
stack. Every audio bug stays a Python bug.

## Install

```bash
git clone <your repo> ~/assistant && cd ~/assistant
./install.sh                     # apt + venv + models, ~15 min
nano .env                        # ANTHROPIC_API_KEY=...
./scripts/setup_credential.sh    # moves it into a systemd credential
```

## Run it by hand (do this first)

Two terminals. Running them apart is how you tell an audio problem from an
assistant problem.

```bash
# Terminal 1 — orchestrator
source .venv/bin/activate
uvicorn core.app:app --host 127.0.0.1 --port 8760

# Terminal 2 — audio (first start takes ~20s to load models)
source .venv/bin/activate
python -m audio.service
```

Open <http://127.0.0.1:8760>. Type a question — that path works with no
microphone at all. Then say the wake word.

## Verify

```bash
./scripts/check_peripherals.sh   # mic, printer, scanner, camera, network
./scripts/measure_network.sh     # wired vs wireless, on YOUR network
.venv/bin/python -m tests.run    # every suite; exits with the number that failed
.venv/bin/python -m tests.run --quiet         # one line each
.venv/bin/python -m tests.run store tools     # just these two
curl -s localhost:8760/healthz | jq
curl -s localhost:8760/stats/wake | jq   # read this daily for a week
```

`tests/run.py` picks up anything matching `tests/test_*.py`, so a new suite
needs no wiring. `test_pipeline` exercises everything except the network call and the sound
card. If it passes and the device still misbehaves, the problem is hardware
or credentials.

On Wi-Fi, do this before anything else — it is worth more than the choice of
link itself:

```bash
sudo iw dev wlan0 set power_save off
printf '[connection]\nwifi.powersave = 2\n' | \
  sudo tee /etc/NetworkManager/conf.d/wifi-powersave.conf
sudo systemctl restart NetworkManager
```

Linux dynamic power management parks the radio between packets — measured at
~280 ms on the first packet after an idle gap, against under 1 ms with it
off. This appliance is idle almost all the time, so it pays that on every
request.

## Tuning

Everything lives in `config.yaml`. The three that matter:

| Setting | Start | Symptom if wrong |
|---|---|---|
| `wake_word.threshold` | 0.55 | Too low: triggers on the television. Too high: you repeat yourself. |
| `vad.silence_ms` | 700 | Too low: cuts you off. Too high: feels dead. |
| `stt.model` | `small.en` | Drop to `base.en` if transcription lags; it is faster and less accurate. |

Tune the wake threshold against `/stats/wake` after a full day of normal
household noise, not against a benchmark. A trigger with an empty transcript
is almost always a false positive.

## As an appliance

```bash
mkdir -p ~/.config/systemd/user
cp systemd/*.service systemd/*.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now assistant-core assistant-audio
systemctl --user enable --now assistant-watchdog.timer   # the timer, not the service
loginctl enable-linger $USER     # runs without you logged in
journalctl --user -u assistant-audio -f
```

## Layout

```
core/app.py       orchestrator, WebSocket protocol, sentence chunking
core/provider.py  the ONLY file that knows which AI platform you use
core/router.py    tier selection (small/mid/top)
core/context.py   history window + prompt-cache stability
core/db.py        SQLite: history, usage, wake events
audio/listener.py wake word -> VAD -> Whisper
audio/speaker.py  Piper TTS + barge-in
audio/service.py  owns the sound devices, talks to core
ui/index.html     kiosk page, vanilla JS
```

## Cost

Logged per request in the `usage` table and surfaced at `/healthz`. Expect
roughly $8–15/month at 40 exchanges a day with tiered routing and prompt
caching on. **Set a hard monthly spend cap in the Anthropic console before
your first request** — the watchdog only warns.
