"""The orchestrator. FastAPI + WebSockets on 127.0.0.1 only.

Two socket endpoints:
  /ws/audio    the audio service connects here (transcripts in, speech out)
  /ws/display  the kiosk browser connects here (state and text out)

Both are pushed the same state changes, so the screen and the speaker never
disagree about what the device is doing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .context import ContextBuilder
from .db import Store
from .provider import Reply, build_provider
from .router import Router

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("assistant.core")

# Break the stream on sentence boundaries so the TTS engine can start
# speaking while the model is still writing. This is most of the perceived
# latency win — without it you wait for the whole reply before hearing a word.
SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=[.!?]["”])\s+')


def load_config() -> dict:
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


class Hub:
    """Fan-out to whatever is currently connected. Nothing is required to be."""

    def __init__(self) -> None:
        self.audio: set[WebSocket] = set()
        self.display: set[WebSocket] = set()

    async def _send(self, targets: set[WebSocket], payload: dict) -> None:
        dead = []
        for ws in list(targets):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            targets.discard(ws)

    async def to_audio(self, payload: dict) -> None:
        await self._send(self.audio, payload)

    async def to_display(self, payload: dict) -> None:
        await self._send(self.display, payload)

    async def broadcast_state(self, state: str, detail: str = "") -> None:
        payload = {"type": "state", "state": state, "detail": detail}
        await self._send(self.display, payload)
        await self._send(self.audio, payload)


cfg = load_config()
logging.basicConfig(
    level=getattr(logging, cfg["logging"]["level"].upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(title="Home AI Appliance")
hub = Hub()
store = Store(ROOT / cfg["storage"]["db_path"])
router = Router(cfg)
ctx = ContextBuilder(cfg, store)
_provider = None          # built lazily so the app starts without a key
_busy = asyncio.Lock()


def provider():
    global _provider
    if _provider is None:
        _provider = build_provider(cfg)
    return _provider


# ---------------------------------------------------------------- routes

@app.get("/")
async def index():
    return FileResponse(ROOT / "ui" / "index.html")


@app.get("/healthz")
async def healthz():
    day_ago = time.time() - 86400
    return JSONResponse({
        "ok": True,
        "audio_connected": len(hub.audio) > 0,
        "display_connected": len(hub.display) > 0,
        "spend_24h_usd": round(store.spend_since(day_ago), 4),
        "identity": cfg["identity"]["name"],
    })


@app.get("/stats/wake")
async def wake_stats(hours: int = 24):
    """Wake-word telemetry. Read this daily for the first week; it is the
    only honest way to tune the detection threshold."""
    s = store.wake_stats(time.time() - hours * 3600)
    scores = sorted(s.pop("scores"))
    s["window_hours"] = hours
    if scores:
        s["score_min"] = round(scores[0], 3)
        s["score_median"] = round(scores[len(scores) // 2], 3)
        s["score_max"] = round(scores[-1], 3)
    return JSONResponse(s)


@app.websocket("/ws/display")
async def ws_display(ws: WebSocket):
    await ws.accept()
    hub.display.add(ws)
    await ws.send_text(json.dumps({
        "type": "hello",
        "name": cfg["identity"]["name"],
        "user": cfg["identity"]["user"],
    }))
    await ws.send_text(json.dumps({"type": "state", "state": "idle"}))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            # The kiosk page can inject typed input — useful before the
            # voice pipeline exists, and a permanent fallback afterwards.
            if msg.get("type") == "text_input":
                asyncio.create_task(handle_utterance(msg.get("text", "")))
    except WebSocketDisconnect:
        pass
    finally:
        hub.display.discard(ws)


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    hub.audio.add(ws)
    log.info("audio service connected")
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")

            if kind == "transcript":
                text = (msg.get("text") or "").strip()
                # A negative score means this came from the follow-up window,
                # not a wake-word trigger. Logging it would pollute the very
                # statistics used to tune the wake threshold — an unanswered
                # follow-up would read as a false positive.
                wake_score = float(msg.get("wake_score", 0.0))
                if cfg["logging"]["log_wake_events"] and wake_score >= 0:
                    store.add_wake_event(
                        score=wake_score,
                        accepted=bool(text),
                        transcript=text or None,
                    )
                if not text:
                    # Wake word fired but nothing intelligible followed.
                    # Almost always a false positive; say nothing.
                    log.info("empty transcript after wake — ignoring")
                    await hub.broadcast_state("idle")
                    continue
                asyncio.create_task(handle_utterance(text))

            elif kind == "state":
                await hub.to_display(msg)

            elif kind == "barge_in":
                log.info("barge-in")
                await hub.to_display({"type": "barge_in"})

    except WebSocketDisconnect:
        pass
    finally:
        hub.audio.discard(ws)
        log.info("audio service disconnected")


# ------------------------------------------------------------- pipeline

async def handle_utterance(text: str) -> None:
    if not text.strip():
        return

    if _busy.locked():
        # A second utterance while one is in flight. Interrupt rather than
        # queue — a queued answer to a question you have moved on from is
        # worse than no answer.
        await hub.to_audio({"type": "stop_speaking"})

    async with _busy:
        await hub.to_display({"type": "transcript", "text": text})
        route = router.route(text)
        log.info("routed to %s (%s): %s", route.tier, route.reason, text[:60])
        await hub.broadcast_state("thinking", route.tier)

        store.add_message("user", text)
        messages = ctx.build(text)
        reply = Reply(tier=route.tier)

        await hub.broadcast_state("speaking")
        buffer = ""
        spoken_any = False

        try:
            stream = provider().stream_reply(
                system=ctx.system,
                messages=messages,
                model=route.model,
                max_tokens=cfg["provider"]["max_tokens"],
                reply=reply,
            )
            async for chunk in stream:
                await hub.to_display({"type": "response_delta", "text": chunk})
                buffer += chunk
                # Emit complete sentences as they form.
                parts = SENTENCE_END.split(buffer)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            await hub.to_audio({
                                "type": "speak", "text": sentence.strip()
                            })
                            spoken_any = True
                    buffer = parts[-1]

            if buffer.strip():
                await hub.to_audio({"type": "speak", "text": buffer.strip()})
                spoken_any = True

        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline failure")
            reply.error = str(exc)
            if not spoken_any:
                await hub.to_audio({
                    "type": "speak",
                    "text": "Something went wrong on my end.",
                })

        await hub.to_audio({"type": "speak_done"})
        await hub.to_display({"type": "response_done", "error": reply.error})

        if reply.text:
            store.add_message("assistant", reply.text,
                              tier=route.tier, model=reply.model)
        if reply.usage:
            store.add_usage(reply.model, route.tier, reply.usage, reply.cost_usd)
            log.info(
                "usage in=%d cached=%d out=%d cost=$%.5f",
                reply.usage.get("input_tokens", 0),
                reply.usage.get("cache_read_input_tokens", 0),
                reply.usage.get("output_tokens", 0),
                reply.cost_usd,
            )

        await hub.broadcast_state("idle")
