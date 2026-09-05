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
from .provider import Reply, build_provider, web_search_tool
from .router import Router
from .scheduler import Scheduler
from .tools import build_tools

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
# Local tools run on this machine. The toolbox needs the store for memory;
# the context builder needs the toolbox to inject remembered facts.
local_tools = build_tools(cfg, store)
ctx = ContextBuilder(cfg, store, local_tools)
scheduler = Scheduler(store, hub, cfg)
_provider = None          # built lazily so the app starts without a key
_busy = asyncio.Lock()
_filler_n = 0             # rotates the "one moment" phrases

# Set when the user dismisses the assistant mid-reply. The audio service has
# already stopped the speaker by the time this arrives — this is what stops
# us generating (and paying for) the rest of an answer nobody is listening to.
_cancel = asyncio.Event()


def provider():
    global _provider
    if _provider is None:
        _provider = build_provider(cfg)
    return _provider


# ---------------------------------------------------------------- routes

async def _backfill_loop():
    """Embed history in the background, slowly and out of the way.

    A first run against months of conversation is thousands of embeddings. Done
    at startup it would leave the assistant deaf for a minute after every
    update, which is a bad trade for a feature nobody is using in that minute.

    So: small batches, a sleep between them, and a long sleep once there is
    nothing left. `embeddings_missing` returns newest first, so even a run that
    never finishes has indexed the half you are most likely to ask about.
    """
    sem = (cfg.get("memory", {}) or {}).get("semantic", {}) or {}
    if not sem.get("enabled", False):
        return
    batch = int(sem.get("backfill_batch", 32))
    gap = float(sem.get("backfill_gap_seconds", 2.0))
    idle = float(sem.get("backfill_idle_seconds", 300.0))

    # Vectors from a different model are not comparable to these ones, so a
    # model change means a re-index rather than a silently mixed space.
    dropped = store.embeddings_drop_other_models(local_tools.embedder.model_name)
    if dropped:
        log.info("embedding model changed — dropped %d stale vectors", dropped)

    await asyncio.sleep(10)   # let the audio service settle first
    while True:
        try:
            done = sum(local_tools.index.backfill(k, batch)
                       for k in ("memory", "message"))
        except Exception:  # noqa: BLE001
            log.exception("backfill failed — continuing")
            done = 0
        if done:
            log.info("indexed %d", done)
        await asyncio.sleep(gap if done else idle)


@app.on_event("startup")
async def _startup():
    # Timers must survive a restart, so the scheduler reloads pending items
    # from SQLite rather than holding them in memory. Starting it here means
    # a reminder set before an update still fires after it.
    scheduler.start()
    asyncio.create_task(_backfill_loop())


@app.get("/")
async def index():
    return FileResponse(ROOT / "ui" / "index.html")


@app.get("/dashboard")
async def dashboard():
    """The kiosk page in board mode. Same file, so there is only ever one
    stylesheet and one renderer to keep correct."""
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


@app.get("/reminders")
async def reminders():
    """Pending timers and reminders, for a glance from the terminal."""
    from .scheduler import describe_when
    rows = store.pending_reminders()
    return JSONResponse({
        "count": len(rows),
        "pending": [
            {"id": r["id"], "kind": r["kind"], "text": r["text"],
             "when": describe_when(r["due"]),
             "due": time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(r["due"]))}
            for r in rows
        ],
    })


@app.get("/board")
async def board():
    """Everything the dashboard shows, in one request.

    One endpoint rather than four because the screen refreshes on a timer:
    four requests every fifteen seconds is four chances for a partial
    render, and a dashboard that shows lists from now and reminders from a
    minute ago is worse than one that is briefly stale in a consistent way.
    """
    from .scheduler import describe_when
    known = (cfg.get("lists", {}) or {}).get("known", {}) or {}
    # Every configured board, even empty ones — an empty board on screen is
    # information ("nothing on army"), whereas a missing card just looks like
    # something broke. Ad-hoc lists appear only when they have something on
    # them.
    names = list(known) + [n for n, _ in store.list_counts()
                           if n not in known]
    midnight = time.time() - (time.time() % 86400)
    boards = []
    for n in names:
        rows = store.list_read(n, include_done=True)
        if not rows and n not in known:
            continue
        boards.append({
            "key": n,
            "title": (known.get(n, {}) or {}).get("say", f"{n} list"),
            "items": [r["text"] for r in rows if not r["done"]],
            "done": [r["text"] for r in rows if r["done"]],
        })
    return JSONResponse({
        "name": cfg["identity"]["name"],
        "open_total": sum(len(b["items"]) for b in boards),
        "done_today": store.done_since(midnight),
        "lists": boards,
        "reminders": [
            {"kind": r["kind"], "text": r["text"],
             "when": describe_when(r["due"])}
            for r in store.pending_reminders()
        ],
        "spend_24h_usd": round(store.spend_since(time.time() - 86400), 4),
        "audio_ok": len(hub.audio) > 0,
    })


@app.get("/case")
async def case():
    """The active investigation, shaped for the screen.

    `sla_due` goes out as an absolute epoch rather than "two hours left", so
    the browser can count down every second against a number that does not go
    stale between the fifteen-second polls. A deadline that only updates four
    times a minute reads as broken on a screen you are watching.
    """
    row = store.case_active()
    if not row:
        return JSONResponse({"active": False})

    sides: dict[str, list] = {"investigation": [], "admin": []}
    for s in store.case_steps(row["id"]):
        sides.setdefault(s["side"], []).append({
            "phase": s["phase"], "text": s["text"],
            "done": bool(s["done"]), "finding": s["finding"],
        })
    prog = store.case_progress(row["id"])
    return JSONResponse({
        "active": True,
        "id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "ref": row["ref"],
        "severity": row["severity"],
        "opened": row["opened"],
        "sla_due": row["sla_due"],
        "now": time.time(),          # lets the page correct for clock skew
        "done_total": sum(d for d, _ in prog.values()),
        "step_total": sum(t for _, t in prog.values()),
        "sides": sides,
        "notes": row["notes"],
        "open_cases": len(store.cases_open()),
    })


@app.get("/cases")
async def cases():
    """Open cases, for a glance from the terminal."""
    out = []
    for r in store.cases_open():
        prog = store.case_progress(r["id"])
        out.append({
            "id": r["id"], "kind": r["kind"], "title": r["title"],
            "ref": r["ref"], "severity": r["severity"],
            "done": sum(d for d, _ in prog.values()),
            "total": sum(t for _, t in prog.values()),
            "sla_due": (time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(r["sla_due"]))
                        if r["sla_due"] else None),
            "sla_seconds_left": (round(r["sla_due"] - time.time())
                                 if r["sla_due"] else None),
        })
    return JSONResponse({"count": len(out), "cases": out})


@app.get("/lists")
async def all_lists():
    """Every list with something on it. A glance at the whole household."""
    counts = store.list_counts()
    return JSONResponse({
        "lists": [
            {"list": n, "count": c, "items": [r["text"] for r in store.list_read(n)]}
            for n, c in counts
        ],
    })


@app.get("/lists/{name}")
async def show_list(name: str = "shopping"):
    rows = store.list_read(name)
    return JSONResponse({"list": name, "count": len(rows),
                         "items": [r["text"] for r in rows]})


@app.get("/memory/index")
async def memory_index():
    """How much of the history is searchable by meaning yet."""
    try:
        return JSONResponse(local_tools.index.stats())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                            status_code=500)


@app.get("/memories")
async def memories():
    """Everything the assistant currently remembers, so you can audit it.

    Worth reading occasionally. Memory that nobody inspects is memory that
    quietly accumulates something wrong.
    """
    rows = store.list_memories()
    return JSONResponse({
        "count": len(rows),
        "memories": [
            {"id": r["id"], "category": r["category"], "text": r["text"],
             "updated": time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(r["updated"]))}
            for r in rows
        ],
    })


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
                mode = msg.get("mode")
                asyncio.create_task(handle_utterance(text, mode=mode))

            elif kind == "state":
                await hub.to_display(msg)

            elif kind == "cancel":
                log.info("dismissed — abandoning any reply in flight")
                _cancel.set()
                await hub.to_display({"type": "cancelled"})

            elif kind == "barge_in":
                log.info("barge-in")
                await hub.to_display({"type": "barge_in"})

    except WebSocketDisconnect:
        pass
    finally:
        hub.audio.discard(ws)
        log.info("audio service disconnected")


# ------------------------------------------------------------- pipeline

async def handle_utterance(text: str, mode: str | None = None) -> None:
    if not text.strip():
        return

    # Dictation arrives already transcribed with patient endpointing. The
    # model needs to know it is transcribing rather than conversing, or it
    # will helpfully summarise what you just spent a minute dictating.
    if mode == "dictation":
        text = (
            "I just dictated the following. Save it as a note using my own "
            "words — tidy up obvious speech errors and add sensible "
            "paragraphs, but do not summarise or rewrite it. Then confirm in "
            "one short sentence.\n\n" + text
        )

    if _busy.locked():
        # A second utterance while one is in flight. Interrupt rather than
        # queue — a queued answer to a question you have moved on from is
        # worse than no answer.
        await hub.to_audio({"type": "stop_speaking"})

    async with _busy:
        _cancel.clear()
        await hub.to_display({"type": "transcript", "text": text})
        route = router.route(text)
        log.info("routed to %s (%s): %s", route.tier, route.reason, text[:60])
        await hub.broadcast_state("thinking", route.tier)

        store.add_message("user", text)
        messages = ctx.build(text)
        reply = Reply(tier=route.tier)

        # Attach the web-search tool only where it is both supported and
        # wanted. Two reasons this is not simply always on: each search costs
        # about a cent, roughly thirty times a plain question; and the small
        # tier may not support server tools at all, which would turn "what
        # time is it" into an API error.
        tools = []
        ws = cfg.get("tools", {}).get("web_search", {})
        if ws.get("enabled") and route.tier in ws.get("tiers", ["mid", "top"]):
            tools.append(web_search_tool(cfg))
        # Local tools go on every tier. They cost nothing per call and
        # "write that down" is exactly the kind of short request the small
        # tier handles.
        tools.extend(local_tools.schemas())
        tools = tools or None

        await hub.broadcast_state("speaking")
        buffer = ""
        spoken_any = False

        # ---- filling the gap -------------------------------------------
        # A web search adds several seconds during which the model produces
        # nothing at all, and silence from a voice assistant is
        # indistinguishable from failure — you repeat yourself, which makes
        # it worse. If nothing has been spoken by the time the filler
        # threshold passes, say something short so the device is audibly
        # alive. The model's own reply then follows normally.
        t0 = time.monotonic()
        timings: dict = {}
        filler_cfg = cfg.get("behaviour", {})
        filler_after = float(filler_cfg.get("filler_after_seconds", 1.2))
        fillers = filler_cfg.get("filler_phrases") or ["One moment."]

        async def maybe_filler() -> None:
            try:
                await asyncio.sleep(filler_after)
            except asyncio.CancelledError:
                return
            if spoken_any or _cancel.is_set():
                return
            # Rotate deterministically. Hearing the same three words every
            # time is worse than the pause it covers.
            global _filler_n
            phrase = fillers[_filler_n % len(fillers)]
            _filler_n += 1
            timings["filler"] = time.monotonic() - t0
            await hub.to_audio({"type": "speak", "text": phrase})

        filler_task = (asyncio.create_task(maybe_filler())
                       if tools and filler_after > 0 else None)

        try:
            stream = provider().stream_reply(
                system=ctx.system,
                messages=messages,
                model=route.model,
                max_tokens=cfg["provider"]["max_tokens"],
                reply=reply,
                tools=tools,
                executor=local_tools,
            )
            async for chunk in stream:
                if _cancel.is_set():
                    log.info("cancelled mid-stream after %.1fs",
                             time.monotonic() - t0)
                    buffer = ""       # do not speak the tail
                    break
                timings.setdefault("first_token", time.monotonic() - t0)
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
                            timings.setdefault("first_speech",
                                               time.monotonic() - t0)
                            spoken_any = True
                    buffer = parts[-1]

            if buffer.strip():
                await hub.to_audio({"type": "speak", "text": buffer.strip()})
                timings.setdefault("first_speech", time.monotonic() - t0)
                spoken_any = True

        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline failure")
            reply.error = str(exc)
            if not spoken_any:
                await hub.to_audio({
                    "type": "speak",
                    "text": "Something went wrong on my end.",
                })

        if filler_task:
            filler_task.cancel()

        # Where the time actually went. Read this before tuning anything —
        # "it feels slow" is not a measurement, and the fix for a slow search
        # is nothing like the fix for slow generation.
        log.info(
            "timing first_token=%.2fs first_speech=%.2fs total=%.2fs%s",
            timings.get("first_token", -1),
            timings.get("first_speech", -1),
            time.monotonic() - t0,
            f" filler_at={timings['filler']:.2f}s" if "filler" in timings else "",
        )

        await hub.to_audio({"type": "speak_done"})
        # Sources go to the screen only. They are what the display is for:
        # carrying the part of an answer that does not survive being read out.
        await hub.to_display({"type": "response_done", "error": reply.error,
                              "sources": reply.sources,
                              "tools": reply.tool_calls})

        if reply.text:
            store.add_message("assistant", reply.text,
                              tier=route.tier, model=reply.model)
        if reply.usage:
            store.add_usage(reply.model, route.tier, reply.usage, reply.cost_usd)
            log.info(
                "usage in=%d cached=%d out=%d searches=%d cost=$%.5f",
                reply.usage.get("input_tokens", 0),
                reply.usage.get("cache_read_input_tokens", 0),
                reply.usage.get("output_tokens", 0),
                reply.usage.get("web_search_requests", 0),
                reply.cost_usd,
            )
        if reply.tool_calls:
            log.info("tools used: %s", ", ".join(reply.tool_calls))
            # Anything that touched a list, or an explicit request to show
            # them, puts the board on screen. Saying six items out loud is
            # tedious; showing them and saying "six things" is better.
            if any(c in ("show_board", "list_add", "list_remove",
                         "list_read", "list_clear", "list_all",
                         "set_reminder", "cancel_reminder", "list_reminders")
                   for c in reply.tool_calls):
                await hub.to_display({"type": "show_board"})
            # Anything that touched the case redraws the checklist. Ticking a
            # step off and not seeing it go grey is the kind of small silence
            # that makes you stop trusting the screen.
            if any(c in ("start_case", "check_step", "uncheck_step",
                         "case_status", "case_next", "case_note", "show_case")
                   for c in reply.tool_calls):
                await hub.to_display({"type": "show_case"})
            if "close_case" in reply.tool_calls:
                await hub.to_display({"type": "hide_case"})

        await hub.broadcast_state("idle")
