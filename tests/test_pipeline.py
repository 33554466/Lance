"""End-to-end smoke test with a fake provider.

Exercises the real orchestrator, the real router, the real SQLite store, the
real WebSocket protocol and the real sentence-chunking — everything except
the network call and the sound card. Run it before you trust a change:

    python -m tests.test_pipeline

Exit code 0 means the plumbing is sound and any remaining problem is
hardware or credentials, which is exactly the split you want when debugging.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import AsyncIterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "\033[1;32m✓\033[0m" if cond else "\033[1;31m✗\033[0m"
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


# ---------------------------------------------------------------- units

def test_router() -> None:
    print("\nRouter")
    import yaml
    from core.router import Router

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    r = Router(cfg)

    cases = [
        ("set a timer for ten minutes", "small"),
        ("what time is it", "small"),
        ("why does the moon look bigger near the horizon", "mid"),
        ("draft a note to the plumber about the leak", "mid"),
        ("think hard about how I should structure this", "top"),
        ("a" * 200, "mid"),
    ]
    for text, expected in cases:
        got = r.route(text)
        check(f"{expected:<5} <- {text[:44]!r}", got.tier == expected,
              f"got {got.tier} ({got.reason})")


def test_store() -> None:
    print("\nStore")
    from core.db import Store

    with tempfile.TemporaryDirectory() as tmp:
        s = Store(Path(tmp) / "t.db")
        s.add_message("user", "where did I put the insurance letter")
        s.add_message("assistant", "You scanned it on Tuesday.")
        s.add_message("user", "and the car registration")
        s.add_message("assistant", "That one is in the vault too.")

        turns = s.recent_turns(2)
        check("history returns 4 messages", len(turns) == 4, str(len(turns)))
        check("history starts on a user turn", turns[0]["role"] == "user",
              turns[0]["role"])

        hits = s.search("insurance")
        check("FTS5 finds 'insurance'", len(hits) >= 1, f"{len(hits)} hits")

        s.add_usage("claude-sonnet-5", "mid",
                    {"input_tokens": 2000, "output_tokens": 500,
                     "cache_read_input_tokens": 1200}, 0.0091)
        check("usage recorded", s.spend_since(0) > 0)

        s.add_wake_event(0.71, True, "hello")
        s.add_wake_event(0.58, False, None)
        st = s.wake_stats(0)
        check("wake stats count triggers", st["triggers"] == 2)
        check("wake stats spot empty transcripts",
              st["likely_false_positives"] == 1)
        s.close()


def test_cost_model() -> None:
    print("\nCost model")
    from core.provider import estimate_cost

    # 1M plain input on Sonnet 5 at $2/M
    c = estimate_cost("claude-sonnet-5", {"input_tokens": 1_000_000})
    check("1M input on sonnet = $2.00", abs(c - 2.00) < 1e-9, f"${c}")

    # Cache reads are a tenth of the input rate — the reason caching matters
    c = estimate_cost("claude-sonnet-5", {"cache_read_input_tokens": 1_000_000})
    check("1M cached input = $0.20", abs(c - 0.20) < 1e-9, f"${c}")

    c = estimate_cost("claude-sonnet-5", {"output_tokens": 1_000_000})
    check("1M output on sonnet = $10.00", abs(c - 10.00) < 1e-9, f"${c}")

    c = estimate_cost("made-up-model", {"input_tokens": 1_000_000})
    check("unknown model costs 0 rather than crashing", c == 0.0)


def test_sentence_split() -> None:
    print("\nSentence chunking (this is what makes speech start early)")
    from core.app import SENTENCE_END

    stream = ("The bins go out on Thursday. Recycling is fortnightly, "
              "so that one is next week. Want me to set a reminder?")
    parts = [p for p in SENTENCE_END.split(stream) if p.strip()]
    check("splits into 3 sentences", len(parts) == 3, f"{len(parts)}: {parts}")
    check("first sentence complete",
          parts[0] == "The bins go out on Thursday.", parts[0])


def test_context() -> None:
    print("\nContext builder")
    import yaml
    from core.context import ContextBuilder
    from core.db import Store

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        s = Store(Path(tmp) / "t.db")
        ctx = ContextBuilder(cfg, s)

        check("system prompt interpolates identity",
              cfg["identity"]["name"] in ctx.system)
        check("system prompt has no timestamp (cache stability)",
              "2026" not in ctx.system and ":" not in ctx.system.split("\n")[0])

        # Byte-identical across calls, or prompt caching never hits.
        a, b = ctx.system, ctx.system
        check("system prompt is stable across reads", a == b)

        msgs = ctx.build("what is the weather")
        check("last message is the user turn", msgs[-1]["role"] == "user")
        check("volatile time rides in the user turn, not the system prompt",
              msgs[-1]["content"].startswith("["))
        s.close()


# ------------------------------------------------------- integration

async def test_full_exchange() -> None:
    print("\nFull exchange through the real orchestrator (fake provider)")
    import uvicorn
    import websockets
    import httpx

    from core import app as appmod
    from core.provider import Reply

    CHUNKS = ["The bins go out ", "on Thursday. ",
              "Recycling is next week. ", "Want a reminder?"]

    class FakeProvider:
        async def stream_reply(self, system, messages, model, max_tokens,
                               reply: Reply) -> AsyncIterator[str]:
            reply.model = model
            for c in CHUNKS:
                reply.text += c
                yield c
                await asyncio.sleep(0.01)
            reply.usage = {"input_tokens": 1200, "output_tokens": 40,
                           "cache_read_input_tokens": 900}
            reply.cost_usd = 0.0031

    # Point the app at a throwaway DB and the fake provider.
    tmp = tempfile.TemporaryDirectory()
    from core.db import Store
    appmod.store = Store(Path(tmp.name) / "t.db")
    appmod.ctx.store = appmod.store
    appmod._provider = FakeProvider()

    config = uvicorn.Config(appmod.app, host="127.0.0.1", port=8799,
                            log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    check("orchestrator starts", server.started)

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://127.0.0.1:8799/healthz")
            check("/healthz responds 200", r.status_code == 200)
            check("/healthz reports no audio attached yet",
                  r.json()["audio_connected"] is False)

            r = await client.get("http://127.0.0.1:8799/")
            check("kiosk page served", r.status_code == 200
                  and "<html" in r.text.lower())

        audio_url = "ws://127.0.0.1:8799/ws/audio"
        disp_url = "ws://127.0.0.1:8799/ws/display"

        async with websockets.connect(audio_url) as audio, \
                   websockets.connect(disp_url) as display:

            hello = json.loads(await display.recv())
            check("display gets hello", hello["type"] == "hello")
            _ = await display.recv()  # initial idle state

            async with httpx.AsyncClient() as client:
                r = await client.get("http://127.0.0.1:8799/healthz")
                check("/healthz now sees the audio service",
                      r.json()["audio_connected"] is True)

            # Simulate the audio service delivering a transcript.
            await audio.send(json.dumps({
                "type": "transcript",
                "text": "when do the bins go out",
                "wake_score": 0.83,
            }))

            spoken, states, deltas, done = [], [], [], False
            async def collect(ws, sink):
                try:
                    while True:
                        sink.append(json.loads(await ws.recv()))
                except Exception:
                    pass

            audio_msgs, disp_msgs = [], []
            await asyncio.wait_for(asyncio.gather(
                asyncio.wait_for(collect(audio, audio_msgs), timeout=4),
                asyncio.wait_for(collect(display, disp_msgs), timeout=4),
                return_exceptions=True,
            ), timeout=6)

            spoken = [m["text"] for m in audio_msgs if m.get("type") == "speak"]
            states = [m["state"] for m in disp_msgs if m.get("type") == "state"]
            deltas = [m["text"] for m in disp_msgs
                      if m.get("type") == "response_delta"]
            done = any(m.get("type") == "response_done" for m in disp_msgs)

            check("audio service was told to speak", len(spoken) > 0,
                  f"{spoken}")
            check("speech was chunked into sentences, not one blob",
                  len(spoken) >= 3, f"{len(spoken)} chunks: {spoken}")
            check("first chunk is a complete sentence",
                  spoken and spoken[0].endswith("."), f"{spoken[:1]}")
            check("display received streaming deltas",
                  len(deltas) == len(CHUNKS), f"{len(deltas)}")
            check("state went thinking -> speaking -> idle",
                  "thinking" in states and "speaking" in states
                  and states[-1] == "idle", f"{states}")
            check("response_done sent", done)
            check("speak_done sent",
                  any(m.get("type") == "speak_done" for m in audio_msgs))

        # Persistence
        turns = appmod.store.recent_turns(5)
        check("exchange persisted (user + assistant)", len(turns) == 2,
              f"{len(turns)}")
        check("usage row written", appmod.store.spend_since(0) > 0)

        # Empty transcript = wake word fired but nothing was said.
        async with websockets.connect(audio_url) as audio:
            await audio.send(json.dumps({
                "type": "transcript", "text": "", "wake_score": 0.61
            }))
            await asyncio.sleep(0.4)
            msgs = appmod.store.recent_turns(10)
            check("empty transcript produces no exchange", len(msgs) == 2,
                  f"{len(msgs)}")
            stats = appmod.store.wake_stats(0)
            check("false positive still logged for tuning",
                  stats["likely_false_positives"] == 1, str(stats))

    finally:
        server.should_exit = True
        await task
        appmod.store.close()
        tmp.cleanup()


# ---------------------------------------------------------------- main

def main() -> int:
    print("\033[1mHome AI Appliance — pipeline smoke test\033[0m")
    test_router()
    test_store()
    test_cost_model()
    test_sentence_split()
    test_context()
    asyncio.run(test_full_exchange())

    print()
    if FAILURES:
        print(f"\033[1;31m{len(FAILURES)} check(s) failed:\033[0m")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\033[1;32mAll checks passed.\033[0m "
          "Plumbing is sound; anything left is hardware or credentials.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
