"""The adapter layer — the only file that knows which AI platform you use.

Everything else in the codebase calls `stream_reply()` and gets back an async
generator of text chunks plus a usage record. Switching providers, or falling
back when one has an outage, happens here and nowhere else.

Only Anthropic is implemented in this scaffold. The shape of `Provider` is
what an OpenAI or Google implementation would need to satisfy.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

# Published prices, USD per million tokens, as of August 2026.
# Verify against the provider's pricing page before trusting the numbers
# your usage dashboard reports — these move.
PRICING = {
    "claude-haiku-4-5":  {"in": 1.00, "out": 5.00},
    "claude-sonnet-5":   {"in": 2.00, "out": 10.00},
    "claude-opus-5":     {"in": 5.00, "out": 25.00},
}
# Anthropic cache economics: writing costs 1.25x the input rate, reading
# costs 0.1x. That read rate is the whole reason caching is worth doing.
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

log = logging.getLogger("assistant.provider")

# Server-side web search is billed per search, on top of the tokens the
# results consume. $10 per 1,000 searches. That is a cent a look-up, which is
# roughly thirty times the cost of a plain question — worth knowing, not worth
# worrying about, but it is why the router does not hand this to every request.
WEB_SEARCH_COST_PER_USE = 0.01


def explain_failure(exc: Exception) -> str:
    """One short spoken sentence naming what actually failed.

    Every failure used to say "Sorry, I could not reach the network just
    then." — a rejected API key, a rate limit, an over-length request, a
    malformed tool schema and a genuine outage all got the same words. That
    sentence has one property worth noting: it is the single explanation
    guaranteed to send you to the router. A wrong key cost an hour of exactly
    that.

    These are deliberately phrased as a person would say them, because they
    are read out loud, and each one names the place to look.
    """
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    status = getattr(exc, "status_code", None)

    # Auth first: it is the one people misdiagnose, and it never fixes itself.
    if status in (401, 403) or "authentication" in low \
            or "invalid x-api-key" in low or "invalid api key" in low:
        return ("My API key is being rejected. It needs replacing in the "
                "environment file, not on your network.")
    if status == 429 or "rate limit" in low or "rate_limit" in low:
        return "I am being rate limited. Give me a minute and ask again."
    if status == 402 or "credit" in low or "quota" in low or "billing" in low:
        return "The account is out of credit, so I cannot answer that."
    if status in (529,) or "overloaded" in low:
        return "The model is overloaded right now. Try again shortly."
    if status == 400 or "invalid_request" in low or "too long" in low \
            or "maximum" in low and "token" in low:
        return ("I sent a request the service would not accept. That is a "
                "bug on my side, and the details are in the log.")
    if status and 500 <= int(status) < 600:
        return "The service returned an error. It is not you, and not here."
    # Genuinely the network: connection, DNS, TLS, timeout.
    if name in ("APIConnectionError", "APITimeoutError", "ConnectionError",
                "TimeoutError", "ConnectTimeout", "ReadTimeout") \
            or any(w in low for w in ("connection", "timed out", "timeout",
                                      "dns", "name resolution", "ssl",
                                      "unreachable")):
        return "Sorry, I could not reach the network just then."
    # Anything left is a bug in this appliance, and should say so rather than
    # blaming infrastructure that is working fine.
    return ("Something went wrong inside me answering that. The details are "
            "in the log.")

# The server tool identifier. This is versioned by date and WILL need updating
# when Anthropic ships a newer one; an out-of-date string fails loudly rather
# than silently, which is the good kind of breakage.
WEB_SEARCH_TOOL_TYPE = "web_search_20260318"


def web_search_tool(cfg: dict) -> dict:
    """Build the web-search tool definition from config.

    user_location is not tracking — it is a hint the search engine uses to
    resolve "near me", "tonight", and local business hours. Without it,
    "is the hardware store still open" is unanswerable no matter how good
    the search is.
    """
    t = cfg.get("tools", {}).get("web_search", {})
    tool = {
        "type": WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        "max_uses": int(t.get("max_uses", 3)),
    }
    loc = cfg.get("location") or {}
    if loc.get("city") or loc.get("timezone"):
        tool["user_location"] = {
            "type": "approximate",
            **{k: v for k, v in loc.items()
               if k in ("city", "region", "country", "timezone") and v},
        }
    if t.get("blocked_domains"):
        tool["blocked_domains"] = list(t["blocked_domains"])
    elif t.get("allowed_domains"):
        # The API accepts one or the other, never both.
        tool["allowed_domains"] = list(t["allowed_domains"])
    return tool


@dataclass
class Reply:
    """Accumulated result of one streamed exchange."""
    text: str = ""
    model: str = ""
    tier: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None
    # Pages the model actually cited, for the screen. Never spoken — reading
    # a URL aloud is noise, and the whole point of the display is to carry
    # what the voice cannot.
    sources: list = field(default_factory=list)
    # Names of local tools that ran, for the log and the display.
    tool_calls: list = field(default_factory=list)
    # The last completed message object, so the tool loop can inspect
    # stop_reason and content without re-requesting anything.
    _last_final: object | None = None


# Environment variables that must never be handed to a child process.
# core/media.py launches yt-dlp and mpv; core/desktop.py launches wmctrl and
# xdotool. yt-dlp in particular is network-facing and runs per-site extractor
# code, and there is no reason any of them should be able to read the key.
SECRET_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
              "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def child_env(base: dict | None = None) -> dict:
    """A copy of the environment with every secret removed.

    Belt and braces on top of LoadCredential: with the credential in place the
    key is not in the environment to begin with, but this file should not have
    to assume the unit is configured correctly to be safe.
    """
    env = dict(base if base is not None else os.environ)
    for name in SECRET_ENV:
        env.pop(name, None)
    return env


def read_api_key() -> tuple[str | None, str]:
    """The API key and where it came from.

    Three places, in order of how much they deserve to be trusted:

      1. systemd's credential directory. The unit declares
         `LoadCredential=anthropic-key:...`; systemd mounts a 0700 tmpfs,
         writes the file 0400, and unmounts it when the service stops. The key
         is never in the process environment, so it is never in
         /proc/PID/environ and never inherited by mpv, yt-dlp or xdotool.
      2. The environment, for a manual run or an EnvironmentFile= unit.
      3. .env in the appliance directory, for `python -m core.app` from a
         terminal, where nothing has loaded it. python-dotenv is a declared
         dependency that nothing ever imported, so this was simply broken:
         the README told you to put the key in .env and then run uvicorn by
         hand, and that combination has never worked.
    """
    d = os.environ.get("CREDENTIALS_DIRECTORY")
    if d:
        path = Path(d) / "anthropic-key"
        try:
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key, "the systemd credential"
        except OSError as exc:
            log.warning("credential directory is set but %s is unreadable "
                        "(%s)", path, exc)

    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key:
        return key, "the environment"

    env_file = Path(__file__).resolve().parent.parent / ".env"
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    return key, str(env_file)
    except OSError:
        pass
    return None, "nowhere"


def _collect_sources(final) -> list[dict]:
    """Pull deduplicated {title, url} out of a finished message's citations."""
    seen, out = set(), []
    for block in getattr(final, "content", []) or []:
        for cit in getattr(block, "citations", None) or []:
            url = getattr(cit, "url", None)
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"url": url, "title": getattr(cit, "title", "") or url})
    return out


class Provider(Protocol):
    async def stream_reply(
        self, system: str, messages: list[dict], model: str,
        max_tokens: int, reply: Reply, tools: list | None = None,
        executor=None, max_rounds: int = 4,
    ) -> AsyncIterator[str]:
        ...


def estimate_cost(model: str, usage: dict) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    plain_in = usage.get("input_tokens", 0)
    cache_w = usage.get("cache_creation_input_tokens", 0)
    cache_r = usage.get("cache_read_input_tokens", 0)
    out = usage.get("output_tokens", 0)
    searches = usage.get("web_search_requests", 0)
    return (
        plain_in / 1e6 * p["in"]
        + cache_w / 1e6 * p["in"] * CACHE_WRITE_MULT
        + cache_r / 1e6 * p["in"] * CACHE_READ_MULT
        + out / 1e6 * p["out"]
        + searches * WEB_SEARCH_COST_PER_USE
    )


class AnthropicProvider:
    def __init__(self, prompt_caching: bool = True):
        # Import lazily so the module can be imported (and unit-tested)
        # on a machine with no SDK and no key.
        from anthropic import AsyncAnthropic

        key, source = read_api_key()
        if not key:
            raise RuntimeError(
                "No API key. On the appliance it arrives through systemd's "
                "LoadCredential, as a file the service can read and nothing "
                "else can:\n"
                "    scripts/setup_credential.sh\n"
                "For a manual run:  export ANTHROPIC_API_KEY=...  or\n"
                "    set -a; . .env; set +a"
            )
        log.info("api key loaded from %s", source)
        self.client = AsyncAnthropic(api_key=key)
        self.prompt_caching = prompt_caching

    async def stream_reply(
        self, system: str, messages: list[dict], model: str,
        max_tokens: int, reply: Reply, tools: list | None = None,
        executor=None, max_rounds: int = 4,
    ) -> AsyncIterator[str]:
        """Stream a reply, running local tools as the model asks for them.

        `executor`, if given, must expose `async run(name, args) -> str`. It
        handles CLIENT-side tools — ones that do something on this machine.
        Server-side tools like web search need none of this; Anthropic runs
        those and the results simply appear in the response.

        The loop is: stream a turn; if it ended because the model wants a
        tool, run the tool, append both the model's turn and the results to
        the conversation, and stream again. `max_rounds` bounds it, because
        a model that saves a note, reads it back, decides to fix it and
        repeats is a model spending your money in a circle.
        """
        # The system block is identical on every request, which makes it
        # the ideal cache target. cache_control marks the boundary; the
        # provider caches everything up to and including it.
        if self.prompt_caching:
            system_param = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_param = system

        reply.model = model
        convo = list(messages)          # never mutate the caller's list
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            for round_no in range(max_rounds):
                final = None
                async for piece in self._stream_once(kwargs, convo, reply):
                    yield piece
                final = reply._last_final

                if not final or final.stop_reason != "tool_use" or not executor:
                    break

                # Run every tool the model asked for, then hand the results
                # back as the next user turn. This is the documented shape:
                # the assistant's ENTIRE content block list goes back
                # verbatim, including any server-tool blocks it contains.
                results = []
                for block in final.content:
                    if getattr(block, "type", "") != "tool_use":
                        continue
                    log.info("tool: %s(%s)", block.name,
                             ", ".join(f"{k}=…" for k in (block.input or {})))
                    out = await executor.run(block.name, block.input or {})
                    reply.tool_calls.append(block.name)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": out,
                    })
                if not results:
                    break
                convo.append({"role": "assistant", "content": final.content})
                convo.append({"role": "user", "content": results})
            else:
                log.warning("hit the %d-round tool limit", max_rounds)

            reply.cost_usd = estimate_cost(model, reply.usage)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            reply.error = f"{type(exc).__name__}: {exc}"
            # Say what actually went wrong, and put the real exception in the
            # journal. Degrade, never die — but degrade honestly.
            log.exception("reply failed: %s", reply.error)
            yield explain_failure(exc)

    async def _stream_once(self, kwargs: dict, convo: list,
                           reply: Reply) -> AsyncIterator[str]:
        """One streamed turn. Accumulates usage rather than replacing it, so
        a multi-round tool conversation reports what it actually cost."""
        try:
            async with self.client.messages.stream(messages=convo,
                                                   **kwargs) as stream:
                # text_stream yields only the assistant's prose. Search
                # queries, results and citation structures arrive as separate
                # content blocks and never leak into what gets spoken — which
                # is exactly what a voice assistant needs. The audible effect
                # of a search is a short pause, usually after the model has
                # already said something like "let me look that up".
                # Iterate EVENTS, not text_stream, so we can see where one
                # content block ends and the next begins.
                #
                # Why that matters: the model says "Let me check." and then
                # immediately issues a tool call. That closes the text block
                # at the period, with no trailing whitespace — and the
                # orchestrator's sentence splitter looks for punctuation
                # FOLLOWED BY whitespace. So the preamble sat in the buffer
                # unspoken until the whole reply finished, measured at twelve
                # seconds after it was generated. Emitting a newline at each
                # block boundary makes that sentence flush the moment it is
                # complete, which is the entire point of streaming.
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "text":
                        chunk = event.text
                        reply.text += chunk
                        yield chunk
                    elif etype == "content_block_stop" and reply.text:
                        if not reply.text.endswith(("\n", " ")):
                            yield "\n"

                final = await stream.get_final_message()

                # Defensive: if this SDK version did not emit the "text"
                # helper events above, fall back to the assembled message
                # rather than saying nothing at all.
                if not reply.text:
                    for block in getattr(final, "content", []) or []:
                        if getattr(block, "type", "") == "text":
                            reply.text += block.text
                    if reply.text:
                        yield reply.text
                u = final.usage
                stu = getattr(u, "server_tool_use", None)
                add = {
                    "input_tokens": getattr(u, "input_tokens", 0) or 0,
                    "output_tokens": getattr(u, "output_tokens", 0) or 0,
                    "cache_creation_input_tokens":
                        getattr(u, "cache_creation_input_tokens", 0) or 0,
                    "cache_read_input_tokens":
                        getattr(u, "cache_read_input_tokens", 0) or 0,
                    "web_search_requests":
                        (getattr(stu, "web_search_requests", 0) or 0)
                        if stu is not None else 0,
                }
                # ACCUMULATE. Each tool round is a separate billed request,
                # and overwriting here would report only the last one — which
                # would quietly understate the cost of every note you save.
                for k, v in add.items():
                    reply.usage[k] = reply.usage.get(k, 0) + v
                reply.sources.extend(
                    s for s in _collect_sources(final)
                    if s["url"] not in {x["url"] for x in reply.sources}
                )
                reply._last_final = final
        finally:
            # Deliberately empty, and deliberately left. Removing it means
            # dedenting the sixty-line block above, which is a real chance of
            # a real bug in exchange for two tidy lines. Not worth it.
            pass


def build_provider(cfg: dict) -> Provider:
    name = cfg["provider"]["name"]
    if name == "anthropic":
        return AnthropicProvider(
            prompt_caching=cfg["provider"].get("prompt_caching", True)
        )
    raise ValueError(
        f"Provider '{name}' is not implemented. Add a class satisfying the "
        f"Provider protocol above and wire it in here — nothing else in the "
        f"codebase needs to change."
    )
