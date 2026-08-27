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

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. On the appliance this arrives "
                "via systemd LoadCredential; for a manual run, export it or "
                "put it in .env with mode 0600."
            )
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
            # Degrade, never die. The appliance says something useful and
            # the wake-word loop keeps running.
            yield "Sorry, I could not reach the network just then."

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
