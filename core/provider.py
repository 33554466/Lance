"""The adapter layer — the only file that knows which AI platform you use.

Everything else in the codebase calls `stream_reply()` and gets back an async
generator of text chunks plus a usage record. Switching providers, or falling
back when one has an outage, happens here and nowhere else.

Only Anthropic is implemented in this scaffold. The shape of `Provider` is
what an OpenAI or Google implementation would need to satisfy.
"""
from __future__ import annotations

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


@dataclass
class Reply:
    """Accumulated result of one streamed exchange."""
    text: str = ""
    model: str = ""
    tier: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    error: str | None = None


class Provider(Protocol):
    async def stream_reply(
        self, system: str, messages: list[dict], model: str,
        max_tokens: int, reply: Reply,
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
    return (
        plain_in / 1e6 * p["in"]
        + cache_w / 1e6 * p["in"] * CACHE_WRITE_MULT
        + cache_r / 1e6 * p["in"] * CACHE_READ_MULT
        + out / 1e6 * p["out"]
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
        max_tokens: int, reply: Reply,
    ) -> AsyncIterator[str]:
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
        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_param,
                messages=messages,
            ) as stream:
                async for chunk in stream.text_stream:
                    reply.text += chunk
                    yield chunk
                final = await stream.get_final_message()
                u = final.usage
                reply.usage = {
                    "input_tokens": getattr(u, "input_tokens", 0) or 0,
                    "output_tokens": getattr(u, "output_tokens", 0) or 0,
                    "cache_creation_input_tokens":
                        getattr(u, "cache_creation_input_tokens", 0) or 0,
                    "cache_read_input_tokens":
                        getattr(u, "cache_read_input_tokens", 0) or 0,
                }
                reply.cost_usd = estimate_cost(model, reply.usage)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            reply.error = f"{type(exc).__name__}: {exc}"
            # Degrade, never die. The appliance says something useful and
            # the wake-word loop keeps running.
            yield "Sorry, I could not reach the network just then."


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
