"""Thin OpenRouter client wrapper.

Uses the OpenAI SDK pointed at OpenRouter's base URL. We expose two helpers:

  - `client()`  — a configured OpenAI() instance
  - `chat()`    — a small wrapper for one-shot chat completions with optional
                  JSON-object response_format and a sensible retry policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI, APIStatusError, RateLimitError, APIConnectionError

from .config import env

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/local/no-op-circuit",
    "X-Title": "no-op-circuit",
}


def client(api_key: str | None = None) -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key or env("OPENROUTER_API_KEY"),
    )


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None
    raw: Any  # the full SDK response object, for accounting / debug
    resolved_model: str | None = None  # the snapshot the alias actually routed to
    response_id: str | None = None     # provider-assigned id (for audit trails)


def chat(
    messages: Iterable[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    response_format: dict | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 4.0,
    cli: OpenAI | None = None,
) -> ChatResult:
    cli = cli or client()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=DEFAULT_HEADERS,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = cli.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            return ChatResult(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason,
                raw=resp,
                resolved_model=getattr(resp, "model", None),
                response_id=getattr(resp, "id", None),
            )
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            last_err = exc
            # Bail immediately on hard 4xx errors that aren't rate-limited.
            status = getattr(exc, "status_code", None)
            if isinstance(exc, APIStatusError) and status is not None and 400 <= status < 500 and status not in (408, 429):
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_seconds * (2**attempt))
    assert last_err is not None
    raise last_err


def parse_json(text: str) -> dict:
    """Parse a JSON-object response, tolerating ```json fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove first fence line and trailing ```
        first_nl = stripped.find("\n")
        if first_nl >= 0:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: -3]
    return json.loads(stripped)
