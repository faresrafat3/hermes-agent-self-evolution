"""OpenRouter API key discovery and rotation primitives (pure stdlib).

The Hermes profile ``.env`` (``~/.hermes/.env``) carries eleven OpenRouter
keys::

    OPENROUTER_API_KEY
    OPENROUTER_API_KEY_2
    ...
    OPENROUTER_API_KEY_11

``evolution.core.config._load_hermes_env`` already loads every ``KEY=VALUE``
line from that file into ``os.environ`` without overriding variables the
caller exported, so by the time any LM is built all eleven names are present
in the process environment (when the file exists).  This module is the
single place that knows the ordered variable names, how to collect them
without ever logging a secret, and how to detect a rate-limit / quota error
that should trigger rotation to the next key.

It deliberately does **not** import ``dspy`` — the DSPy wrapper lives in
:mod:`evolution.core.openrouter_rotation` — so ``config`` can import the
key helpers without pulling in the heavy DSPy stack.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from typing import Any

# Ordered exactly as requested: the un-suffixed key first, then _2 .. _11.
OPENROUTER_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    *(f"OPENROUTER_API_KEY_{i}" for i in range(2, 12)),
)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The evolution fleet, as wired in ``evolution.core.config``:
# primary is the fast, free ``stealth/crouton-alpha`` route; the fallback is
# the ``reflection/beam-1.0-501b-a23b`` reviewer.  On OpenRouter beam is a
# micro-cost route, not ``:free`` — see the long comment in ``config.py`` —
# and its provider-default token budget (131072) exceeds what any single
# key's credit balance can afford, which is why
# :mod:`evolution.core.openrouter_rotation` caps ``max_tokens`` when it
# switches to the fallback model.
DEFAULT_PRIMARY_MODEL = "openrouter/stealth/crouton-alpha"
DEFAULT_FALLBACK_MODEL = "openrouter/reflection/beam-1.0-501b-a23b"

# Status codes that mean "this key is exhausted / throttled, try the next
# one": 429 (rate limit), 402 (OpenRouter: insufficient credits / quota),
# 503 (service unavailable) and 529 (overloaded, Anthropic-style).
ROTATABLE_STATUS_CODES = frozenset({402, 429, 503, 529})

_ROTATABLE_TEXT_MARKERS = (
    "rate limit",
    "rate_limit",
    "rate-limit",
    "too many requests",
    "quota exceeded",
    "quota_exceeded",
    "insufficient credits",
    "insufficient credit",
    "payment required",
    "capacity",
    "overloaded",
    "service unavailable",
    "temporarily unavailable",
    "try again later",
    "429",
    "402",
)


def _clean(value: Any) -> str:
    """Strip whitespace and surrounding quotes from an env value."""
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


def get_openrouter_api_keys(
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the configured OpenRouter keys in ``_1, _2, …, _11`` order.

    Args:
        env: Mapping to read from (defaults to ``os.environ``).  Tests pass
            a plain dict so no real secret is ever involved.

    Empty values are skipped and duplicates are removed (order preserved),
    so a profile that re-exports the same key under two names gets a pool
    of one, not two identical entries.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    keys: list[str] = []
    seen: set[str] = set()
    for name in OPENROUTER_KEY_ENV_VARS:
        value = _clean(source.get(name, ""))
        if value and value not in seen:
            seen.add(value)
            keys.append(value)
    return keys


def key_fingerprint(key: str | None) -> str:
    """A log-safe fingerprint: ``***<last 4>``.  Never log the full key."""
    if not key:
        return "***"
    cleaned = key.strip()
    return f"***{cleaned[-4:]}" if len(cleaned) >= 4 else "***"


def _status_code_from(obj: Any) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    candidates = [obj, getattr(obj, "response", None), getattr(obj, "error", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("status_code", "code", "status"):
            raw = getattr(candidate, attr, None)
            if raw is None:
                continue
            try:
                code = int(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if 100 <= code <= 599:
                return code
    return None


def is_rate_limit_error(error: Any) -> bool:
    """True when *error* looks like a rate-limit / quota / capacity error.

    Accepts an exception instance, a plain string, or ``None``.  Only
    rotatable conditions return True — an authentication failure (401/403),
    a malformed request (400) or a genuine model error must surface to the
    caller instead of pointlessly burning through all eleven keys.
    """
    if error is None:
        return False

    code = _status_code_from(error)
    if code is not None:
        return code in ROTATABLE_STATUS_CODES

    text = str(error).lower()
    if not text:
        return False
    # Class-name hints from SDK exception types (RateLimitError, etc.).
    class_name = type(error).__name__.lower()
    if "ratelimit" in class_name or "toomanyrequests" in class_name:
        return True
    return any(marker in text for marker in _ROTATABLE_TEXT_MARKERS)


class OpenRouterKeyPool:
    """A thread-safe round-robin pool of OpenRouter API keys.

    The pool holds the keys in ``OPENROUTER_KEY_ENV_VARS`` order.  The
    current key is used for the next call; after a successful call the pool
    advances to the next key when ``round_robin`` is enabled, and
    immediately on a rate-limit / quota error.  Keys themselves are never
    printed — only :func:`key_fingerprint` values and counts.
    """

    def __init__(
        self,
        keys: list[str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
        round_robin: bool = True,
    ) -> None:
        if keys is None:
            keys = get_openrouter_api_keys(env)
        cleaned: list[str] = []
        seen: set[str] = set()
        for key in keys:
            value = _clean(key)
            if value and value not in seen:
                seen.add(value)
                cleaned.append(value)
        self.keys: list[str] = cleaned
        self.round_robin = bool(round_robin)
        self._index = 0
        self._lock = threading.Lock()
        # Observability counters — counts and fingerprints only, no secrets.
        self.rotations = 0
        self.rate_limit_hits = 0
        self.successful_calls = 0

    # ── basic state ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.keys)

    def __bool__(self) -> bool:
        return bool(self.keys)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"OpenRouterKeyPool(keys={len(self.keys)}, "
            f"current={key_fingerprint(self.current_key)}, "
            f"round_robin={self.round_robin})"
        )

    @property
    def current_key(self) -> str | None:
        if not self.keys:
            return None
        return self.keys[self._index % len(self.keys)]

    @property
    def current_index(self) -> int:
        return self._index % len(self.keys) if self.keys else 0

    def fingerprints(self) -> list[str]:
        return [key_fingerprint(key) for key in self.keys]

    def reset(self) -> None:
        with self._lock:
            self._index = 0
            self.rotations = 0
            self.rate_limit_hits = 0
            self.successful_calls = 0

    # ── rotation ─────────────────────────────────────────────────────────
    def next_key(self) -> str | None:
        """Advance to the next key (wrap-around) and return it."""
        with self._lock:
            if not self.keys:
                return None
            self._index = (self._index + 1) % len(self.keys)
            self.rotations += 1
            return self.keys[self._index]

    # Backwards-friendly alias.
    rotate = next_key

    def advance_after_success(self) -> str | None:
        """Record a success; advance one step when round-robin is on."""
        with self._lock:
            self.successful_calls += 1
            if self.round_robin and len(self.keys) > 1:
                self._index = (self._index + 1) % len(self.keys)
                self.rotations += 1
            return self.current_key

    def try_rotate_on_error(self, error: Any) -> bool:
        """Rotate iff *error* is a rate-limit / quota error.

        Returns True when the caller should retry with the new current key.
        """
        if not is_rate_limit_error(error):
            return False
        with self._lock:
            self.rate_limit_hits += 1
        if len(self.keys) <= 1:
            return False
        self.next_key()
        return True


_default_pool: OpenRouterKeyPool | None = None
_default_pool_lock = threading.Lock()


def get_default_key_pool(*, round_robin: bool = True) -> OpenRouterKeyPool:
    """Return the process-wide shared pool, creating it on first use.

    Sharing one pool across every LM in a GEPA run means the round-robin
    index is global, not per-LM-instance — otherwise the eval, judge and
    reflection LMs would all start on key 1 and contend for it.
    """
    global _default_pool
    with _default_pool_lock:
        if _default_pool is None:
            _default_pool = OpenRouterKeyPool(round_robin=round_robin)
        return _default_pool


def reset_default_key_pool() -> None:
    """Drop the shared pool (tests, or after the env changes)."""
    global _default_pool
    with _default_pool_lock:
        _default_pool = None


__all__ = [
    "DEFAULT_FALLBACK_MODEL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_PRIMARY_MODEL",
    "OPENROUTER_KEY_ENV_VARS",
    "ROTATABLE_STATUS_CODES",
    "OpenRouterKeyPool",
    "get_default_key_pool",
    "get_openrouter_api_keys",
    "is_rate_limit_error",
    "key_fingerprint",
    "reset_default_key_pool",
]
