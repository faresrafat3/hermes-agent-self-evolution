"""DSPy LM wrapper that rotates OpenRouter keys on rate limits.

The regression this fixes: every LM in the self-evolution engine was built
as ``dspy.LM(model, base_url=OPENROUTER_BASE_URL)`` with **no** ``api_key``,
so LiteLLM silently used the single ``OPENROUTER_API_KEY`` from the
environment.  The first 429 (or, on the paid beam route, 402 quota error)
aborted the whole GEPA run even though ten more keys were sitting in
``~/.hermes/.env``.

:class:`RotatingLM` subclasses ``dspy.LM`` and overrides ``__call__``:

* each call starts with the shared pool's current key (round-robin: the
  index advances after every successful call when enabled);
* on a rate-limit / quota / capacity error it advances to the next key
  and retries, up to ``len(keys)`` attempts per model;
* once every key has been tried on the primary model
  (``openrouter/stealth/crouton-alpha`` by default) it switches to the
  fallback model (``openrouter/reflection/beam-1.0-501b-a23b`` by
  default), caps ``max_tokens`` to :data:`FALLBACK_MAX_TOKENS` — the
  provider-default 131072 exceeds every key's credit affordance on the
  beam route, so the uncapped fallback call can never succeed — and
  retries the key cycle once more;
* any non-rotatable error (401/403/400, genuine model errors) is raised
  immediately, unchanged.

Keys are never logged; only fingerprints and counts are.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import dspy

from evolution.core.openrouter_keys import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_PRIMARY_MODEL,
    OpenRouterKeyPool,
    get_default_key_pool,
    is_rate_limit_error,
    key_fingerprint,
)

logger = logging.getLogger(__name__)

# Beam's provider-default budget is 131072 tokens; the eleven keys afford
# roughly 11k–36k tokens each on that route (measured live — see
# reports/crouton-gepa-benchmark.md).  8192 keeps every key 3–11 viable
# while still leaving ample room for GEPA reflection / judge output.
FALLBACK_MAX_TOKENS = 8192


class RotatingLM(dspy.LM):
    """A ``dspy.LM`` that rotates OpenRouter keys on rate limits.

    Args:
        model: LiteLLM model string, e.g. ``openrouter/stealth/crouton-alpha``.
        pool: Key pool to use.  Defaults to the shared
            :func:`get_default_key_pool` (all ``OPENROUTER_API_KEY[_N]``
            values from the environment).
        fallback_model: Model to switch to after every key is exhausted
            on the primary model.  ``None`` disables model fallback.
        base_url: OpenAI-compatible endpoint.  Defaults to
            ``OPENROUTER_BASE_URL`` / ``https://openrouter.ai/api/v1``.
        api_key: Explicit key override (bypasses the pool for the first
            attempt only; rotation still applies afterwards).
        max_rotation_attempts: Total attempts per model before giving up.
            Defaults to ``max(1, len(pool))``.
        round_robin: Advance the shared index after each successful call.
        **kwargs: Forwarded to ``dspy.LM`` unchanged.
    """

    def __init__(
        self,
        model: str,
        *,
        pool: OpenRouterKeyPool | None = None,
        fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        max_rotation_attempts: int | None = None,
        round_robin: bool = True,
        fallback_max_tokens: int = FALLBACK_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        self.key_pool = pool if pool is not None else get_default_key_pool(
            round_robin=round_robin
        )
        self._round_robin = round_robin
        self.original_model = model
        self.fallback_model = fallback_model
        self.fallback_max_tokens = fallback_max_tokens
        self._using_fallback = False
        self.max_rotation_attempts = (
            max_rotation_attempts
            if max_rotation_attempts is not None
            else max(1, len(self.key_pool))
        )

        init_kwargs: dict[str, Any] = dict(kwargs)
        if base_url is not None:
            init_kwargs["base_url"] = base_url
        elif "base_url" not in init_kwargs:
            init_kwargs["base_url"] = os.getenv(
                "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
            )
        effective_key = api_key if api_key is not None else self.key_pool.current_key
        if effective_key is not None:
            init_kwargs["api_key"] = effective_key
        super().__init__(model, **init_kwargs)

    # ── internal helpers ─────────────────────────────────────────────────
    def _sync_key(self) -> None:
        """Point ``lm.kwargs['api_key']`` at the pool's current key."""
        current = self.key_pool.current_key
        if current is not None:
            self.kwargs["api_key"] = current

    def _switch_to_fallback(self) -> bool:
        """Switch ``self.model`` to the fallback model.  True if switched."""
        if self._using_fallback or not self.fallback_model:
            return False
        if self.fallback_model == self.model:
            return False
        logger.warning(
            "All %s OpenRouter key(s) exhausted on %s; "
            "switching to fallback model %s",
            len(self.key_pool),
            self.model,
            self.fallback_model,
        )
        self.model = self.fallback_model
        self._using_fallback = True
        # The beam route's default budget (131072) exceeds any single
        # key's credit affordance — cap it only when the caller did not
        # set an explicit budget.
        if self.kwargs.get("max_tokens") is None:
            self.kwargs["max_tokens"] = self.fallback_max_tokens
        self._sync_key()
        return True

    def rotate_now(self) -> str | None:
        """Manually advance to the next key and return its fingerprint."""
        key = self.key_pool.next_key()
        self._sync_key()
        return key_fingerprint(key)

    # ── the actual call ────────────────────────────────────────────────
    def __call__(self, prompt=None, *, messages=None, **kwargs):  # type: ignore[override]
        attempts = 0
        max_attempts = max(1, self.max_rotation_attempts)
        last_error: Exception | None = None

        while True:
            try:
                result = super().__call__(prompt=prompt, messages=messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised below if not rotatable
                last_error = exc
                if not is_rate_limit_error(exc):
                    raise

                attempts += 1
                rotated = False
                if len(self.key_pool) > 1:
                    rotated = self.key_pool.try_rotate_on_error(exc)
                    self._sync_key()
                    logger.warning(
                        "Rate limit / quota on key %s (attempt %s); "
                        "rotated to %s",
                        key_fingerprint(self.kwargs.get("api_key")),
                        attempts,
                        key_fingerprint(self.key_pool.current_key),
                    )

                if attempts < max_attempts and (rotated or len(self.key_pool) <= 1):
                    # Either we got a fresh key, or there is only one key
                    # and a plain retry (DSPy's own retries already ran
                    # underneath us) is still worth one more shot — but
                    # avoid an unbounded loop on a single key.
                    if rotated or attempts == 1:
                        continue
                # Keys exhausted on this model: try the fallback model once.
                if self._switch_to_fallback():
                    attempts = 0
                    continue
                raise

            # Success path.
            if self._round_robin:
                self.key_pool.advance_after_success()
                self._sync_key()
            return result

        # Unreachable — the loop either returns or raises — but keep the
        # type checker and readers honest.
        assert last_error is not None  # pragma: no cover
        raise last_error  # pragma: no cover


def make_rotating_lm(
    model: str | None = None,
    *,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    pool: OpenRouterKeyPool | None = None,
    round_robin: bool = True,
    base_url: str | None = None,
    **kwargs: Any,
) -> RotatingLM:
    """Build a :class:`RotatingLM` for *model* with sensible defaults.

    ``model=None`` selects the primary evolution model
    (``openrouter/stealth/crouton-alpha``); the fallback defaults to
    ``openrouter/reflection/beam-1.0-501b-a23b`` unless the caller passes
    ``fallback_model=None`` or the model already *is* the fallback.
    """
    effective_model = model or DEFAULT_PRIMARY_MODEL
    effective_fallback = fallback_model
    if effective_fallback is not None and effective_fallback == effective_model:
        effective_fallback = None
    return RotatingLM(
        effective_model,
        pool=pool,
        fallback_model=effective_fallback,
        round_robin=round_robin,
        base_url=base_url,
        **kwargs,
    )


def configure_rotating_lm(
    model: str | None = None,
    *,
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL,
    pool: OpenRouterKeyPool | None = None,
    round_robin: bool = True,
    base_url: str | None = None,
    **kwargs: Any,
) -> RotatingLM:
    """Build a :class:`RotatingLM` and install it via ``dspy.configure``."""
    lm = make_rotating_lm(
        model,
        fallback_model=fallback_model,
        pool=pool,
        round_robin=round_robin,
        base_url=base_url,
        **kwargs,
    )
    dspy.configure(lm=lm)
    return lm


__all__ = [
    "FALLBACK_MAX_TOKENS",
    "RotatingLM",
    "configure_rotating_lm",
    "make_rotating_lm",
]
