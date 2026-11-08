"""Tests for OpenRouter key rotation (OPENROUTER_API_KEY[_2.._11]).

These tests use fake keys only (``sk-or-v1-test-…``) injected through
``monkeypatch`` / explicit mappings — no network, no real secret, and no
 reliance on the developer's ``~/.hermes/.env``.  Every test that touches
the environment first deletes all eleven numbered variables, so a real
profile with 11 keys loaded at import time cannot leak into the pool
under test.
"""

from __future__ import annotations

import pytest

from evolution.core.openrouter_keys import (
    OPENROUTER_KEY_ENV_VARS,
    OpenRouterKeyPool,
    get_openrouter_api_keys,
    is_rate_limit_error,
    key_fingerprint,
    reset_default_key_pool,
)


@pytest.fixture(autouse=True)
def _clean_openrouter_env(monkeypatch):
    """Start every test with a clean OpenRouter env and a fresh pool."""
    for name in OPENROUTER_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_default_key_pool()
    yield
    reset_default_key_pool()


def _set_keys(monkeypatch, count: int) -> list[str]:
    keys = [f"sk-or-v1-test-{i:02d}-{'x' * 20}" for i in range(1, count + 1)]
    for name, key in zip(OPENROUTER_KEY_ENV_VARS, keys):
        monkeypatch.setenv(name, key)
    return keys


# ── get_openrouter_api_keys ──────────────────────────────────────────────

def test_get_keys_reads_numbered_env_vars_in_order(monkeypatch):
    keys = _set_keys(monkeypatch, 11)
    assert get_openrouter_api_keys() == keys
    assert len(OPENROUTER_KEY_ENV_VARS) == 11


def test_get_keys_skips_empty_and_dedupes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-v1-a  ")
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "")  # empty -> skipped
    monkeypatch.setenv("OPENROUTER_API_KEY_3", "sk-or-v1-a")  # dup -> skipped
    monkeypatch.setenv("OPENROUTER_API_KEY_4", '"sk-or-v1-b"')  # quotes stripped
    assert get_openrouter_api_keys() == ["sk-or-v1-a", "sk-or-v1-b"]


def test_get_keys_from_explicit_mapping():
    env = {
        "OPENROUTER_API_KEY": "k1",
        "OPENROUTER_API_KEY_2": "k2",
        "OPENROUTER_API_KEY_11": "k11",
        "UNRELATED": "ignored",
    }
    assert get_openrouter_api_keys(env) == ["k1", "k2", "k11"]


def test_get_keys_empty_env_returns_empty_list():
    assert get_openrouter_api_keys({}) == []


# ── is_rate_limit_error ──────────────────────────────────────────────────

class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str = "boom"):
        super().__init__(message)
        self.status_code = status_code


def test_is_rate_limit_error_status_codes():
    assert is_rate_limit_error(_FakeHTTPError(429)) is True
    assert is_rate_limit_error(_FakeHTTPError(402)) is True  # quota / credits
    assert is_rate_limit_error(_FakeHTTPError(503)) is True
    assert is_rate_limit_error(_FakeHTTPError(401)) is False  # auth: do not rotate
    assert is_rate_limit_error(_FakeHTTPError(400)) is False  # bad request
    assert is_rate_limit_error(None) is False
    assert is_rate_limit_error("") is False


def test_is_rate_limit_error_text_markers():
    assert is_rate_limit_error(Exception("Rate limit exceeded, try again later")) is True
    assert is_rate_limit_error(Exception("Insufficient credits. (402)")) is True
    assert is_rate_limit_error(Exception("model overloaded")) is True
    assert is_rate_limit_error(Exception("authentication failed")) is False


# ── OpenRouterKeyPool round-robin ────────────────────────────────────────

def test_pool_round_robin_sequence():
    pool = OpenRouterKeyPool(keys=["k1", "k2", "k3"])
    assert pool.current_key == "k1"
    assert pool.next_key() == "k2"
    assert pool.next_key() == "k3"
    assert pool.next_key() == "k1"  # wraps
    assert pool.rotations == 3


def test_pool_advance_after_success_only_when_round_robin():
    pool = OpenRouterKeyPool(keys=["k1", "k2"], round_robin=True)
    pool.advance_after_success()
    assert pool.current_key == "k2"
    assert pool.successful_calls == 1

    sticky = OpenRouterKeyPool(keys=["k1", "k2"], round_robin=False)
    sticky.advance_after_success()
    assert sticky.current_key == "k1"


def test_pool_try_rotate_on_error_rotates_only_on_rate_limit():
    pool = OpenRouterKeyPool(keys=["k1", "k2", "k3"])
    assert pool.try_rotate_on_error(_FakeHTTPError(429)) is True
    assert pool.current_key == "k2"
    assert pool.rate_limit_hits == 1

    assert pool.try_rotate_on_error(_FakeHTTPError(401)) is False
    assert pool.current_key == "k2"  # unchanged
    assert pool.rate_limit_hits == 1


def test_pool_fingerprints_never_expose_full_key():
    pool = OpenRouterKeyPool(keys=["sk-or-v1-secretvalue1234"])
    assert pool.fingerprints() == ["***1234"]
    assert "secretvalue" not in repr(pool)


def test_pool_reads_numbered_env_vars(monkeypatch):
    keys = _set_keys(monkeypatch, 3)
    pool = OpenRouterKeyPool()
    assert pool.keys == keys


# ── EvolutionConfig integration ──────────────────────────────────────────

def test_evolution_config_picks_up_numbered_env_vars(monkeypatch):
    keys = _set_keys(monkeypatch, 4)
    from evolution.core.config import EvolutionConfig

    config = EvolutionConfig()
    assert config.openrouter_api_keys == keys
    assert config.openrouter_key_rotation is True
    assert config.openrouter_round_robin is True
    # The evolution fleet defaults required by the task.
    assert config.optimizer_model == "openrouter/stealth/crouton-alpha"
    assert config.eval_model == "openrouter/stealth/crouton-alpha"
    assert config.fallback_models == ["openrouter/reflection/beam-1.0-501b-a23b"]


# ── RotatingLM / make_rotating_lm ────────────────────────────────────────

def test_make_rotating_lm_uses_first_key_and_defaults(monkeypatch):
    keys = _set_keys(monkeypatch, 3)
    from evolution.core.openrouter_rotation import RotatingLM, make_rotating_lm

    lm = make_rotating_lm()
    assert isinstance(lm, RotatingLM)
    assert lm.model == "openrouter/stealth/crouton-alpha"
    assert lm.fallback_model == "openrouter/reflection/beam-1.0-501b-a23b"
    assert lm.kwargs["api_key"] == keys[0]
    assert lm.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert len(lm.key_pool) == 3


def test_make_rotating_lm_with_explicit_pool(monkeypatch):
    _set_keys(monkeypatch, 11)  # env has 11, but explicit pool wins
    from evolution.core.openrouter_rotation import make_rotating_lm

    pool = OpenRouterKeyPool(keys=["only-a", "only-b"])
    lm = make_rotating_lm("openrouter/stealth/crouton-alpha", pool=pool)
    assert lm.key_pool is pool
    assert lm.kwargs["api_key"] == "only-a"


def test_rotating_lm_call_rotates_on_rate_limit_then_succeeds(monkeypatch):
    _set_keys(monkeypatch, 3)
    from unittest.mock import patch

    import dspy

    from evolution.core.openrouter_rotation import make_rotating_lm

    lm = make_rotating_lm(round_robin=False)
    with patch.object(
        dspy.LM,
        "__call__",
        side_effect=[_FakeHTTPError(429, "rate limit"), "ok-result"],
    ) as mock_call:
        result = lm(prompt="hello")
    assert result == "ok-result"
    assert mock_call.call_count == 2
    # It rotated to the second key before retrying.
    assert lm.key_pool.current_index == 1
    assert lm.key_pool.rate_limit_hits == 1


def test_rotating_lm_round_robin_advances_after_success(monkeypatch):
    _set_keys(monkeypatch, 3)
    from unittest.mock import patch

    import dspy

    from evolution.core.openrouter_rotation import make_rotating_lm

    lm = make_rotating_lm(round_robin=True)
    with patch.object(dspy.LM, "__call__", side_effect=["a", "b"]):
        assert lm(prompt="one") == "a"
        assert lm.key_pool.current_index == 1
        assert lm(prompt="two") == "b"
        assert lm.key_pool.current_index == 2


def test_rotating_lm_non_rate_limit_error_raises_without_rotation(monkeypatch):
    _set_keys(monkeypatch, 3)
    from unittest.mock import patch

    import dspy

    from evolution.core.openrouter_rotation import make_rotating_lm

    lm = make_rotating_lm(round_robin=False)
    with patch.object(dspy.LM, "__call__", side_effect=_FakeHTTPError(401)):
        with pytest.raises(Exception):
            lm(prompt="hello")
    assert lm.key_pool.current_index == 0
    assert lm.key_pool.rate_limit_hits == 0


def test_rotating_lm_switches_to_fallback_after_exhausting_keys(monkeypatch):
    _set_keys(monkeypatch, 2)
    from unittest.mock import patch

    import dspy

    from evolution.core.openrouter_rotation import make_rotating_lm

    lm = make_rotating_lm(round_robin=False)
    # 2 keys on primary (both 429) -> fallback model -> success.
    with patch.object(
        dspy.LM,
        "__call__",
        side_effect=[
            _FakeHTTPError(429, "rate limit"),
            _FakeHTTPError(429, "rate limit"),
            "fallback-ok",
        ],
    ):
        result = lm(prompt="hello")
    assert result == "fallback-ok"
    assert lm.model == "openrouter/reflection/beam-1.0-501b-a23b"
    # The fallback switch caps max_tokens so the beam route stays within
    # a single key's credit affordance (its default budget is 131072).
    assert lm.kwargs["max_tokens"] == 8192


def test_key_fingerprint_helper():
    assert key_fingerprint("sk-or-v1-abcdef1234") == "***1234"
    assert key_fingerprint(None) == "***"
    assert key_fingerprint("") == "***"
