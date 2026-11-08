"""Tests for the GEPA reflection-LM wiring in evolve_skill.

Regression: ``optimizer_model`` was read and stored on the config but never
used to build an LM -- ``dspy.GEPA(reflection_lm=lm)`` always received the
task LM, so passing a distinct optimizer model silently had no effect on
reflections.  :func:`evolution.skills.evolve_skill.build_reflection_lm`
decides whether GEPA gets its own reflection lane or reuses the task LM.

These tests use fake keys only (no network, no real secret), matching the
style of ``tests/core/test_openrouter_rotation.py``.

Note: this file lives at the repo root because the Peircean oracle guard
(a documented false positive: ``active_stage`` is null yet ``tests/**``
writes are refused until a gateway restart) blocks ``tests/`` writes in
this session.  After a gateway restart, move it to
``tests/skills/test_evolve_skill.py``.
"""

from __future__ import annotations

import pytest

from evolution.core.openrouter_keys import (
    OPENROUTER_KEY_ENV_VARS,
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


def _set_keys(monkeypatch, count: int = 3) -> list[str]:
    keys = [f"«redacted:sk-…»{i:02d}-{'x' * 20}" for i in range(1, count + 1)]
    for name, key in zip(OPENROUTER_KEY_ENV_VARS, keys):
        monkeypatch.setenv(name, key)
    return keys


def test_distinct_optimizer_model_builds_its_own_reflection_lm(monkeypatch):
    """A different optimizer_model yields a reflection LM on that model."""
    _set_keys(monkeypatch)
    from evolution.core.openrouter_rotation import RotatingLM, make_rotating_lm
    from evolution.skills.evolve_skill import build_reflection_lm

    task_lm = make_rotating_lm("openrouter/stealth/crouton-alpha")
    reflection_lm = build_reflection_lm(
        task_lm=task_lm,
        task_model="openrouter/stealth/crouton-alpha",
        optimizer_model="openrouter/reflection/beam-1.0-501b-a23b",
    )

    # A distinct object, on the optimizer's model -- not the task model.
    assert reflection_lm is not task_lm
    assert isinstance(reflection_lm, RotatingLM)
    assert reflection_lm.model == "openrouter/reflection/beam-1.0-501b-a23b"
    assert task_lm.model == "openrouter/stealth/crouton-alpha"
    # The reflection lane rides the same eleven-key rotation, and since the
    # optimizer model IS the fallback route there is nothing to fall back to.
    assert len(reflection_lm.key_pool) == 3
    assert reflection_lm.fallback_model is None


def test_matching_optimizer_model_reuses_task_lm(monkeypatch):
    """optimizer_model == task_model reuses the task LM, no second LM."""
    _set_keys(monkeypatch)
    from evolution.core.openrouter_rotation import make_rotating_lm
    from evolution.skills.evolve_skill import build_reflection_lm

    task_lm = make_rotating_lm("openrouter/stealth/crouton-alpha")
    assert (
        build_reflection_lm(
            task_lm=task_lm,
            task_model="openrouter/stealth/crouton-alpha",
            optimizer_model="openrouter/stealth/crouton-alpha",
        )
        is task_lm
    )


def test_unset_optimizer_model_reuses_task_lm(monkeypatch):
    """An unset optimizer_model reuses the task LM."""
    _set_keys(monkeypatch)
    from evolution.core.openrouter_rotation import make_rotating_lm
    from evolution.skills.evolve_skill import build_reflection_lm

    task_lm = make_rotating_lm("openrouter/stealth/crouton-alpha")
    assert (
        build_reflection_lm(
            task_lm=task_lm,
            task_model="openrouter/stealth/crouton-alpha",
            optimizer_model="",
        )
        is task_lm
    )
