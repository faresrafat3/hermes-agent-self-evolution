"""Configuration and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from evolution.core.openrouter_keys import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PRIMARY_MODEL,
    OPENROUTER_KEY_ENV_VARS,
    get_openrouter_api_keys,
)

# The self-evolution runtime is configured to spend zero money: it runs on the
# free models already enabled on this machine. The API keys live in the Hermes
# profile .env, which a bare venv does not read — load it once, without
# overriding anything the caller already exported.
_HERMES_ENV_LOADED = False


def _load_hermes_env() -> None:
    global _HERMES_ENV_LOADED
    if _HERMES_ENV_LOADED:
        return
    _HERMES_ENV_LOADED = True
    candidates = []
    hermes_home = os.getenv("HERMES_HOME")
    if hermes_home:
        candidates.append(Path(hermes_home) / ".env")
    candidates.append(Path.home() / ".hermes" / ".env")
    for path in candidates:
        try:
            if not path.exists():
                continue
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


_load_hermes_env()


# Free model routes. OpenRouter is primary.
#
# The active fleet is the single-leader + two-supporter architecture: the leader
# is meituan/longcat-2.5:free on Nous; the supporters are
# reflection/beam-1.0-501b-a23b:free (reasoning/review) and stealth/crouton-alpha
# (fast operations, 1M context) on Nous and OpenRouter. nemotron and
# north-mini-code are retired from every settings file.
#
# This module's LM calls are rooted at OpenRouter (see OPENROUTER_BASE_URL), so
# the two OpenRouter-reachable supporters are wired here. `stealth/` models are
# free ($0 pricing on OpenRouter). `reflection/beam-1.0-501b-a23b` is free only
# on Nous (`:free` SKU); on OpenRouter it is a micro-cost route ($0.0000003/M
# prompt), used here as the fallback because it is the strongest reviewer in the
# fleet. `meituan/longcat-2.5:free` is a Nous-route model: reaching it needs the
# Nous base_url plus a Nous credential, which this OpenRouter-rooted wiring does
# not carry — pass it explicitly with --optimizer-model / --eval-model once that
# route is wired.
#
# OpenRouter keys: ``~/.hermes/.env`` carries eleven keys
# (OPENROUTER_API_KEY, OPENROUTER_API_KEY_2 … OPENROUTER_API_KEY_11).
# ``_load_hermes_env()`` above loads every KEY=VALUE line into
# ``os.environ``, and :func:`evolution.core.openrouter_keys.get_openrouter_api_keys`
# collects them in that order.  LM construction goes through
# :func:`evolution.core.openrouter_rotation.make_rotating_lm`, which
# round-robins across the pool after each successful call and rotates to
# the next key immediately on a rate-limit / quota (429/402) error,
# switching to the fallback model once every key is exhausted.
FREE_PRIMARY_MODEL = DEFAULT_PRIMARY_MODEL
FREE_FALLBACK_MODEL = DEFAULT_FALLBACK_MODEL
# The openrouter/ prefix is a LiteLLM route: it must be told where the OpenAI-
# compatible endpoint lives, otherwise dspy falls back to the provider default
# and the key in the Hermes .env is never used. Kept overridable by the caller.
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path. Discovered lazily and non-fatally so the config
    # can be constructed even when no repo is present (e.g. unit tests, or
    # callers that pass an explicit path). Use resolve_hermes_agent_path() when
    # an explicit override should win, or get_hermes_agent_path() to require one.
    hermes_agent_path: Optional[Path] = field(default_factory=lambda: _discover_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration — free models only (zero financial cost).
    optimizer_model: str = FREE_PRIMARY_MODEL  # Model for GEPA reflections
    eval_model: str = FREE_PRIMARY_MODEL  # Model for LLM-as-judge scoring
    judge_model: str = FREE_PRIMARY_MODEL  # Model for dataset generation
    fallback_models: list = field(default_factory=lambda: [FREE_FALLBACK_MODEL])

    # OpenRouter key rotation.  The pool is built lazily from the
    # environment (OPENROUTER_API_KEY, OPENROUTER_API_KEY_2 … _11, as
    # loaded from ~/.hermes/.env) at construction time, so tests that set
    # numbered env vars see exactly those keys.  ``openrouter_round_robin``
    # spreads calls across the pool; rate-limit rotation is always on in
    # RotatingLM — these flags are recorded here for observability and
    # for callers that want to disable round-robin spreading.
    openrouter_api_keys: list = field(default_factory=get_openrouter_api_keys)
    openrouter_key_rotation: bool = True
    openrouter_round_robin: bool = True

    # Constraints
    max_skill_size: int = 15_000  # 15KB default
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

    # Eval dataset
    eval_dataset_size: int = 15  # Total examples to generate (small: free models + rate limits)
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    # Off by default: on a free route every loop iteration would re-run the
    # hermes-agent pytest suite, burning the rate limit for a gate that a
    # skill-text evolution cannot affect. Opt in with --run-tests.
    run_pytest: bool = False
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True


def _discover_hermes_agent_path() -> Optional[Path]:
    """Best-effort hermes-agent repo discovery that never raises.

    Returns the discovered path, or None when no repo can be found. Used as
    the EvolutionConfig default so construction never crashes; callers that
    truly require the repo should use get_hermes_agent_path().
    """
    try:
        return get_hermes_agent_path()
    except FileNotFoundError:
        return None


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )


def resolve_hermes_agent_path(hermes_repo: Optional[str] = None) -> Path:
    """Return the hermes-agent repo path, honoring an explicit override.

    An explicit path (for example from ``--hermes-repo``) is expanded and used
    as-is, taking precedence over auto-discovery. This lets callers point at a
    repo in a non-default location without the tool crashing just because
    ``~/.hermes/hermes-agent`` happens to be absent. When no override is given,
    falls back to :func:`get_hermes_agent_path`.
    """
    if hermes_repo:
        return Path(hermes_repo).expanduser()
    return get_hermes_agent_path()
