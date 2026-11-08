"""Ockham budget enforcement & compression for evolved skills.

Root cause this module fixes — *Additive Bloat* in GEPA proposals: the
default proposer in DSPy inflates the skill body with restated, verbose
filler instead of terse rules (measured on a live run: 2,036 → 7,894
chars, +240%). The pipeline previously only *checked* growth after the
fact (``ConstraintValidator._check_growth`` with
``config.max_prompt_growth = 0.2``), so an inflated-but-better candidate
(hit 1.494 → 1.606 on the train subsample) was discarded wholesale by
the growth guard — an overfit on the small sample, a slight regression
on holdout, and the whole free-model quota spent on a rejected run.

The cure applied here is a hard Ockham budget enforced at extraction
time:

1. :func:`compress_skill_body` — a deterministic (no-LLM, zero-cost,
   fully reproducible) distiller. It preserves the condensed rules,
   removes repeated filler, keeps headings and structure, and guarantees
   the output fits the target length.
2. :func:`enforce_skill_growth_budget` — measures the evolved markdown
   against the baseline, and when the growth cap is exceeded, distills
   the module's instructions in place so the extracted SKILL.md complies
   — *without* reverting the new rules the evolution discovered.
3. ``extract_skill_markdown(..., max_growth_ratio=...)`` — the same
   enforcement as a pure extraction-time guard, for call sites that
   never touch the module.

Nothing is ever reverted to the old text: dedupe keeps first
occurrences and unique (new) lines, so evolved content survives in
compressed form and the MD5 still changes.
"""

from __future__ import annotations

import re
from typing import Optional

from evolution.skills.skill_module import reassemble_skill

_HEADING_RE = re.compile(r"^\s*#{1,6}\s")


def growth_ratio(text: str, baseline_text: str) -> float:
    """Fractional size change of ``text`` relative to ``baseline_text``."""
    return (len(text) - len(baseline_text)) / max(1, len(baseline_text))


def meets_budget(text: str, baseline_text: str, max_growth_ratio: float) -> bool:
    """True when ``text`` is within ``max_growth_ratio`` of the baseline size."""
    return growth_ratio(text, baseline_text) <= max_growth_ratio


def _normalize(line: str) -> str:
    """Case/whitespace-insensitive key for duplicate detection."""
    return " ".join(line.lower().split()).strip(" .,-;:")


def _dedupe_lines(text: str) -> str:
    """Collapse blank runs and drop repeated lines, keeping first occurrences.

    Additive-bloat output restates the same procedures in many wordings,
    so the first (Original) occurrence and every genuinely new rule line
    survive while the copied filler is removed.
    """
    out: list[str] = []
    seen: set[str] = set()
    prev_blank = False
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if prev_blank:
                continue
            prev_blank = True
            out.append(line)
            continue
        prev_blank = False
        key = _normalize(line)
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out).strip("\n")


def _trim_filler_lines(lines: list[str], target_len: int) -> list[str]:
    """Drop the longest non-heading lines until the joined length fits.

    Bloat is verbose; the condensed rules added by the evolution are
    typically short imperatives. Trimming longest-first therefore cuts
    filler before it can touch a terse new rule, wherever the rule sits
    in the document (including appended at the tail). Headings are never
    dropped: they carry the document structure.
    """
    lines = list(lines)
    while len("\n".join(lines)) > target_len:
        droppable = [
            i for i, line in enumerate(lines)
            if line.strip() and not _HEADING_RE.match(line)
        ]
        if not droppable:
            break
        longest = max(droppable, key=lambda i: len(lines[i]))
        del lines[longest]
    return lines


def compress_skill_body(body: str, target_len: int) -> str:
    """Deterministically distill ``body`` so it fits ``target_len`` chars.

    Stages:

    1. Dedupe repeated lines and collapse blank runs.
    2. If still over, drop the longest filler lines (bloat is verbose,
       rules are terse), never dropping headings.
    3. Hard guarantee: cut at the last newline that keeps the body at
       or under ``target_len`` (pathological only).

    The result is always ``<= target_len`` chars whenever ``target_len``
    is non-negative, and is a pure function of ``body``.
    """
    target_len = max(0, target_len)
    if len(body) <= target_len:
        return body

    text = _dedupe_lines(body)
    if len(text) <= target_len:
        return text

    lines = _trim_filler_lines(text.split("\n"), target_len)
    text = _dedupe_lines("\n".join(lines))
    if len(text) <= target_len:
        return text

    cut = text.rfind("\n", 0, target_len)
    text = text[: cut if cut > 0 else target_len].rstrip()
    return text


def _body_target_len(
    frontmatter: str,
    baseline_text: str,
    max_growth_ratio: float,
) -> int:
    """Body char budget so the reassembled markdown meets the growth cap."""
    total_budget = int(len(baseline_text) * (1 + max_growth_ratio))
    wrapper_len = len(reassemble_skill(frontmatter, ""))
    return max(0, total_budget - wrapper_len)


def ockham_extract_markdown(
    full_markdown: str,
    frontmatter: str,
    baseline_text: str,
    max_growth_ratio: float,
) -> str:
    """Return ``full_markdown`` untouched when within budget, else distilled.

    Compresses only the body; the YAML frontmatter block is carried over
    from the evolved markdown verbatim, so name/description/metadata
    survive byte-for-byte.
    """
    if meets_budget(full_markdown, baseline_text, max_growth_ratio):
        return full_markdown

    evolved_body = full_markdown.split("---", 2)[2].strip("\n") if full_markdown.startswith("---") else full_markdown
    body_target = _body_target_len(frontmatter, baseline_text, max_growth_ratio)
    return reassemble_skill(frontmatter, compress_skill_body(evolved_body, body_target))


def enforce_skill_growth_budget(
    module,
    max_growth_ratio: float = 0.2,
    baseline_text: Optional[str] = None,
) -> dict:
    """Distill ``module``'s instructions in place when over the growth cap.

    Args:
        module: A ``SkillModule`` (baseline or GEPA-optimized).
        max_growth_ratio: Max fractional growth allowed over the baseline
            markdown size (mirrors ``EvolutionConfig.max_prompt_growth``).
        baseline_text: The reference markdown (defaults to the module's
            captured original raw file).

    Returns:
        A report dict: ``applied`` (was compression needed),
        ``before_size`` / ``after_size`` (extracted markdown sizes),
        ``before_growth`` / ``after_growth`` (fractional growths), and
        ``max_growth_ratio`` (the cap that was enforced).
    """
    from evolution.skills.skill_module import extract_skill_markdown

    baseline = baseline_text if baseline_text is not None else module.original_raw
    if baseline is None:
        raise ValueError(
            "enforce_skill_growth_budget needs a baseline: pass baseline_text "
            "or construct the module with raw_skill=..."
        )

    before = extract_skill_markdown(module)
    before_growth = growth_ratio(before, baseline)

    if meets_budget(before, baseline, max_growth_ratio):
        return {
            "applied": False,
            "before_size": len(before),
            "after_size": len(before),
            "before_growth": before_growth,
            "after_growth": before_growth,
            "max_growth_ratio": max_growth_ratio,
        }

    body_target = _body_target_len(module.frontmatter, baseline, max_growth_ratio)
    compressed_body = compress_skill_body(module.skill_text, body_target)
    module.predictor.signature = module.predictor.signature.with_instructions(compressed_body)

    after = extract_skill_markdown(module)
    return {
        "applied": True,
        "before_size": len(before),
        "after_size": len(after),
        "before_growth": before_growth,
        "after_growth": growth_ratio(after, baseline),
        "max_growth_ratio": max_growth_ratio,
    }
