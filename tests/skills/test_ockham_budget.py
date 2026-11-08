"""Tests for Ockham budget enforcement & compression (anti-bloat guard).

Incident these tests close: the first live GEPA mutation inflated a skill
body by +240% (2,036 → 7,894 chars) with restated filler, collided with the
growth guard (``max_growth_ratio = 20%``), and overfit the tiny subsample.
The fix: :func:`evolution.skills.ockham.enforce_skill_growth_budget`
distills an over-budget evolved body deterministically *before* the
constraint guard sees it, preserving the condensed rules, the YAML
frontmatter, and a changed MD5 (no full revert to the old text).

Every test here is hermetic — no network, no LM, deterministic input.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.skills.ockham import (
    compress_skill_body,
    enforce_skill_growth_budget,
    growth_ratio,
    meets_budget,
    ockham_extract_markdown,
)
from evolution.skills.skill_module import (
    SkillModule,
    extract_skill_markdown,
    load_skill,
)

SAMPLE_SKILL = """---
name: test-skill
description: A skill for testing things
version: 1.0.0
metadata:
  hermes:
    tags: [testing]
---

# Test Skill — Testing Things

## When to Use
Use this when you need to test things.

## Procedure
1. First, do the thing
2. Then, verify it worked
3. Report results

## Pitfalls
- Don't forget to check edge cases
"""

# The one genuinely new rule the evolution "discovered" -- it must survive
# compression; everything else is restated filler.
NEW_RULE = "State the invariant before acting on any mutation."


def _bloated_body(candidate: str, repeats: int = 6) -> str:
    """GEPA-style additive bloat: original body + new rule + repeated filler."""
    filler_blocks = [
        "It is very important to always remember to do the thing carefully "
        "and thoroughly and not to rush through any of the steps in the "
        "procedure because rushing can cause many different kinds of problems.",
        "Please note that verifying the thing worked is an essential part of "
        "the process and it should never be skipped under any circumstances "
        "whatsoever because skipping verification is a common source of error.",
        "As mentioned above, reporting results clearly is necessary so that "
        "everyone involved can understand exactly what happened during the "
        "run and can take the appropriate follow-up actions afterwards.",
    ]
    bloated = [candidate, NEW_RULE]
    for i in range(repeats):
        bloated.append(filler_blocks[i % len(filler_blocks)])
        bloated.append("1. First, do the thing")  # verbatim restatement
    return "\n\n".join(bloated)


def _build_module(tmp_path: Path) -> tuple[SkillModule, dict]:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(SAMPLE_SKILL)
    skill = load_skill(skill_file)
    return (
        SkillModule(skill["body"], frontmatter=skill["frontmatter"], raw_skill=skill["raw"]),
        skill,
    )


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _frontmatter_block(markdown: str) -> str:
    return markdown.split("---", 2)[1]


class TestCompressSkillBody:
    def test_within_budget_returned_unchanged(self):
        body = "# Title\n\nShort body."
        assert compress_skill_body(body, 1000) == body

    def test_deterministic_and_idempotent(self):
        bloated = _bloated_body(SAMPLE_SKILL.split("---", 2)[2].strip())
        once = compress_skill_body(bloated, 400)
        twice = compress_skill_body(once, 400)
        assert once == twice  # pure function of input
        assert len(once) <= 400

    def test_guarantee_never_exceeds_target(self):
        # Pathological: no headings, all unique filler -- tail drop + hard cut.
        body = "\n".join(f"unique filler line number {i} with words" for i in range(500))
        assert len(compress_skill_body(body, 300)) <= 300

    def test_dedupe_kills_verbatim_restatements(self):
        body = "Do the thing.\n\nDo the thing.\n\ndo the thing.\n\nNew rule."
        out = compress_skill_body(body, 30)  # over budget -> compression runs
        assert out.count("Do the thing") == 1
        assert "New rule" in out
        assert len(out) <= 30

    def test_collapses_blank_runs(self):
        body = "A\n\n\n\n\nB\n\n\nC"
        out = compress_skill_body(body, 10)  # 11 chars over a 10 budget -> compresses
        assert "\n\n\n" not in out
        assert "A" in out and "B" in out and "C" in out

    def test_preserves_headings_when_trimming(self):
        body = "# Keep Me\n\n" + "\n".join(f"filler line {i}" for i in range(200))
        out = compress_skill_body(body, 120)
        assert "# Keep Me" in out
        assert len(out) <= 120


class TestGrowthHelpers:
    def test_growth_ratio_math(self):
        assert growth_ratio("a" * 120, "a" * 100) == pytest.approx(0.2)
        assert growth_ratio("a" * 90, "a" * 100) == pytest.approx(-0.1)

    def test_meets_budget_boundary(self):
        assert meets_budget("a" * 120, "a" * 100, 0.2)
        assert not meets_budget("a" * 121, "a" * 100, 0.2)


class TestEnforceSkillGrowthBudget:
    def test_bloated_mutation_is_compressed_into_budget(self, tmp_path):
        module, skill = _build_module(tmp_path)
        bloated = _bloated_body(skill["body"])
        module.predictor.signature = module.predictor.signature.with_instructions(bloated)

        before = extract_skill_markdown(module)
        assert growth_ratio(before, SAMPLE_SKILL) > 2.0  # the +200%-class incident

        report = enforce_skill_growth_budget(module, max_growth_ratio=0.2)

        assert report["applied"] is True
        assert report["before_size"] == len(before)
        assert report["after_size"] < report["before_size"]
        assert report["after_growth"] <= 0.2
        assert report["before_growth"] > 0.2

        after = extract_skill_markdown(module)
        assert len(after) == report["after_size"]
        assert meets_budget(after, SAMPLE_SKILL, 0.2)
        # Not a revert: content changed and the NEW rule survives.
        assert _md5(after) != _md5(SAMPLE_SKILL)
        assert NEW_RULE in after
        # Frontmatter preserved byte-for-byte and still valid YAML.
        assert _frontmatter_block(after) == _frontmatter_block(SAMPLE_SKILL)
        assert yaml.safe_load(_frontmatter_block(after))["name"] == "test-skill"

    def test_within_budget_module_is_untouched(self, tmp_path):
        module, skill = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(
            skill["body"] + "\n\nState it once, briefly."
        )
        report = enforce_skill_growth_budget(module, max_growth_ratio=0.2)
        assert report["applied"] is False
        assert report["before_size"] == report["after_size"]
        assert module.skill_text.endswith("State it once, briefly.")

    def test_unchanged_module_reports_no_op(self, tmp_path):
        module, _ = _build_module(tmp_path)
        report = enforce_skill_growth_budget(module, max_growth_ratio=0.2)
        assert report["applied"] is False
        assert extract_skill_markdown(module) == SAMPLE_SKILL

    def test_custom_budget_is_honored(self, tmp_path):
        module, skill = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(
            _bloated_body(skill["body"])
        )
        report = enforce_skill_growth_budget(module, max_growth_ratio=0.5)
        assert report["applied"] is True
        assert report["after_growth"] <= 0.5

    def test_new_rule_at_tail_survives_longest_first_trim(self, tmp_path):
        """Worst case: GEPA appends the new rule at the very end, after filler."""
        module, skill = _build_module(tmp_path)
        bloated = skill["body"] + "\n\n" + "\n\n".join(
            f"This is an extremely verbose and wordy restatement number {i} of the "
            "original procedure that adds no new information at all whatsoever."
            for i in range(15)
        ) + "\n\n" + NEW_RULE
        module.predictor.signature = module.predictor.signature.with_instructions(bloated)
        report = enforce_skill_growth_budget(module, max_growth_ratio=0.2)
        assert report["applied"] is True
        assert report["after_growth"] <= 0.2
        assert NEW_RULE in module.skill_text

    def test_zero_budget_never_grows(self, tmp_path):
        module, skill = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(
            _bloated_body(skill["body"])
        )
        report = enforce_skill_growth_budget(module, max_growth_ratio=0.0)
        assert report["after_growth"] <= 0.0

    def test_missing_baseline_raises(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)
        module = SkillModule(skill["body"], frontmatter=skill["frontmatter"])  # no raw_skill
        with pytest.raises(ValueError):
            enforce_skill_growth_budget(module, max_growth_ratio=0.2)


class TestExtractWithOckhamBudget:
    def test_extract_enforces_budget_at_extraction_time(self, tmp_path):
        module, skill = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(
            _bloated_body(skill["body"])
        )
        extracted = extract_skill_markdown(module, max_growth_ratio=0.2)
        assert meets_budget(extracted, SAMPLE_SKILL, 0.2)
        assert NEW_RULE in extracted
        assert _md5(extracted) != _md5(SAMPLE_SKILL)
        assert _frontmatter_block(extracted) == _frontmatter_block(SAMPLE_SKILL)

    def test_extract_without_budget_keeps_old_behavior(self, tmp_path):
        module, skill = _build_module(tmp_path)
        bloated = _bloated_body(skill["body"])
        module.predictor.signature = module.predictor.signature.with_instructions(bloated)
        extracted = extract_skill_markdown(module)
        assert growth_ratio(extracted, SAMPLE_SKILL) > 2.0  # untouched, as before

    def test_extract_budget_uses_explicit_baseline(self, tmp_path):
        module, skill = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(
            skill["body"] + "\n\n" + NEW_RULE
        )
        # A tighter explicit baseline (frontmatter + partial body) forces
        # compression even for modest growth that the full baseline would allow.
        tight_baseline = SAMPLE_SKILL[:250]
        extracted = extract_skill_markdown(
            module, max_growth_ratio=0.2, baseline_text=tight_baseline
        )
        assert meets_budget(extracted, tight_baseline, 0.2)

    def test_extract_budget_without_baseline_raises(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)
        module = SkillModule(skill["body"], frontmatter=skill["frontmatter"])
        module.predictor.signature = module.predictor.signature.with_instructions(
            _bloated_body(skill["body"])
        )
        with pytest.raises(ValueError):
            extract_skill_markdown(module, max_growth_ratio=0.2)


class TestDistilledCandidateNowPassesTheGrowthGuard:
    """The closed loop: the exact scenario that failed before the fix."""

    def test_growth_guard_rejection_is_cured_by_enforcement(self, tmp_path):
        module, skill = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(
            _bloated_body(skill["body"])
        )
        config = EvolutionConfig(hermes_agent_path=tmp_path)
        validator = ConstraintValidator(config)

        # Before the cure: raw bloated candidate fails the growth limit --
        # this is the +240% rejection observed on the live run.
        raw_results = {
            c.constraint_name: c
            for c in validator.validate_all(extract_skill_markdown(module), "skill", baseline_text=skill["raw"])
        }
        assert raw_results["growth_limit"].passed is False

        # After Ockham enforcement the SAME candidate passes every guard.
        enforce_skill_growth_budget(module, max_growth_ratio=config.max_prompt_growth)
        distilled = extract_skill_markdown(module)
        results = {
            c.constraint_name: c
            for c in validator.validate_all(distilled, "skill", baseline_text=skill["raw"])
        }
        assert all(c.passed for c in results.values()), [
            (name, c.message) for name, c in results.items() if not c.passed
        ]
        assert meets_budget(distilled, SAMPLE_SKILL, config.max_prompt_growth)
        assert NEW_RULE in distilled
        assert _md5(distilled) != _md5(SAMPLE_SKILL)  # mutation proof intact


class TestOckhamExtractMarkdownPureFunction:
    def test_within_budget_markdown_returned_untouched(self):
        md = SAMPLE_SKILL
        assert ockham_extract_markdown(md, "name: x\ndescription: y", md, 0.2) == md

    def test_frontmatter_comes_from_argument_not_body(self, tmp_path):
        module, skill = _build_module(tmp_path)
        bloated = _bloated_body(skill["body"])
        module.predictor.signature = module.predictor.signature.with_instructions(bloated)
        raw = extract_skill_markdown(module)
        distilled = ockham_extract_markdown(raw, skill["frontmatter"], SAMPLE_SKILL, 0.2)
        assert _frontmatter_block(distilled) == _frontmatter_block(SAMPLE_SKILL)
        assert len(distilled) <= int(len(SAMPLE_SKILL) * 1.2)
