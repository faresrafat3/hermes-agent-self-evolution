"""Tests for extracting the evolved skill markdown from a SkillModule.

These tests simulate exactly what GEPA does to a compiled SkillModule: it
rewrites ``predictor.signature.instructions`` via
``pred.signature = pred.signature.with_instructions(new_text)`` on a deep copy
of the program. They prove the rewrite is reflected in the SKILL.md content
returned by :func:`extract_skill_markdown` while the original YAML frontmatter
is preserved byte-for-byte.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

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

EVOLVED_BODY = """# Test Skill — Testing Things (Evolved)

## When to Use
Use this when you need to test things. State the invariant before acting.

## Procedure
1. Read the relevant reference file before changing anything
2. Do the thing
3. Verify it worked with a fresh command, not from memory
4. Report results with file:line citations

## Pitfalls
- Always check edge cases before finishing"""


def _build_module(tmp_path: Path, **overrides) -> SkillModule:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(SAMPLE_SKILL)
    skill = load_skill(skill_file)
    return SkillModule(
        skill["body"],
        frontmatter=skill["frontmatter"],
        raw_skill=skill["raw"],
        **overrides,
    )


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _frontmatter_block(markdown: str) -> str:
    """Return the raw text between the leading --- markers."""
    return markdown.split("---", 2)[1]


class TestSkillModuleSignatureInstructions:
    """The skill body must live where GEPA can optimize it."""

    def test_body_is_the_predictor_signature_instructions(self, tmp_path):
        module = _build_module(tmp_path)
        assert module.skill_text == _build_module(tmp_path).original_body
        assert module.predictor.signature.instructions == module.original_body

    def test_named_predictors_exposes_instructions_for_gepa(self, tmp_path):
        module = _build_module(tmp_path)
        predictors = list(module.named_predictors())
        assert len(predictors) == 1
        name, predictor = predictors[0]
        assert name == "predictor"
        assert predictor.signature.instructions == module.original_body

    def test_skill_text_property_tracks_signature(self, tmp_path):
        module = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(EVOLVED_BODY)
        assert module.skill_text == EVOLVED_BODY


class TestExtractSkillMarkdownUnchanged:
    def test_unchanged_returns_original_raw_byte_for_byte(self, tmp_path):
        module = _build_module(tmp_path)
        extracted = extract_skill_markdown(module)
        assert extracted == SAMPLE_SKILL
        assert _md5(extracted) == _md5(SAMPLE_SKILL)

    def test_edge_whitespace_is_normalized_not_treated_as_evolution(self, tmp_path):
        """DSPy trims edge whitespace in signature instructions on assignment.

        That normalization alone must not count as an evolved body, so the
        original raw file is still returned verbatim.
        """
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)
        module = SkillModule(
            f"\n\n{skill['body']}\n\n",
            frontmatter=skill["frontmatter"],
            raw_skill=skill["raw"],
        )
        assert extract_skill_markdown(module) == SAMPLE_SKILL


class TestExtractSkillMarkdownEvolved:
    def _evolved(self, tmp_path) -> tuple[SkillModule, str]:
        module = _build_module(tmp_path)
        module.predictor.signature = module.predictor.signature.with_instructions(EVOLVED_BODY)
        return module, extract_skill_markdown(module)

    def test_evolved_instructions_produce_different_markdown(self, tmp_path):
        module, extracted = self._evolved(tmp_path)
        assert module.skill_text == EVOLVED_BODY
        assert extracted != SAMPLE_SKILL
        assert _md5(extracted) != _md5(SAMPLE_SKILL)

    def test_evolved_body_replaces_original_body(self, tmp_path):
        _, extracted = self._evolved(tmp_path)
        assert "State the invariant before acting" in extracted
        assert "file:line citations" in extracted
        # A distinctive line from the original body must now be gone.
        assert "Don't forget to check edge cases" not in extracted
        assert "First, do the thing" not in extracted

    def test_frontmatter_block_is_preserved_exactly(self, tmp_path):
        _, extracted = self._evolved(tmp_path)
        assert _frontmatter_block(extracted) == _frontmatter_block(SAMPLE_SKILL)

    def test_frontmatter_remains_valid_yaml_with_original_fields(self, tmp_path):
        _, extracted = self._evolved(tmp_path)
        parsed = yaml.safe_load(_frontmatter_block(extracted))
        assert parsed["name"] == "test-skill"
        assert parsed["description"] == "A skill for testing things"
        assert parsed["version"] == "1.0.0"
        assert parsed["metadata"] == {"hermes": {"tags": ["testing"]}}


class TestExtractSkillMarkdownAfterGepaStyleCompilation:
    """Mirror DspyAdapter.build_program: deepcopy, then rewrite instructions."""

    def test_deepcopy_and_instruction_update_flow_into_markdown(self, tmp_path):
        baseline = _build_module(tmp_path)
        optimized = baseline.deepcopy()

        for name, predictor in optimized.named_predictors():
            assert name == "predictor"
            predictor.signature = predictor.signature.with_instructions(EVOLVED_BODY)

        extracted = extract_skill_markdown(optimized)
        assert optimized.skill_text == EVOLVED_BODY
        assert _md5(extracted) != _md5(SAMPLE_SKILL)
        assert "State the invariant before acting" in extracted
        assert _frontmatter_block(extracted) == _frontmatter_block(SAMPLE_SKILL)
        # The baseline module must be untouched by the copy's rewrite.
        assert baseline.skill_text == baseline.original_body
        assert extract_skill_markdown(baseline) == SAMPLE_SKILL


class TestExtractSkillMarkdownThroughGepaAdapter:
    """Exercise the real GEPA write-back path (``DspyAdapter.build_program``).

    GEPA builds every candidate by deep-copying the student and rewriting each
    named predictor's signature instructions. This test drives that exact code
    path, so the extraction logic is checked against what ``compile`` returns.
    """

    def test_adapter_build_program_instructions_flow_into_markdown(self, tmp_path):
        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            return 1.0

        baseline = _build_module(tmp_path)
        # The candidate GEPA seeds itself with: current instructions per predictor.
        seed = {
            name: predictor.signature.instructions
            for name, predictor in baseline.named_predictors()
        }
        assert seed == {"predictor": baseline.original_body}

        adapter = DspyAdapter(
            student_module=baseline,
            metric_fn=metric,
            feedback_map={},
            reflection_lm=None,
        )
        optimized = adapter.build_program({"predictor": EVOLVED_BODY})
        extracted = extract_skill_markdown(optimized)

        assert optimized.skill_text == EVOLVED_BODY
        assert extracted != SAMPLE_SKILL
        assert _md5(extracted) != _md5(SAMPLE_SKILL)
        assert "State the invariant before acting" in extracted
        assert "Don't forget to check edge cases" not in extracted
        assert _frontmatter_block(extracted) == _frontmatter_block(SAMPLE_SKILL)
        assert yaml.safe_load(_frontmatter_block(extracted))["name"] == "test-skill"
        # Writing into the copy must not disturb the baseline module.
        assert extract_skill_markdown(baseline) == SAMPLE_SKILL


class TestExtractSkillMarkdownWithoutRawSkill:
    def test_changed_instructions_reassemble_with_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)
        module = SkillModule(skill["body"], frontmatter=skill["frontmatter"])

        module.predictor.signature = module.predictor.signature.with_instructions(EVOLVED_BODY)
        extracted = extract_skill_markdown(module)

        assert "State the invariant before acting" in extracted
        assert _frontmatter_block(extracted) == _frontmatter_block(SAMPLE_SKILL)
        assert yaml.safe_load(_frontmatter_block(extracted))["name"] == "test-skill"


class TestExtractSkillMarkdownOverrides:
    def test_explicit_frontmatter_and_raw_override_are_used(self, tmp_path):
        module = _build_module(tmp_path)
        other_frontmatter = "name: other\ndescription: Something else"
        other_raw = "---\nname: other\n---\n\n# Unrelated\n"

        # Changed instructions + explicit frontmatter.
        module.predictor.signature = module.predictor.signature.with_instructions(EVOLVED_BODY)
        extracted = extract_skill_markdown(module, frontmatter=other_frontmatter)
        assert extracted.startswith(f"---\n{other_frontmatter}\n---\n")

        # Unchanged instructions + explicit raw.
        module = _build_module(tmp_path)
        assert extract_skill_markdown(module, original_raw=other_raw) == other_raw
