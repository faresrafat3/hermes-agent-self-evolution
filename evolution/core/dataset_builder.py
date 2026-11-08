"""Evaluation dataset generation for hermes-agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_split import normalize_task, stratified_split
from evolution.core.openrouter_rotation import make_rotating_lm


@dataclass
class EvalExample:
    """A single evaluation example."""
    task_input: str  # What the user asks
    expected_behavior: str  # Rubric — what a good response looks like
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # Category for stratified eval
    source: str = "synthetic"  # synthetic, sessiondb, golden

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalDataset:
    """Train/val/holdout split of evaluation examples."""
    train: list[EvalExample] = field(default_factory=list)
    val: list[EvalExample] = field(default_factory=list)
    holdout: list[EvalExample] = field(default_factory=list)

    @property
    def all_examples(self) -> list[EvalExample]:
        return self.train + self.val + self.holdout

    def save(self, path: Path):
        """Save dataset splits to JSONL files."""
        path.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            with open(path / f"{split_name}.jsonl", "w") as f:
                for ex in split_data:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """Load dataset splits from JSONL files."""
        dataset = cls()
        for split_name in ["train", "val", "holdout"]:
            split_file = path / f"{split_name}.jsonl"
            if split_file.exists():
                examples = []
                with open(split_file) as f:
                    for line in f:
                        if line.strip():
                            examples.append(EvalExample.from_dict(json.loads(line)))
                setattr(dataset, split_name, examples)
        return dataset

    def to_dspy_examples(self, split: str = "train") -> list[dspy.Example]:
        """Convert a split to DSPy Example objects."""
        data = getattr(self, split)
        return [
            dspy.Example(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("task_input")
            for ex in data
        ]


def _parse_test_cases(raw: str) -> list[dict]:
    """Parse the judge's JSON array of test cases, tolerating wrapped output."""
    try:
        cases = json.loads(raw)
        if isinstance(cases, list):
            return cases
    except json.JSONDecodeError:
        pass
    import re
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        cases = json.loads(match.group())
        if isinstance(cases, list):
            return cases
    raise ValueError(f"Could not parse test cases from LLM output: {raw[:200]}")


class SyntheticDatasetBuilder:
    """Generate evaluation datasets using a strong LLM.

    Reads the target artifact (skill file, tool description, etc.)
    and generates realistic (task_input, expected_behavior) pairs.
    """

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for an agent skill or tool.

        Given the full text of a skill/tool description, generate diverse test cases
        that would exercise different aspects of the skill. Each test case should include:
        - A realistic task_input (what a user would actually ask)
        - An expected_behavior rubric (what a good response should contain/do, NOT exact text)
        - A difficulty level (easy, medium, hard)
        - A category (what aspect of the skill this tests)

        The ``focus`` field names the aspect this batch should emphasize. It
        exists because batched generation must receive a *different* prompt
        per batch: an identical prompt is served from DSPy's result cache, so
        every batch after the first comes back as a verbatim copy of the first
        and the dedupe step then collapses the whole dataset back down to one
        batch.
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        focus: str = dspy.InputField(desc="Specific aspect to emphasize in this batch (e.g. 'authentication setup', 'PR creation', 'edge cases', 'general coverage'). Vary this across batches so batched generation yields diverse, non-repeated cases.")
        test_cases: str = dspy.OutputField(desc="JSON array of test cases, each with: task_input, expected_behavior, difficulty, category")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset with train/val/holdout splits.

        Cases are generated in batches of ``BATCH_CASES``: asking the judge
        for a large ``num_cases`` in a single call yields truncated or
        degenerate JSON, so a big target (the "expand the available sample"
        option) is built from several compliant calls. The pool is deduped
        and split with :func:`evolution.core.dataset_split.stratified_split`
        — stratified by difficulty / input-length / task family, deterministic
        by ``config.split_seed``, and guarded against train/val/holdout
        leakage.
        """
        n = num_cases or self.config.eval_dataset_size

        # Configure DSPy to use the judge model for generation
        fallback = (
            self.config.fallback_models[0]
            if getattr(self.config, "fallback_models", None)
            and self.config.fallback_models[0] != self.config.judge_model
            else None
        )
        lm = make_rotating_lm(
            self.config.judge_model,
            fallback_model=fallback,
            round_robin=getattr(self.config, "openrouter_round_robin", True),
        )

        examples: list[EvalExample] = []
        seen: set[str] = set()
        batch = 15
        # Rotating emphases so each batch produces a different slice of the
        # artifact's surface instead of a cached copy of the first batch: an
        # identical prompt is served from DSPy's result cache, which would
        # hand back a verbatim copy that the dedupe step then collapses.
        _foci = [
            "general coverage of the artifact's main workflows",
            "authentication, setup, and configuration",
            "creating and managing issues, milestones, and labels",
            "pull-request creation and update workflows",
            "code review and reviewer feedback",
            "merging, closing, releasing, and versioning",
            "edge cases, errors, and failure modes",
            "API fallback and tool-preference decisions",
        ]
        # Tolerate a short batch with a bounded number of extra attempts;
        # never loop forever on a judge that keeps coming up short.
        max_attempts = max(1, math.ceil(n / batch) + 2)
        for attempt in range(max_attempts):
            if len(examples) >= n:
                break
            want = min(batch, n - len(examples))
            with dspy.context(lm=lm):
                result = self.generator(
                    artifact_text=artifact_text,
                    artifact_type=artifact_type,
                    num_cases=want,
                    focus=_foci[attempt % len(_foci)],
                )
            for c in _parse_test_cases(result.test_cases):
                task_input = c.get("task_input", "")
                expected_behavior = c.get("expected_behavior", "")
                if not task_input or not expected_behavior:
                    continue
                key = normalize_task(task_input)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(
                    EvalExample(
                        task_input=task_input,
                        expected_behavior=expected_behavior,
                        difficulty=c.get("difficulty", "medium"),
                        category=c.get("category", "general"),
                        source="synthetic",
                    )
                )

        result = stratified_split(
            examples,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            holdout_ratio=self.config.holdout_ratio,
            seed=self.config.split_seed,
            dedupe=True,
            min_holdout=self.config.min_holdout_examples,
        )
        return EvalDataset(**result.dataset_kwargs)


class GoldenDatasetLoader:
    """Load hand-curated evaluation datasets from JSONL files."""

    @staticmethod
    def load(path: Path) -> EvalDataset:
        """Load a golden dataset. If no splits exist, auto-split the single file."""
        if (path / "train.jsonl").exists():
            return EvalDataset.load(path)

        # Single file — auto-split
        golden_file = path if path.suffix == ".jsonl" else path / "golden.jsonl"
        if not golden_file.exists():
            raise FileNotFoundError(f"No golden dataset found at {golden_file}")

        examples = []
        with open(golden_file) as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        result = stratified_split(
            examples,
            train_ratio=0.5,
            val_ratio=0.25,
            holdout_ratio=0.25,
            seed=0,
            dedupe=True,
        )
        return EvalDataset(**result.dataset_kwargs)
