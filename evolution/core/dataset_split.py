"""Deterministic stratified train/val/holdout splitting with a leakage guard.

Root cause this module fixes — *split bias* in the evolution benchmark:

The pipeline used to split eval examples with ``random.shuffle(...)`` then
contiguous chunks (``SyntheticDatasetBuilder.generate``,
``build_dataset_from_external``, ``GoldenDatasetLoader.load``). On the tiny
mined set (7 examples for the ``github`` skill) that produced::

    train   {repo-management x2, code-review x1}            (all medium)
    val     {repo triage and PR/issue signal analysis x1}   (medium)
    holdout {PR lifecycle / fork PR creation x1,
             PR lifecycle / issue-to-PR delivery x1,
             repo-management x1}                            (easy + medium)

i.e. the val and holdout category sets were disjoint from train, so GEPA
optimized one task distribution while "holdout generalization" measured
another. Worse, the same ``task_input`` (md5 ``0859846881``) landed verbatim in
*both* ``train`` and ``holdout`` — a data leak contaminating the baseline vs
evolved comparison. A holdout that contains a training example is not held
out at all.

:func:`stratified_split` fixes all three defects deterministically:

1. **Dedupe** — verbatim / whitespace-only duplicates of ``task_input`` are
   collapsed to their first occurrence, so no example can occupy two splits.
2. **Stratify** — examples are grouped into coarse strata
   ``(difficulty, input-length bucket, category family)`` and distributed
   across splits by a greedy *deficit assignment* against the target split
   sizes, so every split mirrors the overall difficulty / length / task-type
   mix instead of being one contiguous random block.
3. **Deterministic seed** — the per-stratum shuffle is seeded, so a given
   (examples, ratios, seed) always yields the same split. Benchmarks become
   reproducible instead of a dice roll on every run.
4. **Report** — the split ships a balance report (per-stratum counts per
   split, max share deviation) and the residual leakage list, so split
   quality is measurable and unit-tested rather than assumed.

The helpers here are pure stdlib and never import ``dataset_builder`` — the
caller wraps the returned lists in an :class:`EvalDataset`. That keeps this
module free of import cycles and trivially testable offline.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Sequence

# Input-length buckets for the stratum key. Thresholds are inclusive on the
# lower bound: len < SHORT -> "short", < LONG -> "medium", else "long".
_LENGTH_SHORT_CEIL = 200
_LENGTH_LONG_FLOOR = 800

_SPLITS = ("train", "val", "holdout")


def normalize_task(text: str) -> str:
    """Whitespace/case-insensitive canonical form for duplicate detection."""
    return " ".join((text or "").lower().split())


def task_key(example) -> str:
    """Stable id for an example: md5 of its normalized ``task_input``."""
    return hashlib.md5(normalize_task(example.task_input).encode()).hexdigest()


def length_bucket(example) -> str:
    n = len(example.task_input or "")
    if n < _LENGTH_SHORT_CEIL:
        return "short"
    if n < _LENGTH_LONG_FLOOR:
        return "medium"
    return "long"


def category_family(example) -> str:
    """Coarse task-type family from the free-text ``category`` field.

    Categories are LLM-written free text (``"repo-management"``,
    ``"repo triage and PR/issue signal analysis"``,
    ``"PR lifecycle / fork PR creation"``), so stratifying on the raw string
    gives one stratum per example. The first alphanumeric token is a usable
    coarse family: ``repo-management`` -> ``repo``,
    ``PR lifecycle ...`` -> ``pr``, ``code-review`` -> ``code``.
    """
    category = (getattr(example, "category", "") or "").strip().lower()
    if not category or category == "general":
        return "general"
    token = re.split(r"[^a-z0-9]+", category)[0]
    return token or "general"


def stratum_key(example) -> str:
    """Coarse stratification key: difficulty / length bucket / task family."""
    difficulty = (getattr(example, "difficulty", "medium") or "medium").strip().lower()
    return f"{difficulty}:{length_bucket(example)}:{category_family(example)}"


def dedupe_examples(examples: Sequence) -> tuple[list, list]:
    """Collapse duplicate ``task_input`` to their first occurrence.

    Returns ``(unique_examples, dropped)`` where ``dropped`` is the list of
    duplicate examples removed (kept for reporting / testing). Order is the
    original order; the first occurrence of each task wins.
    """
    seen: set[str] = set()
    unique: list = []
    dropped: list = []
    for example in examples:
        key = task_key(example)
        if key in seen:
            dropped.append(example)
            continue
        seen.add(key)
        unique.append(example)
    return unique, dropped


def split_targets(n: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    """Target split sizes, preserving the legacy "remainder to holdout".

    Train and val get their floored ratio (at least one each when ``n >= 3``);
    whatever remains — typically the largest leftover — goes to holdout, so
    the holdout is never the smallest split by construction.
    """
    if n < 3:
        # Too few to split meaningfully: everything trains, val/holdout empty
        # but the caller can still ask for min_holdout enforcement.
        return {"train": n, "val": 0, "holdout": 0}
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_holdout = n - n_train - n_val
    if n_holdout < 1:
        n_holdout = 1
        n_train = max(1, n - n_val - n_holdout)
    return {"train": n_train, "val": n_val, "holdout": n_holdout}


def check_leakage(
    train: Sequence, val: Sequence, holdout: Sequence
) -> list[str]:
    """Return md5 task keys that appear in more than one split.

    An empty list is the leak-free invariant the split guarantees.
    """
    owners: dict[str, str] = {}
    leaked: list[str] = []
    for split, examples in (("train", train), ("val", val), ("holdout", holdout)):
        for example in examples:
            key = task_key(example)
            if key in owners and owners[key] != split:
                leaked.append(key)
            owners[key] = split
    return sorted(set(leaked))


def balance_report(
    train: Sequence, val: Sequence, holdout: Sequence, total: int
) -> dict:
    """Per-stratum counts per split plus the max expected-share deviation.

    For each stratum the expected count in a split is
    ``stratum_total * split_size / total``; the deviation is
    ``abs(actual - expected)``. A perfectly stratified split has max
    deviation 0; larger values mean a stratum is over/under-represented in
    some split.
    """
    splits = {"train": list(train), "val": list(val), "holdout": list(holdout)}
    strata: dict[str, dict[str, int]] = {}
    for name, examples in splits.items():
        for example in examples:
            bucket = strata.setdefault(stratum_key(example), {"train": 0, "val": 0, "holdout": 0})
            bucket[name] += 1

    max_deviation = 0.0
    for key, bucket in strata.items():
        stratum_total = sum(bucket.values())
        for name in _SPLITS:
            split_size = len(splits[name])
            expected = stratum_total * split_size / total if total else 0.0
            max_deviation = max(max_deviation, abs(bucket[name] - expected))

    return {
        "strata": {
            key: {**bucket, "total": sum(bucket.values())}
            for key, bucket in sorted(strata.items())
        },
        "max_stratum_deviation": round(max_deviation, 4),
    }


@dataclass
class SplitResult:
    """A stratified split plus its audit report."""

    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    holdout: list = field(default_factory=list)
    report: dict = field(default_factory=dict)

    @property
    def dataset_kwargs(self) -> dict:
        """Kwargs ready for ``EvalDataset(train=..., val=..., holdout=...)``."""
        return {"train": self.train, "val": self.val, "holdout": self.holdout}


def stratified_split(
    examples: Sequence,
    *,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    holdout_ratio: float = 0.25,
    seed: int = 0,
    dedupe: bool = True,
    min_holdout: int = 0,
) -> SplitResult:
    """Split ``examples`` into train/val/holdout, stratified and leak-free.

    Args:
        examples: EvalExample-like objects (``task_input``, ``difficulty``,
            ``category`` attributes).
        train_ratio / val_ratio / holdout_ratio: Target proportions. The
            holdout receives any remainder (legacy behavior), so it is never
            the smallest split.
        seed: Seed for the per-stratum shuffles — same inputs, same split.
        dedupe: Collapse duplicate ``task_input`` before splitting.
        min_holdout: Floor on the holdout size. When the natural holdout is
            smaller, examples are moved from the largest split (train) until
            the floor is met, and the move is recorded in the report. The
            floor can never exceed ``len(examples) - 2``.

    Returns:
        :class:`SplitResult` with the three lists and a report containing the
        target counts, per-stratum balance, residual leakage (empty when
        clean), the number of duplicates dropped and whether the
        minimum-holdout floor was enforced.
    """
    pool = list(examples)
    dropped: list = []
    if dedupe:
        pool, dropped = dedupe_examples(pool)

    n = len(pool)
    if n == 0:
        return SplitResult(
            report={
                "counts": {"train": 0, "val": 0, "holdout": 0},
                "strata": {},
                "max_stratum_deviation": 0.0,
                "leakage": [],
                "dropped_duplicates": 0,
                "seed": seed,
                "min_holdout_enforced": False,
            }
        )

    targets = split_targets(n, train_ratio, val_ratio)

    # Group into strata; process the largest strata first so the greedy
    # deficit assignment places the bulk of the distribution first.
    by_stratum: dict[str, list] = {}
    for example in pool:
        by_stratum.setdefault(stratum_key(example), []).append(example)

    rng = random.Random(seed)
    splits: dict[str, list] = {name: [] for name in _SPLITS}
    counts = {name: 0 for name in _SPLITS}

    def _assign(example) -> None:
        # The most proportionally under-filled split that still has room:
        # minimizing count/target fills each split in the global target
        # proportions as examples arrive, so every stratum lands distributed
        # like the whole pool. (Maximizing the *absolute* remaining deficit
        # instead lets the first stratum drain train — that was the bug that
        # left a stratum over-represented in one split.) Ties break in
        # _SPLITS order (train, val, holdout).
        room = [name for name in _SPLITS if counts[name] < targets[name]]
        if not room:
            room = list(_SPLITS)
        target = min(
            room,
            key=lambda name: (counts[name] / max(1, targets[name]), _SPLITS.index(name)),
        )
        splits[target].append(example)
        counts[target] += 1

    for key in sorted(by_stratum, key=lambda k: (-len(by_stratum[k]), k)):
        group = list(by_stratum[key])
        rng.shuffle(group)
        for example in group:
            _assign(example)

    # Minimum-holdout floor: steal deterministically from the largest split.
    min_holdout_enforced = False
    if min_holdout and len(splits["holdout"]) < min_holdout:
        donor = max(_SPLITS, key=lambda name: len(splits[name]))
        while len(splits["holdout"]) < min_holdout and len(splits[donor]) > 1:
            splits["holdout"].append(splits[donor].pop())
            min_holdout_enforced = True
        # One example per other split is the floor the arithmetic allows.
        if len(splits["holdout"]) < min_holdout:
            for name in ("train", "val"):
                if len(splits[name]) > 1 and len(splits["holdout"]) < min_holdout:
                    splits["holdout"].append(splits[name].pop())
                    min_holdout_enforced = True

    # Re-sort each split into the original input order for stable, reviewable
    # files (the shuffled assignment decided membership, not order).
    order = {task_key(example): index for index, example in enumerate(examples)}
    for name in _SPLITS:
        splits[name].sort(key=lambda e: order[task_key(e)])

    leakage = check_leakage(splits["train"], splits["val"], splits["holdout"])
    report = balance_report(splits["train"], splits["val"], splits["holdout"], n)
    report.update(
        {
            "counts": {name: len(splits[name]) for name in _SPLITS},
            "leakage": leakage,
            "dropped_duplicates": len(dropped),
            "seed": seed,
            "min_holdout_enforced": min_holdout_enforced,
        }
    )

    return SplitResult(
        train=splits["train"],
        val=splits["val"],
        holdout=splits["holdout"],
        report=report,
    )


__all__ = [
    "SplitResult",
    "balance_report",
    "category_family",
    "check_leakage",
    "dedupe_examples",
    "length_bucket",
    "normalize_task",
    "split_targets",
    "stratified_split",
    "stratum_key",
    "task_key",
]
