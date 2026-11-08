"""Tests for the deterministic stratified split (the split-bias fix).

Incident these tests close: the mined ``github`` eval set (7 examples) was
split by ``random.shuffle`` + contiguous chunks, which (a) put the same
``task_input`` (md5 ``0859846881``) verbatim in both ``train`` and
``holdout`` — a data leak contaminating the baseline comparison — and (b)
left val/holdout with task-type sets disjoint from train, so GEPA optimized
one distribution while "holdout generalization" measured another, and the
holdout regressed by -0.0187.

The fix is :func:`evolution.core.dataset_split.stratified_split`: dedupe,
stratify by difficulty / input-length / task family, seeded, with a balance
report and a leakage check. The dual acceptance gate
(:func:`evolution.skills.evolve_skill.acceptance_gate`) then refuses any
candidate that buys valset gains by regressing on the balanced holdout.

Every test here is hermetic — no network, no LM, deterministic input.
"""

from __future__ import annotations

import pytest

from evolution.core.dataset_builder import EvalExample
from evolution.core.dataset_split import (
    balance_report,
    category_family,
    check_leakage,
    dedupe_examples,
    length_bucket,
    normalize_task,
    split_targets,
    stratified_split,
    stratum_key,
    task_key,
)


def _ex(task, difficulty="medium", category="general", source="synthetic"):
    return EvalExample(
        task_input=task,
        expected_behavior="do the thing",
        difficulty=difficulty,
        category=category,
        source=source,
    )


def _task(tag: str, i: int, length: int) -> str:
    """A distinct task_input of (at least) ``length`` chars."""
    prefix = f"{tag} task {i} "
    return prefix + "y" * max(0, length - len(prefix))


def _pool(n=40):
    """Two equal strata: (easy, short, repo) and (hard, long, pr)."""
    return (
        [_ex(_task("repo", i, 60), "easy", "repo-management") for i in range(n // 2)]
        + [_ex(_task("pr", i, 900), "hard", "PR lifecycle") for i in range(n // 2)]
    )


class TestStratumHelpers:
    def test_length_bucket_boundaries(self):
        assert length_bucket(_ex("x" * 199)) == "short"
        assert length_bucket(_ex("x" * 200)) == "medium"
        assert length_bucket(_ex("x" * 799)) == "medium"
        assert length_bucket(_ex("x" * 800)) == "long"

    def test_category_family_is_coarse(self):
        assert category_family(_ex("t", category="repo-management")) == "repo"
        assert category_family(_ex("t", category="PR lifecycle / fork PR creation")) == "pr"
        assert category_family(_ex("t", category="code-review")) == "code"
        assert category_family(_ex("t", category="")) == "general"
        assert category_family(_ex("t", category="general")) == "general"

    def test_stratum_key_combines_all_three_axes(self):
        ex = _ex("x" * 900, "hard", "PR lifecycle")
        assert stratum_key(ex) == "hard:long:pr"

    def test_normalize_collapses_whitespace_and_case(self):
        assert normalize_task("  Do   the\nThing ") == "do the thing"

    def test_task_key_stable_and_whitespace_insensitive(self):
        assert task_key(_ex("Do the thing")) == task_key(_ex("do   the  thing"))
        assert task_key(_ex("a")) != task_key(_ex("b"))


class TestDedupe:
    def test_collapses_verbatim_duplicates_keeping_first(self):
        pool = [_ex("same task", category="a"), _ex("other", category="b"), _ex("same task", category="c")]
        unique, dropped = dedupe_examples(pool)
        assert len(unique) == 2
        assert len(dropped) == 1
        assert unique[0].category == "a"  # first occurrence wins
        assert dropped[0].category == "c"

    def test_collapses_whitespace_only_duplicates(self):
        pool = [_ex("Do  the thing"), _ex("do the thing")]
        unique, _ = dedupe_examples(pool)
        assert len(unique) == 1

    def test_distinct_tasks_all_survive(self):
        pool = [_ex(f"task {i}") for i in range(10)]
        unique, dropped = dedupe_examples(pool)
        assert len(unique) == 10 and dropped == []


class TestSplitTargets:
    def test_legacy_remainder_to_holdout(self):
        # Matches the pre-fix arithmetic: train/val floored, remainder to holdout.
        assert split_targets(7, 0.5, 0.25) == {"train": 3, "val": 1, "holdout": 3}
        assert split_targets(40, 0.5, 0.25) == {"train": 20, "val": 10, "holdout": 10}
        assert split_targets(10, 0.5, 0.25) == {"train": 5, "val": 2, "holdout": 3}

    def test_tiny_pool_gives_only_train(self):
        assert split_targets(2, 0.5, 0.25) == {"train": 2, "val": 0, "holdout": 0}


class TestStratifiedSplit:
    def test_deterministic_per_seed(self):
        pool = _pool()
        a = stratified_split(pool, seed=7)
        b = stratified_split(pool, seed=7)
        assert [task_key(e) for e in a.train] == [task_key(e) for e in b.train]
        assert [task_key(e) for e in a.val] == [task_key(e) for e in b.val]
        assert [task_key(e) for e in a.holdout] == [task_key(e) for e in b.holdout]

    def test_sizes_follow_targets_and_pool_is_conserved(self):
        pool = _pool(40)
        result = stratified_split(pool, seed=0)
        assert result.report["counts"] == {"train": 20, "val": 10, "holdout": 10}
        assert len(result.train) + len(result.val) + len(result.holdout) == 40

    def test_no_leakage_even_with_duplicates_in_the_pool(self):
        pool = _pool(20)
        pool.append(pool[3])  # verbatim duplicate of a pool member
        pool.append(pool[15])
        result = stratified_split(pool, seed=0)
        assert result.report["leakage"] == []
        assert result.report["dropped_duplicates"] == 2
        assert check_leakage(result.train, result.val, result.holdout) == []

    def test_balanced_strata_mirror_the_pool(self):
        pool = _pool(40)
        result = stratified_split(pool, seed=0)
        report = result.report
        # Each stratum split exactly in the 20/10/10 target proportions.
        assert report["max_stratum_deviation"] == 0.0
        expected = {"easy:short:repo": (10, 5, 5), "hard:long:pr": (10, 5, 5)}
        for key, (train, val, holdout) in expected.items():
            bucket = report["strata"][key]
            assert (bucket["train"], bucket["val"], bucket["holdout"]) == (train, val, holdout)

    def test_every_split_mirrors_each_stratum_not_just_the_total(self):
        # The pre-fix bug: val/holdout carried task types absent from train.
        pool = _pool(40)
        result = stratified_split(pool, seed=3)
        for bucket in result.report["strata"].values():
            for split in ("train", "val", "holdout"):
                assert bucket[split] > 0

    def test_min_holdout_floor_is_enforced_from_train(self):
        pool = _pool(40)
        result = stratified_split(pool, seed=0, min_holdout=15)
        assert len(result.holdout) >= 15
        assert result.report["min_holdout_enforced"] is True
        # The floor is met by moving from train, not by inflating the pool.
        assert len(result.train) + len(result.val) + len(result.holdout) == 40

    def test_empty_pool_returns_empty_split_with_clean_report(self):
        result = stratified_split([], seed=0)
        assert result.train == [] and result.val == [] and result.holdout == []
        assert result.report["leakage"] == []

    def test_seed_changes_membership_but_never_the_sizes(self):
        pool = _pool(40)
        a = stratified_split(pool, seed=1)
        b = stratified_split(pool, seed=2)
        assert a.report["counts"] == b.report["counts"]
        assert a.report["leakage"] == b.report["leakage"] == []


class TestCheckLeakage:
    def test_detects_a_shared_task_across_splits(self):
        shared = _ex("shared task")
        leaked = check_leakage([_ex("a"), shared], [_ex("b")], [shared])
        assert leaked == [task_key(shared)]

    def test_disjoint_splits_are_clean(self):
        assert check_leakage([_ex("a")], [_ex("b")], [_ex("c")]) == []


class TestBalanceReport:
    def test_reports_per_stratum_counts(self):
        pool = _pool(6)
        result = stratified_split(pool, seed=0)
        report = balance_report(result.train, result.val, result.holdout, len(pool))
        assert set(report["strata"]) == {"easy:short:repo", "hard:long:pr"}
        for bucket in report["strata"].values():
            assert bucket["train"] + bucket["val"] + bucket["holdout"] == bucket["total"]


class TestAcceptanceGate:
    """The dual gate: Ockham budget AND non-negative holdout. Val is reported
    as a diagnostic; it does not block deployment (it is GEPA's selection
    criterion, already spent before distillation)."""

    def _gate(self, **overrides):
        from evolution.skills.evolve_skill import acceptance_gate

        kwargs = dict(
            val_before=0.50,
            val_after=0.60,
            holdout_before=0.50,
            holdout_after=0.55,
            budget_ok=True,
        )
        kwargs.update(overrides)
        return acceptance_gate(**kwargs)

    def test_accepts_when_both_conditions_hold(self):
        gate = self._gate()
        assert gate["accepted"] is True
        assert gate["budget_ok"] and gate["holdout_nonnegative"]

    def test_refuses_when_over_the_ockham_budget(self):
        gate = self._gate(budget_ok=False)
        assert gate["accepted"] is False
        assert gate["budget_ok"] is False

    def test_val_regression_alone_does_not_block_deployment(self):
        # Distillation can trade selection-set verbosity for generalization:
        # val falls while the balanced holdout improves. The gate must accept
        # (budget + holdout hold) and merely *report* the val diagnostic.
        gate = self._gate(val_after=0.48, holdout_after=0.55)
        assert gate["val_improved"] is False
        assert gate["accepted"] is True

    def test_refuses_holdout_regression_despite_val_gain(self):
        # The covariate-shift case the gate exists to catch: +val, -holdout.
        gate = self._gate(holdout_after=0.48)
        assert gate["accepted"] is False
        assert gate["holdout_nonnegative"] is False
        assert gate["holdout_delta"] == pytest.approx(-0.02)

    def test_refuses_when_both_budget_and_holdout_fail(self):
        gate = self._gate(budget_ok=False, holdout_after=0.40)
        assert gate["accepted"] is False

    def test_tolerance_allows_measurement_noise(self):
        gate = self._gate(holdout_after=0.495, holdout_tolerance=0.01)
        assert gate["accepted"] is True
        assert gate["holdout_nonnegative"] is True

    def test_zero_holdout_change_is_nonnegative(self):
        gate = self._gate(holdout_after=0.50)
        assert gate["holdout_nonnegative"] is True
        assert gate["accepted"] is True
