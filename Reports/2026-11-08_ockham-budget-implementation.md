# Ockham Budget Enforcement & Compression — Implementation Report

**Repo:** `/home/fares/.hermes/self-evolution` (`main`)
**Date:** 2026-11-08
**Incident closed:** GEPA additive bloat (+240%, 2,036 → 7,894 chars) colliding with the 20% growth guard; inflated-but-better candidates were discarded wholesale.

## The root cause

The growth guard lived *after* the optimizer. `ConstraintValidator._check_growth`
(`evolution/core/constraints.py`) ran only at validation time, so an evolved
candidate that was better on the subsample but fatter than
`config.max_prompt_growth = 0.2` was rejected outright — free-model quota burned,
nothing deployed. The proposer in DSPy is additively biased: it restates
procedures verbosely instead of adding terse rules.

## The cure (deterministic, zero-cost, offline)

New module **`evolution/skills/ockham.py`**:

| Function | Role |
|---|---|
| `compress_skill_body(body, target_len)` | Deterministic distiller. Stage 1: dedupe repeated lines + collapse blank runs. Stage 2: drop longest filler lines first (bloat is verbose, rules are terse). Stage 3: hard cut guarantee. Never drops headings; pure function; output always `≤ target_len`. |
| `enforce_skill_growth_budget(module, max_growth_ratio, baseline_text)` | Measures the evolved markdown against the baseline; when over cap, distills the module's signature instructions **in place** and returns a report dict (`applied`, before/after sizes, before/after growths). |
| `ockham_extract_markdown(...)` | Same enforcement as a pure function on a markdown string; frontmatter travels verbatim. |

Wiring:

- `extract_skill_markdown(..., max_growth_ratio=..., baseline_text=...)` —
  extraction-time budget guard. Byte-identical return for unchanged modules is
  preserved (existing behavior untouched when the flag is absent).
- `evolve()` — enforces the budget **before** constraint validation, prints the
  distillation notice, and records six `ockham_*` fields in `metrics.json`.

No revert anywhere: the first occurrence and every unique (new) line survive,
so evolved rules persist in compressed form and the MD5 still changes.

## Proof

**Unit tests (new):** `tests/skills/test_ockham_budget.py` — 22 tests covering:

- bloat (+200%-class) compressed into the 20% budget with the new rule intact,
  frontmatter byte-identical and valid YAML, MD5 changed (no full revert),
- within-budget / unchanged modules untouched (`applied=False`),
- custom budgets and the zero-budget edge,
- extraction-time enforcement and the missing-baseline `ValueError`,
- **closed loop:** the exact candidate that the growth guard rejected before
  now passes *every* constraint after enforcement,
- a new rule appended at the tail survives the longest-first trim (worst case).

**Full suite:**

```
env -u PYTHONPATH .venv/bin/pytest -q
202 passed in 0.95s
```

180 (baseline) → **202**, 100% green.

**Live reproduction at incident scale** (scripted, then deleted):

```
baseline raw chars: 1431
bloated chars: 7433 | growth: +419.4% | budget ok? False
applied: True | 7433 -> 1532 chars
after growth: +7.1% | budget ok? True
tail rule survived: True
frontmatter intact: True
md5: 0d18d08fd058ecbe576fb44b4b59bfd7 | changed: True
```

## What's left honest

- The distiller is deterministic text compaction — it removes repetition and
  verbose restatement, it does not semantically rewrite. That is deliberate:
  zero quota, fully reproducible, and sufficient for additive-bloat output.
- An optional LLM-based compressor can be layered later through the same
  `enforce_skill_growth_budget` seam if restatement-without-duplication
  (pure paraphrase bloat) ever shows up.
- The overfit on the tiny subsample (holdout −0.0187) is a data-size issue
  (`eval_dataset_size: 15`), orthogonal to the guard — worth its own pass.
