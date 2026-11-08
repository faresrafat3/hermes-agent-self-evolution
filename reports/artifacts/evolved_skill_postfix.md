---
name: github
description: "GitHub via gh CLI: PRs, issues, reviews, repos, auth."
version: 2.0.0
author: Ben Barclay (benbarclay), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [github, gh, git, pull-requests, issues, code-review, repos, auth, ci]
    category: software-development
    related_skills: [codebase-inspection, requesting-code-review]
---

Work GitHub end to end with the `gh` CLI (REST fallback only via `gh api` where noted), in the user's language/dialect (e.g. Egyptian Arabic when the user writes it), and execute — do not stop at a proposal when the user says do it fully / كمل بالكامل / اعمل ال repo. Only ask a question when a value is truly blocking (account/org, visibility, final name) and is not inferable/defaultable; otherwise pick a sensible default (private repo), do the preflight, and proceed, reporting what was actually done and verified.

## Routing - ALWAYS read the matching reference before starting that workflow
| Task | Read first |
|---|---|
| Auth broken / new machine / token or SSH setup / gh login | `references/auth.md` |
| Create, triage, label, assign, close issues | `references/issues.md` |
| Branch, commit, open PR, watch CI, merge | `references/pr-workflow.md` |
| Carry an ISSUE to a verified PR (full delivery loop) | `references/issue-to-pr.md` |
| Review someone's PR: diffs, inline comments, verdict | `references/code-review.md` |
| Clone/create/fork repos, remotes, releases | `references/repo-management.md` |
Supporting assets: `scripts/gh-env.sh` + `scripts/git-credential-token.py`, `templates/` (PR bodies, bug report, feature request), `references/ci-troubleshooting.md`, `references/conventional-commits.md`, `references/github-api-cheatsheet.md`, `references/review-output-template.md`.

## Core discipline (every workflow)
- Preflight once per session: `gh auth status` — if it fails, go to `references/auth.md` before anything else. Do not claim a repo was/will be created before this passes.
- Prefer `gh` over raw REST; drop to `gh api` only for endpoints the porcelain lacks.
- Never report CI green without checking `gh pr checks` yourself; never claim merged without verifying `gh pr view` state/mergedAt; never claim a release/issue state without a fresh `gh` read, never memory.
- Read full context before writing: `gh issue view --comments` / `gh pr view --comments` — decisions live in threads, not titles.
- Sweep for duplicates before creating anything: `gh pr list --search` / `gh issue list --search`.
- The workflow's own reference defines done for that task.

## Scenario A: GitHub as central hub for multiple agents (user wants agents connected via a repo, not manual relay)
- Treat the user's GitHub profile/repos as the shared communication/record area: who did what must be visible from issues, PRs, branches, labels, and handoff files. The user is not the only agent working with them.
- Default pattern, to be actually created with `gh repo create` after auth passes (e.g. `gh repo create <name> --private --description "..."`): a private `research-council` hub with README.md, COUNCIL_CHARTER.md (roles/boundaries = most important file), BOUNDARIES.md, agents/{web-researcher,domain-expert,critic,synthesizer}.md, prompts/{round-1-broad,round-2-deepen,round-3-breakthrough}.md, templates/{research-issue,findings-pr}.md, rounds/.gitkeep. Issues = research questions, PRs = proposals only (not merged without coordinator/user decision), each round labelled.
- Agent prompt must wrap encouragement ("keep going, believe in yourself, make a breakthrough") inside hard boundaries, verbatim in spirit: research and recommendation ONLY; dig deeper / question assumptions / search contradictions to make a breakthrough in UNDERSTANDING, not implementation; do NOT write final code, merge PRs, or change architecture; do NOT act as coordinator — propose, coordinator decides; do NOT invent facts — say `unknown` + what is needed to verify; label every claim [verified]/[inference]/[guess] with source; stay in current issue scope, open new topics as questions; web agents = research only, not developer/judge. End every round with: What we know / What we don't know / Biggest risk / Recommendation + why.
- Loop: Round 1 broad findings on issue, Round 2 deepen after summary, Round 3 critic attacks conclusions then synthesizer produces proposal PR. If an agent is lost, it can re-read the repo to know its role and current round.

## Scenario B: General core vs. platform adapters, future plugins
- Do not hard-couple the project to Hermes. Separate a general, abstract core (ideas, papers, scientific/practical method, plugin contract, schemas, agent protocol, compliance tests, ADR decisions) from per-platform adapters, each with its own AGENTS.md/constitution: Hermes (current working version — continue its existing research/development path there), OpenCode, DSH, and future plugins/projects.
- Recommended concrete layout when creating them: `<name>-core` (AGENTS.md, README.md, docs/{philosophy,architecture,papers,decisions}, spec/{plugin-contract,schemas,agent-protocol}, tests/compliance, templates/), `<name>-plugin-hermes`, `<name>-plugin-opencode`, `<name>-plugin-dsh` (each adapter states which core version it implements, e.g. core-v0.1.0), and a meta/coordination hub (board issues, roadmap.md, agents/{hermes,opencode,dsh}.md, handoffs/). Freeze/import the current Hermes version into its adapter repo first so no work is lost, then extract the core.
- Attribution/governance on GitHub: every task starts as an Issue labelled `agent:hermes|agent:opencode|agent:dsh`, branch per agent e.g. `agent/hermes/issue-12-fix-auth`, every PR states agent, issue #, core version, what changed / what did not, plus a HANDOFF.md (did, left for next, next step). No merge without green `gh pr checks` and review (by another agent or the user). New plugin = new adapter implementing the same plugin-contract and passing compliance tests, not editing existing adapters. Start with a small core v0.1.0 rather than a perfect, stalled one.

## Scenario C: Code review — scoping and honesty pass (e.g. async / slop-reviewer delegation)
Deliver a verdict APPROVE or REQUEST CHANGES, specific and citing exact file and line (not vague paraphrase), never rubber-stamp. Answer concretely:
1. Does the docstring/description describe what the code actually does? Read the real function (e.g. `h5py/_hl/files.py` `File.__init__`); for each documented exception (ValueError, FileNotFoundError, FileExistsError, TypeError) verify it is actually raised on that code path and under what condition (invalid mode, missing file for r/r+, existing file for x/w-, bad argument type), and flag any misleading documented exception or any raised exception missing.
2. Is the change the right size? Compare diff to the stated problem (e.g. silent docstring + 8 lines); flag unrelated/accumulated changes, formatting churn, or scope creep.
3. Do the tests actually fail without the change? Read the test file in full (e.g. `h5py/tests/test_file_raises.py`); say plainly whether tests verify the DOCSTRING or merely the CODE — runtime API tests with no `__doc__`/`inspect.getdoc` assertion pass with or without a docstring edit and are NOT evidence for the doc edit. Record this caveat; it is not automatically a blocker for a docs-only change.
4. Does the PR body claim anything the diff does not support? Check each claim (files touched, content added, no behaviour change, no fix/performance/API claim) against the actual diff.
Apply the same four checks, adapted, to any PR review using `references/code-review.md` and `references/review-output-template.md`: read diffs, issue/PR comments, and the code before judging.

## Response quality
Be concrete and executable: give exact `gh` commands, file trees, and file contents, then run/verify them. Do not end with only 'confirm name / private-or-public / account?' when the user already authorised full execution — if you must ask, ask at most the truly blocking items in one short list and state the defaults you will use. Summarise in the user's language, distinguish [verified] remote state (checked via `gh`) from [inference]/proposal, and never add purposes, promises, owners, or deadlines not in the user's request or the repo.
