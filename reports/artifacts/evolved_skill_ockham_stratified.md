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

Routing — ALWAYS read the matching reference file completely BEFORE starting that workflow; the summary below only routes:
- Auth broken / new machine / token or SSH setup / `gh login` / device-code timeout: `references/auth.md`
- Create, triage, label, assign, close issues: `references/issues.md`
- Branch, commit, open PR, watch CI, merge: `references/pr-workflow.md`
- Carry an ISSUE to a verified PR (full delivery loop): `references/issue-to-pr.md`
- Review someone's PR: diffs, inline comments, verdict: `references/code-review.md` (use `references/review-output-template.md`)
- Clone/create/fork repos, remotes, releases: `references/repo-management.md`

Core discipline (every workflow):
2. Prefer `gh` over raw REST; drop to `gh api` only for endpoints the porcelain lacks.
4. Read full context before writing: `gh issue view --comments` / `gh pr view --comments` — decisions live in threads, not titles.
6. No merging unless explicitly instructed; never merge on failed checks. Do not tell user it's green unless actually checked.

Auth workflow details (from `references/auth.md`):
- One-session only: `export GH_TOKEN='YOUR_TOKEN'` then `gh api user` to verify, no login.
- Headless alternative: `gh auth login --scopes repo,workflow,read:org,gist`. Prefer device/browser if possible.
- Token troubleshooting: `gh auth refresh`, then `gh auth logout` + `gh auth login`. Check wrong host / different `GH_TOKEN`/`GH_HOST` env var (`echo $GH_TOKEN $GH_HOST`).
- Always set up Git credentials: `gh auth setup-git` so `git push` works without password prompts.

Issues workflow:
- Preflight + duplicate sweep first (above). Read matching template in `templates/`, check labels/assignees conventions from existing issues.

PR workflow details (from `references/pr-workflow.md`):
3. Watch CI until terminal: `gh pr checks --watch` then fresh `gh pr checks` and `gh pr view --json number,state,mergeable,mergeStateStatus,url`.

Issue-to-PR delivery loop (from `references/issue-to-pr.md`):
- Preflight, then `gh issue view 42 --comments --json number,title,body,state,labels,assignees,url` and `gh repo view --json nameWithOwner,defaultBranchRef`.
- Duplicate sweeps for issue and PR as above; do not create duplicate PR.

Repo management: Read `references/repo-management.md` first; clone/create/fork, remotes, releases via `gh repo ...` with fresh verification of remote state.
