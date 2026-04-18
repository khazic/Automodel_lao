# Claude Code Guidelines

## Language

- **English only.** All code, comments, docstrings, variable names, commit messages, and PR descriptions must be in English. No other languages anywhere.

## PR & Commit Rules

1. **No Claude co-author tag.** Never add `Co-Authored-By: Claude ...` to any commit message. Commits are solely the user's.

2. **Always sign off commits.** Every commit must include `Signed-off-by: khazic <khazzz1c@gmail.com>` (required for DCO check).

3. **Branch from `upstream/main`, not local `main`.** The local `main` may be ahead of upstream. Always run `git fetch upstream` and branch from `upstream/main` to avoid including unrelated commits in a PR.

4. **After creating a PR, run CI checks.** Run `gh pr checks <PR_NUMBER> --repo NVIDIA-NeMo/Automodel` and wait until all checks pass before reporting done. If any check fails, investigate and fix it.
