---
name: gitflow-github-sync
description: >-
  Manage safe Git branching, atomic conventional commits, .gitignore hygiene,
  and continuous GitHub synchronization across dev and feature branches without
  ever committing directly to main.
---

# Gitflow & GitHub Synchronization Protocol

Use this skill at the start and end of every development phase in `RL/RL-game1` (`antof27/RL-game1`) and when contributing upstream to `C:\Users\anton\Documents\Notelia` (`Prv2106/Notelia`).

---

## 1. Branch Topology (Never Work on `main`)

- **`main`**: Protected baseline branch. Never commit directly to `main`.
- **`dev`**: Active integration branch for verified Milestone work. Always push verified merges on `dev` to `origin/dev`.
- **`feat/<topic>` / `fix/<topic>`**: Short-lived branches created from `dev` (or from the current foundation branch) for each logical phase.

---

## 2. Phase Workflow

1. **Before Starting**: Run `git status -sb` and confirm you are on `dev` or a `feat/*` branch (never `main`).
2. **After Completing & Testing a Phase**:
   - Run all relevant automated checks (`pytest`, `validate_docs.py`, smoke checks). Never commit failing code.
   - Inspect `git status` carefully so large binaries (`*.pt`, `*.zip`, `*.db`, `tb_log/`, `runs/`, `.venv/`) are NEVER staged.
   - Stage only relevant tracked/new source, config, doc, and curated figure/video files.
   - Commit with a clear English conventional message (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
   - Push the feature branch (`git push -u origin <branch>`).
   - Merge cleanly into `dev` (`git checkout dev && git merge --no-ff <branch>`), push `dev` (`git push origin dev`), and delete the merged local/remote feature branch when done (or keep if continuing).

---

## 3. Repository Hygiene & Safety Rules

- **Never run** `git push --force`, `git reset --hard`, or `git clean -fd` without explicit user permission.
- **Preserve legacy work**: Leave the user's pre-existing modified v1 files untouched unless explicitly asked to stage or revert them.
- **Cross-Device Portability**: Always verify that `requirements.txt`, `pyproject.toml`, `configs/`, and `docs/` push cleanly so another workstation can clone `dev`, build `.venv`, and reproduce any experiment immediately.
