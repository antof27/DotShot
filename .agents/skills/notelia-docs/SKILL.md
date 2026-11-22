---
name: notelia-docs
description: >-
  Author, structure, and validate scientific handbooks and experiment logs
  using Notelia's four pillars (pedagogical progression, Scientific Humanizer,
  visual rhythm, and automated citation/link quality gates) adapted for English
  Docsify + KaTeX + Mermaid + BibTeX workflows.
---

# Notelia Docs: Scientific Handbook and Experiment Authoring

This skill adapts the architecture of **Notelia** (`Prv2106/Notelia`) to **DotShot Game**'s living English research handbook under `docs/`.

---

## 1. The Four Notelia Pillars

### Pillar 1: Rigorous Pedagogical Progression
Every chapter in `docs/` (for example `01-evolution-rl-to-marl.md` and `02-sota-marl-taxonomy.md`) must follow a clear learning arc:
1. **Learning Objectives**: 3-5 concrete outcomes answering what the reader will understand or verify by the end of the chapter.
2. **Intuition Before Formalism**: Introduce the physical, game-theoretic, or algorithmic problem with a concrete scenario (e.g., two DotShot agents blocking each other's line of fire) before writing Bellman or policy-gradient equations.
3. **Formal Definition and Derivation**: Present exact notation, assumptions, and theorems with every symbol defined inline right below the display equation.
4. **Worked Example or Concrete Mapping**: Trace how the math maps onto DotShot's Dec-POMDP state, action heads, or training update.
5. **Common Pitfalls and Failure Modes**: Document real bugs (action flattening, missing value normalization, parameter sharing across asymmetric roles, non-stationary replay buffers).
6. **Quick Self-Check**: 2-4 concise conceptual or numerical questions with collapsible `<details>` solutions at the end of theoretical chapters.

### Pillar 2: Scientific Humanizer
All prose must strictly comply with [humanizer/SKILL.md](../humanizer/SKILL.md). Keep paragraphs focused (3-5 sentences), explain *why* a design choice works before listing *what* it contains, and avoid textbook clichés.

### Pillar 3: Visual Rhythm and Formatting
Alternate short prose paragraphs with structured visual anchors:
- **Comparative Tables**: Use Markdown tables to compare algorithms across policy type, action space support, parameter sharing, monotonicity guarantees, and failure modes.
- **Mermaid Diagrams**: Use `flowchart TD/LR` or `sequenceDiagram` to illustrate Dec-POMDP loops, CTDE critic vs actor flow, and simulator architecture. Quote node labels containing parentheses or brackets (e.g. `A["Critic V(s_t)"]`).
- **Semantic Callouts**: Use GitHub-style alerts cleanly (never nest or stack two consecutively):
  - `> [!NOTE]` for Formal Definitions or Intuition
  - `> [!IMPORTANT]` for Theorems, Lemmas, and Key Invariants
  - `> [!TIP]` for Implementation & Hyperparameter Guidance
  - `> [!WARNING]` for Common Pitfalls & Failure Modes
- **KaTeX Safety**: Escape literal currency `$` as `\$` in prose or keep `$VAR` inside backticks. Keep display math `$$ ... $$` on its own lines.

### Pillar 4: Automated Quality Gate
No documentation update is finished until `scripts/validate_docs.py` passes with zero errors:
1. **Bibliography Gate**: Every citation key referenced in `docs/*.md` exists in `docs/references.bib`, and every entry in `docs/references.bib` has verified authors, title, year, venue/arXiv ID, and URL/DOI.
2. **Link & Asset Gate**: Every relative Markdown link, anchor, figure (`docs/assets/figures/*`), and video (`docs/assets/videos/*`) resolves to an existing file.
3. **Humanizer Lint Gate**: Zero banned cliché patterns, zero non-English leakage, and valid chapter structure.

---

## 2. Source Verification Policy

- **Allowed Primary Sources**: Original peer-reviewed papers (NeurIPS, ICLR, ICML, JMLR, AAMAS), arXiv preprints with verified IDs, standard textbooks (Sutton & Barto 2018; Albrecht, Christianos, & Schäfer 2024), and official framework docs (Gymnasium, PettingZoo, Stable-Baselines3, HARL).
- **Zero Fabrication Rule**: Never write an arXiv ID, year, page range, or author list from memory without checking it via search or reading the local PDF in `RL/RL-Literature/` when available. Mark any unverified detail explicitly or verify it before committing.

---

## 3. Upstream Notelia Gitflow Protocol (`C:\Users\anton\Documents\Notelia`)

When updating or contributing skills back to the local Notelia repository:
1. Check `git status` and ensure you are on `dev` (`git checkout dev && git pull origin dev`). Never commit directly to `main`.
2. Branch off `dev` as `feature/<name>` or `fix/<name>`.
3. Add both white-box tests (YAML frontmatter, link integrity, linter rules) and black-box tests (end-to-end compilation/validation).
4. Merge `--no-ff` back into `dev` only when 100% of automated tests pass, then delete the feature branch. Push only when explicitly approved by the user.
