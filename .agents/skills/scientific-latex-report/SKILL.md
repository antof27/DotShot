---
name: scientific-latex-report
description: >-
  Author publication-grade English LaTeX scientific reports (.tex) with
  booktabs tables, verified BibTeX citations, crisp figures and in-game
  screenshots, Humanizer prose, and automated structural validation.
---

# Scientific LaTeX Report Authoring (100% English)

Use this skill when writing or updating `docs/paper/*.tex` reports for **DotShot Game v2**.

---

## 1. Document Architecture and Packages

Use clean standard `article` layout (`11pt`, `margin=1in`) with reproducible compilation compatibility (`pdflatex`/`lualatex` + `bibtex`):
- `amsmath`, `amssymb`, `amsthm` for definitions, lemmas, and aligned derivations.
- `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) for all quantitative and comparative tables; never use vertical grid lines.
- `graphicx` with clean relative paths (`../assets/screenshots/*.png`, `../assets/figures/*.png`) and analytical captions stating *what* the reader should observe.
- `hyperref` and `natbib` (or `\bibliographystyle{plainnat}` + `\bibliography{../references}`) for verified citations.

---

## 2. Prose, Citation, and Number Integrity

1. **Humanizer Discipline**: Apply [humanizer/SKILL.md](../humanizer/SKILL.md) across every section. Avoid inflated claims, cliché openings, forced rule-of-three phrases, and decorative em-dashes.
2. **Exact Citation Parity**: Every `\cite{key}` or `\citep{key}` must match an exact key in `docs/references.bib`. Never invent or alter keys, author names, or years.
3. **Observed Data Only**: Every table cell, confidence interval, step budget, and win rate must come directly from `benchmarks/eval/evaluate.py` stdout or logged `metrics.jsonl` files. Label smoke-test runs explicitly as preliminary Milestone 1 baselines.
4. **Balanced Quantitative + Qualitative Analysis**: Pair every table and learning-curve plot with behavioral analysis explaining *why* agents behaved as observed (aiming dynamics, cover usage, credit assignment, action-space entropy, and truncation horizon).

---

## 3. Automated Pre-Commit Quality Gate

Before accepting any `.tex` file, verify via `scripts/validate_docs.py`:
- Every `\cite{...}` key resolves in `docs/references.bib`.
- Every `\includegraphics{...}` target resolves on disk.
- Every `\begin{env}` matches an `\end{env}`.
- Zero Humanizer cliché lint hits or non-English text leakage.
