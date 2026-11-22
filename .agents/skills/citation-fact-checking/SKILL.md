---
name: citation-fact-checking
description: >-
  Verify every RL and MARL academic citation against primary sources (arXiv,
  peer-reviewed proceedings, JMLR, textbooks, or local PDFs), maintain a clean
  docs/references.bib, and block unverified claims before commits.
---

# Academic Citation & Fact-Checking Protocol

Use this skill whenever writing or editing literature reviews, equations, theorem attributions, or benchmark comparisons in `docs/` or `.agents/docs/`.

---

## 1. Zero-Fabrication Invariant

Never cite a paper, arXiv ID, author list, venue, or publication year from memory alone unless verified during the current session against:
1. Local PDFs in `RL/RL-Literature/` (for example, `1506.02438v6.pdf`, `1707.06347v2.pdf`, `1803.11485v2.pdf`), or
2. Primary web search / official proceedings (`neurips.cc`, `openreview.net`, `proceedings.mlr.press`, `jmlr.org`, `arxiv.org`).

If any detail (exact page range, co-author spelling, conference year vs arXiv preprint year) is uncertain, verify it first or omit the unverified detail.

---

## 2. BibTeX Key & Entry Standards (`docs/references.bib`)

- **Key format**: `<firstauthor><year><shortkeyword>` in lowercase ASCII (for example: `sutton1999policy`, `schulman2017ppo`, `rashid2018qmix`, `yu2022mappo`, `kuba2022happo`, `wen2022mat`, `zhong2024harl`, `amato2024ctde`, `agarwal2021rliable`).
- **Required fields**: `author`, `title`, `year`, `booktitle` or `journal`, and `eprint`/`url` (with exact arXiv ID or proceedings URL).
- **Preprint vs Conference Year**: Always check whether a landmark paper appeared first on arXiv in year $Y$ and at NeurIPS/ICLR/ICML in year $Y+1$ (for example, Kuba et al. arXiv:2109.11251 -> ICLR 2022; Yu et al. arXiv:2103.01955 -> NeurIPS 2022; Rashid et al. ICML 2018). State the peer-reviewed venue year consistently and include the arXiv ID in `eprint`.

---

## 3. Inline Citation Syntax & Automated Gate

In `docs/*.md`, cite references consistently using bracketed BibTeX keys such as `[@yu2022mappo]` or human-readable author-year paired with the key (for example, `Yu et al. (2022) [@yu2022mappo]`), or a clean custom syntax parsed deterministically by `scripts/validate_docs.py`.

Before any documentation commit:
1. Run `.\.venv\Scripts\python.exe scripts/validate_docs.py`.
2. Confirm `0` undefined citation keys and `0` broken BibTeX entries.
