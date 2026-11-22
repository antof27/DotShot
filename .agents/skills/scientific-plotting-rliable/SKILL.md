---
name: scientific-plotting-rliable
description: >-
  Compute Agarwal et al. (2021) rliable aggregate metrics (IQM, Mean, Median,
  95% stratified bootstrap CIs) across multiple seeds and generate publication-grade
  Matplotlib/Seaborn figures and Markdown benchmark tables.
---

# Scientific Plotting & `rliable` Statistical Aggregation

Use this skill when building or running `benchmarks/eval/evaluate.py` and `benchmarks/eval/plot_results.py`, or when updating benchmark tables in `docs/04-experiments-log.md`.

---

## 1. Statistical Aggregation (`rliable` Standard)

Reporting only a single run or an uncalibrated mean hides heavy-tailed variance in RL. Follow Agarwal et al. (2021):

- **Interquartile Mean (IQM)**: Sort all seed-level scores, discard the bottom 25% and top 25%, and take the mean of the middle 50%. IQM is robust to outlier seeds while remaining more statistically efficient than the median.
- **95% Stratified Bootstrap Confidence Intervals**: Resample seeds with replacement $B = 2{,}000$ times per algorithm (stratifying by task/stage when aggregating across stages) and report the 2.5th and 97.5th percentiles.
- **Anchor Normalization**: Normalize returns where helpful against `[Random, RuleBasedHeuristic]` or report raw task metrics (`Win Rate %`, `Core HP %`, `Elemental Accuracy %`) alongside return.

---

## 2. Publication-Grade Visual Design

All generated figures saved under `docs/assets/figures/*.png` must follow strict visual legibility rules:

- **Colorblind-Safe Palette**: Use the Okabe-Ito palette (`#0072B2`, `#E69F00`, `#009E73`, `#CC79A7`, `#D55E00`, `#56B4E9`, `#000000`) with distinct line styles (`-`, `--`, `-.`, `:`) when lines exceed 4 algorithms.
- **Shaded Uncertainty Bands**: Plot seed-mean or IQM trajectories with shaded `± 1 SD` or `95% bootstrap CI` bands (`alpha=0.18`), never bare single lines without spread.
- **Clean Typography & Resolution**: Save at `300 DPI`, tight layout (`bbox_inches="tight"`), remove top and right spines (`ax.spines["top"].set_visible(False)`), add subtle dashed horizontal gridlines (`alpha=0.3`), and label axes with exact units (`Environment Steps`, `Episodic Return`, `Core Survival (%)`).

---

## 3. Automated Handbook Table Sync

`plot_results.py` should emit clean GitHub-Flavored Markdown tables directly to stdout or a summary JSON/MD snippet containing exact observed numbers formatted consistently (e.g., `142.3 [128.1, 155.8]`) so `docs/04-experiments-log.md` never relies on manual copy-paste approximations.
