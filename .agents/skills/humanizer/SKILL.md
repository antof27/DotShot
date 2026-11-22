---
name: humanizer
description: >-
  Remove AI-generated writing patterns, clichés, inflated claims, and uniform
  sentence cadence from technical documentation, research summaries, and
  docstrings. Combines blader/humanizer (Wikipedia Signs of AI Writing) with
  Notelia's Scientific Humanizer standard.
---

# Humanizer: Natural Scientific and Technical Prose

Use this skill whenever drafting, editing, or reviewing prose in `.agents/docs/`, `docs/`, `README.md`, or extended module documentation. Technical precision comes first; natural rhythm follows close behind.

---

## 1. Core Principles

1. **State concrete facts over abstract praise.** Replace vague claims ("revolutionary framework", "groundbreaking results") with specific mechanisms, numbers, equations, or failure modes.
2. **Respect reader intelligence.** Cut throat-clearing openers ("In today's rapidly evolving landscape...", "Let's explore...") and empty summary sign-offs ("In conclusion, these advances open exciting horizons...").
3. **Vary sentence length ("burstiness").** Mix crisp 8-word statements with 25-word analytical sentences. Avoid three consecutive sentences of identical structure or length.
4. **Keep formatting functional.** Use bolding, tables, and lists only when they clarify structure, never to decorate a thin paragraph.

---

## 2. Banned AI Writing Patterns

Scan every draft against these patterns before finalizing:

| Pattern Category | Typical Offenders to Cut | Concrete Replacement Strategy |
| :--- | :--- | :--- |
| **AI Vocabulary Crutches** | *delve, tapestry, testament, pivotal, seamless/seamlessly, holistic, underscore, foster, showcase, leverage, interplay, intricate, paradigm shift, game-changer, landscape, realm, embark* | Use plain verbs: *examine, show, connect, combine, enable, depend on, split, measure, use*. |
| **Inflated Significance** | *"stands as a testament to"*, *"marks a pivotal turning point"*, *"plays a crucial role in shaping"* | Explain the concrete cause and effect directly (e.g., *"MAPPO cuts gradient variance by conditioning the critic on global state $s^t$."*). |
| **Superficial `-ing` Tack-ons** | *"...reducing variance, thereby ensuring robust convergence and fostering stability."* | Split into a clean causal claim or drop the trailing clause. |
| **Forced Rule of Three** | Constant triads: *"fast, scalable, and robust"*; *"clarity, precision, and elegance"* | List only the exact properties actually supported by evidence (one, two, or four if true). |
| **Negative Parallelisms** | *"Not only X, but also Y"*; *"It is not merely A; it is B"* used repeatedly | State B directly, or state A and B plainly without rhetorical windup. |
| **Mechanical Transitions** | Overuse of *"Moreover,"*, *"Furthermore,"*, *"Additionally,"*, *"Consequently,"* at paragraph starts | Connect paragraphs logically through the subject noun or explicit contrast (*"Unlike IPPO, HAPPO..."*). |
| **Promotional Filler** | *"rich visual experience"*, *"cutting-edge"*, *"state-of-the-art"* used as an empty adjective | Reserve *state-of-the-art (SOTA)* strictly for benchmarked baselines or literature eras with citations. |
| **Em-Dash Overuse** | Multiple `—` interruptions per paragraph as a dramatic crutch | Use commas, parentheses, or split into two clean sentences. |
| **Sycophancy & Hedging** | *"Great question!"*, *"It is important to note that..."*, *"While challenges remain..."* | Delete the filler and begin with the actual constraint or observation. |

---

## 3. Scientific Prose Calibration

Technical writing about Reinforcement Learning still needs warmth and clarity, not robotic dryness:
- **Prefer active voice** when describing design choices or algorithm mechanics (*"QMIX constrains mixing weights to be nonnegative"* rather than *"Nonnegative constraints are imposed by QMIX"*).
- **Name exact objects.** Write *"Equation (4)"*, *"the 32-bin aim head"*, or *"Kuba et al. (2022)"* instead of *"this approach"* or *"recent work"*.
- **Own limitations honestly.** If an experiment failed, overfit, or ran only as a smoke test, state that plainly with numbers.

---

## 4. Mandatory 3-Step Self-Edit Pass

Run this pass on every document before marking a task complete:

1. **Draft for technical accuracy:** Verify equations, symbols, file paths, and citation keys first.
2. **Audit for AI tells:** Check for every banned word or structural crutch in Section 2, repeated sentence starters, and uniform paragraph lengths.
3. **Rewrite for directness:** Cut 10-20% dead weight where words repeat meaning, vary cadence, and re-verify that zero technical facts were softened or altered.
