# Sample task descriptions

Variants of this task a future run might target, for context on what "this
class of task" means beyond the exact 8-case suite wired up today:

- Minimize actions-per-level on a different subset of the ARC-AGI-3 game
  roster (the full Kaggle rerun mounts more games than these 8).
- Extend the explorer to clear a level it has never cleared at all (pure
  coverage, ignoring action efficiency).
- Reduce the explorer's worst-case action count on a single named game
  without regressing any other game in the suite.

All variants share the same objective shape: this is a code-improvement task
against a deterministic, model-free game-playing agent, graded by a squared
ratio-to-human-baseline score — not a per-question LLM-inference task.
