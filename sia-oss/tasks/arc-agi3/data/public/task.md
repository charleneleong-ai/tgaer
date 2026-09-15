# ARC-AGI-3 explorer: minimize actions per level

You are given a model-free, training-free game-playing agent for
[ARC-AGI-3](https://arcprize.org/arc-agi/3/) — a suite of small interactive
puzzle games. The agent explores an unknown game by building a directed state
graph of frame signatures and always taking an untested action from the
current state, or routing to the nearest state that still has one
(frontier-directed exploration). It has no access to any LLM at decision time
— this is deliberate: on this benchmark a hand-engineered, deterministic
explorer clears more levels than an LLM-driven agent, and that gap is a known,
measured result from this project's history, not an assumption to challenge.

## Files

- `arc_agi3_explorer.py` — the explorer's core: `StateGraph`, `frame_signature`,
  avatar/lattice induction, `click_targets`, the `act()` decision loop. **This
  is what you are improving.**
- `arc_agi3_grid.py`, `arc_agi3_semantics.py` — supporting modules the explorer
  imports (grid/component primitives, empirical avatar-lattice induction). Only
  touch these if a genuine fix requires it; prefer changes localized to
  `arc_agi3_explorer.py`.
- `reference_target_agent.py` — the entrypoint. Do not change its CLI contract
  (`--dataset_dir`, `--working_dir`, writes `results/submission.json`) — the
  grading harness depends on it.

## Objective — the official RHAE metric (this is what Kaggle scores)

Per level: `S_l = min(1.15, (human_baseline / your_actions)^2)` — squared, so
5x the baseline actions is worth ~4% of the level, 1x is worth 100%.

Per game (environment): a **level-index-weighted** mean, `w_l = l`, **capped
by the weighted fraction of levels completed**. In an 8-level game, level 1
is worth 1/36 of the game and level 8 is worth 8/36; completing only level 1
caps the whole game at 1/36 ≈ 2.8% no matter how efficient you are.

Total: mean over the 8 games, reported as a percent.

Two levers, and which one applies depends on the game:
- **Cap-limited** games (`lp85` — level 1 already at the 1.15 cap; `bp35`,
  `sk48`, `cn04` — nothing cleared): only *clearing the next level* moves
  the score. Efficiency tweaks on level 1 are worth exactly nothing here.
- **Efficiency-limited** games (`ls20` 3.1x, `sp80` 4.9x, `sc25` 13.5x,
  `tu93` 20x/12x): cutting wasted actions on already-cleared levels pays,
  quadratically. `sc25` at 13.5x collects 0.5% of its level; at 2x it would
  collect 25%.

`evaluate.py`'s per-game output prints `limited-by=cap|eff` so you can see
which lever applies to each game.

## Closed ideas — do not re-propose these

Exploration-gate throttling and per-cell frame-signature denoising are both
**closed with measured evidence** (multiple variants each, all net-negative
or exactly neutral). Full detail is in `AGENTS.md` next to this file if you
want it, but you do not need to read it to know these two are off the table.

## Your task this generation — do this, don't deliberate about whether to

**Implement goal-directed pathfinding toward a specific target, instead of
frontier-BFS over only already-discovered edges.** Right now `act()` explores
new actions and routes to the nearest state with an untested edge — it never
does real distance-based pathfinding toward a *specific* candidate object
once one is identified from `click_targets()` or similar. This is the single
highest-confidence untried direction: an independent A*-over-world-model
agent using exactly this mechanism cleared all 25 public ARC-AGI-3 games in
45% fewer actions than the human median (see `AGENTS.md` for the citation).

Concretely: once a salient/interactive target is identified, plan a route to
it directly (A* or similar over the `StateGraph`'s discovered structure, or
over the raw grid if the graph doesn't cover it yet) rather than only walking
edges the frontier search already happened to open. This should help `sp80`
and `sc25` most directly (both already reach and interact with real targets,
just inefficiently) — `bp35`/`sk48`/`cn04` (never-cleared games) are lower
priority since they may need cold-start changes this idea doesn't address.

**Budget your turns**: read `arc_agi3_explorer.py`'s `act()`/`_choose()` and
`click_targets()` once, then implement and test. Do not spend more than 3-4
tool calls deciding whether this is worth doing — it already has external
validation. If you finish this and have turns left, `AGENTS.md`'s "Untried"
section has two more candidates (explore/solve phase split, action-effect
prediction as a staleness gate) ranked below this one.

## Baseline (what "no change" scores)

**RHAE = 0.4263%** across the 8-case suite (lp85 2.78%, ls20 0.37%, sp80
0.20%, sc25 0.03%, tu93 0.04%, the rest 0). 2 of 8 pass their gate cleanly
(`lp85`, `ls20` — control/regression guards; do not regress them). A genuine
improvement raises RHAE above 0.4263% without dropping either control case's
pass status. (The older unweighted `total_level_score` of 131.08 is still
printed for continuity but is not the metric — it was 88% lp85's capped
level 1, which the real metric values at 2.8%.)

## Output format

`reference_target_agent.py` writes `results/submission.json`:

```json
{
  "scorecards": [
    {"game": "lp85", "levels": [{"level": 1, "cleared_in_repeats": 1, "repeats": 1,
      "human_baseline_actions": 17, "our_actions_median": 10.0,
      "ratio_vs_baseline": 0.6, "level_score": 115.0}]},
    {"game": "sk48", "levels": []}
  ]
}
```

Grading (`evaluate.py`, not visible to you) computes `total_level_score` from
this and compares it to the baseline above.
