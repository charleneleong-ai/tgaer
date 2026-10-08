# ARC-AGI-3 on SIA

A [SIA](https://sia.hexo.ai) Foundry project that runs the local scoring
harness (`docs/arc-agi3-kaggle.md`) as its eval suite, so `sia improve` can
propose and score patches to the explorer against the exact metric the
competition pays for — actions per level, not levels cleared.

## Why the project root is `src/tgaer/agents/`, not the repo root

SIA snapshots its whole project directory into `.sia/versions/<name>/` on
every `sia init` and `sia fixes apply` — a full `shutil.copytree`, not a git
worktree. Pointed at the repo root, that copy pulled in `vendor/` (the
starter checkout's own `.venv` and downloaded game sources), `wandb/`, and
`environment_files/`, and used 62GB before the first `sia init` ran the disk
out. Scoped to `src/tgaer/agents/` (536K) the same snapshot is instant, and it
is also the right *patch* boundary: `sia fixes apply` can only write inside
the project root, so scoping it to the agent files is what lets SIA actually
propose changes to `arc_agi3_explorer.py`, not just read it.

## Layout

| what | where |
| --- | --- |
| SIA project root | `src/tgaer/agents/` |
| command adapter | `src/tgaer/agents/sia_adapter.py` |
| eval harness | `src/tgaer/agents/sia_eval.py` |
| eval cases | `src/tgaer/agents/sia_questions.jsonl` |
| project config | `src/tgaer/agents/.sia/config.toml` (tracked) |
| tests | `tests/test_arc_agi3_sia.py` |

`sia_adapter.py` speaks the `sia` command-adapter contract — JSON on stdin,
JSON on stdout — the same contract `docs/arc-agi3-kaggle.md`'s scorer reuses
under the hood (`play`, `level_breakdown`'s sibling `run_levels`,
`inert_features`, `faults`). Given `{"input": "Play lp85."}` it plays that
game `ARC_SIA_REPEATS` times (default 3, different seeds) with the explorer,
and appends a `--- ARC-AGI-3 SCORECARD ---` JSON block to its answer — the
median actions-per-level ratio against the human baseline, which level
scores were, and both diagnostics from the local scorer.

`sia_eval.py` invokes the adapter once per case in `sia_questions.jsonl` and
grades the scorecard **arithmetically** (`require_levels`, `max_ratio`,
`min_levels_any` per case), not with an LLM judge — ARC-AGI-3 pays
`min((baseline / actions)^2 * 100, 115)` per level, so "did this run get
better" has an exact answer, and a judge would add variance to the one number
in this project that already had too much (the public leaderboard sits flat
at 0.13 whether the agent clears 2 levels or 5 — see
`project_arcagi3_metric_is_blind` — which is why scoring moved local in the
first place).

## The eval set

Eight games from `evals/arc_agi3.yaml`'s design: five `ratio-*` cases push
down the actions-per-level ratio on games the explorer already clears but
slowly (ls20, sp80, sc25, tu93, bp35 — the last is one of the six Kaggle
rerun games), one `control-lp85` regression guard on the one game already at
the 115-point cap (a patch that gets cautious everywhere destroys more here
than any ratio case could win back), and two `frontier-*` coverage cases on
Kaggle rerun games the explorer has never cleared a level of at all (sk48,
cn04 — any single clear across the repeats is a pass, since the explorer has
no goal signal before its first win).

## Running

```bash
# fixes are made to sia_adapter.py's own dependency-order copies, so
# `tgaer` must be editable-installed in the shared venv first
uv pip install -e .

cd src/tgaer/agents
sia status                          # what exists, what's next
sia evals run --no-harbor           # score the working tree, no Docker
sia failures detect                 # find what's costing the most ratio
sia fixes propose && sia fixes apply
sia improve --rounds 3              # detect -> fix -> re-score, looped
```

`.venv` is a symlink to the repo's own `.venv` (excluded from every
`.sia/versions/vN/` copy by `sia`'s own versioning — it's meant to be shared
across versions via `PATH`, which `sia` injects before running either
command). `[agent] cmd` and `[engine] eval_command` in `.sia/config.toml`
call bare `python` for exactly that reason — `.venv/bin/python` resolves to
nothing inside a version copy and fails silently in well under a second.

`ARC_SIA_REPEATS`, `ARC_SIA_MAX_ACTIONS`, `ARC_SIA_AGENT`, `ARC_SIA_SEED`
override the adapter's defaults (3 repeats, 600 actions, the explorer,
seed 0) — exported before `sia evals run` so they reach the subprocess.

## Once a fix looks good in dev

`arc_agi3_build_notebook.py` inlines `arc_agi3_kaggle.py` (and whichever of
`arc_agi3_explorer.py`/`arc_agi3_grid.py`/`arc_agi3_semantics.py` it imports)
verbatim into the Kaggle submission notebook, so a version SIA applied under
`.sia/versions/vN/` has to be copied back over the tracked files in
`src/tgaer/agents/` — and re-scored with `arc_agi3_score_local.py --games
competition` — before it is the one that ships. SIA proposes and verifies
patches against the dev eval set; it does not touch the Kaggle build or
submit anything itself.
