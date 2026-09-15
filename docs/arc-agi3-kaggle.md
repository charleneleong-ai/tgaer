# ARC-AGI-3 Kaggle agent

The agent submitted to `arc-prize-2026-arc-agi-3`. It is inlined verbatim into
a Kaggle notebook via `%%writefile`, so `arc_agi3_kaggle.py` must stay
self-contained and stdlib-only — no `tgaer` imports.

## Layout

| what | where |
| --- | --- |
| agent | `src/tgaer/agents/arc_agi3_kaggle.py` |
| local scoring | `src/tgaer/evaluation/arc_agi3_score_local.py` |
| notebook build | `src/tgaer/evaluation/arc_agi3_build_notebook.py` |
| turn-by-turn trace | `src/tgaer/evaluation/arc_agi3_trace.py` |
| tests | `tests/test_arc_agi3_kaggle.py`, `tests/test_arc_agi3_score_local.py` |

## The starter checkout

The competition starter is **not** part of this repo. It holds upstream code,
the bundled `ARC-AGI-3-Agents` SDK, the game sources the SDK downloads into
`environment_files/`, and the offline wheelhouse the kernel installs from —
none of which we author.

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter \
  vendor/ARC-AGI-3-Kaggle-Starter
```

Both scripts find it at `vendor/ARC-AGI-3-Kaggle-Starter` and honour
`ARC_STARTER_ROOT` to point elsewhere.

## Two environments, one of which cannot be forked

Local scoring runs `OperationMode.NORMAL`: the games are in-process, so the
game object can be `deepcopy`-ed and searched directly. The **scored** rerun
does not work that way — it sets `OPERATION_MODE=online` against a gateway
sidecar with `ENVIRONMENTS_DIR=` empty, so there is no local environment at
all. Anything that depends on reaching into the game works perfectly in
development and is inert where it counts.

That constraint is why the agent learns its own forward model
(`ForwardModel`) from observed frames rather than simulating the real game:
frames are all the gateway gives us. Undo does not provide a way around it
either — sk48's undo restores every sprite position and never refunds the move
budget, so a probe is spent whether or not it is taken back, and cn04 caps its
first level at 75 actions.

## Running

```bash
# score the working tree against the six rerun games
uv run python src/tgaer/evaluation/arc_agi3_score_local.py \
  --backend vllm --games competition --max-steps 200 --workers 6

# build the submission notebook
uv run python src/tgaer/evaluation/arc_agi3_build_notebook.py
```

Scoring appends a row per run to `experiments/score_local.jsonl` with the
per-decision counters, the scorecard `score`, and a `levels` breakdown giving
every cleared level's actions against the human baseline.

**Read the ratio, not the level count.** A level scores
`min((baseline / actions_on_that_level)^2 * 100, 115)`, and a game averages
those weighted by level index over *all* its levels, so clearing more levels
slowly loses to clearing fewer quickly: measured on the 25-game roster at 2400
actions, the 12 levels cleared scored 0.174 against the 2.349 the same levels
would be worth at baseline speed — 7.4%. The run prints a `Score by level`
table and a median/best/worst ratio line, and logs both to W&B, so two runs can
be compared on the thing that scores. The level count is also noisy in its own
right: a single completed level failed to reproduce across three targeted
repeats and six full runs.
