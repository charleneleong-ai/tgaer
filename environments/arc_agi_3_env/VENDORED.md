# Vendored — `arc-agi-3-env`

Vendored (not `vf-install`ed) from the Prime Intellect Environments Hub env
[`ryanznie/arc-agi-3-env`](https://github.com/ryanznie/arc-agi-3-env) at upstream
commit `1bd4620808b3` (2026-06-14). Apache/community env — copied in and audited
rather than installed, so we control + can extend it (hub installs execute
untrusted code).

## What it is
A `verifiers` `MultiTurnEnv` (`load_environment(game_family, level_index, game_id, max_turns)`):
- **`simple_maze`** (2 levels) / **`complex_maze`** (5 levels) — *local* games via `arcengine`, no API. The solvable curriculum for RL bootstrapping.
- **`arc_agi`** — remote games via `ARC_API_KEY` + `ROOT_URL=https://three.arcprize.org` (scorecard + RESET/ACTION).
- Reward: binary `1.0 if WIN else 0.0` (+ `num_actions` / `timed_out` metrics). Actions are XML tags: `<action>ACTION1</action>`, `<action>ACTION6</action><x>..</x><y>..</y>`.

## Audit (2026-06-14)
Read in full. No `subprocess`/`eval`/`exec`/`socket`/`pickle`/`os.system`; the only
network is `requests` to the ARC API (`ROOT_URL`). `load_dotenv` reads `.env` for
`ARC_API_KEY`. Clean.

## Our extensions
- **Shaped reward (done):** `shaped_reward.py` — a `ShapedArcAgi3Env` + `load_environment`
  that keeps binary WIN and adds dense novelty / anti-stall signal. Reward logic lives in
  `tgaer.envs.arc_agi3.shaping` (unit-tested); this module is just the verifiers wiring.
- **Planned (see docs/specs/2026-06-14-arc-agi3-rl-design.md):** port the tgaer perception work
  into `_format_observation` (legible encoding, action feedback, board image via
  `tgaer.envs.arc_agi3.rendering.grid_to_png_data_url`); add sub-goal credit for remote games.

## Dependency
Needs `arcengine>=0.9.3` (the local-game engine) — install into the training venv
when wiring rollouts; not added to tgaer's core deps.
