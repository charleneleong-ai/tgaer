# arc-agi-3-env

### Overview
- **Environment ID**: `arc-agi-3-env`
- **Short description**: Multi-turn ARC-style game environment for PrimeIntellect using `arcengine`, with support for `SimpleMaze`, `ComplexMaze`, and ARC-AGI tasks
- **Tags**: `game`, `arc`, `arc-agi`, `maze`, `rl`, `multi-turn`

### Datasets
- **Primary dataset(s)**: Built-in `arcengine` maze levels plus ARC-AGI tasks fetched from the ARC API
- **Source links**: `arcengine` package for local game definitions, `GET {ROOT_URL}/api/games` for ARC-AGI task discovery
- **Split sizes**: Configurable at load time; maze levels are built in, ARC-AGI task count depends on fetched games and evaluation settings

### Task
- **Type**: multi-turn
- **Output format expectations**: XML action tags. For movement: `<action>ACTION1</action>`. For click actions: `<action>ACTION6</action><x>12</x><y>34</y>`
- **Rubric overview**: Binary task reward (`WIN` => 1, else 0), action-count tracking, and timeout tracking

### Quickstart

Configure the repo-level `.env`:

```bash
ARC_API_KEY=your_api_key
ROOT_URL=https://three.arcprize.org
OPENAI_API_KEY=your_openai_key
```

Run a local smoke evaluation:

```bash
set -a
source .env
set +a
PYTHONPATH=environments/arc_agi_3_env uv run --project environments/arc_agi_3_env vf-eval arc-agi-3-env \
  -m gpt-4.1-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  -a '{"game_family":"simple_maze","level_index":0,"max_turns":8}' \
  -n 1 -r 1
```

Notes:
- Run these commands from the repo root.
- The repo `.env` is loaded automatically by `arc_agi_3_env.py`, but `vf-eval` itself still needs the shell to see `OPENAI_API_KEY`, so source `.env` before running the command.
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- `ARC_API_KEY` and `ROOT_URL` are only required for `game_family="arc_agi"`.
- `vf-eval` currently needs `PYTHONPATH=environments/arc_agi_3_env` when run manually from the repo root.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `game_family` | str | `"simple_maze"` | Which game/task family to run: `simple_maze`, `complex_maze`, or `arc_agi` |
| `max_turns` | int | `100` | Maximum number of turns before the rollout times out |
| `system_prompt` | str | built-in | Override the default agent instructions |
| `game_id` | str \| null | `null` | Optional ARC-AGI game identifier to fetch from the API |
| `level_index` | int | `-1` | Maze level index to evaluate for `simple_maze` or `complex_maze`; `-1` means all built-in levels |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward: `1.0` if terminal state is `WIN`, otherwise `0.0` |
| `num_actions` | Number of actions taken by the agent, sourced from the engine action count |
| `timed_out` | `1.0` if the rollout stopped because `max_turns` was reached, otherwise `0.0` |

### Testing

Smoke eval:

```bash
set -a
source .env
set +a
PYTHONPATH=environments/arc_agi_3_env uv run --project environments/arc_agi_3_env vf-eval arc-agi-3-env \
  -m gpt-4.1-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  -a '{"game_family":"simple_maze","level_index":0,"max_turns":8}' \
  -n 1 -r 1
```

Structured recording:

```bash
uv run --project environments/arc_agi_3_env \
  python environments/arc_agi_3_env/evals/run_recording.py \
  --game-family simple_maze \
  --level-index 0 \
  --max-turns 8
```

This writes an ARC-style `.recording.jsonl` under `environments/arc_agi_3_env/evals/recordings/`.

Select a specific ARC-AGI puzzle:

```bash
set -a
source .env
set +a

uv run --project environments/arc_agi_3_env \
  python environments/arc_agi_3_env/evals/run_recording.py \
  --game-family arc_agi \
  --game-id ft09-0d8bbf25 \
  --max-turns 3
```

Discover available ARC-AGI puzzle ids:

```bash
set -a
source .env
set +a

python - <<'PY'
import os, requests
r = requests.get(
    f"{os.environ['ROOT_URL'].rstrip('/')}/api/games",
    headers={"X-API-Key": os.environ["ARC_API_KEY"], "Accept": "application/json"},
    timeout=30,
)
r.raise_for_status()
for item in r.json():
    print(item["game_id"])
PY
```

### How It Works

This environment adapts `arcengine` game loops and ARC-AGI task instances into a PrimeIntellect `MultiTurnEnv`:

1. **Environment Loader** (`arc_agi_3_env.py`):
   - Creates a PrimeIntellect-compatible multi-turn environment
   - Selects a game family from env args
   - Initializes built-in `arcengine` mazes or fetches ARC-AGI tasks from the ARC API

2. **Game Adapter**:
   - Instantiates a fresh game/task per rollout
   - Serializes the current state into a text observation
   - Parses model outputs into `arcengine` actions
   - Advances the environment with `perform_action(...)`
   - Supports `ACTION6` click actions via XML `<x>` / `<y>` tags

3. **Termination and Metrics**:
   - Marks success when the game reaches `GameState.WIN`
   - Marks failure when the game reaches `GameState.GAME_OVER`
   - Marks timeout when `max_turns` is reached
   - Emits `reward`, `num_actions`, and `timed_out`

### Relevant Links

- [PrimeIntellect Verifiers Environments](https://docs.primeintellect.ai/verifiers/environments)
- [ARCEngine Package](https://pypi.org/project/arcengine/)
- [Evals README](/Users/ryanznie/Desktop/work/arc-agi-3-env/environments/arc_agi_3_env/evals/README.md)
