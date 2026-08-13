"""Score `src/tgaer/agents/arc_agi3_kaggle.py` on a local dev set of real ARC-AGI-3 games.

The Kaggle competition allows one submission per day, so comparing agent
revisions against the leaderboard costs a day per hypothesis and returns a
single noisy number. This runs the same games in-process against a local model
so a comparison takes minutes.

Any of the 25 published games can be played; `arc_agi` downloads them into
`environment_files/` on first use. The six the Kaggle rerun actually uses are
COMPETITION_GAMES below.

    .venv/bin/python src/tgaer/evaluation/arc_agi3_score_local.py --backend random
    .venv/bin/python src/tgaer/evaluation/arc_agi3_score_local.py --backend ollama --model qwen3:8b
    .venv/bin/python src/tgaer/evaluation/arc_agi3_score_local.py --agent-rev ae71cdf --label v31

`--agent-rev` scores a committed revision of the agent instead of the working
tree, which is what makes before/after comparisons possible.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

REPO = Path(__file__).resolve().parents[3]
# The competition starter is a checkout of arcprize/ARC-AGI-3-Kaggle-Starter,
# not part of this repo: it carries the bundled SDK and the game sources the
# SDK downloads into environment_files/. Only the path is referenced, so the
# checkout can live anywhere.
ROOT = Path(
    os.environ.get("ARC_STARTER_ROOT", REPO / "vendor" / "ARC-AGI-3-Kaggle-Starter")
)
sys.path.insert(0, str(REPO / "src"))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


def require_starter() -> Path:
    """The starter checkout, or a clear error naming how to get one.

    Checked here rather than at import. Raising SystemExit while a module is
    being imported takes down whatever imported it: the checkout is gitignored,
    so on a clean CI machine this aborted pytest during collection and *no*
    tests ran — a missing optional dependency reported as a total suite
    failure. Only the code that actually plays a game needs the checkout.
    """
    if not ROOT.exists():
        raise typer.BadParameter(
            f"Starter checkout not found at {ROOT}. Clone "
            "https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter there, or set "
            "ARC_STARTER_ROOT to an existing checkout."
        )
    if not VENDOR.exists():
        raise typer.BadParameter(
            f"Framework not found at {VENDOR}. Run `make setup` in {ROOT} first."
        )
    return ROOT


from dotenv import load_dotenv  # noqa: E402

# Without ARC_API_KEY the SDK silently falls back to an anonymous key, so
# scorecards are unattributed and share the anonymous rate limit. The key lives
# in the parent project's .env; a repo-local .env wins if present.
for env_file in (REPO / ".env", ROOT / ".env"):
    if env_file.exists():
        load_dotenv(env_file)
        break

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

# The six games the Kaggle rerun mounts under /kaggle/input (confirmed from a
# rerun log). The other 19 published games make a useful held-out set.
COMPETITION_GAMES = ["sk48", "tn36", "m0r0", "bp35", "cn04", "dc22"]

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


class ContextOverflow(ValueError):
    """Raised when a prompt exceeds n_ctx, mirroring llama-cpp-python.

    llama_cpp/llama.py:1336 (v0.3.34, the wheel the Kaggle notebook installs)
    raises ValueError when `len(prompt_tokens) >= n_ctx`. Reproducing that is
    the whole point of this harness: a backend that silently truncates instead
    would hide the exact failure that made prompt A/B tests meaningless.
    """


class OllamaBackend:
    """A `llama_cpp.Llama`-compatible shim backed by a local ollama server.

    Only `create_chat_completion` is implemented, in the OpenAI response shape
    the agent expects — including tool-call `arguments` as a JSON *string*,
    which llama-cpp emits but ollama returns pre-parsed.
    """

    # ollama truncates the prompt to num_ctx and reports the truncated count, so
    # asking it for exactly n_ctx would hide every overflow. Run with headroom to
    # get a true token count, then apply llama-cpp's limit ourselves.
    RUNTIME_CTX_MULTIPLIER = 4

    def __init__(
        self,
        model: str,
        n_ctx: int,
        host: str,
        timeout: float = 300.0,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.n_ctx = n_ctx
        self.runtime_ctx = n_ctx * self.RUNTIME_CTX_MULTIPLIER
        self.host = host.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.prompt_tokens: list[int] = []
        self.seed = seed

    @property
    def calls(self) -> int:
        return len(self.prompt_tokens)

    @property
    def overflows(self) -> int:
        return sum(1 for t in self.prompt_tokens if t >= self.n_ctx)

    def summary(self) -> str:
        seen = self.prompt_tokens or [0]
        colour = "red" if self.overflows else "green"
        return (
            f"llm calls: {self.calls}   context overflows: "
            f"[{colour}]{self.overflows}[/]   prompt tokens min/max: "
            f"{min(seen)}/{max(seen)} (n_ctx {self.n_ctx})"
        )

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: int = 64,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,  # noqa: ARG002 - ollama has no equivalent
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # qwen3 reasons by default; the agent wants one action
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.runtime_ctx,
                # ollama honours this; dropping it would stamp a seed on the
                # JSONL row that never reached the model.
                **({"seed": self.seed} if self.seed is not None else {}),
            },
        }
        if tools:
            payload["tools"] = tools

        response = self.client.post(f"{self.host}/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()

        # ollama truncates silently; llama-cpp raises. Enforce llama-cpp's rule
        # using the model's own token count so the comparison stays faithful.
        # No default: if ollama ever stops reporting this, fail loudly rather
        # than silently treating every prompt as fitting.
        used = int(body["prompt_eval_count"])
        self.prompt_tokens.append(used)
        if used >= self.n_ctx:
            raise ContextOverflow(
                f"Requested tokens ({used}) exceed context window of {self.n_ctx}"
            )
        return {"choices": [{"message": self._to_openai(body.get("message") or {})}]}

    @staticmethod
    def _to_openai(message: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"content": message.get("content") or ""}
        calls = message.get("tool_calls") or []
        if calls:
            out["tool_calls"] = [
                {
                    "function": {
                        "name": (tc.get("function") or {}).get("name", ""),
                        "arguments": json.dumps(
                            (tc.get("function") or {}).get("arguments") or {}
                        ),
                    }
                }
                for tc in calls
            ]
        return out


# Counters that mean something went wrong. Any of them above zero is a defect,
# not a preference: the agent reached a path it is not supposed to reach.
# image_unavailable is here because it is the counter that would have caught
# vision shipping inert — it read 900 on that run and was noticed only by
# somebody reading the numbers by hand.
FAULT_COUNTERS = (
    "image_unavailable",
    "mouse_without_coords",
    "tool_path_exception",
    "choose_action_exception",
    "repl_error",
)
# Counters that are legitimate in ones and twos and alarming in bulk: the agent
# is limping rather than broken. Compared against the actions actually taken.
DEGRADED_COUNTERS = (("random_fallback", 0.05), ("raw_text_fallback", 0.50))


def inert_features(agent_module: Any, decisions: dict[str, int]) -> list[str]:
    """Enabled features that never once fired during the run.

    Five features shipped this way — vision twice, undo, the mechanic note, and
    chrome detection with the budget model built on it — each built,
    unit-tested, verified in isolation, and then never executed in the real
    loop. Every time the run still reported a tidy zero, which reads exactly
    like a feature that ran and did not help. A feature that cannot fire is a
    different problem from a feature that does not work, and the two must not
    look the same.

    Chrome detection was the one that made the case for registering everything
    here: it was never listed, so nothing ever asked whether it fired, and it
    stayed dead through five agent revisions while its unit tests passed.
    """
    expected = {
        "image_sent": agent_module.SEND_IMAGE,
        "probe": agent_module.PROBE_ACTIONS,
        "repl_call": agent_module.REPL_STEPS > 0,
        "exploit": agent_module.EXPLOIT_REPEATS > 0,
        # Always on: the forward model observes every transition, so predicting
        # nothing across a whole run means it is not being fed.
        "forward_predicted": True,
    }
    return sorted(
        name for name, on in expected.items() if on and not decisions.get(name)
    )


def faults(decisions: dict[str, int], actions: int) -> list[str]:
    """Counters that indicate a broken path rather than an unhelpful feature.

    inert_features asks whether something never happened. This asks the
    opposite question — whether something happened that never should — because
    the two failures look identical in a results table and neither is visible
    in the score.
    """
    found = [
        f"{name}={decisions[name]}" for name in FAULT_COUNTERS if decisions.get(name)
    ]
    for name, limit in DEGRADED_COUNTERS:
        count = decisions.get(name, 0)
        if actions and count / actions > limit:
            found.append(
                f"{name}={count} ({count / actions:.0%} of actions, limit {limit:.0%})"
            )
    return found


def resolve_games(games: str, available: list[str]) -> list[str]:
    """Map the --games selector onto the ids this account can actually play."""
    if games == "competition":
        return [g for g in COMPETITION_GAMES if g in available]
    if games == "all":
        return available
    wanted = {g.strip() for g in games.split(",")}
    chosen = [g for g in available if g in wanted]
    if missing := wanted - set(chosen):
        raise typer.BadParameter(f"unknown game id(s): {sorted(missing)}")
    return chosen


DEFAULT_HOSTS = {"vllm": "http://127.0.0.1:8000/v1", "ollama": "http://localhost:11434"}


def make_backend(
    backend: str,
    agent_module: Any,
    model: str,
    host: str,
    n_ctx: int,
    seed: int | None = None,
) -> Any | None:
    """Pick where inference comes from.

    'vllm' reuses the agent's own HTTPChatBackend, so what the harness measures
    is exactly what the submission runs — including the image payload and the
    chat_template_kwargs that disable reasoning.
    """
    host = host or DEFAULT_HOSTS.get(backend, "")
    if backend == "random":
        return None
    if backend == "vllm":
        return agent_module.HTTPChatBackend(host, model, seed=seed)
    if backend == "ollama":
        return OllamaBackend(model, n_ctx, host, seed=seed)
    raise typer.BadParameter(f"unknown backend {backend!r}")


AGENT_PATH = Path("src/tgaer/agents/arc_agi3_kaggle.py")


def load_agent_class(rev: str | None) -> type[Any]:
    """Import MyAgent from the working tree, or from a committed revision."""
    path = REPO / AGENT_PATH
    if rev:
        source = subprocess.run(
            ["git", "show", f"{rev}:{AGENT_PATH}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tmp = Path(tempfile.mkdtemp()) / "my_agent_rev.py"
        tmp.write_text(source)
        path = tmp

    name = f"scored_agent_{rev or 'worktree'}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Must be in sys.modules before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit(f"{path} must define a class named `MyAgent`")
    # This harness injects a backend per game, so neutralise both production
    # sources. _get_shared_model no longer exists; without these, --backend
    # random silently routes to a real model (or to whatever ARC_LLM_BASE_URL
    # points at) and stops being the random floor it is used as.
    module.REMOTE_BACKEND = None
    module.load_llama = lambda: None
    return module.MyAgent


class WandbRun:
    """Optional W&B logging: per-game metrics plus the frames the model saw.

    The Kaggle kernel has no internet, so this only ever runs locally — which is
    exactly where the failure modes need looking at. Logging the rendered board
    matters because it is the image actually sent to the model: if the render is
    wrong, no amount of prompt reading would reveal it.
    """

    def __init__(self, enabled: bool, label: str, config: dict[str, Any]) -> None:
        self.run = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            console.print("[yellow]wandb not installed; skipping logging[/]")
            return
        self.wandb = wandb
        self.run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "arc-agi-3"),
            name=label,
            config=config,
            reinit=True,
        )

    def log_frames(self, agent_module: Any, game: str, frames: list[Any]) -> None:
        """Log a handful of boards as the model saw them, start to finish."""
        if self.run is None or not frames:
            return
        picks = [0, len(frames) // 2, len(frames) - 1] if len(frames) > 2 else [0]
        images = []
        for index in dict.fromkeys(picks):
            grid = frames[index].frame[-1] if frames[index].frame else None
            if not grid:
                continue
            png = agent_module.render_board_png(tuple(tuple(r) for r in grid))
            if png:
                images.append(
                    self.wandb.Image(io.BytesIO(png), caption=f"{game} step {index}")
                )
        if images:
            self.run.log({f"frames/{game}": images})

    def log(self, data: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(data)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def play(
    agent_cls: type[Any],
    game_id: str,
    arc: Any,
    backend: Any | None,
    max_steps: int,
    seed: int | None = None,
) -> dict[str, Any]:
    env = arc.make(game_id)
    if env is None:
        return {"game": game_id, "error": "env-unavailable"}

    agent = agent_cls(
        card_id="local-dev",
        game_id=game_id,
        agent_name=f"score.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["score-local"],
    )
    agent._llm = backend
    if seed is not None:
        # The agent's own RNG picks random fallbacks and random click targets;
        # seeding only the model would leave that source of variation loose.
        # Mixed with the game id: seeding every game identically made the
        # concurrent games draw the same numbers at the same index, so a run
        # contributed one correlated draw rather than one per game.
        assert hasattr(agent, "_rng"), (
            f"{agent_cls.__name__} has no _rng to seed; --agent-rev may have "
            "loaded a revision that predates it, and the run would record a "
            "seed it never used"
        )
        agent._rng = random.Random(f"{seed}:{game_id}")
    agent.MAX_ACTIONS = max_steps

    started = time.monotonic()
    agent.main()
    final = agent.frames[-1]
    return {
        "game": game_id,
        "state": str(final.state),
        "levels_completed": final.levels_completed,
        "actions": agent.action_counter,
        "seconds": round(time.monotonic() - started, 1),
        "decisions": dict(getattr(agent, "stats", {})),
        "_agent": agent,
    }


def render(
    rows: list[dict[str, Any]],
    label: str,
    score: float,
    levels: int,
    backend: Any | None,
) -> None:
    table = Table(title=f"Dev-set result — {label}", title_style="bold")
    for col in ("game", "levels", "actions", "state", "sec"):
        table.add_column(col, justify="right" if col != "state" else "left")
    for row in rows:
        if row.get("error"):
            table.add_row(row["game"], "-", "-", row["error"], "-")
            continue
        table.add_row(
            row["game"],
            str(row["levels_completed"]),
            str(row["actions"]),
            row["state"].replace("GameState.", ""),
            str(row["seconds"]),
        )
    console.print(table)

    console.print(
        f"[bold]levels completed: {levels}[/]   scorecard score: [bold]{score}[/]"
    )
    # Only the ollama backend counts tokens; the HTTP one has no stats to show.
    # This line is decoration, and it once crashed the run before the inert-feature
    # check downstream of it could report which features had silently died.
    if (summary := getattr(backend, "summary", None)) is not None:
        console.print(summary())


@app.command()
def main(
    games: str = typer.Option(
        "competition",
        help="'competition' (the 6 rerun games), 'all' (25), or a csv of ids.",
    ),
    backend: str = typer.Option(
        "vllm",
        help="'vllm' (A100 or any OpenAI-compatible server), 'ollama', "
        "or 'random' (no model: the floor).",
    ),
    model: str = typer.Option("arc-agent", help="Served model name / ollama tag."),
    host: str = typer.Option(
        "",
        help="Server base URL. Defaults per backend: vllm "
        "http://127.0.0.1:8000/v1 (tunnel the A100 with `make tunnel-a100`), "
        "ollama http://localhost:11434.",
    ),
    n_ctx: int = typer.Option(
        8192, help="Context window; overflow raises, as llama-cpp does."
    ),
    max_steps: int = typer.Option(80, help="Per-game action cap."),
    seed: int | None = typer.Option(
        None,
        help="Seed the model sampler and the agent RNG. Best-effort, not a "
        "fixture: vLLM's seed fixes the sampler but not the batch-dependent "
        "float reductions of continuous batching, so two runs at one seed "
        "still diverge a little (measured on sk48: 1 prediction). Different "
        "seeds diverge about five times as much, which is what makes three "
        "seeds three samples — an A/B whose 'seeds' differed only by label "
        "produced byte-identical results and no variance to compare against.",
    ),
    agent_rev: str | None = typer.Option(
        None, help="Git rev of src/tgaer/agents/arc_agi3_kaggle.py to score."
    ),
    label: str | None = typer.Option(None, help="Name for this run in the output."),
    out: Path = typer.Option(
        Path("experiments/score_local.jsonl"), help="JSONL results path."
    ),
    workers: int = typer.Option(
        6,
        help="Games played concurrently. vLLM batches across them, so this is "
        "close to free until the server's --max-num-seqs is reached.",
    ),
    wandb: bool | None = typer.Option(
        None,
        help="Log metrics and sample frames to W&B. Defaults on when "
        "WANDB_API_KEY is set; the Kaggle kernel has no internet and "
        "never runs this script, so it is local-only by construction.",
    ),
) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    run_label = label or f"{agent_rev or 'worktree'}/{backend}"
    # Without this, three seeded runs share one label in W&B and in the JSONL,
    # which is why earlier runs hand-encoded the seed into --label.
    if seed is not None and label is None:
        run_label = f"{run_label}/seed{seed}"

    require_starter()  # the games and the SDK live there, not in this repo
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    available = [e.game_id.split("-")[0] for e in arc.get_environments()]
    game_ids = resolve_games(games, available)

    agent_cls = load_agent_class(agent_rev)
    agent_module = sys.modules[agent_cls.__module__]
    llm = make_backend(backend, agent_module, model, host, n_ctx, seed=seed)

    # Fail here rather than measure nothing. The agent degrades to random
    # actions when inference fails, so a dead server produces a full run of
    # plausible-looking rows — a whole scoring run was once spent against a
    # vLLM that had exited, and the output was indistinguishable from a result.
    if llm is not None:
        probe = llm.create_chat_completion(
            messages=[{"role": "user", "content": "reply with OK"}], max_tokens=4
        )
        reply = (probe["choices"][0]["message"].get("content") or "")[:40]
        console.print(f"[dim]backend reachable: {backend} -> {reply!r}[/]")
    if wandb is None:
        wandb = bool(os.getenv("WANDB_API_KEY"))
    tracker = WandbRun(
        wandb,
        run_label,
        {
            "agent_rev": agent_rev,
            "backend": backend,
            "model": model,
            "n_ctx": n_ctx,
            "max_steps": max_steps,
            "seed": seed,
            "games": len(game_ids),
        },
    )

    rows: list[dict[str, Any]] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"scoring {run_label}", total=len(game_ids))

        def run(game_id: str) -> dict[str, Any]:
            try:
                return play(agent_cls, game_id, arc, llm, max_steps, seed=seed)
            except Exception as exc:  # noqa: BLE001
                return {"game": game_id, "error": f"{type(exc).__name__}: {exc}"}

        # Concurrently, as the Swarm plays them: a vLLM server batches across
        # requests, so serial play left the GPU mostly idle and made a six-game
        # run take ~50 minutes instead of ~10 — slow enough to discourage the
        # iteration this harness exists for.
        with ThreadPoolExecutor(max_workers=min(workers, len(game_ids))) as pool:
            futures = {pool.submit(run, game_id): game_id for game_id in game_ids}
            for future in as_completed(futures):
                row = future.result()
                agent = row.pop("_agent", None)
                if agent is not None:
                    tracker.log_frames(agent_module, row["game"], agent.frames)
                tracker.log(
                    {
                        f"game/{row['game']}/levels": row.get("levels_completed") or 0,
                        f"game/{row['game']}/actions": row.get("actions") or 0,
                        **{
                            f"decisions/{k}": v
                            for k, v in (row.get("decisions") or {}).items()
                        },
                    }
                )
                rows.append(row)
                progress.advance(task)

    rows.sort(
        key=lambda r: game_ids.index(r["game"]) if r.get("game") in game_ids else 0
    )
    card = arc.get_scorecard()
    score = float(getattr(card, "score", 0.0) or 0.0)
    levels_total = sum(r.get("levels_completed") or 0 for r in rows)

    # Persist before rendering. A formatting bug in the summary discarded a
    # ten-minute run's results once already; the numbers matter more than the
    # table. getattr because only the ollama shim tracks overflows and calls —
    # the vLLM client is stateless so it stays safe across game threads.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "label": run_label,
                    "agent_rev": agent_rev,
                    "backend": backend,
                    "model": model if backend != "random" else None,
                    "n_ctx": n_ctx,
                    "max_steps": max_steps,
                    "score": score,
                    "levels_total": levels_total,
                    "overflows": getattr(llm, "overflows", None),
                    "llm_calls": getattr(llm, "calls", None),
                    "seed": seed,
                    "games": rows,
                }
            )
            + "\n"
        )
    decisions: dict[str, int] = {}
    for row in rows:
        for key, count in (row.get("decisions") or {}).items():
            decisions[key] = decisions.get(key, 0) + count
    # Rendering is decoration and the checks below it are not: a formatting bug
    # here once aborted the run before the inert-feature diagnostic could say
    # which features had silently died.
    try:
        render(rows, run_label, score, levels_total, llm)
    except Exception as exc:
        console.print(f"[yellow]render failed ({exc!r}); results are in {out}[/]")
    if decisions:
        console.print(f"decisions: {dict(sorted(decisions.items()))}")

    # Conditional features cannot be asserted on, but silence about them is how
    # undo went unnoticed, so name them explicitly either way. mechanic_note
    # belongs here rather than in the hard check: it fires only if the model
    # volunteers an optional tool argument, so a quiet model would otherwise get
    # a real scored run thrown away as a wiring bug.
    for name in ("undo", "click_search", "escape_forced", "mechanic_note"):
        if not decisions.get(name):
            console.print(f"[yellow]note: '{name}' never fired this run[/]")

    failures = []
    if inert := inert_features(agent_module, decisions):
        failures.append(
            f"enabled but never fired: {', '.join(inert)}\n"
            "  A feature that cannot fire looks exactly like one that did not help."
        )
    if broken := faults(decisions, sum(r.get("actions") or 0 for r in rows)):
        failures.append(
            f"fault counters above zero: {', '.join(broken)}\n"
            "  These count paths the agent should never reach. image_unavailable "
            "read 900 the run vision shipped inert, and nothing checked it."
        )
    if failures:
        console.print(
            "[bold red]FAILED:[/] "
            + "\n[bold red]FAILED:[/] ".join(failures)
            + "\nFix the wiring before reading this run as a result."
        )
        raise typer.Exit(1)
    tracker.log({"score": score, "levels_total": levels_total})
    tracker.finish()
    console.print(f"[dim]appended to {out}[/]")


if __name__ == "__main__":
    app()
