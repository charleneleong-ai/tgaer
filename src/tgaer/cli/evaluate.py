from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from tgaer.evaluation.dispatch import run_eval

# Load .env (ARC_API_KEY, ...) once at import so every entry path — the
# tgaer-eval console script, `python -m`, or importing `app` — sees it.
# Real env vars take precedence (load_dotenv override=False).
load_dotenv()

app = typer.Typer(
    help="Run a guarded TGAER eval loop, dispatched on the config's env.kind."
)


@app.callback()
def _main() -> None:
    """TGAER eval runner."""


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to an experiment YAML."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved config and exit."
    ),
) -> None:
    cfg = yaml.safe_load(config.read_text())
    if dry_run:
        typer.echo(yaml.dump(cfg, sort_keys=False))
        raise typer.Exit(code=0)
    try:
        result = run_eval(cfg)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(yaml.dump({"score": result.score, **result.details}, sort_keys=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
