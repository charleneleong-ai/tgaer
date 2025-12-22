from __future__ import annotations

import pathlib
import sys

import typer
import yaml

from tgaer.experiments.runner import run_experiment

app = typer.Typer(help="TGAER – Toward General-Purpose Abstraction & Embodied Reasoning")


def _load_experiment_config(name: str) -> dict:
    this_file = pathlib.Path(__file__).resolve()
    project_root = this_file.parents[3]
    cfg_path = project_root / "configs" / "experiments" / f"{name}.yaml"

    if not cfg_path.exists():
        typer.echo(f"[ERROR] Config not found: {cfg_path}", err=True)
        raise typer.Exit(code=1)

    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


@app.command()
def run(
    experiment: str = typer.Argument(..., help="Experiment name, e.g. 'arc_hybrid' or 'orak_geq_opt'."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only load and print config, do not execute."),
) -> None:
    cfg = _load_experiment_config(experiment)
    typer.echo(f"[INFO] Loaded experiment config: {experiment}")

    if dry_run:
        typer.echo("[INFO] Dry run – configuration:")
        typer.echo(yaml.dump(cfg, sort_keys=False))
        raise typer.Exit(code=0)

    run_experiment(cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
