"""Per-game audit of every local 25-game run on dfranzen's harness, in the style of skarin's trajectory audit.

`collect` turns the run summaries into a committed CSV; `plot` draws the figures from that CSV alone, so they
reproduce without the run outputs.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs/figures/arc-agi3/runs_per_game.csv"
FIGS = CSV.parent
GAME = re.compile(
    r"^\s+(\w+)-\w+: score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)",
    re.M,
)
RUNS = {
    "dfranzen_eval25_output": ("unmodified 1", True),
    "dfranzen_eval25_rep2_output": ("unmodified 2", True),
    "dfz-eval25-reattempt_output": ("unmodified 3", True),
    "dfranzen_eval25_prior_output": ("level prior", False),
    "dfranzen_eval25_prior_fade_output": ("prior + fade", False),
    "dfranzen_eval25_prior_fade_rep2_output": ("prior + fade (rep)", False),
    "dfranzen_eval25_winrec_output": ("win records", False),
    "dfranzen_eval25_winrec_rep2_output": ("win records (rep)", False),
    "dfz-eval25-drain40k_output": ("40K drain", False),
    "dfz-eval25-endgame_v2_output": ("endgame rotation", False),
    "dfz-eval25-reattempt_v2_output": ("re-attempt v2", False),
    "dfz_eval25_hicache_s12_output": ("HiCache 12 streams", False),
    "pod_reattempt_v3_output": ("re-attempt v3", False),
    "pod_reattempt_v4_output": ("re-attempt v4", False),
    "pod_rwindow_output": ("reasoning window", False),
    "pod_hicache_ctx160_output": ("HiCache 160K", False),
    "pod_drain24_output": ("24K drain", False),
    "pod_hicache_reattempt_output": ("HiCache 160K + re-attempt", False),
    "pod_img256_output": ("256 px images", False),
}
INK, MUTED, ACCENT, LOSS = "#14181d", "#5d6874", "#0f6f7a", "#9e3324"
STYLE = {
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
}


class Audit:
    """One tidy table: a row per (run, game)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.df["frac"] = self.df.levels / self.df.n_levels
        self.order = (
            self.df.groupby("game")
            .frac.mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

    @staticmethod
    def collect(runs_dir: Path) -> pd.DataFrame:
        rows = []
        for folder, (label, unmodified) in RUNS.items():
            summary = runs_dir / folder / "summary.txt"
            if not summary.exists():
                typer.echo(f"skipping {folder}: no summary.txt")
                continue
            text = summary.read_text()
            for game, score, levels, n, actions, tokens in GAME.findall(text):
                rows.append(
                    {
                        "run": label,
                        "unmodified": unmodified,
                        "game": game,
                        "score": float(score),
                        "levels": float(levels),
                        "n_levels": int(n),
                        "actions": int(actions),
                        "tokens": int(tokens),
                    }
                )
        return pd.DataFrame(rows)

    def save(self, fig: plt.Figure, name: str) -> None:
        fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def heatmap(self) -> None:
        runs = list(dict.fromkeys(self.df.run))
        grid = self.df.pivot(index="game", columns="run", values="frac").loc[
            self.order, runs
        ]
        fig, ax = plt.subplots(figsize=(11, 8.2))
        im = ax.imshow(grid.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(runs)), runs, rotation=60, ha="right")
        ax.set_yticks(range(len(self.order)), self.order)
        ax.axvline(self.df[self.df.unmodified].run.nunique() - 0.5, color="white", lw=2)
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            label="share of the game's levels cleared",
        )
        ax.set_title(
            "Game difficulty sets the score: the same games sit at the top and the bottom in every run",
            loc="left",
            fontweight="bold",
        )
        self.save(fig, "01_levels_heatmap")

    def variability(self) -> None:
        g = self.df.groupby("game").frac.agg(["mean", "min", "max"])
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.scatter(g["mean"], g["max"] - g["min"], s=55, color=ACCENT)
        for game, r in g.iterrows():
            ax.annotate(
                game,
                (r["mean"], r["max"] - r["min"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xlabel(f"mean share of levels cleared, {self.df.run.nunique()} runs")
        ax.set_ylabel("range across runs (max - min)")
        swing = int(((g["max"] - g["min"]) >= 0.5).sum())
        ax.set_title(
            f"{swing} of {len(g)} games swing by half their levels or more between runs",
            loc="left",
            fontweight="bold",
        )
        self.save(fig, "02_game_variability")

    def efficiency(self) -> None:
        cleared = self.df[self.df.levels > 0]
        g = (
            cleared.assign(apl=cleared.actions / cleared.levels)
            .groupby("game")
            .agg(frac=("frac", "mean"), apl=("apl", "median"))
        )
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.scatter(g.frac, g.apl, s=55, color=ACCENT)
        for game, r in g.iterrows():
            ax.annotate(
                game,
                (r.frac, r.apl),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xlabel("mean share of levels cleared")
        ax.set_ylabel("median actions per cleared level")
        ax.set_title(
            "Games that clear few levels also spend the most actions on the ones they do",
            loc="left",
            fontweight="bold",
        )
        self.save(fig, "03_completion_vs_efficiency")

    def effects(self) -> None:
        base = self.df[self.df.unmodified].groupby("game").levels.mean()
        changes = [
            r for r in dict.fromkeys(self.df.run) if not r.startswith("unmodified")
        ]
        delta = np.array(
            [
                [
                    self.df[(self.df.run == r) & (self.df.game == g)].levels.sum()
                    - base[g]
                    for g in self.order
                ]
                for r in changes
            ]
        )
        fig, ax = plt.subplots(figsize=(11, 7))
        lim = np.abs(delta).max()
        im = ax.imshow(delta, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_yticks(range(len(changes)), changes)
        ax.set_xticks(range(len(self.order)), self.order, rotation=60, ha="right")
        fig.colorbar(
            im, ax=ax, fraction=0.03, pad=0.02, label="levels vs the unmodified mean"
        )
        ax.set_title(
            "Gains line up by game, not by change: they mark where the 2-run baseline happened to land",
            loc="left",
            fontweight="bold",
        )
        self.save(fig, "04_per_game_effects")

    def weak_games(self) -> None:
        weak = self.order[-6:]
        fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=False)
        for ax, game in zip(axes.flat, weak):
            d = self.df[self.df.game == game]
            n = d.n_levels.iloc[0]
            ax.hist(d.levels, bins=np.arange(-0.5, n + 1), color=LOSS, alpha=0.8)
            ax.set_title(
                f"{game}: {d.levels.mean():.1f}/{n} levels on average", fontsize=9
            )
            ax.set_xlim(-0.5, n + 0.5)
        fig.supxlabel("levels cleared in a run")
        fig.suptitle(
            f"Six weakest games: {self.df[self.df.game.isin(weak)].levels.sum():.0f} levels in "
            f"{len(set(self.df.run)) * 6} game-runs, none won",
            fontweight="bold",
            x=0.02,
            ha="left",
        )
        fig.tight_layout()
        self.save(fig, "05_weak_games")


app = typer.Typer()


@app.command()
def collect(runs_dir: Path = Path.home() / "arcagi3_runs") -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    Audit.collect(runs_dir).to_csv(CSV, index=False)
    typer.echo(f"wrote {CSV}")


@app.command()
def plot() -> None:
    audit = Audit(pd.read_csv(CSV))
    with plt.rc_context(STYLE):
        for draw in (
            audit.heatmap,
            audit.variability,
            audit.efficiency,
            audit.effects,
            audit.weak_games,
        ):
            draw()
    typer.echo(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    app()
