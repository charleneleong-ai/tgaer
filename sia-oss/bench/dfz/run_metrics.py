"""Low-noise measures for dfranzen eval runs: throughput, KV pressure, turns, parking."""

from __future__ import annotations

import re
import statistics as st
from pathlib import Path
from typing import Any

import typer
from harness import EvalRun

SNAP = re.compile(
    r"^\s+(\w{4})-\w+: score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)"
)
DECODE = re.compile(
    r"Decode batch.*?#running-req: (\d+).*?token usage: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)"
)
PREFILL = re.compile(r"Prefill batch.*?#new-token: (\d+).*?#cached-token: (\d+)")
PARKED_MIN = 10

Series = dict[str, list[tuple[float, float, int, int]]]


class RunMetrics:
    """Summary measures for one run output directory."""

    def __init__(self, root: Path) -> None:
        self.run = EvalRun(root)

    @staticmethod
    def game_series(records: list[dict[str, Any]]) -> Series:
        series: Series = {}
        for rec in records:
            for line in rec["data"].splitlines():
                if m := SNAP.match(line):
                    g, score, _, _, actions, tokens = m.groups()
                    series.setdefault(g, []).append(
                        (rec["time"], float(score), int(actions), int(tokens))
                    )
        return series

    @staticmethod
    def parking(series: Series, won: set[str]) -> dict[str, float]:
        """Minutes before the last snapshot that an unfinished game stopped spending tokens."""
        end = max(t for pts in series.values() for t, *_ in pts)
        parked = {}
        for g, pts in series.items():
            if g in won:
                continue
            pairs = zip(reversed(pts), reversed(pts[:-1]), strict=False)
            last_change = next(
                (t for (t, _, _, tok), (_, _, _, prev) in pairs if tok != prev),
                pts[0][0],
            )
            parked[g] = (end - last_change) / 60
        return parked

    def server(self) -> dict[str, float]:
        path = self.run.root / "serve.log"
        if not path.exists():
            return {}
        text = path.read_text(errors="ignore")
        rows = [(int(r), float(u), float(g)) for r, u, g in DECODE.findall(text)]
        pre = [(int(n), int(c)) for n, c in PREFILL.findall(text)]
        new, cached = sum(n for n, _ in pre), sum(c for _, c in pre)
        return {
            "running_med": st.median(r for r, _, _ in rows),
            "kv_med": st.median(u for _, u, _ in rows),
            "kv_p90": st.quantiles([u for _, u, _ in rows], n=10)[-1],
            "gen_tok_s_med": st.median(g for _, _, g in rows),
            "prefix_hit": cached / max(1, new + cached),
            "retract": len(re.findall(r"retract", text, re.I)),
        }

    def summarise(self) -> list[str]:
        won = {g for g, f in self.run.finished().items() if f.won}
        parked = self.parking(self.game_series(self.run.log_records), won)
        mean = self.run.summary_field("mean score")
        tps = self.run.summary_field("generated tokens/sec")
        tokens = self.run.summary_field("total tokens")
        big = {
            g: round(m)
            for g, m in sorted(parked.items(), key=lambda kv: -kv[1])
            if m >= PARKED_MIN
        }
        lines = [
            f"== {self.run.root.name}",
            f"   score {mean} | won {len(won)} | generated tokens {int(tokens) / 1e6:.2f}M at {tps} tok/s",
            f"   parked >={PARKED_MIN} min: {len(big)} games, {sum(parked.values()):.0f} game-min total  {big}",
        ]
        if s := self.server():
            fmt = (
                f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                for k, v in s.items()
            )
            lines.append("   server " + "  ".join(fmt))
        return lines


def main(runs: list[Path]) -> None:
    for root in runs:
        print("\n".join(RunMetrics(root).summarise()))


if __name__ == "__main__":
    typer.run(main)
