"""Build the ARC-AGI-3 ledger dashboard (docs/arc-agi3-ledger.html) from docs/experiments/arc-agi3-ledger.json."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import typer

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/experiments/arc-agi3-ledger.json"
OUT = ROOT / "docs/arc-agi3-ledger.html"

STYLE = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --fg:#14181d; --muted:#5d6874; --line:#d9dee5; --band:#e7ebf0;
  --accent:#0f6f7a; --gain:#12694a; --loss:#9e3324; --flat:#6b7480;
  --display:"IBM Plex Sans Condensed",-apple-system,"Segoe UI",sans-serif;
  --body:"IBM Plex Sans",-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1317; --panel:#161c22; --fg:#e6eaef; --muted:#93a0ad; --line:#293239; --band:#1e262d;
  --accent:#4fc3cf; --gain:#5ad39b; --loss:#ef8a76; --flat:#8b96a2; color-scheme:dark;}}
:root[data-theme="dark"]{
  --bg:#0f1317; --panel:#161c22; --fg:#e6eaef; --muted:#93a0ad; --line:#293239; --band:#1e262d;
  --accent:#4fc3cf; --gain:#5ad39b; --loss:#ef8a76; --flat:#8b96a2; color-scheme:dark;}
body{background:var(--bg);color:var(--fg);font-family:var(--body);line-height:1.55;margin:0;font-size:15px}
.wrap{max-width:62rem;margin:0 auto;padding-inline:16px;padding-block:2.4rem 4rem}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;letter-spacing:-.01em}
h1{font-size:clamp(1.8rem,5vw,2.7rem);font-weight:700;line-height:1.1}
h2{font-size:1.3rem;font-weight:700;margin-top:2.7rem;padding-top:1rem;border-top:2px solid var(--fg)}
p{margin:.6rem 0;max-width:66ch}
.take{font-weight:600;margin-top:.7rem}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:.6rem}
.lede{font-size:1.05rem;color:var(--muted);max-width:62ch;margin-top:.8rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(10rem,1fr));gap:.7rem;margin:1.8rem 0 .4rem}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.85rem .95rem;min-width:0}
.tile .k{font-family:var(--mono);font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-family:var(--display);font-size:1.6rem;font-weight:700;font-variant-numeric:tabular-nums;margin-top:.2rem}
.tile .n{font-size:.78rem;color:var(--muted)}
.chart{overflow-x:auto;margin:.9rem 0;background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:.6rem}
.chart svg{display:block;width:100%;height:auto;min-width:34rem}
.chart text{font-family:var(--mono);font-size:11px;fill:var(--fg)}
.chart .muted{fill:var(--muted)}
.scroll{overflow-x:auto;margin:1rem 0;border:1px solid var(--line);border-radius:3px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.88rem;min-width:30rem}
th,td{text-align:left;padding:.48rem .8rem;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:var(--mono);font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:500}
tbody tr:last-child td{border-bottom:none}
.pill{display:inline-block;font-family:var(--mono);font-size:.66rem;letter-spacing:.06em;text-transform:uppercase;
  padding:.12rem .4rem;border-radius:2px;border:1px solid currentColor;white-space:nowrap}
.pill.KEEP{color:var(--gain)}.pill.DISCARD{color:var(--loss)}.pill.BASELINE{color:var(--flat)}.pill.PROMOTE{color:var(--accent)}
ul{max-width:66ch;padding-left:1.2rem}li{margin:.35rem 0}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.8rem}
"""

FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600'
    '&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">'
)


class Svg:
    """Horizontal charts on one linear scale, coloured only through theme tokens."""

    LABEL_W, ROW_H, PAD, WIDTH = 290, 24, 40, 760

    @staticmethod
    def esc(s: Any) -> str:
        return html.escape(str(s))

    @classmethod
    def x(cls, v: float, lo: float, hi: float) -> float:
        return cls.LABEL_W + (v - lo) / (hi - lo) * (cls.WIDTH - cls.LABEL_W - cls.PAD)

    @classmethod
    def axis(
        cls, lo: float, hi: float, ticks: list[float], height: float, fmt: str
    ) -> str:
        out = []
        for t in ticks:
            x = cls.x(t, lo, hi)
            out.append(
                f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{height - 18}" stroke="var(--line)"/>'
                f'<text class="muted" x="{x:.1f}" y="{height - 4}" text-anchor="middle">{fmt.format(t)}</text>'
            )
        return "".join(out)

    @classmethod
    def refline(
        cls,
        v: float,
        lo: float,
        hi: float,
        height: float,
        label: str,
        colour: str = "var(--fg)",
    ) -> str:
        x = cls.x(v, lo, hi)
        anchor, dx = ("end", -4) if x > cls.WIDTH * 0.7 else ("start", 4)
        return (
            f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{height - 18}" stroke="{colour}" stroke-dasharray="4 3"/>'
            f'<text x="{x + dx:.1f}" y="14" text-anchor="{anchor}" fill="{colour}">{cls.esc(label)}</text>'
        )

    @classmethod
    def bars(
        cls,
        rows: list[tuple[str, float, str]],
        lo: float,
        hi: float,
        ticks: list[float],
        fmt: str,
        ref: tuple[float, str] | None = None,
        band: tuple[float, float] | None = None,
    ) -> str:
        h = 24 + len(rows) * cls.ROW_H + 22
        parts = [f'<svg viewBox="0 0 {cls.WIDTH} {h}" role="img">']
        if band:
            x0, x1 = cls.x(band[0], lo, hi), cls.x(band[1], lo, hi)
            parts.append(
                f'<rect x="{x0:.1f}" y="20" width="{x1 - x0:.1f}" height="{h - 42}" fill="var(--band)"/>'
            )
        parts.append(cls.axis(lo, hi, ticks, h, fmt))
        for i, (label, v, colour) in enumerate(rows):
            y = 24 + i * cls.ROW_H
            x0, x1 = cls.x(lo, lo, hi), cls.x(v, lo, hi)
            parts.append(
                f'<text x="{cls.LABEL_W - 8}" y="{y + 14}" text-anchor="end">{cls.esc(label)}</text>'
                f'<rect x="{x0:.1f}" y="{y + 4}" width="{max(x1 - x0, 1):.1f}" height="13" fill="{colour}"/>'
                f'<text x="{x1 + 5:.1f}" y="{y + 14}">{fmt.format(v)}</text>'
            )
        if ref:
            parts.append(cls.refline(ref[0], lo, hi, h, ref[1]))
        parts.append("</svg>")
        return '<div class="chart">' + "".join(parts) + "</div>"

    @classmethod
    def dots(
        cls,
        groups: list[tuple[str, list[float], str]],
        lo: float,
        hi: float,
        ticks: list[float],
        ref: tuple[float, str],
    ) -> str:
        h = 24 + len(groups) * cls.ROW_H + 22
        parts = [
            f'<svg viewBox="0 0 {cls.WIDTH} {h}" role="img">',
            cls.axis(lo, hi, ticks, h, "{:.0f}"),
        ]
        for i, (label, vals, colour) in enumerate(groups):
            y = 24 + i * cls.ROW_H + 10
            mean = sum(vals) / len(vals)
            parts.append(
                f'<text x="{cls.LABEL_W - 8}" y="{y + 4}" text-anchor="end">{cls.esc(label)} '
                f'<tspan class="muted">{mean:.2f} (n={len(vals)})</tspan></text>'
            )
            for v in vals:
                parts.append(
                    f'<circle cx="{cls.x(v, lo, hi):.1f}" cy="{y}" r="5" fill="{colour}" fill-opacity=".85"/>'
                )
        parts.append(cls.refline(ref[0], lo, hi, h, ref[1]))
        parts.append("</svg>")
        return '<div class="chart">' + "".join(parts) + "</div>"


class Dashboard:
    """Sections of the ledger page, each a chart with its one-line takeaway above it."""

    COLOUR = {
        "KEEP": "var(--gain)",
        "DISCARD": "var(--loss)",
        "BASELINE": "var(--flat)",
        "PROMOTE": "var(--accent)",
    }

    def __init__(self, d: dict[str, Any]) -> None:
        self.d = d
        self.base = d["baseline"]

    def headline(self) -> str:
        pub = self.d["public"]
        dp = [p["score"] for p in pub if p["group"] == "D'"]
        tiles = [
            (
                "best public draw",
                f"{max(p['score'] for p in pub):.2f}",
                "D' copy, 2026-10-07",
            ),
            (
                "baseline",
                f"{self.base['public_mean']:.2f}",
                "plain dfranzen, mean of 3",
            ),
            ("D' mean", f"{sum(dp) / len(dp):.2f}", f"n={len(dp)}, p~0.34 vs plain"),
            (
                "not moved",
                f"{self.base['suite_episodes']:.0f} / 40",
                "stuck levels cleared, 11 arms",
            ),
            (
                "leader",
                f"{self.d['leaderboard']['top'][0][1]:.2f}",
                self.d["leaderboard"]["top"][0][0],
            ),
        ]
        cells = "".join(
            f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{Svg.esc(n)}</div></div>'
            for k, v, n in tiles
        )
        return f'<div class="tiles">{cells}</div>'

    def public(self) -> str:
        order = ["plain", "D'", "D' + re-attempt", "endgame rotation", "level prior"]
        colour = {"plain": "var(--flat)", "D'": "var(--gain)"}
        groups = [
            (
                g,
                [p["score"] for p in self.d["public"] if p["group"] == g],
                colour.get(g, "var(--loss)"),
            )
            for g in order
        ]
        chart = Svg.dots(
            groups,
            18,
            32,
            [18, 20, 22, 24, 26, 28, 30, 32],
            (self.base["public_mean"], "plain mean 26.95"),
        )
        return (
            "<h2>Public score: only D' sits above plain dfranzen</h2>"
            '<p class="take">Every change we added on top of dfranzen scored at or below the plain copies; D\' leads by 1.7 points, inside the noise of three draws.</p>'
            + chart
            + "<p>Each dot is one submission scored on about 55 hidden games. Identical copies vary by about 4 points. "
            "The new-kernel submission that scored 0.00 is left out: it failed in the scored rerun, and the same code from the proven kernel scored 26.05.</p>"
        )

    def local(self) -> str:
        rows = [
            (r["name"], r["score"], self.COLOUR[r["verdict"]])
            for r in sorted(self.d["local"], key=lambda r: -r["score"])
        ]
        b = self.d["baseline"]["local_mean"]
        chart = Svg.bars(
            rows,
            20,
            55,
            [20, 25, 30, 35, 40, 45, 50, 55],
            "{:.1f}",
            ref=(b, "unmodified mean"),
            band=(b - 7, b + 7),
        )
        return (
            "<h2>Local 25-game runs: nothing cleared the noise band and held</h2>"
            '<p class="take">Of 17 local results, 13 land inside the ±7 band of identical runs. The three below it broke something; the one above it (re-attempt v4, 50.18) fell to 42.56 when stacked.</p>'
            + chart
            + "<p>Shaded: the unmodified mean ±7. Local gains have not carried over to the public score.</p>"
        )

    def suite(self) -> str:
        rows = [
            (
                s["name"],
                s["episodes"],
                "var(--flat)" if s["name"].startswith("baseline") else "var(--accent)",
            )
            for s in self.d["suite"]
        ]
        chart = Svg.bars(
            rows,
            0,
            20,
            [0, 4, 8, 12, 16, 20],
            "{:.0f}",
            ref=(16, "a real gain needs ~16"),
            band=(9 - 3.7, 9 + 3.7),
        )
        return (
            "<h2>Stuck-level suite: no arm has converted stuck levels</h2>"
            '<p class="take">Eleven arms, from prompts to harness probes to restarts, clear 5 to 12 of 40 stuck-level episodes against a baseline of 9.</p>'
            + chart
            + "<p>20 levels where full runs got stuck, each replayed twice from the recorded history. Shaded: 9 ± one standard deviation between identical baselines.</p>"
        )

    def gains(self) -> str:
        c = self.d["reattempt_conversion"]
        rows = [
            ("re-attempt v4", c["v4"][0] / c["v4"][1] * 100, "var(--accent)"),
            ("re-attempt v3", c["v3"][0] / c["v3"][1] * 100, "var(--accent)"),
            ("keep playing (baseline)", c["baseline_continue"] * 100, "var(--flat)"),
        ]
        conv = Svg.bars(rows, 0, 50, [0, 10, 20, 30, 40, 50], "{:.0f}%")
        tps = [
            (r["name"], r["tok_s"], self.COLOUR[r["verdict"]])
            for r in self.d["local"]
            if r.get("tok_s")
        ]
        thr = Svg.bars(
            sorted(tps, key=lambda r: -r[1]),
            300,
            700,
            [300, 400, 500, 600, 700],
            "{:.0f}",
            ref=(self.base["tok_s"], "unmodified"),
        )
        return (
            "<h2>Where the local highs came from</h2>"
            '<p class="take">Re-attempts cleared their stuck level about as often as just carrying on, so the 50.18 run was not caused by them.</p>'
            + conv
            + '<p class="take">Throughput moved for three changes: smaller board images helped by 5%, frequent history trims cost 30%.</p>'
            + thr
        )

    def failures(self) -> str:
        f = self.d["failure_modes"]
        rows = [
            (name, n, "var(--loss)" if i == 0 else "var(--flat)")
            for i, (name, n) in enumerate(f["counts"])
        ]
        return (
            "<h2>How it fails</h2>"
            f'<p class="take">{f["missing_points_uncleared_share"]:.0%} of the missing points are levels never cleared; most stuck levels follow a wrong model of the mechanics.</p>'
            + Svg.bars(rows, 0, 20, [0, 5, 10, 15, 20], "{:.0f}")
            + f"<p>Source: {Svg.esc(f['source'])}.</p>"
        )

    def funnel(self) -> str:
        rows = [(name, n, "var(--accent)") for name, n in self.d["funnel"]]
        return (
            "<h2>The funnel</h2>"
            '<p class="take">21 mechanisms built; one cleared the local noise band and did not reproduce; one configuration beat plain publicly, not significantly.</p>'
            + Svg.bars(rows, 0, 25, [0, 5, 10, 15, 20, 25], "{:.0f}")
        )

    def ledger(self) -> str:
        rows = "".join(
            f'<tr><td>{Svg.esc(n)}</td><td><span class="pill {v}">{v}</span></td><td>{Svg.esc(w)}</td></tr>'
            for n, v, w in self.d["verdicts"]
        )
        lb = "".join(
            f"<li>{Svg.esc(t)}: {s:.2f}</li>" for t, s in self.d["leaderboard"]["top"]
        )
        ne = "".join(f"<li>{Svg.esc(x)}</li>" for x in self.d["not_established"])
        return (
            "<h2>Verdict ledger</h2>"
            '<div class="scroll"><table><thead><tr><th>experiment</th><th>verdict</th><th>why</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
            f"<h2>Not established</h2><ul>{ne}</ul>"
            f"<h2>Leaderboard context ({self.d['leaderboard']['date']})</h2>"
            f"<ul>{lb}</ul><p>{self.d['leaderboard']['teams_at_or_above_32']} teams sit at 32 or above, mostly draws of dfranzen copies. "
            "None of the top five has disclosed a method.</p>"
        )

    def page(self) -> str:
        body = "".join(
            [
                '<div class="eyebrow">ARC Prize 2026 · ARC-AGI-3 · '
                + self.d["as_of"]
                + "</div>",
                "<h1>dfranzen's harness, and what moved the public score</h1>",
                '<p class="lede">Twenty-one changes on top of the strongest public notebook, tested locally, on a stuck-level suite '
                "and in public submissions. One configuration (D') leads plain dfranzen; nothing else moved the hidden-game score.</p>",
                self.headline(),
                self.public(),
                self.local(),
                self.suite(),
                self.gains(),
                self.failures(),
                self.funnel(),
                self.ledger(),
                f"<footer>Built by scripts/build_artifact.py from docs/experiments/arc-agi3-ledger.json · status as of {self.d['as_of']}.</footer>",
            ]
        )
        return f'<title>ARC-AGI-3 Ablation Ledger</title>\n{FONTS}\n<style>{STYLE}</style>\n<div class="wrap">{body}</div>\n'


def main(ledger: Path = LEDGER, out: Path = OUT) -> None:
    out.write_text(Dashboard(json.loads(ledger.read_text())).page())
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
