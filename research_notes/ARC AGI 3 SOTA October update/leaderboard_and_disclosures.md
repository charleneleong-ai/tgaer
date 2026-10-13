# ARC-AGI-3 Kaggle leaderboard and team disclosures, 2026-10-04 to 2026-10-11

Snapshot used throughout: the Kaggle public leaderboard CSV downloaded via the Kaggle CLI, file stamped `2026-10-11T02:48:47` UTC, 4,126 teams ([Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)). Baseline for "change" is the 2026-10-06 state recorded in `reports/ARC AGI 3 vLLM port SOTA.md`.

## Current top of the public leaderboard and movement since 10-04/06

### Takeaway
The top changed hands: Yi-Chia Chen (threerabbits) jumped 48.59 → **62.96** on 10-10 and now leads Tufa Labs (55.89 → 56.52) by 6.4 points. A new tier appeared between 40 and 45 (mtg 45.00, Majkel1337 42.66, NVARC3 40.97), and the ≥32 band roughly doubled again (55 → 115 teams), almost all from 10-09/10-10 submissions.

### Cited Findings
- Top 20 as of 2026-10-11 02:48 UTC (score, submission count, last submission) — [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard):

| Rank | Team | Score | Subs | Last sub | Change vs 10-06 |
|---|---|---|---|---|---|
| 1 | Yi-Chia Chen (threerabbits) | **62.96** | 27 | 10-10 | was 2nd at 48.59 (+14.37) |
| 2 | Tufa Labs (dlorah, driessmit1, infinitecreativity, jeroencottaar, pressman1, stefano1283) | 56.52 | 159 | 10-09 | was 1st at 55.89 (+0.63) |
| 3 | mtg (michaeltgao) | 45.00 | 65 | 10-10 | was 38.33 (+6.67) |
| 4 | Majkel1337 | 42.66 | 13 | 10-09 | new to top band |
| 5 | NVARC3 (cpmpml, darraghdog, eladsarafian, galkaplun, sorokin, yeyinzhu) | 40.97 | 34 | 10-10 | new to top band |
| 6 | the last dance (gklambauer, lukasaichberger, …) | 39.30 | 77 | 10-09 | unchanged best |
| 7 | Son & Mark & Ronen (markbarney, scizical, sonphamorg) | 39.24 | 74 | 10-10 | new |
| 8 | dreach.ai | 39.11 | 34 | 10-09 | new |
| 9 | Nhan Duc Nguyen | 38.84 | 16 | 10-10 | new |
| 10 | artificialagencylab.com | 38.62 | 49 | 10-09 | new |
| 11–20 | fshindo 38.20, 復活の混テキスト 38.15, SparseTech 37.98, keithtyser 37.22, _hans 37.07, Lord Han Solo 36.61, AI_hwlee 36.54, YUTO KOJIMA 36.14, Andrew Reed 36.14, lalalia 35.79 | | | | Lord Han Solo unchanged at 36.61 |

- Band counts now: ≥50: 2, ≥45: 3, ≥40: 5, ≥35: 22 (was 11), ≥32: 115 (was 55), ≥28: 725 (was 402). Rank 50 = 33.37, rank 100 = 32.29 — [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard) (CLI CSV).
- 93 of the top 115 teams made their best submission on 10-09 or 10-10 — same CSV.
- NVARC3's roster includes `sorokin` and `cpmpml` (Ivan Sorokin, Jean-Francois Puget — the NVARC team that won the 2025 Kaggle ARC Prize) and `yeyinzhu` (an author on NVIDIA's AVO ARC-AGI-3 work) — [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard); identity mapping of usernames to people is **unverified inference** from a [search summary of NVIDIA's AVO result](https://runtimewire.com/article/nvidia-avo-arc-agi-3-perfect-public-score) (secondary source).
- Reference public scores of known public-notebook authors: dfranzen 27.89 (91 subs, still submitting 10-10), AFF AI CLUB/shiiin9 33.92 (10-10, up from 31.54), AlekseiProvorov 31.73, zhangyue-xmdx 31.76, Sujan Majhi 31.93, Heliosli 32.60, amatlas/Zhilei Qiu 30.82, kishizaki-sci 30.51, Vladimir Yakunin 29.30, hknight 28.52, rellik13 24.31, Tong Hui Kang 26.88 — [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard).
- Secondary coverage still reports the 10-04 state (Tufa 55.89, Yi-Chia 48.59) and frames the jump as scaffolding-driven — [TechTimes, 2026-10-06](https://www.techtimes.com/articles/328542/20261006/arc-agi-3-scaffolding-beats-model-upgrades-same-ai-two-settings-49x-score-gain.htm) (secondary; no new primary info).

### Inferences
- Yi-Chia's +14.4 in 27 total submissions is the largest single move since the Duck/dfranzen wave and cannot be draw noise (public sd ~3.9 on identical code). Tufa's +0.6 over 159 submissions looks like a plateau or a best-of-many draw.
- The 32–39 band remains dominated by dfranzen-derived copies (see notebook section: two of the top public notebooks are byte-identical to dfranzen on the submission path yet score 31.8–32.6). With ~1,500 copies drawing, the band's upper edge rises by order statistics alone, so a 38–39 entry is not evidence of a better method.
- The 40–45 tier (mtg, Majkel1337, NVARC3) sits ~6+ points above the copy band's apparent ceiling and is probably real method improvement, but none has disclosed anything.

### Gaps
- No per-submission history for other teams; "movement" is only best-score vs the 10-06 report. The unofficial tracker with history ([arc.markbarney.net](https://arc.markbarney.net/kaggle-leaderboard)) did not return content to WebFetch.
- "face-of-agi" (37.94 on 10-06) does not appear by that name in the 10-11 CSV; likely renamed or merged — unverified.

## Disclosures by teams scoring ≥ ~32

### Takeaway
No team above ~34 has published a method since 10-04. The only fresh technical artefacts come from lower-scoring or non-ranked participants: a public GitHub fork distilling GPT-6.1 Sol trajectories into Flash-Next (LamDang), an H200 port and community-game trace-collection branch (junpenglao), and a vLLM/Pennyroyal Duck fork (arcainion). Tufa Labs reportedly stated it will not open-source for Milestone 2.

### Cited Findings
- Tufa Labs' [duck-harness](https://github.com/Tufalabs/duck-harness) repo: last push 2026-07-01; nothing new — GitHub API.
- A Kaggle thread titled "We will not be open-sourcing for the second milestone prize" by Jeroen Cottaar (Tufa) exists with 41 votes / 45 comments — surfaced only via a search-engine snapshot of the [discussion index](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion); post date and contents **unverified** (Kaggle discussion pages render client-side and returned no content to WebFetch).
- No write-ups, repos or X threads found for Yi-Chia Chen, mtg, Majkel1337, NVARC3, dreach.ai, artificialagencylab, Son & Mark & Ronen — multiple searches ([e.g.](https://x.com/arcprize/status/2104590501915787290)); the last ARC Prize "new 1st place" post found is Yi-Chia's 28.34% on 2026-09-28 ([ARC Prize on X](https://x.com/arcprize/status/2104590501915787290)).
- Markbarney (team "Son & Mark & Ronen", 39.24) runs the unofficial leaderboard tracker and a public project page — [arc.markbarney.net](https://arc.markbarney.net/) — but no method write-up was retrievable.
- [LamDang/arc-agi-3-solution-da-fr](https://github.com/LamDang/arc-agi-3-solution-da-fr) (fork of dfranzen, pushed 2026-10-11): commits on 10-10 add "community ARC-AGI-3 games as submodules", an "NVIDIA gpt-6.1-sol driver" that plays community games with GPT-6.1 Sol, "resumable target-token trajectory training", and an `exp/sft-flash-next` experiment comparing Sol replies vs generated-thinking replies by token NLL — i.e. **SFT of Qwen3.8-Flash-Next on frontier-model trajectories from non-competition games**. No LB entry under that name found; results not public — GitHub API/README.
- [junpenglao/arc-agi-3-solution](https://github.com/junpenglao/arc-agi-3-solution) (fork of dfranzen, 10-08): branches `franzen-h200-runtime` (H200/SM90 serving, QSA MTP fallbacks) and `franzen-community-coverage` ("pinned game rosters and complete trace collection", "select community coverage batches") — also points at trace collection on community games; no score — GitHub API.
- [arcainion/duck-harness](https://github.com/arcainion/duck-harness) switched Kaggle inference to Pennyroyal Flash-Next and hardened vLLM startup (10-09/10); team ARCAINION PRIME is at 0.17 — GitHub API; [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard).
- [QingdiMeng/duck-harness](https://github.com/QingdiMeng/duck-harness): "Fix context retention and add DeepSeek thinking support" (10-08); team at 0.33 — GitHub API.

### Inferences
- Two independent forks (LamDang, junpenglao) building trace collection on community ARC-AGI-3 games suggest **offline distillation/SFT from frontier trajectories** is the emerging lever outside the copy band; it is plausible (not shown) that some 40+ teams use fine-tuned weights, which would explain gains no harness copy reproduces. Rules allow offline training as long as the kernel is offline.
- NVARC's 2025 recipe was exactly offline synthetic-data training of a small model; NVARC3 at 40.97 with that roster is consistent with a training-based approach — speculation.

### Gaps
- Kaggle discussion threads since 10-04 could not be read (client-side rendering; Kaggle CLI has no discussion command). Thread IDs >746010 are unknown.
- LamDang's and junpenglao's Kaggle team identities are not established; their artefacts may not correspond to any scored entry.

## New public Kaggle notebooks since 10-04, especially > ~31

### Takeaway
No new public notebook demonstrably beats dfranzen. The highest-ranked new public notebooks are either byte-identical to dfranzen on the scored path (Heliosli 32.60, zhangyue-xmdx 31.76) or prompt/memory add-ons with unattributed scores (Sujan Majhi "v22 Apex", 31.93). shiiin9's team has moved to 33.92 but the public D' notebook is unchanged.

### Cited Findings
Kernel order from `kaggle kernels list --sort-by scoreDescending` (top of list): dfranzen M2, heliosli, sujanmajhisuzan, zhangyues0000, leoprovorov handbook, skarin audit, shiiin9 D′ — [Kaggle code tab](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/code). Diffs below were computed locally against [dfranzen/arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution):

| Notebook | Team best public | What differs from dfranzen |
|---|---|---|
| [heliosli/try-merging-it-you-might-get-lucky-and-win-gold](https://www.kaggle.com/code/heliosli/try-merging-it-you-might-get-lucky-and-win-gold) (run 10-01, 23 votes) | 32.60 (10-04, 2 subs) | 12 diff lines: only demo-mode settings (`n_passes` 4→1, demo game exclusions, 25-min demo cap). Submission path identical. |
| [zhangyues0000/arc-agi-3-m2-franzen-fork](https://www.kaggle.com/code/zhangyues0000/arc-agi-3-m2-franzen-fork) (10-09) | 31.76 (10 subs) | 4 diff lines, demo-only (`n_passes`, 20-min demo cap; "rerun branch untouched"). |
| [sujanmajhisuzan/arc-agi-3-m2-top-submission](https://www.kaggle.com/code/sujanmajhisuzan/arc-agi-3-m2-top-submission) (run 10-11, 75 votes) | 31.93 (10-08, 8 subs) | "v22 Apex": claims "Proven 31.93 base"; adds solved-level episodic memory (capped 3 levels), turn-12 anti-loop UNDO/RESET audit, death-loop warning, efficiency-first prompt, MEMFRAC 0.88, MAXREQ 10, input-path fallbacks, tolerant patch apply. Which version produced 31.93 is not stated. |
| [prvsiyan/arc-agi-3-stock-taaf-action7-shadow](https://www.kaggle.com/code/prvsiyan/arc-agi-3-stock-taaf-action7-shadow) (10-10) | 30.65 | dfranzen V1 rerun with attribution, switched to Jeroen Cottaar's MIT bundle `jeroencottaar/taaf-kaggle-source-share`, watchdog, robustness pack; reports its own unchanged rerun at **27.83**, and "one-shot reruns of unchanged copies averaged about 26". |
| [shiiin9/affectify-arc-31-54-in-a-single-sub](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub) | AFF AI CLUB 33.92 (10-10, 40 subs) | Public version still the D′ slot-priority patch (single change; "how we arrived at this formula … keeping to ourselves"). The 33.92 team score's code is not public. |

- Other new notebooks since 10-04 (no evidence of >31): leoprovorov F001/F002 "fable layer" (10-08; team 31.73), [vladimiryakunin/df-wm](https://www.kaggle.com/code/vladimiryakunin/df-wm) updated 10-09 (team 29.30), [kozykappa FlashNext Duck audited](https://www.kaggle.com/code/kozykappa/arc-agi-3-flashnext-duck-audited-submission) (team 28.39), [superstringdev NVFP4 SGLang public](https://www.kaggle.com/code/superstringdev/arc-agi-3-nvfp4-sglang-public) (team 30.51), [bang1850 sovereign agent](https://www.kaggle.com/code/bang1850/arc-prize-2026-sovereign-agent) (55 votes; team 29.08), [skarin trace parser/token accounting](https://www.kaggle.com/code/skarin/arc-agi-3-trace-parser-token-accounting) (10-08), D′ forks by lwq255/daydreamerthinks (team lwq255 28.24) — [Kaggle code tab](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/code), [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard).

### Inferences
- Two byte-identical-on-submission copies at 32.60 and 31.76, plus prvsiyan's unchanged 27.83, widen the observed range of identical dfranzen code to ~27.6–34.3. Any public notebook in 31–33 is consistent with a lucky draw of the baseline.
- Sujan Majhi's add-ons (memory, anti-loop prompts) are in the class already found null (advice text, journals); its 31.93 is inside the copy distribution. Its stated `TORCH_CUDA_ARCH_LIST 8.9;9.0` on an SM120 card looks like a misconfiguration claim — unverified.

### Gaps
- Kaggle CLI does not expose a notebook's own public score; team best scores were used instead, which may come from a different version or notebook.

## Rule, deadline, prize or announcement changes since 10-06

### Takeaway
No changes found. Deadline stays 2026-11-02, paper deadline 11-08, winners 12-04; no Milestone #3; arcprize.org blog has no post after 09-03.

### Cited Findings
- Dates: competition open 03-25, Milestone #1 06-30, Milestone #2 09-30, final submission 11-02, paper deadline 11-08, winners 12-04; open-source release required before private scores; own code CC0/MIT-0, third-party at least Apache-2.0/GPLv3-style — [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026).
- arcprize.org blog's newest post is 09-03 ("OpenAI's GPT-6 Astra on ARC-AGI-3"); nothing dated after 09-25 — [arcprize.org/blog](https://arcprize.org/blog).
- Milestone #2 winners (announced ~10-01): Daniel Franzen 27.9% ($25K), Lord Han Solo 23.8% ($7.5K), Lohit Siriki 22.5% ($5K) — per [TechTimes](https://www.techtimes.com/articles/328542/20261006/arc-agi-3-scaffolding-beats-model-upgrades-same-ai-two-settings-49x-score-gain.htm) (secondary; consistent with prior report).

### Inferences
- With 22 days left and ~9-hour GPU queues, no structural change is pending; the remaining open question is the licence of Qwen3.8-Flash-Next (`qwen-community-1.0`) for prize eligibility, which prvsiyan's notebook still flags as unresolved and works around with a pre-decided fallback.

### Gaps
- Kaggle pinned host posts (María Cruz) could not be read for any post-10-06 edits; a rule clarification posted only in the forum would be missed.
