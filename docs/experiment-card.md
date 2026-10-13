# Experiment card: changes on dfranzen's ARC-AGI-3 harness (phase 1, 2026-10-01 to 2026-10-13)

**Target.** ARC Prize 2026 ARC-AGI-3 Kaggle public score (RHAE on about 55 of 110 hidden interactive games), under
the competition limits: offline, 9 hours, one RTX PRO 6000 (96 GB), one submission per day. Fixed before testing.

**Base system, frozen.**
- Notebook: [dfranzen/arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)
  (Apache-2.0), harness source [da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution) at `10882e3`.
- Model: `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` v1 with drafter `dfranzen/albucino-qwen3-8-flash-next-drafter` v1.
- Server: SGLang "Pennyroyal" 2.5.3 wheelhouse (`dfranzen/pennyroyal-v253`), FP8 KV cache, 10 streams, MEMFRAC 0.96.
- Image: `gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c`.
- D' variant: [shiiin9/affectify-arc-31-54-in-a-single-sub](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub), byte-identical copy.

**Screens and budgets.**

| Screen | Unit | Runs | Noise |
|---|---|---|---|
| Public score | one submission, ~55 hidden games | 13 submissions (10-01 to 10-13) | identical copies sd ~2.8 (ours, 8 draws), 3.93 (community) |
| Local eval25 | 25 public games, scored-run time share | 19 runs | identical runs differ by up to ~7 |
| Stuck-level suite | 20 stuck levels x 2 repeats, 15 min each | 13 runs | identical baselines 9 and 9 of 40, sd ~3.7 |

**Denominators, failures included.** 23 mechanisms built; 17 tested locally, 12 on the stuck-level suite; 1 above the local noise band, 0
reproduced; 4 submitted as changes on dfranzen, 1 above plain's public mean, 0 significant. One submission
scored 0.00 (new kernel slug; its scored rerun failed) and is counted. Three local runs broke something
(CUDA OOM at MEMFRAC 0.975; 12 streams at -40% throughput; reasoning window at 24.27) and are counted.

**Filters, and when they were set.**
- Pre-registered: the suite bar (about +7 episodes, from the sd of identical baselines) and the public comparison
  against plain dfranzen's mean.
- Post hoc, declared: the re-attempt conversion audit (levels cleared after a re-attempt vs the ~25% rate of
  continuing) was added after v4's 50.18 looked high; it is evidence about that result, not a pre-registered gate.
- Post hoc, declared: "submit only from kernel slugs that have already scored" was added after the 0.00.

**Serving.** One Kaggle run of speculative-decoding settings on 100 replayed requests (two boots of the base setting as the noise band); two of six settings did not boot within 20 minutes and are counted.

**Not measured.** Per-game public scores (Kaggle returns one number); scored-rerun logs; the private score (withheld
until the 2026-11-02 close); 256 px images on the public score; any effect smaller than the screens' noise.

**Results file.** [`docs/experiments/arc-agi3-ledger.json`](experiments/arc-agi3-ledger.json); dashboard built by
[`scripts/build_artifact.py`](../scripts/build_artifact.py); report [`docs/report.pdf`](report.pdf).
