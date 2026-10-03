# dfranzen Milestone 2 notebook: a scheduler change that lost on the hidden games

**Outcome: rejected.** The change scored **20.60** public against **28.39** for the unmodified
notebook submitted the next day, despite looking better locally. We submit the plain notebook.

We submitted [dfranzen's public Milestone 2 notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)
(best public 27.89, Apache-2.0, source [da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution))
with a two-part change to its priority scheduler, then the plain notebook as its control. The notebook itself is not vendored here; it lives
as the private kernel `charyeezy/arc-agi-3-dfranzen-m2-eval25-prior-fade` v1, submitted 2026-10-02
as ref 56761652.

## Diagnosis

On all 25 public games the unmodified notebook scored 40.01 locally and finished 5 games. Games
near the finish were *parked*, not stuck: tr87 got no turns for its last 43 minutes on a final
level worth 28.6 points, ka59 none for 24 minutes. Two biases in `priority_scheduler.priority_value`
cause it:

1. **Fixed human prior.** Expected efficiency is `(h / (h + actions))**2` with `h = 25` on every
   level. Late-level human baselines are 80–150 actions (tr87 L6 is 146), so tr87 at 164 actions
   scored 0.018 against a true ~0.79 and dropped out of the queue.
2. **No future value on the final level.** The `remaining` tail bonus is 8/7/5/0, so a game stuck
   on level 1 keeps +8 while a game on its last level gets 0. The tail only fades over the last
   `ARC3_PRIORITY_TAIL_FADE_FRACTION = 0.2` of the run.

## Change

`h` becomes the median public-25 human baseline for the level number (levels 5+ pooled), and the
tail fades over the last 75% of the run instead of 20%. Games run as threads in-process, so
rebinding `tool_agent.priority_value` takes effect.

```diff
# setup_env.update({...}) cell
-    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
+    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.75',

# end of the benchmark-config cell
+USE_LEVEL_PRIOR = True
+LEVEL_HUMAN_PRIOR = {1: 30, 2: 54, 3: 51, 4: 54}
+LEVEL_HUMAN_PRIOR_LATE = 96
+if USE_LEVEL_PRIOR:
+    import inference.agent.tool_agent as _tool_agent
+    _stock_priority_value = _tool_agent.priority_value
+
+    def _priority_value_level_prior(state, **kwargs):
+        kwargs["human_actions"] = float(
+            LEVEL_HUMAN_PRIOR.get(max(1, state.level), LEVEL_HUMAN_PRIOR_LATE)
+        )
+        return _stock_priority_value(state, **kwargs)
+
+    _tool_agent.priority_value = _priority_value_level_prior
```

The eval kernels also set `demo_excluded_games = []` and give each game its scored-run share of
time (`532*60 * concurrency // 110`). Both edits sit in the local-only branch; the scored run
already plays every game at 532 min.

Simulated alone, the fade did nothing for tr87 and cut ka59's priority from 228 to 90, so it was
only tested together with the prior.

## Results (local, all 25 public games, one run each)

| kernel | change | mean | finished | vs unmodified, paired | tr87 | ka59 |
|---|---|---|---|---|---|---|
| `dfranzen-m2-eval25` | none | 40.01 | 5 | — | 58.2, parked | 53.6, parked |
| `dfranzen-m2-eval25-prior` | prior | 44.68 | 8 | +4.7 ± 8.5 se, 11 up / 12 down | 81.6, won | 10.3, parked on L2 |
| `dfranzen-m2-eval25-prior-fade` | prior + fade 0.75 | **48.85** | 8 | +8.8 ± 7.2 se, 10 up / 7 down | 97.5, won | 53.6, played to the end |

The prior fixes tr87, and the fade keeps ka59 in play. The overall gain is about 1.2 se: single
games swing ±80 points between runs regardless of the change (m0r0 14→100, tu93 100→12), so one
run per variant cannot establish it. The pre-committed switch rule (≥ 43.01 and tr87/ka59 not
parked) passed for prior-fade, so it took the 2026-10-02 slot.

## Replicates and the public test

A byte-identical rerun of each arm moved the unmodified notebook to 44.57 and prior-fade to 48.36,
so the paired gain on two-run means is **+6.3 ± 5.0 se (p = 0.22)**: not established. Identical
configs differ by an sd of ~30-37 points per game. Replaying dfranzen's own `priority_value` over
the recorded game curves put the scheduler's worth at about +1.7 at 25 games and +0.9 at the
scored 110-game scale.

| submission | notebook | public |
|---|---|---|
| ref 56761652 (2026-10-02) | dfranzen + prior + fade 0.75 | 20.60 |
| ref 56785456 (2026-10-03) | dfranzen, unmodified (`charyeezy/arc-agi-3-dfranzen-m2-copy`) | **28.39** |

The plain copy lands in the unmodified notebook's public range (dfranzen 27.89 and 31.47, an
identical copy by fantasy0312 27.26). The likeliest cause is the level prior: its per-level
baselines are medians of the same 25 public games the change was tuned on, so it can misprice
late levels on unseen games, and the fade starves early-level games once 110 compete for 10
slots. Lesson: a local gain inside the noise floor, fitted to the public games, is not evidence
for the hidden set — the same pattern as the explorer's local gains.
