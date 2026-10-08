# Internal audit: everything team tgaer has tried on ARC-AGI-3 (as of 2026-09-30)

Compiled 2026-09-30 from local files only. Source keys used in citations:

- `CH` = `/Users/charleneleong/Dropbox/Mac/Documents/gen-ai/orak-hackathon/tgaer/docs/challenges/arc-agi-3.md` (last modified 2026-09-29)
- `PA` = `.../tgaer/docs/challenges/arc-agi-3-prior-art.md` (2026-09-28)
- `TA` = `.../tgaer/docs/challenges/arc-agi-3-technique-audit.md` (2026-09-28)
- `CO` = `.../tgaer/docs/challenges/arc-agi-3-consolidation.md` (2026-09-28)
- `WM` = `.../tgaer/docs/arc-agi3-world-model.md` (long investigation log, 2026-09-15 to 2026-09-26; `WM §<heading>`)
- `KG` = `.../tgaer/docs/arc-agi3-kaggle.md`; `SIA` = `.../tgaer/docs/arc-agi3-sia.md`
- `REP` = `.../tgaer/reports/arc-agi-3-report.tex` (written while v78 was pending, so it predates the v78 result)
- `MEM:<name>` = `/Users/charleneleong/.claude/projects/-Users-charleneleong-Dropbox-Mac-Documents-gen-ai-orak-hackathon-tgaer/memory/<name>.md`
- `LOG:<file>` = `.../tgaer/logs/<file>`
- `GIT:<ref>` = the git history of the tgaer repo (`git log` / `git branch -a`)
- `AGENTS` = `git show feat/arc-agi3-llm-codegen:sia-oss/tasks/arc-agi3/reference/AGENTS.md`

"pp" means percentage points of RHAE (local score, in %). Local RHAE uses the 25-game public demo set. publicScore is the Kaggle leaderboard column.

## Q1. What agent ships today, and what are its local and Kaggle score histories?

### Takeaway
The shipped agent is `ExplorerArcAgi3Agent`: a model-free, frontier-driven directed state graph over chrome-masked, field-box-cropped frame signatures. No LLM runs in the scored kernel. Local RHAE went from about 0.12% (Aug) to 0.3520% (5 seeds, 600 actions) and about 0.42% with `CHURN_WARMUP=30`. Across 12 explorer-era submissions publicScore has never left {0.13, 0.14}. The team's best-ever public score is 0.17, from the v54 27B LLM agent, which clears 0 levels. v80 (v77 plus `CHURN_WARMUP=30`) was submitted 2026-09-30 00:07 UTC and its score is pending in every file read.

### Cited Findings

**Architecture (shipped)**
- Agents in the repo: `ExplorerArcAgi3Agent` (`src/tgaer/agents/arc_agi3_explorer.py`) is marked "ships — model-free, frontier-driven". `ExplorerAgent` (`arc_agi3_kaggle.py`) is the kernel seam. `MyAgent` is the 27B LLM agent that "clears nothing on the roster but holds the best public score". `ArcAgi3LLMAgent` (Gemini / local vLLM) "scores 0". — [CH §Agents]
- Explorer mechanism ablation against the shipped 0.3520% agent (2026-09-25 re-ablation). Removing each mechanism costs: `_inert` (demote primitives that changed nothing) −0.2177pp, losing ar25 and g50t; frontier routing −0.1651pp, losing g50t and tu93; affordance (steer to a salient object) −0.0725pp, losing ar25; chrome mask −0.0204pp, no game lost; goal induction +0.0281pp, no game lost. — [CH §Agents]; [WM §Re-ablated against the new baseline]
- The `field_box` crop is also load-bearing. `USE_FIELD_CROP=False` gives 0.1859%, sd 0.0000, −0.1661pp, and g50t, sp80 and tu93 stop scoring. — [WM §The field_box crop is load-bearing]
- `nav` (key/door navigation) never fires, so switching it off is byte-identical. Its precondition `_observe_door` is broken: it looks for a vanished colour at level-up and `gone` is always empty. — [WM §Ablation: what each mechanism is actually worth (2026-09-25)]. A fix attempt on `fix/arc-agi3-door-inducer` (2026-09-25) made induction fire (ls20 0%→89%, sp80 0%→68%). It was byte-identical at +0.0000pp because it induced the wrong colour and `_route` then failed on 441/441 (ls20) and 392/409 (sp80) steps. Reverted. — [GIT: fix/arc-agi3-door-inducer e3c409f]
- Other explorer components: avatar pinned by controllability, move lattice (`EmpiricalSemantics.move_lattice()`), blocked-cell learning, position memory (PR #19), stuck-policy switch (`_is_stuck()`, least-taken action reorder), salted tie-break, `CLICK_TARGETS_K = 12` salience-ranked click candidates, `PROBE_LIMIT=1`. — [MEM:project_arcagi3_two_agents]; [MEM:project_arcagi3_stuck_scope_falsified]; [MEM:project_arcagi3_search_not_the_bottleneck]; [WM §Promoted: the bootstrap probe]
- The kernel build must use `ARC_KERNEL_AGENT=explorer`. That also drops the vLLM cells, so the shipped kernel runs no model. — [MEM:project_arcagi3_explorer_in_kernel]; [MEM:project_arcagi3_kaggle_stack]
- Eight fitted constants: `PROBE_LIMIT`, `MIN_NOVELTY`, `WALK_WINDOW`, `STUCK_WINDOW`, `FIELD_SWITCH_MARGIN`, `CHURN_FRACTION`, `CHURN_WARMUP`, `_RECENT_CELLS`. — [CH §Constant tuning is exhausted]

**Local RHAE history (25 games unless noted)**
- 2026-08-20: 0.1226 at 600 actions and 0.1755 at 6000. The scored run had been capped at 400 actions (class `MAX_ACTIONS`) until v63 / PR #26. — [MEM:project_arcagi3_metric_is_blind]
- 2026-08-20: 2400 actions gave aggregate 0.1743 (sum 4.357 / 25). — [MEM:project_arcagi3_score_is_actions_per_level]
- Chrome mask on `frame_signature` (PR #29) took 0.1364 → 0.1879 (+38%). lp85 went 1→3 levels and ar25 0→1. — [AGENTS closed table]; [MEM:project_arcagi3_leaderboard_reality]
- 2026-09-15 budget ladder with chrome mask: 400 → 0.1532% (5 levels / 183); 600 → 0.1879% (9); 2400 → 0.1886% (14); 6000 → 0.1886% (15). — [MEM:project_arcagi3_search_not_the_bottleneck]
- 2026-09-21 with seed error bars: 600 actions / 5 seeds → 0.1561% ± 0.0280 (6–8 levels); 6000 / 3 seeds → 0.1704% ± 0.0296. The single-run baselines 0.1879 and 0.1886 were "favourable draws". — [MEM:project_arcagi3_current_benchmark]
- 2026-09-25 `PROBE_LIMIT=1` (PR #35): 0.1561% → 0.3520% ± 0.0280 (+0.1959pp). — [WM §Promoted]; [CH §Current results]
- At 6000 actions / 3 seeds after PROBE_LIMIT: 0.3663% ± 0.0296. The 0.1704% figure is pre-PROBE_LIMIT. — [CH §Current results]; [MEM:project_arcagi3_headroom_decomposition]
- 2026-09-29 `CHURN_WARMUP` 20→30 (PR #52), 5 seeds: base 0.3518%, sd 0.0280 → cand 0.3851%, sd 0.0491. Delta +0.0333pp, levels per seed [8,8,6,6,6] → [9,9,6,6,9], lp85 1.8 → 2.8 levels. — [MEM:project_arcagi3_churn_warmup_candidate]
- 10-seed re-gate of `CHURN_WARMUP=30` (2026-09-29, not written into any doc or memory): baseline 0.3306% sd 0.0441, candidate 0.3656% sd 0.0666, delta +0.0350pp against pooled sd 0.0565pp. lp85 went 1.3 → 2.6 levels and the verdict was "FAIL — INSIDE NOISE". Per seed, candidate ≥ baseline on all 10 (five up, five tied). — [LOG:gate_warmup10seed_20260929T054847Z.log]
- Current sweep baseline 0.4210% (9 levels) and "Local bench 0.4211%, 9 levels" for v80. — [CH §Constant tuning is exhausted]; [MEM:project_arcagi3_churn_warmup_candidate]
- Games that ever score locally: at 2026-09-21 "exactly 5 … ar25, lp85, ls20, sp80, tu93". After PROBE_LIMIT, ls20 is lost and g50t gained. At 6000 actions the headroom table lists 7 scoring games: lp85, tu93, m0r0, sp80, g50t, ar25, s5i5. — [MEM:project_arcagi3_current_benchmark] (older); [CH §The headroom is speed…] (newer)
- An older 8-game suite read 0.4263% where the full 25 read 0.1364%, a 3x overstatement. It also inverted the chrome-mask verdict. — [MEM:project_arcagi3_leaderboard_reality]

**Kaggle publicScore history**
- LLM era: v26–v31 prompt revisions scored 0.08 → 0.17 → 0.02 → 0.11. This was noise: chat history overflowed `n_ctx=4096` by about step 3 and the agent then played randomly. — [MEM:project_arcagi3_kaggle]
- v52 27B LLM, 0 levels in-kernel, 0.00. v54 27B LLM, 0 levels, **0.17** (best ever). A "countdown probe (expect rejection)" submission also scored exactly 0.17. — [MEM:project_arcagi3_metric_is_blind]; [MEM:project_arcagi3_leaderboard_reality]
- Explorer era, full table (2026-09-29):

  | ver | change | local RHAE | in-kernel levels | public |
  |---|---|---|---|---|
  | v59 | explorer baseline | ~0.12% | 2 | 0.13 |
  | v60 | stuck policy switch | — | 5 over 4 | 0.13 |
  | v61 | widened stuck window | — | 6 over 5 | 0.13 |
  | v63 | action cap 400→6000 | — | — | 0.13 |
  | v64 | inert-action detection | 0.1364% | — | 0.14 |
  | v72 | explorer baseline rebuild | 0.1364% | — | 0.13 |
  | v73 | chrome-masked signature | 0.1879% | 14 over 8 | 0.14 |
  | v75 | salted tie-break | 0.1561% | 8 over 6 | 0.13 |
  | v76 | PROBE_LIMIT=1 | 0.3520% | 9 over 7 | 0.13 |
  | v77 | 12000 actions + colour-agnostic roles | 0.3520% | — | 0.14 |
  | v77r | byte-identical repeat (submission 56623806) | 0.3520% | — | 0.14 |
  | v78 | relative novelty | 0.3518% | — | 0.13 |

  — [MEM:project_arcagi3_public_score_is_pinned]
- v80 (ref `56691280`, 2026-09-30 00:07 UTC) is exactly v77 + `CHURN_WARMUP = 30`, with relative novelty off. The prior stated before scoring: "0.14 is the likely reading … Only 0.15+ would be informative." Score pending. — [MEM:project_arcagi3_churn_warmup_candidate]
- Submission count discrepancy: "21 submissions, range 0.13-0.17" (2026-09-23, includes LLM era) [MEM:project_arcagi3_kaggle_stack] versus "twelve submissions" in the explorer-era table [MEM:project_arcagi3_public_score_is_pinned].
- Leaderboard position, older reading (2026-09-15): rank 2099/3049 at 0.17. Median 0.29, p25 0.14, p75 1.54, p90 3.39, top 18.81 (Tufa Labs). The board keeps the best submission. — [MEM:project_arcagi3_leaderboard_reality]
- Leaderboard position, newer reading (2026-09-24 onward): "0.13 against a 0.30 median and a 19.40 leader over 3284 teams — bottom quartile". — [CH §Current results]; [MEM:project_arcagi3_value_model_is_the_gap]

### Inferences
- The displayed leaderboard score is still 0.17 (v54), because the board keeps the best submission. No explorer submission has exceeded it.
- The 10-seed `CHURN_WARMUP` gate failed the 2-sd rule but meets the team's pre-committed rule: no game regressed, no seed regressed, delta ≥ 0.0300pp. That fits its being merged (PR #52) and submitted as v80.

### Gaps
- The v80 publicScore is not recorded in any local file read.
- No file records a per-game publicScore breakdown. publicScore is only available as an aggregate.

## Q2. Every lever tried: what it was, measured effect, verdict, and why

### Takeaway
More than 30 levers were gated. Four explorer changes were shipped for a measurable local gain: the chrome mask (+38%), PROBE_LIMIT=1 (+0.1959pp), the 12000-action budget (+0.0001pp), and CHURN_WARMUP=30 (+0.033–0.035pp). One (colour-agnostic roles) shipped bit-identical and one (relative novelty) was submitted and reverted. Most other levers were rejected. The LLM-as-policy line and the LLM-codegen line (seven attempts) produced zero promotions. Value-model, effect-prediction and static-ranker programmes were all closed or found underpowered.

### Cited Findings

**A. Action budget / horizon**
- 150→600 actions recovered ls20 (1→2 levels). 4x budget unlocked no new game. — [MEM:project_arcagi3_explorer_in_kernel]
- 600→2400→6000: 3.33 then 0.56 levels per 1k actions. Headline RHAE 0.188600% → 0.188600%, identical to 6 decimals. — [MEM:project_arcagi3_search_not_the_bottleneck]
- 600→6000, 5 vs 3 seeds: +0.0142pp against a 2-sd bar of 0.0625, inside noise. — [MEM:project_arcagi3_current_benchmark]
- 6000→12000: +0.0001pp, one extra level (ar25). **Shipped** in PR #38 / v77. Justified because the scorer is monotone in budget. — [CH §Levers measured and closed]
- 12000 actions is about 80% of the kernel time budget, "not a margin". — [GIT: 6a8cf66, PR #43]

**B. State representation / signature**
- Colour-agnostic `field_box` (modal colour instead of hardcoded GREEN): 2→3 clears (sc25). Shipped (Aug). — [MEM:project_arcagi3_search_not_the_bottleneck]; [GIT: e86542c]
- Cross-frame `FieldTracker`: fixed tr87 (1 signature → 469) but lost lp85 and sc25, net 3→1. Reverted. — [MEM:project_arcagi3_search_not_the_bottleneck]
- Whole-grid hash signature and object-level signature: both lost sc25; the object signature was 2.5x slower. Rejected. — [same]
- Motion veto to whole grid for tr87: 1 signature → 594 states but no clear, and runtime 846s vs ~30s. Reverted. — [same]
- Chrome mask on `frame_signature` (`_settled`, PR #29): **promoted**, 0.1364 → 0.1879. — [AGENTS]; [MEM:project_arcagi3_leaderboard_reality]
- Coarser object key and shape+centroid key (Blind Squirrel style): strictly worse. lp85 431→733 states and rankable 291→146; tu93 rankable 147→46. Rejected 2026-09-24. — [WM §State fragmentation]
- Drop the `field_box` crop (`USE_FIELD_CROP=False`): −0.1661pp, sd 0, three games lost. Rejected 2026-09-26. — [WM §The field_box crop is load-bearing]
- Whole-board `_field` for the five blind games (vc33, tr87, ft09, dc22, tn36): no level count changed. — [WM §field_box blinds the state signature]
- Per-cell frame-signature denoising, all variants: best case −0.11, closed. — [AGENTS closed table]
- `(colour, size)` class compression: not built, because candidates collapse only 1.4x at the median. — [CH §Levers measured and closed]

**C. Exploration / routing / search policy**
- Stall rotation in `_choose`: repeats 3763→2779 actions and ft09 one-action share 98%→8%, but levels stayed 2→2. Discarded, though later noted as a ~1.8x score gain under the quadratic metric. — [MEM:project_arcagi3_search_not_the_bottleneck]; [MEM:project_arcagi3_score_is_actions_per_level]
- Stuck-policy switch: v60 went from 2 to 5 in-kernel levels. **Shipped.** — [MEM:project_arcagi3_metric_is_blind]
- Widened stuck window: v61 reached 6 levels over 5 games. Shipped. — [MEM:project_arcagi3_public_score_is_pinned]
- Cap consecutive walks at 12: gained tu93 but lost ls20 and sc25, net 3→2. Reverted. — [MEM:project_arcagi3_search_not_the_bottleneck]
- Reorder `_nav_affordance` ahead of `_explore_due`: 0.4025, cost sp80. Closed. — [AGENTS]
- Run/momentum prior (`RUN_LIMIT=8`, copying the oracle's 60–80% repeat rate): 0.1879 → 0.1111, six games regressed. Rejected 2026-09-18. — [MEM:project_arcagi3_search_not_the_bottleneck]; [WM §Four changes measured]
- Clicks scan the whole grid: gate PASS 0.1879 → 0.1896 (vc33 0→1) but sweep SPIKE (2/5 values of k beat baseline; at k=16/24 sc25 stops scoring). Out-of-field clicks as a demoted tail: 0.1879, no change. Neither promoted. — [WM §Four changes measured]
- Raising the `click_targets()` cap 12→25: no change on bp35. — [AGENTS]
- Click-based routing fallback: regressed to 130.53 (broke sc25). — [AGENTS]
- `PROBE_LIMIT` 4→1 (bootstrap probe cap): +0.1959pp, g50t 0/5→5/5 and ar25 about 570→39 actions, but ls20 5/5→0/5 and sp80 got worse. **Shipped as a deliberate override of the no-regression rule** (v76). Without g50t the gain is +0.0530pp, inside the 0.0560pp bar. — [WM §Promoted]
- Demand-driven probe (probe only when `_is_stuck()`): 0.3508%, strictly worse (tu93 also lost). Reverted. — [WM §Why ls20 needs the full bootstrap]
- Defer irreversible/regressive edges (deficiency theory): 0.1861%, sd 0, −0.1659pp (8.4 sd), g50t/sp80/tu93 lost. Rejected 2026-09-26 (PR #40). — [MEM:project_arcagi3_irreversible_edges_falsified]
- Re-validate the avatar latch: tu93 2→1 levels. Rejected. — [CH §Levers measured and closed]
- Reject an induced avatar outside the field box: 0.1879 → 0.1852, sp80 got worse. Rejected. — [WM §Four changes measured]
- Per-level scoping of the stuck switch: −0.0012pp, tu93 5/5→0/5 seeds. Rejected 2026-09-28. — [MEM:project_arcagi3_stuck_scope_falsified]
- Fresh novelty window on respawn: identical seed-for-seed to the above (0.3813, 0.3813, 0.3313, 0.3301, 0.3301), so the respawn line was the whole effect. Rejected. — [MEM:project_arcagi3_post_death_switch_is_load_bearing]
- Plan verification (`USE_PLAN_VERIFY`, check a routed step lands where predicted): +0.0000pp, bit-identical, because routes never mislead. — [LOG:gate_planverify_20260928T060052Z.log]; [REP ablation ledger]; [MEM:project_arcagi3_stranded_vs_flooded]
- Learned inverse edges (route over hypothesised reverse edges for tu93): −0.0014pp. Only tu93 moved, and it died (5/5→0/5). Rejected. — [MEM:project_arcagi3_tu93_is_a_knife_edge]
- Door-inducer fix: byte-identical, reverted (see Q1). — [GIT: fix/arc-agi3-door-inducer]
- Position-keyed exploration and position-memory cycle break (Phase 7/8, 2026-06/07): the position memory shipped (PR #19). Phase 8 exists only on remote `a100-fetch/position-keyed` (2026-07-05). — [GIT: a100-fetch/position-keyed]; [MEM:project_arcagi3_two_agents]

**D. Constant tuning (2026-09-29 sweeps, baseline 0.4210%)**
- `MIN_NOVELTY` 0.05–0.30: 0.4206–0.4215, no cliff. `WALK_WINDOW` 12–48: 0.4207–0.4215. `STUCK_WINDOW` 48–192: plateau, with a cliff at 48 (0.4198%, 7 levels). `FIELD_SWITCH_MARGIN` 1.0–2.0: 0.4210 flat at 1.0/1.25/1.5, cliff at 2.0 (0.3314%, 6 levels). 0 of 17 points gain. — [CH §Constant tuning is exhausted]
- `CHURN_FRACTION` plateau over 0.4–0.5. `PROBE_LIMIT` 0 and 1 are identical (0.3825%, 8 levels). `_RECENT_CELLS=0` gives 0.3155% vs 0.3520%, so it is "genuinely not inert". — [MEM:project_arcagi3_churn_warmup_candidate]; [CH]
- `CHURN_WARMUP` 20→30: **shipped** (PR #52, v80). Details in Q1. — [MEM:project_arcagi3_churn_warmup_candidate]
- Relative novelty (`rate < NOVELTY_DROP * peak`, `NOVELTY_DROP=0.5`, replacing `MIN_NOVELTY=0.15`): locally −0.0002pp. Submitted as v78 on the off-roster argument and scored 0.13 vs v77's replicated 0.14. **Rejected**; reverted via `USE_RELATIVE_NOVELTY=False` (PR #54). — [MEM:project_arcagi3_relative_novelty_candidate]
- In-episode dead-colour demotion (`DEAD_COLOUR_TRIES` 1–12): 0.1393–0.1403% (6 levels) when it fires, baseline when it does not. A perfect filter's ceiling is +0.0038pp. Rejected 2026-09-23. — [WM §In-episode colour demotion]

**E. Inertness / effect prediction / cross-level carry**
- Inert-action detection (`_inert`): shipped in v64 (0.1364 local, public 0.14). Now the most load-bearing mechanism (−0.2177pp if removed). — [MEM:project_arcagi3_public_score_is_pinned]; [CH §Agents]
- `_inert` on the chrome-masked view: 0.1879 → 0.1558, lp85 −1 level, no effect on sp80. Rejected 2026-09-17. — [MEM:project_arcagi3_search_not_the_bottleneck]
- Pre-win goal signals (effect-magnitude prior, click-novelty prior): 3 clears unchanged, reverted (2026-08-15). — [same]
- Effect-ranked clicks (2026-09-18). Over all primitives: 0.1879 → **0.3911** (sc25 0.026%→5.215%) but sp80 1→0. Over clicks only: 0.1917, sc25 1→0. Demote dead clicks only: 0.1917, same. Recoverable-demotion cooldown N=10–250: 0.1917 at every value, sc25 still lost. All rejected. — [same]
- Replay effect model (`(action, colour, size)` predicts changed-or-not at 60–100%, across states on 25/25 games), wired into `_inert`: changed `_live` ordering in 492/2756 calls, +0.0000pp bit-identical. A perfect oracle's ceiling is +0.0065pp vs the 0.0300pp floor. **Family closed** 2026-09-28. — [CH §Effect prediction is closed]; [MEM:project_arcagi3_effect_prediction_closed]
- Carry `_inert` across a level. 2026-09-24: −0.0204pp, sd 0. 2026-09-28: raw counts −0.0204pp sd 0; capped −0.0204pp bit-identical; controls-only `("act", id)` +0.0000pp with 0 games moved. **Family closed.** — [WM §Carrying _inert]; [MEM:project_arcagi3_inert_carry_family_closed]
- Colour-agnostic role inference (drop colour literals 3/4 in `_observe_key` / `_observe_door`, PR #41): bit-identical locally. **Shipped** in v77, which scored 0.14 (+0.01 over v76). The "local no-op = OOD gain" heuristic drawn from this was falsified by v78. — [WM §Role inference]; [MEM:project_arcagi3_scale_freedom_heuristic_falsified]

**F. Oracle-derived priors / rankers / value models**
- Offline oracle (fork via `deepcopy`, uninformed BFS): clears a level in 11/25 games. Projected RHAE ceiling 2.75% vs 0.1886%. — [WM §The local simulator as an offline oracle]
- 172 oracle-labelled decisions (281 at 4x budget, all extra from tu93). Recall: proposable 97%, @4 87%, **@1 18%**. — [MEM:project_arcagi3_coverage_not_selection]; [WM §More oracle depth]
- Static ranker (`oracle_rank.py`, leave-one-game-out): recall@1 19%→22%, all gains on lp85. Clicks 0%→25%, simple actions 21%→21%. **Falsified.** — [WM §A learned static ranker does not transfer]
- Within-game ranker, forward-chained: tu93 26% baseline / 27% model / 25% chance (n=138); pooled 22/23/21%. Indistinguishable from random. lp85 anti-transfers (within-group correlation −0.433). — [WM §More oracle depth does not unlock it]
- Value model, GBR vs conv encoder (PR #55 split, PR #56 conv, 2026-09-30, 125 cached 6000-step trajectories). Only lp85 and tu93 clear ≥2 levels. The 10 episodes are 5 distinct trajectories. GBR 47.8% vs chance 45.9%, 4/1, p=0.19. Conv 48.1%, 4/1, p=0.19. Conv vs GBR 2/3, p=0.81. The earlier reading "39% vs 29.5%, p~0.01" came from a tie-blind chance floor. Verdict: the **offline gate is underpowered**. — [MEM:project_arcagi3_value_model_offline_underpowered]
- The conv encoder scores 100% vs GBR 52% (chance) on a toy relative-position task. A strided first conv plateaued at 70%. — [GIT: 83194bd commit message]

**G. LLM approaches**
- June 2026: every frozen baseline scored 0/7 on ls20: random, Gemini, Qwen3.6-35B text, Qwen3-VL-4B, Qwen3-VL-30B-A3B. An RL design (vendored `verifiers` env, shaped reward, GSPO) was scoped as a draft. No results were found. — [docs/specs/2026-06-14-arc-agi3-rl-design.md]
- VL Scientist (Qwen3-VL labels roles per level, PR #13/#17) regressed ls20 to 0 and was superseded by empirical semantics and the explorer. — [docs/specs/2026-06-18-…scientist-design.md; 2026-06-23-arc-agi3-explorer-design.md]
- Kaggle 27B LLM-as-policy (`MyAgent`, Qwen3.6-27B FP8, tool calling). 0 levels on all 25 at 200 actions, with forward model 88.6% correct (2853/3220). In-kernel: 0 levels at 1051 actions/min vs the explorer's 2 levels at 3151. — [MEM:project_arcagi3_two_agents]; [MEM:project_arcagi3_explorer_in_kernel]
- Exploit cap: blind repeats fell 53%→33%, levels 0. Mechanic notes (cross-turn theory): repeats 33%→17%, levels 0 vs 0 in a 3v3 A/B. Falsified 2026-08-11/12. — [MEM:project_arcagi3_model_turns_not_bottleneck]
- REPL harness (`feat/arc-agi3-repl-harness`, 2026-08-22): `ReplController` lets the model call `action([...])` from inside its python tool. 1077 insertions incl. 341 lines of tests. Dormant and unmerged; no result recorded. — [GIT: 1d17eff]
- LLM-codegen line (`feat/arc-agi3-llm-codegen`, SIA supervisor, `policy(frame)` codegen): v2 0.3972 FAIL; v3 exactly baseline; v4 constant policies; v5 REPL loop (4 rounds) with floor intact; v6 0.4230 FAIL (sc25); v7 Qwen3.6-35B-A3B 0.4017 FAIL (sp80). "This closes the line: 7 attempts, 2 models, 0 promotions." — [AGENTS]
- SIA `improve` loop: "14 generations in, only 1 has ever attempted a real change to `explorer.py` (it regressed)". — [AGENTS]; [SIA]
- World model M0 (`deepcopy` perfect simulator + heuristic): lp85 L2 364→8 actions (45x), RHAE 0.1886%→0.4422% projected. Uninformed BFS failed; L3 plateaued in greedy search. — [WM §Milestones]
- M1 (Qwen3.8-27B writes `simulate` / `is_goal` / `distance` for lp85 L1): `is_goal` 9/9 wins with 8/340 false positives; `distance` clears L1 in 5 real actions; `simulate` 0/340 exact, 0.1% of moved objects vs 46% for echo-the-input. — [WM §M1 result]; [WM §How badly does the generated simulate fail]
- M2 phase 1, one-shot model-written `distance` per game over 11 oracle games: **failed its kill criterion**. Only 2/11 games (lp85, ft09) had a usable gradient; tu93 47%, m0r0 43%, s5i5 0%. — [MEM:project_arcagi3_llm_as_programmer]; [LOG:m2_distance_20260924T125125Z.log]
- `wip/kaggle-agent-python-tool` (2026-09-15): richer python tool (`grid_diff`, `object_positions`, `action_history`, `score_history`, `hud_cells`, `step`; `MAX_OUTPUT_TOKENS` 128→256). "Committed as-is, unreviewed and untested." — [GIT: cf91b5e]
- Duck harness on the shared A100 pod (2026-09-30): installed; the first run was stopped. With 16 games concurrent it got 40–130 total gen tok/s, the pod has no native FP8, the card was shared at 99% util, and it risked OOM for another tenant. Not a fair test. — [MEM:reference_tgaer_on_pia100]

### Inferences
- The levers that moved local RHAE are almost all perception and bootstrap fixes (field colour, chrome mask, probe cap) plus `_inert`. Every ordering, routing, prior or carry mechanism tried after 2026-09-18 was rejected, bit-identical, or inside noise.
- An iterative LLM-as-programmer REPL, in which the model acts from its own code, has code on a dormant branch but no recorded measurement.

### Gaps
- No recorded result for `feat/arc-agi3-repl-harness` or `wip/kaggle-agent-python-tool`.
- The June RL (GSPO) design has no results file among those read.
- `gate_drop_minnovelty_20260929T073123Z.log` is empty (0 bytes), so a MIN_NOVELTY-deletion gate result is not recorded.

## Q3. Key structural findings

### Takeaway
Score is action efficiency on levels already cleared (squared), not level count. The local demo set does not predict publicScore in either direction, and publicScore is pinned at 0.13–0.14. Coverage is fine but selection is at chance on the deepest game. tu93 and lp85 hold 64% of reachable local headroom, need opposite fixes, and tu93 is a knife edge. The offline value-model gate is underpowered. The one-shot LLM distance heuristic failed its kill criterion. The Duck clears ≥1 level on 23/25 public games vs this team's 7/25.

### Cited Findings
- **Metric.** Per level: `min((baseline/actions)^2 * 100, 115)`, not cumulative. Per game: weighted by level index over all levels. Best-of-N replay is unreachable in competition mode (`api.py:424`). — [MEM:project_arcagi3_score_is_actions_per_level]; [CH §Scoring]
- **Headroom (6000 actions, 3 seeds, 2026-09-28).** RHAE 0.3663% vs speed-ceiling 2.0571%, so 17.8% captured. 18/25 games score zero with cap zero. Per game: lp85 3/8 levels, 18.7% capture, 37.8% of ceiling; tu93 3/9, 0.3% capture, 25.9%; m0r0 and sp80 9.3% each; g50t at 100% capture; ar25 67.3%; s5i5 0.0%. — [MEM:project_arcagi3_headroom_decomposition]; [CH]
  - Older version (2026-08-20): 12 levels would score 2.349 at baseline speed vs 0.174 achieved (7.4%). lp85 was 64% of score. — [MEM:project_arcagi3_score_is_actions_per_level]
  - Older version (2026-09-21): perfect efficiency on cleared levels is worth +2.203pp vs +0.73pp for winning four uncleared games. — [WM §Every scoring game is efficiency-limited]
- **Where actions go.** On levels that never clear, 80–98% of actions revisit seen boards; tu93 L3 burns 4538 actions at 98% revisit. `rotate` is 0% everywhere, so there is no cycle breaker to build. — [CH]; [WM §Every scoring game is efficiency-limited]
- **Coverage vs selection.** Winning move proposable 97%, first 18%. `_choose` returns `untested[0]` (novelty, not value) and makes 68% of decisions. — [MEM:project_arcagi3_coverage_not_selection]
- **Stranded vs flooded (2026-09-28).** tu93 L2: 66% of decisions have no route, states frozen at 159, 6 untested pairs. lp85 L4: 0% no-route, states grow to 1647 with 15950 untested pairs. — [MEM:project_arcagi3_stranded_vs_flooded]
- **tu93 is a knife edge.** Three routing mechanisms each took it 5/5→0/5 seeds. Its wins rely on the post-death stuck switch firing on stale evidence. — [MEM:project_arcagi3_tu93_is_a_knife_edge]; [MEM:project_arcagi3_post_death_switch_is_load_bearing]
- **No cross-level learning.** tu93 per-level actions run 383→194→886 and lp85 10→364→100→777, with no downward trend. — [CO]
- **Reachability.** Five games are exhausted at 1–62 states (vc33, tr87, ft09, dc22, tn36) because the crop blinds them (tr87: 523 boards → 1 signature). Fourteen have thousands of untested pairs (cd82 31378; the oracle clears cd82 in 5). — [WM §Reachability]; [WM §field_box blinds]
- **Local bench does not predict public.** v75→v76 local 2.25x, public 0.13→0.13 (the predicted ~0.29 is about 6x the 0.027 noise bar). Plateau shape is insufficient evidence of transfer. — [MEM:project_arcagi3_local_bench_does_not_predict]
  - Older and contradicted (2026-09-15): "The local 25-game set predicts Kaggle roughly 1:1". — [MEM:project_arcagi3_leaderboard_reality]. Superseded by the 09-25 test above.
- **publicScore is OOD.** It scores about 55 semi-private games; privateScore covers the other 55 and is withheld; the 25-game roster is never scored (corrected 2026-09-27). — [MEM:project_arcagi3_private_score_withheld]
  - Older and contradicted (2026-09-18): "The scored kernel plays the same 25 games … the '~110 concurrent' in the notebook is thread slots". — [WM §The local simulator as an offline oracle]. Superseded by the 09-27 correction.
- **publicScore reproducibility and pinning.** A byte-identical v77 rebuild also scored 0.14 (n=1 pair). The inference "a local no-op can be an OOD gain" was falsified by v78. Twelve submissions never left {0.13, 0.14} while local moved 3x and in-kernel levels 7x. — [MEM:project_arcagi3_publicscore_is_reproducible]; [MEM:project_arcagi3_public_score_is_pinned]; [MEM:project_arcagi3_scale_freedom_heuristic_falsified]
- **Noise floors.** Local seed sd is about 0.030pp, so the 2-sd bar is about 0.060pp. lp85's level count is "the only source of seed variance in the whole 25-game suite". `sd→0` is a failure signature. — [CH §Bench tools]; [MEM:project_arcagi3_carry_inert_unaged_falsified]; [WM §field_box crop]
- **Value-model offline gate underpowered.** See Q2-F. — [MEM:project_arcagi3_value_model_offline_underpowered]
- **One-shot LLM distance: 2/11**, kill criterion failed. The blocker is bootstrapping without a winning board. — [MEM:project_arcagi3_llm_as_programmer]
- **Duck comparison (2026-09-30)**, from the Duck repo's `example-run/` (500 runs = 25 games × 20 passes) vs this team's cached trajectories (25 × 5 seeds, 6000 steps):
  - Games with ≥1 level: **23/25 vs 7/25**. Mean levels per run 0.62 vs 0.48.
  - L1 actions where both clear: m0r0 67 vs 1708, s5i5 56 vs 1401, sp80 55 vs 458, tu93 69 vs 383.
  - The Duck clears 16 games this team never does. This team is deeper only on lp85 (4 vs 1) and tu93 (3 vs 2).
  - The Duck bootstraps by programmed probes in a Python REPL, then writes its own BFS. It reads segmented objects and the raw grid is hidden.
  - — [MEM:project_arcagi3_duck_bootstraps_by_probing]
- **Effect is predictable but constant in 14/25 games.** 1096/1112 (99%) of `(colour, size)` buckets are unanimous. — [WM §Click effect is predictable]
- **No fixed action effect.** Across lp85/ls20/sp80/tu93/sc25 each action does 24–75 distinct things, with new ones still appearing after 400 sightings. — [WM §Four changes measured]
- **Why five games never clear (2026-09-18).** vc33's winning button lies outside the field box. ft09's winning click was later found proposable (corrected 2026-09-21). cd82, sk48 and m0r0 fail on the continuation. — [WM §Why five games never clear]; [WM §The oracle's labels]

### Inferences
- The Duck comparison also measures level coverage, which the team's 09-28 headroom analysis set aside as "worth nothing until level 1 clears". The Duck's 23/25 suggests level-1 coverage is attainable by a different architecture.

### Gaps
- The Duck's per-game RHAE on the public 25 is not reproduced locally. The pod run was stopped.
- The size of the publicScore noise bar is based on one repeat pair.

## Q4. Infrastructure and constraints discovered

### Takeaway
The scored run is online against a gateway with no game object to fork. Internet is disabled. The GPU is an RTX Pro 6000; this was corrected from an earlier P100 belief. The kernel serves vLLM 0.19 while the dev pod runs 0.26. The runtime limit is unconfirmed (7.5h assumed, <12h per secondary sources). There is one submission per day. Several build traps can burn a slot.

### Cited Findings
- **Scored run is online.** Cell 5 sets `OPERATION_MODE=online`, `ARC_BASE_URL=http://gateway:8001/`, and an empty `ENVIRONMENTS_DIR`, so `deepcopy` planning is impossible there. A shadow instance from the bundled files replays byte-identical (8/8 games, 300 actions), but three load-bearing assumptions are unverified, and adopting it is flagged as "the submitter's call". — [MEM:project_arcagi3_kernel_has_no_game_object]; [WM §Does deepcopy work in the kernel]
- **Internet disabled**, so no frontier API. Accelerators: CPU, T4x2, P100, and RTX 6000 "reserved for ARC-AGI-3 notebooks". — [MEM:feedback_arcagi3_no_frontier_api]; [TA]
- **Hardware.** `kernel-metadata.json` shows `machine_shape: NvidiaRtxPro6000` and `enable_internet: false`, with 27B FP8 weights (`vrfai-qwen3-6-27b-fp8-hf-snapshot`) and a vLLM wheelhouse attached. The "P100 16GB" belief was stale. A torch probe found torch 2.10.0+cu128, 94.4GiB free, and ResNet-18 at about 15 steps/s. — [MEM:project_arcagi3_kaggle]; [MEM:project_arcagi3_relative_novelty_candidate]; [MEM:project_arcagi3_value_model_is_the_gap]
  - **Disagreement:** [TA] and [MEM:feedback_arcagi3_no_frontier_api] (2026-09-28) say the kernel loads `qwen3-14b.Q4_K_M.gguf` via llama-cpp (`arc_agi3_kaggle.py:1159`). [MEM:project_arcagi3_kaggle_stack] (2026-08-14) and [MEM:project_arcagi3_llm_as_programmer] say the LLM agent ran Qwen3.6-27B FP8 on vLLM 0.19.0. Both may be true of different code paths; the shipped explorer loads no model either way.
- **vLLM versions.** The kernel installs vLLM 0.19.0 (wheelhouse `driessmit1/arc3-vllm-h100-wheelhouse-v3`); the A100 dev pod runs 0.26.0; the Duck pins 0.17.2rc1.dev150. — [MEM:project_arcagi3_kaggle_stack]; [MEM:reference_tgaer_on_pia100]
- **Runtime.** Assumed 7.5h (`KERNEL_BUDGET_H`, `_RUN_DEADLINE`); secondary sources say <12h; not stated in official docs. The wall-clock gate fails a candidate above 1.25x baseline or projecting past 7.5h. — [TA]; [MEM:feedback_arcagi3_gate_discipline]
- **Throughput.** Explorer 3151 actions/min, 27B agent 1051. The vLLM install plus 36GB weight load costs 10–15 min. Duck-style codegen for about 110 games at 30s each would take about 1.2h. — [MEM:project_arcagi3_explorer_in_kernel]; [MEM:project_arcagi3_kaggle_stack]; [WM §Scope: LLM-as-programmer]
- **Submissions.** One per day, resetting about 00:00 UTC. CLI needs `-k`, `-v` and `-f`. "0 submissions remaining today" means the submission succeeded. — [MEM:project_arcagi3_kaggle_stack]
- **Build traps:**
  - Env vars read inside the kernel are dead.
  - `ARC_KERNEL_AGENT` defaults to `myagent`.
  - The vendored `my_agent.py` is a stale v33 LLM agent, so `make submit` would ship the wrong agent.
  - The preflight is a single temperature-0.4 sample.
  - Mock output sorted alphabetically hid the scoring games.
  - `_last_action_id` was not maintained.
  - — [MEM:project_arcagi3_explorer_in_kernel]; [MEM:project_arcagi3_kaggle_stack]; [MEM:project_arcagi3_relative_novelty_candidate]
- **Per-level step caps in games.** cn04 L1 allows 75 actions. tu93 StepCounter is 50/50/35/20… and lp85 is 13/60/80/150/80…. — [MEM:project_arcagi3_model_turns_not_bottleneck]; [MEM:project_arcagi3_score_is_actions_per_level]
- **Shared A100 pod (`pi-a100-80gb`).**
  - Use `~/vllm_venv` (base `~/.local` pins transformers 4.40.1) and put ninja on PATH.
  - Only Qwen3.8-27B fits, at 0.88 utilisation and 70.7GB.
  - Disk is about 92% full; use port 8011.
  - The pod has no native FP8.
  - Do not mix pod and Mac numbers, because a GBR split flipped 40% vs 43%.
  - The full test suite (622) passes there.
  - — [MEM:project_pia100_vllm_launch]; [MEM:reference_tgaer_on_pia100]
- **Bench tools** (`sia-oss/bench/`): `measure.py`, `ab.py` (N seeds, 2 pooled sd, per-game frequency and depth, throughput verdict), `sweep.py` (one seed per value), `ablate.py`, `reachability.py`, `oracle_labels.py` / `oracle_recall.py`, `m0_suite.py`, `value_model.py`, `shadow_sync.py`, `action_budget.py`. — [CH §Bench tools]; [MEM:project_arcagi3_churn_warmup_candidate]
- **Gate discipline** (set 2026-09-28): publicScore is the objective. A slot is spent only with no local regression, no wall-clock regression, and an off-roster argument. — [MEM:feedback_arcagi3_gate_discipline]

### Inferences
- The kernel hardware (RTX Pro 6000, about 94GiB, torch present, 27B FP8 attached) is capable of hosting the Duck-style design. Its constraints are runtime and serving-stack divergence, not VRAM.

### Gaps
- The official runtime limit is unverified in every file.
- Whether the gateway serves the same game builds as the bundled `environment_files` is untestable from outside the kernel.

## Q5. Open leads and explicitly deferred work

### Takeaway
The team's own documents converge on architectural change: an LLM-as-programmer REPL in the Duck style, plus verification. Constant tuning is declared exhausted. Several smaller mechanism leads remain untested.

### Cited Findings
- **Resume the REPL branches** toward the Duck design (`feat/arc-agi3-repl-harness` 2026-08-22, `wip/kaggle-agent-python-tool` 2026-09-15). The four named gaps are:
  1. Observations as inspectable Python variables.
  2. Image plus text grids.
  3. Pre-built helpers.
  4. Oldest-message eviction.

  Read `ARC3-Inference/README.md` first. The Duck depends on `tufa-arc-agi-framework` (TAAF), and its licence is declared MIT in `pyproject.toml`. — [PA]; [MEM:reference_arcagi3_prior_art]; [MEM:project_arcagi3_duck_bootstraps_by_probing]
- **Ranked next steps (2026-09-28):**
  1. Verify accelerator/VRAM and runtime.
  2. Add verification to the explorer (since found bit-identical, see Q2-C).
  3. Resume the REPL branches.
  4. Then simplification, then an executable world model, per the EWM ablation order.

  — [TA]
- **Iterative REPL.** Whether iteration clears the bootstrap wall is "the open question". — [MEM:project_arcagi3_llm_as_programmer]
- **Value model.** Offline tuning is abandoned. Paths with statistical power: oracle labels past level 0 on more games, or an online A/B on level-2 clears in games that clear level 1. `report()` must dedup identical trajectories. — [MEM:project_arcagi3_value_model_offline_underpowered]
- **Deliberate post-death switch.** A bounded countdown forcing `_is_stuck()` true for N steps after each death; N is a sweep dimension. Untested. — [MEM:project_arcagi3_post_death_switch_is_load_bearing]
- **Remaining tier-B carry candidates**, untested: goal-colour and role induction. Membership rule: the key must be board-free and expensive to re-earn. — [MEM:project_arcagi3_inert_carry_family_closed]
- **Consolidation target metric.** The slope of actions-per-level vs level index. Explore early, exploit late. — [CO]
- **lp85** is the recommended target because it is an ordering problem. **tu93** is closed as a target. — [MEM:project_arcagi3_tu93_is_a_knife_edge]
- **Adaptive field box** (widen to cells observed to change) for the five blind games. It is described as parked because it re-keys the graph. — [WM §field_box blinds]; [WM §The field_box crop is load-bearing]
- **Effect-scored click candidates**, replacing the fixed K=12 salience budget ("Phase-2 filtering"), with recoverable demotion and a non-per-game resolution of click-vs-act order. — [WM §Four changes measured]; [MEM:project_arcagi3_search_not_the_bottleneck]
- **Irreversible edges.** Retry only after instrumenting edges with no known return path over a 6000-action run. — [MEM:project_arcagi3_irreversible_edges_falsified]
- **Shadow simulator** from bundled files (for free search). Measured and available, "not adopted". — [MEM:project_arcagi3_kernel_has_no_game_object]
- **Model sizing and runtime.** Possibly under-using the GPU; the runtime may leave about 4.5h unused if the limit is 12h. Both must be verified before any change. — [TA]
- **Duck on the pod.** Rerun only when the GPU is idle, at about 4-way concurrency with a longer timeout, and treat the result as a lower bound. — [MEM:reference_tgaer_on_pia100]
- **Earlier M2 phases 2–5** (greedy-on-distance with a fork, hill-climbing without a fork, gate, kernel integration) were never reached because phase 1 failed. — [WM §Scope: LLM-as-programmer]; [MEM:project_arcagi3_llm_as_programmer]

### Inferences
- Every open lead in the files that targets publicScore movement is architectural. The mechanism-level leads (post-death countdown, tier-B goal carry) target local RHAE, which the team has shown does not transfer.

### Gaps
- No file specifies a design or kill criterion for an iterative-REPL experiment beyond the four Duck gaps.
- The v80 outcome, which would settle whether `CHURN_WARMUP=30` moved publicScore, is not yet recorded.
