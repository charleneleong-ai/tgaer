# LLM agents for interactive grid games / ARC-AGI-3: delta for 2026-10-06 to 2026-10-12 (checked 2026-10-12)

This note covers only what is new compared with `research_notes/ARC AGI 3 SOTA October update/models_and_agent_research.md` and `harness_and_serving_techniques.md`. Those notes already cover Memento 3 (2610.11794), VISTA (2610.02200), PRO-LONG (2607.20064), Prime Agent (2608.23552), ASCENT (2610.05303), Agentic-TTT (2610.12002), Twin, Tycho, AERA, NOOA, DreamTeam and OPINE-World, so none of those is repeated here. All numbers below are author-reported; nothing was reproduced.

Searches run: Hugging Face papers (keyword, sorted by date, plus the daily list up to 2026-10-09), arXiv via web search, `gh search repos` for ARC-AGI-3 repos created or pushed since 2026-10-05, and the ARC Prize blog and X account.

## Q1. New papers and repos on ARC-AGI-3 and interactive puzzle agents with measured gains

### Takeaway
One new ARC-AGI-3 paper appeared in the window. *Tracing Thoughts* (2610.11450, 8 Oct) runs the PRO-LONG harness and includes the first published ARC-AGI-3 numbers for a local **Qwen3.8-27B-FP8** backbone: **6.8-8.4% RHAE, 34-42 levels cleared**, against 96.9-99.95% RHAE and 179-183 levels for Opus 5 in the same harness. Its finding is diagnostic: carried-over hard-coded constants cause the costliest failures. It tests no intervention. The most useful new code is Hivemind's public trace-replay analysis of dfranzen's Milestone-2 harness on Qwen3.8-Flash-Next, the same stack we run (see Q2/Q3). One new non-ARC paper reports a large measured gain from falsifiable belief revision, EvoBelief (2610.12789). It uses frontier backbones only.

### Cited Findings

**Tracing Thoughts of a Coding Agent Playing ARC-AGI-3: Lessons for Continual Learning** ([arXiv 2610.11450](https://arxiv.org/abs/2610.11450); Wu, Passenger, Song; submitted 8 Oct 2026; NeurIPS 2026 CL4FMAgents workshop)
- The harness is "taken unmodified from PRO-LONG": an analyzer (Strands ReAct agent, about 10 tool calls per turn) reads `logs.txt` and writes `actions.json`, and a runner executes it. There are 6 sandboxed tools and one workspace per game, and one prompt for all 25 public games that names the actions but not their effects — [arXiv html](https://arxiv.org/html/2610.11450).
- Seven runs. They are not controlled repetitions: model, prompt, action cap and mode all vary — [arXiv html](https://arxiv.org/html/2610.11450):

  | Run | Backbone / condition | RHAE | Levels | Scripts |
  |---|---|---|---|---|
  | R1 | Opus 4.7, action cap 20 | 46.82% | 116 | 898 |
  | R2 | Opus 5, online | 98.33% | 181 | 675 |
  | R3 | Opus 5, competition | 96.87% | 179 | 680 |
  | R4 | Opus 5, prompt priors removed | 99.95% | 183 | 734 |
  | R5 | Qwen3.8-27B-FP8 local (8×H100), offline, cap 50, "exp7" | 8.39% | 42 | 1,274 |
  | R6 | same, "v24" | 6.78% | 36 | 477 |
  | R7 | same, "v45" | 6.84% | 34 | 169 |

  R4 cleared 183 of 183 levels, which implies the full set. The paper does not explain how the Qwen prompt variants differ. In the Qwen runs, three games wrote no interaction record.
- Knowledge does not transfer as code. Only 33 of 630 script references cross a level boundary, and 74% of scripts written before a boundary are abandoned. Of 273 referenced scripts, 263 execute on import and 229 embed at least five numeric literals — [arXiv 2610.11450](https://arxiv.org/abs/2610.11450); [html](https://arxiv.org/html/2610.11450).
- The costliest error is "a hard-coded value carried into a task where it no longer holds":
  - m0r0: all three Opus 5 runs carried the L4 control direction into L5, where it is reversed. R2 and R3 recovered by turning the constant into a parameter (`def solve(polarity, ...)`, `S3 = int(sys.argv[1])`) and at L6 tried both values within a few actions. R4 edited the value in a dict and failed again at L6.
  - tu93 (R6, Qwen): the L2 `move(..., n=2)` parameter was replaced at L3 by a fixed `DIRS` offset table.
  - sp80 (R4): the L5 simulator hard-codes `ROWS=19, COLS=19`, so it is invalid at L6.

  Source: [arXiv html](https://arxiv.org/html/2610.11450).
- Notes are append-only and are never revised. In wa30 R4, a "KEY MECHANIC CORRECTION" (player is 1 cell) and a later "MAJOR REVISION" (back to 2 cells, which was correct) sit 65 lines apart, and neither is withdrawn. Notes grow to 1.3-1.4 MB (Opus) and 288-761 kB (Qwen), and contradictions are settled by reading the raw log — [arXiv html](https://arxiv.org/html/2610.11450).
- No intervention is tested. The only stated lesson is "The parameter is the repair that works". The authors say they will release the code, prompts and the R4 scorecard — [arXiv html](https://arxiv.org/html/2610.11450).

**EvoBelief: Falsifiable World Models for Self-Improving LLM Agents** ([arXiv 2610.12789](https://arxiv.org/abs/2610.12789); Tulane/Amazon; v1 8 Oct 2026). Not ARC-AGI-3.
- The model is frozen. The agent keeps an external belief store split into a per-episode state (z_t) and durable rules (W_t). Each rule is a hypothesis `C ∧ A ⇒ Y`, created when an outcome contradicts a prediction. **Competing explanations are kept side by side until evidence separates them.** Active experiments run only when their result could change a decision, with pass/fail criteria fixed before the run. A rule is promoted after support on 2 or more distinct object bindings in 2 or more episodes, with at least one support from a decision the rule did not itself influence. Counterexamples narrow, demote, then retire a rule. The actor never sees rule text; rules only change predictions, veto actions or supply plans — [arXiv html](https://arxiv.org/html/2610.12789).
- Measured success rates (Haiku 4.5 / GPT-5.4, no memory), from [arXiv html](https://arxiv.org/html/2610.12789):
  - ALFWorld (n=134): EvoBelief 66.42 / 76.12 vs Frozen WM 42.54 / 44.78, ReAct 33.58 / 46.27.
  - ScienceWorld (n=90): 64.44 / 71.11 vs WorldEvolver 52.22 / 65.56.
  - MiniGrid (60 episodes per cell, Haiku): MG-C 61.67 pre-drift and 95.00 post-drift, vs 0 / 0 for both Frozen WM and WorldEvolver.
- Ablation: without active experiments, key rules "never gain independent support", and performance stays at the prior's level — [arXiv html](https://arxiv.org/html/2610.12789).

**LLM-IDEA** ([arXiv 2610.11253](https://arxiv.org/abs/2610.11253); Shetty, Braga-Neto; 8 Oct 2026). Physics and ODE discovery, not games. When progress stalls, an "identifiability engine" classifies the stall as one of three cases: the agent is not capable enough, a better experiment of the same type would settle it, or no allowed experiment can settle it ("certified exhausted"). Without the engine, the agent reached depth ≥3 on 1 of 8 seeds — [arXiv 2610.11253](https://arxiv.org/abs/2610.11253).

**RSI-Anything** ([arXiv 2610.13912](https://huggingface.co/papers/2610.13912), 12 Oct 2026). A JEPA-based predictive-world-model RSI framework evaluated on code, GUI, lab planning and chemistry. It does not evaluate ARC-AGI-3 or any game. Its CodeContests gain over a static verifier is +2.83 points (95% CI +0.17 to +5.50; 200 tasks × 3 seeds) — [HF paper page](https://huggingface.co/papers/2610.13912). Low relevance.

**New repos (GitHub, created or pushed in window)**
- **HivemindEnt/hivemind-arc-agi-3-pub** (created 8 Oct, MIT-0): trace-replay statistics on dfranzen's Milestone-2 harness with Qwen3.8-Flash-Next, using 25 public games × 4 passes per run. Results are under Q2/Q3 — [repo](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub); [FINDINGS.md](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub/blob/main/FINDINGS.md).
- **SimonOuellette35/ARC-AGI-3-Training-Data** (created 6 Oct): 429 games in ARC-AGI-3 format (64×64 frames, ARC-AGI-3 action subset) with solvers and demonstration generators. Sources are augmented public ARC-AGI-3 games (e.g. ar25, ls20, sb26), Gym-Gridworlds, MiniGrid, Procgen, PuzzleScript and homemade games. No licence file — [repo](https://github.com/SimonOuellette35/ARC-AGI-3-Training-Data).
- **XINHAO-ZHANG/diagnose-agent-traces** (created 9 Oct, MIT): an agent skill that splits a run's transcripts into wall-clock waste and six per-level failure codes, including "no revision after contradiction" and "delay after key insight". It compares paired levels against a baseline with 95% intervals and reports agreement between two independent annotation passes. It is a diagnostic only, with no reported gains — [repo](https://github.com/XINHAO-ZHANG/diagnose-agent-traces).
- **r-y-ren/contest-compass-arc-prize-2026** (created 10 Oct): a reproduction of the RRSI self-evolving stack on a local 27B FP8. Its 12 Oct A/B is 0 wins out of 2 episodes for each of four arms. No signal — [runs aggregate](https://github.com/r-y-ren/contest-compass-arc-prize-2026/blob/main/fn_docs/results/2026-10-12-runs-aggregate.md).
- **kanishkpaul/arc-agi3-world-models** (pushed 24 Sep, so outside the window, and found only incidentally): a symbolic rule-induction agent cleared 0 official levels. Its written failure analysis names pixel-first deltas polluting evidence, an ungrounded `ACTION6(0,0)`, and over-confidence in rules learned from one example — [repo](https://github.com/kanishkpaul/arc-agi3-world-models).

### Inferences
- Tracing Thoughts sets a ceiling for our backbone class. In a strong frontier harness (PRO-LONG, about 99% RHAE with Opus 5), a 27B Qwen3.8 reaches only about 7-8% RHAE on the public set. The harness does not close the backbone gap. This is consistent with VISTA's earlier GLM-5.3-Flash 66.93: harness gains shrink sharply with smaller open models.
- The one repair that worked across levels, without being designed in, is *parameterise the constant and branch-test both values at the next level*. A cheap harness-level version would be a level-start instruction to list the constants the previous level's solution relied on and treat each as a parameter to re-test. Hivemind's measured "unanchor" arm (below) is the closest tested analogue, and it did not move levels won.
- EvoBelief's design matches the "keep competing hypotheses until separated" need: rules gate actions rather than appear as prompt text, and promotion requires evidence from a decision the rule did not influence. But it is measured only on text environments with frontier models, so its transfer to a 27B or Flash-class model on pixel games is unmeasured.

### Gaps
- No other ARC-AGI-3 arXiv paper dated 6-12 Oct was found besides 2610.11450. HF daily-paper listings were read only up to 9 Oct, and arXiv IDs from roughly 2610.14xxx onward (10-12 Oct) were checked only through web search.
- The difference between Tracing Thoughts' Qwen prompt variants (exp7/v24/v45) is not stated, so the 6.8-8.4% spread cannot be attributed to anything.
- No ARC Prize blog post dated in the window was found. The latest primary item found is the Milestone #2 results post on X, around 1 Oct, already covered in the earlier notes.

## Q2. Work on "agent anchors on a wrong goal / wrong mechanics" and hypothesis diversity at test time

### Takeaway
New evidence from our exact stack (dfranzen M2 + Qwen3.8-Flash-Next, Hivemind, 25 games × 4 passes) says the stall is mostly **mechanics anchoring at level boundaries, not wrong goal identification**. A prompt that explicitly un-anchors cut the anchoring behaviour about 3x but did not raise levels won (+3%, p=0.82). Tracing Thoughts independently finds the same boundary-anchoring failure at frontier scale.

### Cited Findings
- **Goal identification is mostly not the problem.** In 5 of 9 hand-read failed episodes, the model stated the right goal and kept it — [Hivemind FINDINGS](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub/blob/main/FINDINGS.md).
- **Level-boundary anchoring holds and was replicated.** Failed level-2+ attempts appeal to the previous level 2.0-2.6x more often than won levels. Failed levels spend 0.24 of their actions on the game's last control family, against 0.39 on won levels. The exceptions are three games that never cleared level 2 (bp35, lf52, sk48), each of which adds a new mechanic on level 2 — [Hivemind FINDINGS](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub/blob/main/FINDINGS.md).
- **Where the points go.** 97% of lost points are levels never finished, and 3% are extra actions on finished levels: 51.9 vs 1.6 points in the unanchor run and 58.6 vs 1.9 in ctrlprobe, each 25 games × 4 passes. The first uncleared level is L1 in about 10% of game-passes, L2 in 11-17% and L3 or later in 50%. 22-30% of game-passes clear every level — [Hivemind FINDINGS](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub/blob/main/FINDINGS.md).
- **The agent does revise.** After the game contradicts a "does nothing" statement, it revises in 59% of cases (67% for present-tense beliefs), usually within one turn. Its announced tests target uncertain controls: outcome surprise is 0.65 vs 0.29 bits after 6+ tries. The earlier claim "designs good experiments but draws bad conclusions" was withdrawn — [Hivemind FINDINGS](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub/blob/main/FINDINGS.md).
- **No-ops come from controls that already worked.** 73-82% of no-op actions (about 6-11 per game-pass) hit a control that had worked 2-3 times earlier on that level, and the agent is not told this. A "trackrecord" arm is queued but not yet measured — [Hivemind FINDINGS](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub/blob/main/FINDINGS.md).
- **Tracing Thoughts, frontier scale.** The agent never revises notes, only appends. The level-specific constant carried across a boundary is the costliest error, and the parameterise-and-test repair recovered it in 2 of 3 Opus 5 runs on m0r0 — [arXiv html](https://arxiv.org/html/2610.11450).
- **EvoBelief.** Keeping competing hypotheses side by side, plus decision-relevant active experiments, accounts for the MiniGrid gap (e.g. MG-C 61.67 vs 0). The ablation without experiments falls back to prior-level performance. Frontier models only — [arXiv html](https://arxiv.org/html/2610.12789).
- **Hypothesis diversity at test time.** No paper in the window directly tests hypothesis or goal diversity (e.g. sampling N alternative goal hypotheses) on ARC-AGI-3. The closest prior work found is outside the window: AGNI ([arXiv 2609.33870](https://arxiv.org/abs/2609.33870)), where agents see evidence of an environment change but fail to diagnose it and revise, and ReAdapt ([arXiv 2609.25284](https://arxiv.org/abs/2609.25284)), which adds continue/switch/abandon/clarify policy operations.

### Inferences
- For our harness, the measured target is "carry-over of last level's mechanics on levels 2+", not goal choice. Both independent sources (Hivemind on Flash-Next, Tracing Thoughts on Opus 5 and Qwen 27B) converge on it.
- Prompt-level un-anchoring changes the behaviour but not the outcome. The remaining candidates are therefore structural:
  - telling the agent which controls already worked on the level (Hivemind's trackrecord premise);
  - parameterising carried constants;
  - EvoBelief-style rules that gate actions instead of appearing as prompt text.
- None of these is yet measured on a Flash-class model.
- Three games that add a new mechanic on L2 (bp35, lf52, sk48) never clear L2. A targeted probe helped one of them (lf52, below), which suggests a per-game or new-mechanic-detection trigger rather than a global prompt change.

### Gaps
- Hivemind's goal-identification evidence is a hand read of 9 episodes, which is small n.
- No measured result yet for trackrecord, ctrlprobe-v2 or level inventory; Hivemind lists them as queued.

## Q3. Techniques tested and failed (negative results)

### Takeaway
Several harness interventions on the dfranzen/Duck family were measured null or negative in the window. Hivemind also measured hidden-leaderboard noise directly, which limits what any single submission can show.

### Cited Findings
All from [Hivemind FINDINGS](https://github.com/HivemindEnt/hivemind-arc-agi-3-pub/blob/main/FINDINGS.md), using dfranzen M2 + Qwen3.8-Flash-Next on 25 public games × 4 passes, unless stated.
- **Un-anchoring prompt (null).** The level-start prompt says not to carry previous mechanics forward, to re-test controls and to state alternative goals. Appeals to the previous level fell from 0.098 to 0.034 per turn on failed L2+ levels, but levels won went from 402 to 414 (+3%, p=0.82).
- **ctrlprobe, a sandbox `probe_controls()` diagnostic (negative overall).** lf52 cleared L2 in 3 of 4 passes, against 0 of 4 at baseline, but overall levels fell from 402 to 372 (−7.5%, p=0.2). An integration defect blocked 6 of 62 probes.
- **Statistical power.** A 25-game × 4-pass A/B detects only about a 16% change in levels won at 80% power. A 3% effect needs about 120 passes per arm. Targeted arms therefore run on 6 games × 16 passes with a pre-registered gate.
- **Stuck-level RESET watchdog (premise weak, not built).** The agent almost never resets by choice: 4 voluntary RESETs against 325 forced ones over 400 passes. Levels still open at 45 or 60 actions were later cleared 60% and 56% of the time, so a forced RESET would interrupt many wins. Unfinished levels end at a median of 14 turns (p90 26) against 6 for won levels, which suggests they run out of time rather than stall.
- **Carry-forward notes memory (negative, Qwen3.6-27B + Duck harness, single scored submissions).**
  - Duck v5: 0.14.
  - With archetype-classification and "deterministic first" scaffolding: 0.34. The same patches without that scaffolding: 0.15.
  - Adding carry-forward notes memory to the 0.34 build: **0.04**. It "kept a confidently wrong world model for over 100 turns in one game".
- **Hidden-leaderboard noise.** The unchanged public M2 notebook, submitted on two consecutive days, scored 28.49 and then 23.63. That pair straddles all five of Hivemind's own copies (25.07-25.96). Pooled run-to-run sd is about 2.4, so a +2 gain needs about 24 hidden scores per side.
- **Measurement pitfall.** The agent-visible verdict lines missed 15-39% of executed actions per run (up to 94% on one pass); count actions from the environment's event log instead. A claim that "83% of 'does nothing' claims are later contradicted" was withdrawn, because 73-100% of all actions change the board in every game.
- **Tracing Thoughts (negative observation, not an intervention).** Notes kept per level and script libraries do not transfer: 5% of script references cross a level boundary and 74% of scripts are abandoned. Append-only notes accumulate contradictions — [arXiv 2610.11450](https://arxiv.org/abs/2610.11450).
- **kanishkpaul symbolic rule induction (outside the window).** It cleared 0 official levels. The authors' failure analysis blames one-example rule over-confidence and pixel-first deltas — [repo](https://github.com/kanishkpaul/arc-agi3-world-models).

### Inferences
- Two independent results agree that persistent belief memory can harden wrong mechanics: Hivemind's notes memory (0.34 to 0.04) and Tracing Thoughts' append-only contradictions. Memory gains need a falsification or retirement rule (EvoBelief-style demote/retire). Raw carry-forward should not be shipped.
- At about 2.4 sd on the hidden score, and with 25 × 4 local A/Bs blind below about 16%, small harness tweaks cannot be validated on either scale. Targeted 6-game × 16-pass arms with pre-registered gates are the measured way to evaluate them. This is consistent with our own memory notes that the local bench does not predict public scores.

### Gaps
- Hivemind's interventions have not been submitted to the hidden leaderboard, and its Apache-licensed model swaps (Qwen3.8-27B, gpt-oss-120b, Gemma-4-31B) are queued but unmeasured.
- No negative results on test-time training or ensemble/multi-agent play for ARC-AGI-3 appeared in the window.
