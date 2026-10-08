# ARC-AGI-3 state of the art, late September 2026: who scores what, and how

Research date: 2026-09-30. Two settings are kept separate throughout:
- **[KAGGLE]**: in-kernel, no internet, open weights, one RTX Pro 6000. Scored on hidden environments, not on the 25 public games.
- **[API]**: unrestricted, frontier closed models. Almost always reported as RHAE on the **25 public games / 183 levels**, which the Kaggle score never uses.

The two sets of numbers are not on the same scale and cannot be compared directly. A 100 RHAE result on the public 25 does not mean anyone has a 100 on Kaggle.

## Q1. Current Kaggle leaderboard (arc-prize-2026-arc-agi-3) and how it has moved since June

### Takeaway
[KAGGLE] At the 2026-09-30 00:36 UTC snapshot, 3,492 teams are on the board. The leader is **Tufa Labs at 45.33**, followed by Yi-Chia Chen at 36.73 and Daniel Franzen at 26.55. The median is **0.32**. The whole top of the board moved roughly 20-40x after June, when the Milestone-1 winning score was 1.21. Most of that movement came from swapping in a newer Qwen model (Qwen3.8 27B, then Qwen3.8-Flash-Next NVFP4) under forks of the Duck harness. Tufa's own late-September method has not been published.

### Cited Findings
- Leaderboard snapshot, downloaded with `kaggle competitions leaderboard arc-prize-2026-arc-agi-3 -d` (file `arc-prize-2026-arc-agi-3-publicleaderboard-2026-09-30T00:36:17.csv`). Source: [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
  - 3,492 teams. Median 0.32, mean 1.39. 3,326 teams score above 0, 1,368 score at least 1.0, and 166 score exactly 0.
  - Quantiles: p90 = 3.89, p75 = 2.62, p50 = 0.32, p25 = 0.15.
  - By rank: #1 = 45.33, #5 = 20.53, #10 = 13.40, #20 = 7.99, #50 = 5.55, #100 = 4.82, #200 = 4.31, #500 = 3.57.
  - Top 12 (team, score, submission count, notable members):

    | Rank | Team | Score | Submissions | Notable members |
    |---|---|---|---|---|
    | 1 | Tufa Labs | 45.33 | 149 | jeroencottaar, driessmit1, pressman1 |
    | 2 | Yi-Chia Chen | 36.73 | 15 | |
    | 3 | Daniel Franzen | 26.55 | 86 | |
    | 4 | Lord Han Solo | 22.24 | 77 | |
    | 5 | Tong Hui Kang | 20.53 | 87 | |
    | 6 | the last dance | 20.51 | 66 | includes gklambauer |
    | 7 | Third Intelligence | 18.29 | 60 | |
    | 8 | NVARC3 | 16.07 | 23 | cpmpml, darraghdog, sorokin (NVIDIA Kaggle GMs) |
    | 9 | Son Pham & Mark Barney | 15.68 | 63 | |
    | 10 | rellik13 | 13.40 | 40 | |
    | 11 | Matija L & Zhongwei W & Fususu | 11.64 | 178 | |
    | 12 | Chew Kok Wah | 9.98 | 109 | |

  - Almost every top-30 team last submitted on 2026-09-28 or 2026-09-29, just before the Milestone #2 deadline.
- **June baseline.** Milestone #1 (deadline 2026-06-30, announced 2026-07-06) went to Tufa Labs' Duck, then Reki, then Md Boktiar Mahbub Murad's "forge". The third-place notebook is titled "LB 0.86". Source: [ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1)
  - The Duck's milestone-winning score was **1.21**; Tufa's own notebook says "the notebook that scored our milestone-winning 1.21". Source: [Tufa Duck notebook](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner)
  - AlphaSignal instead lists 1.03. That article also wrongly describes the Duck as a CNN+RL system (that is StochasticGoose), so it is unreliable. Source: [AlphaSignal](https://alphasignal.ai/news/tufa-labs-wins-25k-beating-frontier-ai-on-the-world-s-hardest-benchmark)
- **Mid-course markers.**
  - A public notebook titled "LB-9 arc3 duck v12 with Qwen 3.8 27B" last ran 2026-08-18. It uses `MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"`, so a Duck fork on Qwen3.8-27B reached about LB 9 by mid-August. Source: [Kaggle notebook foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b](https://www.kaggle.com/code/foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b)
  - A 33-day solo journal went from LB 0.87 to 2.56 between 2026-07-31 and 2026-09-01, using Qwen3.8-27B-FP8 with TAAF. Source: [Shih-Yu-Yeh/arc3-dev-agi-journal](https://github.com/Shih-Yu-Yeh/arc3-dev-agi-journal)
- **Milestone #2** (deadline 2026-09-30) pays $25K / $10K / $2.5K. Final submissions are due 2026-11-02 and results are announced 2026-12-04. Source: [arcprize.org competition page](https://arcprize.org/competitions/2026/arc-agi-3); [arXiv 2603.24621 search snippet](https://arxiv.org/abs/2603.24621)
- Kaggle rules: no internet during evaluation, and prize-eligible solutions must be open-sourced. Source: [ARC Prize 2026 docs](https://docs.arcprize.org/arc-prize-2026)

### Inferences
- The step from 1.21 (June) to about 9 (mid-August) to 45 (late September) matches the model upgrades that forks record: Qwen3.6-27B, then Qwen3.8-27B, then Qwen3.8-Flash-Next NVFP4 with MTP. Top notebooks keep the Duck prompts and loop unchanged. That makes model and serving throughput the largest single lever on the Kaggle board, well ahead of harness novelty. This is supported by fork descriptions but not by a controlled ablation.
- Team memory recorded the leader at 19.40 across 3,284 teams on 2026-09-24 (not externally re-sourced). If that is right, #1 more than doubled in the last week before the milestone. The late-September methods are therefore probably unpublished and not reflected in any public notebook.
- A team at 0.13-0.14 sits between p25 (0.15) and the bottom tail, roughly rank 2,700 of 3,492.

### Gaps
- The Kaggle Discussion tab could not be fetched (JS-rendered, and the CLI has no discussion endpoint). No late-September writeups from Tufa, Yi-Chia Chen, Daniel Franzen or NVARC3 were found.
- Milestone #2 winners are not announced yet (the deadline is the research date).
- No public leaderboard time series exists. The "since June" movement above is reconstructed from notebook titles, the milestone post, one journal, and team memory.
- The exact composition of the public-LB game set (how many semi-private games) is not verified here.

## Q2. Milestone #1 placed entries (announced 2026-07-06): methods and scores

### Takeaway
[KAGGLE] All three placed entries are LLM agents running locally.
- **1st, the Duck (Tufa Labs):** Qwen3.6-27B FP8 writes and runs Python in a REPL over the game state.
- **2nd, Reki**, and **3rd, forge:** both use Gemma-4-31B and emit one JSON action per step, with reflection memory and click heuristics.

In the two JSON-action entries, the best configurations turned most of the added machinery off.

### Cited Findings
**Duck (1st, 1.21 public; $25K).** Source: [ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1); [Tufa research page](https://tufalabs.ai/research/duck-harness); [GitHub](https://github.com/Tufalabs/duck-harness); [writeup, Kaggle discussion 717133](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133)
- Game observations are encoded as Python variables in a live REPL. The model inspects them with tool calls and writes code against pre-built helper functions.
- The grid is shown both as an image and as text, plus segmentation tools.
- Play is unbounded: the oldest messages are evicted to keep context small.
- Model: Qwen 3.6 27B FP8, served locally.
- The blog reports that hand-crafted tools hurt performance and letting the model improvise helped.
- Tufa's local result: mean 1.6002 ± 0.4475 over 25 public games × 20 attempts. Some games clear more than 40% of levels consistently; others fail level 1.
- With GPT-5.4 plugged in, the same harness is "an order of magnitude cheaper on each game" than Executable World Models at comparable performance.
- Tufa's readable notebook says the re-run did not reproduce the "lucky" 1.21, so there is real submission-to-submission variance. Source: [Kaggle notebook](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner)
- Authors: Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar, Stefano Viel. The solver ships as TAAF (Tufa ARC-AGI Framework).

**Reki (2nd, $10K).** Source: [ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1); [notebook](https://www.kaggle.com/code/ruichardliu/milestone1-2nd-solution)
- A VLM that reads the board and returns one JSON action per step. Gemma-4-31B, local.
- Reflection memory refreshed about every 10 steps.
- Numpy click heuristics and hardcoded fallback rules.
- A "dead-signature" mechanism marks object interactions that have no effect.
- Every feature can be toggled with an environment variable, for ablation.

**forge (3rd, Md Boktiar Mahbub Murad, LB 0.86, $2.5K).** Source: [ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1); [notebook](https://www.kaggle.com/code/mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone)
- The same JSON-action pattern, driven by config profiles: a candidate-action generator, arbiter scoring, optional confidence prompts, reflection memory, Gemma-4-31B.
- The top-scoring configuration disabled most of the advanced machinery.

**Post-June Duck forks that dominate the public Code tab** (vote-sorted list, retrieved 2026-09-30):
- "Duck Qwen3.8 Flash Next NVFP4 MTP" (ktyser, 2026-09-01). Changes only the serving stack:
  - `RadixArk/Qwen3.8-Flash-Next-NVFP4`, ModelOpt NVFP4 weights with BF16 compute
  - 3-token NEXTN MTP speculative decoding, 32K context, 8K batched-token cap
  - 8 vLLM sequences, CUDA graphs, prefix caching off, 28-way game concurrency, a vLLM watchdog
  - Duck prompts, loop and policy unchanged

  Source: [keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp). The author ktyser (keithtyser) is #15 on the leaderboard at 9.17.
- "taaf-flashnext-sheetu12b-0922" (Scott Le Grand, 2026-09-22). Adds six "agentfix" agent-loop repairs:
  - one board image per request
  - action results always returned
  - tolerant world-model parsing, with no wipe on game-over
  - guards against identical-snippet and no-action loops
  - HUD-band-aware change detection with a no-op guard and animation-frame summaries
  - ACTION7 made executable

  It also sweeps `MULTIMODAL_UPSCALE` away from 4. Source: [scottlegrand/taaf-flashnext-sheetu12b-0922](https://www.kaggle.com/code/scottlegrand/taaf-flashnext-sheetu12b-0922)
- The "Duck Qwen3.8 Anim Base" line (wuliao0, 2026-09-18) adds animation-frame handling. Source: [wuliao0/duck-qwen3-8-anim-base](https://www.kaggle.com/code/wuliao0/duck-qwen3-8-anim-base)
- A solo journal's biggest single gain came from raising `MULTIMODAL_UPSCALE` from 4 to 8, i.e. rendering the board at 512×512 instead of 256×256 (2.56 on 2026-08-25). Every later add-on (graft, NOOA modules) regressed. Source: [Shih-Yu-Yeh/arc3-dev-agi-journal](https://github.com/Shih-Yu-Yeh/arc3-dev-agi-journal)

### Inferences
- Both the Milestone-1 teams and later forks report the same pattern: extra hand-built machinery tends to lower the score. What helped was better perception (image rendering resolution, animation handling), a stronger model, and a more robust loop.
- The Duck/TAAF harness is effectively the public baseline everyone forks. A team with the Kaggle constraint gets more from adopting TAAF with a Qwen3.8 NVFP4 serving stack than from a bespoke explorer.

### Gaps
- The full text of the Duck writeup (discussion 717133) could not be fetched. The TAAF internals (prompt, helper list, eviction policy) came from secondary summaries only.
- No Reki or forge private scores were found.

## Q3. The named methods: AERA, StochasticGoose, Blind Squirrel, Tycho, Executable World Models, Compiled Agency

### Takeaway
Only AERA and the two Preview agents are in-kernel or small-model results:
- **AERA**: 0.30 private with a 0.5B model. [KAGGLE]
- **StochasticGoose**: 12.58% in the Preview.
- **Blind Squirrel**: 6.71% in the Preview.

The rest are frontier-API coding agents on the public 25 games, and they now saturate that set:
- Tycho: 100 RHAE
- Retrodict: 99.86
- NVIDIA AVO: 100
- Executable World Models: 58.12

"Compiled Agency" is not an ARC-AGI-3 paper. It covers a roguelike, StarCraft II and Civilization.

### Cited Findings

**AERA, "Explore Before You Solve" (arXiv 2605.25931, 2026-05-25). [KAGGLE, code track]** Source: [arXiv](https://arxiv.org/abs/2605.25931)
- Three phases: EXPLORE → VERIFY (test hypotheses about mechanics) → PLAN. The paper frames this as a speed-depth Pareto frontier between action efficiency and information gain.
- Model: Qwen2.5-0.5B.
- Public 25 games: RHAE 0.2116, 4 of 25 games solved. The random and no-explore baselines both score 0.0000.
- Private 55 games: 0.30 (code track).
- The paper claims every one of the 25 public games can be reached by non-intelligent strategies.

**StochasticGoose (Tufa Labs, Preview winner, 2025).**
- A CNN plus RL that predicts which actions will change the frame: 64×64 frames, a four-layer conv net, sparse-reward RL. 12.58%, 18 levels. Source: [AlphaSignal](https://alphasignal.ai/news/tufa-labs-wins-25k-beating-frontier-ai-on-the-world-s-hardest-benchmark) (secondary; the same article mislabels the Duck)
- It is the official Kaggle sample submission, "ARC3 Sample Submission - Stochastic Goose" (695 votes). Source: [Kaggle](https://www.kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose)
- The Preview scored levels completed, not quadratic action efficiency.

**Blind Squirrel (Preview 2nd, 6.71%, 13 levels).** Source: [wd13ca/ARC-AGI-3-Agents](https://github.com/wd13ca/ARC-AGI-3-Agents); [ARC Prize Preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- Builds a directed state graph from frames and prunes actions that loop or leave the state unchanged.
- **Back-labelling:** when the score rises, it labels that level's trajectory with distance-to-milestone and retrains a value model over (state, action).
- The value model is a pretrained ResNet-18 with a custom stem (16-colour embedding, then a conv layer) and a head that fuses an action embedding. For clicks, the embedding encodes the button's colour, size and shape.
- The value model is combined with a valid-actions model to pick the next action.

**Tycho (arXiv 2607.28287, Lehmann, Aioanei, Vahdati, 2026-07-30). [API]** Source: [arXiv](https://arxiv.org/abs/2607.28287); [GitHub NIMI-research/Tycho](https://github.com/NIMI-research/Tycho)
- Treats each environment as a "parameterised rendered deterministic Moore machine".
- Separates actionable observations from animation, level-complete and game-over frames. From that history, a coding agent builds, tests, plans with, repairs or bypasses an executable world model. The paper calls this "active abstraction".
- GPT-5.6 Sol and Opus 5 both reach **100.00 RHAE, 183/183 levels** on the public 25. Opus 5 uses about 61% fewer scored actions than the human baselines.
- Cost is about $2,986, per the Retrodict README's comparison. Source: [Retrodict](https://github.com/ryanbbrown/Retrodict)
- **Ablation over 4 orchestration policies:**
  - Actor-requested delegation to a model builder gives the best mean, 88.49.
  - Automatic repair matches transitions better but scores lower RHAE, 83.07.

**Executable World Models (arXiv 2605.05138, 2026-05-06, v2 2026-06-06, AGI-2026). [API]** Source: [arXiv](https://arxiv.org/abs/2605.05138)
- A coding agent keeps an executable Python world model, checks it against observations with verifier programs, and refactors under an MDL-like simplicity bias. No game-specific logic.
- GPT-5.5: 15/25 games fully solved, mean RHAE **58.12%**. GPT-5.4: 8 games, 41.29%.

**Ablation of those components, "Do Coding Agents Need Executable World Models, Simplification, and Verification…" (Rodionov, arXiv 2607.15439, 2026-07-16, rev. 2026-08-27). [API]** Source: [arXiv](https://arxiv.org/abs/2607.15439)
- Four Codex variants on GPT-5.4, 5.5 and 5.6-sol: textual, flexible executable, executable + simplification, and fixed-interface + simplification + exact replay verification.
- Textual beats flexible-executable in both GPT-5.5 conditions.
- Simplification beats executable-only in 3 of 4 settings.
- Full verification ranks first but costs much more compute.
- At GPT-5.6-sol max effort, **plain textual completes every public level with 41% fewer actions than humans**. At that point the three mechanisms are unnecessary; verification only helps at lower effort.

**Compiled Agency (arXiv 2609.18996, Huang & Xiao, 2026-07-17, rev. 2026-09-26).** Source: [arXiv](https://arxiv.org/abs/2609.18996)
- A "Gauntlet" develop, freeze, evaluate protocol. A coding agent builds a model-free game controller from a game description, a raw observation/action interface and an empty policy file. No model calls happen at play time.
- Environment access raises held-out success by 10-78 pp.
- StarCraft II controllers beat the strongest built-in AI. Refreshing the evidence panel mid-session gives +12.4 pts.
- **Not evaluated on ARC-AGI-3** according to the abstract.

### Inferences
- For a Kaggle team, the transferable ideas are:
  - Tycho's **frame-type separation** (animation, level-complete and game-over frames kept apart from actionable frames). Top Duck forks independently added animation handling.
  - Blind Squirrel's **back-labelled value model**.
  - AERA's explicit **explore then verify** gate.
- "Compiled Agency" suggests a hybrid (distil a controller offline, run it without an LLM), but ARC-AGI-3 needs per-game test-time adaptation. Relevance is indirect.

### Gaps
- No private or Kaggle score exists for Tycho, Executable World Models, Retrodict or AVO. They need APIs, which the kernel forbids.
- The Blind Squirrel and StochasticGoose Preview numbers come from secondary summaries. The Preview blog was not fetched in full.

## Q4. New work since July 2026: bootstrapping unwon games and action efficiency

### Takeaway
[API] After July, the public 25 games were saturated by frontier coding agents that keep a text or code hypothesis log and **retrodict**: they test candidate rules against recorded history before acting. The Twin paper names **goal inference** as the hard part, harder than dynamics modelling.

Action efficiency comes from two things:
- Build a model first, then plan: batch moves once the mechanics are confirmed.
- Probe with single actions while uncertain.

[KAGGLE] No in-kernel paper matches this. Kaggle gains come from forking TAAF plus model and serving upgrades.

### Cited Findings
- **GPT-6 Astra (ARC Prize blog, 2026-09-03). [API]** Source: [arcprize.org/blog/astra](https://arcprize.org/blog/astra)
  - 62.7% on the standard harness ($26,098).
  - 99.9% with the Provider Adapter harness, which keeps opaque reasoning state between requests and compacts long conversations ($19,332).
  - Fewer actions than the human baseline on 96.0% of levels, and 51.7% fewer actions per level on average.
  - The model invented its own algebraic notation and small symbolic world models / DSLs. Under PRO-LONG with code execution, it built tools such as maze solvers and patrol trackers.
- **Aggregator leaderboard, as of 2026-09-24:** GPT-6 Astra 62.7, Claude Opus 5 30.2, Gemini 3.8 Flash 10.4, GPT-5.6 Sol 7.8. These are base-model and standard-harness scores. Source: [BenchLM](https://benchlm.ai/benchmarks/arcagi3) (aggregator; verify on [arcprize.org/leaderboard](https://arcprize.org/leaderboard), which could not be parsed)
- **Twin (arXiv 2608.14490, Skoutnev, Acharya, Longhitano, Udell, Ellis, Drori, 2026-08-14). [API]** Source: [arXiv](https://arxiv.org/abs/2608.14490)
  - A test-time "digital twin": a coding agent builds an executable world model by simulation and interaction, with replay validation.
  - 179/183 levels (97.8%). Beats human action efficiency on 158/179 levels.
  - **Infers the goal before any reward on 156 levels (87.2%).**
  - Ablation: base model 7.8%, with harness 61.1%, with Twin 93.3%.
  - Key finding: "building a usable world model is simpler than anticipated, whereas the harder problem is inferring the right goal."
- **PRO-LONG (arXiv 2607.20064, Fox, Wang, Rosu, Dhingra, 2026-07-22). [API]** Source: [arXiv](https://arxiv.org/abs/2607.20064)
  - Keeps the full structured interaction log out of context; the agent queries it with code.
  - +18.0 pp over baseline coding agents, up to 76.1% pass@1, 4.2-5.8x fewer tokens.
  - Fable 5 reaches 97.4% best@2 for $1,750.
- **Retrodict (GitHub, ryanbbrown). [API]** Source: [GitHub](https://github.com/ryanbbrown/Retrodict)
  - 99.86% mean RHAE, all 25 games, $654, 7,703 actions, 660M tokens (about 5.5x fewer than the next baseline). Model: GPT-5.6-sol at max effort.
  - Text game log only, no images.
  - Hypotheses are tested against recorded history with Python **before acting**.
  - Every action states its expected board; a mismatch re-invokes the agent with a diff.
  - A `playbook.md` survives context resets.
  - Escalation: after more than 300 actions stuck on a level, it builds a simulator and runs state-space search.
  - Probes with single actions, then batches moves once mechanics are confirmed.
- **NVIDIA AVO (NVIDIA blog, 2026-08-21). [API]** Source: [NVIDIA](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/)
  - Agentic Variation Operators: persistent memory plus supervisory intervention, on Opus 5.
  - 100.00 RHAE, 183 levels, 6,624 actions. A system called VISTA used 7,542.
  - No component ablation.
- Also surfaced but not read:
  - Prime Agent, a self-improving RLM harness (arXiv 2608.23552). Source: [arXiv](https://arxiv.org/pdf/2608.23552); [AiCybr summary](https://aicybr.com/blog/prime-agent-open-source-rlm-harness-arc-agi-3)
  - DiG-bench, Discovery in Games (arXiv 2608.12593). Source: [arXiv](https://arxiv.org/pdf/2608.12593)
  - Knowledge-Centric Self-Improvement (arXiv 2607.19592). Source: [arXiv](https://arxiv.org/pdf/2607.19592)
  - Graph-Based Exploration for ARC-AGI-3 (arXiv 2512.24156). Source: [arXiv](https://arxiv.org/pdf/2512.24156)

### Inferences
- Bootstrapping without a winning example is handled the same way across Twin, Tycho and Retrodict:
  1. Build a dynamics model from probes.
  2. Hypothesise the goal from the structure of the model and the scene.
  3. Plan to that goal inside the model.

  No example win is needed. This is a different axis from Blind Squirrel-style back-labelling, which only starts working after the first clear.
- For in-kernel use, the cheapest items to transfer are retrodiction (check a hypothesis against the logged history before spending actions) and predicted-next-frame mismatch triggers. Both are compute-light and model-agnostic.

### Gaps
- None of these methods has been shown working with a ≤30B open-weights model in the kernel. The transfer claims above are untested.
- Prime Agent, DiG-bench and Knowledge-Centric Self-Improvement were found but not read. Their results are unverified.

## Q5. What component ablations exist

### Takeaway
The ablation evidence is thin and nearly all [API]. The findings that repeat across sources:
1. At high reasoning effort, executable world models, simplification and verification add little (Rodionov). Verification helps at low effort.
2. Goal inference is the bottleneck, not dynamics (Twin).
3. Model-builder delegation beats auto-repair on RHAE (Tycho).
4. Exploration is necessary: no-explore scores 0 (AERA).
5. On Kaggle, extra hand-built tools hurt, while perception resolution and model upgrades help (Duck, forge, the Yeh journal).

### Cited Findings
- **Verification, simplification, world model:** Textual beats flexible-executable; simplification beats executable in 3 of 4 settings; full verification is best but costly; all are unnecessary at max effort. Source: [arXiv 2607.15439](https://arxiv.org/abs/2607.15439)
- **World model vs goal inference:** base 7.8%, harness 61.1%, Twin 93.3%, and goal inference is the harder problem. Source: [arXiv 2608.14490](https://arxiv.org/abs/2608.14490)
- **Orchestration policy:** actor-requested builder 88.49 vs auto-repair 83.07 RHAE. Source: [arXiv 2607.28287](https://arxiv.org/abs/2607.28287)
- **Exploration:** AERA 0.2116 vs 0.0000 for both the no-explore and random baselines. Source: [arXiv 2605.25931](https://arxiv.org/abs/2605.25931)
- **Tools / machinery [KAGGLE]:**
  - Hand-crafted tools hurt the Duck.
  - forge's best config turned most machinery off.
  - Reki made every feature env-toggleable, but no published ablation table was found.

  Source: [ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- **Perception [KAGGLE]:** rendering upscale 4 → 8 was the single largest gain (LB 2.56), and later add-ons regressed. Source: [Shih-Yu-Yeh journal](https://github.com/Shih-Yu-Yeh/arc3-dev-agi-journal)
- **Harness vs harness [API]:** the Duck harness on GPT-5.4 matches Executable World Models at about 10x lower cost per game. Source: [Tufa research page](https://tufalabs.ai/research/duck-harness)
- **Harness matters as much as the model [API]:** GPT-6 Astra scores 62.7% on the standard harness and 99.9% on the Provider Adapter harness. Source: [arcprize.org/blog/astra](https://arcprize.org/blog/astra)

### Inferences
- For a one-GPU, in-kernel team, the evidence ranks the levers as:
  - (a) model and serving throughput: Qwen3.8 NVFP4 + MTP, more concurrent games
  - (b) perception: a higher-resolution board render, animation and HUD-aware change detection
  - (c) loop robustness: loop guards, no world-model wipe on game-over
  - (d) exploration and verification structure

  Hand-built tools rank last. This ordering is inferred from fork notes and milestone commentary, not from a controlled Kaggle ablation.

### Gaps
- No controlled ablation on the Kaggle private or semi-private set has been published by any team.
- No study isolates segmentation/perception versus search under the RHAE metric in-kernel.
