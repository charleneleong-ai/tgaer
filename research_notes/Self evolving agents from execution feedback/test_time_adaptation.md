# Test-Time Adaptation and In-Context RL for Single-Episode Learning of Novel Environment Dynamics

Scope: 2025–2026 state of the art, prioritised toward interactive grid-game benchmarks (ARC-AGI-3: 64x64 board, 16 colours, discover dynamics + goal by acting, scored on action efficiency). Research date: 2026-09-26.

**Headline framing for the report writer:** the decisive 2026 result is that ARC-AGI-3 went from "all frontier models below 1%" (March 2026) to a claimed 99.9% (September 2026) **without a new adaptation algorithm** — the delta came almost entirely from harness/scaffold design (context retention, executable world models, structured evidence logs, compaction), not from gradient-based test-time training or in-context RL. Gradient TTT remains the winning technique on *static* ARC-AGI-2, but on the *interactive* benchmark the winners are LLM-as-programmer / explicit-world-model harnesses.

---

## Q1: Leading test-time adaptation / test-time training / in-context RL methods published or updated in 2025-2026

### Takeaway
Two distinct lineages. On static ARC-AGI-2, gradient **test-time training (TTT/TTFT)** still owns the leaderboard (NVARC 24.03%). On interactive ARC-AGI-3, the 2026 winners do **no gradient updates at all** — they are scaffolds around frontier LLMs that build explicit, executable, verifiable world models in-context (Tycho, Retrodict, VISTA, AVO, Schema). A third, smaller lineage does true in-episode weight updates on LLM agents (aTTT), with modest gains.

### Cited Findings

**Gradient test-time training (static ARC, the reference lineage)**
- Test-time training is responsible for the top scores in both ARC Prize 2024 (the ARChitects) and 2025 (NVARC); the ARC-AGI-1 jump from near-zero to near-human came from the shift away from massive pre-training toward test-time compute — [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- ARC Prize 2025 ran 2025-03-26 to 2025-11-03; 1,455 teams, 15,154 Kaggle entries — [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- ARC-AGI-2 private eval final standings: **NVARC 24.03%** (1st, $25k) — synthetic data generation + test-time training on a 4B-parameter model; **the ARChitects 16.53%** (2nd, $10k) — 2D-aware masked-diffusion LLM with recursive self-refinement and perspective-based scoring; **MindsAI 12.64%** (3rd, $5k) — TTFT pipeline with augmentation ensembles, tokenizer dropout, new pretraining tricks — [ARC Prize 2025: Technical Report](https://arxiv.org/html/2601.10904v1)
- Leading entry cost **$0.20 per task** in compute — [ARC Prize 2025: Technical Report](https://arxiv.org/html/2601.10904v1)
- **Conflict to flag:** a secondary search summary of the same competition reported MindsAI at 15.42% on ARC-AGI-2; the ARC Prize technical report states 12.64%. Prefer the technical report figure — [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis); vs [ARC Prize 2025: Technical Report](https://arxiv.org/html/2601.10904v1)
- Zero-pretraining / tiny-model line: **Tiny Recursive Model (TRM)** — 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a 7M-parameter network via recursive latent refinement; **CompressARC** — 76K parameters, no pretraining, one model trained on a single target task (MDL-based compression) — [ARC Prize 2025: Technical Report](https://arxiv.org/html/2601.10904v1). A living survey additionally quotes CompressARC at 20–34% on ARC-AGI-1 — [The ARC of Progress towards AGI](https://arxiv.org/html/2603.13372v1)
- Follow-up work exists on **Test-time Adaptation of Tiny Recursive Models** — [arXiv 2511.02886](https://arxiv.org/pdf/2511.02886) (not fetched in full; title-level citation only)
- Direct head-to-head of the two static-ARC adaptation paradigms: **Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning** — [arXiv 2507.15877](https://arxiv.org/pdf/2507.15877) (title-level only; numbers not extracted)

**In-episode gradient TTT on LLM agents (2026)**
- **aTTT (Agentic Test-Time Training)**: token-level reweighting that downweights loss on tokens appearing in repeated n-grams from prior updates while leaving novel tokens fully weighted; applies LoRA updates *inside* a live episode. Gains: **+5.0 points success on ALFWorld** (50-step budget, immediate env feedback) and **+4.9 points on SWE-bench Lite**. Implemented with vLLM's runtime LoRA API in a concurrent serving system; overhead **1.9x** the no-TTT cost. Models: Qwen3.5-4B/9B/27B and Gemma-3-12B — [No Time Like the Present: Agentic Test-Time Training for LLM Agents, arXiv 2607.03441](https://arxiv.org/html/2607.03441v1)
- Adjacent 2026 self-improvement-with-weight-updates work: **SIA (Self Improving AI with Harness & Weight Updates)** — [arXiv 2605.27276](https://arxiv.org/html/2605.27276); **Workspace Optimization: How to Train Your Agent** — [arXiv 2605.09650](https://arxiv.org/pdf/2605.09650); **Self-Improving LLM Agents at Test-Time** — [arXiv 2510.07841](https://arxiv.org/pdf/2510.07841) (all title-level citations; not fetched)
- The field has institutionalised: **NeurIPS 2026 TTCL Workshop — Towards Test-Time Continual Learning Agents** — [ttcl-agents.github.io](https://ttcl-agents.github.io/)

**In-context RL lineage (2025-2026, no weight updates)**
- A 2026 survey organises the ICRL literature around what changes (reward spec, transition dynamics, observation channels, action interfaces, constraints, demonstration distribution), how the change unfolds, and how observable it is. Named method families: **decision-pretrained transformers (DPT)**, **algorithm distillation (AD)**, **long-context meta-RL**, **retrieval-augmented agents**, **value- and model-aware ICRL**, **reward-feedback agents**. Core stated difficulty: "the policy must infer both the current decision rule and which parts of its accumulated evidence still support that rule" — [In-Context Reinforcement Learning under Non-Stationarity: A Survey, arXiv 2607.11906](https://arxiv.org/abs/2607.11906)
- **DICP — Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning** (ICLR 2025): distils an RL algorithm into a transformer that does in-context *model-based* planning rather than pure policy imitation — [arXiv 2502.19009](https://arxiv.org/pdf/2502.19009); [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/5c4a7643ab90cd62320df95c873a1c6f-Paper-Conference.pdf)
- **Vintix: Action Model via In-Context Reinforcement Learning** (2025) — cross-domain ICRL action model — [arXiv 2501.19400](https://arxiv.org/pdf/2501.19400)
- **Yes, Q-learning Helps Offline In-Context RL** (2025) — adds value learning to offline ICRL — [arXiv 2502.17666](https://arxiv.org/pdf/2502.17666)
- **Random Policy Enables In-Context Reinforcement Learning within Trust Horizons** — ICRL from random-policy data, bounded by a "trust horizon" — [arXiv 2410.19982](https://arxiv.org/pdf/2410.19982)
- **CORAL (In-Context RL via Communicative World Models, IJCAI 2026, IBM Research)**: pretrains an *Information Agent* as a world model across diverse tasks, with the objective of building a world model and distilling its understanding into concise messages consumed by a control agent. Authors claim significant sample-efficiency gains and successful **zero-shot adaptation in entirely unseen sparse-reward environments** — [IBM Research publication page](https://research.ibm.com/publications/in-context-reinforcement-learning-via-communicative-world-models) (author claim; no numbers retrieved)
- **Continual ICRL in non-stationary environments**: formally defines the setting and asks which architectures/training strategies let agents master new dynamics while discarding stale information; benchmark suites span symbolic reasoning and physics-based control — [OpenReview: Towards Unpredictable Worlds](https://openreview.net/forum?id=h8OJb8YGNa)
- **Safe ICRL** variants (2025-2026): shielding with function encoders + conformal prediction for unseen OOD environments; constrained decision transformers for OOD generalisation; **EPPO** (exact penalty policy optimization) — [Safe In-Context Reinforcement Learning, arXiv 2509.25582](https://arxiv.org/html/2509.25582v1); [Latent Q-Barrier Shielding, arXiv 2605.25267](https://arxiv.org/pdf/2605.25267)
- Meta-RL methodology audit: **How Should We Meta-Learn Reinforcement Learning Algorithms?** — [arXiv 2507.17668](https://arxiv.org/pdf/2507.17668) (title-level only)

### Inferences
- The two lineages barely intersect. Nothing in the ARC-AGI-3 winner set uses AD/DPT-style in-context RL or gradient TTT; they use frontier LLMs plus a harness. For an ARC-AGI-3-shaped problem, the ICRL literature is currently a source of *framing* (evidence staleness, trust horizons, model-aware ICRL) rather than of transferable implementations.
- aTTT's magnitude (~5 points, 1.9x cost) is the realistic ceiling for in-episode gradient adaptation on a small open model today, and it requires dense per-step environment feedback. ARC-AGI-3's reward is sparse until level completion, so aTTT-style updates have little signal to fit inside a first-contact episode.

### Gaps
- No independently reproduced numbers for CORAL; only the IBM publication abstract was reachable.
- The ICRL non-stationarity survey's abstract carries no benchmark table; per-method numbers would require the full PDF (PDF text extraction failed).

---

## Q2: Which methods learn environment dynamics ONLINE within one episode, and how is the learned model represented?

### Takeaway
The methods that actually work on first-contact interactive environments in 2026 represent learned dynamics as **executable Python world models plus natural-language rule files, held in context and on disk — not in weights**. Tycho's ablation is the cleanest evidence that the explicit model is doing real work: 79.07 → 88.49 RHAE purely from adding a delegated world-model builder.

### Cited Findings

**Representation = executable program + structured evidence log (Tycho)**
- **Tycho** operationalises "active abstraction": constructing explicit, testable models from costly interaction while deciding when the model's cost is justified. Components: **Actor** (sole agent taking environment actions), **Builder** (optional specialist that constructs/repairs executable Python world models), **Evidence** (structured interaction history with typed frames: decision / transient / terminal), **Verification** (replaying recorded transitions against predicted dynamics), **Planning** (search over model states for action sequences reaching level completion) — [Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3, arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
- Matched-policy ablation, Claude Opus 4.8, 25 public games (RHAE): **no world model 79.07**, **single / integrated actor-modeling 85.36**, **orchestrator / builder-on-request 88.49** (selected), **trigger / automatic repair 83.07** — [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
- Counter-intuitive finding: the **Trigger** variant had the **highest transition-prediction accuracy (88.1%)** yet underperformed — "accurate transition prediction alone does not ensure strong gameplay" — [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
- Frontier transfer with the selected policy: **GPT-5.6 Sol 100.00 RHAE** (7,766 actions, all 183 levels); **Opus 5 100.00 RHAE** (6,641 actions, all 183 levels); Opus 5 used **38.8% of aggregate human baseline actions** across all 183 levels — [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
- Inference budget: **$750 per game** in the matched study, **$1,500** for frontier runs (not reached) — [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)

**Representation = hypothesis-as-code replayed over a frame log (Retrodict)**
- **Retrodict** writes every frame returned by the game into a log file and reasons over the log with code rather than images. Before acting it proposes hypotheses about mechanics and tests them against recorded history by "writing python that replays a hypothesis over past frames, where being wrong costs nothing." It then commits a queue of actions **with predicted board states**; the runner plays the queue one action per step and returns to the model only when the plan is exhausted or **a prediction misses**, together with the diff of what went differently. Confirmed rules are curated into a persistent **playbook** file that survives context resets — [Retrodict (GitHub)](https://github.com/ryanbbrown/Retrodict)
- Results: **99.86% mean RHAE**, all 183 levels of all 25 public games, **$654** at API list prices, **660M tokens**, **7,703 actions**, model **gpt-5.6-sol at max reasoning effort**. 23/25 games at 100% RHAE; sk48 98.64%, sp80 97.77%. Support: ThinHarness runner, sandboxed venv with NumPy/SciPy/NetworkX, one vision-model image priming pass before the first move — [Retrodict (GitHub)](https://github.com/ryanbbrown/Retrodict)

**Representation = natural-language hypothesis files + lossless visual memory (VISTA)**
- **VISTA** = visual perception + language reasoning + lossless visual memory. Perception: raw **512x512 PNG** (8x nearest-neighbour upscale of the 64x64 grid) with `read_pixels` and `inspect` tools for pixel-level access. Reasoning: free-form natural-language hypotheses maintained in two files — **`GUIDE.md`** (durable game understanding) and **`WORKING.md`** (level-specific scratchpad). Memory: every frame preserved at full resolution, indexed by turn and frame, retrieved explicitly via `inspect` for pixel-level comparison across game history — [VISTA: A Visual Harness for Reasoning in an Interactive World](https://vista-research.github.io/)
- Results: **Claude Opus 5.0 — RHAE 100.00**, 25/25 games, all 183 levels, 25/25 perfect games, **7,542 actions vs 17,135 human baseline = 56% fewer actions**. **GPT-5.6 Sol — RHAE 98.27**, 25/25 games, 183 levels, 22/25 perfect, 10,063 actions — [VISTA](https://vista-research.github.io/)
- Modality robustness: the same games (S5I5, CD82) were solved from a 1D text grid, a 2D PNG, and a 3D isometric render, with distinct trajectories — so the representation of the *observation* is not the bottleneck — [VISTA](https://vista-research.github.io/)
- VISTA is described by its authors as the first system reaching perfect/near-perfect ARC-AGI-3 scores **without program synthesis**, versus concurrent program-based approaches (Schema, Retrodict, Tycho) at 95.35–100.00 RHAE — [VISTA](https://vista-research.github.io/)

**Representation = persistent memory + supervisor (NVIDIA AVO)**
- **AVO** keeps persistent memory across sessions — prior implementations, evaluation results, compiler/profiler outputs, accumulated reasoning — and runs a loop of hypothesis formation → action → evidence observation → state preservation → refinement. A **supervisor** monitors the broader search trajectory and redirects the main agent when progress stalls — [NVIDIA AVO blog](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/)
- Results: **RHAE 100.00** on all 25 public environments, all 183 levels, **6,624 total environment actions** — ~**12% fewer** than VISTA's 7,542 on identical public tasks (NVIDIA notes different backends and observation representations). Observations were **text-only 64x64 text grids**, not image tokens — [NVIDIA AVO blog](https://developer.nvidia.com/blog/...)
- NVIDIA's own caveat: the 100.00 reflects "the complete agent system, not only the underlying model" — **Claude Opus 5 alone scores approximately 30%** — and results apply to the public set only — [NVIDIA AVO blog](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/)
- An open-source Qwen-backed reimplementation exists — [avo-qwen-arcagi3 (GitHub)](https://github.com/criticaldata/avo-qwen-arcagi3)

**Representation = weights updated online (reset-free RL, non-LLM)**
- **Online Agent (OA)** for continual RL uses shallow-but-wide networks supporting efficient **Follow-The-Leader online learning**, fitting a linear model online **per time step** on top of a high-dimensional sparse feature encoder. The setting is explicitly reset-free: acting and learning both happen per-timestep, with no episode resets — [Continual Reinforcement Learning by Planning with Online World Models, arXiv 2507.09177](https://arxiv.org/pdf/2507.09177)
- **Reinforcement World Model Learning for LLM-based Agents** — [arXiv 2602.05842](https://arxiv.org/abs/2602.05842); **Qwen-AgentWorld: Language World Models for General Agents** — [arXiv 2606.24597](https://arxiv.org/html/2606.24597v1) (title-level only)

**Representation = induced symbolic rules across episodes (not single-episode)**
- **Cogito, Ergo Ludo**: after each episode the agent enters a reflective phase, with the LLM performing **rule induction** by analysing episode trajectories in light of previously held rules — an explicit natural-language rule base plus a planner — [arXiv 2509.25052](https://arxiv.org/pdf/2509.25052) (mechanism from search snippets; PDF text extraction failed, so no numbers)

### Inferences
- Across all four ARC-AGI-3 winners the learned dynamics live in **externalised, re-readable artefacts** — a Python model, a frame log, a `GUIDE.md`, a playbook — precisely so they survive context compaction. That is the load-bearing design choice, not the model backend. For a tgaer-style agent, the actionable pattern is: append-only frame log + hypothesis-as-code replay + a durable rule file that survives context resets.
- Tycho's Trigger result (best transition accuracy, worse RHAE) argues against optimising a dynamics model for prediction accuracy as a proxy objective. The useful model is the one that *changes the plan*, and building it costs actions, so model-building must be gated on expected action savings.
- Retrodict's "being wrong costs nothing" reframing is the single most transferable idea for an action-efficiency-scored benchmark: move hypothesis falsification off the environment (replay against the log) and onto free compute. Cost evidence supports this: $654 / 660M tokens at 99.86% versus a 98.97% baseline at $2,722 / 3,640M tokens.

### Gaps
- No ablation anywhere isolating *how much* of the 100% RHAE is the frontier model versus the harness, beyond NVIDIA's single data point (Opus 5 bare ≈30% vs 100.00 with AVO) and Schema's 42.83% → 98.98%.
- Cogito Ergo Ludo's benchmark numbers could not be retrieved (PDF extraction failure).

---

## Q3: Does algorithm distillation / in-context RL actually work on genuinely unseen environments?

### Takeaway
The evidence is mixed and mostly in-distribution-adjacent. Original-author claims of OOD generalisation hold for tasks drawn from a *perturbed* version of the training distribution; documented failure modes appear as soon as the exploration strategy or the action interface changes. No ICRL/AD result was found on ARC-AGI-3 or any comparable first-contact interactive benchmark.

### Cited Findings
- Algorithm Distillation trains a transformer to sequentially model the *entire learning histories* of a source RL algorithm across tasks, so it replicates the source algorithm's exploration-exploitation behaviour and can tackle novel tasks purely in-context, with no parameter updates — [In-context Reinforcement Learning with Algorithm Distillation, arXiv 2210.14215](https://arxiv.org/pdf/2210.14215) — **2022 paper; still the reference result for this lineage**
- Original-author claim: AD is more data-efficient than the source RL algorithm that generated its training data, shows accelerated learning curves with fewer environment interactions, and can generalise to tasks "significantly different from the training distribution," which the authors attribute to learned latent dynamics of policy improvement — [arXiv 2210.14215](https://arxiv.org/pdf/2210.14215); secondary summary — [alphaXiv](https://www.alphaxiv.org/abs/2210.14215v1)
- Independent-ish replication attempt exists as a public write-up series — [Experiments With Algorithm Distillation, Part 1](https://1a3orn.com/sub/machine-learning-algorithm-distillation-1.html) (blog; not peer-reviewed, and I did not verify its conclusions in detail)
- Documented failure mode 1: **AD-EPS** (AD with injected action noise) fails to generalise to OOD environments — the noise produces random action variation rather than *structured* exploration — [Safe In-Context Reinforcement Learning, arXiv 2509.25582](https://arxiv.org/pdf/2509.25582) (search-snippet level; flag as lower confidence)
- Documented failure mode 2: AD struggles with **novel action spaces**, which motivated a dedicated line of work — [In-Context Reinforcement Learning for Variable Action Spaces, arXiv 2312.13327](https://arxiv.org/pdf/2312.13327)
- Structural limitation restated in 2025 work: ICRL models "inherit the suboptimal behaviors of the algorithms they imitate"; if the source algorithm is poor, distillation will not fix it — [Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning, arXiv 2502.19009](https://arxiv.org/pdf/2502.19009)
- The 2026 survey's framing concedes the open problem: accumulated in-context evidence becomes "stale, misleading, or useful again" as the task shifts, and the policy must simultaneously infer the current decision rule and which of its evidence still supports that rule — [arXiv 2607.11906](https://arxiv.org/abs/2607.11906)
- Reported robustness gains in 2026 ICRL work are narrower than "novel environment": e.g. RL eliciting contextual learning of *unseen language translation* — [arXiv 2606.06428](https://arxiv.org/abs/2606.06428)

### Inferences
- "Unseen environment" in the AD/ICRL literature almost always means a held-out task from a parameterised family (new goal location, new reward vector, perturbed dynamics) with a **fixed observation and action interface**. ARC-AGI-3 violates that: each environment has different dynamics, different semantics for the same five action ids, and an unknown goal. The AD lineage has no published result under those conditions.
- The contamination-free verdict on ICRL-for-ARC-AGI-3 is therefore "untested", not "disproven". But the burden is high: AD needs a large corpus of learning histories across a task family, and ARC-AGI-3 environments are hand-built and few, so the generator distribution would have to be synthesised — which is exactly the NVARC recipe for *static* ARC, not for interactive.

### Gaps
- I found no paper that runs AD / DPT / Vintix on ARC-AGI-3 or a comparable first-contact interactive suite. This appears to be a genuine hole in the literature as of 2026-09.
- The AD-EPS failure claim rests on a search snippet from the Safe ICRL paper; PDF text extraction failed, so treat as unverified.

---

## Q4: Handling the no-reset constraint, where every exploratory action is permanently spent

### Takeaway
ARC-AGI-3 is not literally reset-free — there is an Undo action and RESET exists — but every action including RESET is **scored**, so exploration is priced. The 2026 winners solve this by making falsification free: they replay hypotheses against a recorded transition log in Python instead of testing them in the environment, and they gate expensive model-building on whether it will save actions.

### Cited Findings
- ARC-AGI-3's action set: five key actions, plus an **Undo** action (reverting to the previous state), plus one cell-selection action specifying grid coordinates. An "action" is defined as a discrete interaction with the environment — internal reasoning steps that do not alter environment state are explicitly excluded — [ARC-AGI-3 Technical Report, arXiv 2603.24621](https://arxiv.org/html/2603.24621v1)
- The benchmark deliberately measures the number of moves required to solve a new environment **upon first contact**, to penalise brute-force exploration relative to efficient modelling; agents encounter each environment once on first contact — [arXiv 2603.24621](https://arxiv.org/html/2603.24621v1)
- Tycho's accounting: every environment action increments the per-level counter, and **RESET costs one scored action**; level-completion and game-over transitions are unscored but close the attempt. Tycho's typed-frame protocol distinguishes decision frames (where actions occur) from transient animations and terminal frames, which prevents fabricating transitions between non-consecutive decision points — a correctness requirement for replay-based falsification — [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
- Retrodict's explicit answer to the constraint: test hypotheses by replaying them over *past* frames in Python, "where being wrong costs nothing", and only spend environment actions on a queued plan with predicted states, returning to the model on a prediction miss — [Retrodict (GitHub)](https://github.com/ryanbbrown/Retrodict)
- Tycho's "active abstraction" is framed as deciding **when acquiring or using the model justifies its expense** — i.e. an explicit exploration-cost gate rather than always-on modelling — [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1)
- ARC Prize's failure analysis of the pre-harness era names wasted exploration as a top failure mode: models map unfamiliar mechanics onto known games (Tetris, Frogger, Sokoban), and "a local visual resemblance becomes a full gameplay theory, then the model wastes actions testing the wrong affordances" — [Analyzing GPT-5.5 & Opus 4.7 with ARC-AGI-3](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)
- Epistemic-agent framing of the same trade-off: **Explore Before You Solve: The Speed–Depth Trade-off in Epistemic Agents for ARC-AGI-3** — thorough exploration improves solution quality but costs steps; the agent allocates a finite exploration budget before solving, treating exploration as an investment with diminishing returns — [arXiv 2605.25931](https://arxiv.org/pdf/2605.25931) **(low confidence: PDF extraction was partial and the fetch summary may have conflated details; no scores were recoverable)**
- True reset-free RL (non-LLM, for contrast): reset-free online learning requires acting and learning per timestep with no state resets — [arXiv 2507.09177](https://arxiv.org/pdf/2507.09177). Older reference result: **MoReFree** (model-based reset-free, world-model-driven) outperformed state-of-the-art model-free methods in **7 of 8 tasks** in a setting where the environment is reset only once at the very beginning of training — [Reset-free RL with World Models (Lacuna summary)](https://lacuna.tiptreesystems.com/work/reset-free-reinforcement-learning-with-world-models/wrk_ed69c68892c0261afa05a7dd18176a27) — **pre-2025 work; label as older**

### Inferences
- The operational rule the 2026 results support: **the only actions worth spending are ones whose outcome you cannot compute from the log.** Everything else (rule confirmation, plan validation, counterfactual checks) should be moved into a replay simulator. Undo is cheap *relative to* a wrong multi-action plan but is still scored, so it is not free backtracking.
- Because RHAE squares the ratio (see Q5), a 2x action overrun costs 75% of the level's score. That makes action budgeting the dominant optimisation target, well ahead of level-completion rate once completion is achievable — which matches Tycho and AVO competing on action counts (6,641 vs 6,624) at identical 100.00 RHAE.

### Gaps
- No paper reports a clean ablation of "exploration actions spent" versus "replay-falsified hypotheses" as a ratio; the efficiency gains are reported only as aggregate action counts.

---

## Q5: Results on ARC-AGI-3 and comparable interactive/agentic benchmarks — exact scores and approaches

### Takeaway
Public-set ARC-AGI-3 is saturated (four independent systems at 99.86–100.00 RHAE, all in 2026, all harness-based, all on frontier LLMs). The official semi-private leaderboard tells a different story: 62.7% for GPT-6 Astra under ARC Prize's Standard harness versus a claimed 99.9% under the Provider Adapter harness — the same model, ~37 points apart.

### Cited Findings

**Benchmark definition and metric**
- Observation: a **64x64 grid, each cell one of 16 colours**; agents receive a frame or sequence of frames per turn, with frame sequences enabling non-interactive animations — [arXiv 2603.24621](https://arxiv.org/html/2603.24621v1)
- Environments are strictly turn-based — nothing changes until the agent acts — so the benchmark rewards careful reasoning over reflexes — [DataCamp overview](https://www.datacamp.com/blog/arc-agi-3) (secondary source)
- **RHAE (Relative Human Action Efficiency)** per level: **S(l,e) = min(1.0, h(l,e) / a(l,e))²**, where h is the **second-best human action count** and a is the agent's count. Squared to penalise inefficiency, weighted by level difficulty (later levels weighted more heavily), averaged across environments — [arXiv 2603.24621](https://arxiv.org/html/2603.24621v1)
- Human baseline: median environment attempt **7.4 minutes**, successful attempts **8.1 minutes**; **100% solvability** confirmed, every environment passed by ≥2 independent human testers — [arXiv 2603.24621](https://arxiv.org/html/2603.24621v1). Human cost reference ≈ **$12.78 per attempted game** — [ARC Prize: GPT-6 Astra](https://arcprize.org/blog/astra)

**Official semi-private scores, chronological**
- March 2026 semi-private: **Gemini 3.1 Pro Preview 0.37%**, **GPT-5.4 (High) 0.26%**, **Opus 4.6 (Max) 0.25%**, **Grok-4.20 0.00%** — all frontier models below 1% — [arXiv 2603.24621](https://arxiv.org/html/2603.24621v1)
- **GPT-5.5 0.43%**, **Opus 4.7 0.18%** (semi-private) — [Analyzing GPT-5.5 & Opus 4.7](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)
- **GPT-6 Astra, Standard harness (minimal, provider-neutral): 62.7% semi-private at max reasoning effort, $26,098.** **Provider Adapter harness (provider context-management: preserved opaque reasoning state between requests + compaction): 99.9% semi-private at high reasoning effort, $18,817.** In the Provider Adapter setup Astra used fewer actions than the human baseline on **96.0% of levels** and **51.7% fewer actions per level on average**. Provider Adapter runs were ~**3.66x faster** by aggregate elapsed time and used **49% fewer total tokens** — [ARC Prize: GPT-6 Astra](https://arcprize.org/blog/astra)
- ARC Prize's own caution alongside the 99.9%: ARC-AGI-3 has "a tightly bounded scope and format" with "deterministic, closed-ended mechanics and goals", and saturating it "would not represent proof of achieving AGI" — [ARC Prize: GPT-6 Astra](https://arcprize.org/blog/astra)
- Same-model harness delta on the **public** set: **GPT-5.6 Sol scored 13.3% with the official harness and 38.3% with retained reasoning + compaction** — [OpenAI: How enabling two settings tripled our ARC-AGI-3 scores](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/) (vendor blog; author claim). Press framing of the harness question — [TNW: OpenAI's AGI number came from a harness, not the model](https://thenextweb.com/news/openai-astra-arc-agi-3-harness-62-7-vs-99-9-benchmark-revisions)
- Aggregator leaderboard (September 2026, **secondary source, set/harness unlabelled — use with caution**): GPT-6 Astra 62.7%, Claude Opus 5 30.2%, Gemini 3.8 Flash 10.4%, GPT-5.6 Sol 7.8%, Grok 4.6 2.1%, Claude Opus 4.8 1.5%, GPT-5.6 Terra 0.8%, GPT-5.5 0.4%, Gemini 3.1 Pro 0.4%, Grok 4.5 0.3%, GPT-5.4 0.2%, Claude Opus 4.7 (Adaptive) 0.2%, GPT-5.6 Luna 0.2%, GPT-6 Luna 0.1%, Grok 4.20 0.1% — [BenchLM ARC-AGI-3](https://benchlm.ai/benchmarks/arcagi3)

**Public-set harness/agent results (all 25 games, 183 levels)**
| System | RHAE | Backend | Actions | Cost | Source |
|---|---|---|---|---|---|
| NVIDIA AVO | 100.00 | Claude Opus 5 | 6,624 | n/r | [NVIDIA](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/) |
| Tycho (orchestrator) | 100.00 | Opus 5 | 6,641 | ~$2,986 est. | [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1); cost per [Retrodict](https://github.com/ryanbbrown/Retrodict) |
| Tycho (orchestrator) | 100.00 | GPT-5.6 Sol | 7,766 | — | [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1) |
| VISTA | 100.00 | Claude Opus 5.0 | 7,542 | — | [VISTA](https://vista-research.github.io/) |
| Retrodict | 99.86 | gpt-5.6-sol (max) | 7,703 | $654, 660M tokens | [Retrodict](https://github.com/ryanbbrown/Retrodict) |
| Schema | 98.98 | frontier + Claude Code | — | — | [Schema](https://schema-harness.github.io/) |
| "baseline1" best run | 98.97 | — | — | $2,722, 3,640M tokens | [Retrodict](https://github.com/ryanbbrown/Retrodict) |
| VISTA | 98.27 | GPT-5.6 Sol | 10,063 | — | [VISTA](https://vista-research.github.io/) |
| Tycho (no world model) | 79.07 | Opus 4.8 | — | — | [arXiv 2607.28287](https://arxiv.org/html/2607.28287v1) |
| Claude Code scratch snapshot (Schema's baseline) | 42.83 | Claude Code | — | — | [Schema](https://schema-harness.github.io/) |
| Bare Claude Opus 5 (no harness) | ~30 | Opus 5 | — | — | [NVIDIA](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/) |

- Human baseline for the public set: **17,135 actions** aggregate across the 183 levels — [VISTA](https://vista-research.github.io/)
- **Contamination caveat, stated by VISTA's own authors:** "the models used were released after the public ARC-AGI-3 games, so we cannot rule out that these games were seen during training; the private set remains the real test of generalization." NVIDIA independently notes its results apply to the public set only — [VISTA](https://vista-research.github.io/); [NVIDIA](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/)
- Another 2026 harness: **Arcgentica** (Symbolica AI) — orchestrator–subagent architecture where the top-level orchestrator never touches the environment and instead delegates to specialised subagents that return **compressed textual summaries** — [ARC Prize / search summary](https://arcprize.org/blog/astra) (mechanism from search snippet; no score retrieved)
- Related 2026 executable-world-model work: **Executable World Models for ARC-AGI-3 in the Era of Coding Agents** (S. Rodionov) — [arXiv 2605.05138](https://arxiv.org/pdf/2605.05138) (PDF extraction failed; no numbers)

**Failure modes ARC Prize documented before the harness era (useful diagnostic checklist)**
- **Local perception without global rules:** "models understand which action produced a change, but they fail to translate the effect into a global rule" — Opus recognised that ACTION3 rotated objects but could not apply it strategically — [ARC Prize analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)
- **Misaligned abstraction from pretraining:** mapping unfamiliar mechanics onto Tetris / Frogger / Sokoban, then wasting actions testing wrong affordances — [ARC Prize analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)
- **Victory without comprehension:** winning a level by coincidence without transferable mechanics (examples KA59, AR25) — [ARC Prize analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)
- Contrast of model styles: GPT-5.5 generated wider hypotheses but committed poorly and failed at compression; Opus "compressed its observations into a confident-but-wrong theory" — [ARC Prize analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)

**Competition constraints (relevant if targeting the prize track)**
- ARC Prize 2026 ARC-AGI-3: **$700K grand prize** for the first agent reaching 100%; **$75K** top-score awards ($40K/$15K/$10K/$5K/$5K); **$75K** milestone prizes for open-sourced solutions (deadlines 2026-06-30 and 2026-09-30). **No internet access during evaluation**; all code and methods must be open-sourced to be prize-eligible; hardware/compute limits announced at launch — [ARC Prize 2026 competition page](https://arcprize.org/competitions/2026/arc-agi-3)
- ARC Prize maintains a formal verified-testing policy governing harness conditions — [ARC Prize Verified Testing Policy](https://arcprize.org/policy)

**Comparable interactive benchmarks with 2026 adaptation results**
- **ALFWorld** (household instruction-following, immediate feedback, 50-step budget) and **SWE-bench Lite**: aTTT improves success by up to **5.0** and **4.9** points respectively — [arXiv 2607.03441](https://arxiv.org/html/2607.03441v1)
- **BabyAI** grid-world: automated prompt-optimisation framework improves LLM game-agent performance (monolithic and multi-subagent) — [Environment-Grounded Automated Prompt Optimization for LLM Game Agents, arXiv 2606.17838](https://arxiv.org/html/2606.17838)
- Broader agentic scaling and goal-directedness measurement: [Benchmark Test-Time Scaling of General LLM Agents, arXiv 2602.18998](https://arxiv.org/html/2602.18998v1); [A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents, arXiv 2602.08964](https://arxiv.org/pdf/2602.08964); [LLMs for Text-Based Exploration and Navigation Under Partial Observability, arXiv 2604.09604](https://arxiv.org/pdf/2604.09604)
- Gaming-agent evaluation infrastructure: [GamingAgent (ICLR 2026)](https://github.com/lmgame-org/GamingAgent)

### Inferences
- The ~37-point Standard-vs-Provider-Adapter gap on the *same* model, plus Schema's 42.83 → 98.98 and NVIDIA's ~30 → 100.00, means **the harness is currently worth more than the model** on this benchmark. Any comparison of ARC-AGI-3 numbers is meaningless without naming both the harness and the evaluation set.
- Converging design across four independent 100%-class systems: (1) externalised durable memory that survives compaction, (2) explicit typed transition log separating decision frames from animation frames, (3) hypothesis falsification by replay rather than by acting, (4) cost-gated model building, (5) plan-with-prediction execution that returns control only on a prediction miss. Treat that five-part list as the empirical recipe.
- Observation modality is not a differentiator: AVO used text grids, VISTA used 512x512 PNGs, and VISTA solved the same games from 1D text, 2D image and 3D render. Effort is better spent on the evidence log and the falsification loop.
- Cost spread is enormous at equal accuracy — $654 (Retrodict) vs ~$2,986 (Tycho) vs $18,817–$26,098 (Astra semi-private). On a constrained budget, the Retrodict shape (text log + code replay + playbook, one vision priming pass) is the most reproducible target.

### Gaps
- No public ARC-AGI-3 **semi-private/private** score for any of the open harnesses (Tycho, Retrodict, VISTA, AVO, Schema). Every 100%-class figure is public-set, with an explicit contamination caveat. The generalisation question is therefore open.
- ARC Prize's 2026 competition page does not publish per-submission leaderboard scores or the action-budget/reset rules; those are referenced but not stated on the page I fetched.
- Arcgentica, Schema and Executable World Models have no fully extracted per-game numbers here (site/PDF extraction limits).
- The BenchLM leaderboard does not label harness or evaluation set; its numbers should not be cited without cross-checking arcprize.org/results.
