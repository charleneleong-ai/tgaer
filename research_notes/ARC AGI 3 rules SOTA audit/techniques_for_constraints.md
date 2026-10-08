# Techniques for an ARC-AGI-3 agent under in-kernel constraints (Qwen3.6-27B-FP8, 1x RTX Pro 6000, no internet, ~110 OOD games, RHAE)

Research date: 2026-09-30. Scope: techniques that fit a Kaggle kernel with no internet, one 96 GB Blackwell GPU, open weights, several hours for ~110 unseen games, scored by RHAE.

Ranked summary. Each item gives mechanism, evidence, cost and main risk. The cited findings are in the sections below.

1. **LLM-as-programmer REPL harness (Duck-style): hypothesis, probe, revise, then search in code.** Mechanism: the observation is exposed as Python variables, the model calls `action()` from inside code, and it writes its own BFS once it understands the rules. Evidence: 1st place in Milestone 1 with Qwen 3.6 27B FP8, the same model this team has attached. 1.6002 +/- 0.4475 over 25 games x 20 passes. The team's own measurement: it clears at least one level in 23/25 games vs 7/25 for our explorer, with L1 in 55-69 actions vs 383-1708. Cost: fits one GPU at about 16 games concurrently. Risk: token throughput per game is tight (see Q4), and the harness is non-deterministic (+/-0.45 sd).
2. **Verification before commitment (exact replay of a hypothesis against observed transitions).** EWM's ablation ranks verification first. Mechanism: before spending real actions, check that the candidate rule or world model reproduces every observed frame. Evidence comes only from frontier models (GPT-5.x), so transfer to 27B is unproven. Cost: CPU-cheap, but it adds LLM calls. Risk: verification "required substantially more computational resources".
3. **Explicit EXPLORE, then VERIFY, then PLAN phase separation with action-efficiency-aware stopping (AERA).** 0.30 private RHAE (55 games) with Qwen2.5-0.5B, so it is compute-trivial. Risk: most of its public-set gains may come from "non-intelligent strategies" that the authors themselves say solve 24/25 public games. OOD transfer is shown only by the 0.30 private result.
4. **A cheap model-free prior inside the LLM loop (frame-change prediction / dead-signature pruning / state-graph hashing).** Evidence: StochasticGoose (12.58%, Preview), Blind Squirrel (6.71%), and Reki's "dead-signature" detection plus a numpy click heuristic (2nd place in Milestone 1). Cost: negligible GPU. Risk: the Duck team reports that hand-built tools "actually hurt the model". Expose these as optional helpers, not forced policy.
5. **Serving: non-thinking or short-thinking mode, prefix caching, 8-16 concurrent games, MTP speculative decoding.** This is the throughput multiplier that every design above depends on (see Q4).
6. **Agentic test-time training (per-episode LoRA via vLLM runtime LoRA).** +4.9 pts on SWE-bench Lite for Qwen3.5-27B. It "stabilizes existing competence rather than teaching new abilities", and in the paper the updates ran on separate training GPUs. Low priority on a single GPU.
7. **Not applicable (frontier API):** Tycho, Executable World Models headline numbers, Symbolica 36.08%, GPT-5.4 baselines. Internet is disabled in the kernel.

## Q1. LLM-as-programmer / code-agent approaches: what makes them bootstrap on unseen environments?

### Takeaway
The only in-kernel, open-weights approach with a published top result is the Duck: a minimal REPL where Qwen 3.6 27B FP8 writes and runs Python against the live game. It bootstraps by running discriminating probes and revising a written world model, then switching to self-written search. It does not need a pre-known goal or a winning board. Frontier-only work (EWM, Compiled Agency) consistently finds that verification and access to the environment matter more than having an executable world model as such.

### Cited Findings
- Milestone #1 (through 2026-06-30) winners were Tufa Labs "Duck", Reki and forge. The Duck converts game state into Python variables accessed through a REPL. It perceives via a rendered image, the raw ASCII grid and a segmentation tool, and uses "infinite play via eviction" (evicting the oldest messages). — [ARC Prize blog, Milestone 1](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- Duck lessons: gains came from "multimodality and better base models, not hand-built tools". Hand-crafted tools "actually hurt the model; letting it improvise worked better". It was the only winner that used code generation. — [ARC Prize blog, Milestone 1](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- 2nd (Reki) and 3rd (forge) were both **vision-LLM-as-policy** agents on Gemma-4-31B served locally. Reki used labelled image rendering, reflection memory refreshed about every 10 steps, a numpy click heuristic that prefers button-like shapes, "dead-signature" detection to stop clicking object types that never change the frame, JSON self-repair and a 1-4 action plan queue. forge's winning configuration *disabled* extra machinery, and forge found local public-game checks unreliable as a leaderboard proxy. — [ARC Prize blog, Milestone 1](https://arcprize.org/blog/arc-prize-2026-milestone-1). (Contradicts the claim in our memory that "LLM-as-policy is the configuration nobody wins with". LLM-as-policy took 2nd and 3rd, just below the programmer approach.)
- Duck: mean 1.6002 +/- 0.4475 on 25 public games with 20 tries each. It is "an order of magnitude cheaper on each game" than a GPT 5.4 approach. Some games reach >40% of levels, others no first level. — [Tufa Labs Duck write-up](https://tufalabs.ai/research/duck-harness/). The write-up gives no component ablations, token budgets, vLLM settings or concurrency. — same source
- Team's own analysis of the Duck repo's `example-run/` (2026-09-30): it clears at least one level in 23/25 games vs 7/25 for our explorer. L1 actions: m0r0 67, s5i5 56, sp80 55, tu93 69 (vs 1708/1401/458/383 for our explorer). The prompt tells the model to "program a discriminating probe and revise the world model" when confidence is low, and to "stop probing and search" once the rules are understood. It runs 16 games concurrently on one vLLM. — internal memory `project_arcagi3_duck_bootstraps_by_probing.md`; source repo [Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness)
- Executable World Models (Rodionov, 2026-07-16, rev. 2026-08-27) ablates executable world model vs simplification prompts vs exact replay verification on GPT-5.4/5.5/5.6-sol. Verification ranked highest overall but "required substantially more computational resources". Simplification beat plain executable in 3 of 4 settings. Textual variants outperformed executable-only in some configurations. "Model capability increases yielded greater gains than architectural differences." — [arXiv 2607.15439](https://arxiv.org/abs/2607.15439). Frontier API only, so not directly applicable.
- Compiled Agency (Jul/Sep 2026): coding agents build standalone game players under a develop-freeze-evaluate protocol, and the frozen program runs with zero model calls. Environment access gives "10 to 78 percentage points of held-out success over construction-only controls". Refreshing the validation panel mid-session improved shipped success by 12.4 points in 9/9 pairs. The models used were not stated in the abstract. — [arXiv 2609.18996](https://arxiv.org/abs/2609.18996)
- WorldCoder (NeurIPS 2024) represents dynamics and reward as Python functions refined through a synthesize-repair loop, guided by transition consistency plus an optimism constraint. — [Awesome-Code-World-Models](https://github.com/lxycopper/Awesome-Code-World-Models)
- AutumnBench (2025-10-22): 517 humans vs Claude 4 Sonnet, Gemini 2.5 Pro and o3 on 43 environments / 129 tasks. Humans (80th percentile about 0.935) beat all models. Humans used resets about 12.5% and no-ops about 12.5% of actions as "experimental tools to test hypotheses", while Claude spent 98.6% of actions on clicks and moves. More compute helped in only 25 of 43 environments. — [arXiv 2510.19788](https://arxiv.org/html/2510.19788v1)
- "Agents Explore but Agents Ignore": GPT-OSS-120B discovers the relevant documentation in 97.54% of runs but calls the solution API in 0.53%. LLM agents fail to act on what they discover. — [arXiv 2604.17609](https://arxiv.org/abs/2604.17609)
- PatchWorld (2026-07-29): gradient-free optimisation of executable world models with open-weight models. Numbers could not be extracted from the PDF. — [arXiv 2605.30880](https://arxiv.org/pdf/2605.30880)

### Inferences
- The things that make these agents bootstrap are (a) experiments run *in context* against the live environment, (b) a written hypothesis that is revised on contradiction, and (c) handing off to programmatic search once the rules are known. A one-shot "write a heuristic/distance function" does not include (a), which matches this team's failed phase-1 test (2/11 games usable).
- The AutumnBench and "explore but ignore" results predict the typical 27B failure mode: probing without exploiting what it learns, and under-using reset. A harness prompt that explicitly legitimises reset-as-experiment and "stop probing, now search" addresses both.
- Because the Duck uses the same model this team has attached, porting it is the lowest-risk way to get a large gain. Its 23/25 vs 7/25 coverage gap dwarfs any explorer-constant tuning.

### Gaps
- No published component ablation (eviction, multimodality, helpers, thinking mode) exists for the Duck. The effect of each piece is unknown.
- No verification-first study on an open ~27B model was found. EWM's ranking is frontier-only.
- The Duck's hidden-set score and the Reki and forge numeric scores were not in the sources fetched.

## Q2. Test-time adaptation, in-context RL and TTT/LoRA inside a time budget

### Takeaway
Per-episode LoRA TTT gives single-digit gains, and only where the model is already competent. It needs asynchronous training compute (a separate GPU in the paper), so on one shared GPU it competes with inference. In-context adaptation (the REPL's own revise loop) is the cheaper form of "test-time adaptation" here. The in-kernel precedent for weight-level learning is a small value network retrained online (Blind Squirrel), not LoRA on the LLM.

### Cited Findings
- Agentic TTT (aTTT, 2026-07): episode-specific LoRA (rank 8, alpha 16, lr 5e-4, 2 gradient steps every K=5 steps) trained on the agent's recent tokens, observations or a summary, with repetition-aware token reweighting. Results: ALFWorld Qwen3.5-9B 50.7 -> 55.7. Qwen3.5-27B gained little on 50-step ALFWorld but went 57.8 -> 62.7 on SWE-bench Lite. — [arXiv 2607.03441](https://arxiv.org/html/2607.03441v1)
- aTTT costs 1.9x wall-clock vs no-TTT on ALFWorld. It served 16 concurrent episodes through the vLLM runtime LoRA API, with "dedicated training GPUs" computing updates asynchronously and hot-swapping adapters in <100 ms. Unfiltered online TTT degrades performance. aTTT "stabilizes existing competence rather than teaching new abilities". Static pre-rollout adaptation gave minimal gains. The same summary text injected in-context scored 49.3 vs 54.3 with LoRA. — [arXiv 2607.03441](https://arxiv.org/html/2607.03441v1)
- Blind Squirrel (Preview 2nd, 6.71%, 13 levels) builds state graphs from frames. When progress happens it "back-labels that level with distances and retrains a small ResNet18-based value model". — [ARC Prize Preview 30-day learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- Other 2025-26 TTA work: syntactic alignment plus dynamics grounding from deployment-time interaction ([arXiv 2511.04847](https://arxiv.org/abs/2511.04847)) and CANOPY outcome-only RL ([arXiv 2609.01245](https://arxiv.org/abs/2609.01245v1)). Their numbers were not extracted.

### Inferences
- On one GPU that is fully used for 16-way inference, LoRA TTT roughly halves throughput (1.9x wall-clock) for a gain of about 5 points on tasks the model can already do. ARC-AGI-3's bottleneck is bootstrapping, i.e. no competence yet, which is exactly where aTTT does not help. Rank it low.
- A small CNN or ResNet value or effect model trained on-the-fly (StochasticGoose, Blind Squirrel) is the cheap weight-level adaptation. The team's own memory, though, records that effect prediction is closed under RHAE and that the value model is offline-underpowered.

### Gaps
- No study was found that applies LoRA TTT to grid games or ARC-AGI-3 with a single shared GPU.

## Q3. Action-efficient exploration and the exploration-vs-RHAE trade-off

### Takeaway
RHAE is min(1.15, human/agent actions)^2 per level, with linearly increasing level weights. Wasted exploration on a level is therefore penalised quadratically, but a cleared level is worth far more than an efficient failure. The best evidence favours a short, deliberate probe phase (hypothesis-driven, including resets) followed by planned execution. Cheap learned or heuristic priors (frame-change prediction, dead-signature pruning, state hashing) mainly trim the probe phase.

### Cited Findings
- RHAE: S = min(1.15, h/a)^2 per level, where h is the upper-median best human action count. Levels are weighted linearly (level 5 of 5 counts 5/15). An environment's score is capped at the weighted fraction of completed levels. Frontier LLMs scored <=0.50% in March 2026. — [ARC-AGI-3 technical report, arXiv 2603.24621](https://arxiv.org/html/2603.24621v2)
- StochasticGoose (Preview 1st, 12.58%, 18 levels): a 4-layer CNN on 64x64 frames predicts which actions change the frame, with a 64x64 click-coordinate head, a 200K deduplicated state-action buffer and BCE loss. — [DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution); [tech report](https://arxiv.org/html/2603.24621v2). It cut initial wasted actions from about 350 to focused exploitation. — [Preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- In the Preview, winners separated exploratory from execution actions. LLM entries (Fluxonian 8.04%, Play Zero 4.37%, Tomas Engine 3.70%) "struggled with efficiency metrics". — [Preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- AERA (2026-05-25): EXPLORE -> VERIFY -> PLAN with Qwen2.5-0.5B scored public RHAE 0.2116 (4/25 solved) and private 0.30 (55 games). Baselines scored 0.0000. It frames RHAE's quadratic as a second-order penalty for leaving the Pareto frontier between action efficiency and information gain. The authors also report that 24/25 public games are solvable by "non-intelligent strategies" and that a null-coordinate vulnerability bypasses 18 games in 1 step. — [arXiv 2605.25931](https://arxiv.org/abs/2605.25931)
- Humans on AutumnBench used about 12.5% resets as hypothesis tests. — [arXiv 2510.19788](https://arxiv.org/html/2510.19788v1)

### Inferences
- The Preview metric (levels completed) and the current metric (RHAE) reward different things. StochasticGoose's result is evidence about *coverage*, and its RHAE value is untested.
- Given the level-cap rule, per-level speed only matters for levels you clear. On OOD games, coverage (clearing L1 at all) comes first and efficiency second. That matches the Duck's advantage: 23/25 coverage plus L1 in about 55-69 actions.
- The AERA null-coordinate exploit is a public-set artefact. Do not rely on it for the 110 private games (and it may be patched).

### Gaps
- Whether a per-level action cap exists in the Kaggle harness was not verified in these sources.

## Q4. Serving a 27B FP8 model for many concurrent games on one RTX Pro 6000

### Takeaway
Expect roughly 40-46 tok/s single-stream and about 90-190 tok/s aggregate at low concurrency without speculative decoding. MTP speculative decoding and prefix caching can raise this substantially. Over a 6-7 h run shared across 110 games, the budget is on the order of tens of thousands of output tokens per game. That argues for non-thinking or short-thinking mode, code that batches many actions per LLM call, short evicted contexts that keep the prefix cacheable, and 8-16 concurrent games.

### Cited Findings
- Millstone AI (2026-05-27), Qwen3.6-27B FP8 on 1x RTX Pro 6000 Blackwell with vLLM, prefix caching and speculative decoding **disabled**, 1,024 output tokens. Single request: 46.1 tok/s at 1K context, 43 at 32K, 30.4 at 256K. Aggregate with 5 concurrent: 189.3 tok/s at 1K, 92.5 at 32K, 23.4 at 128K. TTFT is 170 ms at 1K, 8.2 s at 32K with 3 requests, and 70 s at 256K. Only concurrency 1-5 was tested. — [Millstone AI benchmark](https://www.millstoneai.com/inference-benchmark/qwen3-6-27b-fp8-1x-rtx-pro-6000-blackwell)
- lastloop-ai guide: Qwen3.6-27B (**INT4 AutoRound**, not FP8) on RTX PRO 6000 with vLLM 0.19.2 nightly, flashinfer, FP8 KV cache, `--max-num-seqs 8`, MTP with 3 speculative tokens. About 100 tok/s single-stream, mean acceptance length 3.19, per-position acceptance 0.87/0.72/0.60. FP8 KV is 15% slower at short context. Blackwell sm_120 needs the CUDA 13 toolkit for flashinfer JIT. Stable 0.19.1 had an MTP bug. — [vllm-blackwell-guide](https://github.com/lastloop-ai/vllm-blackwell-guide)
- A Max-Q user with vLLM 0.27.1 measured 46.8 tok/s single without speculative decoding and 62.2 tok/s with 2-token MTP. — search snippet via [loFT LLC](https://loftllc.dev/en/docs/tech/llm-research/qwen3-6-27b-nvfp4-mtp-vllm-benchmark/) (not fetched directly; treat as secondary).
- The team's kernel installs vLLM 0.19.0 from a pinned wheelhouse (vs 0.26 on the dev A100). The chat template gates on `enable_thinking`. — internal memory `project_arcagi3_kaggle_stack.md`
- The Duck runs 16 games concurrently on one local vLLM at about 75 min per game. — internal memory `project_arcagi3_duck_bootstraps_by_probing.md` (from the repo)

### Inferences
- Rough budget: about 92 tok/s aggregate at 32K context (the conservative Millstone number) x 6.5 h is about 2.2M output tokens, or about 20K output tokens per game for 110 games. With short contexts (1-8K via eviction) and higher concurrency, aggregate plausibly rises several-fold, but that is not benchmarked above concurrency 5. Actual throughput should be measured in-kernel.
- Long reasoning traces are unaffordable at this budget: one 8K-token thinking turn is about 40% of a game's allowance. Prefer `enable_thinking=False` or a tight thinking budget, and let code do the loops (one call -> dozens of actions).
- Eviction keeps context short, which both preserves decode speed (46 vs 30 tok/s) and cuts TTFT (170 ms vs tens of seconds). A stable system-prompt prefix makes prefix caching pay off across 16 concurrent games.
- MTP on vLLM 0.19.0 is risky: the guide reports an MTP bug in 0.19.1 stable. Verify in-kernel before relying on it.

### Gaps
- No benchmark covers Qwen3.6-27B-FP8 at 8-32 concurrency with prefix caching on RTX Pro 6000.
- The kernel runtime limit is not in official docs. The team assumes 7.5 h.

## Q5. Hybrid designs: model-free explorer plus LLM

### Takeaway
The evidence is mixed but leans toward LLM-led with light model-free priors: the winners used LLMs with small heuristic aids, not an explorer that hands off to an LLM. Hand-built tooling hurt the Duck, while Reki's cheap heuristics (click prior, dead-signature pruning) were part of a 2nd-place system. No published ablation isolates the hybrid's contribution on ARC-AGI-3.

### Cited Findings
- Reki (2nd, Milestone 1): Gemma-4-31B policy plus a numpy click heuristic plus "dead-signature" pruning of object types that never change the frame. Environment-variable toggles were built in for ablation, but the numbers were not in the blog. — [ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- Duck: hand-crafted tools "actually hurt the model". — [ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- In the Preview (levels metric), non-LLM directed exploration (StochasticGoose 12.58%, Blind Squirrel 6.71%) beat LLM entries (<=8.04%) and frontier LRMs. — [Preview learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings); [tech report](https://arxiv.org/html/2603.24621v2)
- Internal: our model-free explorer clears 2/25 in the scored kernel. One team that ran both lines measured 0.11 (model-free) vs 1.70 (LLM) on the hidden set. — internal memory `project_arcagi3_llm_as_programmer.md` (secondary; original source not re-verified here)

### Inferences
- The best-supported hybrid gives the LLM REPL the explorer's cheap machinery (state hashing and graph, BFS, inert-object detection) as *optional library functions* it may call, rather than an explorer-first pipeline. This keeps the Duck's "let the model improvise" property while giving it efficient search primitives once rules are inferred.
- A second viable hybrid is to run the explorer on games where the LLM stalls (a timeout fallback), since the explorer is GPU-free and could run in CPU slack. This is untested.

### Gaps
- No ARC-AGI-3 ablation of LLM with vs without explorer primitives was found.
- The numeric private scores of Reki and forge were not found.
