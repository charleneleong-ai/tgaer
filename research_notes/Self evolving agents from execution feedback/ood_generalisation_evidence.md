# Which Self-Improving / Adaptive Agent Methods Actually Generalise Out-of-Distribution

Scope note: all numbers below are author-reported unless flagged otherwise. None of the 2026 results in this note has a published independent reproduction that I could find. Older (pre-2025) results are labelled.

---

## Q1. Quantitative evidence for in-distribution gains shrinking or vanishing out-of-distribution

### Takeaway
The single cleanest paired dataset is RRSI's ablation table: unregularized harness self-improvement produces the **largest** development-split gain (+3.4 pts) and an out-of-distribution gain of **+0.6 pts, described by the authors as "nearly indistinguishable" from the un-evolved baseline** — i.e. the exact dev-gain/zero-OOD-gain pattern. Across RRSI, EvoAgentBench and SkillOpt, the reported shrinkage factor from dev to OOD is roughly 3x in the best case and sign-flipping negative in the worst; two published methods (TTHE, Memento) show *negative* OOD transfer while gaining in-distribution.

### Cited Findings

**RRSI (Google, arXiv 2609.24972, submitted 2026-09-25) — the headline framing**
- The paper's own problem statement: recursive harness evolution "may overfit by memorizing the training tasks, showing large in-distribution gains that shrink or even vanish on out-of-distribution benchmarks" — [arXiv 2609.24972](https://arxiv.org/abs/2609.24972)
- Headline paired result: "up to 14.1 points on the split it evolves against and up to 4.7 points on the five out-of-distribution benchmarks", while producing a harness that runs on **30% fewer policy tokens** than unregularized evolution — [arXiv 2609.24972](https://arxiv.org/abs/2609.24972)
- Eight benchmarks over three domains. Evolve splits: **Terminal-Bench 2.1** (coding), **Harvey LAB** (agentic workspace), **EngDesign** (engineering design). Held-out: **SWE-bench Verified**, Harvey LAB in-distribution held-out split, **JobBench**, **GDPval**, **APEX-Agents**, **Frontier-Eng** — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)

**RRSI Table 1 (agentic workspace) — per-method paired numbers**

| Method | Harvey LAB (evolve) | Harvey LAB (ID held-out) | JobBench (OOD) | GDPval (OOD) | APEX-Agents (OOD) |
|---|---|---|---|---|---|
| H₀ baseline | 89.4 | 86.9 | 36.0 | 48.8 | 34.2 |
| Meta-Harness | 93.0 (+3.6) | 89.2 | 37.1 (+1.1) | 49.1 (+0.3) | 35.7 (+1.5) |
| AHE | 90.7 (+1.3) | 88.7 | 37.2 (+1.2) | 47.2 (**−1.6**) | 33.1 (**−1.1**) |
| TTHE | 91.1 (+1.7) | 88.5 | 35.2 (**−0.8**) | 47.0 (**−1.8**) | 31.7 (**−2.5**) |
| HarnessX | 91.8 (+2.4) | 89.1 | 36.3 (+0.3) | 48.5 (**−0.3**) | 34.3 (+0.1) |
| RRSI | 90.5 (+1.1) | 89.2 | 40.7 (+4.7) | 52.3 (+3.5) | 37.9 (+3.7) |

Source: [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1). Read the ordering carefully — **the method with the largest evolve-set gain (Meta-Harness, +3.6) delivers +0.3 to +1.5 OOD; the method with the smallest evolve-set gain (RRSI, +1.1) delivers +3.5 to +4.7 OOD.** Within this table dev gain and OOD gain are *anti*-correlated. Two methods (AHE, TTHE) are net-negative OOD while positive in-distribution.

- Coding domain: Terminal-Bench 2.1 evolve **74.2 → 80.2 (+6.0)**; SWE-bench Verified OOD **82.0 → 83.8 (+1.8)**. Shrinkage ≈ 3.3x — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Engineering domain: EngDesign evolve **+4.9**; Frontier-Eng OOD **+4.3 Medal points (24.3% relative)** — the best-transferring domain in the paper — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- RRSI posts "the smallest evolve-set gain of any evolved harness" (90.5 vs 93.0 for Meta-Harness) while being "the only out-of-distribution average that clears H₀ by more than a point" — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)

**RRSI Table 2 (ablation) — the decisive paired table**

| Variant | Harvey LAB (evolve) | Harvey LAB (ID held-out) | OOD avg | Tokens/trial (M) |
|---|---|---|---|---|
| H₀ (no evolution) | 89.4 | 86.9 | 39.7 | 1.56 |
| **Unregularized evolution** | **92.8 (+3.4, best dev)** | 88.9 | **40.3 (+0.6)** | 3.80 |
| w/o proposal regularizers | 90.7 | 88.8 | 41.9 (+2.2) | 2.69 |
| w/o acceptance regularizers | 91.5 | 88.7 | 41.0 (+1.3) | 3.59 |
| RRSI (full) | 90.5 (+1.1, worst dev) | 89.2 | **43.6 (+3.9, best OOD)** | 2.42 |

Source: [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1). Authors' reading: with both regularizer families removed, "evolve peaks at 92.8, OOD collapses to 40.3 (nearly indistinguishable from H₀)". Unregularized evolution also **2.4x-es token cost** (1.56M → 3.80M) for that ~zero OOD gain.

**SkillOpt vs SkillBoost (arXiv 2607.26643) — measured generalization gaps**
- "Skill overfitting" defined as agents that "gain accuracy on short-term observations while losing robustness to later tasks from different distributions", caused by optimizing loss only on the current batch of limited trajectories — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)
- SkillOpt baseline generalization gap (Test − Train): SpreadsheetBench **−4.6% to −12.4%**; LiveMath **−4.7% to −26.7%**; BFCL-v4 **−3.1% to −9.0%** — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)
- SkillBoost (regularized) gap: SpreadsheetBench **−0.9% to +2.5%**; LiveMath **−1.1% to +1.7%**; BFCL-v4 **−0.7% to +0.8%** — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)

**EvoAgentBench (arXiv 2607.05202) — transfer measured by construction**
- 528 training / 267 test tasks, instance-disjoint but sharing "reusable procedural capabilities"; domains BrowseComp-Plus, LiveCodeBench, SWE-Bench Verified, GDPVal — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)
- **Memento** (raw case retrieval): **−2.4%** average on Qwen3.5-27B, +1.5% on Qwen3.5-397B, with a **−36.3%** catastrophic cell — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)
- **ReasoningBank** (distilled reasoning memory): +0.4% to +3.6% average, but **six negative per-domain cells** despite positive averages — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)
- **GEPA** (evolved global prompt): +1.2% / +3.5% / +5.7% average across three backbones, including a **−12.3%** cell — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)
- **Anchor Skill** (curator-side reference procedures with *oracle* routing): +7.5% / +10.5% / +5.8%, **positive in all 24 method–domain–setting cells** — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)
- Authors' conclusion: "reusable procedural content transfers reliably across model families" when correctly delivered, but "automatic methods remain brittle" at extracting and routing it; the gap "implicates method-side mechanisms rather than task difficulty or evaluation variance" — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)

**Counter-evidence: methods that DO transfer**
- Darwin Gödel Machine (arXiv 2505.22954, 2025): agent evolved on SWE-bench scored **28.9% on held-out Polyglot vs 14.2% baseline**; agent evolved on Polyglot scored **24.5% on held-out SWE-bench vs 20.0% baseline**. Neither ever accessed the alternate benchmark — [arXiv 2505.22954](https://arxiv.org/abs/2505.22954)
- DGM cross-backbone transfer: on SWE-bench, swapping Claude 3.5 Sonnet → o3-mini gave base 23.0% vs DGM 33.0%; → Claude 3.7 Sonnet gave base 19.0% vs DGM 59.5% — [arXiv 2505.22954](https://arxiv.org/abs/2505.22954)
- AIDE² / "Recursive self-improvement of AI research agents" (arXiv 2609.26457, 2026-09-23): selection-benchmark grade **0.703 → 0.778**; **7 rewrites accepted out of a 100-node trajectory** (~7%, at steps 2, 6, 28, 39, 47, 63, 85; two other runs accepted 2 and 4). AIDE₈₅ "matches or exceeds AIDEhuman on all four external benchmarks" (ALE-Bench, MLE-Bench, FML-Bench, WeatherBench 2), and "some of the largest performance gains appear on the out-of-distribution benchmark" (WeatherBench 2) — [arXiv 2609.26457](https://arxiv.org/html/2609.26457)
- SkillBoost cross-benchmark transfer: LiveMath-optimized skill applied to **OlympiadBench**: DeepSeek-v4-pro 44.0% → 50.0% (+6.0pp); Kimi-k2.6 46.0% → 60.0% (+14.0pp). Human-written baseline skills gave 0–3pp on the same transfer tasks — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)
- SkillBoost cross-model transfer (skills optimized on Claude-opus-4-6, applied on DeepSeek): BFCL-v4 **+14.3pp** over human baseline, LiveMath **+6.4pp**; 0.7–14.3pp across four benchmarks — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)

### Inferences
- A 2.25x dev-set improvement with exactly zero OOD movement sits **inside the published distribution of outcomes, not outside it**. RRSI's unregularized row (+3.4 dev, +0.6 OOD) and TTHE (+1.7 dev, −0.8/−1.8/−2.5 OOD) are the same phenomenon.
- Where transfer *is* reported (DGM, AIDE², SkillBoost), the ratio of OOD gain to dev gain is roughly 0.3–1.0 and sometimes >1. Where it fails, the ratio is ~0 to negative. There appears to be no middle regime of "small but reliable transfer" — the outcome is closer to bimodal, which matters because a dev gain gives almost no prior on which mode you are in.
- The transferring cases share a structural feature: selection was made against a signal **decoupled** from the inner optimisation loop (AIDE²), or across a **domain/backbone boundary** (DGM, SkillBoost). The failing cases selected on the same signal they optimised.

### Gaps
- I could not recover exact per-benchmark deltas for AIDE² on its four external benchmarks — the paper reports "matches or exceeds" qualitatively rather than as a paired table. This is the weakest of the "transfer works" claims in numerical terms.
- No paper I found reports a **correlation coefficient** between dev-set gain and OOD gain across many accepted edits. RRSI has the raw material for this and does not compute it (see Q5).
- No independent reproduction exists for any 2026 result cited here. RRSI, AIDE², EvoAgentBench and SkillBoost are all author-reported.

---

## Q2. Regularisation and selection disciplines reported to IMPROVE transfer

### Takeaway
Five mechanisms have paired numbers attached in 2026 work: a **noise-calibrated acceptance floor**, **non-compensatory (multi-criterion, non-tradeable) acceptance**, an **annealed edit budget**, a **leakage critic** on the diff, and **pruning of components that stopped paying**. RRSI's ablation shows acceptance-side and proposal-side constraints each contribute independently (−2.6 and −1.7 OOD points when removed), and the compression literature adds a sixth: **constrain the description length of the learned artifact, and constrain the bandwidth of the dev-set feedback channel**.

### Cited Findings

**Noise-adjusted acceptance floor (RRSI)**
- Acceptance requires the noise-adjusted floor **Ŝ(H′) ≥ S★ − δ**, where δ is empirically calibrated before evolution by "evaluat[ing] the unchanged base harness repeatedly" — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Calibrated δ values: coding domain δ ≈ **3 passes out of 178 trials** (~1.7%); agentic workspace δ ≈ **60 criteria out of ~14,100 criterion verdicts** (~0.43%) — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- **No statistical hypothesis testing is used**; decisions rest on the empirical noise band, not a significance test — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)

**Non-compensatory acceptance (RRSI)**
- "A candidate must satisfy several non-compensatory criteria before its score can justify replacing the incumbent" — i.e. a validity regression cannot be bought off with a score gain — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Concrete engineering-design guards: reject if valid-output rate falls by **>0.03** or no-submission rate rises **>0.02** — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)

**Edit budget annealing (RRSI)**
- The proposer operates with a "temporally annealed budget, limiting how many edits a candidate can bundle"; the budget decreases from b_max to b_min over rounds on a **cosine schedule**, creating soft pressure toward sparsity in later rounds — [arXiv 2609.24972](https://arxiv.org/abs/2609.24972); [HTML](https://arxiv.org/html/2609.24972v1)

**Leakage critic (RRSI)**
- The critic inspects candidate **diffs before full evaluation** and rejects edits containing "task names, entity names, task-specific values, answers, or other logic specific to the evolve benchmark", plus inert machinery additions — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- "Generic prompt or tool-description improvements remain valid candidates" — the screen is on benchmark-specific *content*, not on component type — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)

**Pruner (RRSI)**
- The pruner "removes changes that are too small, too expensive, or no longer useful"; components with no strictly positive gain over a fixed pruning window (n_prune rounds) are marked for removal — [arXiv 2609.24972](https://arxiv.org/abs/2609.24972); [HTML](https://arxiv.org/html/2609.24972v1)

**Novelty / exploration bonus (RRSI)**
- A novelty bonus rewards candidates that touch **structural** components (skills, memory, subagents) "that have never previously appeared in a winning edit" — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Authors' conclusion on ablation: proposal regularization matters because "steering where the search looks matters even when nothing is rejected" (removing it costs 1.7 OOD points *without* changing the evolve score much: 90.7 vs 90.5) — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Collective aim: the constraints "favor reusable agent mechanisms over benchmark-specific ones or even noises" — [arXiv 2609.24972](https://arxiv.org/abs/2609.24972)

**Verified acceptance with explicit regression cap (SkillBoost)**
- Accept a candidate skill edit only when **Repair(s′) > 0 AND Regression(s′) < ε** — new edits must fix more cases than they break — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)
- SkillBoost's three stages: structured exploitation (failure attribution to skill components via workflow-compliance checks, reasoning-chain reconstruction, root-cause clustering), prior-guided exploration (N candidate repairs of varying scope), verified acceptance — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)
- Effect size: collapses the Test−Train gap from as bad as −26.7% to within ±2.5% on three benchmarks (numbers in Q1) — [arXiv 2607.26643](https://arxiv.org/html/2607.26643v2)

**Description-length / bandwidth regularisation (arXiv 2606.11045, Bertran, Roth, Wu)**
- Central claim: "what fits into few tokens doesn't overfit" — compressing an agent's discovery into a small token budget is itself the generalisation guarantee — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)
- Output compression: **32-token reproducers match or exceed the unconstrained explorer on 38 of 41 improvement checkpoints (92.7%)** under a one-sided 5% relative-gap criterion, with validation tracking holdout across all 8 datasets — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)
- Input compression ("ladder"): replacing the numerical validation score with **a single bit of feedback** still "matches or slightly exceeds the score-based explorer on holdout performance on all 8 datasets" — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)
- Overfitting *detection* via compression: when agents were deliberately steered toward exploitation, compressed prompts identified validation-specific gains with **100% sensitivity and 91% specificity**; otherwise validation–holdout gaps were **0.0–1.9 pp**, within confidence bounds — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)
- Simultaneous confidence-interval width ε_j = O(√((j·log T_max + log(1/δ_j))/n)); under K_max=7, T_max=50, n=5,000, δ=0.05, the bound widens from **~2.4pp at the first checkpoint to ~4.7pp at the final checkpoint** — the price of adaptive reuse of the dev set — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)

**Decoupled inner/outer selection signals (AIDE², arXiv 2609.26457)**
- Design explicitly "decouples the inner-loop optimization signals from the outer-loop selection signals" to "prevent the inner-loop agent from directly optimizing the criterion used for outer-loop selection" — [arXiv 2609.26457](https://arxiv.org/html/2609.26457)
- The authors state the discipline's own limit: "the private grades used during recursive self-improvement cannot by themselves establish generalization beyond candidate selection", motivating a separate "second-order generalization" test on four external benchmarks — [arXiv 2609.26457](https://arxiv.org/html/2609.26457)

**Held-out slices and replay (weaker / secondary sources)**
- SEAGym "tracks overfitting by replaying previous checkpoints on held-out tasks" — search-result summary, not verified against the primary paper (flagged low confidence) — [search summary, Self-Improvements survey listing](https://arxiv.org/pdf/2607.13104)
- Practitioner-level: a held-out test slice catches rubric-gaming "when the gap between validation and test performance exceeds a few points", and without one the overfitting "remains undetected until production finds it" — [Future AGI, automatic prompt optimization 2026](https://futureagi.com/blog/automatic-prompt-optimization/) (vendor blog — treat as opinion, not evidence)
- CMVF (arXiv 2607.24354) reports the smallest average validation-to-test gap among compared prompt optimizers, attributed to aggregating over *recurring* patterns rather than isolated validation errors — described as a form of implicit regularisation. Snippet-level only; I did not fetch the paper — [arXiv 2607.24354](https://arxiv.org/pdf/2607.24354)

### Inferences
- The two independently-effective axes in RRSI map cleanly onto two different failure modes: **acceptance-side** constraints stop you from banking noise (−2.6 OOD when removed), **proposal-side** constraints stop you from ever generating benchmark-specific candidates in the first place (−1.7 OOD when removed). A noise floor alone is not sufficient; you also have to bias *what gets proposed*.
- The compression result is the most directly actionable: it says the discriminator between a real improvement and a dev-set artifact is **whether the change can be stated in few tokens**, and that a 32-token summary loses essentially nothing when the gain is real. An adaptive mechanism that needs a large, per-case, enumerated body to express is prima facie a dev-set fit.
- The one-bit-feedback result is the strongest argument for throttling how much of the dev-set score you let the optimiser see. If a single bit of "better/worse" reaches the same holdout performance, then feeding the full 25-game score vector is pure overfitting capacity with no measured upside.
- RRSI's δ calibration protocol is directly transplantable: evaluate the *unchanged* baseline repeatedly, take the empirical spread as δ, and refuse any edit that does not clear incumbent − δ. For a 25-environment dev set this floor will be wide, which is itself the finding.

### Gaps
- No paper isolates the **leakage critic alone** in an ablation. RRSI's Table 2 lumps it with all other acceptance-side constraints, so the critic's individual contribution to the +3.9 OOD gain is unmeasured.
- No reported method uses **cross-validation across tasks** (leave-one-task-out selection) as its acceptance rule in the 2026 agent literature that I found. The nearest analogues are AIDE²'s decoupled private grade and DGM's cross-benchmark check — both post-hoc validation, not a selection criterion. This looks like a genuine hole in the literature rather than a negative result.
- I could not extract the numerical magnitude of overtuning from the HPO paper (see Q5) — its PDF resisted text extraction.

---

## Q3. Benchmark design that deliberately resists overfitting

### Takeaway
ARC-AGI-3 is the clearest published case of **out-of-distribution-by-construction**: the private environments are stated to be "intentionally out-of-distribution relative to the public set" with "limited overlap" in game mechanics, the public set is explicitly demoted from a training resource to a demonstration interface, and **the official leaderboard will never report public-set scores for any system** because the public set is materially easier. Under that design, a public-set gain carries no designed-in claim on the private set at all.

### Cited Findings

**ARC-AGI-3 split and rationale (arXiv 2603.24621 / arcprize.org)**
- Split is **25 public demo / 55 semi-private / 55 fully private** environments. Semi-private is for testing frontier models behind an external API (accepting "a small risk of data leakage"); fully private is for the official ARC Prize competition and "tightly guarded" — [arXiv 2603.24621 (HTML)](https://arxiv.org/html/2603.24621v1); [ARC-AGI-3 Technical Report PDF](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- ARC-AGI-3 **inverts** ARC-AGI-2's "roughly 10:1 public-to-private ratio". The public set shifts "from a training resource to a demonstration interface, while the private set becomes the primary basis for evaluation" — [arXiv 2603.24621 (HTML)](https://arxiv.org/html/2603.24621v1)
- Public environments are "intentionally easier for both humans and AI, with a stronger emphasis on clarity and fun"; private sets are "significantly more difficult" and designed to "more rigorously test generalization" — [arXiv 2603.24621 (HTML)](https://arxiv.org/html/2603.24621v1)
- **OOD by construction, stated explicitly**: private environments are "intentionally out-of-distribution relative to the public set" and cover "a broader and more diverse set of mechanics with limited overlap with the mechanics found in the public environments" — [arXiv 2603.24621 (HTML)](https://arxiv.org/html/2603.24621v1)
- "Because the public set is materially easier than the private set, the official leaderboard will never report public set scores of any system" — [search summary of ARC-AGI-3 materials](https://arxiv.org/html/2603.24621v1)
- The official harness is "intentionally generic, without tools or special features", a design choice to prevent domain-specific overfitting so leaderboard evaluation measures generalisation rather than harness-specific performance — [ARC-AGI-3 Technical Report PDF](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- Evidence of within-public instability: early harness testing showed "extreme bimodal performance", with one model scoring **97.1% on one public environment and 0.0% on a different public environment using the same harness** — [arXiv 2603.24621 (HTML)](https://arxiv.org/html/2603.24621v1)
- Scoring metric **RHAE** (Relative Human Action Efficiency): per level **S_{l,e} = min(1.0, h_{l,e} / a_{l,e})²**, where h is the *second-best* human action count and a is AI actions taken; environment score is a linearly weighted average across levels with level weights increasing 1…n — [arXiv 2603.24621 (HTML)](https://arxiv.org/html/2603.24621v1)

**ARC Prize 2025 technical report (arXiv 2601.10904)**
- "State-of-the-art scores are only reported on the Semi-Private and Private Evaluation task sets to reduce the risk of overfitting and data contamination" — [arXiv 2601.10904 (HTML)](https://arxiv.org/html/2601.10904v1)
- ARC-AGI-2 composition for contrast: 400 public training tasks (imported from ARC-AGI-1), 120 semi-private, 120 private — [arXiv 2601.10904 (HTML)](https://arxiv.org/html/2601.10904v1)
- New contamination channel identified: frontier models produce **correct ARC colour mappings in their reasoning without explicit mention in training**, suggesting "ARC data are well-represented in the underlying model". Authors' verdict: "we assess that this new form of 'overfitting' assists models in solving ARC", while acknowledging they "cannot precisely quantify the magnitude of this effect" — [arXiv 2601.10904 (HTML)](https://arxiv.org/html/2601.10904v1)
- Competition scale: 1,455 teams, 15,154 entries. **No public-vs-private ranking-divergence statistics are reported** — [arXiv 2601.10904 (HTML)](https://arxiv.org/html/2601.10904v1)

**Public-set saturation claims on ARC-AGI-3 (all public-only, none with a paired private number)**
- Schema harness reports **~99% on ARC-AGI-3 Public** with Claude Opus 4.8 and Fable 5, and **95.35%** with GPT-5.6 Sol — [schema-harness.github.io](https://schema-harness.github.io/)
- An open-source harness reports **100% on all public environments using human replay** — [search result, ARC-AGI-3 harness listings](https://schema-harness.github.io/)
- OpenAI reports GPT-5.6 Sol at **13.3% on the ARC-AGI-3 public set with the official harness, rising to 38.3%** with retained reasoning and compaction (~2.9x) — [OpenAI: How enabling two settings tripled our ARC-AGI-3 scores](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/)

### Inferences
- A 2.25x public-set gain producing zero private-set change is **the designed behaviour of this benchmark**, not an anomaly. The split was explicitly constructed so that public-set mechanics do not recur privately; a heuristic that exploits public mechanics has no support set on the private side by construction.
- The RHAE metric structure compounds this. Because the per-level score is `min(1, h/a)²` against *second-best human* action counts, clearing more levels at unchanged action efficiency moves the score very little, and the squaring means sub-human efficiency is penalised quadratically. A dev-set improvement measured in *levels cleared* can be near-orthogonal to the scored quantity. Two independent multipliers therefore sit between a public gain and a private score change: distribution shift, and metric shape.
- The 97.1% / 0.0% bimodality *within the public set* is a strong prior that 25 environments cannot support a stable estimate of anything. If the variance between two public environments is that large, the variance of a 25-environment mean is large enough to swallow most candidate effect sizes — which is exactly what a noise-floor calibration (Q2) would have revealed before acceptance.
- Everyone reporting near-100% on ARC-AGI-3 Public is reporting on the set the benchmark authors said they would never score. Those numbers are unfalsifiable by design and should not be read as harness capability.

### Gaps
- **No source reports how well public-set performance predicts private-set performance on ARC-AGI-3.** The technical report does not provide it, and no third party I found has published a paired public/private number for the same agent. This is the single most valuable missing number for the user's question.
- ARC Prize's reports do not publish Kaggle-style shakeup statistics (rank-correlation between semi-private and private), so the magnitude of divergence on ARC specifically is unquantified in public sources.
- I saw a search-result title (GitHub issue jjakimoto/research-issues #1313) asserting that a "Prime Agent" claim of raising ARC-AGI-3 RHAE Best@1 from 30% to 95.5% is "a split-choice and metric-definition artifact rather than a harness-capability result". I did not fetch or verify it, and it is a third-party issue tracker, not a primary source. Recorded only as a pointer.

---

## Q4. Which CLASSES of change transfer better — runtime-adaptive mechanisms vs tuned constants / static priors

### Takeaway
The evidence leans toward **structural and runtime-adaptive mechanisms** transferring better than static tuned artifacts, but it is indirect: RRSI's novelty bonus explicitly privileges structural components (skills, memory, subagents) while prompts and configuration are the only things routed through the leakage screen; AIDE²'s seven accepted rewrites are all runtime-adaptive (bandit search policy, context management, robustness safeguards); and EvoAgentBench shows the most static delivery mechanism (GEPA's single broadcast prompt) carrying a −12.3% worst cell. No paper runs a head-to-head ablation of adaptive vs constant.

### Cited Findings

**RRSI — implicit ranking, not a measured one**
- Structural components are privileged by design: the novelty bonus rewards candidates touching "skills, memory, subagents" that have never previously appeared in a winning edit, "suggesting these are expected to generalize" — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Prompt edits and configuration changes receive **no** novelty bonus and are subject to leakage screening before evaluation, "implying these are suspected of encoding task-specific patterns" — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- "Cost growth does not transfer": unregularized evolution overfits via "unnecessary complexity", so mechanisms purchased through added inference cost are less portable. Consistent with the token numbers (unregularized 3.80M tok/trial for +0.6 OOD; RRSI 2.42M for +3.9 OOD) — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- The paper cites prior work on **delta attribution**, separating "edits that install a reusable mechanism from those that merely fit the evolution tasks" — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Explicit caveat: the paper **does not directly categorize** which harness edits transfer versus which remain benchmark-specific — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)

**AIDE² — what the seven accepted rewrites actually were**
- Accepted rewrites concentrated on: a new **search policy using bandit selection with periodic forking**; **context management** with bounded prompts and failure-memory mechanisms; and **robustness safeguards against lucky one-off scores** — [arXiv 2609.26457](https://arxiv.org/html/2609.26457)
- All three are runtime-adaptive or variance-reducing. None is a tuned constant. (Inference from the list, not an authors' claim.)

**EvoAgentBench — delivery mechanism matters more than content**
- Paradigm-level patterns: Memento (raw case storage) is "vulnerable to surface mismatch; assumes query similarity predicts solution similarity"; ReasoningBank (abstracted strategies) "adds abstraction layer reducing surface dependence but shows modest gains"; GEPA (static evolved prompt) has "no per-task routing; broadcasts single prompt to all test instances" — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)
- The oracle-routed Anchor Skill beats every learned method and is positive in all 24 cells, while automatic extraction/routing is brittle — "the gap implicates method-side mechanisms rather than task difficulty" — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)
- The paper does **not** explicitly compare runtime-adaptive vs static artifact types across the three automatic methods — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)

**Compression paper — adaptive process, static artifact**
- Adaptive runtime mechanisms (exploration trajectories refined through validation interaction) "compress effectively into static descriptions when strategies are genuine, but this compression fails precisely when validation-specific exploitation occurs" — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)

**Survey framing (arXiv 2607.13104)**
- Scaffolding improvement "is typically faster and more easily reversible; it improves the agent by updating structural components", versus parameter updates which are a "slower but more stable form of long-term consolidation". The survey does **not** claim either class generalises better — [arXiv 2607.13104](https://arxiv.org/html/2607.13104v1)

### Inferences
- The defensible version of the claim is not "adaptive beats static" but **"mechanism beats memorisation"**. What transfers is a *procedure* that reads the current situation and branches (bandit selection, retrieval with routing, failure memory consulted at runtime). What does not transfer is a *value* or *rule* fitted to observed dev-set situations — a tuned constant, an enumerated case list, a broadcast prompt that encodes the dev set's mechanics.
- The compression finding gives a cheap test that cuts across the adaptive/static distinction: if the change survives being restated in ~32 tokens, it is a mechanism; if restating it requires enumerating the dev-set cases, it is a fit. This is more operational than "is it adaptive?", because a badly-designed adaptive mechanism whose thresholds were tuned per-environment is a fit wearing a mechanism's clothes.
- RRSI's cost finding suggests a second cheap proxy: **an edit that raises token spend and raises dev score is the highest-risk category.** Unregularized evolution bought +3.4 dev points for 2.4x tokens and +0.6 OOD.

### Gaps
- **No head-to-head experiment exists** comparing "adaptive mechanism" against "tuned constant" edits on the same OOD suite. Every claim in this section is inferred from which edits happened to be accepted, or from design intent, not from a controlled ablation. This is the weakest-evidenced of the five questions.
- Delta attribution is cited by RRSI as prior work but I did not locate and verify the underlying paper, so I cannot report its method or numbers.

---

## Q5. Does a broad, stable dev-set plateau constitute evidence of robustness?

### Takeaway
No. The HPO literature states directly that flat or plateau regions in a validation landscape are **not** evidence of robust generalisation, because optimisation can exploit shallow validation variations that do not reflect test behaviour; and RRSI's own algorithm treats a plateau as a *stall signal to redirect search*, not as a robustness signal. RRSI's ablation table goes further: within it, dev-set score and OOD gain are anti-correlated, so a high, stable dev score is if anything a mild negative indicator. I found **no paper that reports a plateau-width-versus-transfer analysis**, so the direct answer to "is my plateau meaningful?" is untested in the literature.

### Cited Findings

**Overtuning in Hyperparameter Optimization (arXiv 2506.19540, 2025)**
- "Overtuning" defined as hyperparameter-optimisation gains on the validation set failing to transfer to the test set — [arXiv 2506.19540](https://arxiv.org/pdf/2506.19540)
- On landscape shape: flat or plateau regions in the hyperparameter landscape are **not necessarily evidence of robust generalization**; optimisation "can exploit shallow variations in validation performance that don't reflect test set behaviour", so "landscape flatness alone is insufficient to guarantee generalization robustness" — [arXiv 2506.19540](https://arxiv.org/pdf/2506.19540)
- Magnitude depends on validation set size, optimisation budget, and algorithm — [arXiv 2506.19540](https://arxiv.org/pdf/2506.19540)

**RRSI treats a plateau as exhausted search, not as robustness**
- Stall detection: progress is defined as stalled "when its progress over the previous w rounds remains within the empirical noise band δ", implemented as **σ_t = 1[Ŝ_t − Ŝ_{t−w} ≤ δ]**. On stall, exploration budget is **reserved for underexercised components** — i.e. the response to a plateau is to go look elsewhere — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- RRSI has **no plateau-based termination**; experiments run a fixed T rounds, and the paper does not report whether evolution terminates on plateau detection — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)
- Anti-correlation within Table 2: the highest dev score (unregularized, 92.8) pairs with the lowest OOD (40.3); the lowest dev score among evolved variants (RRSI, 90.5) pairs with the highest OOD (43.6) — [arXiv 2609.24972 (HTML)](https://arxiv.org/html/2609.24972v1)

**The dev curve does not reveal exploitation; a separate test does**
- In the compression paper's deliberate-exploitation experiment, validation-specific gains were caught by the **compressibility test** at 100% sensitivity / 91% specificity — the detection came from an external discriminator, not from the shape of the validation trajectory. In the non-exploiting runs the validation–holdout gap stayed at 0.0–1.9pp — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)
- The adaptive-reuse bound widens monotonically with checkpoint count (~2.4pp → ~4.7pp over 7 checkpoints at n=5,000), meaning the *longer* you sit on a dev set the *less* a stable reading means — [arXiv 2606.11045](https://arxiv.org/html/2606.11045v1)

**Averaged plateaus hide sign flips (EvoAgentBench)**
- ReasoningBank shows positive averages across backbones while containing **six negative per-domain cells**; GEPA averages +1.2% to +5.7% while containing a **−12.3%** cell — a stable-looking aggregate concealing per-domain reversals — [arXiv 2607.05202](https://arxiv.org/html/2607.05202v1)

**Countervailing older evidence: Kaggle holdout reuse (Roelofs et al., NeurIPS 2019 — PRE-2025)**
- Meta-analysis of **over 100 Kaggle competitions** found "little evidence of substantial overfitting" from repeated holdout evaluation, with robustness holding "across different data domains, loss functions, model classes, and human analysts" — [Roelofs et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf)
- But overfitting **does** appear under specific conditions: **small public test sets** ("practitioners with access to repeated feedback on limited test data can more easily adapt models to noise rather than genuine patterns") and **small overall datasets**, which "showed greater divergence between public rankings and final private test rankings" — [Roelofs et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf)
- General shakeup framing: final standings use "a disjoint subset of the test data" precisely to prevent public-leaderboard overfitting; shakeup is typically measured as mean absolute percentage change in rank, or Spearman rank correlation between public and private — [davidthaler/shakeup](https://github.com/davidthaler/shakeup); [Kaggle Handbook: Surviving a Shake-up](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)

### Inferences
- Roelofs et al. is the strongest-looking counter-argument and it **does not apply** to the user's situation, for two reasons worth stating plainly to the report writer: (a) it measures *adaptive overfitting to a same-distribution holdout*, whereas ARC-AGI-3's private set is stated to be out-of-distribution by construction — a different failure mode entirely; and (b) its own boundary condition (small public test sets, small datasets) is exactly the 25-environment regime. Roelofs predicts overfitting *here*, not against it.
- A plateau on a dev set is a statement about the **variance of the dev-set estimator**, not about the transfer of the mechanism. A flat region over a parameter range means the dev set cannot distinguish those parameter values — which is equally consistent with "the parameter doesn't matter" and "the dev set is too small/too easy to resolve it". With 25 environments showing 97.1%/0.0% bimodality between neighbours, the second reading is the likelier one.
- Combining RRSI's δ calibration with the plateau observation gives the diagnostic the user actually wants: **calibrate δ by re-running the unchanged baseline on the 25-game set several times.** If the plateau's width is comparable to δ, the plateau is noise-flat, not robustness-flat. This is a cheap, decisive experiment and nobody in the literature appears to have published it for an agent harness.
- Practical upshot for the user's 2.25x: three independent mechanisms each suffice to explain zero private movement — distribution shift (private mechanics disjoint by design), metric shape (RHAE's squared human-relative efficiency term means level-count gains need not score), and estimator variance on 25 samples. None of them requires the improvement to be fake on the public set; the public gain can be entirely real and still worth nothing privately.

### Gaps
- **No documented case in the literature explicitly frames "broad stable parameter plateau on dev, zero transfer" as an observed result.** The closest available statements are the HPO paper's assertion that flatness is insufficient, and RRSI's anti-correlated Table 2. The user's observation appears to be **novel as a reported finding**, which is worth saying in the report rather than pretending a citation exists.
- I could not extract quantitative overtuning magnitudes from arXiv 2506.19540 (PDF text extraction failed); only its qualitative claims are recorded above. Someone should re-fetch the HTML version for the effect sizes.
- No source quantifies the relationship between dev-set size and transfer reliability for *agent harness* selection specifically. Roelofs' small-test-set finding is the nearest analogue and comes from supervised-learning competitions, pre-2025.
- arXiv 2510.08413 ("Prompts Generalize with Low Data: Non-vacuous Generalization Bounds for Optimizing Prompts with More Informative Priors") appeared in search and is directly on-topic for bounding prompt-optimisation generalisation from small dev sets. **I did not fetch it** and record no claims from it — flagged as the highest-value unread source for a follow-up pass.
