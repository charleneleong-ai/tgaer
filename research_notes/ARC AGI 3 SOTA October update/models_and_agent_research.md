# Open-weight models and interactive-agent research, delta since 2026-10-06 (checked 2026-10-11)

Scope: what is new relative to `reports/ARC AGI 3 vLLM port SOTA.md` (Qwen 4 not released, Kolibri-1, DeepSeek-V4.1-Flash and GLM-5.3 noted as too big, Qwen3.8-27B slower than Flash-Next) and `reports/Self evolving agents from execution feedback.md` (Tycho, AVO, VISTA headline, Retrodict, OPINE-World, Twin, Rodionov, AERA, RRSI, aTTT). Items already covered there are not repeated except where a new number changes them. Every benchmark figure below is author- or vendor-reported unless stated otherwise; nothing here was reproduced.

## Q1. Is there a new open-weight model that could beat Qwen3.8-Flash-Next W4A16 within 96 GB on SM120?

### Takeaway
No. As of 2026-10-11, no lab has published a new general LLM checkpoint on Hugging Face since about 2026-09-15 that fits 96 GB at 4 to 8 bit. Qwen 4 is still unreleased, the only open-weight model with a modern ARC-AGI-3 number (GLM-5.3-Flash, 321B) is too large, and the new community artefacts are quantisations or fine-tunes of models already in hand.

### Cited Findings
**Qwen**
- The Qwen HF org's newest LLM checkpoints are still Qwen3.8-Flash-Next and its FP8 variant (created 24 Aug 2026, arch `qwen4_exp`, licence "other"). Qwen3.8-27B (5 Aug, Apache-2.0) and Qwen3.8-2.4T-A95B (8 Aug) are older. Everything created since is image generation (Qwen-Image-2.1, 14 Sep; Qwen-Image-2.1-Turbo, 9 Oct) — [HF Qwen org, sorted by created](https://huggingface.co/Qwen).
- Qwen 4 was announced at Apsara on 2026-09-22 in four tiers (Max, Plus, Flash, and an open-weights 27B), with no specs, licence, benchmarks or date. No Qwen 4 repo exists on HF. These details come from secondary blogs, not Alibaba primary sources, so treat them as unverified — [Yotta Labs](https://www.yottalabs.ai/post/qwen-4-release-date-what-is-known-how-to-prepare-2026); [orcarouter](https://www.orcarouter.ai/blog/qwen-4-max-lineup-announced-apsara-2026); [cellcog](https://cellcog.ai/blog/qwen-4-release-date/).
- vLLM PR "Add qwen4 fuse op" (#53909) has been open since 26 Aug and was still unmerged as of the latest secondary report — [vLLM PRs](https://github.com/vllm-project/vllm/pulls), as reported by [orcarouter](https://www.orcarouter.ai/blog/qwen-4-leak-vllm-fuse-op).

**Other labs (HF orgs sorted by creation date)**
- zai-org: GLM-5.3 and GLM-5.3-Flash were both released 25 Aug 2026. GLM-5.3-Flash has 321.3B params (HF metadata), is multimodal (`glm5_next`) and MIT-licensed. Nothing newer has appeared — [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). At 4 bit that is about 160 GB of weights, so it does not fit 96 GB.
- deepseek-ai: DeepSeek-V4.1-Flash (10 Sep, MIT, 763.2B params) has an NVIDIA NVFP4 build (16 Sep). Neither fits 96 GB — [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash); [nvidia NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4).
- moonshotai: the newest model is Kimi-K3 (13 Jun) — [moonshotai](https://huggingface.co/moonshotai).
- mistralai: the only post-September items are Voxtral-Mini-4B-Realtime-Arabic and LIDstral-Arabic (8 Oct). The newest general LLMs are Leanstral-1.5-119B-A6B (1 Jul, a Lean prover) and Mistral-Medium-3.5-128B (31 Mar) — [mistralai](https://huggingface.co/mistralai).
- google: there is no new Gemma LLM. The newest items are embeddinggemma-2 (14 Sep), TimesFM 3.0 and TIPS — [google](https://huggingface.co/google).
- openai: there is no new gpt-oss. The newest LLM is still gpt-oss-120b/20b (Aug 2025) — [openai](https://huggingface.co/openai).

**Community artefacts trending since late September (all derivatives)**
- Cloudflare/clef (and clef-flash) is Qwen3.8-27B post-trained into a classifier that "returns a probability for every allowed option… There is no free-form text generation." It cannot act as a generative agent backbone. Its vendor-reported GPQA-Diamond is 48.0 — [Cloudflare/clef](https://huggingface.co/Cloudflare/clef).
- ConwayResearch/Underdog-Saluki-27B-1.0 is a 7.89 GB IQ2 GGUF of Qwen3.8-27B for llama.cpp. Vendor-reported: BFCL subset 88/120 vs 84 for the full model, but MuSR 67.5 vs 79.6 and AIME25 79.2 vs 96.7 — [card](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0). The reasoning loss rules it out for ARC.
- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF provides non-uniform GGUF quants of Flash-Next (512 experts per layer, 48 layers, 10 active; a 51.2B per-layer n-gram embedding table held at IQ4_NL). A Coder variant drops to 256 experts (58.4 GB). These are llama.cpp/Ollama formats, not vLLM/SGLang — [card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
- Venastine-Research/Xing4.0-29B-A4B-GGUF is a 29B/4B-active MLA MoE from an unknown lab. Its card gives architecture only, with no comparable reasoning numbers in the part read — [card](https://huggingface.co/Venastine-Research/Xing4.0-29B-A4B-GGUF).

### Inferences
- The model axis is effectively frozen until Qwen 4 27B lands. If Qwen 4 slips past roughly 10-25, there is no time to validate it before the 11-02 deadline.
- GLM-5.3-Flash is the only open model with an ARC-AGI-3 harness result (see Q2: VISTA, RHAE 66.93 public). Fitting it into 96 GB would need about 2-bit weights. Given the reasoning loss Saluki shows at IQ2 on a 27B, that trade looks poor.
- The GSQ-RCO 256-expert pruned Flash-Next (58.4 GB) would free about 40 GB of KV. That is the "memory, not model" lever from the prior report. It ships only as GGUF, though, so it would need either a llama.cpp backend or a vLLM GGUF-MoE path, and neither has been verified on SM120 here.

### Gaps
- No primary Alibaba statement on Qwen 4 timing or licence was found; only third-party blogs.
- No SM120 (RTX PRO 6000) vLLM/SGLang support status was found for GLM-5.3-Flash (`glm5_next`) or for the GSQ-RCO GGUFs.
- The Xing4.0 card was read only partially, and the lab is unknown.

## Q2. New research with measured gains on ARC-AGI-3 or similar interactive benchmarks

### Takeaway
New since the prior reports: Memento 3 (2610.11794, 8 Oct) reaches RHAE 100.0 using a verified natural-language-rulebook-to-code world model, VISTA adds an open-weight backbone result and context/image-scale ablations, PRO-LONG puts programmatic memory over a raw interaction log, and the Prime Agent REPL harness goes from 30% to 95.5% RHAE. All use frontier API models on the 25 public games. The one transferable result with an open model is VISTA + GLM-5.3-Flash at 66.93 RHAE.

### Cited Findings
**Memento 3, model-based recursive self-improvement via reflective rulebooks** ([arXiv 2610.11794](https://arxiv.org/abs/2610.11794), UCL/Huawei, 8 Oct 2026)
- The agent keeps `world_model.md` (a natural-language rulebook of revisable hypotheses, with unknowns left underspecified) and compiles it into an executable Python world model. Code is accepted only if an LLM judge finds it faithful to the rulebook and cell-exact replay over all 64×64 cells reproduces the observed transitions. A planner then searches on the verified model — [arXiv 2610.11794](https://arxiv.org/html/2610.11794).
- Author-reported on 25 public games: every level is cleared, mean RHAE is 100.0, the agent uses 44% of the human action count (7,518 actions total), and it scores 59.3 points above Claude Opus 5 under the ARC Prize Standard harness. The backbone is Claude Opus 5 at extra-high effort via Claude Code. Comparators in the same table: baseline1 99.0, NOOA 85.1 (19/25), OPINE-World 78.4 (20/25), DreamTeam 38.1 (6/25), Continual Harness 20.5 (3/25) — [arXiv 2610.11794](https://arxiv.org/html/2610.11794).
- Rulebook ablation on three games (ft09, ka59, cn04; all runs reach RHAE 100): the rulebook cuts actions 9% (617 vs 677) and agent turns 18% (678 vs 830). This is n=1 run per game, author-reported — [arXiv 2610.11794](https://arxiv.org/html/2610.11794).
- A population variant runs several world models in parallel, shares evidence, and uses their disagreement to steer exploration — [arXiv 2610.11794](https://arxiv.org/abs/2610.11794).

**VISTA, visual harness, new details** ([arXiv 2610.02200](https://arxiv.org/abs/2610.02200), 1 Oct 2026)
- Open-weight backbone: VISTA with **GLM-5.3 Flash (320B)** reaches **RHAE 66.93** on the 25 public games, "substantially outperforming" the official minimal harness on the same model. Author-reported — [arXiv 2610.02200](https://arxiv.org/html/2610.02200).
- Context ablation (Opus 5.0): the 200K max context before compaction gives the best RHAE, 99.00. 100K loses only 0.4 RHAE while roughly halving tokens. 780K costs 105.7M tokens per game vs 30.7M at 200K — [arXiv 2610.02200](https://arxiv.org/html/2610.02200).
- Image-scale ablation: 4× (256 px per side, about 77 image tokens per frame) matches 8× (512 px, about 308 tokens) at 75% fewer image tokens. 2× (128 px, about 20 tokens) degrades, and 16× (1,024 px) adds nothing — [arXiv 2610.02200](https://arxiv.org/html/2610.02200).
- Official baselines quoted there: Opus 5.0 high 40.68, GPT-5.6 Sol max 13.33 — [arXiv 2610.02200](https://arxiv.org/html/2610.02200).

**PRO-LONG, programmatic memory for long-horizon agents** ([arXiv 2607.20064](https://arxiv.org/abs/2607.20064), Duke, Jul 2026)
- The agent keeps a complete structured interaction log and lets a coding agent search it programmatically, instead of summarising or compacting it. On the full ARC-AGI-3 public set it averages +18.0 pp over the base coding agent across frontier models and reaches up to 76.1% pass@1, matching specialised harnesses with 4.2–5.8× fewer billed tokens. Fable 5 gets 97.4% best@2 at $1,750, and 94.6 pass@1 at a 2,000-action budget. All author-reported — [arXiv 2607.20064](https://arxiv.org/html/2607.20064).

**Prime Agent, self-improving RLM harness** ([arXiv 2608.23552](https://arxiv.org/abs/2608.23552), Prime Intellect/MIT, Aug 2026)
- A persistent IPython-REPL, daemon-backed recursive harness. It lifts ARC-AGI-3 RHAE Best@1 from 30% to 95.5%. The open models tested (Kimi K3, DeepSeek V4 Pro, GLM 5.3) appear on nanoGPT/coding tasks, not ARC — [arXiv 2608.23552](https://arxiv.org/html/2608.23552).

**Test-time weight adaptation (non-ARC, new)**
- ASCENT ([arXiv 2610.05303](https://arxiv.org/abs/2610.05303)) runs online test-time training by self-distilling verified trajectories into persistent LoRA fast weights, with a frozen initial copy as the teacher and invalid-action turns removed. Authors report gains in success and efficiency on ALFWorld, WebShop and AppWorld. It needs a verification signal per task.
- Agentic-TTT ([arXiv 2610.12002](https://arxiv.org/abs/2610.12002)) learns a policy for when and which TTT procedure to invoke, and "nearly doubles utility over backbone model" on the authors' own benchmark.
- "One Rollout Is All You Get" ([arXiv 2609.34321](https://arxiv.org/abs/2609.34321)) does fully test-time adaptation for GUI agents from a single rollout. Only the HF abstract snippet was read; no numbers were extracted.

**ARC Prize / frontier labs since September**
- No arcprize.org blog post on September ARC-AGI-3 results was found. The leaderboard page renders scores dynamically, so they could not be extracted — [arcprize.org/leaderboard](https://arcprize.org/leaderboard). ARC Prize posted on X that Yi-Chia Chen took first place on the Kaggle track at 28.34% (date not confirmed) — [@arcprize](https://x.com/arcprize/status/2104590501915787290). This is superseded by the 48.59 in the prior report.

### Inferences
- The converging frontier recipe has three parts: (1) an explicit, revisable hypothesis store (rulebook or ontology), (2) compiled code verified by exact replay of every observed transition, and (3) planning on the verified model. Memento 3's ablation suggests the rulebook layer buys efficiency (9% fewer actions), not solvability, and that is on frontier models only.
- VISTA's GLM-5.3-Flash result (66.93 vs 99–100 for frontier models) is the only data point on how much of the frontier harness gain survives an open backbone. Roughly two-thirds survives, on public games, with a 320B model. A 125B/6B-active model under 9 h and a 4-bit quant should be expected to keep less.
- Two cheap, transferable knobs come from VISTA: about 256 px boards (77 tokens per frame) and a compaction threshold near 100K. Both conserve KV without measured loss. This lines up with the prior report's "spend memory on history" lever, but it implies image tokens can be cut before history is trimmed.
- PRO-LONG's "keep the full log, search it with code" approach is the context-management result most compatible with a fixed KV budget. It replaces summaries (null in the prior report) with retrieval.
- Gradient TTT papers (ASCENT, Agentic-TTT) need per-task verification and multi-task streams. In ARC-AGI-3, a level clear is the only verifier, so within-game LoRA updates would train on very few verified trajectories. This is consistent with the prior report's aTTT verdict.

### Gaps
- No October 2026 ARC-AGI-3 paper was found that evaluates a model that fits 96 GB (for example Qwen3.8-Flash-Next or Qwen3.8-27B) on the public or semi-private set.
- No ARC Prize team analysis of what worked at Milestone 2 was found; the leaderboard data was not extractable.
- Memento 3's population-variant numbers and its wall-clock/token costs were not extracted. Whether any of these systems would run within 9 h on one GPU is not reported.
- OpenAI's July post "How enabling two settings tripled our scores on the ARC-AGI-3 benchmark" is cited in DiG-bench ([arXiv 2608.12593](https://arxiv.org/pdf/2608.12593)) but was not located.
