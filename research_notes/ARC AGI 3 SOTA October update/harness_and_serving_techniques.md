# Harness and serving techniques for dfranzen/Duck/TAAF harnesses, ARC-AGI-3 Kaggle, 2026-10-01 to 2026-10-11

Scope: only items that are not already in `reports/ARC AGI 3 vLLM port SOTA.md` and its research notes (dated 2026-10-06). Those notes already cover: rellik13's +8.0 from history depth, D' slot priority (31.54 / 26.98), skarin's 100-trajectory audit, leoprovorov's handbook and the journal-replacement null, lordhansolo's v2/v3 vLLM runtimes, the primitive-ai / local-inference-lab / RadixArk quants, and Son Pham's copy-variance numbers (mean 25.77, sd 3.93).

Method: Kaggle discussion threads and notebook sources were pulled through Kaggle's JSON endpoints (forum 10403401, competition 133468) with our API token. Notebook sources came from `api/v1/kernels/pull`. GitHub repos and releases came through `gh api`, model cards through the Hugging Face Hub. Raw files are in the session scratchpad (`kd/`, `nb/`, `pr3_rel.md`, `son_sprint.md`). Kaggle pages render client-side, so the links below go to the page; the text was read from the JSON.

Context for the numbers below. The public LB CSV of 2026-10-11 02:48 UTC shows a large move since 10-06:

| Rank | Team | Score | Prior note |
|---|---|---|---|
| 1 | Yi-Chia Chen | 62.96 | was 48.59 |
| 2 | Tufa Labs | 56.52 | was 55.89 |
| 3 | mtg | 45.00 | |
| 4 | Majkel1337 | 42.66 | |
| 5 | NVARC3 | 40.97 | was 21.79 |
| 6 | the last dance | 39.30 | |
| 7 | Son & Mark & Ronen | 39.24 | |
| 8 | dreach.ai | 39.11 | |
| 16 | Lord Han Solo | 36.61 | unchanged |

Source: [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard). None of the top five has disclosed a method. CPMP (NVARC3) says "I can't disclose anything about what we do before end of competition" ([746010](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746010)).

## Q1. In-game retry/restart, context/history management and cross-game scheduling: who reported what since 10-01

### Takeaway
Nobody has published a controlled public-LB gain for a retry, history or scheduling change since 10-01. The most substantial new work is Son Pham's team's "level-start sprint": exact carried-context checkpoints at stuck levels, used to A/B harness variants on the levels that decide the score. It is infrastructure plus one harvest result, not a score. Their best local configuration runs 12 lanes with HiCache and a ~131K window and scores 55.9–60.6 on the public 25. One prompt arm lost 12 points there. A harness-computed "level log" (leoprovorov F002) scored 18.95 public vs 25.98 for F001. That is one draw each, but it points the same way as the earlier journal null. Our level re-attempt (RESET after 80K tokens, restore level-start history) has no public analogue with numbers. The closest are Son's carried-context restarts and rellik13's earlier null "fresh-context restart".

### Cited Findings

**Level restart / carried-context checkpoints (Son Pham, Mark Barney and Ronen (Kaggle `scizical`); team "Son & Mark & Ronen", LB 39.24)**
- On 2026-10-09 the team started a "level-start sprint". They tallied every public-game level cleared under ~80% of the time in their nine-hour runs, then built a notebook that plays only those stuck levels from saved starts, about ten lanes at once with a 30-minute clock. The stuck levels include Buoyant Pontoons L3 (67%), Skewer Kebabs L5 (33%), Leapfrog L6 (17%) and Streaming Purple L5 (0%) — [sonpham-org/arc-3 docs/2026-10-09-level-start-sprint.md](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)
- The notebook supports four start kinds:
  - `exact`: restore a saved carried-context checkpoint (conversation plus game state). The first request is checked byte-identical to the saved one.
  - `warmup`: replay the winning line to the level before, then let the model play into the target level carrying its own context.
  - `replay`: the stuck level with no context.
  - `auto`: the default since version 3.

  Restored checkpoints carried 70–91 messages of history and 7–35 retained functions — [same doc](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)
- Measured so far (one pass per arm, by the authors' account noisy):
  - The no-context, 30-minute, one-pass control "reads zero everywhere": 0–1 of 11 stuck levels cleared, flipping from run to run.
  - The 6-hour carried-context "start harvest" (16 lanes on 12 request slots, 5,000-action cap) cleared the stuck level on 9 of 16 lanes, covering 6 of 9 targeted levels.
  - Their verdict: "half hour often not enough to clear even the level before." The sprint is "only good with saved starts."
  — [same doc, §3 and §5](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)
- The same team found a Kaggle dataset-version race. A notebook pushed four seconds before Kaggle finished processing a dataset version silently ran the previous version's code and checkpoints, and `kaggle datasets status` reported "ready" for the old version. They now gate on a build-ID marker file — [same doc](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)

**Prompt / information changes (local, public-25)**
- Son's "draftB" short system prompt was tested on the INT8 v2 build with 12 lanes, HiCache and a ~131K window:
  - It scored 46.69 (123 levels) against four same-build controls at 59.51, 58.64, 55.87 and 60.58 (127–137 levels).
  - It lost on Reaching Lurch, Mirror Rendezvous, Functional Tiles, Axis Reflectors, Toggle Runes and Coded Notches.
  - It did better on Sliding Indicator, Sucking Up and Streaming Purple, spending ~1,800 more actions.
  - The author's reading: "more probing, less early commitment". Traces were not recoverable.

  — [docs/plans/2026-10-08-draftb-trace-diff.md](https://github.com/sonpham-org/arc-3/blob/main/docs/plans/2026-10-08-draftb-trace-diff.md)
- leoprovorov's "Fable layer" notebooks on the unmodified dfranzen stack:
  - F001 pins harness-computed facts (legend, tray and counter changes; a low-budget line; a pinned LEVEL LOG on levels that have taken a long time). It scored **25.98** public.
  - F002 adds a block, on the first turn of every level after the first, stating what each key did on cleared levels. It scored **18.95** public.

  One draw each — [F001](https://www.kaggle.com/code/leoprovorov/f001-fable-layer-notebook); [F002](https://www.kaggle.com/code/leoprovorov/f002-fable-layer-notebook) (scores from the Kaggle kernel list `bestPublicScore`)
- Shehab Anwer added "board-view / lexicon datafication" to the dfranzen base and got 27.05 vs 24.95 for his own baseline. He calls it "no effect" relative to the 25.77 ± 3.93 copy distribution. He also reports eight draws of his Flash-Next Duck variants (mean 3.07, sd 0.35, CV 11%) and a save-score vs LB Pearson r of −0.32 across 5 versions on a 6-game regime — [745062](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745062)

**Context vs concurrency**
- rellik13 (10-01): "logs showed me that kv pool sat somewhere 91-97% full [and] requests queued [for] cache space, memory [was the] actual limit". FP8 KV alone took him from 14.5 to 22.5 by spending the freed memory on history (trims 37k→27k became 57k→45k). He found that harness wording fixes such as UNDO instead of ACTION7, "level restarted" instead of "game over" and removing the "ignore budget bar" lines helped "simply to not mislead [the] agent" — [744792](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792)
- xz (10-01) says he ablated NVFP4 KV and found "very little NLL penalty" and has used it since. He also found his harness "less action-efficient" than the milestone solutions at matched tokens and did not know whether KV quantisation or prompts were the cause. rellik13 replied that a short-context NLL check may miss 60–100K-token effects, and that a token-matched run changing only the KV dtype would settle it — [744792](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792)
- Son's local runner (09-10 changelog) admits 13 games on 10 slots in his Kaggle notebook. The sprint used 16 lanes on 12 request slots, described as "about the same over-admission as Son's own notebook" — [level-start-sprint doc §5](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)

**Scheduling**
- Son's team compared Affectify's D' with dfranzen's priority rule side by side. The structural differences:
  - D' adds the levels-left bonus outside the hazard term, so stuck games keep a floor.
  - Its pace term rewards cheap clears.
  - Its hazard decays linearly over ~225K tokens, against dfranzen's 62K half-life.
  - Its fade starts at 40% rather than 20%.
  - Unstarted games compete on the same formula instead of jumping the queue.

  Their stance: "promising candidate, not proven gain", gated on repeat passes — [docs/2026-10-05-dprime-slot-priority-vs-franzen.md](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-05-dprime-slot-priority-vs-franzen.md)
- New D' derivatives since 10-05 have single public draws: lwq255 "arc3-dprime-guarded" 29.15, and amatlas's D-prime fork best 30.82 (two D' draws were already noted) — [lwq255](https://www.kaggle.com/code/lwq255/arc3-dprime-guarded); [amatlas](https://www.kaggle.com/code/amatlas/dfranzen-fork-d-prime)

**Local public-25 reference points (self-reported, unverified)**
- Nick2187: 56.76 (124 levels, 11 games fully won). Nick Pellegrin: ~105 levels on the public 25 (~40%) across 3 runs, scoring ~23 on the LB. Scott Le Grand: 53.74, "haven't been able to repeat it" — [732854](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732854)
- Mark Barney names seven public games as nearly unsolvable ("Slippery Seven"). rellik13: "some of these games dont budge beyond 2nd level no matter what i do" — [732854](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732854)

### Inferences
- Son's team and our level re-attempt attack the same failure: stuck mid-game levels. Their harvest suggests carried context plus long wall time (6 h) clears stuck levels a fresh 30-minute attempt cannot. That is consistent with our result that re-attempt with restored level-start history (keeping context, keeping the slot) beat the unmodified harness locally, while rellik13's no-context restart was null. The common factor is keeping the model's own history, which matches every history result to date.
- Son's local controls (55.9–60.6, 12 lanes, HiCache, ~131K window) sit well above our unmodified 42.9 mean on the same 25 games. Their team's LB is only 39.24 (best draw). Local public-25 again overstates LB, as in our own memory. Their configuration does show that 12 streams with HiCache and a bigger window is viable on this card, which our 12-stream test (−40% throughput) did not.
- Two prompt data points (draftB −12 locally, F002 −7 public, single draws) reinforce the existing rule: removing or adding prompt information is at best null and often harmful.

### Gaps
- No public A/B of level re-attempt / RESET-and-restore on the LB exists. Son's sprint has not yet reported its `auto` baseline vs variant results.
- Son's "INT8 v2" build is not documented publicly. The kernel name `arc3-daniel-8bit-w4kw8d-final` suggests W4 weights, 8-bit KV and an 8-bit drafter, but this is an inference from the name.
- The top five (Yi-Chia 62.96, Tufa 56.52, mtg 45.00, Majkel1337 42.66, NVARC3 40.97) have disclosed nothing.

## Q2. Serving: SGLang HiCache, vLLM builds, speculative decoding/MTP, longer context, NVFP4/FP8 on SM120, decode speedups

### Takeaway
The main new serving item is **Penny Royal 3.0** (released 2026-10-10), the successor of the Pennyroyal 2.5.3 fork that dfranzen and we run. Vendor-reported changes:
- +17% single-request decode
- a 2.7% bigger token pool
- adaptive MTP
- host-side Mamba checkpoint retention for sessions
- HiCache fixes

Its adaptive MTP only adapts at concurrency 1. At our 10 streams the draft width stays fixed at 4, so most of the headline speedup will not transfer. Upstream vLLM 0.31.0 (10-05) absorbed most of lordhansolo's overlay ports. lordhansolo's datasets have not been updated since 09-30. Some teams appear to be training custom MTP drafters on ARC traces, but no numbers are published.

### Cited Findings

**Penny Royal 3.0 (jpezzulli/sglang-rtxpro6000, tag `pennyroyal-v3.0.0`, 2026-10-10)** — vendor-reported, on the OrcaRouter ModelOpt NVFP4 checkpoint, not the Intel W4A16 we use — [release notes](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v3.0.0); [RESULTS.md](https://github.com/jpezzulli/sglang-rtxpro6000/blob/pennyroyal-v3.0.0/RESULTS.md#penny-royal-30)
- Adaptive MTP is on by default, "adapts only [at] C=1": a single active request switches between 4- and 8-token windows by draft acceptance. "C=2 [and] above, draft width [is] fixed [at] four." FR-Spec stays enabled.
- Speed figures:
  - C1 mean rose 186.79 → 218.63 tok/s (+17.0%) over 2.5.3, but the baseline ran medium reasoning and the 3.0 run xhigh.
  - Four concurrent requests reach 489.35 tok/s aggregate (mean). Six concurrent requests reach 433.54 tok/s aggregate, lower than at four.
  - Cold prefill runs at 17,236 tok/s at 64K and 8,840 tok/s at 490K.
- The one-card C6 GPU token pool grew 1,039,040 → 1,067,072 tokens (+2.7%) with 4.19 GiB free at startup. The per-request context limit stays 524,288.
- Memory savings came from FP8 small-batch SM120 rowwise projections and output head, an NVFP4 draft-MoE, and FP32→BF16 changes in extend/prefill and QSA scratch. "These savings recovered adaptive MTP's extra memory cost."
- Long-session fixes:
  - speculative Mamba buffers are reserved to actual need, with context room left for lookahead
  - "Keep useful host Mamba checkpoints, release abandoned session branches, track shared-backup ownership", with session retention keyed on client session IDs
  - separate persistent-cache namespaces per FP8 representation and GDN mode
  - image requests whose shared-memory segment has vanished are rejected instead of crashing the scheduler
- Packaging and runtime: FlashInfer 0.7.0.post1 on CUDA 13. Container `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.0` (~7.65 GB). Configuration now lives outside the container.
- 2.5.3 (09-28) for reference: MTP checkpoint-selection fix, BF16 NVMe PLE hotfix, parallel tokenisation of long prompts, HiCache persistence fixes — [v2.5.3 notes](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v2.5.3)

**Upstream vLLM v0.31.0 (2026-10-05)** — [release notes](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)
- Qwen3.8-Flash-Next (Qwen4Exp) items include:
  - FP8 main KV on the QSA path (#55557)
  - fused HC down-projection plus SiLU (#58957)
  - lower PLE metadata overhead (#58114)
  - profiling KV cache released (#58961)
  - pinned PLE prefetch IDs kept out of the CUDA-graph pool (#58489)
  - indexed expert mapping (#58720)

  Four of these (#58114, #58961, #58489, #58720) are PRs lordhansolo's v3 overlay back-ported.
- Hybrid-model prefix caching:
  - `--enable-mamba-shared-prefix-checkpoint` (#57382)
  - prompt-tail hits restored under MTP (#58368)
  - GDN prefix-caching changes (#51565)
- Scheduler: `--max-num-active-seqs` caps running admission independently of `max_num_seqs` (#56758).
- Breaking: `VLLM_PLE_CPU_OFFLOAD` is removed; use `--engram-config` (#57937).
- Upstream SGLang v0.5.21 (10-02) has no Flash-Next/SM120-specific items. Its spec-decode and HiCache work targets DeepSeek/GLM, SM100 and PD disaggregation — [release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)

**lordhansolo stack status**
- `lordhansolo/vllm-main-e975732-arc3` is still version 3 (last updated 2026-09-30 11:38) and `taaf-kaggle-source` is still v305 (09-30) — Kaggle datasets API ([vllm-main-e975732-arc3](https://www.kaggle.com/datasets/lordhansolo/vllm-main-e975732-arc3), [taaf-kaggle-source](https://www.kaggle.com/datasets/lordhansolo/taaf-kaggle-source)). His LB 36.61 remains undisclosed.

**Custom / tuned MTP drafters (unquantified)**
- Son's team ships private datasets `flashnext-mtp-drafter-fx10` and `flashnext-mtp-drafter-tuned`. Their serving cell lists "noborder, tuned drafter, hot map, temperature, toolfast, host tier, rejection sampling" on Flash-Next INT4 + drafter with twelve request slots — [level-start-sprint doc](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)
- wkdrbwnd1 published lordhansolo-based notebooks named `mtp-featdump-full`, `mtp-featdump-verify` and `mtp-gtcapture` (10-06). By name, they dump features for drafter training. No results — [mtp-featdump-full](https://www.kaggle.com/code/wkdrbwnd1/mtp-featdump-full)

**Other public stacks**
- superstringdev's "ARC-AGI-3 NVFP4 / SGLang" integrated a ModelOpt NVFP4 W4A16 block-16 SM120 Marlin MoE with BF16 PLE. It scored **9.8** public. Its own notes flag unresolved error-identity review failures and "Submit-readiness ... not approved" — [notebook](https://www.kaggle.com/code/superstringdev/arc-agi-3-nvfp4-sglang-public)
- Practitioner comments: Scott Le Grand: coding agents were "useless ... for understanding vllm and sglang". NNbody reports "redundancy in mamba slot allocation for Qwen 3.8 Flash Next" and that "every time I get TPS improvement, some errors show up" — [744792](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792)

### Inferences
- **Penny Royal 3.0 is the cheapest serving upgrade to test, but expect less than +17%.** That figure is single-request with adaptive width; our 10 streams run fixed width 4. Its C4 → C6 aggregate *falls* (489 → 434 tok/s on their workload). That backs our finding that more streams do not buy throughput. What may matter for us:
  - the +2.7% pool
  - host Mamba checkpoint retention, which could make level re-attempt (restore level-start history) and HiCache reuse cheaper
  - the scheduler fix for missing image segments

  The benchmarks used the NVFP4 checkpoint. Whether the SM120 FP8 small-batch paths help the Intel W4A16 checkpoint is unknown.
- **vLLM 0.31.0 lowers the cost of a vLLM port** because lordhansolo's key ports are now upstream. The PLE-offload flag rename would break his launcher on 0.31. This only matters if a trace-replay benchmark still shows vLLM ahead.
- **A drafter tuned on ARC traces is the one serving lever two teams are pursuing that we have not tried.** With 10 streams at fixed width 4, a higher acceptance rate is the main way to raise decode tokens per step. No acceptance or score numbers are public.

### Gaps
- No public tok/s or acceptance-rate numbers for Penny Royal 3.0 on the Intel W4A16 + Albucino drafter checkpoint, or at C=10.
- No public measurement of tuned-drafter acceptance on ARC traces.
- No public NVFP4-KV vs FP8-KV token-matched ARC comparison (xz's NLL check is short-context only).

## Q3. Quantisation or checkpoint changes for Qwen3.8-Flash-Next (quality or speed on 96 GB)

### Takeaway
There are many new community quants in October, but none has an ARC-AGI-3 result. Two have measurable quality claims:
- **GPTQ NVFP4 "Strata"**: 45% of the expert-output error of NVIDIA's NVFP4. It only loads in the Strata fork, not in SGLang or vLLM.
- **local-inference-lab CSF-QAD**: losslessly compressed QAD scales (10-07). Its GPU inference and quality are untested by the author.

Swift 1.5, a reasoning-compressed Flash-Next fine-tune (09-24, not in the prior notes), claims 63% fewer thinking tokens. Our own evidence that stripping reasoning collapses the score argues for caution.

### Cited Findings
- **Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata** (10-09):
  - Every routed expert is in NVFP4 by GPTQ, calibrated on 57.5K tokens of the engine's own MoE inputs, with a block-scale search.
  - Same size and decode cost as NVIDIA's NVFP4 (63.3 GiB of experts).
  - Held-out expert-output error is 0.00561 rel-MSE, against 0.01252 for NVIDIA's MSE-calibrated checkpoint and 0.01684 for RTN.
  - "Only for Strata NVFP4 ... They do not load in transformers, vLLM or llama.cpp."
  - Author-reported. — [model card](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)
- **local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD** (updated 10-07):
  - QAD NVFP4 routed experts with losslessly compressed ("CSF") E4M3 block scales.
  - MXFP8 attention, NVFP4 MTP routed experts, NVFP4/MXFP8 vision.
  - Loads in vLLM via `modelopt_mixed`.
  - "GPU inference and multimodal quality evaluation have not been run for this checkpoint." — [model card](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
  - claidler repacks its BF16 `lm_head` (~1.2 GiB) into NVFP4 (0.21% relative error) — [claidler card](https://huggingface.co/claidler/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-LMHead-NVFP4)
- **ukisai/Swift1.5-Qwen3.8-Flash-Next** (09-24): a fine-tune claiming "63.4% fewer thinking tokens, with a 1.8x speed up while keeping the accuracy loss <1% vs base on xhigh", built by penalising "pathological overthinking" tokens and then RL/OPD. It is gated under a custom "swift-open-license-1.0". An NVFP4 variant exists. Vendor-reported. — [model card](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next); [NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4)
- Expert pruning remains negative on ARC. Son's team pruned a "336g line" on 10-02, and it "passed every public-set test [but] still scored 8.14 and 3.93 on Kaggle" — [docs/2026-10-05-lessons-from-tyser-ppo-postmortem.md](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-05-lessons-from-tyser-ppo-postmortem.md)
- The Penny Royal 3.0 benchmark checkpoint is "OrcaRouter Qwen3.8 Flash-Next ModelOpt NVFP4" — [release notes](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v3.0.0)
- Licence: two threads ask whether Qwen Community License 1.0 weights are prize-eligible under rule 5.a.3. As of 10-06 there was no organiser answer — [745079](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745079); [745837](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745837)

### Inferences
- None of these replaces the Intel W4A16 checkpoint without a matched local A/B. The GPTQ pack is the best-measured quality gain, but it is runtime-locked to Strata. The CSF-QAD pack is the same QAD weights the prior note already flagged, now with lossless scale compression, so it adds no quality.
- Swift 1.5 attacks token count, which matters given that throughput is the binding constraint. But the Duck harness relies on retained reasoning: our "strip old reasoning" test collapsed to 24.27, and dfranzen saw a "large regression" when removing it. Its licence also adds a prize-eligibility question on top of the Qwen one. Treat it as low priority.

### Gaps
- No Flash-Next quant has ARC-AGI-3 local or LB numbers since 10-01.
- Swift 1.5's evaluation tables (beyond the headline) were truncated in retrieval. Its agentic-benchmark deltas are not captured here.

## Q4. Known submission failure modes (score 0.00 in rerun, mount paths, image changes)

### Takeaway
Three new failure modes have appeared since 10-01:
1. The **1 October Kaggle image update to Python 3.13** breaks the cp312 Pennyroyal wheelhouse unless the notebook pins its original environment.
2. **Dataset mount failures** at save time (`ERRORED_MOUNTING_DATASET`).
3. **Dataset-version races** where a just-pushed notebook runs the previous dataset version.

Separately, the RTX PRO 6000 queue has been 6–17 h since 10-05. Submissions have also sat in PENDING for days.

### Cited Findings
- On 10-01 the default images became Ubuntu 24.04 / Python 3.13. Until 09-30 the default was the 2026-06-29 build (Ubuntu 22.04, Python 3.12). Vitalii Berezniak asked which image a scored rerun uses (pinned or current default); there was no organiser answer as of 10-03 — [745654](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745654)
- kozykappa's audited dfranzen copy states the consequence. The Pennyroyal v2.5.3 wheelhouse contains `cp312` native wheels, and "A Kaggle image using Python 3.13 cannot install wheels into Python 3.13". The notebook looks for an installed Python 3.12 and uses it only for the SGLang server venv. The fix it gives is to copy dfranzen's original notebook and set "Settings → Session options → Environment → Pin original environment". "Merely importing an `.ipynb` into a new notebook, or editing `language_info.version`, does not pin an older Docker image." It also adds a submission preflight that fails early on missing inputs or a wrong accelerator — [kozykappa audited submission](https://www.kaggle.com/code/kozykappa/arc-agi-3-flashnext-duck-audited-submission)
- CPMP (NVARC3) reported a save failure on 10-08: `mount data: ERRORED_MOUNTING_DATASET ... retry budget exhausted (30 attempts): rpc error: code = Internal desc = dataset loading failed` — [747438](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/747438)
- Dataset-version race (Son's team, 10-10): the notebook ran previous-version code and checkpoints because it was pushed seconds before the new version finished processing, and the status call said "ready". Their mitigation is an `EXPECT_DATASET_BUILD` marker checked in the setup cell before the server starts. They also note Kaggle unpacks uploaded `.gz` files, so the setup cell re-gzips them — [level-start-sprint doc](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)
- Son's team also reports that `kaggle kernels logs` returns nothing while a run is live, and that a wrong accelerator (P100) "fails within minutes". They check the card from the log afterwards — [same doc §5](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-09-level-start-sprint.md)
- Early completion: Son Pham had a run finish in ~4 h and immediately output a score (10-01). He asked whether this was a harness bug or VM preemption; it is unresolved — [744995](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744995)
- Queues and pending submissions: "waiting for ~9 hours" (10-05), ">6 hrs ... over 15 hrs" and "17 hours" (10-06), "queued for 9 hours" (10-07). Nick Pellegrin says it is not possible to run public-25 experiments. Penguin (10-08): "even submitted notebooks are taking longer now" — [745951](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745951). XIAO MA (10-11): submissions stuck in PENDING "for days" and unable to submit anything new — [748173](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/748173)
- Public scores on dfranzen's notebook page: the page showed 31.47, then 34.30, as "best score" while his LB stayed 27.89. xz ran it 4 times at "consistently 27-28". CPMP: "You don't inherit scores of people who fork your notebook" and "Public LB reflects your best submission". The cause is unexplained — [746010](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746010)
- I found no new report since 10-01 of the `/kaggle/input/<slug>` vs `/kaggle/input/datasets/<owner>/<slug>` path difference, or of a kernel scoring 0.00 in the scored rerun after a working save run. The superstringdev notebook does note that its competition reruns "obtain game list [from the] gateway" while ordinary runs use an offline filter, and lists unresolved TimeoutError/cleanup error-identity issues — [superstringdev](https://www.kaggle.com/code/superstringdev/arc-agi-3-nvfp4-sglang-public)

### Inferences
- **The Python 3.13 image change is the most urgent item for us.** Any new notebook created after 10-01 gets the 3.13 image by default, and our Pennyroyal wheelhouse is cp312. Our submissions must either be versions of a notebook pinned to the original environment, or launch the server from a real Python 3.12 venv. Moving to Penny Royal 3.0 would mean re-checking which Python its wheels target. Whether the scored rerun uses the pinned image is officially unconfirmed. The audited notebook's pin-and-preflight approach is the defensive default.
- Queue times of 9–17 h mean each local or public experiment now costs roughly a day. That argues for replay-style or level-start harnesses (Son's 30-minute sprint, our 25-game local runs elsewhere) over full public draws.

### Gaps
- No organiser answer yet on which Docker image the scored rerun uses (745654), or on Qwen licence eligibility (745079, 745837).
- No root cause is public for the 4-hour early completion or the notebook-page score display.
