# Models and serving delta for the offline ARC-AGI-3 agent, 2026-10-06 to 2026-10-12 (checked 2026-10-12)

Baseline: `research_notes/ARC AGI 3 SOTA October update/models_and_agent_research.md` and `harness_and_serving_techniques.md` (written 2026-10-11). Those notes already cover: Qwen 4 announced at Apsara (09-22) but unreleased; vLLM PR #53909 open; GLM-5.3-Flash, DeepSeek-V4.1-Flash, Kimi-K3 too big; Cloudflare clef, Underdog-Saluki, ISTA GSQ-RCO GGUF, Xing4.0; Penny Royal 3.0 (10-10); vLLM 0.31.0 (10-05); SGLang 0.5.21 (10-02); GPTQ-Strata NVFP4, CSF-QAD, Swift 1.5. Only what is new or changed is listed below. Every benchmark number is author- or vendor-reported; nothing was reproduced. Release status was read from the Hugging Face API and GitHub API on 2026-10-12.

Current stack, for reference: Intel/Qwen3.8-Flash-Next-W4A16-AutoRound + MTP drafter on SGLang Pennyroyal 2.5.3, one RTX PRO 6000 (96 GB, SM120), ~10 streams, ~130K context.

## Q1. Qwen 4 (or other Qwen) open-weight release status

### Takeaway
Unchanged: no Qwen 4 weights exist, and there is still no primary Alibaba date. The Qwen HF org added no LLM in the window (only Qwen-Image-2.1-Turbo, 10-09). The vLLM Qwen 4 kernel PR has not moved since 09-01. Nothing found signals a release before 10-25.

### Cited Findings
- Qwen HF org, newest repos: Qwen-Image-2.1-Turbo (updated 10-09), Qwen-Image-2.1 PE variants (09-20), Qwen-Image-2.1, Qwen-Drive-1.0-4B (09-02), then Qwen3.8-Flash-Next-FP8 and Qwen3.8-Flash-Next (Aug). No Qwen 4 repo — [HF Qwen org](https://huggingface.co/Qwen).
- QwenLM GitHub org: newest repos are D2K-Bench (09-29), RecreationWorld (09-18), Omnilingua-Bench (09-16), Qwen-Image-2.1 (09-14). No Qwen 4 repo (`QwenLM/Qwen4` returns 404) — [QwenLM](https://github.com/QwenLM).
- vLLM PR #53909 "[Kernel][Qwen] Add qwen4 fuse op" is still open, last updated 2026-09-01 — [vllm#53909](https://github.com/vllm-project/vllm/pull/53909).
- Test stubs `hf-tiny-v2/tiny-random-Qwen4Exp{ForCausalLM,ForConditionalGeneration,Model,TextModel}` appeared 10-06. `qwen4_exp` is the architecture name Qwen3.8-Flash-Next already uses, so these are test fixtures for the existing preview architecture, not a Qwen 4 signal — [tiny-random-Qwen4ExpForCausalLM](https://huggingface.co/hf-tiny-v2/tiny-random-Qwen4ExpForCausalLM).
- The only "release window" claims are still third-party blogs (Yotta Labs, orcarouter, cellcog), which say "October or early November" with no Alibaba source — [Yotta Labs](https://www.yottalabs.ai/post/qwen-4-27b-release-date-specs-hardware-what-is-known-2026); [orcarouter](https://www.orcarouter.ai/blog/qwen-4-leak-vllm-fuse-op).

### Inferences
- With the Kaggle deadline on 11-02, a Qwen 4 27B that lands after ~10-25 cannot be validated (W4 quant, SGLang/SM120 support, a 9-hour draw). Plan on Flash-Next as the final model.
- If Qwen 4 does appear, the reusable `qwen4_exp` plumbing (transformers stubs, vLLM Qwen4Exp paths) suggests it may load on existing Flash-Next code paths, but that is a guess from naming, not a stated fact.

### Gaps
- No primary Alibaba statement (blog, X, GitHub) on Qwen 4 timing, sizes or licence was found.

## Q2. Other new open-weight models that fit 96 GB and beat Qwen3.8-Flash-Next

### Takeaway
None. The window's model news is two huge announcements without weights (Mistral Large 4, 1.05T; Reflection AI Beam, 501B), MiMo-V2.6 checkpoint refreshes (310B), and one small coder (JetBrains Mellum2.1, 12B-A2.5B) that is weaker than Qwen3.5-9B on its own card. No new checkpoint from Qwen, Z.ai, DeepSeek, Moonshot, Google, NVIDIA or OpenAI.

### Cited Findings
- **Mistral Large 4 ("Le Chonk")**, announced 2026-10-06 as a preview API. 1.05T total / ~49-52B active MoE, 1M context. Vendor-reported DeepSWE v1.1 61.7%. Weights not on HF as of 10-12. Reported (secondary, citing VentureBeat) to arrive on 10-27 under a custom Mistral licence — [Mistral announcement](https://mistral.ai/news/mistral-large-4/); [MarkTechPost](https://www.marktechpost.com/2026/10/06/mistral-ai-releases-mistral-large-4-le-chonk-a-1-05t-parameter-open-weight-multimodal-moe/); [betanews](https://betanews.com/article/mistral-large-4-preview-open-weights/); [HF mistralai](https://huggingface.co/mistralai) (newest repos: Voxtral-Mini-4B-Realtime-Arabic, LIDstral-Arabic, 10-09). The Mistral page did not load through WebFetch, so the specs come from secondary coverage.
- **Reflection AI Beam**, announced 2026-10-05. 501B MoE, text-only, open weights promised with FP8 and NVFP4 builds, no release date, early access by sign-up. No ReflectionAI HF org — [technology.org](https://www.technology.org/2026/10/06/reflection-ai-beam-open-weight-model-501b/); [datacamp](https://www.datacamp.com/blog/reflection-ai-beam) (secondary; no primary Reflection page was fetched).
- **XiaomiMiMo MiMo-V2.6-Flash-RL / -MOPD and Pro-RL / -MOPD** were updated 10-09. They were created 09-21 and 09-27, so these are refreshes, not new models. MiMo-V2.6-Flash is 310.8B params (MIT, FP8) — [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL); [MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD). Community 2.3-bpw EXL3 of MiMo-V2.6-Flash exists (beamster, 10-07) — [beamster](https://huggingface.co/beamster/MiMo-V2.6-Flash-Sushi-2.3bpw).
- **JetBrains Mellum2.1-12B-A2.5B-Thinking** (repo created 09-20, card updated 10-07, Apache-2.0, 12.1B). Vendor-reported vs Qwen3.5 (9B): SWE-bench Verified 47.0 vs 50.0, SWE-bench Pro 28.0 vs 38.0, Terminal-Bench 2.1 17.4 vs 21.7, GPQA-Diamond 64.6 vs 77.8, AIME 25/26 83.3 vs 86.7 — [card](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking).
- No new LLM repos in the window from zai-org (newest GLM-5.3 family, Sep), deepseek-ai (DeepSeek-V4.1-Flash card touched 10-12, created 09-10), moonshotai (Kimi-K3), google (embeddinggemma-2, 10-06) or nvidia (robotics checkpoints only) — [zai-org](https://huggingface.co/zai-org); [deepseek-ai](https://huggingface.co/deepseek-ai); [moonshotai](https://huggingface.co/moonshotai); [google](https://huggingface.co/google); [nvidia](https://huggingface.co/nvidia).
- HF trending (10-12) LLM entries are all already known or derivatives: Qwen3.8-27B, Kolibri-1, Underdog-Saluki-27B, Xing4.0-29B-A4B, Cloudflare clef, ISTA GSQ-RCO, and a Qwen3.8-27B distillation merge (`nerkyor/Qwen3.8-27B-Coder390-...-MTP-DFlash2`) — [HF trending](https://huggingface.co/models?sort=trending).

### Inferences
- Mistral Large 4 and Beam would need ~2-bit weights to approach 96 GB (and Large 4 even then does not fit), so neither is a candidate even if released before the deadline.
- The model axis remains frozen on Flash-Next for the rest of the competition.

### Gaps
- No primary Reflection AI page or Mistral primary page text was successfully read; parameter counts and dates for both rely on secondary press.

## Q3. New SGLang / Pennyroyal / vLLM releases or features with measured gains

### Takeaway
Small. Penny Royal 3.0.1 (10-11) is a startup bugfix release, with no performance change over 3.0. Upstream vLLM and SGLang shipped no new release in the window (latest are vLLM 0.31.0 on 10-05 and SGLang 0.5.21 on 10-02). Open PRs relevant to Flash-Next spec decoding and HiCache exist but are unmerged. The one measured gain for this model class came from a community vLLM recipe on DGX Spark: weight-only NVFP4 GDN projections (+4-8% decode) and a retrained MTP drafter (+3-8% decode). See Q4.

### Cited Findings
**Penny Royal 3.0.1** (jpezzulli/sglang-rtxpro6000, tag `pennyroyal-v3.0.1`, 2026-10-11) — [release](https://github.com/jpezzulli/sglang-rtxpro6000/releases/tag/pennyroyal-v3.0.1)
- Fixes: RAM-PLE illegal memory access on WSL2 (device pointer from `cudaHostGetDevicePointer`, #64); TileLang kernel cache moved to the persistent cache directory (#65); NVMe-PLE reader compatibility hashes, where the stale 2.5.3 hashes had rejected the 3.0 attention backend at load (#66).
- Launchers print the Penny Royal version and source commit. "These fixes keep the same model and performance defaults." Container `ghcr.io/jpezzulli/sglang-rtxpro6000:v3.0.1`.

**Upstream releases**
- vLLM: latest release still v0.31.0 (2026-10-05) — [vLLM releases](https://github.com/vllm-project/vllm/releases).
- SGLang: latest release still v0.5.21 (2026-10-02) — [SGLang releases](https://github.com/sgl-project/sglang/releases).
- transformers v5.19.0 (2026-10-06) adds expert-parallel token dispatch for Qwen3 MoE and Mellum. Nothing found that is specific to Flash-Next serving — [v5.19.0](https://github.com/huggingface/transformers/releases/tag/v5.19.0).

**Open or closed vLLM PRs in the window touching Qwen4Exp / MTP / SM12x (none merged for Flash-Next)**
- #58863 "[Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN and the PLE short conv" (open) — [vllm#58863](https://github.com/vllm-project/vllm/pull/58863).
- #60861 "[Bugfix][Qwen4Exp] Carry checkpoint PLE short-conv state in align mode" (open, 10-09) — [vllm#60861](https://github.com/vllm-project/vllm/pull/60861).
- #60992 "[Perf][Model][Qwen4Exp] QSA decode FP8 e4m3 bytes four per register" (open, 10-10; targets sm_80) — [vllm#60992](https://github.com/vllm-project/vllm/pull/60992).
- #61018 "[Model] Qwen4Exp: support batch-sharded sampling" (closed unmerged, 10-10) — [vllm#61018](https://github.com/vllm-project/vllm/pull/61018).
- #61174 "[Bugfix][Spec Decode] Clear stale block-table rows for padded MTP decode" (open, 10-12) — [vllm#61174](https://github.com/vllm-project/vllm/pull/61174).
- #61178 "[Perf][Attention] FlashInfer prefill DiffKV attention on SM12x" (open, 10-12) — [vllm#61178](https://github.com/vllm-project/vllm/pull/61178).
- #61166 / #61066 NVFP4 global-scale bugfixes for MoE and compressed-tensors linears (open) — [vllm#61166](https://github.com/vllm-project/vllm/pull/61166); [vllm#61066](https://github.com/vllm-project/vllm/pull/61066).

**Open SGLang HiCache PRs (10-11 to 10-12)**
- #43727 "[HiCache] Export host Mamba pool occupancy next to host KV"; #43808 "[HiCache] Back up component state of backed-up nodes on write-back eviction"; #43778 "Support HiCache with KV Cache Sharding". All open, none Flash-Next specific — [sglang#43727](https://github.com/sgl-project/sglang/pull/43727); [sglang#43808](https://github.com/sgl-project/sglang/pull/43808); [sglang#43778](https://github.com/sgl-project/sglang/pull/43778).

### Inferences
- If we move to Penny Royal 3.x, take 3.0.1 rather than 3.0. The NVMe-PLE hash fix in particular matters if our launcher uses the NVMe PLE path. The prior note's caveat still stands: 3.0's headline +17% is C=1 adaptive MTP and will mostly not transfer to 10 streams.
- Nothing upstream in the window changes the vLLM-port question from the prior note.

### Gaps
- Still no public Penny Royal 3.x numbers on the Intel W4A16 checkpoint or at C≥8.
- Penny Royal 3.0.1's wheel Python ABI (cp312 vs cp313, relevant to the Kaggle image change) was not checked.

## Q4. New quantised checkpoints of Qwen3.8-Flash-Next or drafters on Hugging Face

### Takeaway
The one result worth acting on is a **retrained MTP drafter for quantised Flash-Next** (ursuciprian, D1 on 10-07 and D3 on 10-10). Its author retrained only the drafter's dense BF16 tensors with a KL loss to the served quantised target, on ~3.5M (D1) then ~10.7M (D3) anchors of the model's own outputs. Live acceptance at draft positions 1-4 rose from 0.82/0.66/0.54/0.43 to 0.86/0.72/0.60/0.50, and decode rose +3-8% at 4-8 concurrent requests on DGX Spark (vLLM). It is the first public number for the "tune the drafter on our own traces" lever flagged in the prior note. The same checkpoint's weight-only NVFP4 GDN projections add +4-8% decode. Everything else new is NVFP4/MLX/EXL3/GGUF derivatives without agentic numbers. The Intel W4A16 checkpoint we run is unchanged since 08-31.

### Cited Findings
**ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE** (created 10-05; revision 2 with drafter D1 on 10-07, revision 3 with D3 on 10-10; Qwen Community License 1.0) — [card](https://huggingface.co/ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE)
- Base: local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (QAD NVFP4 experts, MXFP8 GDN) at revision `7c4f1bc1`. Two changes:
  - the 108 GDN projection tensors (`in_proj_qkv`, `in_proj_z`, `out_proj` × 36 layers) are stored as weight-only NVFP4 with an MSE block-scale search (relative error 0.0877 vs 0.0937 absmax);
  - the 24 dense BF16 `mtp.*` drafter tensors are retrained.
- GDN change, paired screen on one DGX Spark, temperature 0: decode at 8 requests +4.4% (noise 1.9%); 16K context at 4 requests +5.9% (3.0%); counting task at 8 requests +7.8% (2.1%); tg512 at 1 request 50.0 → 59.7 tok/s; prefill pp2048 −0.9% (1.0%). Prefill keeps an MXFP8 copy for calls of ≥41 rows (`VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS=41`, +1.1 GiB). Quality gate: hard-mode suite 91/100 (base runs 86-93), tool calling 100/100, 128K retrieval 20/20.
- Drafter D1 (10-07): ~3.5M tokens of the recipe's own outputs (agentic, chat, code, math, tool), KL to the served target's top-20 distribution at 6 chained depths, 72 steps of 8,192 anchors on one DGX Spark. Training emulates the served FP8 drafter KV cache.
  - Offline T=0 acceptance by position, original → D1: 0.878→0.905, 0.750→0.797, 0.635→0.699, 0.534→0.614, 0.449→0.542, 0.377→0.480.
  - Live, two Sparks, positions 1-4: 0.819→0.856, 0.661→0.713, 0.535→0.593, 0.431→0.489 (dgx-01; dgx-02 similar).
  - Decode speed vs original: +4.9% / +7.9% at 4 requests, +6.8% / +8.1% at 8 requests, +5.5% / +5.7% at 16K context with 4 requests (dgx-01 / dgx-02; noise bands 2.6-3.6%).
- Drafter D3 (10-10): continued from D1 on ~10.66M anchors (11.9× D1), one pass. Offline +2.2 points per position on average. Live positions 1-4 on dgx-01: 0.849→0.860, 0.699→0.720, 0.575→0.598, 0.472→0.498. Decode vs D1: +3.3-3.4% at 4-8 requests on dgx-01 (beyond noise), inside noise on dgx-02. Quality gate 93/100.
- Served via the recipe repo [ursuciprian/qwen3.8-flash-next-1x-dgx-spark](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark) v2.2.0 (10-10). It needs a vLLM build with the "b12x" W4A16 NVFP4 linear backend (vLLM `5dad364d`, b12x `21e0b201`), pinned in image `ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-...` — [v2.2.0 release](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark/releases/tag/v2.2.0).
- Mirror: Kanposer/Qwen3.8-Flash-Next-NVFP4-GDN-MSE (created 10-12) is a re-upload with the same card — [Kanposer](https://huggingface.co/Kanposer/Qwen3.8-Flash-Next-NVFP4-GDN-MSE).

**claidler/Qwen3.8-Flash-Next-NVFP4-MXFP8-QAD-LMHead-NVFP4-GDNMSE** (10-11) — [card](https://huggingface.co/claidler/Qwen3.8-Flash-Next-NVFP4-MXFP8-QAD-LMHead-NVFP4-GDNMSE)
- The local-inference-lab NVFP4 base, plus an NVFP4 `lm_head`, plus ursuciprian's retrained drafter D1 (rev `16c9bd54`) swapped in. Supersedes claidler's CSF variant (CSF does not load in vLLM 0.30).
- Measured on 2× DGX Spark TP=2 (vLLM 0.30, K=5, marlin MoE), on its abliterated twin with the same drafter: spec-decode acceptance 39.11% vs 38.11%; decode 81.7 vs 75.2 tok/s median (+8.6%); TTFT 117 vs 121 ms; 120K needle 3/3.

**Other new Flash-Next quants in the window (no agentic or ARC numbers)**
- Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata and -Q8_0-Strata (10-11): an NVFP4 + Q8_0 mix at 105 GiB, which needs a 128 GB-RAM PC. Its own quality table: 95.0% vs 93.6% for their NVFP4-GPTQ (69 GiB) and 92.9% for NVIDIA NVFP4 (69 GiB). Strata runtime only — [card](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata).
- Jundot/Qwen3.8-Flash-Next-oQ{2.5e…8e}-mtp and -fp16-mtp (10-11 to 10-12): oMLX (Apple MLX) mixed-precision quants with the MTP head; oQ4e is 4.78 bpw, ~100 GiB including the n-gram table — [oQ4e-fp16-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-fp16-mtp). Not usable on SGLang or vLLM.
- beamster/Qwen3.8-Flash-Next-Sushi-{2,2.6,4}bpw (EXL3/Apple, updated 10-07) — [Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw).
- Uncensored/abliterated W4A16, NVFP4, GGUF and EXL3 variants (mikechambers84, beginning-ai, huihui-ai, gdeyoung and others, 10-10 to 10-12) and Swift 1.5 derivatives (wallawalla47 NVFP4-Dense8, poisonxa PXQN-32GB) — [HF search "Qwen3.8-Flash-Next"](https://huggingface.co/models?search=Qwen3.8-Flash-Next&sort=created).
- Intel/Qwen3.8-Flash-Next-W4A16-AutoRound: unchanged since 08-31 (sha `4c67bf68`); no new Intel Flash-Next repo — [Intel](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound).

### Inferences
- **The drafter refit is the transferable recipe, not the checkpoint.** It targets a different base (QAD NVFP4) and runtime (vLLM b12x on SM121 DGX Spark), so the D3 weights will not drop into our Intel W4A16 + Pennyroyal stack. The method is cheap and backend-agnostic, though: retrain only the dense `mtp.*` tensors with a KL loss to the *served quantised* target on its own outputs, emulating the served drafter KV dtype. On our own ARC traces (long agentic contexts, repetitive action/JSON patterns) the acceptance gain may be larger than on the author's general mix. At our fixed draft width of 4 with ~10 streams, +0.05 acceptance at positions 2-4 is roughly the +5-8% decode the author saw at 4-8 requests. This matches what Son's team appears to be doing privately (`flashnext-mtp-drafter-tuned`, prior note).
- The GDN weight-only NVFP4 trick (+4-8% decode, half the bytes for small-row GEMMs) needs a W4A16 NVFP4 linear kernel in the serving stack. Whether Pennyroyal's GDN path can use one on SM120 is unknown. It is a lower priority than the drafter.
- Neither gain was measured at ~130K contexts with 10 streams, and both come from a single author on different hardware (DGX Spark, unified memory).

### Gaps
- No ARC-AGI-3 or Kaggle result for any checkpoint listed here.
- No measurement of the retrained drafter at contexts above 16K, or on SGLang.
- Whether ursuciprian's training scripts are public was not checked (the card describes the method; the recipe repo pins the image).
