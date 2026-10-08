# Open-weight models, quantizations and decode kernels for one RTX PRO 6000 (SM120, 96 GB), as of 2026-10-06

Scope: can any model or kernel change beat Qwen3.8-Flash-Next (125B total / 6B active, 51B n-gram table, MTP) served as Intel W4A16 AutoRound or mixed NVFP4+FP8, on agentic reasoning per unit of decode throughput, for an offline 9 h / 110-game ARC-AGI-3 Kaggle run (deadline 2026-11-02).

## Q1. Qwen 4 (and its 27B open tier) status; other Sept-Oct 2026 open models that fit 96 GB

### Takeaway
Qwen 4 is announced, not released: no weights, specs, licence, benchmarks or date for any tier, including the 27B. No other open model released Sept-Oct 2026 that fits 96 GB has published numbers beating Qwen3.8-Flash-Next on reasoning; the strong new releases (DeepSeek-V4.1-Flash, GLM-5.3) are far too large.

### Cited Findings
- Qwen 4 was announced at Apsara on 2026-09-22 with four tiers (Max, Plus, Flash, 27B open-weights). Status: "in training", release "very soon". No date, licence, architecture, context or benchmarks were given for the 27B — [Yotta Labs, 2026-09-23](https://www.yottalabs.ai/post/qwen-4-27b-release-date-specs-hardware-what-is-known-2026); [Yotta Labs, Qwen 4 overview](https://www.yottalabs.ai/post/qwen-4-release-date-what-is-known-how-to-prepare-2026)
- The October / early-November release window "traces back to [nothing] Alibaba has published"; it comes from X watchlists and tracker pages. vLLM PR #53909 ("qwen4 fuse op") was still open at the time of the post. PLE-offload work merged on Sept 9 and Sept 25, and the config was upstreamed to Transformers on Oct 2 — [orcarouter blog](https://www.orcarouter.ai/blog/qwen-4-leak-vllm-fuse-op)
- The Qwen org on Hugging Face has no Qwen 4 repo. The newest LLM repos are Qwen3.8-Flash-Next / -FP8 (Aug 24), Qwen3.8-27B (Aug 5, Apache-2.0) and Qwen3.8-2.4T-A95B (Aug 8). Releases since then are image, ASR and driving models only — [HF Qwen org listing, sorted by creation date](https://huggingface.co/Qwen)
- The Qwen3.8-Flash-Next model card labels it a preview of the Qwen 4 architecture (HF arch tag `qwen4_exp`). Licence: `qwen-community-1.0`. Context: 262K native — [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); [Yotta Labs](https://www.yottalabs.ai/post/qwen-4-release-date-what-is-known-how-to-prepare-2026)
- Model-card benchmarks, Flash-Next vs Qwen3.8-27B: SWE-bench Pro 62.5 vs 61.7; DeepSWE 58.7 vs 42.2; Toolathlon 73.5 vs 67.1; JobBench 55.7 vs 33.4; GPQA-D 91.7 vs 89.2; LCB v6 91.9 vs 90.3; HLE 35.9 vs 30.8. These are vendor claims — [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Aleph-Alpha Kolibri-1 (released 2026-10-03, Apache-2.0): 78B MoE with 3.46B active, about 78 GB in FP8, 1M context, German/English focus. Claims GPQA-D 84.3 and AIME-2026 96.0, which is below Flash-Next's claimed GPQA-D 91.7. It ships a custom `kolibri1` arch in vLLM — [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
- DeepSeek-V4.1-Flash (MIT, Sept 10) is 763B parameters. GLM-5.3 (Aug 25) is a large `glm_moe_dsa` MoE that the community runs at 3.0 bpw EXL3. Neither fits one 96 GB GPU at usable quality — [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash); [HF trending listing](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)
- Xing4.0-29B-A4B (XingChen-AGI, Sept 17, Apache-2.0, custom code) is trending, but I did not open its benchmarks — [XingChen-AGI/Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B)

### Inferences
- Unless Qwen 4 27B ships in the next ~2 weeks, it cannot realistically be integrated, validated and submitted before 2026-11-02. Treat it as a contingency, not a plan.
- Kolibri-1 is the only new release that fits comfortably in 96 GB with KV headroom (3.46B active). Its claimed reasoning scores sit below Flash-Next, and nothing shows it is better at ARC-like tasks. It is not worth a slot.

### Gaps
- I did not inspect the benchmark tables of Xing4.0-29B-A4B, IQuest-Q1 or Naive-N0.5-Flash.
- No ARC-AGI-3 results exist for any of the newer models.
- I could not read the Kaggle discussion thread itself; the fetch returned only the page title.

## Q2. New quantizations of Qwen3.8-Flash-Next that run on SM120

### Takeaway
Several downloadable SM120-validated checkpoints exist. The most relevant new ones are:
- **local-inference-lab NVFP4+MXFP8 QAD** (~98 GiB, distilled): its quality claims are the strongest.
- **primitive-ai NVFP4**: single RTX PRO 6000 validated, with BF16 n-gram table offloaded to host.
- **RadixArk NVFP4**: the basis of the fastest SM120 SGLang numbers.

ISTA's GSQ-RCO is GGUF/llama.cpp only, so it does not apply to a vLLM/SGLang stack.

### Cited Findings
- **primitive-ai/Qwen3.8-Flash-Next-NVFP4** (Sept 2026), validated on RTX PRO 6000.
  - Recipe: routed experts in NVFP4 (group 16), weights-only RTN; the 51.2B n-gram table stays BF16; attention, shared experts and MTP are BF16. 186 GB on disk; needs ~100 GB host RAM for the n-gram table.
  - Speed (vLLM, 8K in / 512 out): 74.4 tok/s single-stream, 483.8 tok/s at c=32, TTFT ~570 ms. MTP with 3 spec tokens reaches 142.6 tok/s (+56%); 1 spec token hangs boot.
  - Quality: tool-call accuracy 84.6, abstain accuracy 56.7.
  - Known issues: PLE-offload startup race (overlay fix shipped); grammar + MTP can log FSM errors.
  - [primitive-ai/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4)
- **local-inference-lab/Qwen3.8-Flash-Next-NVFP4**: routed experts and n-gram tables in NVFP4, shared experts and attention in MXFP8, trained with quantization-aware distillation (QAD) against a BF16 teacher.
  - Size: ~98 GiB on disk. Targets a single RTX 6000 with PLE offload.
  - Claims GPQA-D 89.9, and that it "scored higher than NVFP4 PTQ while using 9% fewer tokens on average".
  - No tok/s published.
  - [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
- **nvidia/Qwen3.8-Flash-Next-NVFP4** (Sept 2): routed experts W4A4 NVFP4 with MSE-calibrated scales; attention, shared experts and the rest stay BF16; ~2.7x smaller than BF16. Its published speed (540 tok/s at bs=1 with MTP, accept length 3.3) is for TP4 on B200, not SM120 — [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
- **RadixArk/Qwen3.8-Flash-Next-NVFP4** (Aug 25) is the checkpoint behind the fastest single-GPU SGLang SM120 numbers (see Q3) — [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- **Inferact/Qwen3.8-Flash-Next-NVFP4** (modelopt, Aug 26) exists; I did not inspect its card — [Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
- **Intel/Qwen3.8-Flash-Next-W4A16-AutoRound** (the current stack): `linear_attn`, `self_attn`, embeddings, `lm_head` and vision stay unquantized. It averages 0.8332 vs 0.8362 for BF16 (99.64%) on GSM8K/MMLU/PIQA/HellaSwag. No speed numbers — [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
- **Qwen/Qwen3.8-Flash-Next-FP8** is 172.8 GiB, too big for one GPU without offload. On 4x RTX PRO 6000 without NVLink, TP2 beats TP4 for a single user (81.45 vs 64.61 tok/s). The same post's NVFP4 attempt fit in 74.75 GiB but failed server init — [kubesimplify, 2026-08-27](https://blog.kubesimplify.com/running-qwen3-8-flash-next-on-dgx-spark-and-rtx-pro-6000)
- **ISTA-DASLab GSQ-RCO** (Sept 7; Coder variant with expert pruning Sept 26): GGUF only, 2.2M downloads — [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- **REAP expert-pruned variants** (288 and 320 of 512 experts) exist only as MLX and GGUF — [sh0wie REAP-288 MLX](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit); [AnonimousA REAP-320 GGUF](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF)
- **ExLlamaV3 EXL3 4.05 bpw** is an alternative non-vLLM backend on one RTX PRO 6000 (numbers in Q3) — [MarcoPizeta bench](https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench)

### Inferences
- The ~98 GiB QAD checkpoint with the n-gram table also in NVFP4 is the most promising untested swap. It shrinks the on-GPU footprint, and therefore grows KV capacity, more than the 126 GB checkpoint used in a Duck run (Q4). It also claims a quality gain over PTQ and 9% fewer tokens.
- These claims are unverified by third parties. A head-to-head against the AutoRound W4A16 on the local harness would be needed before submitting.
- 177 GB host RAM covers primitive-ai's ~100 GB host-side n-gram table. It is tight if other host buffers grow.

### Gaps
- No published MXFP4 checkpoint or "ISTA MoESQ" vLLM checkpoint for Flash-Next was found.
- No independent quality eval compares AutoRound W4A16, NVFP4 PTQ and NVFP4 QAD on the same reasoning or agentic suite.

## Q3. Decode-speed improvements on SM120 since mid-September 2026

### Takeaway
The biggest measured single-GPU SM120 gains come from community SGLang forks: Flash-Next NVFP4 with 3-step native MTP reaches ~180-193 tok/s single-stream. Stock vLLM measures ~100-143 tok/s single-stream. vLLM v0.30.0 (2026-09-22) added Flash-Next-specific kernels and SM120 NVFP4 work but published no SM120 tok/s.

### Cited Findings
- **vLLM v0.30.0 (2026-09-22)**, as summarised by a third-party release digest:
  - Dedicated Triton kernels for Flash-Next: separate prefill and decode QSA indexer paths, fused PLE kernels, an FP8 indexer cache, padded-index skipping in sparse GQA, and fused PLE residual and QSA output gate.
  - CPU/UVA PLE offload via `--engram-config`.
  - Removed `torch.compile`, which had consumed ~50 GB extra and caused OOM; FP8 now loads on a single GPU.
  - "W4A4 NVFP4 on SM120/121" prioritised. Init time fell from 28.9 s to 8.2 s (graph capture from 12 s to 2 s).
  - Sources: [localmodelwatch digest](https://localmodelwatch.tsuchitsuchi.com/en/2026/09/22/vllm-v0-30-0-released/); [vLLM releases](https://github.com/vllm-project/vllm/releases)
- **vLLM nightlies, early September:** the SM120 selector picks Marlin by default for W4A16. RecoverSSM, an accepted-state recovery path for MTP verification on hybrid SSM/linear-attention layers, was built for Flash-Next on SM120 — [search summary of vLLM sources](https://github.com/vllm-project/vllm/releases). This is from a search snippet; I did not verify it against the primary PR.
- **SSHdotCodes SGLang 0.5.20 build** (RadixArk NVFP4; NEXTN MTP with 3 steps and 4 verify tokens; FP8 E4M3 KV with 524,288 tokens; 262K context; max 2 concurrent requests; ~50 GiB host RAM for the PLE table):
  - 24 Sept release: 179.4-192.7 tok/s single-stream, +13-17% over its Sept 19 build. Verify step went from 68.1 to 79.1.
  - Custom decode kernels for batch 1-8 and exact rejection sampling.
  - [SSHdotCodes/qwen-3.8-flash-next-pro6000](https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000)
- **h3po/wojciak SGLang SM120 fork** (FP8 E4M3 KV, FR-Spec 65,536-token draft vocab):

  | Model | Weights | Speculative decoding | bs=1 (1024 tok) | 4 concurrent | Measured |
  |---|---|---|---|---|---|
  | Flash-Next | NVFP4 | native NEXTN MTP, 4 tokens | 181.72 tok/s | 446.49 tok/s | Sept 9 |
  | Qwen3.8-27B | FP8 | DFlash2, 8 tokens | 108.31 tok/s | 374.98 tok/s | Sept 10 |

  [h3po/sglang-rtxpro6000](https://github.com/h3po/sglang-rtxpro6000); [wojciak fork](https://github.com/wojciak/sglang-rtxpro6000)
- **MarcoPizeta bench** (one RTX PRO 6000, 64 GB host RAM):
  - vLLM (`qwen38-flash-next` image built 26/08) with primitive-ai mixed NVFP4-FP8 and MTP: 101.6 tok/s at 1k c=1, 296.4 tok/s at 1k c=4, 140.5 tok/s at 8k c=1, 98.9 tok/s at 32k c=1.
  - EXL3 4.05 bpw with MTP: 168.9, 221.4, 129.1 and 63.1 tok/s for the same four settings.
  - vLLM prefill ~28k tok/s flat from 8k to 128k; EXL3 prefill 10.3k tok/s at 8k.
  - MTP adds +30% on vLLM and +60% on EXL3 at c=1.
  - KV capacity: 217k tokens with MTP vs 503k without.
  - [MarcoPizeta/flash-next-rtxpro6000-bench](https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench)
- **kubesimplify** (FP8, TP4 on RTX PRO 6000): MTP lifted real-prompt decode from 49.90 to 124.93 tok/s at 55.3% acceptance. At c=32, MTP lowered aggregate throughput (805 to 694 tok/s) — [kubesimplify](https://blog.kubesimplify.com/running-qwen3-8-flash-next-on-dgx-spark-and-rtx-pro-6000)
- **Other runtime numbers:** one SGLang fork streams the FP8 n-gram table from SSD and reports 164.7 tok/s single-stream with MTP on one RTX PRO 6000 (search snippet; source not opened) — [search results](https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000). The SGLang day-0 blog is LMSYS, 2026-08-26 — [LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)

### Inferences
- **What MTP is for:** it is a single-stream lever. Under high concurrency it can lower aggregate throughput, and it more than halves KV capacity (217k vs 503k tokens). For a 110-game parallel agent loop that is KV-bound (see Q4), MTP depth should be tuned against concurrency, not maximised.
- **SGLang forks vs stock vLLM:** the forks are ~1.3-1.8x faster per stream at low concurrency. They are capped at 2-4 concurrent requests in their published configs and rely on custom patches. Risky to port into a Kaggle kernel within 4 weeks.
- **NVFP4 KV cache:** I found no evidence of NVFP4 KV cache for Flash-Next on SM120. All measured configs use FP8 E4M3 KV, and the MXFP8 KV path cited in v0.30 is for DeepSeek-V4.1.

### Gaps
- No primary-source measured tok/s for vLLM v0.30.0 on SM120 with Flash-Next.
- No measured gain from v0.30's fused QSA/PLE kernels on RTX PRO 6000.
- No data on MTP depth beyond 3 on SM120.
- No primary vLLM PR numbers or GitHub links were collected for RecoverSSM or the SM120 NVFP4 MoE kernel.

## Q4. Does a smaller dense model (Qwen3.8-27B) or another MoE do better per wall-clock in agent loops?

### Takeaway
Measured evidence is mixed:
- Flash-Next decodes ~1.2-1.7x faster than Qwen3.8-27B on the same SM120 GPU.
- The 27B scores higher on one independent tool-calling bench and passes forced tool calls that Flash-Next on vLLM fails.

The ARC-AGI-3 field (Milestone 2 winners) runs Flash-Next. The documented bottleneck in a Duck/Flash-Next run was KV capacity and serving, not model choice.

### Cited Findings
- **Decode speed** on one RTX PRO 6000 (same fork, same week): Flash-Next NVFP4 at 181.72 / 446.49 tok/s (bs=1 / c=4) vs Qwen3.8-27B FP8 with DFlash2 at 108.31 / 374.98 tok/s — [h3po/sglang-rtxpro6000](https://github.com/h3po/sglang-rtxpro6000)
- **Tool-calling quality** (tool-eval-bench):
  - Qwen3.8-27B NVFP4 scored 91.0 ± 1.5 (Pass@8 96.6%). Flash-Next scored 87.9 ± 1.9 on vLLM (Pass@8 89.8%) and 87.4 ± 1.5 on EXL3.
  - With `tool_choice="required"`, Flash-Next on vLLM failed 0/8 in every run; the 27B on SGLang passed 8/8.
  - [MarcoPizeta bench](https://github.com/MarcoPizeta/flash-next-rtxpro6000-bench)
- **Vendor-claimed agentic benchmarks** favour Flash-Next over 27B, e.g. DeepSWE 58.7 vs 42.2 and Toolathlon 73.5 vs 67.1 — [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Duck/Flash-Next validation run post-mortem** (2026-09-22):
  - The 126 GB checkpoint left a 5 GiB KV cache. vLLM ran ~3 requests while ~21.6 waited, at ~236 tok/s aggregate.
  - Each game got only ~42 LLM calls in a 2 h window. Prefix caching was disabled.
  - Local score was 10.02 against a target of 18-20.
  - [Hisernberg/arc-agi-3 PR #1](https://github.com/Hisernberg/arc-agi-3/pull/1)
- **Field and memory:** per the same search summary, Milestone Prize 2 winners run Tufa's Duck harness and serve Flash-Next, using quants including INT4 MTP, NVFP4+FP8 mixed and AutoRound W4A16 — [search summary citing Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792). Secondary: I could not read the thread itself. On Spark/GB10 the community runs Flash-Next with offload — [NVIDIA dev forum](https://forums.developer.nvidia.com/t/qwen3-8-flash-next/381228?page=4)

### Inferences
- **Per wall-clock:** for a throughput-bound 110-game loop, the 27B's tool-calling edge is small (91.0 vs 87.9, overlapping-ish CIs) against a ~19% aggregate decode deficit at c=4. It would need to be clearly better on ARC-style multi-step play to win, and there is no ARC-AGI-3 evidence either way.
  - The 27B's practical advantage is memory. At ~14-27 GB of weights (NVFP4/FP8), it leaves far more KV room for many concurrent games.
  - This matters because the documented failure mode is KV starvation: 5 GiB of KV and 3 running requests.
- **Highest-leverage change on the current stack:** shrink the on-GPU footprint (QAD NVFP4+MXFP8 at ~98 GiB, n-gram table offloaded to host via `--engram-config`/UVA), enable prefix caching, and tune MTP depth vs `max-num-seqs`. This is likely worth more than a model swap. It is consistent with the prior finding that KV capacity is the portable lever.
- **Forced tool calls:** if the harness relies on `tool_choice="required"`, Flash-Next on vLLM is a correctness risk to test explicitly.

### Gaps
- No head-to-head of Flash-Next vs Qwen3.8-27B (or any other model) on ARC-AGI-3 or a similar interactive game loop at matched wall-clock.
- No measured aggregate throughput at high concurrency (≥16) for the 27B vs Flash-Next on one SM120 GPU.
- No public data on how winners configured KV and concurrency inside the 9 h kernel.
