# vLLM (lordhansolo e975732 overlay) vs SGLang (dfranzen Pennyroyal 2.5.3) for dfranzen's harness, as of 2026-10-06

Scope: should we move dfranzen's harness from SGLang Pennyroyal 2.5.3 (Intel W4A16 + Albucino MTP draft, FP8 KV, 10 streams) to lordhansolo's vLLM main `e975732` runtime plus overlay (primitive-ai mixed NVFP4+FP8, built-in NVFP4 MTP)? This note covers whether it would raise throughput and what could break. Target hardware is one RTX PRO 6000 (SM120, 96 GB), 177 GB host RAM, offline, 9 h, 110 games.

Local copies of the files read (all small text, downloaded read-only):
- `scratchpad/lhs/v3/`: `lordhansolo/vllm-main-e975732-arc3`, its README, VALIDATION, PATCH_IDENTITY, runtime-manifest and applier
- `scratchpad/lhs/v2/`: `lordhansolo/arc3-vllm-e975732-gdn-recoverssm`, its README and VALIDATION
- `scratchpad/lhs/src/`: from `lordhansolo/taaf-kaggle-source`, `inference.json`, `framework/kaggle.py`, `openai_compat.py`, `vllm_metrics.py`, `cache_diagnostics.py` and `setup_commands.json`
- `scratchpad/thk/COMPARISON.md`: Tong Hui Kang's side-by-side

## 1. What exactly is in lordhansolo's stack

### Takeaway
It is the vLLM nightly `0.29.1rc1.dev573+ge97573215`, shipped as six Docker image layers plus a hash-pinned 32-file Python overlay. The overlay carries 9 upstream PR ports and about 16 custom fixes, and the model runs as `qwen4_exp`. **The scored 23.84 run did not use the dataset named in the brief.** It used the earlier `arc3-vllm-e975732-gdn-recoverssm` v2 runtime. The newer `vllm-main-e975732-arc3` v3 has, in the author's words, not beaten that score in any run. Every v3 change is validated only on an RTX PRO 4000 laptop and has never been benchmarked on the RTX PRO 6000.

### Cited Findings
- The notebook header says the Milestone 2 run (v293) used `taaf-kaggle-source v297`, the runtime `arc3-vllm-e975732-gdn-recoverssm v2` and the model `qwen3-8-flash-next-mixed-nvfp4-fp8 / hf-mixed-mtp-nvfp4 v1`. The current notebook uses `vllm-main-e975732-arc3 v3`: "It should be faster and have a larger KV cache pool than the runtime used in the highest-scoring run. Unfortunately, no run with it has beaten that score so far." — [lordhansolo notebook](https://www.kaggle.com/code/lordhansolo/built-on-tufa-labs-duck-harness-milestone-2) (local copy `scratchpad/forks/lordhansolo/*.ipynb`)
- Baseline per the v2 README: vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19` (image `vllm/vllm-openai:nightly-e975732…`, `0.29.1rc1.dev573+ge97573215`). Serving is "C14, MTP3, BF16 GDN state, prefix cache align (unit 128, retention 0), `max_num_batched_tokens` 2048, `max_model_len` 123392, FP8 KV, Triton GDN decode, Model Runner V2, async scheduling". — [arc3-vllm-e975732-gdn-recoverssm README](https://www.kaggle.com/datasets/lordhansolo/arc3-vllm-e975732-gdn-recoverssm)
- v3 adds three things to the GDN bundle:
  - exact CUDA graphs at 44 and 52 tokens for 14 sequences with MTP3, with an SM120 GEMM at M52
  - ports of vLLM #58489 (PLE prefetch ids), #58961 (QSA profiling KV), #58430 (allocator split during `profile_run`), #58114 (PLE metadata without a GPU sync), #58434 (one-token prompt tail under MTP), #58784 (rejection of padded draft slots) and #58720 (expert mapping in `load_weights`)
  - overlay SHA256 `e3a6fe0d…`, 32 files

  The README says: "Neither the KV pool delta nor the throughput gain has been measured on the RTX PRO 6000." VALIDATION says the rc1/rc2 tests "passed 205 of 205" on the e975732 runtime, and "No RTX PRO 6000 server benchmark or competition wave has been run for this bundle." — [vllm-main-e975732-arc3 README/VALIDATION](https://www.kaggle.com/datasets/lordhansolo/vllm-main-e975732-arc3)
- **GDN RecoverSSM** (`VLLM_ARC3_GDN_RECOVERSSM=1`, on by default in the launcher) changes how MTP drafts are verified. Stock vLLM writes per-draft recurrent states. RecoverSSM instead writes small per-row records and replays only the accepted tokens. This frees "9 blocks per running request", projected at "≈ 123 blocks ≈ 2.62 GiB, about 13% of the 923-block pool" at C14. Bit-exactness against the stock kernel is shown on a laptop GPU and on a tiny dummy-weight model. When on, the startup validator **rejects KV connectors (P/D, offload)**, Model Runner V1, TP>1 and `mamba_cache_mode` `all`. — [gdn-recoverssm README/VALIDATION](https://www.kaggle.com/datasets/lordhansolo/arc3-vllm-e975732-gdn-recoverssm)
- Model: `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 @ 07915ee`, mirrored as `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1` "with the repo's mtp_nvfp4/ overlay already flattened into the root, so the NVFP4 MTP head is what loads". — `taaf-kaggle-source/src/ARC3-Inference/inference/framework/kaggle.py` L21–24
- Checkpoint layout per the primitive-ai card:
  - routed experts: NVFP4 group-16
  - QSA attention (12 layers) and GDN projections (36 layers): FP8 E4M3 per-channel; the GDN projections need `VLLM_GDN_DECODE_KERNEL=triton`, because "the default CUDA kernel deterministically stalls" at about 32 concurrent requests
  - n-gram/PLE table: BF16 (51.2B params), offloaded to host
  - checkpoint size: 183.7 GB

  Accuracy is at "parity" with BF16: 90.3% overall on 1,370 items, "within the ±1.0 tie band" of plain NVFP4. — [primitive-ai model card](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8)
- The harness is a forked rewrite of ARC3-Inference (commit `ca1bd02`, branch `solution-improve-qwen38-flash-next`), 40 files, +5,246/−2,337. The overlay is "pinned by SHA256 and installed by an applier that refuses modified targets". — [COMPARISON.md](https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/COMPARISON.md)

### Inferences
- "lordhansolo's stack" is two different runtimes. The 933 tok/s and 1.42M-token figures come from the Save & Run of 2026-09-30 13:11 UTC (taaf-kaggle-source v305). By the dates, that is the v3 runtime (uploaded 09-30 11:38), not the runtime behind the 23.84. The best-scoring runtime and the best-throughput runtime are not the same artifact.
- Porting means taking a 1,475-line Kaggle launcher, a hash-pinned image-layer runtime (~8 GB of layer blobs) and a 184 GB model mirror. Any vLLM upgrade breaks the pins by design.

### Gaps
- `PATCH_README.md` was not downloaded separately; the README content is identical in size (5,273 B), so it is most likely the same file.
- lordhansolo's Kaggle write-up (`kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2`) needs JS and was not read.
- The full list of the "16 custom fixes" was taken from COMPARISON.md and not re-verified line by line in `PATCH_IDENTITY.json`.

## 2. Server launch flags, boot time, and where 933 vs 588 tok/s come from

### Takeaway
The two headline numbers are **not comparable**:
- 933 is lordhansolo's own harness on 25 public games, 14 at once, with ≤81.5k-token prompts and 66-token board images.
- 588 is dfranzen's harness on 10 games for 25 min, with 60k–118k prompts and ~402-token images.

Both are "total generated tokens / whole notebook wallclock including boot". Per stream, SGLang/dfranzen was actually faster: ~95 vs ~81 tok/s at peak. The vLLM gain in aggregate tokens comes mainly from more concurrent streams that fit because of the larger pool and the shorter contexts.

### Cited Findings
Launch command (lordhansolo `kaggle.py` `build_vllm_server_command`, values from `configs/inference.json`):
- Core: `vllm serve <model> --served-model-name primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 --load-format safetensors --dtype bfloat16 --tensor-parallel-size 1 --distributed-executor-backend mp`
- Context and batching: `--max-model-len 147072 --max-num-seqs 14 --max-num-batched-tokens 2048 --async-scheduling --enable-chunked-prefill --max-cudagraph-capture-size 56`
- Prefix caching: `--enable-prefix-caching --prefix-match-unit 128 --prefix-cache-retention-interval 0`. These two prefix options are overlay-only.
- Parsers and template: `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --generation-config vllm`, plus `--chat-template <model>/chat_template.jinja` if present, and `--default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "xhigh"}'`
- Offload: `--engram-config '{"cpu_offload": true}' --cpu-offload-params embed_tokens`
- Multimodal: `--mm-processor-kwargs '{"max_pixels": (64*upscale)^2}' --limit-mm-per-prompt '{"video": 0}'`
- Model, cache and speculative decoding: `--quantization compressed-tensors --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' --gpu-memory-utilization 0.98 --kv-cache-dtype fp8_e4m3 --attention-config '{"indexer_kv_dtype":"fp8"}' --mamba-ssm-cache-dtype bfloat16`
- Environment: `VLLM_GDN_DECODE_KERNEL=triton`, `VLLM_ARC3_GDN_RECOVERSSM=1`, `VLLM_QWEN4_EXP_DRAFT_VOCAB=configs/draft_vocab_32k.json`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, server ready timeout 3,600 s.

Source: `taaf-kaggle-source` `inference.json` + `kaggle.py` L1286–1413, L1036–1090.

- Harness settings: analyzer `target_context` 81,536, `context_window` 127,488, temperature 0.6, top_p 0.95, top_k 20, `yield_seconds` 150, `concurrent_jobs` 14, `multimodal.upscale` 4 (so boards are 256 px). — `inference.json`
- Measured Save & Run figures (server logs, 2026-09-30, demo runs, not scored reruns):

  | Metric | dfranzen | lordhansolo | rellik13 |
  |---|---|---|---|
  | Run shape | "10 games, 25 min each" | "25 games, 14 at once" | "25 games, 20 min" |
  | Peak decode tok/s | 946 | 1,135 | 1,159 |
  | Median decode tok/s | 722 | 976 | 899 |
  | Running requests at peak | 10 | 14 | 16 |
  | Median accepted draft length | 2.7 | 2.9 | 2.6 |
  | KV pool | 1,011,264 | 1,417,100 | 1,004,288 |
  | Prefix-cache hit | not logged | median 91% | not logged |
  | Job-wide gen tok/s | 588 | 933 | 688 |
  | Notebook start to ready | 531 s | 544 s | 615 s |
  | Server launch to ready | 478 s | 430 s | 536 s |

  The comparison's own notes on these figures:
  - "Job-wide tok/s is total generated tokens over the whole notebook wallclock, including server boot and idle time."
  - "Peak throughput tracks the number of concurrent requests more than the server … Per-sequence decode speed is therefore highest for dfranzen at about 95 tok/s per stream, versus about 81 and 72."
  - vLLM logs a 10 s average, while SGLang logs per batch, so the SGLang peaks are single-batch readings.
  - vLLM reported "19.69 GiB available for KV", that is 1,417,100 tokens or "9.64 full-context requests". Draft acceptance was 63–71%.

  — [COMPARISON.md §6](https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/COMPARISON.md)
- Token-capacity differences that change decode cost:

  | Setting | dfranzen | lordhansolo |
  |---|---|---|
  | Prompt budget before trim | ~118k | ~126k |
  | Trim target | ~59k | 81,536 |
  | Steady-state prompt | 60k → 118k | ≤ ~81k |
  | Tokens per board image | ~402 (640 px) | ~66 (256 px) |
  | Prefill chunk | 8,192 | 2,048 |
  | Scheduler | SGLang `lpm` | vLLM async |

  lordhansolo's pickled target "still names an older kernel slug and a 2,100 s budget". — [COMPARISON.md §1–2, Caveats](https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/COMPARISON.md)
- dfranzen's own earlier demo: prefix caching 93.43% token-weighted, P90 aggregate decode 836.6 tok/s, harness-side generation 609.0 tok/s. — [dfranzen WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md)
- Our measurements of dfranzen's harness on Kaggle (eval25, 25 public games, ~2 h runs; SGLang `serve.log` parsed by `scratchpad/run_metrics.py`):

  | Run | Median running | Median gen tok/s | Prefix hit | Total gen tokens | Wallclock |
  |---|---|---|---|---|---|
  | eval25 (10 streams) | 9 | 739.8 | 0.939 | — | — |
  | eval25 rep2 | 9 | 743.1 | 0.933 | 4,463,629 | 2h 1m 53s |
  | prior_fade | 9 | 744.5 | 0.933 | 4,464,793 | — |

  Job-wide that is about 610 tok/s. — local outputs `dfranzen_eval25_*_output/`

### Inferences
- The "+59%" (933/588) overstates the gain the server swap would give dfranzen's harness. Most of the aggregate gain is 14 vs 10 concurrent streams. Our own SGLang tests show that adding streams alone does not scale aggregate decode for this harness (section 5). lordhansolo also decodes at shorter contexts (≤81.5k vs up to 118k) with ~6× cheaper board images, which lowers attention and prefill cost per token.
- The like-for-like evidence that vLLM's kernels are faster is the median decode at peak concurrency: 976 tok/s at 14 streams (~70/stream median) vs 722 at 10 (~72/stream). That is roughly equal per stream, at different contexts. **No evidence shows vLLM decoding a token faster than Pennyroyal at matched context and concurrency.**
- Boot time is a wash (544 s vs 531 s to ready).

### Gaps
- No run of lordhansolo's runtime with 118k prompts or 640 px images exists, so decode speed in dfranzen's regime is unknown.
- The exact Save & Run duration for the 933 figure is unclear: a 2,100 s notebook budget is pickled, alongside a 3,918 s per-game cap. If the run lasted ~35 min, boot (544 s) is ~26% of wallclock and the steady-state rate would be higher still. Either way the figure is not comparable to a 9 h submission.

## 3. Compatibility of dfranzen's harness with vLLM

### Takeaway
At the API level the harness is largely vLLM-native. It is Tufa's ARC3-Inference, which was written against vLLM. Its payload builder has a `vllm` provider path (`top_k`, `chat_template_kwargs`). It never sends `priority` in requests, and its admission control (`_PriorityGate`, `ARC3_MAX_ACTIVE_STREAMS`) runs on the harness side. The real porting risks are lower-level:
- the 640 px board images would be capped to 256 px by lordhansolo's `max_pixels`
- SGLang-only launch flags (`lpm`, `--enable-cache-report`) and our `serve.log` parsing
- the reasoning-history key
- whether vLLM's align-mode Mamba prefix cache keeps ~93% hits under dfranzen's drain-trim pattern at 118k contexts

### Cited Findings
- `openai_compat.build_chat_payload` (dfranzen):
  - sends `priority` only if the caller passes one ("scheduling hint, ignored by servers that do not implement it")
  - for provider `vllm`, sets `top_k`, `chat_template_kwargs.enable_thinking` and `seed`
  - `tool_agent._chat_completion` calls it **without** `priority`, then merges `preserve_thinking` / `reasoning_effort` into `chat_template_kwargs`

  — `dfz_repo/.../ARC3-Inference/inference/utils/openai_compat.py` L51–96; `inference/agent/tool_agent.py` L5200–5262
- `_PriorityGate` docstring: "Harness-side admission control, for when the server ignores `priority`… limiting how many games hold a slot at once". dfranzen's SGLang launch has `--schedule-policy lpm --enable-cache-report --enable-metrics` and no priority-scheduling flag. — `tool_agent.py` L1976–2000; dfranzen launch args in `kaggle_dfz_eval25_mem975/*.ipynb`; WRITEUP: "The remaining games wait at the harness side (either by the concurrency limit, or the priority gate mechanism if priority scheduling is enabled)" ([WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md))
- vLLM accepts a request `priority` only when started with `--scheduling-policy priority`. Lower values mean higher priority. — [vLLM PR #5958](https://github.com/vllm-project/vllm/pull/5958); [vLLM forum](https://discuss.vllm.ai/t/priority-in-batch-api/2376) (search summary, not verified against e975732 source)
- Reasoning fields: the harness reads `message["reasoning"]` and falls back to `reasoning_content`. The history key defaults to `reasoning` (`ARC3_REASONING_HISTORY_KEY`). The docstring warns that a template that reads only `reasoning_content` silently drops reasoning stored under `reasoning`: "measured: 69 prompt tokens vs 3670 for the same text under the two keys". — `tool_agent.py` L1255–1270, L3434–3438
- Context-overflow detection matches vLLM's wording ("maximum context length") as well as SGLang's. — `tool_agent.py` `_is_context_length_error`
- Image cap on lordhansolo's side: `--mm-processor-kwargs {"max_pixels": (64*upscale)^2}`. The docstring explains why: vLLM profiled a 4096×4096 dummy image (16,384 tokens) that "set the whole 2.19 GiB activation peak, which the KV pool gives up". — lordhansolo `kaggle.py` L1322–1327
- Prefix caching on the hybrid GDN/QSA model in vLLM:
  - Upstream: experimental, enabled with `--enable-prefix-caching --mamba-cache-mode align` ([vLLM Ascend Qwen3.8-Flash-Next doc](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-Flash-Next.html)).
  - Align-mode hit rate measured at **74.3–78.6%**, against 90.2–95.3% for the proposed `all` mode. On 68,266 production multi-turn agent requests, reuse was 81% vs 70%. PR #50172 is **still open** (conflicts as of 2026-09-06) and validated only for Qwen3-Next ([vLLM PR #50172](https://github.com/vllm-project/vllm/pull/50172)).
  - Open bug: with MTP on a hybrid GDN model, "the first repeat of an identical prompt misses the prefix cache entirely", because the EAGLE-adjusted boundary differs from the sparse Mamba retention boundary ([vLLM issue #53504](https://github.com/vllm-project/vllm/issues/53504), Qwen3.8-27B-FP8 on 2× RTX 5090).
- lordhansolo's overlay targets exactly this issue:
  - Mamba align-state retention and prompt-tail state "for multi-turn prefix reuse"
  - a finer `--prefix-match-unit 128` over coarser physical Mamba pages, which "recovers almost all of MTP's boundary loss"
  - retention interval 0, which "keeps only request-replay boundaries"
  - the vLLM #58434 port, so GDN does not fold placeholder drafts into a one-token prompt tail
  - fail-closed hash checks on six files before fine-grained caching is enabled

  Measured median prefix hit with his harness: 91%. — lordhansolo `kaggle.py` L36–45, L1237–1283; [COMPARISON.md](https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/COMPARISON.md)
- dfranzen's SGLang patch keeps the end-of-prefill Mamba checkpoint and refreshes its LRU, because "due to re-tokenization, the tokenized conversation might not exactly match the generated token sequence". He also notes that "Having enough KV capacity alone did not guarantee a cache hit." — [dfranzen WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md)

### Inferences
Compatibility checklist for a port:

| Item | Status | Detail |
|---|---|---|
| Chat completions, `qwen3_coder` tool calls, `qwen3` reasoning parser, `preserve_thinking` via default template kwargs | OK | Same flags on both stacks. |
| `priority` | OK | Never sent. If someone enables it, vLLM needs `--scheduling-policy priority` or the request may be rejected. |
| `ARC3_MAX_ACTIVE_STREAMS` / priority gate | OK | Server-agnostic. Set it equal to `--max-num-seqs`. |
| SGLang `lpm` | Not available in vLLM (FCFS or priority only) | With harness admission equal to the server slot cap, the server queue is usually empty, so LPM matters less. Unmeasured. |
| `--enable-cache-report` / our `serve.log` regexes (`#cached-token`, `gen throughput`) | Must change | Switch to vLLM Prometheus `vllm:prefix_cache_*` (lordhansolo's `vllm_metrics.py` scraper is reusable) and/or `--enable-prompt-tokens-details`. |
| Board images 640 px | **Silent break** | Must raise `max_pixels` to at least 640² (or dfranzen's upscale²·64²) or boards are downscaled. That grows the vision profiling peak and shrinks the KV pool. |
| Reasoning history key | **Must verify** | Smoke test: prompt tokens must grow by the reasoning length turn over turn (the harness's own 69 vs 3,670 diagnostic). Dropping retained reasoning is a "large regression" per dfranzen. |
| Context | OK by flags | dfranzen needs a 139,264 server context; lordhansolo's `max_model_len` is 147,072. |
| KV at full depth | Preemption risk | 1.417M pool ÷ ~118k peak prompt ≈ 12 full-depth streams. The Mamba checkpoints share that pool in vLLM, so the safe count is likely 10–12, not 14. |
| Prefill chunk | Tunable | 2,048 (lordhansolo) vs 8,192 (dfranzen). dfranzen's 57k-token re-prefill after each drain would run in 2k chunks. Async scheduling may hide this; unmeasured. |
| Prefix caching | Not equivalent by construction | lordhansolo's overlay is the only vLLM path near SGLang's ~93%; upstream align mode is ~75–79%. His 91% was measured with ≤81.5k prompts and a different trim pattern (trim to 81.5k, not drain to 57k). |

### Gaps
- I did not verify which response key the e975732 `qwen3` reasoning parser emits (`reasoning` vs `reasoning_content`). I also did not verify which key the model's `chat_template.jinja` reads back from assistant history.
- I did not verify that vLLM e975732 rejects a non-zero `priority` without the priority policy. The claim comes from search summaries.

## 4. Has anyone run dfranzen's (or the Duck) harness on lordhansolo's vLLM stack? Bugs and quality differences

### Takeaway
I found **no public run of dfranzen's harness on lordhansolo's runtime**. A Kaggle kernel search for `e975732` and `lordhansolo` returns only lordhansolo's own notebook. Many "vLLM" hits are dfranzen copies on SGLang. The only like-for-like serving evidence is single Save & Run demos with different harnesses. Quality: mixed NVFP4+FP8 matches BF16 within ±1 on generic suites, and dfranzen found NVFP4 (RadixArk) and Intel W4A16 "similar". There is no ARC-AGI-3 score A/B between the quantisations at fixed harness. Several SM120 vLLM bugs affect nearby configurations.

### Cited Findings
- Kaggle `kernels list --competition arc-prize-2026-arc-agi-3 -s e975732` (run 2026-10-06) returned only `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2`. The `-s vllm` hits are mostly dfranzen forks such as `lwq255/arc3-dfranzen-m2-guarded` and `amatlas/dfranzen-fork-d-prime`. Their notebooks were not opened, and the search matches text, not runtime. — Kaggle CLI, 2026-10-06
- Demo mean scores (not comparable, different harnesses and game counts): dfranzen 36.56 (10 games), lordhansolo 5.28 (25), rellik13 6.89 (25). Public LB: dfranzen 27.89, lordhansolo 23.84, rellik13 22.53. — [COMPARISON.md](https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/COMPARISON.md)
- dfranzen: Intel W4A16 "provided similar quality and throughput" to RadixArk NVFP4, "while leaving more VRAM available for the KV cache". — [dfranzen WRITEUP](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md)
- primitive-ai mixed NVFP4+FP8 card:
  - accuracy matches plain NVFP4 "within the ±1.0 tie band" (tool calling ±1.5 across runs)
  - single-stream 84.4 tok/s, 13% faster than plain NVFP4; concurrency-32 526 tok/s, +8.7%
  - "`num_speculative_tokens: 1` hangs at startup"

  — [model card](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8)
- rellik13: MXFP8 on non-expert layers made decoding "about 25% slower". — [rellik13 WRITEUP](https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/main/WRITEUP.md)
- SM120 / vLLM bugs near this config:
  - **vLLM #59768**: illegal memory access (Xid 13) with `SimpleCPUOffloadConnector` on Qwen3.8-Flash-Next, single RTX PRO 6000, **vLLM 0.29.1rc1.dev573 (the same nightly as e975732)**, MTP, FP8 KV. It fires every 45–80 min under 4–8 concurrent requests of 80k–185k tokens with the KV pool at 92–98%. Open, no workaround. Otherwise it gave "80–90% prefix cache hit rates". — [vLLM #59768](https://github.com/vllm-project/vllm/issues/59768)
  - The official NVIDIA NVFP4 checkpoint's FP8 MTP head cuts acceptance from ~3.0 to 1.50, about 40% decode throughput. This does not apply to the primitive-ai/lordhansolo NVFP4 MTP head, whose measured acceptance is 2.9. — [untcoder2 notes](https://github.com/untcoder2/qwen38-flash-next-nvfp4-sm120-tp2)
  - FlashInfer b12x NVFP4 MoE illegal memory access on SM120 with 512 experts. — [flashinfer #5446](https://github.com/flashinfer-ai/flashinfer/issues/5446)
  - lordhansolo's launcher notes that the b12x micro kernel crashes on padding rows with expert id −1 and disables it. His run used the auto `flashinfer_cutlass` MoE backend. — lordhansolo `kaggle.py` L1030–1035; COMPARISON §4
  - Startup hangs in the PLE-offload worker on a single RTX PRO 6000. — [dbirks/home-k8s #112](https://github.com/dbirks/home-k8s/issues/112)
  - lordhansolo's own OOM history: pinning an 8 GiB KV "ran the card out of memory during warmup with 40 MiB free", because FlashInfer autotune ran a 3.39 GiB dummy forward that profiling did not count. — `kaggle.py` L46–54
- The upstream vLLM recipe for 1× RTX PRO 6000 (nightly `a9eafde`, 2026-09-27) is far more conservative: `--max-num-seqs 4`, `--gpu-memory-utilization 0.93`, `--mamba-cache-mode align`, MTP K=2, giving a 602,931-token pool at 262k context. PR #1055 is still open. — [vllm-project/recipes #1055](https://github.com/vllm-project/recipes/pull/1055)

### Inferences
- Swapping the quantisation (W4A16 → mixed NVFP4+FP8) and the drafter changes the sampling distribution. Any score change from a port will confound server, quant and drafter. The run-to-run sd (~3.9 public; ±7 on our eval25 mean) means a single A/B cannot separate them.
- A host-RAM KV tier is **not available** on the vLLM path. RecoverSSM rejects KV connectors, and the stock CPU-offload connector crashes on this exact nightly at high KV pressure. The only way to add streams is a larger GPU pool.

### Gaps
- No ARC-AGI-3 score comparison exists between W4A16 and mixed NVFP4+FP8 at a fixed harness.
- No public vLLM run of dfranzen's (or Duck's) harness with throughput logs was found. Forum dumps through 2026-10-04 contain none, and I did not open newer dfranzen-fork notebooks.

## 5. rellik13's SGLang + HiCache + FP8 PLE (688 tok/s) as a middle option

### Takeaway
We already ran the HiCache middle option on dfranzen's harness. At 12 streams it **fixed the prefix-eviction collapse**: hit rate 0.93 vs 0.55 without HiCache. It **did not raise throughput**: total generated tokens were +1% over the 10-stream baseline in the same ~2 h. rellik13's 688 tok/s comes from a different harness (69k context, 66-token images, 16 streams) and is not transferable.

### Cited Findings
- Our runs on Kaggle (25 public games, ~2 h each, dfranzen harness, SGLang Pennyroyal):

  | Run | Median running | Median gen tok/s | Prefix hit | KV usage med / p90 | Total gen tokens | Mean score (n=1) |
  |---|---|---|---|---|---|---|
  | 10 streams | 9 | 743 | 0.933 | 0.70 / 0.84 | 4,463,629 | 44.57 |
  | 10 streams, prior_fade | 9 | 745 | 0.933 | — | 4,464,793 | 48.85 |
  | 12 streams, no HiCache | 11 | 778 | **0.546** | — | **2,686,131** | 37.09 |
  | 12 streams + HiCache | 11 | 755 | **0.933** | 0.89 / 0.97 | **4,512,514** | 47.49 |

  The HiCache run's server reported `max_total_num_tokens=961344`, `max_running_requests=12`, `hicache_attached=True`, `hybrid_ssm=True`. — local `dfz_eval25_hicache_s12_output/serve.log`, `summary.txt`; `dfranzen_eval25_*_output/` via `run_metrics.py`
- rellik13's config:
  - Pennyroyal 2.5.0 + 4 patches, 48 GB host KV+Mamba tier, FP8 PLE (47.68 GB host), FP8 KV
  - 16 × 69k context, pool 1,004,288 tokens, 1.59 GB of CUDA graphs
  - 688 job-wide tok/s on a 25-game, 20-min Save & Run
  - boot log warns `fp8_unscaled_warning=True`

  — [COMPARISON.md §3, §6](https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/COMPARISON.md)
- rellik13's write-up credits the 14.49 → 22.53 jump to "FP8 KV, history 37k→57k; no prompt changes", that is, memory spent on history rather than streams. — [rellik13 WRITEUP](https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/main/WRITEUP.md)

### Inferences
- On SGLang, with dfranzen's long contexts, aggregate decode looks saturated at about 740–780 tok/s whether 9 or 11 streams are running. The bottleneck is per-step decode cost at 60–118k context, not KV capacity or cache hits. That predicts the vLLM port gains only if vLLM's per-token decode is faster at matched context, which no evidence currently shows (section 2).
- HiCache is the low-risk way to run 12 streams without losing prefix hits. It buys little by itself and costs host RAM: BF16 PLE (95 GB) plus 48 GB host tier ≈ 143 GB of 177 GB. FP8 PLE would free ~48 GB.

### Gaps
- There is one run per arm, and the score differences are inside the ±7 run-to-run spread.
- The truncated `dfz_eval25_mem975` run (1h12m, 763 tok/s median, 0.935 hit) is not comparable on totals.

## Decision summary (inference, for the report writer)

**Expected throughput gain for dfranzen's harness: small and unproven.**

Evidence against a large gain:
- The 933 vs 588 headline is apples to oranges: different harness, contexts, image size, stream count and run length.
- Per stream, SGLang was faster (~95 vs ~81 tok/s at peak).
- Our own data shows extra streams do not raise aggregate decode for this harness.

Evidence for some gain:
- The 1.417M pool (+40%) and RecoverSSM's freed blocks would let dfranzen keep 10–12 streams at full 118k depth with less eviction.
- The NVFP4 MoE / FP8-projection kernels may decode somewhat faster.

**What could break:**
- 640 px images silently capped at 256 px.
- Reasoning-history key mismatch, which would be a silent large regression.
- Upstream-quality (~75–79%) prefix hits if the overlay pins are not reproduced exactly.
- No host KV tier on this nightly (crash bug #59768).
- A laptop-validated v3 runtime that has never beaten the author's own v2 score.
- A confounded quantisation and drafter change.
- KV preemption if 14 streams run at 118k.

**Cheapest discriminating test:** a single fixed-input throughput benchmark. Replay recorded dfranzen request traces (118k prompts, 640 px images) against both servers at 10 and 12 concurrency. Compare decode tok/s and prefix hit before any score runs, then run at least 3 score runs per arm.
