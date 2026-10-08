# ARC-AGI-3 Kaggle: leaderboard movement and disclosures, 2026-10-04 to 2026-10-06

Prior state (2026-10-04 sweep, `scratchpad/research/arc3_leaders_2026-10-04.md`): Tufa Labs led at 55.89 and Yi-Chia Chen (threerabbits) was second at 48.59. Below them, 44 teams sat between 31.6 and 37.5 (mtg 37.54, Andrew Reed 36.14, lalalia 35.79). Neither leader had disclosed anything technical; Tufa said it would open-source only after the competition. dfranzen's notebook was the only public one above 31 (31.47 on its page, 27.89 on the LB). Son Pham had measured first-day dfranzen copies at a mean of 25.77 with sd 3.93, so ranks 3 to 44 were best explained by copy variance.

Data pulled for this note (raw files in `scratchpad/lb1006/`):
- The LB CSV via the Kaggle CLI. The file is stamped `2026-10-05T23:31:14`, so it is the latest snapshot Kaggle serves.
- The Kaggle kernel list via `ListKernels`, sorted by score and by date created.
- Forum topics active since 2026-10-03 12:00 UTC, via `GetTopicListByForumId` and `GetForumTopicById` with forumId 10403401.
- The `.ipynb` of each new notebook above 28, pulled and diffed cell by cell against dfranzen's notebook.

## 1. Current public leaderboard top 50 and movement since 2026-10-04

### Takeaway
The top two have not moved: Tufa 55.89 and Yi-Chia 48.59, with no new best from either. Below them the copy band shifted up by about 1 to 2 points and doubled in size. Teams at 32 or more went from 33 to 55, and teams at 35 or more from 6 to 11. The only new score above 37.54 is "the last dance" at 39.30. Several big jumps (+8 to +30) came from teams that were near 20 to 28 two days earlier. The gap to the leaders is unchanged at about 16 points (to Tufa) and 9 points (to Yi-Chia).

### Cited Findings
Unless noted, everything here comes from the [Kaggle LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard) CSV of 2026-10-05 23:31, compared with the 2026-10-04 04:36 CSV. Each row is: rank. team — score (submission count) [old rank / old score].

**Top 50 (measured):**
1. Tufa Labs — 55.89 (154) [1 / 55.89, unchanged; last sub 10-03 07:17]
2. Yi-Chia Chen (threerabbits) — 48.59 (21) [2 / 48.59; one more sub on 10-04 with no gain]
3. **the last dance 🕺** — **39.30** (72) [252 / 27.91]. Members: dwellement0baser, fses91, gklambauer, lukasaichberger.
4. mtg — 38.33 (60) [3 / 37.54]
5. **face-of-agi** — **37.94** (44) [603 / 20.17]. Members: cmechevalier, richardcsaky.
6. gng — 36.78 [7 / 34.77]
7. **Lord Han Solo** (Milestone 2 #2) — **36.61** (83) [492 / 24.81]
8. YUTO KOJIMA — 36.14 (119) [129 / 29.69]
9. Andrew Reed — 36.14 (4)
10. lalalia — 35.79
11. Kamal Kadakara — 35.44
12. Nhan Duc Nguyen — 34.74
13. TDSAI Lab — 34.36 [24 / 32.33]
14. Arke — 34.30 [375 / 26.46]
15. markintell — 34.21
16. EISLab_hwlee — 34.12
17. Malla — 34.00 [28 / 32.13]
18. Nic Barthelemy — 33.79 [180 / 28.74]
19. HiroyukiSasaki — 33.76
20. Jeki Wan Taufik — 33.72 [793 / 3.72]
21. Solutions SafeHive — 33.68
22. kiwi719 — 33.54 (first-ever submission)
23. Albert Wang — 33.44 [179 / 28.78]
24. Shreekumar Shah — 33.37 [146 / 29.29]
25. fshindo — 33.31 [52 / 31.46]
26. How bad can it go? — 33.29
27. Patrick Chan — 33.19
28. Kunal Aarse — 33.19
29. shineef — 33.09 [694 / 4.22]
30. Noir — 33.09
31. _hans — 33.00 [183 / 28.70]
32. Arnav Singh — 32.99
33. Rogers Johnson — 32.99
34. Majkel1337 — 32.97 [132 / 29.61]
35. Northstar — 32.96 [234 / 28.11]
36. Haraguchi-T — 32.91
37. cihan atak — 32.89
38. Rebecca EW — 32.88 [402 / 26.19]
39. keithtyser — 32.74 [218 / 28.26]
40. Mahmoud Nasser — 32.73
41. ocean240812 — 32.63
42. Cyrus — 32.62 [461 / 25.31]
43. Kravets — 32.60
44. Heliosli — 32.60 [64 / 31.06]
45. green algeria — 32.38 [317 / 27.16]
46. Edith Yong — 32.37 [392 / 26.27]
47. ShelterW — 32.30
48. Maren Sajdaras — 32.29
49. Akhilesh godugu — 32.24
50. Junhua Yang — 32.13 (186 subs)

**Other named teams:**
- Nick2187 — 31.82, rank 63
- Son Pham & Mark Barney — 31.63, rank 71 (merged team)
- AFF AI CLUB (shiiin9 / Affectify) — 31.54, rank 79
- Scott Le Grand — 30.34, rank 161
- Daniel Franzen — 27.89, rank 414 (last sub 09-30)
- Tong Hui Kang — 24.02
- rellik13 — 22.53
- NVARC3 (CPMP and others) — 21.79, rank 826

**Distribution shift (measured):**

| Threshold | Teams on 10-04 | Teams on 10-05 |
|---|---|---|
| ≥37.5 | 3 | 5 |
| ≥35 | 6 | 11 |
| ≥32 | 33 | 55 |
| ≥30 | 105 | 192 |
| ≥28 | 244 | 402 |

Total teams went from 3,652 to 3,834.

**Queue congestion:** RTX PRO 6000 queues returned on 10-05 ("waiting for ~9 hours"; [745951](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745951)). CPMP also mentions being "stuck in queue as everyone else" ([746010](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746010)). Scored runs are 9 h on the scarce GPU, so many teams get only a few draws per day.

### Inferences
- **The band is still consistent with copy variance.** The dfranzen code alone, unchanged, has produced public scores of 27.62, 27.89, 28.52, 31.47 and 34.3 (section 3). With roughly 400 teams now at 28 or more and hundreds of fresh draws, maxima of 36 to 39 are what a mean of about 28 to 29 and an sd of about 3 to 4 predict at that sample size.
- **Lord Han Solo is the one notable exception, because he has his own harness and server.** He went from 24.81 to 36.61. His public notebook is still the 10-01 version (23.84), so whatever he ran is unpublished. It could be a variance draw, a merge of dfranzen's harness onto his vLLM stack, or both. **This is not evidence of a method.**
- **The jumps are not explained by any disclosure.** "the last dance" includes Günter Klambauer and Lukas Aichberger (JKU Linz names). It went from 27.91 to 39.30 in two submissions. face-of-agi includes Richard Csaky, whose public M2 notebook scored 20. It went from 20.17 to 37.94. Neither has posted anything. Both jumps fit a team that switched to the dfranzen base and drew well.
- **Nothing on the LB closes the gap.** The two leaders sit 9 to 17 points above the copy band's tail, and no team between 39.3 and 48.59 has appeared.

### Gaps
- The CLI exposes only each team's best score and last submission time, not when the best was set or the full submission history. sonpham-org runs a half-hourly LB poller with history ([commit 9e367e3](https://github.com/sonpham-org/arc-3/commit/9e367e3)), now moved to the "ARC Explainer" public page ([commit 814c999](https://github.com/sonpham-org/arc-3/commit/814c999)). I did not fetch that page.
- The 2026-10-06 LB is not yet published; the latest CSV is 10-05 23:31 UTC.

## 2. New disclosures from Tufa Labs, Yi-Chia Chen and teams above ~32

### Takeaway
None. In the forum activity since 10-03, Tufa Labs, Yi-Chia Chen and every team above 33 have disclosed nothing technical. The only forum material is about rules and logistics: license eligibility of Qwen3.8-Flash-Next, the docker image pinning, queues, and why dfranzen's 34.3 is not on the LB. Two non-top teams added a little colour: Nick Pellegrin on local-vs-hidden transfer, and Scott Le Grand on an unreplicated 53.74 local score.

### Cited Findings
- **Tufa Labs:** the latest activity on its "will not open-source for M2" thread is only team-up requests on 10-04 ([742801](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742801)). No new post, notebook or repo commit: [Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness) has no commits since 2026-10-01 (`gh api .../commits?since=2026-10-01`).
- **Yi-Chia Chen:** no forum post or notebook found. The only web trace is the earlier ARC Prize tweet of her as new 1st place at 28.34% ([X](https://x.com/arcprize/status/2104590501915787290/photo/1)). Her 21st submission on 10-04 18:00 did not raise 48.59 (LB CSV).
- **Nick Pellegrin** (team score 23.09) on 10-04, in the "What are your agents scoring on the 25 public games?" thread ([732854](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732854)):
  - Claimed: about 105 levels on the public 25, scoring about 40% locally, which "scores ~23% on the private set".
  - Claimed: across 3 runs on the public set he got 105, 103 and 106 levels.
  - He says it is "not just resubmitting the milestone #2 notebooks" but pulls in a lot of their work.
- **Nick2187** (LB 31.82), same thread, 10-04: claimed a local best of 56.76 on the 25 public games, with 124 levels and 11 games fully won. It had no official score at the time of posting, and the team is at 31.82 now (claim; [732854](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732854)).
- **Scott Le Grand** (LB 30.34), same thread, 10-05: "53.74 and I haven't been able to repeat it". The thread is about local scores on the 25 public games. Earlier he used Claude-built per-game solvers, which are frontier-assisted development and are not run in the kernel.
- **Mark Barney and rellik13**, same thread, 10-04: about seven of the 25 public games are "extremely difficult for the current solutions" (the "Slippery Seven"). rellik13 adds that "some of these games don't budge beyond 2nd level no matter what I do".
- **Rules: Qwen Community License.** Two threads ask whether Qwen3.8-Flash-Next is prize-eligible, given rule 2.5.a (open-source model and weights) versus the 5.a.3 exemption for incompatible-license pretrained models ([745079](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745079), [745837](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745837)). As of 10-05 10:25 there was **no host answer**. All three M2 winners use this model, so the question bears on most of the top 50.
- **Docker image pin:** on 10-01 Kaggle's default image moved to Ubuntu 24.04 / Python 3.13, while the competition wheels are cp312 ([745654](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745654)). A non-staff user says scored reruns use the pinned image of the submitted notebook version, and that new notebooks must select "Pin to original environment". **No staff confirmation.**
- **dfranzen's 34.3:** CPMP asks why dfranzen's notebook shows "v1 scored 34+" when the LB shows 27.89 ([746010](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/746010), 8 comments). The comments did not come back from the API, so the answer is unknown.
- **Keith Tyser** (rank 39, 32.74) published a blog post-mortem on PPO in three other Kaggle simulation competitions ([keithtyser.com](https://keithtyser.com/blog/three-simulation-competitions-three-ppo-mistakes.html)). sonpham-org read its "next competition" list as his ARC-3 playbook ([sonpham doc](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-05-lessons-from-tyser-ppo-postmortem.md)). The post is not an ARC-3 method disclosure.
- **Son Pham & Mark Barney** (31.63) are building an RL / self-play training track on Flash-Next, with held-out "copycat" and recolor games. They report no score result yet ([sonpham-org/arc-3 commits 10-04/05](https://github.com/sonpham-org/arc-3/commits/main)).

### Inferences
- Pellegrin's numbers are the clearest local-to-hidden transfer point in the forum: about 40% local maps to about 23% hidden. They fit our memory note that local RHAE does not predict public. They also suggest Nick2187's and Scott Le Grand's local 54 to 57 should not be read as LB-level claims.
- The license question is a tail risk for everyone on Flash-Next, the leaders very likely included. It does not explain the gap.

### Gaps
- No technical information exists on Tufa's 45→55.89 jump or on Yi-Chia's 48.59. X/Twitter search returned nothing new from either after 10-04 (searches limited; no X login).
- The comments on 746010 could not be retrieved.

## 3. New public notebooks above ~31 or with new approaches

### Takeaway
Two notebooks are new above 31. Diffs show that neither changes the model, the serving stack or the prompts:
- **Affectify / shiiin9 "D′", 31.54 on one submission.** It replaces only dfranzen's slot-priority formula.
- **sigeward, 31.27.** It adds robustness and teardown wrappers only.

The most useful new data point is a calibration, not a technique: a **byte-identical copy of dfranzen's code (hknight3.0) scored 28.52**, while dfranzen's own notebook page now shows a best of **34.3**. The same code therefore spans at least 27.62 to 34.3 publicly, and nothing public above 31 is distinguishable from that spread.

### Cited Findings
From the Kaggle ListKernels pull, sorted by score, 2026-10-06; diffs are against `dfranzen/arc-agi-3-milestone-2-solution`:
- **dfranzen, "ARC-AGI-3 Milestone 2 Solution": bestPublicScore now 34.3** (was 31.47 on 10-04). The version date is still 2026-10-03 13:58 ([notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)). The [da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution) repo has no commits since 10-01. A notebook page's score is "the best of all submissions made from that notebook version, so it shows the luckiest run" ([Affectify notebook](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub)). His LB entry is still 27.89 from 09-30, which is what CPMP's 746010 asks about.
- **shiiin9 / Affectify (AFF AI CLUB), "31.54 in a Single Sub"**, created 10-04 ([notebook](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub)). The diff is 23 lines: one patch cell replaces `tool_agent.priority_value` and `ProgressPace`, plus rehearsal-mode settings. The new formula, D′, is:
  - `priority = A·M·C + B·φ`
  - A: level and action efficiency, `(300/(300+a))^2.5`
  - M: pace, `clip((30000/p)^0.4, 0.25, 4)` on tokens per cleared level
  - C: linear patience to a floor, over about 225k tokens
  - B: levels-left bonus of 16 / 14 / 10, added outside the hazard
  - φ: tail fade over the last 40%
  - Unstarted games are priced by the same formula and no longer jump the queue

  The authors' own table puts unchanged dfranzen at 27.89, 31.47 and their own resubmission 27.62, against D′ at 31.54 on one submission. They state that "one submission of D′ is not enough to say by how much it is better". The coefficients came from their own simulator (`arceval/sim/simulate.py`), which is not published. sonpham-org's comparison concludes "31.54 is inside the run-to-run spread of the unchanged base" ([doc](https://github.com/sonpham-org/arc-3/blob/main/docs/2026-10-05-dprime-slot-priority-vs-franzen.md)).
- **sigeward, "ARC-AGI-3 Milestone 2 Solution" (v3-opt1): 31.27**, 10-04 ([notebook](https://www.kaggle.com/code/sigeward/arc-agi-3-milestone-2-solution)). The diff is 72 lines and wrapper-only:
  - fail-fast input checks
  - a server startup timeout of 7 min instead of 12
  - graceful fallback if the MTP draft or FR-Spec map is missing
  - a 120 s teardown reserve in real submissions
  - provenance JSON

  The scoring path is unchanged.
- **hknight888 "hknight3.0": 28.52**, 10-05 ([notebook](https://www.kaggle.com/code/hknight888/hknight3-0)). **0 differing code lines** from dfranzen: a pure copy.
- **vladimiryakunin "DF-WM": 29.3**, 10-04 ([notebook](https://www.kaggle.com/code/vladimiryakunin/df-wm)). It is dfranzen plus an executable world-model scaffold, with `ARC3_WORLD_MODEL_SIM` on by default (`simulate()`, `check_sim()`, BFS `search()`, `act_checked()`). It was unscored on 10-04. At 29.3 it lies inside the copy band, so it shows no measurable gain, in line with earlier negative world-model results.
- **skarin**, "27.80 LB | 100-Trajectory Audit": the page best moved from 27.80 to 29.17 ([notebook](https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-100-trajectory-audit)).
- **Other dfranzen-titled copies on 10-04** (not diffed): lizw1233 26.9, svegio 24.28, anthonyxlyu 24.21.
- **amatlas "dfranzen fork + D-prime"**: still 26.98, with a version dated 10-05 ([notebook](https://www.kaggle.com/code/amatlas/dfranzen-fork-d-prime)).
- **No public notebook exists from Tufa, Yi-Chia, the last dance, mtg, face-of-agi, gng, or from Lord Han Solo's 36.61 configuration.** lordhansolo's public notebook is still the 10-01 version at 23.84.
- **Other new notebooks since 10-03** are all low or unscored (≤0.25 or None):
  - graph-exploration, memory-guided and "Digital Mouse" variants
  - GPT-OSS "brain"
  - STEP_adapt_SFT / GRAPH_SFT training notebooks
  - The one exception is manwithacat's "Digital Mouse Qwen Genome Baseline" at 26.23 (not diffed).

### Inferences
- **Observed spread of unchanged dfranzen code:** 27.62, 27.89, 28.52 and 31.47, up to 34.3 at best. That is a range of at least 6.7 points. The new public notebooks (31.54, 31.27, 29.3) cannot be told apart from it.
- **Scheduler change is the only new technique, and it remains a candidate.** D′ is cheap to A/B on our base (one cell). Note that our own earlier "prior-fade" scheduler change dropped 28.39 to 20.60 (MEMORY), so scheduler changes can hurt as well as help.
- **No public artefact explains the jump from about 34 to 55.89.** Every public notebook above 28 is the dfranzen harness with SGLang Pennyroyal and Intel W4A16 Flash-Next. The leaders' advantage remains undisclosed: model, serving, harness or search.

### Gaps
- Per-submission scores for the D′ and sigeward notebooks beyond the page best are unknown.
- The 34.3 on dfranzen's page cannot be attributed to a particular submission or person.
- lizw1233, svegio, anthonyxlyu and manwithacat were not diffed.

## 4. ARC Prize official announcements since 2026-09-30

### Takeaway
There is nothing new since the 2026-10-01 Milestone 2 winner announcement. The arcprize.org blog index shows no post after the Astra post.

### Cited Findings
- The **Milestone 2 winners** were announced on 2026-10-01: Daniel Franzen 27.9% ($25K), Lord Han Solo 23.8% ($7.5K), Lohit Siriki 22.5% ($5K). Deadline 2026-09-30 ([ARC Prize on X](https://x.com/arcprize/status/2105737436450201734); [search summary](https://arcprize.org/competitions/2026)).
- The [arcprize.org/blog](https://arcprize.org/blog) index, fetched 2026-10-06, lists "OpenAI's GPT-6 Astra on ARC-AGI-3" ([/blog/astra](https://arcprize.org/blog/astra)) as the most recent post. There is no Milestone 2 blog post or paper.
- **Competition timeline:** final submission deadline 2026-11-02, results 2026-12-04 ([arcprize.org/competitions/2026](https://arcprize.org/competitions/2026), via search summary).
- **The hosts have not answered** the Qwen-license or docker-pinning questions as of 10-05 ([745079](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745079), [745654](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745654)).
- **Frontier-API results are not allowed in the offline kernel.** sonpham-org published "Astra lessons with trace evidence across all 183 public levels" on 10-05 ([commits #81/#82](https://github.com/sonpham-org/arc-3/commits/main)). Astra is GPT-6, an API model, so these are analysis inputs only, not a portable technique.

### Inferences
- With no official write-up of the M2 winners beyond their own repos, the M2 material is unchanged from the 10-04 note.

### Gaps
- The arcprize.org blog index fetch may have rendered incompletely. ARC Prize's X feed after 10-01 could not be read (X needs login), so a tweet-only announcement could have been missed.
- The Digg item "Tufa Labs raises the leading score on the ARC-AGI-3…" returned 503 and could not be read ([digg](https://digg.com/tech/k8o9t7me)).
