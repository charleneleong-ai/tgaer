# ARC-AGI-3 Kaggle leaderboard and disclosures: delta 2026-10-11 to 2026-10-12

Baseline: the 10-11 notes (`ARC AGI 3 SOTA October update/leaderboard_and_disclosures.md`), Kaggle CLI CSV stamped `2026-10-11T02:48:47Z`. Everything below is new since that snapshot unless marked otherwise.

Data source for this delta: the Kaggle CLI was not used. Instead I read the JSON behind Mark Barney's public tracker, which scrapes the full Kaggle public board every 30 min: [arc.markbarney.net/api/kaggle/arc-prize-2026-arc-agi-3/board](https://arc.markbarney.net/api/kaggle/arc-prize-2026-arc-agi-3/board) (UI: [arc.markbarney.net/kaggle-leaderboard](https://arc.markbarney.net/kaggle-leaderboard)). The snapshot was `fetched 2026-10-12T09:04:12Z`, 4,195 teams; each row holds rank, team id, name, Kaggle last-submission time, score, entries and members. The tracker is third-party, and its author (`markbarney`) is on the competing team "Son & Mark & Ronen". Its rows agree with the 10-11 Kaggle CSV on every team checked (Yi-Chia Chen 62.96, Tufa 56.52, mtg 45.00, etc.). The tracker's event timestamps record when it *observed* a change, typically about 9 h after the Kaggle submission time, since a run takes about 9 h.

## Current top 10, teams ≥ 35, and who moved

### Takeaway
Yi-Chia Chen widened the lead again: **62.96 → 67.49** (observed 10-12 05:33Z; submitted 10-11 20:28 UTC, entry 28). The lead over Tufa Labs is now **10.97** points. Tufa's best is unchanged at 56.52, even though it submitted again (entries 159 → 160, last sub 10-11 12:55). The 40+ tier grew from 5 to 7 teams: the last dance rose to 42.55 and Son & Mark & Ronen to 41.69. AFF AI CLUB (the shiiin9 affectify authors) jumped 33.92 → **38.98**, the biggest move among teams with a public notebook.

### Cited Findings
- Top 10 at 2026-10-12 09:04Z. Last sub is Kaggle's last-submission time (UTC). The 10-11 column comes from the baseline notes. Source: [tracker board JSON](https://arc.markbarney.net/api/kaggle/arc-prize-2026-arc-agi-3/board).

| Rank | Team | Score | Entries | Last sub | 10-11 02:48Z → now |
|---|---|---|---|---|---|
| 1 | Yi-Chia Chen (threerabbits) | **67.49** | 28 | 10-11 20:28 | 62.96 → 67.49 (**+4.53**) |
| 2 | Tufa Labs (dlorah, driessmit1, infinitecreativity, jeroencottaar, pressman1, stefano1283) | 56.52 | 160 | 10-11 12:55 | unchanged (1 more entry) |
| 3 | mtg (michaeltgao) | 45.00 | 66 | 10-11 09:44 | unchanged |
| 4 | Majkel1337 | 42.66 | 15 | 10-11 22:31 | unchanged (2 more entries) |
| 5 | the last dance 🕺 (dwellement0baser, fses91, gklambauer, lukasaichberger) | 42.55 | 79 | 10-11 20:12 | 39.30 → 42.55 (**+3.25**), was 6th |
| 6 | Son & Mark & Ronen (markbarney, scizical, sonphamorg) | 41.69 | 75 | 10-11 16:45 | 39.24 → 41.69 (**+2.45**), was 7th |
| 7 | NVARC3 (cpmpml, darraghdog, eladsarafian, galkaplun, sorokin, yeyinzhu) | 40.97 | 35 | 10-11 17:30 | unchanged, 5th → 7th |
| 8 | 復活の混テキスト (8 members) | 39.53 | 82 | 10-11 16:52 | 38.15 → 39.53 (+1.38), was 12th |
| 9 | dreach.ai | 39.11 | 36 | 10-11 08:24 | unchanged, 8th → 9th |
| 10 | AFF AI CLUB (shiiin9, taiiiiiichi, ydi102143) | 38.98 | 41 | 10-11 05:08 | 33.92 → 38.98 (**+5.06**), was ~30th |

- Ranks 11-28 (all ≥ 35), same source: Nhan Duc Nguyen 38.84 (unch.), artificialagencylab.com 38.62 (unch.), fshindo 38.20 (unch.), SparseTech 37.98 (unch.), keithtyser 37.22 (unch.), _hans 37.07 (unch.), **Lord Han Solo 36.98** (+0.37 from 36.61, observed 10-11 05:31Z), **Md Boktiar Mahbub Murad 36.89** (new, 30.06 → 36.89, rank 363 → 16 at the time; now the gold line, rank 18), AI_hwlee 36.54, **Team Alicia 36.49** (new, 28.52 → 36.49), YUTO KOJIMA 36.14, Andrew Reed 36.14, **Nick2187 36.11** (33.70 → 36.11), lalalia 35.79, **yktori 35.76** (33.97 → 35.76), **Hugo Thorvaldsen 35.49** (29.67 → 35.49, rank 441 → 23), Kamal Kadakara 35.44, mikelou1 35.42 — [tracker JSON](https://arc.markbarney.net/api/kaggle/arc-prize-2026-arc-agi-3/board).
- Other risers into 32+ since 10-11: phnghiapro 29.29 → 34.28; havid aqoma 28.58 → 31.81 (10-12 09:04Z, the last event in the feed) — same source, `events` array.
- Yi-Chia Chen's full climb as the tracker observed it: 48.59 (10-04) → 55.77 (10-08 05:42Z) → 59.17 (10-09 03:51Z) → 62.96 (10-10 08:59Z) → 67.49 (10-12 05:33Z). That is +18.9 in eight days on 28 entries. Tufa's last improvement was 55.89 → 56.52, observed 10-09 04:21Z — same source, `history.trails`.
- ARC Prize's own post on X announced the 59.17 high score ("They taken lead over @tufalabs") — [ARC Prize on X](https://x.com/arcprize/status/2108577347192676612) (text from the search snippet only; X returned HTTP 402 to WebFetch). I found no ARC Prize post for 62.96 or 67.49 (unverified absence).
- Band counts now (was at 10-11 02:48Z): ≥60: 1 (1), ≥50: 2 (2), ≥45: 3 (3), **≥40: 7 (5)**, ≥38: 13, **≥35: 28 (22)**, **≥32: 133 (115)**, ≥30: 422, ≥28: 801 (725). Rank 18 (gold line) = 36.89; rank 50 = 33.47 (33.37); rank 100 = 32.60 (32.29); silver line (rank 209) = 31.36; bronze line (rank 419) = 30.02 — computed from [tracker JSON](https://arc.markbarney.net/api/kaggle/arc-prize-2026-arc-agi-3/board) `latest.rows`.
- The team count rose 4,126 → 4,195 — same source.

### Inferences
- The gap between first place and the field keeps widening: Yi-Chia Chen gained +4.5 on a single entry, while Tufa's one new entry did not move its best. With about 1 in 28 entries setting a new best, Yi-Chia's gains look like real method changes rather than best-of-many reruns. Tufa's 160 entries for +0.6 since 10-04 look like the opposite.
- The 40-45 tier (mtg, Majkel1337, NVARC3) is flat; the movers are teams that were at 38-39 (the last dance, Son & Mark & Ronen). The ≥32 band keeps growing by about 18 teams per day, mostly from rank-300-plus teams jumping 5-8 points in one submission (Murad, Team Alicia, Hugo Thorvaldsen, phnghiapro). That pattern fits adopting a stronger public or shared recipe rather than incremental tuning. This is speculation; none of them disclosed anything.
- AFF AI CLUB went 33.92 → 38.98 on its 41st entry. Its public notebook still shows the older slot-priority patch (see the notebooks section), so the 38.98 code is not public.

### Gaps
- No per-entry score history (only best-score changes), so I cannot tell whether Tufa's latest entry scored low or is still running.
- Entries submitted on 10-12 had not finished scoring when the snapshot was taken (each run takes about 9 h), so the 10-12 picture is incomplete.

## Method disclosures by top teams (10-06 to 10-12)

### Takeaway
None of the top-10 teams published a method, repo or write-up in the window. The only public signals are a teaser from keithtyser ("secret sauce ... works on both ARC 2 and 3") and a Tufa Labs post on X replying to losing first place (content unverified). Tufa's [duck-harness](https://github.com/Tufalabs/duck-harness) repo has no commits since 10-06.

### Cited Findings
- Tufa Labs posted on X in response to Yi-Chia Chen taking the lead (10-09), linked as [x.com/tufalabs/status/2108463895128269084](https://x.com/tufalabs/status/2108463895128269084) by the tracker's newsroom piece "Rabbit and spider" ([dispatch JSON](https://github.com/82deutschmark/arc-explainer/blob/main/content/news/dispatches/2026-10-09-arc3-rabbit-and-spider.json)). I could not read the post itself (X returned HTTP 402), so its content is **unverified**.
- Keith Tyser (keithtyser, 37.22, ARC-3 rank 15) wrote privately to the tracker's publisher: "I think I've finally found the secret sauce, seems to work on both ARC 2 and 3". He climbed 70th → 14th in a day (+4.48). On ARC-AGI-2 his team "ARC Raiders" is 8th at 58.75 — [arc-explainer special edition, 10-10](https://github.com/82deutschmark/arc-explainer/blob/main/content/news/evidence/2026-10-10-special-late-what-is-keith-cooking.json). The message was supplied second-hand by Mark Barney, a competitor; no method is given.
- The tracker's newsroom identifies Kaggle `raist321` (Peter Findley, 33.60, rank 42) as "Peter_Findley", a human who holds the best recorded score on every one of the 25 public games on the official human leaderboard. The desk says it "has not independently confirmed" the cross-platform identity — [dispatch, 10-11](https://github.com/82deutschmark/arc-explainer/blob/main/content/news/dispatches/2026-10-11-peter-findley-human-mastery.json); human leaderboard e.g. [arcprize.org/arc-agi/3/leaderboard?task=tu93](https://arcprize.org/arc-agi/3/leaderboard?task=tu93).
- Background on Yi-Chia Chen, not a method disclosure: 26th in ARC Prize 2024, and author of the CAV verification pipeline that won the AIMO-3 MathCorpus prize. This comes from a search summary only, so it is unverified (snippet referencing the [AIMO-3 discussion](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/708484)).
- Low-ranked public research activity on 10-10 to 10-12, for context: [LamDang/arc-agi-3-solution-da-fr](https://github.com/LamDang/arc-agi-3-solution-da-fr) made the "speed stack" the default for SFT training (10-11 20:25Z) and added W4A16 Triton expert kernels, with timings of 16K tokens in 24.0 s and 120K in 166.0 s. This continues the Qwen3.8-Flash-Next distillation line from the baseline notes. [cubres/arc-prize-2026-arc-agi-3](https://github.com/cubres/arc-prize-2026-arc-agi-3) (prvsiyan) added an "ARC3 observation-memory proposal" and a "visible-repeat advisory" (10-11), and records its Version 17 at 30.65 — GitHub API commit logs.

### Inferences
- Because write-ups are due only after the 11-02 deadline, no disclosure from the top two should be expected before then. Keith Tyser's "works on ARC-2 and ARC-3" hint suggests something general (training- or test-time-compute based) rather than an ARC-3 harness trick. This is speculation.

### Gaps
- Kaggle discussion threads still cannot be read (they render client-side, and the CLI was unavailable), so forum posts made in the window may have been missed.
- No posts found from mtg, Majkel1337, NVARC3, the last dance, AFF AI CLUB or 復活の混テキスト.

## New public notebooks scoring above ~31

### Takeaway
I found no public notebook above 33.92 in the window. One new listing appeared: "ARC3 M2 Solution", public score **33.22**, author not shown. On the evidence available it is another dfranzen Milestone 2 copy. The two new 38-39 team scores with public authors (AFF AI CLUB 38.98, the last dance 42.55) have no matching public code.

### Cited Findings
- A search-engine capture of the Code tab lists "ARC3 M2 Solution" (updated "16h ago", score 33.22, bronze medal) at the top of the score-sorted list. Below it: arc-agi-3-m2-top-submission 31.93, "ARC-AGI-3 M2 open-source reproduction (Franzen)" 26, and "ARC-AGI-3 Champion Solution" 25.68 — [Kaggle Code tab](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/code) (search snippet; author, slug and capture date **unverified**).
- The 10-10 top-40 score-sorted listing from the Kaggle API, archived by cubres, contains no notebook titled "ARC3 M2 Solution". Its top entries are dfranzen, heliosli, sujanmajhisuzan, zhangyues0000, leoprovorov (Fable & Astra Handbook), skarin (100-trajectory audit), shiiin9 (affectify), and many `*/arc-agi-3-milestone-2-solution` copies — [cubres frontier listing, 10-10](https://github.com/cubres/arc-prize-2026-arc-agi-3/blob/main/frontier/2026-10-10-public-notebook-listing.md). The 33.22 notebook therefore appeared after 10-10.
- The public [shiiin9/affectify-arc-31-54-in-a-single-sub](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub) is still listed in search captures under its "31.54 in Single Sub" title. I found no evidence it was updated to the 38.98 code (unverified; the notebook page cannot be read without the CLI).

### Inferences
- The highest public notebook score (~33) now sits about 34 points below the leader and about 6 below the 10th team. The working recipes above ~38 are private.

### Gaps
- I could not diff "ARC3 M2 Solution" against dfranzen because the Kaggle pages render client-side and the CLI was unavailable. Running `kaggle kernels list --competition arc-prize-2026-arc-agi-3 --sort-by scoreDescending` plus `kaggle kernels pull` would settle its author and diff.

## Rules, host announcements, deadline and compute changes

### Takeaway
No changes found. The arcprize.org blog's newest post is still 09-03; the final submission deadline stays 2026-11-02.

### Cited Findings
- arcprize.org blog, newest first: 09.03.26 "OpenAI's GPT-6 Astra on ARC-AGI-3", 07.06.26 "Milestone Prize #1", 05.01.26 ... No post since 09-03 — [arcprize.org/blog](https://arcprize.org/blog) (fetched 10-12).
- The ARC Prize X account's only competition post in the window is the 59.17 high-score announcement (10-09) — [ARC Prize on X](https://x.com/arcprize/status/2108577347192676612).
- Dates are unchanged on the competition page: final submission 11-02, paper 11-08, winners 12-04 — [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026) (as recorded in the 10-11 notes; no change surfaced in 10-12 searches).

### Inferences
- With 21 days left and no Milestone #3, the next official signal is the close on 11-02 and the private-board reveal.

### Gaps
- Kaggle host forum posts (rule clarifications, compute or GPU changes) could not be read; a forum-only change would have been missed.
