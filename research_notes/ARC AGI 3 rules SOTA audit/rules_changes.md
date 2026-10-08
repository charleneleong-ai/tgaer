# ARC Prize 2026 ARC-AGI-3 (Kaggle): current rules and recent changes, audited 2026-09-30

Method note: the Kaggle competition pages are rendered client-side, so WebFetch returns only a title. The page text, competition settings and discussion threads below were pulled on 2026-09-30 from Kaggle's own JSON endpoints (`competitions.CompetitionService/GetCompetition`, `competitions.PageService/ListPages`, `discussions.DiscussionsService/GetForumTopicById`) for competition id 133468. The rest of this note calls those settings "Kaggle settings". Raw copies are in the session scratchpad (`kgl/`). Kaggle does not show a revision history for Overview pages, so the date a page was last edited can only be inferred from organiser posts that say "I updated the Overview tab".

Source tiers used below:
- **[OFFICIAL-KAGGLE]**: Overview, Rules and Data pages, plus the competition settings object
- **[ORGANISER]**: posts or comments on the Kaggle forum by Kaggle staff (María Cruz, LucyHe2, Dustin, inversion) or by the host (Greg Kamradt, ARC Prize President), and arcprize.org pages and blog posts
- **[COMMUNITY]**: forum posts by participants
- **[SECONDARY]**: third-party reporting

## Kernel/runtime limits (runtime hours, GPU, CPU/RAM, disk) and changes

### Takeaway
The current limit is **9 hours** for both CPU and GPU notebooks (540 minutes in the Kaggle settings). It was 6 hours until 2026-05-07, when it was raised to 9 hours at the same time as the H100 to RTX PRO 6000 switch. Neither 7.5 h nor 12 h is correct. The "<12 hours" figure comes from ARC Prize's *Verified Testing Policy* (the arcprize.org verified leaderboard), not from the Kaggle competition, and the host confirmed on 2026-07-27 that the figure for v3 is 9 h. The GPU is an NVIDIA RTX PRO 6000 (Blackwell, 96 GB GDDR7) on a GCP `g4-standard-48` machine, with internet off.

### Cited Findings
- [OFFICIAL-KAGGLE, current as of 2026-09-30] Code Requirements: "CPU Notebook <= 9 hours run-time; GPU Notebook <= 9 hours run-time; Internet access disabled; Freely & publicly available external data is allowed, including pre-trained models; Submission file will be automatically generated." — [Kaggle Overview > Code Requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/code-requirements)
- [OFFICIAL-KAGGLE settings, 2026-09-30] `maxCpuRuntimeMinutes: 540`, `maxGpuRuntimeMinutes: 540`, `onlyAllowKernelSubmissions: true`, `usesSynchronousReruns: true`, `rerunMaxStaggerMinutes: 10`, `submissionSizeLimitMb: 20480`, `requiredSubmissionFilename: submission.parquet`. — [Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [ORGANISER, 2026-05-07, María Cruz, Kaggle] "Given the change in accelerators from H100s to RTX 6000 ... we have adjusted the code requirements and extended the maximum notebook runtime. It has increased from 6 hours to 9 hours." — [Update on Code Requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944)
- [ORGANISER, 2026-07-27, Greg Kamradt] Q: "Is the effective notebook wall-clock limit ... exactly the 9 hours ... or is there additional headroom (one official page mentions 'under 12 hours')?" A: "For v3 it is 9hrs. Where do you see 12 hours? we should switch that" — [Three clarifications on final scoring mechanics](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)
- [ORGANISER, undated policy page] The "<12 hours" line reads: "Solutions must be submitted via a Kaggle notebook and run in <12 hours to ensure reproducibility." It sits in ARC Prize's Verified Testing Policy, whose submission rules also say "You are allowed to use internet access and call external APIs". That policy covers the verified leaderboard, not the Kaggle prize. — [arcprize.org/policy](https://arcprize.org/policy)
- [ORGANISER, accelerator timeline]
  - 2026-04-28: "we just upgraded the accelerators for ARC-AGI-3. This competition now has access to Kaggle's pool of powerful new H100 accelerators" — [Upgraded accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/695158)
  - 2026-05-07: "Kaggle faced a stockout of H100 processors. If we are able to reallocate these processors in the coming months, we will. In the meantime, we have changed to RTX 6000 Pro." — [Update on accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697720). Same day, LucyHe2 (Kaggle): the machine type "should be 'g4-standard-48'".
- [OFFICIAL-KAGGLE, current] "We've added RTX 6000 machines to the ARC-AGI-3 hardware pool ... Kaggle uses machine type `g4-standard-48`. ARC-AGI-3 Only ... use of RTX for any other activity could result in moderation action ... all RTX sessions must have internet disabled." — [Overview > Upgraded accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/upgraded-accelerators)
- [ORGANISER / GCP docs] `g4-standard-48` has 48 vCPU, 180 GB instance memory, 1x RTX PRO 6000 with 96 GB GDDR7, and up to 1,500 GiB Titanium SSD. — [GCP GPU machine types](https://cloud.google.com/compute/docs/gpus)
- [ORGANISER, 2026-07-17, Greg Kamradt relaying the Kaggle team] "The default disk quota is 20 GB. If your submission writes large intermediate files to /kaggle/working/ ... the kernel will be terminated with an 'out of disk' error." Also: "Docker logs are capped at 10 MB per container"; there is no RLIMIT_NPROC; "spawning many threads on a 4-core CPU allocation will lead to contention". — [Submit Error ~30min?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724841)
- [ORGANISER, docs] The starter kit offers accelerator choices cpu / T4x2 / P100 / RTX 6000 (`g4-standard-48`, "ARC-AGI-3 exclusive, burns GPU quota faster") and requires Python 3.12 for the `arc-agi` package. The repo was last pushed 2026-05-27, and no runtime-limit changes appear in its commits. — [docs.arcprize.org/arc-prize-2026](https://docs.arcprize.org/arc-prize-2026); [ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- [ORGANISER, 2026-06-25, Greg Kamradt] The 5x-human-actions cap per level applies only on the verified leaderboard: "We don't have the 5x action cap during the kaggle competition ... Kaggle has another mechanism for a cap by way of the limited compute." — [Action Cap clarification](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713921)
- [ORGANISER, Aug–Sep 2026] RTX queue and capacity problems. 2026-08-14, María Cruz: "actively investigating potential capacity constraints for the RTX 6000 pool" — [thread 735147](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735147). 2026-09-21/22, Dustin (Kaggle): "We're working on getting more capacity" → "Capacity is restored" → "Queue has fully cleared" — [thread 742148](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742148). These were operational changes; the rules did not change.

### Inferences
- The 7.5 h figure a team was using does not appear in any official source. It may be a self-imposed safety margin under 9 h, or it may come from a mix-up with the old 6 h limit.
- Greg's "4-core CPU allocation" (2026-07-17) conflicts with the 48 vCPU of a `g4-standard-48`. The 4-core figure probably describes CPU or T4 sessions, or a container share. The effective vCPU and RAM visible inside an RTX scoring rerun should be measured (for example `nproc` and `free -g` logged to stderr) rather than assumed.
- The H100 was withdrawn "in the meantime" (2026-05-07), and no later post restores it. As of 2026-09-30, RTX PRO 6000 is the only premium accelerator documented for this competition.

### Gaps
- Kaggle does not state the exact CPU and RAM allotted inside the rerun container for RTX sessions. 48 vCPU / 180 GB is the GCP machine spec, not a Kaggle-stated container limit.
- I found no official statement that the 9 h limit has changed since 2026-05-07. I also found no revision history for the Code Requirements page to prove it has not been edited since then.

## Internet access and external/frontier APIs

### Takeaway
Internet is disabled during the scored rerun, and all RTX sessions must run offline. Frontier APIs (GPT, Claude and similar) therefore cannot be used in the Kaggle prize track. This has been the rule since launch, and no change was found.

### Cited Findings
- [OFFICIAL-KAGGLE] "Internet access disabled" is a condition for the Submit button to activate. — [Code Requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/code-requirements)
- [OFFICIAL-KAGGLE] "all RTX sessions must have internet disabled." — [Upgraded accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/upgraded-accelerators)
- [ORGANISER] "Internet access is not available during Kaggle evaluation (no API-based systems like GPT/Claude/etc.)" — [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026); "No internet access during evaluation" — [arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3)
- [ORGANISER, 2026-09-28, Greg Kamradt] Asked how to package a 27B–31B model offline: "No official way, but plenty of templates you can pick from." He pointed to public competition notebooks. — [thread 743785](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743785)
- [OFFICIAL-KAGGLE Rules §6(b)] The general Kaggle template still contains an "LLM subscription (e.g. Gemini Advanced) is acceptable if Reasonable" clause. In practice it cannot be used at scoring time because internet is off. — [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules)

### Inferences
- The only allowance for external APIs is on the ARC Prize verified or community leaderboard ([policy](https://arcprize.org/policy)). That leaderboard is separate from the Kaggle prize track and is where the "<12 h" and "internet allowed" language comes from.

### Gaps
- None found. No announcement relaxing the internet ban exists as of 2026-09-30.

## Allowed models and weights

### Takeaway
Any "freely & publicly available" external data or pretrained model may be used, as long as it is attached offline (in practice as a Kaggle Dataset or Model). Kaggle's rules set no parameter-count cap; the effective cap is the 96 GB of VRAM and the 9 h limit. To be *prize-eligible*, the rules require an open-source system, model and weights as defined by the OSI Open Source AI checklist. No changes were found.

### Cited Findings
- [OFFICIAL-KAGGLE] "Freely & publicly available external data is allowed, including pre-trained models." — [Code Requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/code-requirements)
- [OFFICIAL-KAGGLE Rules §5(a)(1)(a)] "Submissions are required to have open source system, open source model, and open source weights/parameters, as defined in the checklist from the Open Source AI definition by the Open Source Initiative." — [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules)
- [OFFICIAL-KAGGLE Rules §5(a)(3)] "In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license ... for that data and/or model(s)." This sits in tension with §5(a)(1)(a). — [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules)
- [OFFICIAL-KAGGLE Rules §6(a)] External data must be "publicly available and equally accessible to use by all Participants ... at no cost", or meet the Reasonableness standard. — [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules)
- [ORGANISER, 2026-07-13] Milestone #1 winners ran local open-weights models: Gemma-4-31B for 2nd and 3rd place, a local model inside a Python REPL harness for 1st. — [Milestone Prize #1 post](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/725002)
- [COMMUNITY] Participants report running GPT-OSS-120B, Qwen 3.x 27B and "Qwen 3.8 Next Flash NVFP4" on the RTX PRO 6000. — [738599](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738599), [742835](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742835)

### Inferences
- Gemma-family weights ship under the Gemma terms, which are not OSI-approved. Taken literally, §5(a)(1)(a) could conflict with using them. §5(a)(3) and the fact that Gemma-based entries won Milestone 1 suggest the conflict is not enforced for pretrained weights. A team relying on non-OSI weights should get this confirmed.

### Gaps
- Two September 2026 eligibility questions had no host answer when checked on 2026-09-30:
  - [743753](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743753) (2026-09-26): does a dependency dataset labelled "License: Unknown" break Milestone 2 open-source eligibility?
  - [742940](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742940) (2026-09-24): does self-generated synthetic game data count as "External Data"?

## Scoring (metric, weighting, caps) and game-set sizes

### Takeaway
The metric is **RHAE-style action efficiency**. Each completed level scores min(human_actions / agent_actions, 1) **squared**. Level scores are averaged within a game, weighted by level index (1-indexed). Game scores are averaged across games, and the total is capped at 100%. There is **no 5x action cap** on Kaggle. Evaluation uses **110 private games**: 50% (55 games) back the public leaderboard and 50% (55 games) back the private leaderboard. Every submission plays all 110 when it is scored, and private scores are fixed at that moment with no end-of-competition rerun. The 25 public games are for development only and are never scored. No change to the metric or the split was found.

### Cited Findings
- [OFFICIAL-KAGGLE Evaluation] "Scores for individual game range from 0 to 100% ... While an agent could theoretically exceed 100% by using fewer moves, scores are capped at 100%. The final score averages individual game scores across levels." It links to [docs.arcprize.org/methodology](https://docs.arcprize.org/methodology). — [Evaluation](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/evaluation)
- [OFFICIAL-KAGGLE Data] "Per-level score = min(human_actions / agent_actions, 1.0), then squared ... Per-game score = Weighted average of level scores (weighted by level index, 1-indexed). Total score = Average of all individual game scores." The human baseline is "first-time test-testers". — [Data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data)
- [OFFICIAL-KAGGLE Data] "Competition evaluation uses a separate, private set of 110 games that your agent has never seen. Half of these are used for the Public Leaderboard score, and the other half for the Private Leaderboard score." The dataset includes "the 25 public game files". Kaggle settings: `leaderboardPercentage: 50`. — [Data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data)
- [ORGANISER, 2026-03-26, inversion (Kaggle)] "Submissions run on all 110 tasks." — [684852](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/684852)
- [ORGANISER, 2026-07-27, Greg Kamradt] "Private scores are calc'd at original run time. They aren't rerun ... the submission plays both datasets." — [729985](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)
- [ORGANISER, 2026-04-23, Greg Kamradt] "a seed is used for each one. Games weren't designed with randomness or proc gen ... there are stable seeds." — [694153](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/694153)
- [ORGANISER, 2026-06-25] No 5x action cap on Kaggle (see the runtime section). — [713921](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713921)
- [OFFICIAL-KAGGLE settings] The metric is "ARC-AGI-3 Metric" (algorithm id 5571, `latestCompetitionMetricVersionId: 63243`), and `scoreTruncationNumDecimals: 2`. — [Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)

### Inferences
- The leaderboard shows 2 decimals of a 0–1 score. Differences under 0.01 cannot be seen on the leaderboard.
- The public leaderboard is a genuine held-out (semi-private) signal on 55 unseen games. The final ranking uses the other 55, which have already been scored for every submission.

### Gaps
- [COMMUNITY, 2026-09-01] No host answer was found on whether the same 55 games back the public leaderboard for every submission, or whether seeds are identical across reruns. Byte-identical agents scoring 0.20 vs 0.03 were reported without a host reply. — [738762](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738762), [726552](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/726552)
- The metric version id is visible, but I could not find a date or changelog for metric versions. No post announces a metric change.

## Submission limits, final selection, deadlines, milestones

### Takeaway
The limit is **1 submission per day**. It was briefly and mistakenly 5 per day from 2026-05-27 to 2026-06-08; the surplus submissions were invalidated. Teams select up to **2 final submissions**. Teams have at most 8 members. The entry and team-merger deadline is 2026-10-26, and the **final submission deadline is 2026-11-02 23:59 UTC**. The paper track closes **2026-11-09 23:59 UTC** on Kaggle, although arcprize.org says "Papers due Nov 8". Winners are announced 2026-12-04. **Milestone #2 is 2026-09-30 23:59 UTC (today)**. The criterion is leaderboard rank at that moment, and the notebook must be public under an open-source licence by then. Kaggle lists the payout as $25K / $7.5K / $5K. arcprize.org still shows $25K / $10K / $2.5K, which is a discrepancy. There is no Milestone #3.

### Cited Findings
- [OFFICIAL-KAGGLE Rules §2] "You may submit a maximum of 1 (1) Submissions per day. You may select up to two (2) Final Submissions for judging." Kaggle settings: `maxDailySubmissions: 1`, `numScoredSubmissions: 2`, `maxTeamSize: 8`. — [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules)
- [ORGANISER, 2026-06-09, María Cruz] "on May 27 ... we mistakenly change[d] the competition settings to allow 5 submissions per day ... Yesterday, June 8, we restored the submission limit back to one submission per day ... We are working now to invalidate any additional submissions teams made after their first successful submission for each day during that period." — [705405](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705405)
- [OFFICIAL-KAGGLE Timeline, current] Start 2026-03-25; Milestone 1 2026-06-30 (optional); Milestone 2 2026-09-30 (optional); Entry deadline and team merger 2026-10-26; Final submission 2026-11-02; Winners 2026-12-04; "All deadlines are at 11:59 PM UTC." Kaggle settings: `deadline 2026-11-02T23:59:00Z`, `teamMergerExplicitDeadline 2026-10-26T23:59Z`, `prohibitNewEntrantsExplicitDeadline 2026-10-26T11:59Z` (note the 11:59, not 23:59), and `kernelsPublishingDisabledDeadline 2026-10-26T23:59Z`. — [Timeline](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/timeline)
- [OFFICIAL-KAGGLE settings] Paper track `deadline: 2026-11-09T23:59:00Z`. — [Paper track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track); arcprize.org says "November 8, 2026 - Papers due" — [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026)
- [ORGANISER, 2026-06-24, María Cruz] "the deadline to [open-source] is 11:59pm UTC on the corresponding day ... there are 2 milestone prizes, the first one ... June 30th, and the second one is on September 30th. I have updated the Timeline and the Prizes sections." — [713634](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713634)
- [OFFICIAL-KAGGLE Prizes] "Milestone Prizes: $75,000. These prizes are based on the leaderboard score on two specific dates ... Notebooks must be made public under an open source license by the corresponding milestone dates to qualify." Milestone 1 and Milestone 2 are each First $25,000 / Second $7,500 / Third $5,000. The Final Leaderboard pays $40K/$15K/$10K/$5K/$5K. The $700K Grand Prize requires 100% and pays $350K/$175K/$70K/$70K/$35K. — [Prizes](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/prizes)
- [ORGANISER, arcprize.org] "Milestone #1 (June 30, 2026): 1st: $25K, 2nd: $10K, 3rd: $2.5K. Milestone #2 (September 30, 2026): 1st: $25K, 2nd: $10K, 3rd: $2.5K." Both pages total $37.5K per milestone, but the 2nd and 3rd place split differs. — [arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3)
- [ORGANISER, 2026-07-13 Kaggle / 2026-07-06 arcprize.org blog] Milestone #1 ($37.5K total) went to 1st Tufa Labs "The Duck", 2nd Reki, 3rd Md Boktiar Mahbub Murad "forge". — [Kaggle post](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/725002); [arcprize.org blog](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- [COMMUNITY, 2026-06-08] A grandmaster asked for a freeze-then-claim window like AIMO's, because open-sourcing drops a team's rank as copies flood the leaderboard. No change was made: the June 24 clarification kept a single 23:59 UTC deadline. — [705043](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705043)

### Inferences
- If the top teams decline to open-source, milestone money goes to the highest-ranked teams that do publish. This is how community members read the rule in [742935](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742935). No organiser has stated it explicitly.
- Kaggle's Rules page is the legally binding text ("These Rules form a binding legal agreement"), so the $7.5K/$5K split should prevail over arcprize.org's $10K/$2.5K.

### Gaps
- No official statement was found on the *exact instant* the Milestone 2 ranking is snapshotted, beyond "11:59 PM UTC", or on how ties and late-finishing reruns are handled.

## Open-source / licensing requirements

### Takeaway
There are three official texts, and they disagree:
- Kaggle Rules: the winner licence is **CC-BY 4.0** for the winning submission and its source, and the submission must use an OSI-defined open system, model and weights.
- Kaggle Prizes page: milestone notebooks must be public "under an open source license".
- arcprize.org: self-authored code must be under a **permissive public-domain licence (CC0 or MIT-0)**, and third-party code must be under at least an open-source licence.

The data licence is Apache 2.0. Winners must deliver training and inference code, write-ups and an interview.

### Cited Findings
- [OFFICIAL-KAGGLE Rules] "WINNER LICENSE TYPE: CC-BY 4.0"; "DATA ACCESS AND USE: Apache 2.0"; §5 and §8 winner obligations: deliver the code (training and inference), give a reproducible methodology description, and "conduct an interview with the sponsor and work with a technical writer". — [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules)
- [OFFICIAL-KAGGLE Prizes] "participants eligible for a prize will be removed from the competition if they do not open source their solutions." — [Prizes](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/prizes)
- [ORGANISER, arcprize.org] "all code and methods authored by the submitter must be made open source under a permissive public domain license (eg. CC0 or MIT-0). Additionally, any 3rd party code ... must be available under, at least, an open source license which allows public sharing (eg. Apache-2.0, GPLv3)." — [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026)
- [ORGANISER] Kaggle settings: `requiresIdentityVerification: true`, `witholdFinalLeaderboardUntilItHasBeenVerified: true`. Minors may take part with parental consent through Kaggle's process (Greg Kamradt, 2026-09-28). — [743785](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743785)

### Inferences
- The safe choice that meets all three texts is to release self-authored code under CC0 or MIT-0, a licence strictly more permissive than CC-BY 4.0, and to use only OSI-licensed dependencies.

### Gaps
- No organiser post reconciles CC-BY 4.0 with CC0/MIT-0. The "License: Unknown" dependency question (743753) is unanswered.

## Rule changes in late August / September 2026

### Takeaway
**I found no rule change announced in late August or September 2026.** Kaggle staff and the host made no new rules announcements after the 2026-07-13 Milestone #1 post. Their only September posts were operational: RTX capacity restored (2026-09-21/22) and a reply on minors and offline packaging (2026-09-28). Every material change this year came earlier:

| Date | Change |
|---|---|
| 2026-04-28 | H100 added |
| 2026-05-07 | H100 → RTX PRO 6000; runtime raised from 6 h to 9 h |
| 2026-05-27 → 06-08 | Submission limit accidentally 5/day, then restored to 1/day with surplus invalidated |
| 2026-06-24 | Milestone open-source deadline clarified as 23:59 UTC, with Timeline and Prizes pages edited |

The September "drama" is about teams' *choices*, not the rules. Tufa Labs announced on 2026-09-23 that they will not open-source for Milestone 2.

### Cited Findings
- [ORGANISER] Every forum topic by an ADMIN or HOST, or with an ADMIN/HOST as last commenter, was enumerated on 2026-09-30 across the whole forum (120 topics). Official announcements are dated 2026-03-25, 04-02, 04-28, 05-07 (x2), 06-09, 06-24, 07-13 and 07-17. After that there are only host replies in individual threads. — [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion)
- [ORGANISER, 2026-09-21/22, Dustin] RTX capacity restored and the queue cleared. — [742148](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742148)
- [COMMUNITY, 2026-09-23, Tufa Labs] "We ... will not be sharing our solution on September 30 for the second milestone prize ... We still plan to release our final solution when the competition ends." — [742801](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742801); follow-ups: [742935](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742935), [743624](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743624)
- [COMMUNITY, 2026-09-10, CPMP, not an organiser] "There are no rerun at end of competition ... 55 are used to compute the score shown on public LB ... 55 are used to compute a hidden score." This matches the official statements above. The same post says "100 games", which is a typo for 110. — [697944 comments](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944)
- [SECONDARY] Search snippets from aggregators and GitHub issues repeat the "<12 hours" and "$10K/$2.5K" figures. These come from the arcprize.org policy and competition pages, not from the Kaggle rules. — e.g. [scout-engine issue](https://github.com/rahulsiiitm/scout-engine/issues/9)

### Inferences
- The user may be thinking of one of these events:
  - the May runtime and accelerator change, which some teams may have missed
  - the September RTX queue and capacity episode
  - the September 30 Milestone-2 open-sourcing debate
- It may also be a change posted outside Kaggle (Discord or X), which I could not check.

### Gaps
- I did not check the ARC Prize Discord or X (@arcprize) for September posts. The Kaggle Overview and Rules pages have no public revision history, so a silent page edit cannot be ruled out. The current page text is recorded above as the baseline for 2026-09-30.
