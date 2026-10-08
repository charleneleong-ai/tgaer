# Methods that work under extremely sparse reward (no success signal for whole episodes)

Scope note: the applied setting is an interactive deterministic grid benchmark (ARC-AGI-3) where the agent
clears a level in only ~5 of 25 games, there is **no reset** (every action is permanently spent), and the
transition function is deterministic. Throughout, methods are tagged **[needs resets]**, **[needs a first
success]**, or **[reset-free / works with zero reward]** because that distinction is the load-bearing one here.

---

## Q1. Leading methods for sparse-reward exploration as of 2025-2026 (count-based, pseudo-counts, curiosity, Go-Explore lineage)

### Takeaway
The classical lineage (pseudo-counts → RND → Go-Explore) still supplies the standard numbers, but every member
of it is an *episodic, reward-bootstrapped* method: the bonus tells the agent where to go, and the extrinsic
reward is what eventually shapes the policy. The 2025-2026 shift is away from scalar intrinsic bonuses toward
**explicit archives + explicit world models driven by a model-error / uncertainty signal**, with the
foundation model supplying the notion of "interesting" (Intelligent Go-Explore, ICLR 2025) or the world model
supplying it as a Bayesian diagnostic (OPINE-World, 2026). On ARC-AGI-3 specifically, the best published
non-LLM baseline is not an intrinsic-bonus method at all — it is training-free systematic graph traversal.

### Cited Findings

**Classical, still the reference numbers (clearly older):**
- **Pseudo-counts (Bellemare et al., NeurIPS 2016)** — CTS density model → pseudo-count bonus
  `R+ = β(N̂(x)+0.01)^(-1/2)`, β=0.05, on DQN with mixed Monte Carlo update. At 50M frames the bonus agent had
  explored **15 rooms of Montezuma's Revenge vs 2 rooms without the bonus**; one run consistently reached
  **6,600 points by 100M frames**, state of the art at the time — [Unifying Count-Based Exploration and Intrinsic Motivation](https://proceedings.neurips.cc/paper_files/paper/2016/file/afda332245e2af431fb7b672a68b659d-Paper.pdf); [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)
- **Neural density models for pseudo-counts (Ostrovski et al., ICML 2017)** replaced CTS with PixelCNN, the
  standard follow-up in the count-based line — [Count-Based Exploration with Neural Density Models](https://proceedings.mlr.press/v70/ostrovski17a.html)
- **RND (Burda et al., 2018/ICLR 2019)** — prediction error against a fixed random target network. Reported
  Montezuma score **8,152** (mean), best agent discovered **22 of 24 rooms** on level 1, over half the rooms
  consistently — [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894); score figure via [search summary](https://arxiv.org/abs/1810.12894)
- **ICM (Pathak et al., 2017)** — forward-model prediction error as curiosity; cited as enabling exploration of
  Super Mario Bros with *no extrinsic reward at all*, the canonical demonstration that curiosity alone produces
  progress — [cited in Rudakov et al. 2025](https://arxiv.org/pdf/2512.24156)
- **Go-Explore (Ecoffet et al., Uber AI; Nature 2021 "First return, then explore")** — archive of cells,
  *return* to a cell, then explore from it. Reported: Montezuma **mean 469,209 / max >2,000,000 (level 159)**
  with domain knowledge after robustification, **mean 35,410 without domain knowledge**, vs prior SOTA
  **11,347 mean / 17,500 max** and **human expert average 34,900**; Pitfall **>21,000** where **no prior
  algorithm scored above zero** — [Uber AI blog](https://www.uber.com/us/en/blog/go-explore/); paper [arXiv:2004.12919](https://arxiv.org/abs/2004.12919)
- Go-Explore's return step is explicitly **reset/state-restore based in its efficient form**: "Atari is
  resettable, so for efficiency reasons we return to previously visited cells by loading the game state…
  this optimization allows us to solve the first level **45× faster** than by replaying trajectories" —
  [Uber AI blog](https://www.uber.com/us/en/blog/go-explore/). The stochastic-robust variant replaces
  restore with a **goal-conditioned policy** that has to walk back — [arXiv:2004.12919](https://arxiv.org/abs/2004.12919)

**2025-2026:**
- **Intelligent Go-Explore (IGE), Lu, Hu & Clune, ICLR 2025** — replaces Go-Explore's hand-designed cell
  representation and selection heuristics with a foundation model that decides *which archived state to return
  to*, *which action to take*, and *whether a state is interestingly new and worth archiving*. Reported to
  "strongly exceed classic RL and graph-search baselines" and to succeed "where prior state-of-the-art FM
  agents like Reflexion completely fail", on Game of 24, BabyAI-Text and TextWorld; intelligent filtering
  "drastically reduce[s] the size of the archive" — [arXiv:2405.15143](https://arxiv.org/abs/2405.15143); [ICLR 2025 PDF](https://arxiv.org/pdf/2405.15143); [code](https://github.com/conglu1997/intelligent-go-explore)
- **DISCOVER (NeurIPS 2025)** — automated curricula for sparse-reward, *very long-horizon* goal-conditioned RL;
  selects exploratory goals **in the direction of the target task** rather than by undirected novelty —
  [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116697)
- **Metric-based exploration bonuses (NeurIPS 2024)** — argues the critical design choice is *how novelty
  between adjacent states is quantified*, and that bisimulation-metric bonuses have a theory-practice gap that
  makes them struggle precisely on hard-exploration tasks — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a39cf3b666f8bdb2223f253981f3869-Paper-Conference.pdf)
- **LLM-driven intrinsic motivation (2025)** — LLM generates the reward signal from an environment description;
  demonstrated only on MiniGrid DoorKey with an actor-critic agent — [arXiv:2508.18420](https://arxiv.org/abs/2508.18420)
- **Intrinsic Motivation in RL: A Research Agenda for Adaptive Self-Organisation (Belikov, Sept 2026)** — recent
  survey; taxonomy = curiosity (RND, ICM), exploration bonuses (counts, pseudo-counts), information-theoretic
  (empowerment, information gain), representation-based (successor features) — [arXiv:2609.17325](https://arxiv.org/pdf/2609.17325)
- Standing taxonomy from the field's own survey literature: knowledge-based, data-based, and competence-based
  intrinsic motivation, the first two aiming at maximal state coverage when extrinsic reward is sparse or
  absent — [awesome-exploration-rl](https://github.com/opendilab/awesome-exploration-rl); [A survey on intrinsic motivation in RL](https://arxiv.org/pdf/1908.06976)

### Inferences
- The classical numbers are not transferable evidence for our setting. Every headline Montezuma/Pitfall figure
  above comes from **millions to billions of environment frames across millions of episodes with free resets**,
  and Go-Explore's specific advantage was measured *because* it could teleport. A method whose published win is
  "45× faster than replaying trajectories" is reporting a benefit we structurally cannot have.
- The part of Go-Explore that survives without resets is the **archive + frontier bookkeeping**, not the return
  mechanism. That is exactly what the ARC-AGI-3 graph explorer (Q2) reimplements, and it is why it works.
- IGE is the closest 2025 method in spirit to our problem (it needs no reward at all to decide what is
  interesting), but its reported environments are text/puzzle domains with cheap resets, and its archive
  selection presumes you can go back to an archived state.

### Gaps
- IGE's exact numbers (success rates per environment, sample counts) could not be extracted — the arXiv
  abstract page carries only qualitative claims and the results tables are in the PDF body. Report writer
  should treat IGE as "qualitatively strong, numbers not captured here".
- A conflict exists in the Montezuma literature: one search summary reported "Go-Explore 43,000 (2021)" while
  the Uber blog reports 469,209 (domain knowledge) / 35,410 (no domain knowledge) and attributes 11,347 to the
  prior SOTA. These are different configurations (exploration phase vs robustified, with/without domain
  knowledge) and possibly different papers (2019 preprint vs Nature 2021). **Do not state a single canonical
  Go-Explore Montezuma number without checking the Nature 2021 table.** The Nature page is paywalled/redirects
  to an auth endpoint, so it was not verified here.

---

## Q2. Deterministic environments with NO reset: what is optimal or near-optimal? Systematic traversal and the cost of returning to a frontier

### Takeaway
This is the one question with a genuinely satisfying answer: when the environment is deterministic and
reward-free, the problem *is* online graph exploration, and there are tight classical competitive bounds.
The critical structural distinction is **undirected (actions reversible) vs directed (actions irreversible)**:
undirected online exploration is Θ(log n)-competitive with simple greedy algorithms, whereas directed
exploration costs blow up with the graph's *deficiency* — the Chinese-Postman-style measure of how far the
graph is from Eulerian. And on ARC-AGI-3 itself, a training-free systematic graph traversal is the strongest
published non-frontier-LLM result, beating both a random policy and GPT-4.1+DSL.

### Cited Findings

**Online graph exploration (undirected, weighted, unknown graph; must return to start — i.e. the
Chinese-Postman / TSP-with-unknown-graph formulation):**
- The best known competitive ratio for arbitrary undirected weighted graphs is **O(log n)**, attained by
  **Nearest Neighbour**; hierarchical DFS also achieves **Θ(log n)**. A matching **Ω(log n) lower bound** holds
  even for unweighted graphs and trees, so greedy-nearest-frontier is asymptotically optimal among known
  algorithms for general graphs — [Online graph exploration: New results on old and new algorithms](https://www.sciencedirect.com/science/article/pii/S0304397512006445); [Robustification of Online Graph Exploration Methods (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/21208/20957)
- Constant-competitive results exist for structured graphs: **planar graphs 16-competitive**, **bounded genus g
  16(1+2g)**, **graphs with k distinct edge weights 2k**, **Nearest Neighbour 1.5-competitive on cycles** —
  [same survey](https://www.sciencedirect.com/science/article/pii/S0304397512006445); [Online Graph Exploration on Trees, Unicyclic Graphs and Cactus Graphs](https://arxiv.org/pdf/2004.06690)
- 2026 developments in the pure theory: **a lower bound of 4** for online graph exploration
  ([arXiv:2607.15113](https://arxiv.org/pdf/2607.15113)) and **randomisation provably breaks the deterministic
  lower bound on cycles** ([arXiv:2607.11203](https://arxiv.org/pdf/2607.11203)) — i.e. the field is still
  actively tightening constants, general-graph constant-competitiveness remains open.
- Improved general lower bound work: [Improved Lower Bound for Competitive Graph Exploration](https://arxiv.org/pdf/2002.10958)

**Directed / irreversible-action exploration (the case that matches "no reset, actions permanently spent"):**
- For an unknown **strongly connected directed** graph, define **deficiency d = the number of edges that must
  be added to make the graph Eulerian** (d=0 ⇒ a single Eulerian tour explores every edge in m traversals, the
  Chinese-Postman optimum). Deng & Papadimitriou's algorithm may need **d^O(d)·m** edge traversals; **Albers &
  Henzinger's "Balance" improves this to d^O(log d)·m**. Best known lower bounds: **Ω(d²m)** deterministic and
  **Ω(d²m / log d)** randomised — [Albers & Henzinger, Exploring Unknown Environments](https://infoscience.epfl.ch/server/api/core/bitstreams/fc967a90-a76f-4bc9-8621-5a9ae818962e/content); [Directed Graph Exploration (survey chapter)](https://link.springer.com/chapter/10.1007/978-3-642-35476-2_11)

**Reward-free exploration complexity (RL theory):**
- There is a class of **deterministic** systems with linear Q-functions where **any** reward-free algorithm needs
  **Ω(2^H)** samples in the exploration phase to later produce a 0.1-optimal policy with probability ≥0.9 —
  a hardness result specifically about deterministic dynamics, showing determinism buys you nothing by itself
  once the horizon is long and you only have value-based structure —
  [On Reward-Free RL with Linear Function Approximation (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/ce4449660c6523b377b22a1dc2da5556-Paper.pdf)
- Contrast: under a *model-based* (linear MDP) assumption, reward-free RL has polynomial sample complexity;
  linear-Q* is strictly weaker than linear-MDP in the reward-free setting — [same paper](https://proceedings.neurips.cc/paper/2020/file/ce4449660c6523b377b22a1dc2da5556-Paper.pdf). For linear mixture MDPs the lower bound is **Ω(d²H³/ε²)** episodes — [arXiv:2303.10165](https://arxiv.org/pdf/2303.10165)

**Applied directly to ARC-AGI-3 — Rudakov, Shock & Cowley, "Graph-Based Exploration for ARC-AGI-3
Interactive Reasoning Tasks" (arXiv:2512.24156, 30 Dec 2025):**
- The games are explicitly **deterministic**: "the same action taken from the same state always produces the
  same outcome. This property enables systematic state-space exploration strategies and graph-based
  representations of explored states. However, determinism does not imply simplicity; the complexity arises
  from the large state and action spaces and the lack of prior knowledge about which actions lead toward goal
  states." — [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)
- Method: frame segmentation into single-colour connected components + status-bar masking + **state hashing**
  (the hash of the masked frame is the node id) + a directed graph of state→action→state. Per action it stores
  priority tier π(a), tested/untested status, successor, and **minimal distance to the nearest unexplored
  frontier**. Action selection (their Algorithm 1): take an untested action at the current priority threshold
  if one exists here; else **move along the shortest path to the nearest reachable state that still has an
  untested action at that priority**; else raise the priority threshold and recurse. This is literally
  Nearest-Neighbour frontier exploration with a salience prior, i.e. the Θ(log n) algorithm from the theory
  above — [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)
- Results: **median 30 of 52 levels across six games** (median **16 levels on the 3 private games**, **14 on
  the 3 public games**, 5 independent 8-hour runs), **3rd on the private leaderboard**, "substantially
  outperforming frontier LLM-based agents". The officially submitted run solved **12 private levels** (still
  3rd) — the gap was an implementation bug in handling **reset-inducing actions**: an action that triggered a
  reset appeared as a self-edge from the start node, so the agent kept selecting it and kept resetting the game
  — [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)
- Budget-matched ablation at **4,000 interactions per game** (the LLM baseline's effective ceiling; the
  benchmark in principle allows **96,000 steps**): random agent **6 private levels + 3 public**; **GPT-4.1 +
  DSL code-writing agent 5 private levels — i.e. the frontier-LLM method underperforms a uniform random
  policy**; random + frame segmentation adds public-game levels (5 on vc33, 2 on ft09); "untested actions
  favoured but no full state graph" solves only 4; **the complete graph explorer solves 19 levels** (ft09 2,
  ls20 2, vc33 5, sp80 1, lp85 2, as66 7) — [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)
- Stated failure modes: cost grows with state-space size ("limiting scalability to levels with moderate
  complexity", degrading on ft09 level 6+ and ls20 level 3+), and the approach "assumes deterministic, fully
  observable environments and would fail under" stochasticity/partial observability —
  [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)
- Benchmark structure as they describe it: 6 games (public ft09, ls20, vc33; private sp80, lp85, as66), **8-10
  levels each**, a **single sparse reward signal (level completion)**, scoring on levels completed *and* total
  actions, and when the step counter hits zero **the current level resets to its initial state** —
  [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)

### Inferences
- **The applicable optimality result for our setting is online graph exploration, not RL theory.** If the
  reachable state graph is effectively undirected (actions undoable), greedy nearest-frontier traversal is
  within O(log n) of the optimal tour and no known algorithm does better on general graphs — so "walk to the
  nearest state with an untested action" is close to the best defensible default, and it is what the 3rd-place
  ARC-AGI-3 agent does.
- **If actions are irreversible, the cost model changes qualitatively, not quantitatively.** The directed
  bounds (d^O(log d)·m upper, Ω(d²m) lower) say the penalty is governed by deficiency d — how many one-way
  doors there are. Practically: irreversible actions should be *deferred* (explored last from any state) so
  that the explored subgraph stays as close to Eulerian as possible. That is the principled version of the
  reset-handling bug that cost the ARC-AGI-3 graph agent 4 private levels.
- Note the asymmetry with our constraint: the graph-exploration literature assumes the agent *can* walk back
  (it charges you the path length). Our "no reset, actions permanently spent" is the same charging model — we
  pay for the return — so the theory applies directly. What does *not* apply is Go-Explore's teleport.
- The Ω(2^H) deterministic reward-free lower bound is a warning about the *representation*, not the traversal:
  determinism plus value-based structure alone is exponentially hard; you need model-based structure. This is
  independent evidence for the OPINE-World-style programmatic-world-model route over bonus-shaped value
  learning.

### Gaps
- No paper found that *explicitly* frames the ARC-AGI-3 no-reset return cost as Chinese Postman / deficiency.
  The connection above is my inference from the graph-exploration bounds; it is not, as far as this search
  found, stated in the RL or ARC literature.
- The graph-explorer paper's per-level action counts (Tables 1-2, appendix) were not extracted, so no
  actions-per-level efficiency figure for it.

---

## Q3. Goal inference without reward — inferring what the environment wants from structure alone (empowerment, successor features, 2025-2026 goal inference)

### Takeaway
Empowerment and successor features are the standing information-theoretic answers, and 2024-2026 work made
empowerment *scalable* by routing it through learned successor features (ESR) — but empowerment answers "keep
your options open", not "here is the goal this level wants". The 2026 result that actually infers goal
structure from environment structure alone is **OPINE-World**: it builds an object-centric *programmatic*
world model online and drives exploration by **ontology error**, a Bayesian measure of how badly its current
object-type partition explains what it has seen. That is a progress signal that exists on step 1 of a game
that will never emit a reward.

### Cited Findings

**Empowerment / successor features:**
- **Empowerment via Successor Representations (ESR), Myers, Ellis et al., NeurIPS 2024** — "an objective for
  training agents intrinsically motivated to assist humans **without requiring a model of the human's reward
  function**", maximising the influence of actions on the environment, via a **scalable model-free objective
  derived from learned successor features that encode which states may be wanted given the current action**.
  Reported to significantly outperform prior empowerment baselines as environment complexity increases, where
  **prior empowerment methods perform worse than a random controller**; the objective provides a lower bound on
  reward maximisation under stated assumptions — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/83a4ea71b13bc86308a2bd0b5e07fb61-Paper-Conference.pdf); [project page](https://empowering-humans.github.io/)
- **Information-Theoretic Policy Pre-Training with Empowerment (2025)** — empowerment as a pre-training
  objective — [arXiv:2510.05996](https://arxiv.org/html/2510.05996)
- **When Empowerment Disempowers (2025)** — a counter-result on empowerment-as-assistance —
  [arXiv:2511.04177](https://arxiv.org/html/2511.04177v1)
- **Evaluating Agents without Rewards** — the reward-free-evaluation framing (compares curiosity, empowerment
  and input entropy as proxies for behaviour quality) — [arXiv:2012.11538](https://arxiv.org/pdf/2012.11538)
- **Agent-centric learning: from external reward maximisation to internal knowledge curation (2025)** — argues
  existing intrinsic motivation is "fundamentally environment-centric in ways that can lead to overfitting to
  the specifics of the current environment" and that the focus on "what is out there to be known or done" is
  the wrong frame — [arXiv:2507.22255](https://arxiv.org/pdf/2507.22255)

**OPINE-World (Courtis, Li & Sanner, U. Toronto; arXiv:2607.01531v2, 15 Jul 2026) — the directly relevant
2026 result:**
- Two cooperating LLM agents: one acts in the environment, one synthesises an **object-centric programmatic
  world model in source code**, refined by **counterexample-guided inductive synthesis (CEGIS)** with replay
  verification and model-based planning — [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- **The reward-free progress signal, concretely:** each object transition is reduced to a coarse **effect
  signature** e = ρ(Δ) (which attributes changed, values discarded), filed into a row keyed by
  (type, action, local context) with counts C_t(j,e); a symmetric Dirichlet prior gives a posterior mean
  `q̂_j(e) = (α₀ + C_t(j,e)) / (mα₀ + Σ_e' C_t(j,e'))`. Two normalised entropies — **type uncertainty**
  `U^type_i = H[Pr(τ_i|D_t)]/log K` and **row (effect) uncertainty** `U^row_j = H(q̂_j)/log m` — combine by
  noisy-OR into per-object **ontology error** `η_i = 1 − (1−U^type_i)(1−U^row_j)`, averaged to an aggregate η_t
  that directs exploration toward what the current types do not explain. "A row that mixes signatures is
  under-observed or missing a context feature" — [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- Explicit assumption: **Observable-Markov determinism** — a representation exists under which T is a
  deterministic function of state and action; under hidden state "the ontology error … has a floor it cannot
  drive to zero" — [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- **No resets:** "OPINE-World plays each game online under the live, on-policy budget, **with no reset to
  re-sample a level**." It is not trained or fine-tuned per game; it learns each game's mechanics online during
  the scored run — [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- Results on the 25-game ARC-AGI-3 set: **clears 20 of 25 games** and reaches **action-efficiency score 78.4**,
  vs **baseline1 (GPT-5.5 high reasoning, single-agent object-centric program-synthesis world model, Rodionov
  2026) 14 games / 63.8**, and a pretrained continual-learning **Vision agent 12 games / 63.2** (the Vision
  agent explores the public evaluation games offline first, so it is not a no-training comparison).
  **WorldCoder and neural latent world models clear zero games under the benchmark's budget.** —
  [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- The margin is concentrated on hard games: on the six games baseline1 never clears (re86, tn36, vc33, m0r0,
  sc25, sp80) baseline1 burns **10,874 actions in total and fails**, OPINE-World clears **all six in 2,578
  actions** (~¼ of baseline1's spend) against a human reference of **3,994 actions** (**0.65× the human
  budget**). Plus a seventh, dc22, at 1,479 actions. On the 13 games both clear, action counts are within a
  fraction of a percent. Averaged over its 20 wins the human/agent action ratio is **1.7**, best on m0r0 (4.3),
  lp85 (3.5), cn04 (3.0); it beats the human count on **16 of 20** wins —
  [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- The scoring rule (important for goal-setting incentives): each cleared level scores
  **min(1.15, (human_actions / agent_actions)²)**, levels weighted by index, per-game score capped at 100 —
  [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)

**ARC Prize's own framing and baselines:**
- ARC-AGI-3 explicitly names **goal-setting under sparse reward** as one of four tested components:
  "Exploration, where agents must actively obtain information by interacting with their surroundings;
  Modeling, where agents must turn raw observations into a generalizable model that can predict future states
  and outcomes; **Goal-setting, where agents must identify target future states with only sparse rewards**;
  and Planning and execution" — [ARC-AGI-3 technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf); [arXiv:2603.24621](https://arxiv.org/html/2603.24621v1)
- Environments have "**deterministic, closed-ended mechanics and goals**"; agents receive no instructions —
  [ARC Prize, Astra post](https://arcprize.org/blog/astra)
- Human baseline: ~500 general-public participants, ~9 games per session, **median action count among those
  who completed each level**, no code interpreter or scratch pad — [ARC Prize, Astra post](https://arcprize.org/blog/astra)
- Frontier models scored **below 1% as of March 2026**, humans 100% — [search summary of arXiv:2603.24621](https://arxiv.org/html/2603.24621v1)
- **OpenAI GPT-6 "Astra" (2026)**: **62.7% on Semi-Private with the standard harness ($26K)** and **99.9% with
  a Provider Adapter harness ($19K)**; used **fewer actions than the human baseline on 96.0% of levels** and
  **51.7% fewer actions per level on average** — [ARC Prize, Astra post](https://arcprize.org/blog/astra)

### Inferences
- Ontology error is the cleanest available answer to "infer what the environment wants from structure alone":
  it is a *typed* information-gain signal (uncertainty over the object partition, not over pixels), so it does
  not suffer the noisy-TV pathology the way pixel-prediction curiosity does, and it is defined before any
  reward has ever been observed. For a game where we will never see a level clear, ontology error still
  monotonically drives the agent to resolve mechanics.
- Empowerment is the wrong tool for our specific problem. It tells you to preserve optionality, which under
  irreversible actions and a fixed action budget is a *constraint* worth having but not a *direction*. ESR's
  own framing is assistance (inferring a human's latent goal), not inferring a level's win condition.
- The ARC-AGI-3 scoring rule squares the human/agent action ratio and caps per-level credit at 1.15, so the
  marginal value of clearing an *additional* level far exceeds the value of clearing an already-cleared level
  more efficiently — consistent with the applied note that our solved levels capture a small fraction of their
  nominal worth. Any exploration policy should be budgeted per level against the human median, not globally.

### Gaps
- No 2025-2026 paper found that infers a *goal predicate* (a win condition) purely from structure without any
  reward observation. OPINE-World synthesises a `reward_function(state)` returning a predicted reward and a
  goal flag, but the paper's planning loop plans "to reward from the entry states of the levels already
  cleared" — i.e. it *does* use observed clears where available. Whether its goal flag is inferable with zero
  clears anywhere in the game is not established by what was extracted here. Flag this explicitly: this is the
  precise hole in the literature for the applied setting.
- Successor features as a *goal-inference* device (as opposed to a transfer/representation device) has no
  2025-2026 result with numbers that this search surfaced.

---

## Q4. Which methods degrade gracefully to "do something systematic" when no reward ever arrives, versus collapsing to random behaviour?

### Takeaway
The dividing line is whether the method maintains an **explicit, persistent record of what has and has not been
tried** (archive/graph/world model) or only a **scalar bonus fed to a value learner**. Explicit-record methods
degrade to exhaustive systematic traversal — which is exactly the right zero-reward behaviour and is
demonstrably 3-6× better than random on ARC-AGI-3. Bonus-driven methods degrade toward random, and worse: the
documented failure modes (vanishing bonuses, detachment, de-synchronisation) actively *un-explore* previously
reached frontiers.

### Cited Findings
- **Degrades gracefully (explicit record, needs no reward at all):** the ARC-AGI-3 graph explorer is
  **training-free** and has no reward term whatsoever — its policy is "shortest path to the nearest untested
  state-action pair" — yet it solves a **median 30/52 levels**, vs **9 levels** for a random agent at matched
  4,000-step budget (6 private + 3 public) and **5 private levels** for GPT-4.1+DSL. The component ablation
  shows the ordering random < random+segmentation < untested-action-preference-without-graph (4) < full graph
  (19) at matched budget — the *graph*, not the perception, is where most of the gain sits —
  [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)
- **Degrades gracefully (explicit world model, no reward needed for the signal):** OPINE-World's ontology error
  is defined from transition counts alone and "OPINE-World … avoids spending thousands of actions on a game it
  cannot yet model", whereas baseline1 exhausts its budget on six games it cannot clear — the
  model-uncertainty signal functions as a *stopping/allocation* rule as well as a direction —
  [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- **Collapses (documented failure modes of bonus-based intrinsic motivation):** the 2026 survey lists
  **noisy-TV** ("agents become distracted by random, uncontrollable environmental noise"), **generalisation
  failure** (bonuses fail to transfer across tasks/environments), **vanishing bonuses** ("exploration
  incentives diminish as agents become familiar with states, potentially prematurely halting productive
  exploration"), **episodic-reset dependency** ("many methods require explicit episode boundaries, limiting
  applicability in continuous learning scenarios"), and unsolved intrinsic/extrinsic reward scaling —
  [arXiv:2609.17325](https://arxiv.org/pdf/2609.17325)
- **A named, mechanistic collapse:** in sequential coordination tasks novelty bonuses cause **"coordination
  de-synchronisation, where agents repeatedly traversing earlier coordination points gradually exhaust their
  intrinsic motivation to revisit these critical locations"**; **"lifelong novelty bonuses deteriorate with
  increasing task complexity, while augmenting with episodic bonuses substantially improves performance"** —
  [When Intrinsic Motivation Fails: Exploration Challenges in Decentralized MARL (2026)](https://link.springer.com/chapter/10.1007/978-3-032-19105-2_3)
- **Go-Explore's own diagnosis of why bonus methods collapse:** *detachment* (the agent loses track of
  promising frontiers it had reached) and *derailment* (it cannot reliably get back to a deep state). Go-Explore
  fixes both by **remembering and returning** rather than by a better bonus —
  [Uber AI blog](https://www.uber.com/us/en/blog/go-explore/); [arXiv:2004.12919](https://arxiv.org/abs/2004.12919)
- **A concrete anti-pattern from our own benchmark:** the graph explorer's 16→12 private-level regression came
  from a **self-edge on a reset-inducing action** — the agent "repeatedly selected it, resetting the game and
  effectively entering" a loop. Systematic methods fail *catastrophically and silently* when the graph
  abstraction mis-models an irreversible transition — [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)
- **LLM-as-policy collapses below random here:** GPT-4.1+DSL solves 5 private levels vs random's 6, "meaning
  that the LLM-based method underperforms even a random policy" — partly a budget artefact (every step gated by
  an LLM call ⇒ ~4,000 interactions vs 96,000 allowed) but reported as a substantive finding —
  [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156)

### Inferences
- The right architecture under zero reward is: **persistent state-hash graph + untested-action frontier +
  shortest-path return + a model-uncertainty score to decide where to spend the marginal action**. Each
  component is independently evidenced above. A scalar intrinsic bonus added to a value learner has no
  published evidence of working in a no-reset, single-life, reward-never-arrives regime, and has three named
  mechanisms (vanishing bonus, detachment, de-synchronisation) for degrading below systematic.
- "Vanishing bonuses" and "detachment" are *specifically* diseases of representing exploration state in
  network weights. A graph/archive cannot vanish or detach; it can only grow. Under a hard per-episode action
  budget with no reset, that monotonicity is worth more than sample efficiency.
- Irreversible actions must be modelled as first-class in the graph (a typed edge), not discovered accidentally
  — the 4-level regression is the empirical cost of not doing so.

### Gaps
- No head-to-head study found of bonus-based intrinsic motivation (RND/ICM) *versus* archive/graph exploration
  under a **single-life, no-reset** budget with matched interactions. The ARC-AGI-3 comparison is
  graph-vs-random-vs-LLM; RND/ICM are cited as motivation in that paper but not run as baselines.

---

## Q5. Does intrinsic motivation transfer to genuinely novel environments (not held-out levels of a training environment)?

### Takeaway
The evidence is negative-to-absent, and the 2025-2026 literature says so in its own words: intrinsic motivation
is "environment-centric in ways that can lead to overfitting to the specifics of the current environment".
Every method in this report that demonstrably works on *genuinely novel* games is **training-free or
learns-online-only** (graph explorer, OPINE-World); every method that accumulates weights either was not
evaluated cross-environment or, in the one ARC-AGI-3 case where it was, had trained on the very games it was
scored on.

### Cited Findings
- "While intrinsic motivation has demonstrably improved learning about the current environment, it remains
  fundamentally **environment-centric in ways that can lead to overfitting to the specifics of the current
  environment**"; "novelty seeking encourages visiting all environmental states, and learning progress drives
  the formation of a more accurate transition model **of the current environment**, but the focus remains on
  'what is out there to be known or done'" — [Agent-centric learning (2025), arXiv:2507.22255](https://arxiv.org/pdf/2507.22255)
- The 2026 survey lists **"Generalization Failures: intrinsic bonuses often fail to transfer across different
  tasks or environments"** as a standing limitation — [arXiv:2609.17325](https://arxiv.org/pdf/2609.17325)
- "Intrinsic methods can lead to suboptimal policies if internal reward signals diverge from task objectives"
  — [search summary of the intrinsic-exploration literature](https://umontreal.scholaris.ca/bitstreams/0c09d32e-5f12-4ee4-ac27-1adba9fddc3e/download)
- **The cleanest contamination example, on our exact benchmark:** OPINE-World's paper flags that the
  continual-learning **Vision** agent "explores the public game set offline before the scored run, so its scores
  are not a no-training comparison"; "its offline phase plays the same public games and grows the weights it
  then freezes, so the action counts in its scored run follow earlier passes over those games". It reaches
  ~63% raw on the public set and clears 12/25 — i.e. *worse* than OPINE-World's 20/25 despite the advantage —
  [arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)
- Conversely the two methods that work on unseen games carry **no learned weights about those games**:
  OPINE-World is "not trained, fine-tuned, or" adapted per game and "learns each game's mechanics online during
  the scored run" ([arXiv:2607.01531](https://arxiv.org/pdf/2607.01531)); the graph explorer is explicitly
  "training-free" and still ranks 3rd ([arXiv:2512.24156](https://arxiv.org/pdf/2512.24156))
- ARC-AGI-3's whole design premise is that generalisation must be to *novel* environments: the benchmark
  measures "skill-acquisition efficiency", agents get no instructions, and levels within a game deliberately
  add new mechanics so "knowledge transfer between levels could accelerate learning, but the levels are
  connected on" a progressively-changing mechanic set — [arXiv:2512.24156](https://arxiv.org/pdf/2512.24156); [ARC-AGI-3 technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)
- Positive-transfer counterpoint (older, and *within*-domain): **Motif** derives intrinsic reward from LLM
  preference feedback on NetHack and is the standard example of an intrinsic signal carrying prior knowledge
  rather than pure novelty — [arXiv:2310.00166](https://arxiv.org/pdf/2310.00166). **AMIGo** (adversarially
  motivated intrinsic goals) is the standard teacher-student variant — [arXiv:2006.12122](https://arxiv.org/pdf/2006.12122)

### Inferences
- There is no published evidence that a *learned* intrinsic-motivation module transfers to a genuinely novel
  environment. What transfers in the 2025-2026 results is **prior knowledge carried by a pretrained model**
  (IGE's notion of interestingness, OPINE-World's LLM synthesiser, Motif's LLM preferences) and
  **environment-agnostic algorithmic structure** (graph traversal, CEGIS, Dirichlet effect counts). Those are
  different things from a trained bonus network, and only the latter two are available to us on a game we have
  never seen.
- The Vision-vs-OPINE comparison on ARC-AGI-3 is close to a natural experiment: weights grown on the public
  games *and then scored on those same games* still lose to an online, weightless method on the full set. That
  is direct evidence against the "pretrain an exploration prior" route for this benchmark.

### Gaps
- No study found that measures the *same* intrinsic-motivation method on a train/test split of **distinct
  environments** (as opposed to procedurally-generated levels of one environment) with reported numbers. This
  is the biggest single gap in the literature relative to Q5 as asked; the claims of generalisation failure are
  stated in survey/position papers rather than backed by a controlled cross-environment benchmark in what was
  surfaced here.

---

## Cross-cutting summary table for the report writer

| Method | Year / venue | Reward needed to start? | Resets needed? | Headline number |
|---|---|---|---|---|
| Pseudo-counts (CTS) | 2016 NeurIPS | no (bonus), yes to learn a policy | **yes, episodic** | 15 vs 2 Montezuma rooms @50M frames; 6,600 pts @100M |
| RND | 2018/ICLR 2019 | no (bonus), yes to learn a policy | **yes, episodic** | Montezuma 8,152; 22/24 rooms |
| Go-Explore | 2019 / Nature 2021 | **yes — archive is scored by reward** | **yes — state restore (45× speedup) or a return policy** | Montezuma mean 469,209 / max >2M w/ domain knowledge; 35,410 w/o; Pitfall >21,000 (prior SOTA 0) |
| Intelligent Go-Explore | ICLR 2025 | **no** — FM judges interestingness | **yes** — returns to archived states | qualitative: exceeds RL + graph search; Reflexion fails |
| ESR (empowerment via successor features) | NeurIPS 2024 | no | not stated | beats prior empowerment, which loses to a random controller as complexity grows |
| Graph explorer (ARC-AGI-3) | arXiv 2512.24156, Dec 2025 | **no — training-free, no reward term** | **no — walks the shortest path back** | median 30/52 levels, 3rd on private LB; 19 vs 9 (random) vs 5 (GPT-4.1+DSL) @4k actions |
| OPINE-World | arXiv 2607.01531, Jul 2026 | **no — ontology error** | **no — "no reset to re-sample a level"** | 20/25 games, score 78.4 vs 63.8 baseline1, 63.2 Vision; WorldCoder & neural WMs 0/25 |
| GPT-6 Astra | ARC Prize, 2026 | n/a (frontier agent) | n/a | 62.7% semi-private (standard harness), 99.9% (provider adapter); 51.7% fewer actions/level than humans |
| Online graph exploration (theory) | 1990s-2026 | **no** | **no — return cost is charged** | NN / hierarchical DFS Θ(log n)-competitive, Ω(log n) lower bound; planar 16; directed d^O(log d)·m upper, Ω(d²m) lower |
| Reward-free RL theory | NeurIPS 2020 | no | yes (episodic) | Ω(2^H) samples for **deterministic** systems with linear Q* |
