# ARC-AGI-3 Landscape Analysis — August 2026

## Executive Summary

ARC-AGI-3 is an interactive reasoning benchmark where agents must explore novel environments, infer goals, build world models, and plan actions without instructions. **Humans solve 100% of environments; frontier LLMs score <1%; purpose-built agents reach ~30%.**

---

## Current Leaderboard (August 2026)

| Rank | Model | Lab | Score | Notes |
|------|-------|-----|-------|-------|
| 1 | Claude Opus 5 | Anthropic | **30.2%** | Leading purpose-built agent |
| 2 | GPT-5.6 Sol | OpenAI | 7.8% | |
| 3 | Claude Opus 4.8 | Anthropic | 1.5% | |
| 4 | GPT-5.6 Terra | OpenAI | 0.8% | |
| 5 | GPT-5.5 | OpenAI | 0.4% | |
| 6 | Gemini 3.1 Pro | Google | 0.4% | |
| 7 | Grok 4.5 | xAI | 0.3% | |
| 8 | GPT-5.4 | OpenAI | 0.2% | |
| 9 | GPT-5.6 Luna | OpenAI | 0.2% | |
| 10 | Claude Opus 4.7 | Anthropic | 0.2% | |
| 11 | Grok 4.20 | xAI | 0.1% | |

**Key insight:** Raw LLM scale does NOT buy interactive competence. The top score (30.2%) comes from a purpose-built agent, not a frontier model used directly.

---

## Top Approaches from Developer Preview

### 1. StochasticGoose (Tufa Labs) — 12.58%, 1st Place
- **Architecture:** CNN with reinforcement learning
- **Key innovation:** Predicts which actions will cause frame changes
- **How it works:**
  - 4-layer CNN backbone (32→64→128→256 channels)
  - Binary classification: will this action change the frame?
  - Hierarchical sampling: first select action type, then coordinates if ACTION6
  - 200K unique state-action pairs with hash-based deduplication
  - Dynamic model reset when reaching new levels
- **Why it works:** Biases exploration toward actions that actually matter, avoiding wasted actions on inert moves

### 2. Blind Squirrel — 6.71%, 2nd Place
- **Architecture:** Directed state graphs from observed frames
- **Key innovation:** Graph-based exploration of reachable states
- **How it works:** Builds explicit state-space graph, uses graph properties to identify winning paths

### 3. Symbolica AI's Arcgentica — Solved all 3 public environments
- **Architecture:** Orchestrator–subagent architecture
- **Key innovation:** Top-level orchestrator delegates to specialized subagents
- **How it works:**
  - Orchestrator doesn't interact with environment directly
  - Subagents return compressed textual summaries
  - Constrains context growth
  - Maintains higher-level plan without exceeding context limits

---

## Why Our Click Sweep Scored 0

### The Problem
Our click agent (`PlannerArcAgi3Agent` with `CLICK_DEFAULT`) scored 0/19 ACTION6 games. The controller's key→door abstraction doesn't match the diverse click game mechanics.

### Root Causes
1. **Games have richer mechanics than one-click toggle**
   - ft09: background pockets / top-row legend tiles (inert targets)
   - r11l: large cascade reactions (not simple toggle)
   - Most click games require multi-step interaction sequences

2. **The controller assumes a key→door pattern**
   - Click games don't follow this pattern
   - Need game-specific exploration and modeling

3. **HTTP 400 errors on 3 games (s5i5, tn36, r11l)**
   - Games advertise ACTION6 but reject the click
   - Possible reasons: different coordinate system, game-specific validation, or the click mechanic works differently than expected

### What's Missing
- **Exploration:** Need to try different click targets and observe effects
- **Modeling:** Need to build a world model of how clicks affect the environment
- **Goal inference:** Need to figure out what "winning" means in each game
- **Adaptive planning:** Need to adjust strategy based on feedback

---

## Technical Insights from the Paper

### Core Capabilities Tested
1. **Exploration:** Actively gather information by interacting with surroundings
2. **Modeling:** Turn raw observations into a generalizable world model
3. **Goal-Setting:** Identify desirable future states without explicit instructions
4. **Planning & Execution:** Map action paths from current state to identified goals

### Scoring Methodology (RHAE)
- **Action efficiency:** Number of actions to solve a task on first contact
- **Power-law scoring:** Efficiency is squared (penalizes inefficiency heavily)
- **Level weighting:** Later levels count more (linear weight)
- **Per-environment cap:** Max score = fraction of levels completed

### Environment Design Constraints
- **Core Knowledge priors only:** Objectness, basic geometry, basic physics, agentness
- **No language or cultural symbols:** No numbers, letters, recognizable clip-art
- **Novelty:** Must be novel vs existing games and previously created environments
- **Human solvable:** Must be solvable within ~20 minutes
- **Difficulty through composition:** Later levels combine concepts learned earlier

---

## ACTION6 API Details

From the official docs:
- **Purpose:** "click/tap at (x,y)", "place a tile", "shoot a projectile"
- **Coordinates:** Zero-based grid coordinates (0-63 inclusive)
- **x:** Horizontal coordinate (0 = left, 63 = right)
- **y:** Vertical coordinate (0 = top, 63 = bottom)
- **400 errors:** Unknown game_id, guid not found, x/y outside 0-63 range

---

## What We Need to Add to Our Plan

### Immediate: Game-Specific Exploration Agent
Instead of a generic click agent, we need an agent that:
1. **Explores systematically:** Try different actions and observe effects
2. **Builds a state model:** Track what changes after each action
3. **Infers goals:** Identify what constitutes "progress" in each game
4. **Adapts strategy:** Change approach based on what works

### Medium-Term: Hybrid Architecture
Combine our strengths with winning approaches:
1. **CNN-based change prediction** (like StochasticGoose)
2. **Graph-based state exploration** (like Blind Squirrel)
3. **Orchestrator-subagent decomposition** (like Arcgentica)
4. **Our existing controller** for LS20-family games

### Long-Term: RL Training
The vendored `arc-agi-3-env` with GRPO training is the right direction, but needs:
1. Better reward shaping (not just level completion)
2. Exploration bonuses (try novel actions)
3. Curriculum learning (start with easier games)

---

## References

- [ARC-AGI-3 Technical Paper](https://arxiv.org/abs/2603.24621)
- [ARC-AGI-3 Docs](https://docs.arcprize.org/)
- [StochasticGoose Solution](https://github.com/DriesSmit/ARC3-solution)
- [Official Leaderboard](https://arcprize.org/leaderboard)
- [BenchLM Leaderboard](https://benchlm.ai/benchmarks/arcagi3)
