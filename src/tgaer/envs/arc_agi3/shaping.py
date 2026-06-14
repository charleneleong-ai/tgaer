"""Granular / shaped reward for ARC-AGI-3 RL.

The binary WIN reward is too sparse on the hard games (every frozen baseline
scored 0/7 → zero advantage variance → no GRPO/GSPO gradient). These pure
functions add dense intermediate signal — exploration novelty and an anti-stall
penalty — computed from the per-step observation text the env already emits.

State contract (accumulated by ``observe`` once per env step, read by the reward
functions at rollout end). Everything is JSON-serializable so it survives
whatever verifiers does to rollout state.
"""

from __future__ import annotations

import hashlib

_GRID_MARKER = "grid:"


def extract_grid(observation: str) -> str:
    """The grid block of an observation — everything after the ``grid:`` marker,
    so volatile counters (num_actions, state) don't pollute the fingerprint."""
    if not observation:
        return ""
    idx = observation.find(_GRID_MARKER)
    return (
        observation[idx + len(_GRID_MARKER) :].strip()
        if idx >= 0
        else observation.strip()
    )


def fingerprint(observation: str) -> str:
    return hashlib.sha1(extract_grid(observation).encode()).hexdigest()


def observe(state: dict, observation: str) -> None:
    """Accumulate exploration stats for one env step. Novelty = a board not seen
    before this rollout; stall = the board is identical to the previous step."""
    seen: list[str] = state.setdefault("shaping_seen", [])
    fp = fingerprint(observation)
    state["shaping_steps"] = state.get("shaping_steps", 0) + 1
    if fp == state.get("shaping_last_fp"):
        state["shaping_stall_steps"] = state.get("shaping_stall_steps", 0) + 1
    if fp not in seen:
        state["shaping_novel_steps"] = state.get("shaping_novel_steps", 0) + 1
        seen.append(fp)
    state["shaping_last_fp"] = fp


def win_reward(state: dict) -> float:
    return 1.0 if state.get("game_state") == "WIN" else 0.0


def novelty_reward(state: dict) -> float:
    """Fraction of steps that revealed a board never seen before this rollout."""
    steps = state.get("shaping_steps", 0)
    return state.get("shaping_novel_steps", 0) / steps if steps else 0.0


def anti_stall_penalty(state: dict) -> float:
    """Negative fraction of steps that left the board unchanged (got stuck)."""
    steps = state.get("shaping_steps", 0)
    return -(state.get("shaping_stall_steps", 0) / steps) if steps else 0.0
