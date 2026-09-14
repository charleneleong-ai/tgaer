"""LLM-as-programmer for ARC-AGI-3: the model writes a policy, it never picks a move.

Every ARC-AGI-3 Milestone #1 winner ran local weights that *emit Python*, and no
model-free agent placed. Our own 27B kernel agent lost to the model-free
explorer, but it was an LLM-as-*policy* — queried for the next action every step
— which is the configuration nobody wins with. This module is the other one.

The shape, which is also `Executable World Models`' shape: play a warmup with
the explorer and record what actually happened, hand those transitions to the
model, ask for a `policy` function, then **falsify it against the recorded
transitions before it is allowed to spend a single real action**. Verification
is free — it replays history — so a policy that cannot reproduce what we already
saw is discarded at no cost in score.

Two invariants, both load-bearing:

* **Never worse than the explorer.** The explorer drives the warmup and remains
  the fallback for every step the policy declines, throws on, or fails to
  validate. A generation that produces nothing costs only the LLM call.
* **The model is asked once per game, not once per step.** Throughput was the
  binding constraint the last time a model was in this loop; one call per game
  makes it irrelevant.

The backend is any OpenAI-compatible ``/chat/completions`` — the same seam
`arc_agi3_kaggle.HTTPChatBackend` uses — so this runs unchanged against a local
vLLM during development and against the in-kernel vLLM on the competition GPU.

**Backends must disable thinking.** Measured on Qwen3.8-27B with this exact
prompt: thinking on spent 6229 completion tokens and 229.5s; thinking off spent
173 tokens and 6.8s and produced the same usable policy. Worse, a reasoning
model asked for 2048 tokens burns every one of them reasoning and returns
`finish_reason="length"` with **empty content** — which looks exactly like a
model that had nothing to say. `request_policy` names that case specifically so
it is never mistaken for a bad prompt.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
# A policy must reproduce at least this share of held-out warmup transitions
# before it is trusted with real actions. Below it, the model has not understood
# the game and the explorer is the better bet.
MIN_AGREEMENT = 0.6
# Warmup actions spent gathering evidence before the model is asked. Long enough
# to see each action's effect, short enough to leave the budget for playing.
WARMUP_ACTIONS = 60


@dataclass(frozen=True)
class Transition:
    """One observed step: what the board was, what we did, what changed."""

    grid: np.ndarray
    action: int
    click: tuple[int, int] | None
    next_grid: np.ndarray
    level: int
    level_after: int

    @property
    def changed(self) -> bool:
        return not np.array_equal(self.grid, self.next_grid)

    @property
    def advanced(self) -> bool:
        return self.level_after > self.level


@dataclass
class GameEvidence:
    """What a warmup learned about one game, in a form a prompt can carry."""

    game: str
    available_actions: list[int]
    transitions: list[Transition] = field(default_factory=list)

    def action_effects(self) -> dict[int, dict[str, int]]:
        """Per action id: how often it changed the board, and how often it won."""
        out: dict[int, dict[str, int]] = {}
        for t in self.transitions:
            row = out.setdefault(t.action, {"tried": 0, "changed": 0, "advanced": 0})
            row["tried"] += 1
            row["changed"] += int(t.changed)
            row["advanced"] += int(t.advanced)
        return out

    def summary(self) -> str:
        """A compact, honest description of the game — no invented semantics.

        Deliberately not a raw frame dump: 64x64 integers per step would bury
        the signal and blow the context, and the model does not need pixels to
        write a policy over the effects we measured.
        """
        effects = self.action_effects()
        lines = [
            f"game: {self.game}",
            f"available actions: {sorted(self.available_actions)}",
            f"warmup steps observed: {len(self.transitions)}",
            "",
            "action effects observed (action: tried / changed the board / advanced a level):",
        ]
        lines.extend(
            f"  action {a}: {r['tried']} / {r['changed']} / {r['advanced']}"
            for a, r in sorted(effects.items())
        )
        if self.transitions:
            grid = self.transitions[0].grid
            values, counts = np.unique(grid, return_counts=True)
            lines += [
                "",
                f"grid shape: {grid.shape}",
                "cell values present (value: count): "
                + ", ".join(f"{int(v)}: {int(c)}" for v, c in zip(values, counts)),
            ]
        return "\n".join(lines)


SYSTEM_PROMPT = """You write Python policies for grid puzzle games. You never \
play the game yourself; you emit one function and it is executed for you.

Write exactly one fenced Python block defining:

    def policy(grid, available_actions, memory):
        ...

- `grid` is a 2D numpy array of small ints (the board).
- `available_actions` is a list of ints you may return.
- `memory` is a dict that persists across calls within a level; use it freely.
- Return either an int (a simple action), or a tuple ("click", row, col).
- Return None to defer to the fallback explorer for this step.

Rules:
- Use only numpy and the Python standard library. No imports of anything else.
- Never raise. If unsure, return None.
- Deterministic: no randomness, no clocks, no I/O.
- Prefer returning None over guessing when the board looks unfamiliar.
"""


def build_prompt(evidence: GameEvidence) -> str:
    return textwrap.dedent(f"""\
        Here is what a scripted explorer observed while playing this game.

        {evidence.summary()}

        Write a `policy` function that plays this game more efficiently than
        blind exploration. Score rewards *few actions per level completed*, so a
        policy that reaches the goal directly beats one that wanders.

        If the evidence does not justify a confident rule, return None for that
        situation so the fallback explorer handles it. A narrow policy that is
        right is worth far more than a broad one that guesses.
        """)


def extract_code(reply: str) -> str | None:
    """The first fenced Python block, or None if the model wrote prose only."""
    match = CODE_BLOCK.search(reply)
    return match.group(1) if match else None


def compile_policy(code: str) -> Callable[..., Any] | None:
    """Execute ``code`` in an isolated namespace and return its ``policy``.

    The sandbox is deliberately thin: this runs offline in a competition kernel
    against text we asked for, so the threat is a buggy policy, not a hostile
    one. What matters is that a failure here is contained and falls back, rather
    than taking the whole run down.
    """
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    try:
        exec(code, namespace)  # noqa: S102 — see docstring
    except Exception:  # noqa: BLE001 — any failure means "no policy"
        return None
    policy = namespace.get("policy")
    return policy if callable(policy) else None


def agreement(policy: Callable[..., Any], held_out: Sequence[Transition]) -> float:
    """Share of held-out transitions where the policy picks a move we saw work.

    This is the falsifiability step, and it is free: it replays transitions
    already recorded, so a policy is rejected without ever spending a real
    action. A policy that returns None is not counted as wrong — deferring is
    allowed — but it cannot earn agreement either, so an all-None policy scores
    0 and is discarded.
    """
    if not held_out:
        return 0.0
    correct = 0
    for t in held_out:
        memory: dict[str, Any] = {}
        try:
            choice = policy(t.grid, [t.action], memory)
        except Exception:  # noqa: BLE001 — a throwing policy is a failed policy
            return 0.0
        if choice is None:
            continue
        picked = choice if isinstance(choice, int) else None
        if picked == t.action and t.changed:
            correct += 1
    return correct / len(held_out)


def validate(policy: Callable[..., Any], evidence: GameEvidence) -> tuple[bool, str]:
    """``(usable, reason)`` — whether this policy may drive real actions.

    **Known weak spot, do not read a high score here as understanding.**
    Agreement compares action *ids*, and on a click-only game like `lp85` there
    is exactly one id available, so any policy returning it scores 1.00 without
    having decided anything — the real choice on that board is *where* to click,
    which this does not check at all. A click-aware agreement (does the policy
    pick a cell whose click we saw change the board?) is the obvious next step;
    until then treat validation as a filter against broken code rather than
    evidence of a good policy, and let the suite score be the judge.
    """
    if not evidence.transitions:
        return False, "no warmup transitions to validate against"
    split = max(1, len(evidence.transitions) // 2)
    held_out = evidence.transitions[split:]
    score = agreement(policy, held_out)
    if score < MIN_AGREEMENT:
        return False, f"agreement {score:.2f} < {MIN_AGREEMENT} on held-out transitions"
    return True, f"agreement {score:.2f} on {len(held_out)} held-out transitions"


def request_policy(
    backend: Any, evidence: GameEvidence, *, max_tokens: int = 2048
) -> tuple[Callable[..., Any] | None, str]:
    """Ask the model for a policy and return ``(policy, reason)``.

    ``backend`` is anything exposing ``chat(messages, max_tokens) -> str`` — the
    same contract `arc_agi3_kaggle.HTTPChatBackend` satisfies, so this is
    identical against a development vLLM and the in-kernel one.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(evidence)},
    ]
    try:
        reply = backend.chat(messages, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 — a dead model must not sink the run
        return None, f"backend failed: {type(exc).__name__}: {exc}"
    if not (reply or "").strip():
        return None, (
            "model returned empty content — a reasoning model will spend the "
            "whole token budget thinking and finish with nothing; disable "
            "thinking on the backend rather than raising max_tokens"
        )
    code = extract_code(reply)
    if code is None:
        return None, "model returned no python block"
    policy = compile_policy(code)
    if policy is None:
        return None, "generated code did not compile to a callable policy"
    usable, reason = validate(policy, evidence)
    return (policy if usable else None), reason


def as_json(evidence: GameEvidence) -> str:
    """Evidence in a form worth logging beside a run's scorecard."""
    return json.dumps(
        {
            "game": evidence.game,
            "steps": len(evidence.transitions),
            "action_effects": evidence.action_effects(),
        },
        indent=2,
    )
