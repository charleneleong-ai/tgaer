"""LLM-as-programmer for ARC-AGI-3: the model writes a policy, it never picks a move.

Every ARC-AGI-3 Milestone #1 winner ran local weights that *emit Python*, and no
model-free agent placed. Our own 27B kernel agent lost to the model-free
explorer, but it was an LLM-as-*policy* — queried for the next action every step
— which is the configuration nobody wins with. This module is the other one.

The shape, which is also `Executable World Models`' shape: play a warmup with
the explorer and record what actually happened, hand those transitions to the
model, ask for a `policy` function, then **falsify it against the recorded
transitions before it is allowed to spend a single real action**. Verification
is free — it replays history — so a policy that proposes what we already watched
fail is discarded at no cost in score. Note the direction: this falsifies, it
does not imitate. A policy is meant to *beat* the explorer, so being scored on
reproducing the explorer's moves would reward the wrong thing.

The bar is **progress, not change**. An option counts only where the warmup saw
it clear a level or reach a board state never visited before — with state
identity taken after masking self-animating chrome, without which lp85 reads as
95% novel while looping on one cell. Judging mere change scored every generated
policy a perfect 1.00, including ones that destroyed levels.

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
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
# Share of *judged* choices that must avoid known-dead options. This is a
# falsification bar, not an imitation bar — see `productivity`.
MIN_PRODUCTIVITY = 0.8
# A policy that defers on nearly everything is not a policy. It must commit on
# at least this share of held-out states, and at least this many must be ones we
# have evidence about, or there is nothing to judge and it is rejected.
MIN_COMMIT_RATE = 0.25
MIN_JUDGED = 5
# How many times an option must be seen doing nothing before it counts as dead.
# One no-op can be a transient; a cell clicked repeatedly to no effect is not.
DEAD_AFTER = 2
# A cell changing on more than this share of warmup steps is board chrome — an
# animated border or counter that moves on its own. It has to be masked out
# before novelty means anything: lp85's level 2 shows 95% "novel" frames while
# the agent re-clicks one cell, purely because a ring recolours every step.
# Masking is safe here in a way it was not inside the search: this runs offline
# against recorded history, so it cannot perturb a rollout.
CHROME_FRACTION = 0.5
# Warmup steps needed before the chrome mask is trusted. Below it, too few
# samples to tell an animation from a consequence, so nothing is masked.
CHROME_WARMUP = 20
# If a mask would cover more than this share of the board, the board is not
# animated — the game simply redraws a lot — and masking it would collapse every
# state into one, making all progress invisible. Better to mask nothing.
MAX_CHROME_SHARE = 0.25
# Share of an option's observed outcomes that must be a level-up or a genuinely
# new state for it to count as progress. An option that reaches somewhere new
# once and then loops for the rest of the warmup is not a way forward.
PROGRESS_RATE = 0.5
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

    def option(self, transition: Transition) -> tuple[Any, ...]:
        """The choice a policy would have to make to reproduce this transition."""
        if transition.click is not None:
            return ("click", *transition.click)
        return ("act", transition.action)

    def chrome_mask(self) -> np.ndarray | None:
        """Cells that animate on their own, or None while evidence is too thin."""
        if len(self.transitions) < CHROME_WARMUP:
            return None
        shape = self.transitions[0].grid.shape
        counts = np.zeros(shape, dtype=np.int32)
        for t in self.transitions:
            if t.grid.shape == shape and t.next_grid.shape == shape:
                counts += t.grid != t.next_grid
        mask = counts > CHROME_FRACTION * len(self.transitions)
        if not mask.any() or mask.mean() > MAX_CHROME_SHARE:
            return None
        return mask

    @staticmethod
    def settled_key(grid: np.ndarray, chrome: np.ndarray | None) -> bytes:
        """A state identity that ignores self-animating cells."""
        if chrome is None or chrome.shape != grid.shape:
            return grid.tobytes()
        return np.where(chrome, 0, grid).tobytes()

    def outcomes(self) -> dict[tuple[Any, ...], Counter[str]]:
        """Per option, how its observed effects break down.

        Four outcomes, and the distinction between the middle two is the whole
        point: ``advanced`` cleared a level, ``novel`` reached a board
        configuration never seen before, ``cyclic`` returned to one already
        visited, and ``dead`` changed nothing at all once chrome is masked.

        `cyclic` is what change-detection cannot see. lp85 spent 542 actions on
        a cell that changed the board every time and went nowhere; under a
        chrome-masked state key that behaviour is visibly a loop, which is
        exactly what "judge progress, not change" has to mean here.
        """
        chrome = self.chrome_mask()
        seen: set[bytes] = set()
        out: dict[tuple[Any, ...], Counter[str]] = {}
        for t in self.transitions:
            before = self.settled_key(t.grid, chrome)
            after = self.settled_key(t.next_grid, chrome)
            seen.add(before)
            row = out.setdefault(self.option(t), Counter())
            if t.advanced:
                row["advanced"] += 1
            elif after == before:
                row["dead"] += 1
            elif after in seen:
                row["cyclic"] += 1
            else:
                row["novel"] += 1
            seen.add(after)
        return out

    @staticmethod
    def progress_rate(row: Counter[str]) -> float:
        """Share of an option's outcomes that went somewhere."""
        total = sum(row.values())
        return (row["advanced"] + row["novel"]) / total if total else 0.0

    def progressive_options(self) -> set[tuple[Any, ...]]:
        """Options that mostly clear levels or reach genuinely new states."""
        return {
            opt
            for opt, row in self.outcomes().items()
            if self.progress_rate(row) >= PROGRESS_RATE
        }

    def unproductive_options(self) -> set[tuple[Any, ...]]:
        """Options seen enough times to judge that mostly loop or do nothing.

        A rate rather than a flat "never went anywhere": an option that reaches
        somewhere new once and then cycles for the rest of the warmup is not a
        way forward, and under the old binary test it escaped judgement
        entirely.
        """
        return {
            opt
            for opt, row in self.outcomes().items()
            if sum(row.values()) >= DEAD_AFTER
            and self.progress_rate(row) < PROGRESS_RATE
        }

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


def as_click(choice: Any) -> tuple[int, int] | None:
    """``("click", row, col)`` as a cell, or None if this is not a click."""
    if isinstance(choice, (tuple, list)) and len(choice) == 3 and choice[0] == "click":
        try:
            return int(choice[1]), int(choice[2])
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class Productivity:
    """How a policy fared against what the warmup proved about this game."""

    judged: int
    productive: int
    committed: int
    total: int

    @property
    def score(self) -> float:
        return self.productive / self.judged if self.judged else 0.0

    @property
    def commit_rate(self) -> float:
        return self.committed / self.total if self.total else 0.0


def productivity(
    policy: Callable[..., Any],
    held_out: Sequence[Transition],
    evidence: GameEvidence,
    available: Sequence[int],
) -> Productivity:
    """Falsify a policy against what the warmup already proved — never imitate it.

    Two earlier versions of this were too weak, and each failed differently.
    Asking whether the policy reproduced the explorer's action was wrong twice
    over: the policy is supposed to *beat* the explorer, and it was handed
    ``[t.action]``, so it was shown the answer. Replacing that with "avoids
    options that change nothing" then scored a perfect 1.00 on every game that
    produced a policy at all — including two that went on to destroy levels —
    because on these boards almost every click changes *something*.

    So the bar is progress, not change. An option earns credit only where the
    warmup saw it clear a level or reach a board configuration never visited
    before, with state identity taken after masking self-animating chrome.
    Options that only ever looped or did nothing are counted against, which is
    the lp85 behaviour — 542 actions of real, cycling change — made checkable.
    Choices we have no evidence for are skipped rather than guessed at, so
    `judged` is reported separately and a minimum is required.
    """
    progressive = evidence.progressive_options()
    unproductive = evidence.unproductive_options()
    judged = productive = committed = 0

    for t in held_out:
        memory: dict[str, Any] = {}
        try:
            choice = policy(t.grid, list(available), memory)
        except Exception:  # noqa: BLE001 — a throwing policy is a failed policy
            return Productivity(judged=0, productive=0, committed=0, total=len(held_out))
        if choice is None:
            continue
        committed += 1

        if (cell := as_click(choice)) is not None:
            option: tuple[Any, ...] = ("click", *cell)
        elif isinstance(choice, int):
            if choice not in available:
                judged += 1  # proposing an unavailable action is always wrong
                continue
            option = ("act", choice)
        else:
            judged += 1  # a malformed answer is a wrong one
            continue

        if option in progressive:
            judged += 1
            productive += 1
        elif option in unproductive:
            judged += 1
    return Productivity(judged, productive, committed, len(held_out))


def distinct_states(held_out: Sequence[Transition]) -> int:
    """How many genuinely different boards the policy is shown."""
    return len({t.grid.tobytes() for t in held_out})


def distinct_choices(
    policy: Callable[..., Any], held_out: Sequence[Transition], evidence: GameEvidence
) -> int:
    """How many different answers the policy gives across those boards."""
    answers: set[Any] = set()
    for t in held_out:
        try:
            choice = policy(t.grid, list(evidence.available_actions), {})
        except Exception:  # noqa: BLE001 — a throwing policy has no answers
            return 0
        if choice is not None:
            answers.add(as_click(choice) or (choice if isinstance(choice, int) else None))
    return len(answers)


def validate(policy: Callable[..., Any], evidence: GameEvidence) -> tuple[bool, str]:
    """``(usable, reason)`` — whether this policy may drive real actions.

    Three independent bars, because each catches a different bad policy: it must
    commit often enough to be worth running at all, enough of those commitments
    must be ones we can judge, and enough of the judged ones must avoid options
    the warmup proved dead. Passing is still only a licence to *try* — the suite
    score gated by `bench/gate.py` is what decides whether it ships.
    """
    if not evidence.transitions:
        return False, "no warmup transitions to validate against"
    split = max(1, len(evidence.transitions) // 2)
    held_out = evidence.transitions[split:]
    result = productivity(policy, held_out, evidence, evidence.available_actions)


    detail = (
        f"productivity {result.score:.2f} on {result.judged} judged "
        f"of {result.committed} committed / {result.total} held-out"
    )
    if result.commit_rate < MIN_COMMIT_RATE:
        return False, f"defers too often ({result.commit_rate:.2f} commit rate) — {detail}"

    # A constant policy is a loop by construction, and it is invisible to
    # `productivity`: one known-good option, replayed on every held-out state,
    # scores a perfect 1.00 while doing the same thing forever. Both earlier
    # bars scored exactly 1.00 on every game that produced a policy at all, and
    # this is why. If the board varies and the answer does not, it is not a
    # policy — it is a constant wearing one.
    if distinct_states(held_out) > 1 and distinct_choices(policy, held_out, evidence) < 2:
        return False, "ignores the board — same choice on every held-out state"
    if result.judged < MIN_JUDGED:
        return False, f"too little evidence to judge it ({result.judged} < {MIN_JUDGED}) — {detail}"
    if result.score < MIN_PRODUCTIVITY:
        return False, f"picks known-dead options — {detail}"
    return True, detail


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
