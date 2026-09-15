"""Kaggle submission wrapper for the REPL agent with local LLM.

Adapts our ArcAgi3ReplAgent to the Kaggle ARC-AGI-3 Agent interface. Inference
comes from whichever backend the notebook configures: a vLLM server over HTTP
(the default) or a local llama-cpp model pool.

Usage:
    1. Copy this file to the ARC-AGI-3-Kaggle-Starter repo as agent/my_agent.py
    2. Run: make play-local
    3. Run: make submit
"""

from __future__ import annotations

import base64
import builtins
import io
import json
import os
import random
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

from arcengine import FrameData, GameAction, GameState

# Models are pooled, not per-agent. The Swarm runs one thread per game (110 in
# the competition rerun) and each thread once loaded its own Llama() — 110x9GB
# OOMs any GPU and fails the run with a generic "system error". One shared
# instance fixed that but serialized every game behind a single lock.
#
# LLAMA_POOL_SIZE instances are shared instead: a llama.cpp instance must be
# used by one thread at a time, but batch-1 decoding leaves most of the GPU
# idle, so concurrent instances raise aggregate throughput until VRAM runs out.
# Defaults to 1, which is exactly the old behaviour; the notebook raises it to
# suit the GPU it actually gets.
POOL_SIZE = max(1, int(os.environ.get("LLAMA_POOL_SIZE", "1")))
_INFERENCE_LOCK = threading.Lock()

# Wall-clock budget for the whole run, shared by every game. The old per-game
# limit started at agent construction — i.e. simultaneously for all games — while
# inference is globally serialized, so N concurrent games split one window N ways
# and most of the notebook budget went unused. One global deadline lets games use
# the full budget and still stops in time to close the scorecard. Anchored at
# import, which happens in the play subprocess *after* setup, so it errs short.
RUN_BUDGET_S = float(os.environ.get("ARC_RUN_BUDGET_S", str(7.5 * 3600)))
_RUN_DEADLINE = time.monotonic() + RUN_BUDGET_S

ARC_SYMBOLS: dict[int, str] = {
    0: ".",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
    13: "M",
    14: "N",
    15: "O",
}
ARC_LEGEND = "'.'=0 (empty), A-O=values 1-15; hex letter if >15"

# RESET (0) starts/restarts the game: the gateway returns an empty frame while
# NOT_PLAYED, so the very first action MUST be a reset or the model plays blind.
ACTION_NAMES: dict[int, str] = {
    0: "RESET",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "SPACE",
    6: "MOUSE",
    7: "ACTION7",
}
NAME_TO_ID: dict[str, int] = {v: k for k, v in ACTION_NAMES.items()}

COMPLEX_ACTION_ID = 6
GRID_SIZE = 64
# One action call is ~12 tokens raw-text, ~30 as a tool call. This is a guard
# against a rambling generation, not a target — generation stops at EOS. Kept
# well above the need because hitting the cap is indistinguishable from a
# refusal: a reasoning model burned all 64 tokens on <think> and returned
# nothing usable at all.
MAX_OUTPUT_TOKENS = int(os.environ.get("ARC_MAX_OUTPUT_TOKENS", "128"))
# "required" forces a tool call through guided decoding; "auto" leaves the model
# to emit a <tool_call> block the server's parser picks up. Which one works is a
# property of the serving stack, not of us: the scored kernel installs vLLM
# 0.19 from a fixed wheelhouse while local runs are on 0.26, and the preflight
# probes both and sets this to whichever actually returns a call.
TOOL_CHOICE = os.environ.get("ARC_TOOL_CHOICE", "required")
# Python inspections allowed before an action must be chosen. Measured at 2:
# throughput fell 163 -> 41 actions/min (168 projected actions per game, down
# from 668) and levels completed stayed at 0, so the loop cost four turns' worth
# of budget and bought nothing yet. Off by default until it earns its place;
# raise it to A/B the sandbox again once the cheaper levers are settled.
REPL_STEPS = int(os.environ.get("ARC_REPL_STEPS", "0"))
# Send the board as a picture as well as text. The served model has a vision
# tower and the 2nd/3rd place entries used rendered images alone; 0 disables it
# so the mock can measure whether it actually helps.
SEND_IMAGE = os.environ.get("ARC_SEND_IMAGE", "1") not in {"0", "false", ""}
# After this many consecutive actions that change nothing, stop asking and try
# an action family the model has not tried on this board. Traced on sk48: 41
# turns, every one a MOUSE click inside a 4x4 patch, 98% with no effect, while
# UP/DOWN/LEFT/RIGHT sat untried. Telling the model was not enough — it kept
# clicking — so the escape has to be taken rather than suggested. Costs no
# inference, and the score squares wasted actions.
ESCAPE_AFTER = int(os.environ.get("ARC_ESCAPE_AFTER", "3"))
# Times to repeat an action that is working before handing back to the model.
# A published survey of this competition's public games found 8 of 25 are won
# by one action repeated 50-200 times; the agent never repeated anything, so
# persistence is worth more here than another opinion per turn. Each repeat
# also costs no inference.
EXPLOIT_REPEATS = int(os.environ.get("ARC_EXPLOIT_REPEATS", "8"))
# The model must get a turn at least this often. Exploiting re-armed itself on
# every apparent success, and early in a level a ticking timer looks like
# success because HUD detection needs a few transitions before it engages — so
# cn04 ran 201 actions in 1.0s without a single inference, and tn36 did the
# same before it. Cheap policies may fill the gaps; they may not take the game.
MAX_POLICY_STREAK = int(os.environ.get("ARC_MAX_POLICY_STREAK", "10"))
# Probe every available action once at the start of a level before asking the
# model to choose. ~6 actions out of a ~400 budget buys a real effect model.
PROBE_ACTIONS = os.environ.get("ARC_PROBE", "1") not in {"0", "false", ""}
# Try what has not been tried in this state, and walk back to a state that
# still has something untried. Ported from the explorer, where instrumenting
# its two wins showed both came from this frontier and not from its avatar
# induction — lp85 cleared at step 19 with the avatar unpinned and an empty
# move lattice. Random play with five seeds clears neither game, so the
# coverage is doing work that undirected sampling does not.
# On, decided by a 3-seed A/B over all 25 games. With the frontier the agent
# cleared lp85 on 3 of 3 runs, deterministically (172 frontier actions, 18 model
# calls, identical across seeds); without it, 1 of 2 valid runs — the third
# control is excluded because 20% of its actions hit choose_action_exception
# when the backend degraded.
#
# Read that at its real strength. Only one game was cleared by either arm in six
# runs, it is the game whose click targets were tuned against, and the 16 games
# never inspected during development produced 0 levels from 6782 frontier
# actions. So this buys reliability on one known game, not a general capability,
# and the honest expectation on unseen games is no contribution at all. It is
# still the only configuration measured to beat the alternative on anything.
FRONTIER = os.environ.get("ARC_FRONTIER", "1") not in {"0", "false", ""}
# Let the model carry one sentence of its own understanding across turns. Every
# other note in the prompt is harness bookkeeping — dead actions, budget, chrome
# — so the model re-derives the mechanic ("arrows push tiles") from scratch on
# every turn. orak's 2048 agent is handed the rules outright; ARC-AGI-3 hides
# them by design, so the nearest equivalent is a note the model writes itself.
# One evolving line, not a log: history is what overflowed n_ctx in v31.
# Off by default: a 3v3 A/B scored 0 levels either way. It fires reliably and
# writes true rules, and it halved blind repeats, but none of that converted
# into a completed level, and an over-long note can still truncate the tool call
# that carries it (MAX_OUTPUT_TOKENS is shared with the thinking block). Kept
# behind the flag rather than deleted, so the arm can be re-run cheaply.
MECHANIC_NOTES = os.environ.get("ARC_MECHANIC_NOTES", "0") not in {"0", "false", ""}
MAX_NOTE_CHARS = int(os.environ.get("ARC_MAX_NOTE_CHARS", "240"))
IMAGE_CELL_PX = int(os.environ.get("ARC_IMAGE_CELL_PX", "8"))
PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run Python against the board and print what you want to know. "
            "Available: grid (tuple of rows of ints), objects (segmentation dict "
            "with 'nodes' and 'adjacency'), prev (previous board or None), "
            "SYMBOLS (int->char). "
            "Use print(); each call starts fresh."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python to run"}},
            "required": ["code"],
        },
    },
}

# Qwen3 reasons by default and emits <think>...</think> before the answer. At
# MAX_OUTPUT_TOKENS that reasoning is what gets truncated, so the action never
# arrives. "/no_think" is Qwen3's switch for turning it off; Qwen3.6 ignores it
# and needs chat_template_kwargs instead (see HTTPChatBackend), so both are set.
NO_THINK = " /no_think"
TOOL_SYSTEM = (
    "You are a game agent. Each turn call exactly one action function that best "
    "progresses the game, based on the board and the feedback in the user message."
    + NO_THINK
)
RAW_TEXT_SYSTEM = (
    "You solve grid puzzles by choosing actions. "
    "ALWAYS output EXACTLY ONE line starting with action([...]). "
    "Nothing else. No explanation. Just the action call." + NO_THINK
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop complete <think> blocks, then any stray tags.

    A stray opening tag must NOT discard everything after it: the reply seen on
    the competition GPU was `<think>\\n\\n<tool_call>{...}</tool_call>` — an
    unclosed tag wrapped around the real answer. Dropping the remainder there
    would throw away the action.
    """
    text = THINK_BLOCK_RE.sub("", text)
    return text.replace("<think>", "").replace("</think>", "").strip()


def parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Recover tool calls that the chat template left as raw text.

    llama-cpp's qwen3 template does not always populate `tool_calls`: the model
    emits a correct `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
    block and the field comes back empty. Observed on the competition GPU, where
    it silently disabled the entire tool-calling path.
    """
    calls: list[dict[str, Any]] = []
    for blob in TOOL_CALL_RE.findall(text):
        try:
            parsed = json.loads(blob)
        except ValueError:
            continue
        if name := parsed.get("name"):
            calls.append(
                {
                    "function": {
                        "name": name,
                        "arguments": json.dumps(parsed.get("arguments") or {}),
                    }
                }
            )
    return calls


def clamp_coord(v: int) -> int:
    return max(0, min(GRID_SIZE - 1, v))


Grid = tuple[tuple[int, ...], ...]

# The ARC palette. Distinct hues so adjacent values stay separable once the
# board is scaled down to a few hundred pixels.
ARC_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
    10: (100, 65, 165),
    11: (0, 128, 128),
    12: (255, 255, 255),
    13: (60, 60, 60),
    14: (200, 120, 60),
    15: (80, 200, 180),
}


# Why the last render failed, so a silently disabled image path can be
# diagnosed from the run log rather than guessed at.
_RENDER_FAILURE: str = ""


def render_board_png(
    grid: Grid, cell_px: int = 8, gridline_every: int = 8
) -> bytes | None:
    """Draw the board as a PNG, or None if PIL is unavailable.

    The served model is `Qwen3_5ForConditionalGeneration` — it has a vision
    tower, and the second and third place entries in this competition drove a
    model with rendered images alone. Spatial relations that take a paragraph to
    describe in ASCII are immediate in a picture.

    Faint gridlines every `gridline_every` cells give the model something to
    count against, since MOUSE actions need absolute coordinates.
    """
    global _RENDER_FAILURE
    try:
        # Image only. ImageDraw is deliberately avoided: the competition image
        # ships a pillow whose ImageDraw raises
        # "cannot import name '_Ink' from 'PIL._typing'", which silently
        # disabled every board image. Reinstalling pillow is what broke it, so
        # this uses the stock install and the narrowest API that works.
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        _RENDER_FAILURE = f"{type(exc).__name__}: {exc}"
        return None

    rows, cols = len(grid), len(grid[0]) if grid else 0
    if not rows or not cols:
        _RENDER_FAILURE = f"empty grid ({rows}x{cols})"
        return None

    # One pixel per cell, then nearest-neighbour upscale: no drawing primitives.
    image = Image.new("RGB", (cols, rows))
    image.putdata(
        [ARC_COLOURS.get(value, (255, 0, 255)) for row in grid for value in row]
    )
    image = image.resize((cols * cell_px, rows * cell_px), Image.NEAREST)

    # Gridlines every `gridline_every` cells give the model something to count
    # against, since MOUSE needs absolute coordinates. Written pixel by pixel
    # for the same reason as above.
    pixels = image.load()
    if pixels is not None:
        for c in range(0, cols, gridline_every):
            for y in range(rows * cell_px):
                pixels[c * cell_px, y] = (70, 70, 70)
        for r in range(0, rows, gridline_every):
            for x in range(cols * cell_px):
                pixels[x, r * cell_px] = (70, 70, 70)

    buffer = io.BytesIO()
    try:
        image.save(buffer, format="PNG")
    except Exception as exc:  # noqa: BLE001
        globals()["_RENDER_FAILURE"] = f"save failed: {type(exc).__name__}: {exc}"
        return None
    return buffer.getvalue()


def image_data_uri(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode()}"


def segment(grid: Grid, background: int = 0) -> dict[str, Any]:
    """Parse the board into 4-connected same-colour objects.

    Duck's harness hides the raw numeric grid from the model and offers this as
    the primary view, on the reasoning that games are about objects — moving
    one, aligning two, putting one inside another — and a wall of digits buries
    that. The same view is built here.

    Each node carries a `hash` of its shape and colour that ignores position, so
    the same object is recognisable after it moves or when it appears twice.
    `children` lists objects fully enclosed by another (a key relation in these
    games), and `adjacency` pairs objects that touch.
    """
    rows, cols = len(grid), len(grid[0]) if grid else 0
    node_at: dict[tuple[int, int], int] = {}
    nodes: list[dict[str, Any]] = []

    for r in range(rows):
        for c in range(cols):
            if (r, c) in node_at or grid[r][c] == background:
                continue
            colour = grid[r][c]
            stack, cells = [(r, c)], []
            node_at[(r, c)] = len(nodes)
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if (
                        0 <= nr < rows
                        and 0 <= nc < len(grid[nr])
                        and (nr, nc) not in node_at
                        and grid[nr][nc] == colour
                    ):
                        node_at[(nr, nc)] = len(nodes)
                        stack.append((nr, nc))
            min_r = min(cr for cr, _ in cells)
            min_c = min(cc for _, cc in cells)
            shape = frozenset((cr - min_r, cc - min_c) for cr, cc in cells)
            nodes.append(
                {
                    "id": len(nodes),
                    "colour": ARC_SYMBOLS.get(colour, str(colour)),
                    "pixels": len(cells),
                    "bbox": [
                        min_r,
                        min_c,
                        max(cr for cr, _ in cells),
                        max(cc for _, cc in cells),
                    ],
                    "hash": f"{ARC_SYMBOLS.get(colour, colour)}{abs(hash(shape)) % 100000:05d}",
                    "cells": sorted(cells),
                    "children": [],
                }
            )

    adjacency = sorted(
        {
            (
                min(node_at[(r, c)], node_at[(nr, nc)]),
                max(node_at[(r, c)], node_at[(nr, nc)]),
            )
            for (r, c), _ in node_at.items()
            for nr, nc in ((r + 1, c), (r, c + 1))
            if (nr, nc) in node_at and node_at[(nr, nc)] != node_at[(r, c)]
        }
    )

    # Containment: every cell of the inner object's bounding box lies strictly
    # inside the outer object's box, and the outer object encloses it.
    for outer in nodes:
        o_r0, o_c0, o_r1, o_c1 = outer["bbox"]
        for inner in nodes:
            if inner["id"] == outer["id"]:
                continue
            i_r0, i_c0, i_r1, i_c1 = inner["bbox"]
            if o_r0 < i_r0 and o_c0 < i_c0 and i_r1 < o_r1 and i_c1 < o_c1:
                outer["children"].append(inner["id"])

    return {"nodes": nodes, "adjacency": [list(pair) for pair in adjacency]}


def describe_segmentation(seg: dict[str, Any], limit: int = 24) -> str:
    """One line per object, largest first, for the prompt."""
    nodes = sorted(seg["nodes"], key=lambda n: -n["pixels"])
    lines = [
        f"  #{n['id']} {n['colour']} {n['pixels']}px box=rows {n['bbox'][0]}..{n['bbox'][2]} "
        f"cols {n['bbox'][1]}..{n['bbox'][3]} hash={n['hash']}"
        + (f" contains={n['children']}" if n["children"] else "")
        for n in nodes[:limit]
    ]
    if len(nodes) > limit:
        lines.append(f"  ... and {len(nodes) - limit} smaller objects")
    if seg["adjacency"]:
        lines.append(f"  touching pairs: {seg['adjacency'][:20]}")
    return "\n".join(lines) or "  (no objects; the board is empty)"


SANDBOX_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    )
}


def run_python(code: str, namespace: dict[str, Any], max_output: int = 1500) -> str:
    """Run model-written code against the board and return whatever it printed.

    Duck lets the model write Python to interrogate the board instead of reading
    a wall of digits, which is the largest remaining difference between their
    harness and this one. Scope here is deliberately small: no imports, no file
    or network access, and only the builtins needed to inspect a grid — the code
    is generated by a model and runs inside the submission kernel.

    Errors come back as text on purpose. A traceback is information the model
    can act on, and a failed inspection must not cost the game its turn.
    """
    buffer: list[str] = []
    scope = dict(namespace)
    scope["print"] = lambda *args, **kwargs: buffer.append(
        (kwargs.get("sep", " ")).join(str(a) for a in args)
    )
    scope["__builtins__"] = {**SANDBOX_BUILTINS, "print": scope["print"]}
    try:
        exec(compile(code, "<agent>", "exec"), scope)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        buffer.append(f"{type(exc).__name__}: {exc}")
    output = "\n".join(buffer).strip() or "(no output; use print() to see values)"
    if len(output) > max_output:
        output = output[:max_output] + f"\n... truncated at {max_output} chars"
    return output


class ActionModel:
    """What each action family has been observed to do on this level.

    The agent had no idea what its buttons did. It chose one action per turn
    from a description of the board, and a trace showed the predictable result:
    41 clicks in a 4x4 patch while the movement keys sat untried.

    Two cheap, deterministic policies come out of knowing the effects:

    probe   - try each available family once, at ~6 actions out of a ~400-action
              budget, so the choice is informed rather than guessed.
    exploit - when something works, keep doing it. A published survey of this
              competition's public set found 8 of 25 games are won by a single
              action repeated 50-200 times, and our agent never repeated
              anything.
    """

    # An action is called costly once most of its uses move the meter.
    # Budget awareness lived here and has been removed with the chrome
    # detection it was built on. `spent` was only ever incremented when a HUD
    # cell moved, and no HUD cell was ever detected, so `costly` was always
    # False: every action was labelled "(free)", free_first was the identity,
    # and the "SPENDS the limited move budget" note was never sent.
    #
    # The budgets themselves are real and decisive — sk48 decrements a life
    # counter in its arrow handler and never in its click handler, and cn04
    # caps its first level at 75 actions — so this needs rebuilding on a signal
    # that works. Reading the meter from pixels is not one: no cell in any of
    # the 25 games changes on more than 48% of transitions, and no colour
    # population is monotonic across a respawn.

    def __init__(self) -> None:
        self.tried: dict[str, int] = {}
        self.worked: dict[str, int] = {}
        self.last_effective: str | None = None

    def reset(self) -> None:
        self.__init__()

    def record(self, family: str, changed_gameplay: bool) -> None:
        self.tried[family] = self.tried.get(family, 0) + 1
        if changed_gameplay:
            self.worked[family] = self.worked.get(family, 0) + 1
            self.last_effective = family
        elif self.last_effective == family:
            self.last_effective = None  # it stopped working; stop exploiting it

    def unprobed(self, valid_names: list[str]) -> list[str]:
        return [name for name in valid_names if name not in self.tried]

    def summary(self, valid_names: list[str]) -> str:
        """A line per action: how often it did something. Empty until probed."""
        lines = [
            f"  {name}: worked {self.worked.get(name, 0)}/{self.tried[name]}"
            for name in valid_names
            if name in self.tried
        ]
        return "\n".join(lines)


class ClickSearch:
    """Systematic clicking, one object at a time, instead of guessing pixels.

    Some games only offer MOUSE, and the whole decision is *where*. Choosing
    from 64x64 leaves 4096 options and the model picked badly: a trace showed
    41 clicks inside one 4x4 patch. The board segments into far fewer objects —
    44 on sk48, 7 on m0r0 — so the objects are the real candidate set.

    A click that changes nothing leaves the state untouched, so the next
    candidate can be tried immediately; replaying from a RESET is only needed
    once the candidates run out. Levels are deterministic and RESET restores the
    initial state exactly (both verified against the engine), so that outer loop
    is sound when it is needed.
    """

    def __init__(self) -> None:
        self.tried: set[tuple[int, int]] = set()
        self.exhausted = False

    def reset(self) -> None:
        self.__init__()

    # Objects covering more than this share of the board are scenery — the
    # backdrop, a wall, a play area — not something to click.
    BACKGROUND_SHARE = 0.10

    @staticmethod
    def targets(
        segmentation: dict[str, Any], board_cells: int = GRID_SIZE * GRID_SIZE
    ) -> list[tuple[int, int]]:
        """One representative cell per object, most clickable first.

        Largest-first was wrong, and measurably so: clicking the centre of each
        of sk48's six biggest objects changed nothing, while the model's own
        click at (38,23) — a small object — moved 12 cells. The big regions are
        background; the interactive pieces are small. So scenery is dropped and
        the rest are ordered largest-first among what remains, which puts real
        pieces ahead of single-pixel noise (bp35 segments into 190 objects).
        """
        candidates = [
            node
            for node in segmentation["nodes"]
            if node["cells"]
            and node["pixels"] <= board_cells * ClickSearch.BACKGROUND_SHARE
        ]
        return [
            node["cells"][len(node["cells"]) // 2]
            for node in sorted(candidates, key=lambda n: -n["pixels"])
        ]

    def next_target(self, segmentation: dict[str, Any]) -> tuple[int, int] | None:
        """The largest object not yet clicked on this board, or None."""
        for cell in self.targets(segmentation):
            if cell not in self.tried:
                return cell
        self.exhausted = True
        return None

    def record(self, cell: tuple[int, int], changed_gameplay: bool) -> None:
        self.tried.add(cell)
        if changed_gameplay:
            # The board moved, so every object may now behave differently.
            self.tried = {cell}
            self.exhausted = False


class ObjectTracker:
    """Follows objects between frames so a turn can be described as movement.

    Reading sk48's source showed what the agent was missing: the game is a
    pushing puzzle. The player shoves coloured tiles until their arrangement
    matches a fixed reference pattern, and the win compares what sits *under*
    each piece. Told "96 cells changed", a model cannot see any of that. Told
    "you moved up 6 and the blue piece was pushed up 6", it can.

    Objects are matched between frames by their position-invariant shape hash,
    nearest first, which is what makes "the same piece, moved" expressible at
    all. Anything unmatched appeared or vanished.
    """

    def __init__(self) -> None:
        self.previous: list[dict[str, Any]] = []
        self.move_counts: dict[str, int] = {}
        # The same translations the prose describes, kept as numbers so a
        # forward model can learn from them; prose cannot be searched over.
        self.last_moves: list[tuple[int, int]] = []
        # Whether any object was followed across the frame at all. An empty
        # last_moves is ambiguous on its own: the action may have done nothing,
        # or a piece may have changed colour or shape, grown past the scenery
        # cut, or swapped with an identical twin — all of which read as no
        # movement. Learning "this action does nothing" from those would prune
        # a working action out of every candidate sequence.
        self.matched = False

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def centroid(node: dict[str, Any]) -> tuple[float, float]:
        cells = node["cells"]
        return (
            sum(r for r, _ in cells) / len(cells),
            sum(c for _, c in cells) / len(cells),
        )

    # Scenery reshapes constantly as pieces pass over it, so its hash changes
    # and it reads as vanished-then-appeared every turn. Track only pieces,
    # sized against the actual board rather than an assumed 64x64.
    SCENERY_SHARE = 0.10

    def update(
        self, segmentation: dict[str, Any], board_cells: int = GRID_SIZE * GRID_SIZE
    ) -> list[str]:
        """Match this frame against the last and describe what changed."""
        limit = board_cells * self.SCENERY_SHARE
        current = [n for n in segmentation["nodes"] if n["pixels"] <= limit]
        unmatched = list(current)
        events: list[str] = []
        self.last_moves = []
        self.matched = False

        for old in self.previous:
            same_shape = [n for n in unmatched if n["hash"] == old["hash"]]
            if not same_shape:
                events.append(f"{old['colour']} object ({old['pixels']}px) vanished")
                continue
            old_at = self.centroid(old)
            new = min(
                same_shape,
                key=lambda n: (
                    abs(self.centroid(n)[0] - old_at[0])
                    + abs(self.centroid(n)[1] - old_at[1])
                ),
            )
            unmatched.remove(new)
            self.matched = True
            dr = round(self.centroid(new)[0] - old_at[0])
            dc = round(self.centroid(new)[1] - old_at[1])
            if dr or dc:
                self.last_moves.append((dr, dc))
                where = (
                    f"{'down' if dr > 0 else 'up' if dr else ''}"
                    f"{'' if not (dr and dc) else ' and '}"
                    f"{'right' if dc > 0 else 'left' if dc else ''}"
                )
                events.append(
                    f"{new['colour']} object ({new['pixels']}px) moved {where} "
                    f"by ({abs(dr)},{abs(dc)})"
                )
                self.move_counts[new["hash"]] = self.move_counts.get(new["hash"], 0) + 1

        for new in unmatched:
            events.append(f"{new['colour']} object ({new['pixels']}px) appeared")

        self.previous = current
        return events

    def movers(self) -> list[str]:
        """Shape hashes seen to move, commonest first — the pieces in play."""
        return [h for h, _ in sorted(self.move_counts.items(), key=lambda kv: -kv[1])]


class ForwardModel:
    """What each action does to the board, learned by watching it happen.

    Searching over candidate action sequences needs a simulator, and there is
    no way to get one from the environment where it counts. The scored run sets
    OPERATION_MODE=online against a gateway sidecar with no local environments,
    so there is no game object to fork the way the local sk48 solver forked one
    with deepcopy. Trying a sequence for real and taking it back does not work
    either: sk48's undo restores every sprite position and never touches the
    move budget, so a probe is spent whether or not it is undone, and cn04
    allows 75 actions on its first level.

    So the only sequence we can search is one we can predict ourselves. This
    learns the translation each action applies, and — because a model that
    cannot predict makes search worthless — scores every prediction against
    what actually happened before learning from it.
    """

    # Two sightings before predicting: one is indistinguishable from a fluke,
    # and a confident wrong model is worse for search than an absent one.
    MIN_OBSERVATIONS = 2
    # Sightings of one action before its history is halved, so the model can
    # notice that an action stopped working within a few turns instead of tens.
    WINDOW = 8

    def __init__(self) -> None:
        self.effects: dict[str, dict[tuple[int, int], int]] = {}
        self.predicted = 0
        self.correct = 0
        # Shape hashes seen to move. A rollout has to know which pieces an
        # action carries and which are scenery, and there is no way to tell
        # from one frame — only from having watched them.
        self.moving: set[str] = set()

    # Deliberately no reset(): every sibling is cleared on level-up, and this
    # one must not be. sk48's eight levels carry identical physics flags and
    # differ only in piece count, so what an action does is the one thing that
    # does carry over. See the level-change block in _choose_action_inner.

    @staticmethod
    def summarise(moves: list[tuple[int, int]]) -> tuple[int, int] | None:
        """The translation shared by most of the pieces that moved.

        A push moves the pusher and the pushed by the same vector, so the modal
        translation is the action's effect, and a stray piece drifting on its
        own does not outvote a push. A tie is not evidence, though: breaking it
        by picking a vector meant the largest tuple won, so a player moving up
        6 lost to an enemy drifting down 1 — and since negative deltas always
        lose that comparison, UP and LEFT were the mislearned directions.
        """
        if not moves:
            return (0, 0)
        counts: dict[tuple[int, int], int] = {}
        for move in moves:
            counts[move] = counts.get(move, 0) + 1
        best = max(counts.values())
        winners = [move for move, count in counts.items() if count == best]
        return winners[0] if len(winners) == 1 else None

    def predict(self, action: str) -> tuple[int, int] | None:
        """The translation this action is expected to apply, or None if unsure.

        Weighted towards recent sightings, because the effect of an action is
        not constant: a piece that has been moving up 6 all game moves 0 the
        turn it reaches a wall. On lifetime counts a long history of successes
        outvotes the wall for many turns, so the model stays confident exactly
        where it has started being wrong.
        """
        seen = self.effects.get(action)
        if not seen:
            return None
        total = sum(seen.values())
        best = max(seen.values())
        winners = [vector for vector, count in seen.items() if count == best]
        if total < self.MIN_OBSERVATIONS or len(winners) > 1 or best * 2 <= total:
            return None  # no majority: the action is not deterministic here
        return winners[0]

    def observe(self, action: str, moves: list[tuple[int, int]]) -> None:
        """Score the standing prediction, then learn from what really happened.

        Scoring first is the point: learning from an outcome and then claiming
        to have predicted it measures nothing.
        """
        if (actual := self.summarise(moves)) is None:
            return  # a tie tells us nothing; recording either vector is a guess
        if (guess := self.predict(action)) is not None:
            self.predicted += 1
            self.correct += guess == actual
        seen = self.effects.setdefault(action, {})
        if sum(seen.values()) >= self.WINDOW:
            for vector in list(seen):
                seen[vector] //= 2  # halve the past so recent evidence wins
                if not seen[vector]:
                    del seen[vector]
        seen[actual] = seen.get(actual, 0) + 1

    @property
    def accuracy(self) -> float:
        """Share of predictions that matched. 0.0 when nothing was predicted."""
        return self.correct / self.predicted if self.predicted else 0.0

    def step(
        self,
        cells: dict[str, frozenset[tuple[int, int]]],
        action: str,
        rows: int,
        cols: int,
    ) -> dict[str, frozenset[tuple[int, int]]] | None:
        """Where the pieces sit after one action, or None if unpredictable.

        Summing translations cannot express a wall: told UP moves a piece -6,
        a five-step rollout adds -6 five times and walks it off the board. This
        moves the pieces that move and stops the ones that cannot, which is the
        least a sequence has to model to be worth searching.

        Everything that moved before is moved together. A push carries pusher
        and pushed by the same vector, so treating movers as one group gets the
        common case right without needing to know which piece is the player.
        """
        vector = self.predict(action)
        if vector is None:
            return None
        dr, dc = vector
        if (dr, dc) == (0, 0):
            return dict(cells)

        # Keys may carry a position to keep two identical shapes apart
        # ("<hash>@<cell>"); what was learned to move is the shape itself.
        movers = {key for key in cells if key.split("@")[0] in self.moving}
        if not movers:
            return None  # nothing is known to move; the outcome is a guess
        settled = {key: patch for key, patch in cells.items() if key not in movers}
        blockers = frozenset().union(*settled.values()) if settled else frozenset()

        moved: dict[str, frozenset[tuple[int, int]]] = {}
        for key in movers:
            shifted = frozenset((r + dr, c + dc) for r, c in cells[key])
            offboard = any(not (0 <= r < rows and 0 <= c < cols) for r, c in shifted)
            moved[key] = cells[key] if offboard or (shifted & blockers) else shifted
        return {**settled, **moved}


class StateGraph:
    """States seen, what has been tried in each, and how to get back to one.

    This is the only mechanism in either agent measured to beat random play.
    The explorer clears ls20 and lp85 and nothing else, and instrumenting it
    shows both clears came from this frontier rather than from its avatar
    induction: lp85 cleared at step 19 with the avatar still unpinned and an
    empty move lattice, so no induction had happened at all. Random play with
    five seeds clears neither game, so the coverage is doing real work.

    Nothing here predicts or models anything. It records what has been tried
    where, and walks back to somewhere with something left to try. The Kaggle
    agent had no equivalent: it re-decided from scratch every turn, which is
    why 89% action-effect prediction converted into no levels — knowing what an
    action does is worth little without a record of what has been tried.
    """

    def __init__(self) -> None:
        self._untested: dict[Any, list[str]] = {}
        self._edges: dict[Any, list[tuple[str, Any]]] = {}

    def reset(self) -> None:
        self.__init__()

    def register(self, sig: Any, options: list[str]) -> None:
        """Record a state's options on first sighting only.

        Re-registering would resurrect options already taken, turning the
        frontier into a loop over the same few actions.
        """
        if sig not in self._untested:
            self._untested[sig] = list(options)

    def take(self, sig: Any, option: str) -> None:
        rest = self._untested.get(sig)
        if rest and option in rest:
            rest.remove(option)

    def connect(self, src: Any, option: str, dst: Any) -> None:
        edges = self._edges.setdefault(src, [])
        if (option, dst) not in edges:
            edges.append((option, dst))

    def untested_at(self, sig: Any) -> list[str]:
        return self._untested.get(sig, [])

    def path_to_frontier(self, start: Any, limit: int = 12) -> list[str]:
        """Options leading from `start` to the nearest state with something
        untested. Empty when `start` is itself a frontier or none is reachable.

        Bounded: a long walk back spends real budget, and cn04 allows 75
        actions on its first level.
        """
        if self.untested_at(start):
            return []
        prev: dict[Any, tuple[Any, str]] = {}
        seen, queue = {start}, deque([start])
        while queue:
            node = queue.popleft()
            for option, dst in self._edges.get(node, []):
                if dst in seen:
                    continue
                seen.add(dst)
                prev[dst] = (node, option)
                if self.untested_at(dst):
                    path = []
                    at = dst
                    while at != start:
                        at, option = prev[at]
                        path.append(option)
                    return list(reversed(path)) if len(path) <= limit else []
                queue.append(dst)
        return []


class UndoDetector:
    """Finds an action that reverts the board, and reports when it is worth using.

    sk48 has one: ACTION7 restores a snapshot, and unlike the arrow keys it does
    not spend the move budget. Which action does this is game-specific, so it is
    inferred rather than named — an action is an undo when taking it returns the
    board to the state before the previous action, and it costs nothing.

    The value is not a refund; the budget is spent either way. It is that
    reversing a move costs zero instead of another point, which makes trying a
    costly action and stepping back affordable. Under a score that squares
    actions taken, cheap reversal is worth more than it first appears.
    """

    def __init__(self) -> None:
        self.history: list[Grid] = []
        self.candidate: str | None = None
        self._ruled_out: set[str] = set()

    def reset(self) -> None:
        self.__init__()

    def observe(self, action: str, board: Grid) -> None:
        """Record a transition and decide whether `action` reverted the board."""
        family = action.split("@")[0]
        if (
            self.candidate is None
            and family not in self._ruled_out
            and len(self.history) >= 2
            and board == self.history[-2]
            and board != self.history[-1]
        ):
            self.candidate = family
        self.history.append(board)
        del self.history[:-3]

    def rule_out(self, family: str) -> None:
        """Stop trusting an action that failed to revert anything."""
        self._ruled_out.add(family)
        if self.candidate == family:
            self.candidate = None


class LevelMemory:
    """What playing the current level has revealed, summarised for the prompt.

    The Duck harness keeps (action, before, after) transitions and lets the
    model query them by writing Python. This agent picks one action per turn
    and cannot run code, so the same transitions are kept here and reduced to a
    few lines of text instead.

    The idea worth stealing is HUD discrimination. Duck's prompt warns the model
    not to treat a moving timer or step counter as evidence that a move worked.
    That is detectable rather than a matter of judgement: cells that change
    under nearly every action are chrome, whatever the action was. Excluding
    them turns "the board changed" into "the game changed", which is what
    decides whether an action was wasted — and wasted actions are squared in the
    score.
    """

    # Chrome detection lived here and has been removed. It required a cell to
    # change on 80% of transitions; measured across all 25 games, no cell
    # changes on more than 48% (cn04 0.33, sk48 0.28, ls20 0.12, dc22 0.15), so
    # it never fired once and the prompt line it produced was never sent. It
    # cannot be repaired by lowering the threshold either: there is no gap
    # between chrome and gameplay to put one in, and any value low enough to
    # fire marks the busiest gameplay cells as chrome. The alternative
    # discriminator — cells that change under every action — is no better,
    # flagging 651 cells on ls20 and none on sk48 or dc22. These boards carry
    # incidental churn everywhere (741 frame signatures for 30 avatar cells),
    # so what flickers cannot identify chrome.
    MAX_LISTED_CELLS = 12

    def __init__(self) -> None:
        self.transitions = 0
        self._last: tuple[str, list[tuple[int, int, int, int]]] | None = None
        self._dead: set[str] = set()
        # Every action family tried since the board last changed, so the agent
        # can be told what it has NOT tried. MOUSE@(12,39) and MOUSE@(13,38) are
        # different strings but the same idea, so families are the useful unit:
        # tracked per name, a click never repeats and nothing is ever "dead".
        self._tried_families: set[str] = set()
        self._no_effect_streak = 0
        # The model's own one-line theory of the mechanic, overwritten each time
        # it offers a new one. Kept as a single line rather than a history so the
        # prompt stays constant-size whatever the level costs.
        self.mechanic = ""

    def reset(self) -> None:
        """A new level is a new board, but not a new game.

        Everything here describes this board — which cells are chrome, which
        actions are dead on it, what changed last — and none of that survives a
        new layout. The mechanic does: sk48's eight levels carry byte-identical
        flags and differ only in how many pieces they hold, and cn04 keeps its
        rules and merely raises the step budget. Clearing the theory discarded
        it at the one moment the level just proved it right.
        """
        mechanic = self.mechanic
        self.__init__()
        self.mechanic = mechanic

    def record_mechanic(self, note: object) -> bool:
        """Store the model's latest theory, replacing any earlier one.

        Anything that is not a non-empty string is dropped rather than coerced:
        a model that sends `note: null` would otherwise overwrite a working
        theory with the string "None".
        """
        if not isinstance(note, str):
            return False
        cleaned = " ".join(note.split())[:MAX_NOTE_CHARS]
        if not cleaned or cleaned == self.mechanic:
            return False
        self.mechanic = cleaned
        return True

    @property
    def dead_actions(self) -> set[str]:
        return set(self._dead)

    @property
    def no_effect_streak(self) -> int:
        return self._no_effect_streak

    def untried(self, valid_names: list[str]) -> list[str]:
        """Action families not yet tried since the board last changed."""
        return [name for name in valid_names if name not in self._tried_families]

    def record(self, action: str, before: Grid | None, after: Grid) -> None:
        if before is None:
            return
        self.transitions += 1
        changed = [
            (r, c, before[r][c], value)
            for r, row in enumerate(after)
            for c, value in enumerate(row)
            if r < len(before) and c < len(before[r]) and before[r][c] != value
        ]
        # Every change now counts as gameplay: the chrome filter that used to
        # sit here never classified a single cell in any of the 25 games.
        gameplay = changed
        self._last = (action, gameplay)
        family = action.split("@")[0]
        self._tried_families.add(family)
        if gameplay:
            self._dead.clear()
            self._tried_families = {family}
            self._no_effect_streak = 0
        else:
            self._dead.add(action)
            self._no_effect_streak += 1

    def describe_last(self) -> str:
        if self._last is None:
            return "n/a (first look at this board)"
        action, gameplay = self._last
        if not gameplay:
            return (
                "NOTHING changed in the game (only a timer or counter, if anything) - "
                "that action achieved nothing here; try a different one"
            )
        cells = ", ".join(
            f"({r},{c}) {ARC_SYMBOLS.get(before, before)}->{ARC_SYMBOLS.get(after, after)}"
            for r, c, before, after in gameplay[: self.MAX_LISTED_CELLS]
        )
        if len(gameplay) > self.MAX_LISTED_CELLS:
            rows = [r for r, _, _, _ in gameplay]
            cols = [c for _, c, _, _ in gameplay]
            return (
                f"{len(gameplay)} cells changed, spanning rows {min(rows)}..{max(rows)}, "
                f"cols {min(cols)}..{max(cols)}; first {self.MAX_LISTED_CELLS}: {cells}"
            )
        return f"{len(gameplay)} cell(s) changed: {cells}"

    def prompt_notes(self, valid_names: list[str] | None = None) -> str:
        """Extra lines for the prompt: dead actions, untried ones, and chrome."""
        notes = []
        if dead := sorted(self._dead):
            notes.append(
                f"Actions already shown to change NOTHING on this board: {', '.join(dead)}"
            )
        if valid_names and (untried := self.untried(valid_names)):
            notes.append(
                f"You have NOT yet tried: {', '.join(untried)}. Prefer one of these "
                f"over repeating something that changed nothing."
            )
        if self._no_effect_streak >= 3:
            notes.append(
                f"WARNING: {self._no_effect_streak} actions in a row changed nothing. "
                f"Whatever you are doing is not the mechanic here - switch to a "
                f"different kind of action, or a very different part of the board."
            )
        return ("\n" + "\n".join(notes)) if notes else ""


@dataclass
class ArcAction:
    id: int
    x: int | None = None
    y: int | None = None


def load_llama() -> Any:
    """Construct one Llama instance, or None if it can't be loaded.

    Returns None rather than raising (e.g. CUDA init fails on the rerun host);
    callers must degrade gracefully to random actions.
    """
    try:
        from llama_cpp import Llama

        model_path = os.environ.get("LLAMA_MODEL_PATH", "/tmp/qwen3-14b.Q4_K_M.gguf")
        # 8192, not 4096: a board spanning the 64x64 grid is ~1.2-2.4k tokens,
        # and 4096 left no headroom (640 MiB KV at 4096 -> 1.25 GiB at 8192).
        n_ctx = int(os.environ.get("LLAMA_N_CTX", "8192"))
        n_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", "99"))
        print(
            f"[MyAgent] Loading model from {model_path} "
            f"(n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})"
        )
        llm = Llama(
            model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )
        print("[MyAgent] Model loaded")
        return llm
    except Exception as exc:  # noqa: BLE001
        print(
            f"[MyAgent] WARNING: model load failed ({exc!r}); "
            "degrading to random actions"
        )
        return None


class HTTPChatBackend:
    """OpenAI-compatible chat client, shaped like `llama_cpp.Llama`.

    Points at a local vLLM server (the Duck harness's stack). vLLM batches
    concurrent requests itself, so unlike a llama.cpp instance this is safe to
    call from every game thread at once — which is the whole point: ~110 games
    run as concurrent threads, and serializing them behind one model was the
    throughput ceiling.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        seed: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Sent as OpenAI's `seed` so a repeat is a genuine repeat. Without it an
        # A/B cannot tell a real effect from sampling: three "seeds" that only
        # differed by label produced byte-identical results in the arm whose
        # decisions were mostly deterministic, and no variance to compare
        # against in the arm that was not.
        self.seed = seed

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **({"seed": self.seed} if self.seed is not None else {}),
            # Qwen3.6's chat template gates reasoning on this flag alone; the
            # "/no_think" string that works for Qwen3 is ignored by it. Without
            # this the model spends its whole token budget inside <think> and
            # returns no action (measured: 64/64 tokens, empty content).
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            return json.loads(response.read())


class ModelPool:
    """Up to `max_size` Llama instances, lent out one thread at a time.

    Instances are created on demand rather than up front: the gateway requires
    a first action within ~15 minutes, and pre-loading the whole pool would put
    that deadline behind several minutes of model loading. Growing lazily also
    sizes the pool to the concurrency that actually shows up.
    """

    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._idle: list[Any] = []
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(self._max_size)
        self._created = 0
        self._failed = False

    @property
    def size(self) -> int:
        with self._lock:
            return self._created

    @contextmanager
    def acquire(self) -> Iterator[Any | None]:
        """Lend an instance, or None if the model cannot be loaded at all."""
        self._slots.acquire()
        llm = None
        try:
            llm = self._checkout()
            yield llm
        finally:
            if llm is not None:
                with self._lock:
                    self._idle.append(llm)
            self._slots.release()

    def _checkout(self) -> Any | None:
        with self._lock:
            if self._idle:
                return self._idle.pop()
            if self._failed or self._created >= self._max_size:
                return None
            self._created += 1  # claim the slot before the slow load
        llm = load_llama()
        if llm is None:
            with self._lock:
                self._created -= 1
                self._failed = True  # a second attempt would fail the same way
        return llm


MODEL_POOL = ModelPool(POOL_SIZE)

# When a vLLM server is available (ARC_LLM_BASE_URL), it replaces the local
# llama.cpp pool entirely: vLLM batches across threads, so no pool and no lock.
LLM_BASE_URL = os.environ.get("ARC_LLM_BASE_URL", "").strip()
LLM_MODEL = os.environ.get("ARC_LLM_MODEL", "").strip()
REMOTE_BACKEND = HTTPChatBackend(LLM_BASE_URL, LLM_MODEL) if LLM_BASE_URL else None

# When run inside the ARC-AGI-3-Agents framework
try:
    from agents.agent import Agent
except ImportError:
    # Local testing — provide a stub base class
    class Agent:  # type: ignore[no-redef]
        """Stub base class for local testing."""

        game_id: str = ""
        MAX_ACTIONS: int = 80
        arc_env: Any = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        @property
        def name(self) -> str:
            return f"Agent.{self.game_id}"

        def _convert_raw_frame_data(self, raw: Any) -> Any:
            return raw


class MyAgent(Agent):
    # Whether this agent reaches for the model. Read by the scoring harness:
    # the model-path features below are only meaningful when it does.
    USES_MODEL = True

    """Agent for ARC-AGI-3, choosing one action per turn via function calling.

    Inference is whatever `_model()` yields: a vLLM server, a pooled llama-cpp
    instance, or an injected stub in tests.
    """

    # The framework's own loop bound (agents/agent.py: `action_counter <=
    # MAX_ACTIONS`), default 80. We used to *lower* it to 50, hard-capping every
    # game at 51 actions — far fewer than an ARC-AGI-3 level needs. The real
    # safety valve is the global deadline in is_done(), so keep this generous.
    MAX_ACTIONS = int(os.environ.get("ARC_MAX_ACTIONS", "400"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Models come from the shared pool (see ModelPool). Tests and the local
        # scoring harness inject an instance here instead.
        self._llm: Any = None
        self._rng = random.Random()

        # Session state. Deliberately no chat history: every turn is a fresh
        # single-shot prompt carrying only the current board (see _call_llm).
        self._prev_levels = -1
        self._step = 0
        self._last_action_id: int | None = None
        self._last_action_name: str = "None"
        self._last_grid: Grid | None = None
        self.memory = LevelMemory()
        self._segmentation: dict[str, Any] = {"nodes": [], "adjacency": []}
        self._seg_board: tuple[tuple[int, ...], ...] | None = None
        self.actions = ActionModel()
        self.clicks = ClickSearch()
        self.undo = UndoDetector()
        self.objects = ObjectTracker()
        self.forward = ForwardModel()
        self.graph = StateGraph()
        self._sig: tuple[tuple[int, str, int, int], ...] | None = None
        self._frontier_plan: deque[str] = deque()
        self._undo_pending = False
        self._last_events: list[str] = []
        self._exploit_left = 0
        self._exploited: set[str] = set()
        self._policy_streak = 0
        self._last_click: tuple[int, int] | None = None
        # How each action was decided. Without this a run that silently fell
        # back to random every turn is indistinguishable from one that played
        # deliberately — both just report actions taken.
        self.stats: dict[str, int] = defaultdict(int)
        # Action payload passed to arc_env.step() by do_action_request.
        self._pending_data: dict[str, int] | None = None
        self._pending_reasoning: str | dict[str, str] | None = None

    @property
    def name(self) -> str:
        return f"{super().name}.repl"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop when we win the whole game, or when the shared run budget is spent."""
        try:
            if latest_frame.state is GameState.WIN:
                return True
            if time.monotonic() > _RUN_DEADLINE:
                print(
                    f"[MyAgent] {self.game_id}: shared run budget ({RUN_BUDGET_S:.0f}s) spent"
                )
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"[MyAgent] {self.game_id}: is_done error ({exc!r}); returning False")
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Convert FrameData to our format, call LLM, convert back.

        Never raises: any LLM/framework error falls back to a random action so
        a single bad inference can't stall the gateway loop and empty the
        scorecard (the documented silent-crash failure mode).
        """
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as exc:  # noqa: BLE001
            self.stats["choose_action_exception"] += 1
            print(
                f"[MyAgent] {self.game_id}: choose_action error ({exc!r}); "
                "using random fallback"
            )
            action = self._random_action(
                latest_frame.available_actions or [1, 2, 3, 4, 5]
            )
            # Record it, or _last_action_name still names the previous action
            # and the next frame's movement is credited to it. A mis-scored
            # effect costs one turn; a world model taught that UP does what a
            # random action did is wrong for the rest of the game.
            grid = latest_frame.frame[-1] if len(latest_frame.frame) else []
            self._record_action(grid, ArcAction(id=action.value))
            return action

    def _random_action(self, available: list[int]) -> GameAction:
        """Random non-reset action as a safe fallback."""
        return self._arc_to_game_action(self._random_arc_action(available))

    def _choose_action_inner(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Reset on new level. Ignore transient levels_completed=-1 emitted by
        # the gateway during boot (a change to -1 is not a real level up).
        if (
            latest_frame.levels_completed is not None
            and latest_frame.levels_completed >= 0
            and latest_frame.levels_completed != self._prev_levels
        ):
            self._prev_levels = latest_frame.levels_completed
            self._step = 0
            self._last_action_id = None
            self._last_action_name = "None"
            self._last_grid = None
            self.memory.reset()
            self.actions.reset()
            self.clicks.reset()
            self.undo.reset()
            self.objects.reset()
            self._undo_pending = False
            self._last_events = []
            self._exploit_left = 0
            self._exploited.clear()
            # A new level is a new arrangement: states from the old board are
            # unreachable, and a stale plan routes toward one of them.
            self.graph.reset()
            self._sig = None
            self._frontier_plan.clear()

        # Start / restart the game. The gateway returns an empty frame while
        # NOT_PLAYED and a dead board after GAME_OVER; RESET yields the real
        # initial board so the model never acts blind. If a RESET didn't take
        # effect, fall back to a real input rather than spinning.
        if self._unplayable(latest_frame):
            if self._last_action_id == 0:
                return self._random_action(
                    latest_frame.available_actions or [1, 2, 3, 4, 5]
                )
            return self._make_reset()

        grids = latest_frame.frame if latest_frame.frame else []
        grid = grids[-1] if grids else []
        if not grid:
            return self._make_reset()

        self._step += 1

        board = tuple(tuple(row) for row in grid)
        # Exactly once per turn, and before anything reads the segmentation or
        # the movement events.
        self.observe_frame(grid)
        # The board in hand is the result of the previous action, so score that
        # transition now — before the prompt is built, or the effect shown is
        # always one turn stale.
        if self._last_grid is not None:
            self.memory.record(
                self._last_action_name,
                self._last_grid,
                tuple(tuple(row) for row in grid),
            )
            family = self._last_action_name.split("@")[0]
            worked = self.memory.no_effect_streak == 0
            self.actions.record(family, worked)
            board_now = tuple(tuple(row) for row in grid)
            if (
                self._undo_pending
                and self.undo.candidate == family
                and worked is False
                and board_now == self._last_grid
            ):
                # It claimed to be an undo and changed nothing; stop trusting it.
                self.undo.rule_out(family)
            self._undo_pending = False
            self.undo.observe(self._last_action_name, board_now)
            if family == "MOUSE" and self._last_click is not None:
                self.clicks.record((self._last_click[1], self._last_click[0]), worked)
            # Arm once per newly-discovered action, keyed on the full identity so
            # a click's cell counts: every click shares the family "MOUSE", so
            # keying on family alone would exploit only the first cell that ever
            # worked and ignore every productive cell found after it.
            # Re-arming on each success never drained the counter on a movement
            # game — every arrow changes the board — so the agent held one
            # direction until the move budget ran out: 53% of v59's actions.
            if worked and self._last_action_name not in self._exploited:
                self._exploited.add(self._last_action_name)
                self._exploit_left = EXPLOIT_REPEATS
        effect = self.memory.describe_last()

        # Map GameState
        state_map = {
            GameState.NOT_PLAYED: "NOT_FINISHED",
            GameState.NOT_FINISHED: "NOT_FINISHED",
            GameState.WIN: "WIN",
            GameState.GAME_OVER: "GAME_OVER",
        }
        state = state_map.get(latest_frame.state, "NOT_FINISHED")

        # Build prompt
        valid_names = [
            ACTION_NAMES.get(a, f"ACTION{a}")
            for a in (latest_frame.available_actions or [1, 2, 3, 4, 5, 6])
        ]

        # The raw-text prompt is only needed if the tool call misses, so build it
        # lazily rather than assembling a second ~5KB string on every turn.
        def raw_prompt() -> str:
            return self.prompt_for(grid, valid_names, state, effect)

        tool_prompt = self.prompt_for(grid, valid_names, state, effect, tool_mode=True)

        available = latest_frame.available_actions or [1, 2, 3, 4, 5, 6]

        # Cheap policies first. Probing and exploiting need no inference, and
        # both address what a trace showed the model doing badly: it never
        # learned what its buttons did, and never repeated anything that worked.
        action = self._policy_action(valid_names, available)
        if action is None:
            self._policy_streak = 0
            action = self._call_llm(raw_prompt, tool_prompt, available, board)

        action = self._escape_if_stuck(action, valid_names, available)

        # Convert to GameAction
        game_action = self._arc_to_game_action(action)
        self._record_action(grid, action)
        return game_action

    def _policy_action(
        self, valid_names: list[str], available: list[int]
    ) -> Any | None:
        """Probe unknown actions, then exploit what works. None means ask the model.

        Both branches are free — no inference — and the score squares wasted
        actions, so an informed cheap choice beats an uninformed expensive one.
        """
        if self._policy_streak >= MAX_POLICY_STREAK:
            self._policy_streak = 0
            self.stats["policy_yield"] += 1
            return None

        playable = [n for n in valid_names if NAME_TO_ID.get(n) not in (None, 0)]

        # Where to click is the whole decision when clicking is the mechanic.
        # Rather than guess a pixel, walk the objects: the board segments into
        # tens of them, not 4096 cells, and an unchanged board means the next
        # candidate can be tried straight away.
        clickable = COMPLEX_ACTION_ID in available
        if clickable and self.memory.no_effect_streak >= ESCAPE_AFTER:
            if (target := self.clicks.next_target(self._segmentation)) is not None:
                self.stats["click_search"] += 1
                self._policy_streak += 1
                return ArcAction(id=COMPLEX_ACTION_ID, x=target[1], y=target[0])

        # Click-only games used to hand straight back to the model here, which
        # is why the frontier never fired once on lp85, tn36 or the other four
        # MOUSE-only games. The reason for the hand-back was that the policy
        # clicked at *random* positions and learned nothing; the frontier picks
        # ranked object targets and records what each did, so it earns a turn.
        # MAX_POLICY_STREAK still guarantees the model one turn in ten.
        click_only = all(NAME_TO_ID.get(n) == COMPLEX_ACTION_ID for n in playable)
        if click_only and not (FRONTIER and self._sig is not None):
            return None

        # Probe the simple actions only. One click somewhere random says nothing
        # about whether clicking works, so MOUSE is left to the model.
        simple = [n for n in playable if NAME_TO_ID.get(n) != COMPLEX_ACTION_ID]
        if PROBE_ACTIONS and (unprobed := self.actions.unprobed(simple)):
            self.stats["probe"] += 1
            self._policy_streak += 1
            return self._named_action(unprobed[0])

        # Frontier: try what has not been tried in *this* state, and otherwise
        # walk back to a state that still has something untried. This is the
        # only mechanism measured to beat random on any game — it is what
        # cleared ls20 and lp85 for the explorer, while random play with five
        # seeds cleared neither. It sits after probing so a level still starts
        # by learning what the actions do.
        if FRONTIER and self._sig is not None:
            options = self._frontier_options(simple, bool(clickable))
            if (frontier := self._frontier_action(options)) is not None:
                return frontier

        # An action that achieved nothing leaves the board somewhere we did not
        # want, and the score squares the total, so reverting with the free undo
        # beats walking back. This used to require the action to be costly,
        # which was never true once chrome detection turned out never to fire.
        if self.undo.candidate in playable and self.memory.no_effect_streak == 1:
            self.stats["undo"] += 1
            self._policy_streak += 1
            self._undo_pending = True
            return self._named_action(self.undo.candidate)

        # Exploring must not cost the game. Arrow keys in sk48 decrement a life
        # counter and lose at zero, while clicks are free — so once the meter
        # has revealed which is which, wander with the free actions.
        idle = list(playable)
        if self.memory.no_effect_streak >= ESCAPE_AFTER and idle:
            choice = idle[0]
            self.stats["explore_free"] += 1
            self._policy_streak += 1
            return self._named_action(choice)

        if self._exploit_left > 0 and self.actions.last_effective in playable:
            self._exploit_left -= 1
            self.stats["exploit"] += 1
            self._policy_streak += 1
            return self._repeat_last(self.actions.last_effective)

        return None

    def _repeat_last(self, family: str) -> Any:
        """Repeat the last action exactly — same target, for a click.

        Re-rolling the coordinates would not be repetition at all; the point of
        exploiting is that this precise action produced a change.
        """
        action_id = NAME_TO_ID[family]
        if action_id == COMPLEX_ACTION_ID and self._last_click is not None:
            x, y = self._last_click
            return ArcAction(id=action_id, x=x, y=y)
        return self._named_action(family)

    # Click targets offered to the frontier per state. Unbounded is not an
    # option: bp35 segments into 190 objects, so a state would take 190 actions
    # to exhaust and cn04 allows 75 on its first level. Ranked by ClickSearch,
    # which puts real pieces ahead of single-pixel noise.
    FRONTIER_CLICKS = 6

    def _frontier_options(self, simple: list[str], clickable: bool) -> list[str]:
        """What is worth trying in this state, clicks included.

        Without clicks the frontier cannot fire at all on the six MOUSE-only
        games — a fifth of the set, and the only games where this agent has
        ever cleared a level. Clicks are named exactly as _record_action names
        them so a graph edge matches the action that produced it.
        """
        options = list(simple)
        if clickable:
            options += [
                f"MOUSE@({c},{r})"
                for r, c in ClickSearch.targets(
                    self._segmentation, len(self._seg_board or ()) ** 2 or GRID_SIZE**2
                )[: self.FRONTIER_CLICKS]
            ]
        return options

    def _option_action(self, option: str) -> Any:
        """An ArcAction for a frontier option, which may carry a click target."""
        if option.startswith("MOUSE@("):
            x, y = option[len("MOUSE@(") : -1].split(",")
            return ArcAction(id=COMPLEX_ACTION_ID, x=int(x), y=int(y))
        return self._named_action(option)

    def _named_action(self, name: str) -> Any:
        """An ArcAction for a family name, with a random target for clicks."""
        action_id = NAME_TO_ID[name]
        if action_id == COMPLEX_ACTION_ID:
            return ArcAction(
                id=action_id,
                x=self._rng.randrange(GRID_SIZE),
                y=self._rng.randrange(GRID_SIZE),
            )
        return ArcAction(id=action_id)

    def _escape_if_stuck(
        self, action: Any, valid_names: list[str], available: list[int]
    ) -> Any:
        """Force variety once the model is provably going nowhere.

        Only fires after ESCAPE_AFTER consecutive no-effect actions, and only
        towards families it has not tried on this board, so a working strategy
        is never interrupted.
        """
        if self.memory.no_effect_streak < ESCAPE_AFTER:
            return action
        untried = [
            name
            for name in self.memory.untried(valid_names)
            if NAME_TO_ID.get(name) in available and NAME_TO_ID.get(name) != 0
        ]
        if not untried:
            return action
        # Cheapest first: an escape that spends the budget can end the level.
        choice = NAME_TO_ID[untried[0]]
        self.stats["escape_forced"] += 1
        if choice == COMPLEX_ACTION_ID:
            return ArcAction(
                id=choice,
                x=self._rng.randrange(GRID_SIZE),
                y=self._rng.randrange(GRID_SIZE),
            )
        return ArcAction(id=choice)

    @staticmethod
    def _unplayable(latest_frame: FrameData) -> bool:
        """Whether the board cannot be acted on: empty before the first RESET,
        dead after GAME_OVER. What to do about it is the caller's policy."""
        return latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER)

    def _make_reset(self) -> GameAction:
        """Return a RESET GameAction, recording it as the last action."""
        self._last_action_id = 0
        self._last_action_name = "RESET"
        self._last_grid = None
        self._pending_data = None
        self._pending_reasoning = "REPL agent start/restart game"
        return GameAction.RESET

    def _encode_grid(
        self, grid: list[list[int]]
    ) -> tuple[str, tuple[int, int, int, int] | None]:
        """Encode a grid compactly: crop to the non-empty bounding box and return
        (body, bbox) where bbox=(min_r, min_c, max_r, max_c) or None when the whole
        board is shown. The grid dominates prefill cost, so smaller prompts mean
        more actions per game."""
        if not grid or not grid[0]:
            return "empty", None
        rows, cols = len(grid), len(grid[0])
        min_r, max_r, min_c, max_c = rows, -1, cols, -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 0:
                    if r < min_r:
                        min_r = r
                    if r > max_r:
                        max_r = r
                    if c < min_c:
                        min_c = c
                    if c > max_c:
                        max_c = c
        if max_r < min_r:
            return "all zeros", None
        crop = [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]
        body = "\n".join(
            "".join(ARC_SYMBOLS.get(v, f"{v:x}") for v in row) for row in crop
        )
        if (min_r, min_c) == (0, 0) and max_r == rows - 1 and max_c == cols - 1:
            return body, None
        return (
            f"{body}\n[board is {rows}x{cols}; shown region = rows {min_r}..{max_r}, "
            f"cols {min_c}..{max_c}]",
            (min_r, min_c, max_r, max_c),
        )

    def _grid_changed(self, grid: list[list[int]]) -> bool:
        key = tuple(tuple(row) for row in grid)
        return self._last_grid is None or key != self._last_grid

    def _record_action(self, grid: list[list[int]], arc_action: Any) -> None:
        """Remember the action taken and the board it was chosen from.

        The outcome is scored at the start of the next turn, when the resulting
        board arrives (see _choose_action_inner).
        """
        name = ACTION_NAMES.get(arc_action.id, f"ACTION{arc_action.id}")
        if (
            arc_action.id == COMPLEX_ACTION_ID
            and getattr(arc_action, "x", None) is not None
        ):
            name = f"MOUSE@({arc_action.x},{arc_action.y})"
        if (
            arc_action.id == COMPLEX_ACTION_ID
            and getattr(arc_action, "x", None) is not None
        ):
            self._last_click = (arc_action.x, arc_action.y)
        self._last_action_id = arc_action.id
        self._last_action_name = name
        self._last_grid = tuple(tuple(row) for row in grid)

    def _frontier_action(self, options: list[str]) -> Any | None:
        """An untried option here, or the next step of a walk back to one."""
        self.graph.register(self._sig, options)
        untested = [n for n in self.graph.untested_at(self._sig) if n in options]
        if untested:
            self.graph.take(self._sig, untested[0])
            self.stats["frontier"] += 1
            self._policy_streak += 1
            self._frontier_plan.clear()
            return self._option_action(untested[0])

        # A plan is a route through states, so it is void the moment a step
        # cannot be played here — following it blind would walk a route that
        # no longer exists.
        if self._frontier_plan and self._frontier_plan[0] not in options:
            self._frontier_plan.clear()
        if not self._frontier_plan:
            self._frontier_plan = deque(self.graph.path_to_frontier(self._sig))
        if self._frontier_plan:
            self.stats["frontier_walk"] += 1
            self._policy_streak += 1
            return self._option_action(self._frontier_plan.popleft())
        return None

    def state_signature(
        self, board_cells: int
    ) -> tuple[tuple[int, str, int, int], ...]:
        """Identity of the board as an arrangement of pieces.

        Keyed on objects rather than pixels. The explorer keys its frontier on
        every in-field pixel and its own TODO records what that costs: on live
        ls20 the board produced 741 distinct signatures for 30 avatar
        positions, so incidental churn fragmented one state into many and the
        frontier never recognised a revisit. Scenery is already filtered out
        here by size, and a piece is identified by colour, shape and position,
        so a flickering background does not create a new state.
        """
        limit = board_cells * ObjectTracker.SCENERY_SHARE
        return tuple(
            sorted(
                (node["colour"], node["hash"], *min(node["cells"]))
                for node in self._segmentation["nodes"]
                if node["pixels"] <= limit
            )
        )

    def observe_frame(self, grid: list[list[int]]) -> None:
        """Take in one frame: segment it, match objects, score the world model.

        Called exactly once per turn from the turn loop. It has to be separate
        from prompt_for, which runs twice whenever the tool call missed and so
        fed the model a second, self-matched frame in which nothing had moved.
        """
        board = tuple(tuple(row) for row in grid)
        self._segmentation = segment(board)
        self._seg_board = board
        self._last_events = self.objects.update(
            self._segmentation, len(grid) * len(grid[0])
        )
        previous_sig = self._sig
        self._sig = self.state_signature(len(grid) * len(grid[0]))
        if previous_sig is not None and self._last_action_name not in {"None", "RESET"}:
            self.graph.connect(previous_sig, self._last_action_name, self._sig)
        # RESET has no predecessor, and the "movement" from a fresh layout is
        # not an action's effect.
        if self._last_action_name.split("@")[0] in {"None", "RESET"}:
            return
        # Clicks all share the family MOUSE but not their target, so one key
        # would average unrelated effects into a translation no search could
        # act on: it carries no coordinates. Learn the full identity instead.
        if self.objects.matched:
            self.forward.observe(self._last_action_name, self.objects.last_moves)
            self.forward.moving.update(self.objects.movers())
            self.stats["forward_predicted"] = self.forward.predicted
            self.stats["forward_correct"] = self.forward.correct

    def prompt_for(
        self,
        grid: list[list[int]],
        valid_names: list[str],
        state: str = "NOT_FINISHED",
        effect: str | None = None,
        tool_mode: bool = False,
    ) -> str:
        """Build a turn's prompt from a board.

        The single entry point for prompt construction, so callers outside the
        turn loop — the notebook preflight in particular — cannot drift from
        what the agent actually sends. A preflight that reassembled this by hand
        broke on a signature change and failed a build.

        A pure builder: it is called twice on any turn whose tool call missed,
        once eagerly and once through the raw-text thunk, and the preflight
        calls it with a synthetic board. Anything stateful here runs a variable
        number of times per turn — which is how a phantom no-move entry got
        into the world model. Frame-to-frame state belongs in observe_frame.
        """
        encoded, bbox = self._encode_grid(grid)
        # Reuse what observe_frame computed for this exact board; segment only
        # for a board it has not seen, which is the preflight's synthetic one.
        # Keyed on the board rather than on emptiness: _segmentation starts as
        # an empty-but-not-None dict, so a None check would never fire and the
        # preflight would describe a board with no objects in it.
        board = tuple(tuple(row) for row in grid)
        if getattr(self, "_seg_board", None) != board:
            self._segmentation = segment(board)
            self._seg_board = board
        return self._build_prompt(
            encoded,
            describe_segmentation(self._segmentation),
            bbox,
            valid_names,
            state,
            self.memory.describe_last() if effect is None else effect,
            tool_mode=tool_mode,
        )

    def _progress_line(self) -> str:
        """Where the level stands: how much has been done, and what it cost.

        The score divides a level's baseline action count by the actions taken
        and squares it, so wasted turns are expensive in a way nothing on the
        board shows. Stating the count keeps that in front of the model.
        """
        parts = [f"level {self._prev_levels}", f"action {self._step}"]
        if self.undo.candidate:
            parts.append(f"{self.undo.candidate} undoes the last move for free")
        return "; ".join(parts)

    def _build_prompt(
        self,
        encoded_grid: str,
        objects: str,
        bbox: tuple[int, int, int, int] | None,
        valid_names: list[str],
        state: str,
        effect: str,
        tool_mode: bool = False,
    ) -> str:
        """Build prompt for the LLM.

        Base is the v26 prompt (the highest-scoring build, 0.17) plus a
        full-board click guide. v27's "Reasoning:" line, recent-actions history
        and softer system prompt appear to have hurt the agent, so they are
        dropped. `tool_mode` swaps the strict raw-text output instruction for
        a function-calling one.
        """
        notes = self.memory.prompt_notes(valid_names)
        moved = "; ".join(self._last_events[:6]) or "nothing moved"
        progress = self._progress_line()
        # Marked as the model's own guess, not fact: it is asserted with the same
        # confidence whether it was inferred from one frame or fifty, and a wrong
        # theory presented as a rule is worse than none.
        # Only in tool mode: the note can only be written through a tool argument,
        # so on the raw-text path this would be a permanently false line saying no
        # theory exists, costing tokens every turn and never becoming actionable.
        theory = ""
        if MECHANIC_NOTES and tool_mode:
            theory = (
                (
                    f"\nYour theory so far (yours, may be wrong - revise it): "
                    f"{self.memory.mechanic}"
                )
                if self.memory.mechanic
                else ("\nYou have not yet recorded a theory of this game's mechanic.")
            )
        click_guide = ""
        if bbox is not None:
            min_r, min_c, max_r, max_c = bbox
            click_guide = (
                " MOUSE row/col are FULL-BOARD coordinates (0-63), not offsets "
                f"into the shown region. The shown region is rows {min_r}..{max_r}, "
                f"cols {min_c}..{max_c}; to click the cell at relative (r, c) inside "
                f"it, use row={min_r} + r, col={min_c} + c."
            )
        if tool_mode:
            instructions = (
                "INSTRUCTIONS:\n"
                "1. Analyze the board and infer the game rules as you go.\n"
                "2. Call EXACTLY ONE function for the best action.\n"
                f"3. The available functions are: {', '.join(valid_names)}\n"
                "4. For clicks, call MOUSE with the full-board row/col (0-63)."
                + click_guide
                + (
                    "\n5. Pass `note`: at most 12 words stating a RULE that is true "
                    "every turn, e.g. 'arrows push blocks, blocks stop at walls'. "
                    "NOT your plan for this turn and NOT the board state. Repeat "
                    "your previous note unchanged unless the board disproved it."
                    if MECHANIC_NOTES
                    else ""
                )
            )
        else:
            instructions = (
                "INSTRUCTIONS:\n"
                "1. Analyze the board and infer the game rules as you go.\n"
                "2. Choose the best action from the valid actions list.\n"
                "3. You MUST output EXACTLY ONE line in this format: action(['ACTION_NAME'])\n"
                f"   Where ACTION_NAME is one of: {', '.join(valid_names)}\n"
                "4. For mouse clicks, use: action([{'action': 'MOUSE', 'row': <row>, 'col': <col>}])"
                + click_guide
                + "\n"
                "Example output: action(['RIGHT'])"
            )

        return f"""You are solving a grid-based puzzle game. You MUST choose exactly one action from the valid actions list.

Current state: {state}, step {self._step}
Last action: {self._last_action_name}
Effect of that action: {effect}
What moved: {moved}
Progress: {progress}{theory}{notes}

Valid actions: {", ".join(valid_names)}

Objects on the board (4-connected same-colour; hash ignores position):
{objects}

Current board (symbols: {ARC_LEGEND}):
{encoded_grid}

{instructions}"""

    def _call_llm(
        self,
        raw_prompt: Callable[[], str],
        tool_prompt: str,
        available: list[int],
        board: Grid | None = None,
    ) -> Any:
        """Call the local LLM and parse the response.

        `raw_prompt` is a thunk: the raw-text prompt is only built if the tool
        call misses. Past the run deadline we skip inference entirely so the
        remaining games drain at network speed rather than inference speed.
        """
        if time.monotonic() > _RUN_DEADLINE:
            self.stats["past_deadline"] += 1
            return self._random_arc_action(available)

        # Prefer structured tool calls (qwen3 native function calling): more
        # reliable than free-text parsing. Uses a dedicated tool-mode prompt so
        # the model calls a function rather than echoing raw-text instructions.
        action = self._call_tool(available, tool_prompt, board)
        if action:
            return action

        # Raw-text fallback: one more call, parsing action([...]) from the reply.
        self.stats["raw_text_fallback"] += 1
        response = self._complete(RAW_TEXT_SYSTEM, raw_prompt(), temperature=0.6)
        if response is None:
            self.stats["no_model"] += 1
            return self._random_arc_action(available)
        content = strip_thinking(
            response["choices"][0]["message"].get("content", "") or ""
        )
        if parsed := self._parse_action_from_code(content, available):
            self.stats["raw_text_parsed"] += 1
            return parsed
        self.stats["random_fallback"] += 1
        return self._random_arc_action(available)

    @contextmanager
    def _model(self) -> Iterator[Any | None]:
        """Borrow a model: the injected one for tests, else one from the pool."""
        if self._llm is not None:
            with _INFERENCE_LOCK:  # a single instance is not thread-safe
                yield self._llm
        elif REMOTE_BACKEND is not None:
            yield REMOTE_BACKEND  # vLLM batches concurrent requests itself
        else:
            with MODEL_POOL.acquire() as llm:
                yield llm

    def _user_content(self, prompt: str, board: Grid | None) -> Any:
        """The turn's message: text, plus the board as a picture when possible.

        Falls back to plain text if the image cannot be produced, so a missing
        PIL or an odd board costs nothing.
        """
        if not (SEND_IMAGE and board):
            return prompt
        png = render_board_png(board, IMAGE_CELL_PX)
        if png is None:
            if not self.stats["image_unavailable"]:
                print(f"[MyAgent] no board image: {_RENDER_FAILURE or 'unknown'}")
            self.stats["image_unavailable"] += 1
            return prompt
        self.stats["image_sent"] += 1
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(png)}},
        ]

    def _inspect(self, arguments: str, board: Grid) -> str:
        """Run one model-written inspection against the current board."""
        try:
            code = json.loads(arguments).get("code", "") if arguments else ""
        except (TypeError, ValueError):
            code = arguments or ""
        return run_python(
            code,
            {
                "grid": board,
                "objects": self._segmentation,
                "prev": self._last_grid,
                "SYMBOLS": ARC_SYMBOLS,
            },
        )

    def _complete_messages(
        self, messages: list[dict[str, Any]], temperature: float, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Send an explicit message list (the REPL loop builds one)."""
        with self._model() as llm:
            if llm is None:
                return None
            return llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                **kwargs,
            )

    def _complete(
        self, system: str, user: str, temperature: float, **kwargs: Any
    ) -> dict[str, Any] | None:
        """One single-shot completion, or None if no model is available.

        Single-shot is the point: a board spanning the 64x64 grid is ~1.2-2.4k
        tokens on its own, so accumulating turns in a chat history overflowed
        n_ctx within a few steps, after which every call raised and the agent
        silently degraded to random moves for the rest of the game.
        """
        with self._model() as llm:
            if llm is None:
                return None
            return llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                **kwargs,
            )

    def _call_tool(
        self, available: list[int], prompt: str, board: Grid | None = None
    ) -> Any | None:
        """Choose an action, optionally inspecting the board with Python first.

        Up to REPL_STEPS `python` calls may precede the action, each running
        against the current board and printing back to the model. This is the
        Duck harness's central idea: let the model interrogate the board rather
        than read a wall of digits. The loop is bounded and every exchange is
        small (code plus output), so the board still appears exactly once.

        Returns an ArcAction, or None on any failure so the caller falls back to
        raw-text parsing.
        """
        tools = self._build_tools(available)
        if not tools:
            return None
        if board is not None and REPL_STEPS:
            tools = [*tools, PYTHON_TOOL]
        messages = [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": self._user_content(prompt, board)},
        ]
        try:
            for step in range(REPL_STEPS + 1):
                # The last turn must produce an action, so drop the python tool.
                turn_tools = (
                    tools if step < REPL_STEPS else self._build_tools(available)
                )
                response = self._complete_messages(
                    messages, temperature=0.4, tools=turn_tools, tool_choice=TOOL_CHOICE
                )
                if response is None:
                    return None
                message = response["choices"][0]["message"]
                content = strip_thinking(message.get("content") or "")
                calls = (message.get("tool_calls") or []) or parse_text_tool_calls(
                    content
                )

                code = next(
                    (
                        (c.get("function") or {}).get("arguments")
                        for c in calls
                        if ((c.get("function") or {}).get("name") or "").lower()
                        == "python"
                    ),
                    None,
                )
                if code is None or board is None:
                    break
                self.stats["repl_call"] += 1
                output = self._inspect(code, board)
                if output.split("\n")[-1].startswith(
                    (
                        "NameError",
                        "SyntaxError",
                        "TypeError",
                        "ValueError",
                        "KeyError",
                        "AttributeError",
                    )
                ):
                    self.stats["repl_error"] += 1
                messages.extend(
                    [
                        {"role": "assistant", "content": f"python:\n{code}"},
                        {
                            "role": "user",
                            "content": f"python output:\n{output}\n\nNow act.",
                        },
                    ]
                )
            else:
                return None
            # The template may leave the call as text in `content` instead of
            # populating `tool_calls`; recover it rather than paying for a
            # second inference (or losing the click coordinates entirely).
            for tc in calls:
                fn = tc.get("function") or {}
                name = (fn.get("name") or "").upper()
                aid = NAME_TO_ID.get(name)
                if aid is None or aid not in available:
                    continue
                args: dict[str, Any] = {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {}
                self.stats[
                    "tool_call" if message.get("tool_calls") else "tool_from_text"
                ] += 1
                if MECHANIC_NOTES and self.memory.record_mechanic(
                    args.get("note") or ""
                ):
                    self.stats["mechanic_note"] += 1
                if aid != COMPLEX_ACTION_ID:
                    return ArcAction(id=aid)
                try:
                    x = int(args.get("x", args.get("col")))
                    y = int(args.get("y", args.get("row")))
                except (TypeError, ValueError):
                    self.stats["mouse_without_coords"] += 1
                    continue
                return ArcAction(
                    id=COMPLEX_ACTION_ID, x=clamp_coord(x), y=clamp_coord(y)
                )
            # No usable tool call, but the reply may still name an action.
            # Parsing it here saves a second inference round-trip per turn.
            return self._parse_action_from_code(content, available)
        except Exception as exc:  # noqa: BLE001
            self.stats["tool_path_exception"] += 1
            print(
                f"[MyAgent] {self.game_id}: tool-call failed ({exc!r}); using raw-text path"
            )
        return None

    @staticmethod
    def _build_tools(available: list[int]) -> list[dict[str, Any]]:
        """Build JSON function definitions for the actions currently available.

        Deliberately minimal. The schema is re-prefilled every turn, and
        per-action descriptions only restate the function name the prompt
        already lists — ~260 tokens per turn of pure boilerplate.
        """
        tools: list[dict[str, Any]] = []
        for aid, name in ACTION_NAMES.items():
            if aid not in available:
                continue
            if aid == COMPLEX_ACTION_ID:
                params: dict[str, Any] = {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "integer",
                            "description": f"Column 0-{GRID_SIZE - 1}",
                        },
                        "y": {
                            "type": "integer",
                            "description": f"Row 0-{GRID_SIZE - 1}",
                        },
                    },
                    "required": ["x", "y"],
                }
            else:
                params = {"type": "object", "properties": {}}
            if MECHANIC_NOTES:
                # Optional, and on every action, so the theory rides along with the
                # move the model was already making rather than costing a turn.
                # Bounded in the schema, not just on storage: MAX_OUTPUT_TOKENS is
                # 128 and shared with the thinking block, so a long theory can run
                # the JSON out of budget mid-object. The call then fails to parse
                # and a MOUSE loses its coordinates entirely.
                params["properties"]["note"] = {
                    "type": "string",
                    "maxLength": MAX_NOTE_CHARS,
                    "description": "At most 12 words: your theory of this game's mechanic",
                }
            tools.append(
                {"type": "function", "function": {"name": name, "parameters": params}}
            )
        return tools

    def _random_arc_action(self, available: list[int]) -> Any:
        """Random ArcAction helper (used when the model is unavailable or unparseable)."""
        choices = [a for a in (available or []) if a != 0]
        if not choices:
            choices = [1, 2, 3, 4, 5]
        choice = self._rng.choice(choices)

        if choice == COMPLEX_ACTION_ID:
            return ArcAction(
                id=COMPLEX_ACTION_ID,
                x=self._rng.randrange(GRID_SIZE),
                y=self._rng.randrange(GRID_SIZE),
            )
        return ArcAction(id=choice)

    def _parse_action_from_code(self, text: str, available: list[int]) -> Any | None:
        """Parse action from raw LLM text output."""
        # Pattern 1: action(['ACTION_NAME']) or action(["ACTION_NAME"]) or action([ACTION_NAME]).
        # First match wins: the v26 prompt forbids extra lines, so the first
        # action([...]) is the intended one.
        match = re.search(r"action\(\[['\"]?(\w+)['\"]?\]\)", text)
        if match:
            act_name = match.group(1).upper()
            act_id = NAME_TO_ID.get(act_name)
            if act_id is not None and act_id in available:
                return ArcAction(id=act_id)

        # Pattern 1b: action([ACTION_NAME (incomplete - missing closing bracket)
        match = re.search(r"action\(\[([A-Z]+)", text)
        if match:
            act_name = match.group(1).upper()
            act_id = NAME_TO_ID.get(act_name)
            if act_id is not None and act_id in available:
                return ArcAction(id=act_id)

        # Pattern 2: action([{...MOUSE...}]) for click. Extract the coord keys
        # order-independently (models write x/y or row/col, in either order).
        match = re.search(
            r"action\(\[\{.*?['\"]action['\"]\s*:\s*['\"]MOUSE['\"].*?\}(?:\s*\])?\)",
            text,
            re.DOTALL,
        )
        if match:
            block = match.group(0)
            coords = {
                key: int(m.group(1))
                for key in ("row", "y", "col", "x")
                if (m := re.search(rf"['\"]{key}['\"]\s*:\s*(\d+)", block)) is not None
            }
            row = coords.get("row", coords.get("y"))
            col = coords.get("col", coords.get("x"))
            if row is not None and col is not None:
                return ArcAction(
                    id=COMPLEX_ACTION_ID, x=clamp_coord(col), y=clamp_coord(row)
                )

        # Pattern 3: keyword in text (fallback). RESET is excluded: it is a
        # common English word, so a reasoning sentence would false-positive.
        text_upper = text.upper()
        for name, aid in NAME_TO_ID.items():
            if aid == 0:
                continue
            if aid in available and name in text_upper:
                # Only match if it's a clear keyword (not embedded in a word)
                if re.search(rf"\b{name}\b", text_upper):
                    return ArcAction(id=aid)

        return None

    def _arc_to_game_action(self, arc_action: Any) -> GameAction:
        """Pick the GameAction member and stage its payload on this agent.

        GameAction members are process-wide Enum singletons, so `set_data()`
        writes state every concurrent game shares. We stage the payload per
        agent instead and hand it to `arc_env.step()` as an argument, which is
        what the wrapper actually reads — the enum's mutable fields never come
        into it, so two games cannot clobber each other's coordinates.
        """
        game_action = GameAction.from_id(arc_action.id)

        if not game_action.is_complex():
            self._pending_data = None
            self._pending_reasoning = f"REPL agent action {arc_action.id}"
            return game_action

        # A complex action always needs coordinates; the text parser can name
        # MOUSE without them, so fall back to a random click rather than (0, 0).
        x = arc_action.x if arc_action.x is not None else self._rng.randrange(GRID_SIZE)
        y = arc_action.y if arc_action.y is not None else self._rng.randrange(GRID_SIZE)
        self._pending_data = {"x": x, "y": y}
        self._pending_reasoning = {"why": f"REPL agent click at ({x}, {y})"}
        return game_action

    def do_action_request(self, action: GameAction) -> Any:
        """Dispatch with the staged payload, bypassing the shared enum entirely.

        The base class reads `action.action_data` / `action.reasoning` off the
        Enum member; `arc_env.step()` only ever uses its `data`/`reasoning`
        arguments (it touches `action` solely for RESET and `.value`), so
        passing them directly removes the shared mutable state and the lock it
        would otherwise need — and keeps the gateway round-trip concurrent.
        """
        return self._convert_raw_frame_data(
            self.arc_env.step(
                action,
                data=self._pending_data,
                reasoning=self._pending_reasoning_dict(),
            )
        )

    def _pending_reasoning_dict(self) -> dict[str, str] | None:
        """Normalize staged reasoning to the dict shape the wrapper expects."""
        if self._pending_reasoning is None or isinstance(self._pending_reasoning, dict):
            return self._pending_reasoning
        return {"text": str(self._pending_reasoning)}


class ExplorerAgent(MyAgent):
    """Drive the model-free explorer through the framework's Agent interface.

    Only the seam lives here; action conversion, the random fallback and the
    run budget are MyAgent's. The observation keys mirror
    `ArcAgi3Environment._obs`, which is what feeds the explorer locally — the
    two are assembled from different frame types and must stay in step.
    """

    USES_MODEL = False  # induction only; it never calls the model

    # The scored rerun uses the class default: it launches `main.py --agent
    # explorer`, and the mock's `agent.MAX_ACTIONS = MOCK_ACTIONS` override
    # runs only in the mock. At MyAgent's 400 the explorer finished 110 games
    # in ~12 minutes of a 7.5h budget — 2.7% of the wall clock — while levels
    # were still arriving: on the local roster 600 -> 6000 actions took 6
    # levels to 14, over 5 games to 8.
    #
    # 6000 x 110 games at the measured 3811 actions/min is ~2.9h, leaving
    # margin for slower hidden games; is_done()'s global deadline remains the
    # real safety valve. The LLM agent keeps 400: at 1051 actions/min the same
    # budget would need 10.5h and blow the deadline.
    MAX_ACTIONS = 6000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Local import: my_agent.py is inlined standalone into the kernel, so a
        # top-level one would kill every game when the package is not staged.
        from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent

        self._explorer = ExplorerArcAgi3Agent()

    def _record_branch(self) -> None:
        """Count which policy chose this action.

        The scorer's fault counters are what separate a healthy run from one
        that has silently degraded, and the explorer filled none of them: a run
        stuck in stall rotation reported the same empty dict as one navigating
        and clearing levels.
        """
        trace = getattr(self._explorer, "trace", None)
        if trace and trace.get("branch"):
            self.stats[f"branch_{trace['branch']}"] += 1

    def _choose_action_inner(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        observation = {
            "frame": latest_frame.frame or [],
            "available_actions": latest_frame.available_actions or [1],
            "levels_completed": latest_frame.levels_completed,
            "terminal": latest_frame.state is GameState.GAME_OVER,
        }
        # Asked even when the board is dead: this is the frame that teaches the
        # explorer which edge killed it, and it refuses to repeat a recorded
        # one. On a dead board the answer is then dropped for the restart.
        arc_action = self._explorer.act(observation)
        self._record_branch()

        # An empty board before the first RESET and a dead one after GAME_OVER
        # can only be restarted. _last_action_id == 0 means the last thing sent
        # was a RESET that did not take, so play a real input rather than spin.
        if self._unplayable(latest_frame) and self._last_action_id != 0:
            return self._make_reset()

        # MyAgent keeps this truthful via _record_action, which also feeds a
        # world model the explorer does not have. Without the write the id
        # stays 0 after the first restart and every later death is played out
        # on a dead board — the exact budget burn the check above prevents.
        self._last_action_id = arc_action.id
        return self._arc_to_game_action(arc_action)
