from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
import verifiers as vf
from arcengine import ARCBaseGame, ActionInput, GameAction as ArcGameAction
from datasets import Dataset
from dotenv import load_dotenv
from verifiers.types import UserMessage

from arc_games import build_complex_maze, build_simple_maze

ACTION_PATTERN = re.compile(r"<action>\s*(RESET|ACTION[1-7])\s*</action>", re.IGNORECASE)
X_PATTERN = re.compile(r"<x>\s*(\d+)\s*</x>", re.IGNORECASE)
Y_PATTERN = re.compile(r"<y>\s*(\d+)\s*</y>", re.IGNORECASE)
SUPPORTED_GAME_FAMILIES = ("simple_maze", "complex_maze", "arc_agi")

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass
class StepResult:
    observation: str
    game_state: str | None
    num_actions: int
    available_actions: list[dict[str, Any]]
    full_reset: bool = False
    error: str | None = None


def reward(state: dict[str, Any]) -> float:
    return 1.0 if state.get("game_state") == "WIN" else 0.0


def num_actions(state: dict[str, Any]) -> float:
    return float(state.get("num_actions", 0))


def timed_out(state: dict[str, Any]) -> float:
    is_terminal = state.get("game_state") in {"WIN", "GAME_OVER"}
    return 1.0 if not is_terminal and state.get("agent_turns", 0) >= state.get("max_turns", 0) else 0.0


class ArcAgi3Rubric(vf.Rubric):
    def __init__(self) -> None:
        super().__init__(funcs=[reward], weights=[1.0])
        self.add_metric(num_actions)
        self.add_metric(timed_out)


def _parse_action(text: str) -> tuple[str, dict[str, int]] | None:
    match = ACTION_PATTERN.search(text or "")
    if not match:
        return None
    action_name = match.group(1).upper()
    data: dict[str, int] = {}
    x_match = X_PATTERN.search(text or "")
    y_match = Y_PATTERN.search(text or "")
    if x_match:
        data["x"] = int(x_match.group(1))
    if y_match:
        data["y"] = int(y_match.group(1))
    return action_name, data


def _frame_to_grid(frame: Any) -> list[list[int]]:
    if not frame:
        return []
    if isinstance(frame, list) and frame and isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        return frame[-1]
    return frame


def _local_game_grid(game: ARCBaseGame) -> list[list[int]]:
    grid_size = getattr(game.current_level, "grid_size", None)
    if not grid_size:
        return []
    width, height = grid_size
    pixels = game.get_pixels(0, 0, width, height)
    return pixels.tolist()


def _grid_to_text(grid: list[list[int]]) -> str:
    if not grid:
        return "<empty>"
    return "\n".join(" ".join(f"{cell:>2}" for cell in row) for row in grid)


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    if not grid:
        return (0, 0)
    return (len(grid[0]), len(grid))


def _available_action_specs(raw_actions: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for action in raw_actions or []:
        if isinstance(action, int):
            specs.append({"name": ArcGameAction.from_id(action).name})
        elif hasattr(action, "name"):
            spec: dict[str, Any] = {"name": str(action.name)}
            data = getattr(action, "data", None)
            if data:
                spec["data"] = dict(data)
            specs.append(spec)
        else:
            specs.append({"name": str(action)})
    return specs


def _available_action_text(action_specs: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for spec in action_specs:
        name = str(spec.get("name", ""))
        data = spec.get("data")
        if data and "x" in data and "y" in data:
            parts.append(f"{name}(x={data['x']}, y={data['y']})")
        else:
            parts.append(name)
    return ", ".join(parts) if parts else "NONE"


def _format_observation(*, game_id: str, game_state: str, grid: list[list[int]], available_actions: list[dict[str, Any]], num_actions: int, extra: str | None = None) -> str:
    width, height = _grid_shape(grid)
    parts = [
        f"game_id: {game_id}",
        f"state: {game_state}",
        f"num_actions: {num_actions}",
        f"grid_size: width={width}, height={height}",
        f"available_actions: {_available_action_text(available_actions)}",
        "grid:",
        _grid_to_text(grid),
    ]
    if extra:
        parts.extend(["", extra])
    return "\n".join(parts)


class LocalArcEngineBackend:
    def __init__(self, family: str, level_index: int) -> None:
        self.family = family
        self.level_index = level_index
        self.game: ARCBaseGame | None = None
        self.factory: Callable[[], ARCBaseGame] = build_simple_maze if family == "simple_maze" else build_complex_maze

    def step(self, action_name: str, action_data: dict[str, int] | None = None) -> StepResult:
        if self.game is None:
            if action_name != "RESET":
                return StepResult(
                    observation="The game has not started yet. First send `<action>RESET</action>`.",
                    game_state=None,
                    num_actions=0,
                    available_actions=[{"name": "RESET"}],
                    error="game_not_started",
                )
            self.game = self._build_game()

        arc_action = ArcGameAction[action_name]
        action_payload = {"game_id": self.game.game_id, **(action_data or {})}
        frame = self.game.perform_action(ActionInput(id=arc_action, data=action_payload))
        available_actions = [{"name": "RESET"}, *_available_action_specs(frame.available_actions)]
        game_state = frame.state.value
        num_actions = int(getattr(self.game, "_action_count", 0))
        observation = _format_observation(
            game_id=str(frame.game_id),
            game_state=game_state,
            grid=_local_game_grid(self.game),
            available_actions=available_actions,
            num_actions=num_actions,
        )
        return StepResult(
            observation=observation,
            game_state=game_state,
            num_actions=num_actions,
            available_actions=available_actions,
            full_reset=bool(frame.full_reset),
        )

    def _build_game(self) -> ARCBaseGame:
        game = self.factory()
        levels = list(getattr(game, "_clean_levels"))
        if self.level_index < 0 or self.level_index >= len(levels):
            raise ValueError(f"level_index={self.level_index} is out of range for {self.family}")
        selected_level = levels[self.level_index].clone()
        game._levels = [selected_level.clone()]
        game._clean_levels = [selected_level.clone()]
        game._win_score = 1
        game.set_level(0)
        return game


class RemoteArcAgiBackend:
    def __init__(self, root_url: str, api_key: str, game_id: str) -> None:
        self.root_url = root_url.rstrip("/")
        self.game_id = game_id
        self.card_id: str | None = None
        self.guid: str | None = None
        self.num_actions = 0
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key, "Accept": "application/json"})

    def step(self, action_name: str, action_data: dict[str, int] | None = None) -> StepResult:
        if self.card_id is None:
            if action_name != "RESET":
                return StepResult(
                    observation="The remote game has not started yet. First send `<action>RESET</action>`.",
                    game_state=None,
                    num_actions=0,
                    available_actions=[{"name": "RESET"}],
                    error="game_not_started",
                )
            self.card_id = self._open_scorecard()

        payload: dict[str, Any] = {"game_id": self.game_id}
        if action_data:
            payload.update(action_data)
        if action_name == "RESET":
            payload["card_id"] = self.card_id
        if self.guid:
            payload["guid"] = self.guid

        response = self.session.post(f"{self.root_url}/api/cmd/{action_name}", json=payload, timeout=30)
        response.raise_for_status()
        frame = response.json()

        if "guid" in frame and frame["guid"]:
            self.guid = str(frame["guid"])

        if action_name != "RESET":
            self.num_actions += 1
        elif frame.get("full_reset"):
            self.num_actions = 0

        available_actions = [{"name": "RESET"}, *_available_action_specs(frame.get("available_actions", []))]
        game_state = str(frame.get("state", "NOT_PLAYED"))
        extra = None
        if any(spec.get("name") == "ACTION6" for spec in available_actions):
            extra = "\n".join(
                [
                    "ACTION6 usage:",
                    "- Send click actions as `<action>ACTION6</action><x>X</x><y>Y</y>`.",
                    "- Coordinates use the rendered grid origin at the top-left corner.",
                    "- `x` increases to the right and `y` increases downward.",
                ]
            )
        observation = _format_observation(
            game_id=str(frame.get("game_id", self.game_id)),
            game_state=game_state,
            grid=_frame_to_grid(frame.get("frame", [])),
            available_actions=available_actions,
            num_actions=self.num_actions,
            extra=extra,
        )
        return StepResult(
            observation=observation,
            game_state=game_state,
            num_actions=self.num_actions,
            available_actions=available_actions,
            full_reset=bool(frame.get("full_reset", False)),
        )

    def _open_scorecard(self) -> str:
        response = self.session.post(f"{self.root_url}/api/scorecard/open", json={"tags": ["vf", "arc-agi-3-env"]}, timeout=30)
        response.raise_for_status()
        data = response.json()
        card_id = data.get("card_id")
        if not card_id:
            raise ValueError("ARC API did not return a card_id when opening a scorecard")
        return str(card_id)


class ArcAgi3Env(vf.MultiTurnEnv):
    def __init__(self, *, dataset: Dataset, rubric: vf.Rubric, max_turns: int) -> None:
        super().__init__(dataset=dataset, rubric=rubric, max_turns=max_turns)
        self.max_turns = max_turns

    async def is_completed(self, state: dict[str, Any], **kwargs: Any) -> bool:
        state.setdefault("trajectory", [])
        if await super().is_completed(state, **kwargs):
            return True
        return state.get("game_state") in {"WIN", "GAME_OVER"}

    async def env_response(self, messages: Any, state: dict[str, Any], **kwargs: Any) -> list[UserMessage]:
        state["agent_turns"] = int(state.get("agent_turns", 0)) + 1
        state["max_turns"] = self.max_turns
        info = kwargs.get("info") or state.get("info") or {}
        state["info"] = info

        last_message = messages[-1] if messages else {}
        parsed_action = _parse_action(last_message.get("content", ""))
        if parsed_action is None:
            return [
                UserMessage(content="Invalid action format. Reply with `<action>...</action>`. For ACTION6, also include `<x>...</x>` and `<y>...</y>`.")
            ]
        action_name, action_data = parsed_action

        backend = state.get("backend")
        if backend is None:
            backend = self._build_backend(info)
            state["backend"] = backend

        try:
            result = backend.step(action_name, action_data)
        except Exception as exc:
            return [UserMessage(content=f"Environment error: {type(exc).__name__}: {exc}")]

        state["game_state"] = result.game_state
        state["num_actions"] = result.num_actions
        state["available_actions"] = result.available_actions
        state["last_observation"] = result.observation

        if result.game_state in {"WIN", "GAME_OVER"}:
            terminal_message = f"{result.observation}\n\nTerminal state reached: {result.game_state}."
            return [UserMessage(content=terminal_message)]

        return [UserMessage(content=result.observation)]

    def _build_backend(self, info: dict[str, Any]) -> LocalArcEngineBackend | RemoteArcAgiBackend:
        family = str(info["game_family"])
        if family == "arc_agi":
            root_url = os.environ.get("ROOT_URL", "").strip()
            api_key = os.environ.get("ARC_API_KEY", "").strip()
            if not root_url or not api_key:
                raise ValueError("ARC_API_KEY and ROOT_URL must be set for game_family='arc_agi'")
            return RemoteArcAgiBackend(root_url=root_url, api_key=api_key, game_id=str(info["game_id"]))
        return LocalArcEngineBackend(family=family, level_index=int(info["level_index"]))


def _build_prompt(game_family: str, identifier: str) -> list[dict[str, str]]:
    lines = [
        f"You are playing `{identifier}` from the `{game_family}` environment.",
        "Reply with exactly one XML action tag on every turn.",
        "Use `RESET` to initialize the game before any movement.",
        "After each action, you will receive the latest game state as a text grid.",
        "Your objective is to reach terminal state `WIN` while minimizing `num_actions`.",
    ]
    if game_family == "arc_agi":
        lines.append("For click-style actions, use `<action>ACTION6</action><x>12</x><y>34</y>`.")
    else:
        lines.append("Valid examples: `<action>RESET</action>`, `<action>ACTION1</action>`, `<action>ACTION2</action>`, `<action>ACTION3</action>`, `<action>ACTION4</action>`.")
    instructions = "\n".join(lines)
    return [{"role": "user", "content": instructions}]


def _list_arc_agi_games(root_url: str, api_key: str) -> list[str]:
    response = requests.get(
        f"{root_url.rstrip('/')}/api/games",
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    games = response.json()
    return [str(item["game_id"]) for item in games]


def _build_dataset(game_family: str, level_index: int, game_id: str | None) -> Dataset:
    rows: list[dict[str, Any]] = []

    if game_family == "simple_maze":
        level_indices = [level_index] if level_index >= 0 else [0, 1]
        for idx in level_indices:
            rows.append(
                {
                    "prompt": _build_prompt(game_family, f"simple_maze_level_{idx}"),
                    "info": {"game_family": game_family, "game_id": "simple_maze", "level_index": idx},
                }
            )
    elif game_family == "complex_maze":
        level_indices = [level_index] if level_index >= 0 else [0, 1, 2, 3, 4]
        for idx in level_indices:
            rows.append(
                {
                    "prompt": _build_prompt(game_family, f"complex_maze_level_{idx}"),
                    "info": {"game_family": game_family, "game_id": "complex_maze", "level_index": idx},
                }
            )
    elif game_family == "arc_agi":
        root_url = os.environ.get("ROOT_URL", "").strip()
        api_key = os.environ.get("ARC_API_KEY", "").strip()
        if not root_url or not api_key:
            raise ValueError("ARC_API_KEY and ROOT_URL must be set for game_family='arc_agi'")
        game_ids = [game_id] if game_id else _list_arc_agi_games(root_url=root_url, api_key=api_key)
        for remote_game_id in game_ids:
            rows.append(
                {
                    "prompt": _build_prompt(game_family, remote_game_id),
                    "info": {"game_family": game_family, "game_id": remote_game_id, "level_index": -1},
                }
            )
    else:
        raise ValueError(f"Unsupported game_family '{game_family}'. Expected one of {SUPPORTED_GAME_FAMILIES}.")

    return Dataset.from_list(rows)


def load_environment(
    game_family: str = "simple_maze",
    max_turns: int = 100,
    level_index: int = -1,
    game_id: str | None = None,
    system_prompt: str | None = None,
) -> vf.Environment:
    if game_family not in SUPPORTED_GAME_FAMILIES:
        raise ValueError(f"Unsupported game_family '{game_family}'. Expected one of {SUPPORTED_GAME_FAMILIES}.")

    dataset = _build_dataset(game_family=game_family, level_index=level_index, game_id=game_id)
    if system_prompt:
        rows = []
        for row in dataset:
            prompt = [{"role": "system", "content": system_prompt}, *row["prompt"]]
            rows.append({"prompt": prompt, "info": row["info"]})
        dataset = Dataset.from_list(rows)

    return ArcAgi3Env(dataset=dataset, rubric=ArcAgi3Rubric(), max_turns=max_turns)
