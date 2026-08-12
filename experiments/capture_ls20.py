"""Capture one live ls20 game's (obs, action) stream for offline telemetry replay.

The single paid step of the induction diagnostic: drives the real explorer over
one ls20 game through a RecordingAgent. No scorecard metric matters — only the
trajectory. Replay it for free with experiments/replay_telemetry.py.

    PYTHONPATH=src ./.venv/bin/python experiments/capture_ls20.py
"""

from __future__ import annotations

import os

import requests

from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
from tgaer.agents.arc_agi3_recorder import RecordingAgent
from tgaer.envs.arc_agi3.arc_agi3_client import BASE_URL, ArcAgi3Client
from tgaer.envs.arc_agi3.arc_agi3_env import ArcAgi3Environment
from tgaer.evaluation.arc_agi3_eval import evaluate_arc_agi3_agent

OUT = "/workspace/tgaer/experiments/ls20_capture.jsonl"
# reset_on_game_over keeps the episode alive across deaths/levels, so a generous
# action budget buys a long induction trajectory (per-game API cap notwithstanding).
MAX_ACTIONS = int(os.environ.get("MAX_ACTIONS", "1000"))


def _ls20_game_id(key: str) -> str:
    games = requests.get(f"{BASE_URL}/api/games", headers={"X-API-Key": key}).json()
    return next(g["game_id"] for g in games if g["game_id"].startswith("ls20"))


def main() -> None:
    key = os.environ["ARC_API_KEY"]
    gid = _ls20_game_id(key)
    print(f"[capture] ls20 = {gid}", flush=True)

    client = ArcAgi3Client(api_key=key)
    client.open_scorecard()  # the API only accepts actions under an open card
    env = ArcAgi3Environment(
        client, gid, max_actions=MAX_ACTIONS, reset_on_game_over=True
    )
    agent = RecordingAgent(ExplorerArcAgi3Agent(), OUT)
    try:
        result = evaluate_arc_agi3_agent(
            agent, env, {"guards": [], "max_steps": MAX_ACTIONS}
        )
        print(f"[capture] DONE score={result.score} -> {OUT}", flush=True)
    finally:
        client.close_scorecard()  # always release the paid card, even on crash


if __name__ == "__main__":
    main()
