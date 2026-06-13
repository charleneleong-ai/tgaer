from __future__ import annotations

from typing import Any

from tgaer.envs.arc_agi3.arc_agi3_api import ArcFrame

BASE_URL = "https://three.arcprize.org"


def _to_frame(data: dict) -> ArcFrame:
    return ArcFrame(
        game_id=data["game_id"],
        guid=data["guid"],
        frame=data["frame"],
        state=data["state"],
        levels_completed=data["levels_completed"],
        win_levels=data["win_levels"],
        available_actions=data["available_actions"],
    )


class ArcAgi3Client:
    """Transport for the hosted ARC-AGI-3 API at ``three.arcprize.org``.

    Implements the ``ArcTransport`` protocol (``reset``/``act``). A ``session``
    may be injected for tests; otherwise a ``requests.Session`` is created lazily
    so it carries the API's session-affinity cookies across requests.
    """

    def __init__(
        self,
        api_key: str,
        card_id: str | None = None,
        base_url: str = BASE_URL,
        session: Any = None,
    ) -> None:
        if session is None:
            import requests  # lazy: only the live transport needs the dependency

            session = requests.Session()
        self._api_key = api_key
        self._card_id = card_id
        self._base_url = base_url.rstrip("/")
        self._session = session

    def _post(self, path: str, body: dict) -> dict:
        resp = self._session.post(
            f"{self._base_url}{path}",
            json=body,
            headers={"X-API-Key": self._api_key, "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

    def open_scorecard(self) -> str:
        self._card_id = self._post("/api/scorecard/open", {})["card_id"]
        return self._card_id

    def close_scorecard(self) -> dict:
        return self._post("/api/scorecard/close", {"card_id": self._card_id})

    def reset(self, game_id: str) -> ArcFrame:
        body = {"game_id": game_id, "card_id": self._card_id, "guid": None}
        return _to_frame(self._post("/api/cmd/RESET", body))

    def act(
        self,
        game_id: str,
        guid: str,
        action_id: int,
        x: int | None = None,
        y: int | None = None,
    ) -> ArcFrame:
        body: dict = {"game_id": game_id, "guid": guid}
        if action_id == 6:
            body["x"] = x
            body["y"] = y
        return _to_frame(self._post(f"/api/cmd/ACTION{action_id}", body))
