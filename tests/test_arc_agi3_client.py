from __future__ import annotations

from tgaer.envs.arc_agi3.arc_agi3_api import ArcFrame
from tgaer.envs.arc_agi3.arc_agi3_client import ArcAgi3Client


def _frame_json(state="NOT_FINISHED", levels=0, guid="g1"):
    return {
        "game_id": "ls20-016295f7601e",
        "guid": guid,
        "frame": [[[0] * 64 for _ in range(64)]],
        "state": state,
        "levels_completed": levels,
        "win_levels": 254,
        "available_actions": [1, 2, 3, 4],
    }


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts: list[dict] = []

    def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self._responses.pop(0))


def _client(responses, card_id="card-1"):
    session = _FakeSession(responses)
    return ArcAgi3Client(api_key="k", card_id=card_id, session=session), session


class TestOpenScorecard:
    def test_posts_to_open_and_stores_card_id(self):
        client, session = _client([{"card_id": "abc"}], card_id=None)
        assert client.open_scorecard() == "abc"
        assert (
            session.posts[0]["url"] == "https://three.arcprize.org/api/scorecard/open"
        )
        assert session.posts[0]["headers"]["X-API-Key"] == "k"


class TestReset:
    def test_posts_reset_with_card_id_and_parses_frame(self):
        client, session = _client([_frame_json(levels=0, guid="g9")])
        frame = client.reset("ls20-016295f7601e")
        assert isinstance(frame, ArcFrame)
        assert frame.guid == "g9"
        post = session.posts[0]
        assert post["url"] == "https://three.arcprize.org/api/cmd/RESET"
        assert post["json"] == {
            "game_id": "ls20-016295f7601e",
            "card_id": "card-1",
            "guid": None,
        }


class TestAct:
    def test_simple_action_posts_to_numbered_endpoint_without_coords(self):
        client, session = _client([_frame_json()])
        client.act("ls20-016295f7601e", "g1", 3)
        post = session.posts[0]
        assert post["url"] == "https://three.arcprize.org/api/cmd/ACTION3"
        assert post["json"] == {"game_id": "ls20-016295f7601e", "guid": "g1"}

    def test_complex_action_posts_action6_with_xy(self):
        client, session = _client([_frame_json()])
        client.act("ls20-016295f7601e", "g1", 6, x=12, y=34)
        post = session.posts[0]
        assert post["url"] == "https://three.arcprize.org/api/cmd/ACTION6"
        assert post["json"] == {
            "game_id": "ls20-016295f7601e",
            "guid": "g1",
            "x": 12,
            "y": 34,
        }
