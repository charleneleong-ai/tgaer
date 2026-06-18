from __future__ import annotations

import numpy as np

from tgaer.agents.arc_agi3_grid import LS20_DEFAULT, Semantics
from tgaer.agents.arc_agi3_scientist import Scientist, ScientistPlannerAgent
from tgaer.envs.arc_agi3.arc_agi3_api import ArcAction


# board uses indices 3 (floor), 12 (avatar), 0/1 (key), 9 (door), 4 (wall)
def _frame():
    g = np.full((8, 8), 3, dtype=int)
    g[0, :] = 4
    g[2, 2] = 12
    g[4, 4] = 0
    g[6, 6] = 9
    return [g.tolist()]


def _sci(reply: str) -> Scientist:
    s = Scientist(model="openai/fake", api_base="http://localhost:8000/v1")
    s._complete = lambda _p, _img=None: reply  # type: ignore[method-assign]
    return s


VALID = '{"avatar": 12, "keys": [0, 1], "door": 9, "walls": [4], "verb": "navigate"}'


class TestInfer:
    def test_valid_reply_parses_to_semantics(self):
        sem = _sci(VALID).infer(_frame())
        assert sem == Semantics(12, (0, 1), 9, (4,), "navigate")

    def test_reply_in_prose_and_fences_is_parsed(self):
        reply = f"The avatar is darkred.\n```json\n{VALID}\n```"
        assert _sci(reply).infer(_frame()) == Semantics(12, (0, 1), 9, (4,), "navigate")

    def test_index_absent_from_grid_rejected(self):
        # avatar=7 never appears on the board -> hallucination -> None
        bad = '{"avatar": 7, "keys": [0], "door": 9, "walls": [4], "verb": "navigate"}'
        assert _sci(bad).infer(_frame()) is None

    def test_bad_verb_rejected(self):
        bad = '{"avatar": 12, "keys": [0], "door": 9, "walls": [4], "verb": "teleport"}'
        assert _sci(bad).infer(_frame()) is None

    def test_unparseable_reply_returns_none(self):
        assert _sci("no json at all").infer(_frame()) is None

    def test_press_verb_accepted(self):
        ok = '{"avatar": 12, "keys": [0], "door": 9, "walls": [4], "verb": "press"}'
        assert _sci(ok).infer(_frame()).verb == "press"

    def test_complete_exception_returns_none(self):
        s = Scientist(model="openai/fake", api_base="x")

        def _boom(_p, _img=None):
            raise RuntimeError("server down")

        s._complete = _boom  # type: ignore[method-assign]
        assert s.infer(_frame()) is None

    def test_hallucinated_keys_rejected(self):
        # keys=[7,8] — neither index is on _frame() → hallucination → None
        bad = '{"avatar": 12, "keys": [7, 8], "door": 9, "walls": [4], "verb": "navigate"}'
        assert _sci(bad).infer(_frame()) is None

    def test_hallucinated_walls_rejected(self):
        # walls=[99] — absent from _frame() → hallucination → None
        bad = (
            '{"avatar": 12, "keys": [0], "door": 9, "walls": [99], "verb": "navigate"}'
        )
        assert _sci(bad).infer(_frame()) is None

    def test_partial_multi_index_key_accepted(self):
        # keys=[0,1]: only 0 is on _frame() (1 is absent) → any-present rule accepts it
        reply = '{"avatar": 12, "keys": [0, 1], "door": 9, "walls": [4], "verb": "navigate"}'
        sem = _sci(reply).infer(_frame())
        assert sem is not None
        assert sem.keys == (0, 1)


class _FakeSci:
    def __init__(self, sem):
        self.sem = sem
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        return self.sem


def _obs(levels=0, actions=(1, 2, 3, 4)):
    g = np.full((10, 10), 3, dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 4
    g[2, 2] = 12
    g[5, 5] = 0
    g[7, 7] = 9
    return {
        "frame": [g.tolist()],
        "available_actions": list(actions),
        "levels_completed": levels,
        "state": "NOT_FINISHED",
    }


class TestScientistAgent:
    def test_emits_legal_action_from_scientist_semantics(self):
        sci = _FakeSci(LS20_DEFAULT)
        act = ScientistPlannerAgent(scientist=sci).act(_obs())
        assert isinstance(act, ArcAction) and act.id in (1, 2, 3, 4)

    def test_scientist_queried_once_per_level(self):
        sci = _FakeSci(LS20_DEFAULT)
        a = ScientistPlannerAgent(scientist=sci)
        for _ in range(5):
            a.act(_obs(levels=0))
        assert sci.calls == 1  # cached after first query of the level

    def test_requeries_on_new_level(self):
        sci = _FakeSci(LS20_DEFAULT)
        a = ScientistPlannerAgent(scientist=sci)
        a.act(_obs(levels=0))
        a.act(_obs(levels=1))
        assert sci.calls == 2

    def test_none_from_scientist_falls_back_to_ls20(self):
        sci = _FakeSci(None)  # infer returns None -> degrade
        act = ScientistPlannerAgent(scientist=sci).act(_obs())
        assert act.id in (1, 2, 3, 4)  # still plays via LS20_DEFAULT
