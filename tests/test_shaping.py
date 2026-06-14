from __future__ import annotations

from tgaer.envs.arc_agi3 import shaping

OBS1 = "game_id: m\nstate: NOT_FINISHED\nnum_actions: 1\ngrid:\n0 1\n2 3"
# same board, different counter — must fingerprint identically (grid only)
OBS1_LATER = "game_id: m\nstate: NOT_FINISHED\nnum_actions: 9\ngrid:\n0 1\n2 3"
OBS2 = "game_id: m\nstate: NOT_FINISHED\nnum_actions: 2\ngrid:\n0 1\n2 9"


class TestFingerprint:
    def test_extract_grid_takes_block_after_marker(self):
        assert shaping.extract_grid(OBS1) == "0 1\n2 3"

    def test_same_grid_diff_counter_same_fingerprint(self):
        assert shaping.fingerprint(OBS1) == shaping.fingerprint(OBS1_LATER)

    def test_different_grid_different_fingerprint(self):
        assert shaping.fingerprint(OBS1) != shaping.fingerprint(OBS2)


class TestAccumulate:
    def test_novel_and_stall_counts(self):
        st: dict = {}
        for obs in (OBS1, OBS2, OBS2):  # new, new, repeat
            shaping.observe(st, obs)
        assert st["shaping_steps"] == 3
        assert st["shaping_novel_steps"] == 2  # OBS1, OBS2 distinct
        assert st["shaping_stall_steps"] == 1  # the OBS2 repeat
        assert isinstance(st["shaping_seen"], list)  # json-safe, not a set

    def test_seen_is_serializable(self):
        st: dict = {}
        shaping.observe(st, OBS1)
        import json

        json.dumps(st)  # must not raise


class TestRewards:
    def test_win_reward(self):
        assert shaping.win_reward({"game_state": "WIN"}) == 1.0
        assert shaping.win_reward({"game_state": "GAME_OVER"}) == 0.0

    def test_novelty_fraction(self):
        st = {"shaping_steps": 4, "shaping_novel_steps": 3}
        assert shaping.novelty_reward(st) == 0.75

    def test_anti_stall_is_negative_fraction(self):
        st = {"shaping_steps": 4, "shaping_stall_steps": 2}
        assert shaping.anti_stall_penalty(st) == -0.5

    def test_rewards_safe_on_empty_state(self):
        assert shaping.novelty_reward({}) == 0.0
        assert shaping.anti_stall_penalty({}) == 0.0
