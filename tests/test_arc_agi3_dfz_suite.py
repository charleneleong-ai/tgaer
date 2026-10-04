"""dfranzen stuck-level suite tools; data-backed tests need DFZ_SRC, DFZ_WORKDIR and DFZ_ENV_FILES."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import arc_agi
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sia-oss/bench/dfz"))

import candidates  # noqa: E402
import gepa_score  # noqa: E402
import harness  # noqa: E402
import kernels  # noqa: E402
from messages import MARK, Messages  # noqa: E402
from replay import RecordedGame  # noqa: E402
from warm_start import WarmStart  # noqa: E402

COLOURS = [
    "white",
    "gray",
    "grey",
    "charcoal",
    "black",
    "magenta",
    "pink",
    "red",
    "blue",
    "sky",
    "yellow",
    "orange",
    "green",
    "purple",
]
SPLIT = json.loads((REPO / "sia-oss/bench/dfz/split.json").read_text())
PATCHED = (
    "_ensure_session",
    "_build_user_message",
    "_build_user_prompt",
    "_trim_messages_for_context",
)


def located(var: str) -> Path | None:
    value = os.environ.get(var)
    return Path(value) if value and Path(value).exists() else None


SRC = located("DFZ_SRC")
WORKDIR = located("DFZ_WORKDIR")
if SRC is not None:
    harness.Harness.install(SRC)


@pytest.fixture(scope="module")
def workdir() -> harness.Workdir:
    if WORKDIR is None:
        pytest.skip("DFZ_WORKDIR run outputs not available")
    return harness.Workdir(WORKDIR)


@pytest.fixture(scope="module")
def ta() -> ModuleType:
    if SRC is None:
        pytest.skip("DFZ_SRC dfranzen harness not available")
    import inference.agent.tool_agent as tool_agent  # importable only after Harness.install

    return tool_agent


@pytest.fixture(scope="module")
def env_files() -> Path:
    path = located("DFZ_ENV_FILES")
    if path is None or not any(path.iterdir()):
        pytest.skip("DFZ_ENV_FILES offline games not available")
    return path


@pytest.fixture(scope="module")
def taaf() -> SimpleNamespace:
    game_api = pytest.importorskip("taaf.game_api", exc_type=ImportError)
    return SimpleNamespace(
        game_api=game_api,
        RunSession=pytest.importorskip("taaf.game", exc_type=ImportError).RunSession,
    )


@pytest.fixture(scope="module")
def warm_records(workdir: harness.Workdir) -> dict[str, dict[str, Any]]:
    records = {
        p.stem: json.loads(p.read_text()) for p in workdir.warm_data().glob("*-L*.json")
    }
    if not records:
        pytest.skip("warmdata/ not built")
    return records


@pytest.fixture
def warm_agent(ta: ModuleType, warm_records: dict[str, dict[str, Any]]) -> type:
    seeded = dict(
        warm_records,
        **{"m0r0-L3": dict(warm_records["m0r0-L3"], win_records={"1": "a", "2": "b"})},
    )
    cls = type("WarmAgent", (ta.ToolAgent,), {})
    WarmStart.install(cls, seeded)
    return cls


@pytest.fixture
def hook(ta: ModuleType) -> Any:
    from win_records import WinRecords  # importable only after Harness.install

    return WinRecords(ta._render_auto_frame_diff, ta._AUTO_FRAME_DIFF_MAX_GROUP)


@pytest.fixture
def hooked_agent(hook: Any, ta: ModuleType) -> type:
    cls = type("HookedAgent", (ta.ToolAgent,), {})
    hook.install(cls)
    return cls


@pytest.fixture(scope="module")
def tr87(workdir: harness.Workdir, ta: ModuleType) -> list[Any]:
    from win_records import WinRecords

    return WinRecords.history_from_events(workdir.eval25.events("tr87"))


def state_path(tmp_path: Path, episode: str) -> Path:
    return tmp_path / f"{episode}_p0_tool_runtime_state.json"


def tool_call(code: str) -> dict[str, Any]:
    return {"function": {"arguments": json.dumps({"code": code})}}


def opener_kwargs(entries: list[Any], *, transition: bool) -> dict[str, Any]:
    summary = {
        "level_transition": transition,
        "level": entries[-1].frame.level,
        "executed_actions": ["UP"],
    }
    return {
        "valid_actions": ["UP"],
        "current_frame": entries[-1].frame,
        "history_entries": entries,
        "previous_step_summary": summary,
    }


def chat(n: int) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs += [
            {"role": "user", "content": f"turn {i} " + "x" * 4000},
            {"role": "assistant", "content": f"reply {i} " + "y" * 4000},
        ]
    return msgs


def summary(levels: dict[str, float]) -> str:
    return "".join(
        f"  {ep}: score=0.00, levels={lv}/8, actions=1, tokens=1\n"
        for ep, lv in levels.items()
    )


class TestReplay:
    """Recorded actions reproduce the recorded boards and give a JSON-safe level prefix."""

    def test_the_recorded_boards_are_reproduced(
        self, workdir: harness.Workdir, env_files: Path
    ) -> None:
        r = RecordedGame(workdir.eval25.events("m0r0"), env_files).replay(3)
        assert (
            r["mismatches"] == [] and r["levels_completed"] == 2 and r["replayed"] > 0
        )

    def test_the_prefix_survives_a_json_round_trip(
        self, workdir: harness.Workdir
    ) -> None:
        prefix = RecordedGame(workdir.eval25.events("ar25")).prefix(7)
        assert [tuple(p) for p in json.loads(json.dumps(prefix))] == prefix and any(
            d for _, d in prefix
        )

    @pytest.mark.parametrize("eid", ["m0r0-L3r1", "re86-L8r0"])
    def test_episode_ids_round_trip(self, eid: str) -> None:
        parsed = harness.parse_episode_id(eid)
        assert parsed is not None and harness.episode_id(*parsed) == eid
        assert harness.parse_episode_id("ar25-0c556536") is None


class TestLevelStart:
    """A LevelStartGameAPI episode begins on the recorded start of its stuck level."""

    @pytest.mark.parametrize(
        ("code", "level"),
        [("tr87", 6), ("m0r0", 3), ("sk48", 1), ("vc33", 7), ("su15", 5)],
    )
    def test_the_game_starts_on_the_recorded_level_start(
        self,
        workdir: harness.Workdir,
        env_files: Path,
        taaf: SimpleNamespace,
        code: str,
        level: int,
    ) -> None:
        from level_start import LevelStartGameAPI  # needs taaf

        game = RecordedGame(workdir.eval25.events(code), env_files)
        spec = taaf.game_api.ArcadeSpec(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=str(env_files),
        )
        api = LevelStartGameAPI(
            env_name=game.game_id, arcade_spec=spec, prefix=tuple(game.prefix(level))
        )
        state = api._start_game(taaf.RunSession())
        assert state.raw.levels_completed == level - 1
        assert RecordedGame.board(state.raw) == game.start_board(level)


class TestWarmStart:
    """Warm records rebuild the retained functions and seed each suite episode once."""

    @pytest.mark.parametrize(
        ("bodies", "kept"),
        [
            (
                ["Retained your function f(x)", "Retained your function g(x)"],
                {"f", "g"},
            ),
            (
                [
                    "Retained your function f(x) and g(",
                    "Your function g was not retained",
                ],
                {"f"},
            ),
            (
                [
                    "Retained your function f(x)",
                    "retained functions were cleared",
                    "Retained your function h(",
                ],
                {"h"},
            ),
        ],
    )
    def test_kept_functions_follow_the_tool_results(
        self, bodies: list[str], kept: set[str]
    ) -> None:
        defs = (
            "def f(x):\n    return x\n\ndef g(x):\n    return x\n\ndef h():\n    pass\n"
        )
        history = [
            {"role": "assistant", "tool_calls": [tool_call(defs), {"function": {}}]}
        ]
        history += [
            {"role": "tool", "content": [{"type": "text", "text": b}]} for b in bodies
        ]
        assert set(WarmStart.kept_functions(history)) == kept

    def test_a_suite_episode_is_seeded_once(
        self, warm_agent: type, warm_records: dict[str, Any], tmp_path: Path
    ) -> None:
        agent = warm_agent()
        agent._ensure_session(state_path(tmp_path, "m0r0-L3r1"))
        assert agent._history_messages == warm_records["m0r0-L3"]["history"]
        assert agent._kept_functions == warm_records["m0r0-L3"]["kept"]
        assert agent._win_records == {1: "a", 2: "b"}
        agent._history_messages.append({"role": "user", "content": "later turn"})
        agent._ensure_session(state_path(tmp_path, "m0r0-L3r1"))
        assert agent._history_messages[-1]["content"] == "later turn"

    def test_the_first_opener_is_the_recorded_one_then_normal(
        self, warm_agent: type, warm_records: dict[str, Any], tmp_path: Path
    ) -> None:
        agent = warm_agent()
        agent._ensure_session(state_path(tmp_path, "tr87-L6r0"))
        opener = warm_records["tr87-L6"]["opener"]
        assert agent._build_user_message("ignored", None) == opener
        assert (
            agent._build_user_message("fresh prompt", None)["content"]
            != opener["content"]
        )

    @pytest.mark.parametrize("episode", ["ft09-L2r0", "ar25-0c556536"])
    def test_an_unknown_episode_is_left_cold(
        self, warm_agent: type, tmp_path: Path, episode: str
    ) -> None:
        agent = warm_agent()
        agent._ensure_session(state_path(tmp_path, episode))
        assert agent._history_messages == [] and "_warm_opener" not in agent.__dict__

    @pytest.mark.parametrize("code", ["tr87-L6", "m0r0-L3", "vc33-L7"])
    def test_rebuilt_functions_are_valid_python(
        self, warm_records: dict[str, Any], code: str
    ) -> None:
        for src in warm_records[code]["kept"].values():
            compile(src, code, "exec")


class TestWinRecords:
    """Win records come from the real tr87 trajectory and stay pinned through history trimming."""

    @pytest.mark.parametrize(
        "content",
        [
            "hi",
            [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "u"}},
            ],
        ],
    )
    def test_re_pinning_replaces_the_previous_block(
        self, content: str | list[dict[str, Any]]
    ) -> None:
        pinned = Messages.pin(
            Messages.pin({"role": "user", "content": content}, {1: "a"}),
            {1: "a", 2: "b"},
        )
        text = json.dumps(pinned["content"])
        assert text.count(MARK) == 1 and "\\nb\\n" in text and "hi" in text
        if isinstance(content, list):
            assert [p["type"] for p in pinned["content"]] == [
                "text",
                "text",
                "image_url",
            ]
        else:
            assert Messages.strip(pinned["content"]) == "hi"

    @pytest.mark.parametrize(
        ("level", "actions"), [(1, 37), (2, 30), (5, 158), (6, None)]
    )
    def test_counts_match_the_logged_per_level_actions(
        self, hook: Any, tr87: list[Any], level: int, actions: int | None
    ) -> None:
        record = hook.record(tr87, level)
        if actions is None:
            assert record is None
            return
        assert (
            record.startswith(f"Level {level}: cleared by ")
            and f"after {actions} actions" in record
        )
        assert len(record) < 4000 and record.count("\n") >= 3

    def test_a_level_transition_appends_its_record_once(
        self, hooked_agent: type, tr87: list[Any]
    ) -> None:
        agent = hooked_agent()
        entries = tr87[: next(i for i, e in enumerate(tr87) if e.frame.level == 2) + 1]
        first = agent._build_user_prompt(
            len(entries), **opener_kwargs(entries, transition=True)
        )
        again = agent._build_user_prompt(
            len(entries), **opener_kwargs(entries, transition=True)
        )
        assert first.count(MARK) == 1 and "Level 1: cleared by" in first
        assert MARK not in again and list(agent._win_records) == [1]

    def test_an_ordinary_turn_is_untouched(
        self, hooked_agent: type, tr87: list[Any]
    ) -> None:
        assert MARK not in hooked_agent()._build_user_prompt(
            10, **opener_kwargs(tr87[:10], transition=False)
        )

    def test_eviction_pins_a_single_block_at_the_head(self, hooked_agent: type) -> None:
        agent = hooked_agent()
        agent._context_budget_tokens = 12000
        agent._win_records = {1: "Level 1: cleared by 'UP'."}
        out = agent._trim_messages_for_context(chat(20))
        assert len(out) < 41 and out[1]["role"] == "user"
        assert out[1]["content"].startswith(MARK) and "turn " in out[1]["content"]
        out = agent._trim_messages_for_context([*out, *chat(6)[1:]])
        assert sum(str(m.get("content")).count(MARK) for m in out) == 1

    def test_without_records_trimming_is_unchanged(
        self, hooked_agent: type, ta: ModuleType
    ) -> None:
        hooked, stock = hooked_agent(), ta.ToolAgent()
        hooked._context_budget_tokens = stock._context_budget_tokens = 12000
        assert hooked._trim_messages_for_context(
            chat(20)
        ) == stock._trim_messages_for_context(chat(20))


class TestCandidates:
    """Prompt candidates stay game-agnostic and are scored on a fixed tune/holdout split."""

    @pytest.mark.parametrize("name", list(candidates.CANDIDATES))
    def test_a_candidate_names_no_game_or_colour(self, name: str) -> None:
        text = candidates.CANDIDATES[name].lower()
        games = {key.split("-L")[0] for key in SPLIT["tune"] + SPLIT["holdout"]}
        assert not [g for g in games if g in text]
        assert not [c for c in COLOURS if re.search(rf"\b{c}\b", text)]

    def test_the_split_keeps_tune_and_holdout_disjoint(self) -> None:
        assert not set(SPLIT["tune"]) & set(SPLIT["holdout"])
        assert len(SPLIT["tune"]) + len(SPLIT["holdout"]) == 20

    def test_install_appends_once_to_every_new_agent(
        self, ta: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stock = ta.ToolAgent()._system_prompt
        monkeypatch.setattr(ta, "_build_system_prompt", ta._build_system_prompt)
        candidates.install(ta, candidates.CANDIDATES["c1_protocol"])
        a, b = ta.ToolAgent(), ta.ToolAgent()
        assert (
            a._system_prompt
            == stock + candidates.CANDIDATES["c1_protocol"]
            == b._system_prompt
        )

    def test_a_level_clears_only_when_every_repeat_does(self) -> None:
        base = gepa_score.SuiteScore.clears(
            summary({"ar25-L7r0": 7.0, "ar25-L7r1": 6.0, "g50t-L1r0": 1.0})
        )
        cand = gepa_score.SuiteScore.clears(
            summary({"ar25-L7r0": 7.0, "ar25-L7r1": 7.0, "g50t-L1r0": 1.0})
        )
        score = gepa_score.SuiteScore(cand, base)
        line = score.line("tune", ["ar25-L7", "g50t-L1", "sk48-L1"])
        assert "both-repeat clears 0 -> 1" in line and "episodes 2/6 -> 3/6" in line
        assert "gained ['ar25-L7'] lost []" in line
        assert "3/40 vs baseline mean 2.0/40 (gain +1.0" in score.pooled_line(base, 20)


class TestKernels:
    """Suite kernels are rebuilt from the base notebook and their hooks run against the real harness."""

    @pytest.mark.parametrize(
        ("features", "slug", "dataset"),
        [
            (kernels.Features(), "arc-agi-3-dfranzen-m2-stallsuite", None),
            (
                kernels.Features(warm=True),
                "arc-agi-3-dfranzen-m2-stallsuite-warm",
                kernels.WARM_DATASET,
            ),
            (
                kernels.Features(winrec=True),
                "arc-agi-3-dfranzen-m2-stallsuite-warm-winrec",
                kernels.WINREC_DATASET,
            ),
            (
                kernels.Features(candidate="c2_protocol_ledger"),
                "arc-agi-3-dfz-warm-c2-protocol-ledger",
                kernels.WARM_DATASET,
            ),
        ],
    )
    def test_features_name_the_kernel_and_its_dataset(
        self, features: kernels.Features, slug: str, dataset: str | None
    ) -> None:
        assert (features.slug, features.dataset) == (slug, dataset) and len(
            slug
        ) <= kernels.MAX_SLUG

    @staticmethod
    def isolate(
        monkeypatch: pytest.MonkeyPatch, ta: ModuleType, modules: list[str]
    ) -> None:
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(ta, "_build_system_prompt", ta._build_system_prompt)
        for attr in PATCHED:
            monkeypatch.setattr(ta.ToolAgent, attr, getattr(ta.ToolAgent, attr))
        for name in modules:  # force the hook to import the files the notebook wrote
            monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
            del sys.modules[name]

    @staticmethod
    def emulate_writefile(cells: list[str], dfz_dir: Path) -> None:
        exec(next(c for c in cells if "os.makedirs" in c), {})
        for src in cells:
            if src.startswith(f"%%writefile {dfz_dir}/"):
                head, body = src.split("\n", 1)
                Path(head.split()[1]).write_text(body)

    def test_the_built_kernel_runs_its_hooks(
        self,
        workdir: harness.Workdir,
        ta: ModuleType,
        env_files: Path,
        taaf: SimpleNamespace,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        features = kernels.Features(winrec=True, candidate="c1_protocol")
        datasets = tmp_path / "datasets"
        (datasets / harness.KAGGLE_USER).mkdir(parents=True)
        (datasets / features.dataset).symlink_to(workdir.warm_data(winrec=True))
        nb = kernels.Kernel.build(
            workdir, features, str(tmp_path / "dfz"), str(datasets)
        ).notebook
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.isolate(monkeypatch, ta, features.modules)
        self.emulate_writefile(cells, tmp_path / "dfz")
        hook = next(c for c in cells if kernels.HOOK_ANCHOR in c)
        bm = SimpleNamespace(solver=SimpleNamespace(), games=None)
        ns: dict[str, Any] = {
            "bm": bm,
            "TRUE_SUBMISSION": False,
            "sys": sys,
            "json": json,
            "Path": Path,
        }
        ns |= {"arc_agi": arc_agi, "competition_env_files": str(env_files)}
        exec(hook[hook.index(kernels.SUITE_START) :], ns)
        games = next(c for c in cells if kernels.GAMES_ANCHOR in c)
        exec(
            "if True:\n"
            + games[games.index(kernels.SUITE_GAMES_START) :].split("\n\n")[0],
            ns,
        )

        n_levels = len(workdir.stuck_levels())
        assert len(bm.games) == harness.STALL_REPEATS * n_levels
        for g in bm.games:
            _, level, _ = harness.parse_episode_id(g.external_game_id)
            assert g._start_game(taaf.RunSession()).raw.levels_completed == level - 1
        agent = ta.ToolAgent()
        agent._ensure_session(state_path(tmp_path, "m0r0-L3r0"))
        assert agent._history_messages and agent._win_records
        assert agent._system_prompt.endswith(candidates.CANDIDATES["c1_protocol"])
        assert "WinRecords" in ta.ToolAgent._trim_messages_for_context.__qualname__
        shipped = {
            Path(sys.modules[m].__file__).resolve().parent for m in features.modules
        }
        assert shipped == {(tmp_path / "dfz").resolve()}
