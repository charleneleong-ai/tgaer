# ruff: noqa: F821  # runs in the dfranzen M2 notebook, after the harness is importable
# --- turn archive: let the model search its own turns after they are trimmed from context ---
# The board history already survives trimming (`history`, `transitions`); the model's own reasoning, code and tool
# output do not. Archive each assistant/tool message when it leaves the context, tagged with the level it was written
# on, and expose `search_turns()` in the Python tool. Retrieval only - nothing is re-injected into the prompt.
import re as _ta_re
import threading as _ta_threading
import time as _ta_time

import inference.agent.python_tool_sandbox as _ta_sandbox
import inference.agent.tool_agent as _ta_mod

TURN_ARCHIVE = {
    "max_entries": 600,
    "send_entries": 300,
    "send_chars": 1200,
    "archived": 0,
    "searches": 0,
    "python_calls": 0,
}
_ta_local = _ta_threading.local()
_TA_PROMPT = (
    "\n- `search_turns(query, level='current', last=5)` searches your own earlier turns that were trimmed from the "
    "context (your reasoning, code and tool output) and returns short excerpts. They record what you thought, not "
    "what is true: check a claim against `history` before relying on it. `level=None` searches every level.\n"
)
_TA_SEARCH = '''
_turn_archive = list((initial.get("state") or {}).get("turn_archive") or [])


def search_turns(query="", level="current", last=5, width=400):
    """Excerpts of your own trimmed turns containing `query` (case-insensitive), newest last."""
    if level == "current":
        level = getattr(runtime_globals.get("current_frame"), "level", None)
    needle = str(query).lower()
    hits = [e for e in _turn_archive
            if (level is None or e.get("level") == level) and needle in str(e.get("text", "")).lower()]
    out = []
    for e in hits[-max(1, int(last)):]:
        text = str(e.get("text", ""))
        start = max(0, text.lower().find(needle) - int(width) // 2) if needle else 0
        out.append(f"[unverified: your turn {e.get('n')}, level {e.get('level')}, {e.get('role')}] "
                   + text[start:start + int(width)])
    return out


runtime_globals["search_turns"] = search_turns
'''


def _ta_text(message: dict) -> str:
    content, parts = message.get("content"), []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        parts += [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
    parts += [
        message[k]
        for k in ("reasoning", "reasoning_content")
        if isinstance(message.get(k), str)
    ]
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        if fn.get("arguments"):
            parts.append(f"[{fn.get('name', 'tool')}] {fn['arguments']}")
    return "\n".join(p for p in parts if p).strip()


ROLES = ("assistant", "tool")


def _ta_store(agent) -> dict:
    run_dir = getattr(agent, "_session_runtime_dir", None)
    store = agent.__dict__.get("_ours_turn_archive")
    if store is None or store["run_dir"] != run_dir:  # a new game starts a new archive
        store = agent.__dict__["_ours_turn_archive"] = {
            "run_dir": run_dir,
            "entries": [],
            "seen": set(),  # digests of archived turns
            "level": {},  # digest -> level it was written on, for turns still in context
            "text": {},  # id(message) -> (message, role, text), so each message is rendered once
        }
    return store


def _ta_capture(agent, before: list, kept: list) -> None:
    store, level, cache = _ta_store(agent), getattr(agent, "_ours_level", None), {}

    def render(message: dict) -> tuple:
        hit = store["text"].get(id(message))
        if hit is None or hit[0] is not message:
            hit = (message, str(message.get("role", "")), _ta_text(message))
        cache[id(message)] = hit
        return hit[1:]

    kept_digests = {hash(render(m)) for m in kept if m.get("role") in ROLES}
    for message in before:
        if message.get("role") not in ROLES:
            continue
        role, text = render(message)
        digest = hash((role, text))
        store["level"].setdefault(digest, level)
        if not text or digest in kept_digests or digest in store["seen"]:
            continue
        store["seen"].add(digest)
        store["entries"].append(
            {
                "n": len(store["seen"]),
                "level": store["level"].pop(digest),
                "role": role,
                "text": text[: TURN_ARCHIVE["send_chars"]],
            }
        )
        TURN_ARCHIVE["archived"] += 1
    store["text"] = {k: v for k, v in cache.items() if k in {id(m) for m in kept}}
    del store["entries"][: -TURN_ARCHIVE["max_entries"]]


_ta_cls = _ta_mod.ToolAgent
if not getattr(_ta_cls._persistent_history_messages, "_ours_ta", False):
    _ta_orig_history, _ta_orig_python = (
        _ta_cls._persistent_history_messages,
        _ta_cls._run_python_tool,
    )
    _ta_orig_sandbox, _ta_orig_prompt = (
        _ta_mod.run_sandboxed_python,
        _ta_mod._build_system_prompt,
    )

    def _persistent_history_messages(self, messages, *args, **kwargs):
        kept = _ta_orig_history(self, messages, *args, **kwargs)
        try:
            _ta_capture(self, messages, kept or [])
        except Exception as exc:
            print(f"#OURS_ARCHIVE capture failed: {exc!r}"[:300], flush=True)
        return kept

    def _run_python_tool(self, state_path, arguments, *args, **kwargs):
        _ta_local.agent = self
        _ta_local.searching = "search_turns(" in str((arguments or {}).get("code", ""))
        TURN_ARCHIVE["python_calls"] += 1
        TURN_ARCHIVE["searches"] += _ta_local.searching
        try:
            return _ta_orig_python(self, state_path, arguments, *args, **kwargs)
        finally:
            _ta_local.agent = None

    def run_sandboxed_python(*args, initial_state=None, **kwargs):
        agent = getattr(_ta_local, "agent", None)
        if agent is not None and isinstance(initial_state, dict):
            agent._ours_level = (initial_state.get("current_frame") or {}).get("level")
            if getattr(
                _ta_local, "searching", False
            ):  # only snippets that search pay for the payload
                entries = _ta_store(agent)["entries"][-TURN_ARCHIVE["send_entries"] :]
                initial_state = {**initial_state, "turn_archive": entries}
        return _ta_orig_sandbox(*args, initial_state=initial_state, **kwargs)

    def _build_system_prompt(*args, **kwargs):
        return _ta_orig_prompt(*args, **kwargs) + _TA_PROMPT

    _anchor = _ta_re.search(
        r'^( *)_refresh_state\(initial\.get\("state"\) or \{\}\)\n',
        _ta_sandbox._SANDBOX_BOOTSTRAP,
        _ta_re.M,
    )
    if _anchor is None:
        raise RuntimeError("turn archive: sandbox anchor not found")
    _indent = _anchor.group(1)
    _ta_sandbox._SANDBOX_BOOTSTRAP = (
        _ta_sandbox._SANDBOX_BOOTSTRAP[: _anchor.end()]
        + "".join(
            _indent + line if line.strip() else line
            for line in _TA_SEARCH.splitlines(keepends=True)
        )
        + _ta_sandbox._SANDBOX_BOOTSTRAP[_anchor.end() :]
    )
    _persistent_history_messages._ours_ta = True
    _ta_cls._persistent_history_messages, _ta_cls._run_python_tool = (
        _persistent_history_messages,
        _run_python_tool,
    )
    _ta_mod.run_sandboxed_python, _ta_mod._build_system_prompt = (
        run_sandboxed_python,
        _build_system_prompt,
    )


def _ta_report() -> None:
    while True:
        _ta_time.sleep(1800)
        print(
            "#OURS_ARCHIVE " + " ".join(f"{k}={v}" for k, v in TURN_ARCHIVE.items()),
            flush=True,
        )


_ta_threading.Thread(target=_ta_report, name="ours-archive-report", daemon=True).start()
print("#OURS_ARCHIVE armed", flush=True)
