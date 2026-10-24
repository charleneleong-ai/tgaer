# ruff: noqa: F821  # runs in the dfranzen M2 notebook after the launcher cell defines NOTEBOOK_START_TIME, SERVED_MODEL_*, TRUE_SUBMISSION
# --- keep-alive: never let the scorecard go 15 minutes without an action ---
# Kaggle closes the run's scorecard after 15 min without an action. Since 10-20 weight loads on scored machines have
# taken 35-63 min, longer than the single warm-up RESET plus the 15-min first-request grace cover, and the same close
# can fire mid-run (server restart, every stream in a long generation). A watchdog RESETs one live game, round robin,
# whenever no action has gone out for 5 min while the server loads or 10 min once it is up; each game's first model
# request may wait up to 90 min.
import os
import threading as _ka_threading
import time as _ka_time
import urllib.request as _ka_request

import inference.framework.solver as _ka_solver

os.environ["ARC3_HTTP_RETRY_INITIAL_SECONDS"] = str(90 * 60)
KEEPALIVE = {
    "poll_s": float(os.environ.get("OURS_KEEPALIVE_POLL_S", 60)),
    "idle_loading_s": float(os.environ.get("OURS_KEEPALIVE_IDLE_LOADING_S", 300)),
    "idle_running_s": float(os.environ.get("OURS_KEEPALIVE_IDLE_RUNNING_S", 600)),
    "sessions": [],
    "resets": 0,
    "last_action": _ka_time.time(),
    "lock": _ka_threading.Lock(),
}
_ka_cls = _ka_solver._HarnessGameSession


def _ka_healthy() -> bool:
    try:
        with _ka_request.urlopen(
            f"http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/health", timeout=5
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _ka_reset(session, why: str) -> None:
    try:
        session._execute_auto_reset()
        KEEPALIVE["resets"] += 1
        print(
            f"#OURS_KEEPALIVE {why} RESET {session.game.game_run.game_id} "
            f"at {int(_ka_time.time() - NOTEBOOK_START_TIME)}s",
            flush=True,
        )
    except Exception as exc:
        print(f"#OURS_KEEPALIVE {why} RESET failed: {exc!r}"[:300], flush=True)


if not getattr(_ka_cls._play_inner, "_ours_ka", False):
    _ka_orig_play_inner, _ka_orig_execute = _ka_cls._play_inner, _ka_cls._execute_action

    def _play_inner(self, *args, **kwargs):
        with KEEPALIVE["lock"]:
            KEEPALIVE["sessions"].append(self)
            first = len(KEEPALIVE["sessions"]) == 1
        if first and not TRUE_SUBMISSION:
            _ka_reset(
                self, "save-run self-test"
            )  # exercises the reset path once, where it costs nothing
        return _ka_orig_play_inner(self, *args, **kwargs)

    def _execute_action(self, *args, **kwargs):
        try:
            return _ka_orig_execute(self, *args, **kwargs)
        finally:
            KEEPALIVE["last_action"] = _ka_time.time()

    _play_inner._ours_ka = True
    _ka_cls._play_inner, _ka_cls._execute_action = _play_inner, _execute_action


def _ka_loop() -> None:
    turn = 0
    while True:
        _ka_time.sleep(KEEPALIVE["poll_s"])
        idle = _ka_time.time() - KEEPALIVE["last_action"]
        if idle < KEEPALIVE["idle_loading_s"]:
            continue
        healthy = _ka_healthy()
        if healthy and idle < KEEPALIVE["idle_running_s"]:
            continue
        with KEEPALIVE["lock"]:
            live = KEEPALIVE["sessions"] = [
                s for s in KEEPALIVE["sessions"] if not s.should_stop()
            ]
        if live:
            _ka_reset(
                live[turn % len(live)],
                "keep-alive" if healthy else "keep-alive (server loading)",
            )
            turn += 1


_ka_threading.Thread(target=_ka_loop, name="ours-keepalive", daemon=True).start()
print(
    f"#OURS_KEEPALIVE armed: healthy={_ka_healthy()} at {int(_ka_time.time() - NOTEBOOK_START_TIME)}s",
    flush=True,
)
