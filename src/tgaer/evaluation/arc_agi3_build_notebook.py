"""Build Kaggle notebook for ARC-AGI-3 submission with local LLM.

This script creates a notebook that:
1. Installs dependencies from offline dataset
2. Loads Qwen3.6-27B GGUF model
3. Runs the REPL agent against games

Usage:
    python src/tgaer/evaluation/arc_agi3_build_notebook.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent

REPO = Path(__file__).resolve().parents[3]
AGENT_SRC = REPO / "src" / "tgaer" / "agents" / "arc_agi3_kaggle.py"
# Modules copied into the kernel verbatim, keeping their package path so the
# imports in the source work unchanged. Copying the *text* rather than
# re-implementing it is the point: the explorer is under active development
# (its position-memory semantics changed in #19), and a hand-maintained second
# copy is the failure mode that produced three features which looked
# maintained and were never actually the code that ran.
KERNEL_PKG = "/tmp/kernel_pkg"
PORTED_MODULES = (
    "tgaer/core/agent_base.py",
    "tgaer/envs/arc_agi3/arc_agi3_api.py",
    "tgaer/agents/arc_agi3_grid.py",
    "tgaer/agents/arc_agi3_semantics.py",
    "tgaer/agents/arc_agi3_explorer.py",
)
# The notebook and its Kaggle metadata stay in the starter checkout: they are
# pushed by the Kaggle CLI from that directory, which also holds the offline
# wheelhouse and model snapshot the kernel needs.
ROOT = Path(
    os.environ.get("ARC_STARTER_ROOT", REPO / "vendor" / "ARC-AGI-3-Kaggle-Starter")
)
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"

# We had never actually run on a GPU: `kaggle quota` showed 0.00h of 30h used,
# ever. The kernel metadata carried "accelerator"/"enable_gpu", but the CLI reads
# neither — kaggle_api_extended.py:6450 takes the accelerator from `--accelerator`
# or the "machine_shape" key alone, and ours said "None". Hence no libcuda in
# every build, and llama_cpp failing to import.
#
# RTX Pro 6000 (Blackwell, 96GB) is what the competition offers and what the
# leading solutions request. It is a large step up from the P100 (compute 6.0,
# 16GB) the reruns used: the P100 is too old for vLLM, which requires >= 7.0.
ACCELERATOR = "rtx6000"

# "vllm"     - the Duck harness stack: vLLM serving Qwen3.6-27B-FP8, batching
#              across all game threads. Requires the datasets and pinned docker
#              image in notebooks/kernel-metadata.json.
# "llamacpp" - the previous stack: llama-cpp-python + a 14B GGUF, one model per
#              pool slot. Validated end to end as kernel v38.
BACKEND = os.environ.get("ARC_BACKEND", "vllm")

# Values accepted by the API (kagglesdk kernels_api_service.py). Exact casing
# matters: "nvidiaTeslaP100" is silently not a valid machine shape.
ACCELERATORS = {
    "cpu": {"machine_shape": "None", "gpu": False},
    "t4": {"machine_shape": "NvidiaTeslaT4", "gpu": True},
    "p100": {"machine_shape": "NvidiaTeslaP100", "gpu": True},
    "a100": {"machine_shape": "NvidiaTeslaA100", "gpu": True},
    "l4": {"machine_shape": "NvidiaL4", "gpu": True},
    "rtx6000": {"machine_shape": "NvidiaRtxPro6000", "gpu": True},
}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        # Cell bodies are plain (non-f) strings — they are full of braces that
        # f-string interpolation would eat — so a placeholder is the way to get
        # a build-time constant into one.
        "source": source.replace("__KERNEL_PKG__", KERNEL_PKG),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    # Cell 1: Install dependencies
    install_cell = code_cell(
        dedent("""\
        import os, glob

        # Debug: list all mounted input directories
        print("=== /kaggle/input contents ===")
        for item in sorted(os.listdir("/kaggle/input")):
            full = f"/kaggle/input/{item}"
            if os.path.isdir(full):
                files = glob.glob(f"{full}/**/*", recursive=True)
                print(f"  {item}/ ({len(files)} files)")
                for f in files[:10]:
                    print(f"    {os.path.relpath(f, full)}")
            else:
                print(f"  {item}")
        print("==============================")

        # Install arc-agi from offline competition dataset.
        # Single line (no shell continuation): a trailing "\" in a Python
        # triple-quote would be eaten by line-joining, and "\\" produces a
        # literal backslash that breaks --find-links in the shell.
        !pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv

        # Install llama-cpp-python from local dataset (pre-built CUDA wheel)
        wheel_dir = "/kaggle/input/llama-cpp-python-cuda-wheels"
        wheels = glob.glob(f"{wheel_dir}/*.whl")
        print(f"Found wheels: {wheels}")
        for w in wheels:
            !pip install "{w}" --no-deps

        print(f"CUDA available: {os.path.exists('/usr/local/cuda')}")
        """)
    )

    # Cell2: Download and load model
    load_model_cell = code_cell(
        dedent("""\
        import os
        import shutil
        import subprocess
        from pathlib import Path

        # Model path from our dataset (mounted at /kaggle/input/<dataset-slug>/)
        MODEL_FILE = "qwen3-14b.Q4_K_M.gguf"
        MODEL_PATH = Path("/kaggle/input/qwen3-14b") / MODEL_FILE

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        # Copy to /tmp for faster loading (1.2TB available)
        TMP_MODEL = Path("/tmp") / MODEL_FILE
        if not TMP_MODEL.exists():
            print(f"Copying model to {TMP_MODEL}...")
            shutil.copy2(MODEL_PATH, TMP_MODEL)
            print("Done.")

        # Set environment for our agent
        os.environ["LLAMA_MODEL_PATH"] = str(TMP_MODEL)
        os.environ["LLAMA_N_CTX"] = "8192"
        os.environ["LLAMA_N_GPU_LAYERS"] = "99"

        # Size the model pool to the GPU we actually get. The rerun runs ~110
        # games as concurrent threads through this one process, so a single
        # instance serializes the entire run; batch-1 decoding leaves the GPU
        # mostly idle, and extra instances convert that into throughput.
        # 16GB P100 -> 1 instance; 96GB RTX PRO 6000 -> ~8.
        MODEL_GB = Path(os.environ["LLAMA_MODEL_PATH"]).stat().st_size / 1e9
        KV_GB = 1.4  # measured: 1280 MiB of KV cache at n_ctx=8192
        RESERVE_GB = 6.0  # compute buffers, fragmentation, and the gateway
        POOL_CAP = 8  # past this, prefill contention outweighs the parallelism

        free_gb = 0.0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            free_gb = int(out.stdout.split()[0]) / 1024
        except Exception as exc:
            print(f"Could not read VRAM ({exc!r}); defaulting to a single model")

        pool = int((free_gb - RESERVE_GB) // (MODEL_GB + KV_GB)) if free_gb else 1
        pool = max(1, min(POOL_CAP, pool))
        os.environ["LLAMA_POOL_SIZE"] = str(pool)

        print(f"Model path: {os.environ['LLAMA_MODEL_PATH']} ({MODEL_GB:.2f} GB)")
        print(f"Free VRAM: {free_gb:.1f} GB -> LLAMA_POOL_SIZE={pool}")
        """)
    )

    # Cell3: Write agent code
    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    # Cell3b: stage the ported tgaer modules as a real package. %%writefile
    # cannot create directories and refuses to write into one that does not
    # exist, so the tree and its __init__.py files are made first, and the
    # module bodies follow one magic cell each — the magic copies the rest of
    # the cell verbatim, so no escaping of docstrings or quotes is involved.
    pkg_dirs = sorted({str(Path(m).parent) for m in PORTED_MODULES})
    stage_pkg_cell = code_cell(
        dedent(f"""\
        import sys
        from pathlib import Path

        PKG = Path({KERNEL_PKG!r})
        for rel in {pkg_dirs!r}:
            d = PKG / rel
            d.mkdir(parents=True, exist_ok=True)
        # Every level needs an __init__.py, including the intermediate ones.
        for rel in {pkg_dirs!r}:
            part = PKG
            for name in Path(rel).parts:
                part = part / name
                (part / "__init__.py").touch()
        if str(PKG) not in sys.path:
            sys.path.insert(0, str(PKG))
        print(f"[pkg] staged {{PKG}} on sys.path")
        """)
    )
    module_cells = [
        code_cell(
            f"%%writefile {KERNEL_PKG}/{rel}\n" + (REPO / "src" / rel).read_text()
        )
        for rel in PORTED_MODULES
    ]

    # Cell4: preflight (commit mode only). Skipped during the rerun, where
    # main.py loads the model in a subprocess — a second in-kernel copy would
    # hold ~9GB and OOM the P100 (2x9GB > 16GB).
    #
    # Two tiers, because commit builds are CPU-only (see ACCELERATOR above):
    #   - agent checks run everywhere and are STRICT. They catch the failure
    #     mode that actually costs slots: an agent that no longer imports,
    #     parses, or offers the tools it is supposed to.
    #   - the inference check needs a GPU. It is skipped loudly when libcuda is
    #     absent, rather than failing a build that could never have passed.
    # Anything unexpected still raises: a build that cannot validate the agent
    # must not reach a submission, since the daemon only submits on COMPLETE.
    preflight_cell = code_cell(
        dedent("""\
        import ctypes
        import json
        import os
        import random
        import sys

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("Competition rerun: skipping in-kernel preflight (main.py loads the model).")
        else:
            sys.path.insert(0, "/tmp")  # cell 3 wrote the agent here
            import my_agent

            # The ported modules must import here, not merely have been written.
            # A staged package that cannot be imported in the kernel is the same
            # failure as a feature that never fires: it looks present and does
            # nothing. numpy in particular is assumed rather than installed, and
            # this project has already had a dependency it "obviously had" turn
            # out to be broken in the kernel.
            sys.path.insert(0, "__KERNEL_PKG__")
            import numpy as _np
            from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
            from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics
            assert EmpiricalSemantics().move_lattice() == {}, "cold lattice must be empty"
            print(f"Preflight: ported modules OK — numpy {_np.__version__}, "
                  f"{ExplorerArcAgi3Agent.__name__} importable")

            n_ctx = int(os.environ["LLAMA_N_CTX"])
            available = [1, 2, 3, 4, 5, 6]

            # --- agent checks: no GPU needed, so these are unconditional ---
            tools = my_agent.MyAgent._build_tools(available)
            names = [t["function"]["name"] for t in tools]
            assert "MOUSE" in names, f"MOUSE tool missing from schema: {names}"
            mouse = next(t for t in tools if t["function"]["name"] == "MOUSE")
            # Coordinates must be required; the optional mechanic note may or may
            # not be present depending on ARC_MECHANIC_NOTES, so assert on what
            # the click actually needs rather than on the exact property set.
            assert mouse["function"]["parameters"]["required"] == ["x", "y"], mouse
            assert {"x", "y"} <= set(mouse["function"]["parameters"]["properties"]), mouse

            # A realistic board: random cells so it does not compress away under
            # BPE, built through the agent's own prompt builder so the preflight
            # cannot drift from what a real turn sends.
            rng = random.Random(0)
            board = [[rng.randrange(16) for _ in range(64)] for _ in range(64)]
            agent = my_agent.MyAgent()
            agent._step = 1
            prompt = agent.prompt_for(board, ["UP", "MOUSE"], tool_mode=True)
            assert "Call EXACTLY ONE function" in prompt, prompt[:200]
            assert my_agent.MyAgent.MAX_ACTIONS >= 80, my_agent.MyAgent.MAX_ACTIONS
            print(f"Preflight: agent OK — {len(names)} tools {names}, "
                  f"prompt {len(prompt)} chars, MAX_ACTIONS={my_agent.MyAgent.MAX_ACTIONS}, "
                  f"POOL_SIZE={my_agent.POOL_SIZE}")

            # --- inference check: needs a GPU the commit build does not have ---
            # Load the exact library llama_cpp loads. ctypes.util.find_library
            # ("cuda") is NOT equivalent: on a box with the CUDA toolkit but no
            # driver — precisely the Kaggle build image — it returns a path and
            # the import still dies on libcuda.so.1 (that mistake failed v34).
            try:
                ctypes.CDLL("libcuda.so.1")
                has_cuda_driver = True
            except OSError:
                has_cuda_driver = False

            if not has_cuda_driver:
                print("Preflight: NO GPU in this build (libcuda absent) — skipping the "
                      "inference check. Model load and tool calling are exercised only "
                      "in the rerun; validate them locally with arc_agi3_score_local.py.")
            else:
                from llama_cpp import Llama

                llm = Llama(model_path=os.environ["LLAMA_MODEL_PATH"], n_ctx=n_ctx,
                            n_gpu_layers=int(os.environ["LLAMA_N_GPU_LAYERS"]), verbose=True)
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": my_agent.TOOL_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    tools=tools,
                    tool_choice="required",
                    temperature=0.4,
                    max_tokens=my_agent.MAX_OUTPUT_TOKENS,
                )
                message = out["choices"][0]["message"]
                content = my_agent.strip_thinking(message.get("content") or "")
                # Same recovery the agent uses: this template often leaves the
                # call as text in `content` rather than filling `tool_calls`.
                calls = message.get("tool_calls") or my_agent.parse_text_tool_calls(content)
                structured = bool(message.get("tool_calls"))
                prompt_tokens = out["usage"]["prompt_tokens"]
                print(f"Preflight prompt_tokens={prompt_tokens} structured={structured} "
                      f"tool_calls={json.dumps(calls)[:300]}")
                assert calls, (
                    "no tool call, structured or recoverable from text; "
                    f"content={content[:200]!r}"
                )
                assert "<think>" not in (message.get("content") or ""), (
                    "model is still emitting <think> despite /no_think; it will "
                    "burn the token cap before producing an action"
                )
                called = [(c.get("function") or {}).get("name", "").upper() for c in calls]
                assert any(my_agent.NAME_TO_ID.get(n) in available for n in called), (
                    f"tool call names {called} resolve to no available action"
                )
                print(f"Preflight OK — n_ctx={n_ctx}, {prompt_tokens} prompt tokens.")
        """)
    )

    # Cell5: Run agent (competition rerun only — gateway sidecar is only available then)
    run_cell_source = dedent(
        """\
        import os
        import subprocess
        import sys

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            AGENTS_WD = '/kaggle/working/ARC-AGI-3-Agents'

            # Wait for the gateway sidecar to be ready.
            subprocess.run(
                ['curl', '--fail', '--retry', '999', '--retry-all-errors',
                 '--retry-delay', '5', '--retry-max-time', '600',
                 'http://gateway:8001/api/games'],
                check=True,
            )
            print('[Cell5] gateway reachable')

            # Copy the framework into a writable location.
            subprocess.run(
                ['rm', '-rf', AGENTS_WD], check=True)
            subprocess.run(
                ['cp', '-r',
                 '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents',
                 AGENTS_WD],
                check=True,
            )
            print('[Cell5] framework copied')

            # Drop our agent in as a framework template.
            subprocess.run(
                ['cp', '-f', '/tmp/my_agent.py',
                 f'{AGENTS_WD}/agents/templates/my_agent.py'],
                check=True,
            )
            print('[Cell5] agent template installed')

            # Register MyAgent in the framework's agent registry.
            with open(f'{AGENTS_WD}/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open(f'{AGENTS_WD}/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")
            print('[Cell5] .env written')

            # Run it. The gateway records every action and emits submission.parquet.
            # Explicit env passthrough so LLAMA_MODEL_PATH etc. from Cell 2 survive.
            env = os.environ.copy()
            env['MPLBACKEND'] = 'agg'
            result = subprocess.run(
                [sys.executable, 'main.py', '--agent', 'myagent'],
                cwd=AGENTS_WD,
                env=env,
            )
            print(f'[Cell5] main.py exited with rc={result.returncode}')
            if result.returncode != 0:
                raise SystemExit(result.returncode)

            parquet = '/kaggle/working/submission.parquet'
            if os.path.exists(parquet):
                print(f'[Cell5] submission.parquet written ({os.path.getsize(parquet)} bytes)')
            else:
                print('[Cell5] WARNING: submission.parquet NOT found after run')
        """
    )
    run_cell = code_cell(run_cell_source)

    # Cell5: Dummy submission for commit mode
    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    # --- vLLM backend, matching the Duck harness environment -----------------
    # Same pinned docker image, the same offline vLLM wheelhouse and the same
    # Qwen3.6-27B-FP8 snapshot (see notebooks/kernel-metadata.json). vLLM batches
    # concurrent requests, so the ~110 game threads stop queueing behind one
    # model — the structural difference between us and the leading solutions.
    vllm_install_cell = code_cell(
        dedent("""\
        import glob, os, subprocess, sys

        print("=== /kaggle/input ===")
        for item in sorted(os.listdir("/kaggle/input")):
            print(" ", item)

        def find_input(*names):
            \"\"\"Kaggle mounts a dataset at /kaggle/input/<slug> or
            /kaggle/input/datasets/<owner>/<slug>; probe both.\"\"\"
            for name in names:
                for candidate in (f"/kaggle/input/{name}",
                                  *glob.glob(f"/kaggle/input/**/{name}", recursive=True)):
                    if os.path.isdir(candidate):
                        return candidate
            raise FileNotFoundError(f"none of {names} found under /kaggle/input")

        ARC_WHEELS = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
        VLLM_WHEELS = find_input("arc3-vllm-h100-wheelhouse-v3")
        MODEL_DIR = find_input("vrfai-qwen3-6-27b-fp8-hf-snapshot")
        print(f"vLLM wheelhouse: {VLLM_WHEELS}")
        print(f"model snapshot : {MODEL_DIR}")

        def pip_install(*args):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                 "--no-warn-conflicts", "--disable-pip-version-check", *args],
                stdout=subprocess.DEVNULL,
            )

        # Do NOT install pillow. The image already ships a working one, and
        # installing the wheelhouse copy over it left ImageDraw broken
        # ("cannot import name '_Ink' from 'PIL._typing'"), which silently
        # disabled every board image in kernels v50 and v51.
        pip_install("--find-links", ARC_WHEELS, "arc-agi", "python-dotenv")
        from PIL import Image  # noqa: F401  - fail here, not mid-game
        print(f"pillow {__import__('PIL').__version__} (stock)")
        pip_install("--find-links", VLLM_WHEELS, "vllm")
        import vllm
        print(f"vllm {vllm.__version__} installed")
        """)
    )

    vllm_serve_cell = code_cell(
        dedent("""\
        import os, socket, subprocess, sys, time
        from urllib.request import urlopen

        # The CUDA libraries are off the linker path on Kaggle GPU images, so
        # torch/vllm cannot find libcuda without this (the Duck harness does the
        # same). Set before the server process is spawned.
        CUDA_LIB = "/usr/local/nvidia/lib64"
        for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
            os.environ[var] = os.pathsep.join(
                [CUDA_LIB, *[p for p in os.environ.get(var, "").split(os.pathsep) if p]]
            )

        SERVED_MODEL = "arc-agent"
        PORT = 8000
        os.environ["ARC_LLM_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
        os.environ["ARC_LLM_MODEL"] = SERVED_MODEL
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

        # The server lives in this kernel in both modes; the play subprocess
        # reaches it over HTTP via ARC_LLM_BASE_URL, which it inherits.
        server_log = open("/kaggle/working/vllm_server.log", "wb")
        server = subprocess.Popen(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
             "--model", MODEL_DIR,
             "--served-model-name", SERVED_MODEL,
             "--port", str(PORT),
             # One board is ~2.9k tokens and replies are capped at 64, so 8k is
             # ample; a smaller window leaves more KV cache for concurrency.
             "--max-model-len", "8192",
             "--max-num-seqs", "20",
             "--gpu-memory-utilization", "0.85",
             # Qwen emits <tool_call> blocks; this parser turns them into
             # structured tool_calls (the agent can also recover them itself).
             "--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
            stdout=server_log, stderr=subprocess.STDOUT, env=os.environ.copy(),
        )

        # Loading 36GB of FP8 weights takes minutes; the gateway wants a first
        # action within ~15, so fail loudly rather than let games start blind.
        DEADLINE_S = 15 * 60
        started = time.time()
        while time.time() - started < DEADLINE_S:
            if server.poll() is not None:
                print(open("/kaggle/working/vllm_server.log").read()[-4000:])
                raise RuntimeError(f"vLLM exited early with rc={server.returncode}")
            try:
                with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        print(f"vLLM ready in {time.time() - started:.0f}s")
                        break
            except Exception:
                time.sleep(5)
        else:
            raise TimeoutError(f"vLLM not ready within {DEADLINE_S}s")
        """)
    )

    vllm_preflight_cell = code_cell(
        dedent("""\
        import json, os, random, sys

        sys.path.insert(0, "/tmp")  # cell 3 wrote the agent here
        import my_agent

        # The ported modules must import here, not merely have been written. A
        # staged package that cannot be imported is the same failure as a
        # feature that never fires: present, and doing nothing. numpy is assumed
        # to be in the kernel rather than installed, and a dependency this
        # project "obviously had" has already turned out to be broken here.
        sys.path.insert(0, "__KERNEL_PKG__")
        import numpy as _np
        from tgaer.agents.arc_agi3_explorer import ExplorerArcAgi3Agent
        from tgaer.agents.arc_agi3_semantics import EmpiricalSemantics
        assert EmpiricalSemantics().move_lattice() == {}, "cold lattice must be empty"
        print(f"Preflight: ported modules OK — numpy {_np.__version__}, "
              f"{ExplorerArcAgi3Agent.__name__} importable")

        # In a real rerun this reports but never raises: a broken tool path
        # still degrades to the raw-text fallback and can score, so killing the
        # run would be strictly worse. In the commit build it must fail loudly.
        MUST_PASS = not os.getenv("KAGGLE_IS_COMPETITION_RERUN")
        available = [1, 2, 3, 4, 5, 6]
        tools = my_agent.MyAgent._build_tools(available)
        assert my_agent.REMOTE_BACKEND is not None, "ARC_LLM_BASE_URL not set"

        rng = random.Random(0)
        board = [[rng.randrange(16) for _ in range(64)] for _ in range(64)]
        # A real construction, not __new__: skipping __init__ leaves out every
        # instance attribute the prompt touches, which is how a missing one
        # reached a build (AttributeError: no attribute '_dead_actions').
        # prompt_for is the agent's own builder, so this cannot drift from a
        # real turn — reassembling it by hand broke three builds in a row.
        agent = my_agent.MyAgent()
        agent._step = 1
        prompt = agent.prompt_for(board, ["UP", "MOUSE"], tool_mode=True)

        # Send exactly what a turn sends, image included. vLLM rejects images
        # outright if the server was not built to accept them for this model, so
        # this must be exercised here rather than discovered mid-game.
        content = agent._user_content(prompt, tuple(tuple(r) for r in board))
        sent_image = not isinstance(content, str)
        print(f"Preflight sending image: {sent_image}")
        if not sent_image:
            probe = my_agent.render_board_png(tuple(tuple(r) for r in board))
            print(f"Preflight image render: png={probe is not None} "
                  f"SEND_IMAGE={my_agent.SEND_IMAGE} reason={my_agent._RENDER_FAILURE!r}")
        try:
            out = my_agent.REMOTE_BACKEND.create_chat_completion(
                messages=[{"role": "system", "content": my_agent.TOOL_SYSTEM},
                          {"role": "user", "content": content}],
                tools=tools, tool_choice="required", temperature=0.4,
                max_tokens=my_agent.MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            if not sent_image:
                raise
            # Fall back to text so one unsupported feature cannot cost the run,
            # and say so loudly enough to act on.
            # The env var, not just the module flag: the mock re-imports
            # my_agent and the rerun plays in a subprocess, so only the
            # environment carries the decision to either of them.
            print(f"WARNING: server rejected the image ({exc}); falling back to text "
                  "for the whole run.")
            os.environ["ARC_SEND_IMAGE"] = "0"
            my_agent.SEND_IMAGE = False
            # Every later probe sends this too, so it must be the text payload
            # rather than the image the server just refused.
            content = prompt
            out = my_agent.REMOTE_BACKEND.create_chat_completion(
                messages=[{"role": "system", "content": my_agent.TOOL_SYSTEM},
                          {"role": "user", "content": content}],
                tools=tools, tool_choice="required", temperature=0.4,
                max_tokens=my_agent.MAX_OUTPUT_TOKENS,
            )
        def report(mode, out):
            \"\"\"Whether this tool mode yields a usable action, and why not.

            Prints the raw message, not the stripped one. v53 failed here with
            `content=''` after stripping, which is what an unterminated <think>
            block and a tool call the server's parser rejected both look like —
            indistinguishable, so the build could not say which it was.
            \"\"\"
            message = out["choices"][0]["message"]
            raw = message.get("content") or ""
            content = my_agent.strip_thinking(raw)
            calls = message.get("tool_calls") or my_agent.parse_text_tool_calls(content)
            called = [(c.get("function") or {}).get("name", "").upper() for c in calls]
            usable = any(my_agent.NAME_TO_ID.get(n) in available for n in called)
            print(f"[{mode}] usage={out.get('usage')} "
                  f"finish={out['choices'][0].get('finish_reason')!r} "
                  f"structured={bool(message.get('tool_calls'))} usable={usable}")
            print(f"[{mode}] called={called} raw={raw[:300]!r}")
            return usable

        # Probe the runtime path first, then the alternative. The scored kernel
        # serves vLLM 0.19 from a pinned wheelhouse and local runs are on 0.26,
        # so which mode returns a call is a property of the server, not the
        # prompt, and picking it here costs one request instead of a submission.
        # Retry the runtime mode before blaming it. This is one sample from a
        # temperature-0.4 model: v53 drew 128 tokens and no call, failed the
        # build, and v54 drew a clean call from a byte-identical request. A
        # single miss says nothing about whether tool calling works.
        usable = report("required", out)
        for attempt in range(2):
            if usable:
                break
            print(f"[required] miss; resampling ({attempt + 1}/2)")
            out = my_agent.REMOTE_BACKEND.create_chat_completion(
                messages=[{"role": "system", "content": my_agent.TOOL_SYSTEM},
                          {"role": "user", "content": content}],
                tools=tools, tool_choice="required", temperature=0.4,
                max_tokens=my_agent.MAX_OUTPUT_TOKENS,
            )
            usable = report("required", out)
        if not usable:
            for mode in ("auto",):
                probe_out = my_agent.REMOTE_BACKEND.create_chat_completion(
                    messages=[{"role": "system", "content": my_agent.TOOL_SYSTEM},
                              {"role": "user", "content": content}],
                    tools=tools, tool_choice=mode, temperature=0.4,
                    max_tokens=my_agent.MAX_OUTPUT_TOKENS,
                )
                if report(mode, probe_out):
                    # The env var as well as the flag: the rerun plays in a
                    # subprocess that re-imports my_agent, and only the
                    # environment survives that.
                    os.environ["ARC_TOOL_CHOICE"] = mode
                    my_agent.TOOL_CHOICE = mode
                    print(f"Preflight selected tool_choice={mode!r}")
                    usable = True
                    break
        if MUST_PASS:
            assert usable, "no tool mode returned a usable action; see the raw output above"
        elif not usable:
            print("WARNING: tool calling unusable; the agent will use raw-text fallback")
        print(f"Preflight {'OK' if usable else 'DEGRADED'} — vLLM serving "
              f"with tool_choice={my_agent.TOOL_CHOICE!r}.")
        """)
    )

    # Mock submission: play real games in-kernel, on the real GPU, in commit
    # mode. A commit run costs GPU quota but NOT the daily submission slot, so
    # this is the only way to exercise the whole pipeline — agent, server, game
    # loop, scoring — before spending the one submission we get per day.
    # Games run concurrently, as the Swarm runs them, so the number that
    # matters (actions per minute under concurrency) is measured, not assumed.
    mock_submission_cell = code_cell(
        dedent("""\
        import os, shutil, sys, threading, time

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("Rerun: skipping the mock submission; the real games follow.")
        else:
            COMP = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'
            AGENTS_WD = '/kaggle/working/ARC-AGI-3-Agents'
            # Threads, not games: only ~6 games ship in environment_files, but
            # the rerun runs ~110 concurrently. Server contention is what we
            # need to measure, so cycle several threads over the games we have.
            # Depth, not breadth. Level-1 baselines in the bundled games are
            # 19-32 actions, so a 20-action run cannot complete a level however
            # good the agent is — 0 levels was arithmetic, not a verdict. Give
            # each game enough actions for level 1 to be reachable, so the mock
            # reports the thing that actually scores. Concurrency was measured
            # separately; raise ARC_MOCK_THREADS to re-test contention.
            MOCK_THREADS = int(os.environ.get('ARC_MOCK_THREADS', '6'))
            MOCK_ACTIONS = int(os.environ.get('ARC_MOCK_ACTIONS', '150'))

            shutil.rmtree(AGENTS_WD, ignore_errors=True)
            shutil.copytree(f'{COMP}/ARC-AGI-3-Agents', AGENTS_WD)
            shutil.copy('/tmp/my_agent.py', f'{AGENTS_WD}/agents/templates/my_agent.py')
            # Slim the registry so importing it doesn't drag in langgraph etc.
            with open(f'{AGENTS_WD}/agents/__init__.py', 'w') as fh:
                fh.write('from .agent import Agent\\n')
            sys.path.insert(0, AGENTS_WD)

            import arc_agi
            from arc_agi import OperationMode
            sys.path.insert(0, '/tmp')
            # The preflight already imported my_agent while the framework was
            # NOT importable, so it bound the local stub base class — which has
            # no main() and no frames. Re-import now that agents/ is on the
            # path, or every game thread dies on the first attribute access.
            sys.modules.pop('my_agent', None)
            import my_agent
            assert hasattr(my_agent.MyAgent, 'main'), (
                'MyAgent still has the stub base; the framework is not importable'
            )

            # OFFLINE: read the bundled games straight off disk, no gateway and
            # no scorecard API (the kernel has no internet).
            arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                                 environments_dir=f'{COMP}/environment_files')
            game_ids = [e.game_id.split('-')[0] for e in arc.get_environments()]
            slots = [(f'{game_ids[i % len(game_ids)]}#{i}', game_ids[i % len(game_ids)])
                     for i in range(MOCK_THREADS)]
            print(f'Mock submission: {MOCK_THREADS} concurrent threads over '
                  f'{len(game_ids)} games x {MOCK_ACTIONS} actions')

            results = {}
            latencies = []
            decisions = {}

            def play(slot, game_id):
                # Threads swallow exceptions by default; record them instead, or
                # a failure looks identical to "played 0 actions".
                try:
                    env = arc.make(game_id)
                    if env is None:
                        results[slot] = ('no-env', 0, 0)
                        return
                    agent = my_agent.MyAgent(
                        card_id='mock', game_id=game_id, agent_name=f'mock.{slot}',
                        ROOT_URL='http://localhost', record=False, arc_env=env,
                        tags=['mock'])
                    agent.MAX_ACTIONS = MOCK_ACTIONS
                    t0 = time.time()
                    agent.main()
                    frame = agent.frames[-1]
                    if agent.action_counter:
                        latencies.append((time.time() - t0) / agent.action_counter)
                    results[slot] = (str(frame.state), frame.levels_completed,
                                     agent.action_counter)
                    for key, count in agent.stats.items():
                        decisions[key] = decisions.get(key, 0) + count
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    results[slot] = (f'ERROR {exc!r}'[:120], 0, 0)

            started = time.time()
            threads = [threading.Thread(target=play, args=(s, g)) for s, g in slots]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.time() - started

            actions = sum(r[2] for r in results.values())
            levels = sum(r[1] or 0 for r in results.values())
            errors = {s: r[0] for s, r in results.items() if str(r[0]).startswith('ERROR')}
            for slot, (state, lvl, acts) in sorted(results.items())[:8]:
                print(f'  {slot}: levels={lvl} actions={acts} state={state}')
            if errors:
                print(f'  ERRORS in {len(errors)}/{len(results)} threads: '
                      f'{list(errors.values())[:2]}')
            # How each action was actually decided. A run that fell back to
            # random every turn otherwise looks identical to one that played.
            print(f'  decisions: {dict(sorted(decisions.items()))}')
            mean_latency = sum(latencies) / len(latencies) if latencies else 0
            print(f'Mock submission: {actions} actions in {elapsed:.0f}s '
                  f'= {actions / elapsed * 60:.1f} actions/min across '
                  f'{MOCK_THREADS} concurrent threads; {levels} levels completed; '
                  f'mean {mean_latency:.1f}s per action per thread')

            # The rerun plays ~110 games; project whether the budget is usable.
            print(f'Projected at this rate: {actions / elapsed * 3600 * 7.5 / 110:.0f} '
                  f'actions per game over a 7.5h run of 110 games')
            assert actions > 0, 'no actions were taken - the agent never played'
            # A timeout degrades silently to random actions, so a partial
            # failure must fail the build rather than look like a slow run.
            assert not errors, f'{len(errors)} threads failed: {list(errors.values())[:3]}'
        """)
    )

    if ACCELERATOR not in ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: {sorted(ACCELERATORS)}"
        )
    accel = ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["machine_shape"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                f"Agent driven by a local LLM (`{BACKEND}` backend).\n"
                "Built from `src/tgaer/agents/arc_agi3_kaggle.py` via `src/tgaer/evaluation/arc_agi3_build_notebook.py`."
            ),
            stage_pkg_cell,
            *module_cells,
            *(
                [
                    vllm_install_cell,
                    write_agent_cell,
                    vllm_serve_cell,
                    vllm_preflight_cell,
                ]
                if BACKEND == "vllm"
                else [install_cell, load_model_cell, write_agent_cell, preflight_cell]
            ),
            mock_submission_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(
        f"[arc_agi3_build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
        f"(backend: {BACKEND}, accelerator: {ACCELERATOR})"
    )

    # Sync metadata. machine_shape is the field the CLI actually reads; keep
    # enable_gpu consistent even though it is deprecated and ignored.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        accel = ACCELERATORS[ACCELERATOR]
        wanted = {"machine_shape": accel["machine_shape"], "enable_gpu": accel["gpu"]}
        if any(meta.get(k) != v for k, v in wanted.items()):
            meta.update(wanted)
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[arc_agi3_build_notebook] Synced {wanted}")


if __name__ == "__main__":
    main()
