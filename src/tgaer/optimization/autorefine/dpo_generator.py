#!/usr/bin/env python3
"""
TGAER AutoRefine / DPO Generator Node

Integrates with Hermes Agent's isolated execution environments to read a GSPO
rollout JSONL, find failed trajectories (reward=0.0), and use a Teacher model 
to generate a corrected (chosen) trajectory. Exports a DPO-ready dataset: 
{prompt, chosen, rejected}
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Find the hermes-agent path explicitly to ensure absolute resolution
hermes_agent_dir = os.path.expanduser("~/.hermes/hermes-agent")
if hermes_agent_dir not in sys.path:
    sys.path.insert(0, hermes_agent_dir)

from run_agent import AIAgent
from agent.agent_init import init_agent
from tools.environments.local import LocalEnvironment
from hermes_constants import get_hermes_home

def generate_dpo_pair(task: dict, verifier_cmd: str, teacher_model: str, teacher_provider: str) -> dict:
    """Takes a failed task, runs a teacher to fix it, and returns a DPO pair."""
    original_prompt = task["prompt"][0]["content"]
    
    # Find a rejected completion
    rejected_completion = None
    for comp, rew in zip(task["completions"], task["rewards"]):
        if rew == 0.0:
            rejected_completion = comp
            break
            
    if not rejected_completion:
        print("No failed trajectory found to correct.")
        return None

    # Format the failed trajectory to show the teacher what went wrong
    failed_log = ""
    for msg in rejected_completion:
        failed_log += f"[{msg['role']}]: {msg['content']}\n\n"

    teacher_prompt = (
        f"[System directive: You are a senior engineer correcting an AI student's mistake.]\n"
        f"Original Task: {original_prompt}\n\n"
        f"The student attempted this task but failed verification. Here is their trajectory:\n"
        f"--- STUDENT ATTEMPT ---\n{failed_log}\n--- END ATTEMPT ---\n\n"
        f"Please complete the task correctly. Do not repeat their mistakes."
    )

    print(f"Spawning Teacher ({teacher_model}) to correct the trajectory...")
    
    # Setup isolated workspace for the teacher
    dpo_tmp_dir = os.path.join(get_hermes_home(), "dpo_tmp")
    os.makedirs(dpo_tmp_dir, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="hermes_dpo_teacher_", dir=dpo_tmp_dir))
    
    task_id = f"dpo_teacher_{workdir.name}"
    from tools.terminal_tool import register_task_env_overrides
    register_task_env_overrides(task_id, {"cwd": str(workdir)})

    # Enforce isolation path in prompt
    isolated_prompt = f"[System directive: Your absolute working directory is {workdir}. All file operations MUST be strictly relative to this directory.]\n\n{teacher_prompt}"

    raw_agent = AIAgent()
    init_agent(
        raw_agent,
        platform="cli",
        model=teacher_model,
        provider=teacher_provider,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=10,
        quiet_mode=True,
    )
    
    try:
        result = raw_agent.run_conversation(isolated_prompt, task_id=task_id)
        raw_trajectory = raw_agent._convert_to_trajectory_format(
            result["messages"], isolated_prompt, result["completed"]
        )
    except Exception as e:
        print(f"Teacher crashed: {e}")
        shutil.rmtree(workdir, ignore_errors=True)
        return None

    print("Teacher finished. Verifying...")
    term = LocalEnvironment(cwd=str(workdir))
    verify_result = term.execute(verifier_cmd, timeout=30)
    
    code = verify_result.get("returncode", verify_result.get("exit_code", -1))
    if code != 0:
        print(f"Teacher failed verification too (exit code {code}). Output: {verify_result.get('output')} Discarding pair.")
        shutil.rmtree(workdir, ignore_errors=True)
        return None

    print("Teacher succeeded! Formatting chosen trajectory...")
    
    # Format the chosen completion
    chosen_sequence = []
    for msg in raw_trajectory:
        if isinstance(msg, dict):
            role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
            role = role_map.get(msg.get("from", "user"), "user")
            content = msg.get("value", "")
            
            # Skip the massive teacher prompt injection so it perfectly matches the original user prompt format
            if role != "user" and role != "system":
                chosen_sequence.append({"role": role, "content": content})

    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "prompt": [{"role": "user", "content": original_prompt}],
        "chosen": chosen_sequence,
        "rejected": rejected_completion
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input GSPO JSONL file with failed trajectories")
    parser.add_argument("--output", required=True, help="Output DPO JSONL file")
    parser.add_argument("--verifier", required=True, help="Verifier script command")
    parser.add_argument("--teacher-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--teacher-provider", default="gemini")
    args = parser.parse_args()

    success_count = 0
    with open(args.input, "r") as f_in, open(args.output, "a", encoding="utf-8") as f_out:
        for line in f_in:
            task = json.loads(line)
            # Only process if there's a failed reward
            if 0.0 in task.get("rewards", []):
                print(f"\nFound failed trajectory for prompt: '{task['prompt'][0]['content'][:50]}...'")
                dpo_pair = generate_dpo_pair(task, args.verifier, args.teacher_model, args.teacher_provider)
                
                if dpo_pair:
                    f_out.write(json.dumps(dpo_pair) + "\n")
                    f_out.flush()
                    success_count += 1
                    print(f"✅ Saved DPO pair.")

    print(f"\n🎉 Finished! Generated {success_count} perfect DPO pairs.")

if __name__ == "__main__":
    main()
