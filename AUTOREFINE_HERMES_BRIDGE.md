# TGAER + Hermes: AutoRefine Bridge

This document outlines how TGAER's `AutoRefine` module integrates with `hermes-agent` as its primary execution and data-generation engine. 

By bridging TGAER and Hermes, we achieve zero-human-in-the-loop synthetic data generation for DPO and GSPO alignment.

## The Architecture

1. **Environment Node (Hermes):** TGAER no longer needs to mock execution environments. It calls `hermes rollout` to spin up isolated, sandboxed terminal/browser workspaces.
2. **Evaluator Node (RLVR):** TGAER passes verifier scripts (e.g., `pytest`, AST checkers) to Hermes. Hermes returns the binary sequence-level rewards (0.0 or 1.0).
3. **AutoRefine Node (DPO Generator):**
    - When Hermes returns a `0.0` reward (Failed Trajectory), AutoRefine triggers.
    - It extracts the exact Hermes `<tool_call>` sequence and the resulting traceback.
    - It prompts a Teacher Model (e.g., Claude 3.5 Sonnet): "The student agent failed with this traceback. Correct the trajectory to solve the user's prompt."
    - The Teacher's successful trajectory (Reward `1.0`) becomes the `chosen` sequence.
    - The Student's failed trajectory becomes the `rejected` sequence.

## Implementation Steps (Weekend Deep Work)
1. Write `autorefine_dpo_export.py` in TGAER.
2. Consume the `gspo_dataset.jsonl` output by `hermes rollout`.
3. Format the chosen/rejected pairs into the HuggingFace `trl` format.
4. Test the generated DPO dataset on ARC-AGI-3 baseline models.
