"""GEPA-style prompt candidates for dfranzen's harness: general reasoning procedure only."""

from __future__ import annotations

from typing import Any

# Content rule: a candidate describes how to form, test and revise rules, never naming a game, object,
# colour or mechanic.
PROTOCOL = (
    "- A level is always solvable. If your plan or search finds no solution, or a long plan fails, "
    "a rule in your model is wrong or missing: before searching again, name the assumption you are "
    "least sure of and test it with the cheapest single action that could falsify it.\n"
    "- Effects often depend on the situation, not only on the action: the same action can behave "
    "differently next to another object, after another change, or in another position. Before "
    "concluding that something is impossible, list the action-in-situation combinations you have "
    "not tried yet and test the cheapest one.\n"
    "- When an observation surprises you, or two attempts that look identical give different "
    "results, write down the smallest rule that explains both (consider your own earlier actions "
    "as a cause) and test that rule directly before continuing.\n"
)

LEDGER = (
    "- Keep a retained Python ledger of what you have observed: for each action you have taken, the "
    "situation it was taken in (what was adjacent or selected, what had changed before) and the "
    "effect it had. Consult it before each plan, and prefer probing a combination it does not "
    "cover yet over repeating one it already explains.\n"
)

G2A_PROBE_THROUGHPUT = "- Time is your scarcest resource, not actions. When stuck, keep each turn short: state one hypothesis, run one cheap test of 1-3 actions, read the result, repeat. Do not write a long derivation or build a simulator for a rule you have not yet seen work in the game.\n- Before testing a guess about the goal, make sure you know what every available action does to every kind of object here, including actions you have rarely used. Test an unknown action-on-object pair before any multi-step plan.\n- Do not re-test a fact you already verified. If a result surprises you, test the smallest explanation directly (it may be a delayed effect of your own earlier action) instead of patching your plan or simulator.\n"

G2B_EVIDENCE_LEDGER = "- A level is always solvable. When stuck, keep a short ledger and update it after every test: (a) for each action and each kind of object, what it was observed to do; (b) the goal hypotheses already ruled out, and by which test.\n- Choose the next test from gaps in the ledger. Prefer an action-on-object pair you have never tried, and a test whose outcome rules out the most remaining goal hypotheses. Never re-test a ledger entry unless new evidence contradicts it.\n- Solvability tells you your model is wrong somewhere. It never tells you that a specific move is safe.\n- When the game surprises you, record the smallest rule that explains it, including delayed effects of your own earlier actions, and test that rule next.\n"

CANDIDATES: dict[str, str] = {
    "c1_protocol": PROTOCOL,
    "c2_protocol_ledger": PROTOCOL + LEDGER,
    "g2a_probe_throughput": G2A_PROBE_THROUGHPUT,
    "g2b_evidence_ledger": G2B_EVIDENCE_LEDGER,
}


def install(tool_agent_module: Any, addendum: str) -> None:
    build = tool_agent_module._build_system_prompt

    def _build_system_prompt(*args: Any, **kwargs: Any) -> str:
        return build(*args, **kwargs) + addendum

    tool_agent_module._build_system_prompt = _build_system_prompt
