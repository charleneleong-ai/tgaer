from __future__ import annotations

from typing import Any, Dict, Optional

import dspy


class ARCReasoningSignature(dspy.Signature):
    task_description: str = dspy.InputField()
    examples: str = dspy.InputField()
    hypothesis: str = dspy.OutputField()


class ARCActionSignature(dspy.Signature):
    hypothesis: str = dspy.InputField()
    input_grid: str = dspy.InputField()
    output_grid: str = dspy.OutputField()


class OrakPlanningSignature(dspy.Signature):
    mission_description: str = dspy.InputField()
    observation_summary: str = dspy.InputField()
    history: str = dspy.InputField()
    plan: str = dspy.OutputField()
    next_action: str = dspy.OutputField()


class OrakReflectSignature(dspy.Signature):
    mission_description: str = dspy.InputField()
    observation_summary: str = dspy.InputField()
    last_action: str = dspy.InputField()
    outcome_summary: str = dspy.InputField()
    strategy_hint: str = dspy.OutputField()


def build_arc_program(
    lm: Optional[dspy.LM] = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> dspy.Module:
    if lm is None:
        lm = dspy.OpenAI(
            model="gpt-4.1-mini",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    dspy.settings.configure(lm=lm)

    class ARCProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.reason = dspy.Predict(ARCReasoningSignature)
            self.act = dspy.Predict(ARCActionSignature)

        def forward(self, task_description: str, examples: str, input_grid: str) -> Dict[str, str]:
            hyp = self.reason(
                task_description=task_description,
                examples=examples,
            )
            out = self.act(
                hypothesis=hyp.hypothesis,
                input_grid=input_grid,
            )
            return {
                "hypothesis": hyp.hypothesis,
                "output_grid": out.output_grid,
            }

    return ARCProgram()


def build_orak_program(
    lm: Optional[dspy.LM] = None,
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> dspy.Module:
    if lm is None:
        lm = dspy.OpenAI(
            model="gpt-4.1",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    dspy.settings.configure(lm=lm)

    class OrakProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.plan_step = dspy.Predict(OrakPlanningSignature)
            self.reflect = dspy.Predict(OrakReflectSignature)

        def step(
            self,
            mission_description: str,
            observation_summary: str,
            history: str,
        ) -> Dict[str, str]:
            res = self.plan_step(
                mission_description=mission_description,
                observation_summary=observation_summary,
                history=history,
            )
            return {
                "plan": res.plan,
                "next_action": res.next_action,
            }

        def reflect_step(
            self,
            mission_description: str,
            observation_summary: str,
            last_action: str,
            outcome_summary: str,
        ) -> str:
            res = self.reflect(
                mission_description=mission_description,
                observation_summary=observation_summary,
                last_action=last_action,
                outcome_summary=outcome_summary,
            )
            return res.strategy_hint

    return OrakProgram()
