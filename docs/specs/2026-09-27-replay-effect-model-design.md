# Replay-based effect model — design

Date: 2026-09-27
Status: approved for stage 1

## Why

The explorer spends 1463–1707 actions on games an oracle finishes in 13–47, and RHAE
squares actions-per-level, so action efficiency is where the score is. Five levers have
been measured and closed:

| lever | result |
| --- | --- |
| action budget 6000 → 12000 | `+0.0001pp` |
| defer irreversible edges | `-0.1659pp`, 3 games lost |
| drop the `field_box` crop | `-0.1661pp`, 3 games lost |
| re-validate the avatar latch | tu93 2 levels → 1 |
| `(colour, size)` class compression | candidates collapse only 1.4x |

Three of those removed search breadth and regressed ~8 sd. What has *not* been tried is
making each action teach more: fitting an effect model from recorded transitions so a
rule confirmed once need not be re-tested, and so hypotheses are falsified by replay
rather than by spending actions.

Two prior failures bound the design. A full-board simulator scored 0.1% against a 46%
baseline ("the heuristic transfers, the simulator does not"), so predicting frames is
out. A cross-game action ranker did not transfer, because an action id means something
different per game. Both point the same way: predict a **coarse, typed effect**, keyed on
features that mean the same thing in every game.

## Non-goals

- No LLM. The competition forbids internet access during evaluation, and a dev-time
  model dependency the kernel cannot reproduce is worse than none.
- No policy change. Stage 1 must not be able to regress the agent.
- No goal/reward inference. No published work establishes a win predicate is inferable
  with zero observed clears, and our value-model attempt failed.

## Architecture

One new bench tool, `sia-oss/bench/replay_effects.py`. **Nothing under `src/` changes**,
so there is no gate and no regression risk. It drives the roster through the existing
`play(..., on_step=hook)` seam that `action_budget.py` and `value_model.py` already use,
inheriting the harness rather than inventing one.

## The effect signature

A hashable tuple derived from `(prev, action, cur)`, computed on `_settled` frames so
chrome cannot pollute it — the same masking the agent's own state key uses:

```
(level_advanced,        # bool
 avatar_delta,          # (dr, dc) from _det.avatar centroid, None when unpinned
 appeared,              # frozenset of colours in cur but not prev
 vanished,              # frozenset of colours in prev but not cur
 churn_bucket)          # 0 if no cell changed, else 1 + floor(log2(n_changed))
```

`size_bucket` in the key below is the same shape: `0` for a single cell, else
`1 + floor(log2(cells))`. Buckets rather than raw counts so a 9-cell and a 10-cell
blob share a rule — raw counts would fragment the key the way raw boards fragment
the state signature.

Coarse by construction, and a strict superset of `_inert`, which is exactly
`churn_bucket == 0`. That containment is deliberate: it makes the measurement able to say
whether the model beats the mechanism already shipped, rather than merely differing
from it.

## The model

`(action, colour, size_bucket) -> Counter[effect]`, predicting the modal effect, with
back-off to `(action, colour)` then `(action)`. Colour and size come from the clicked
component; simple actions key on the action id alone. The back-off level used is recorded
per prediction, so the report can say how often the specific key was even available.

Back-off is the point rather than a convenience: a rule confirmed at one granularity
still fires at a context never seen, which is what an out-of-distribution palette needs.
A flat table fails closed exactly where generalisation is required.

## Evaluation: prequential

For each recorded transition, predict with the model **as it stands**, score the
prediction, then update. No train/test split, no leakage by construction, and it mirrors
what the agent would experience online. Fitting and scoring on one trajectory would be
circular — the flaw that cost six iterations in the value-model harness.

## Metrics, per game

| metric | decides |
| --- | --- |
| prequential accuracy, class-keyed | is the model right on-policy at all |
| same, state-keyed (signature in the key) | does cross-state generalisation buy anything |
| same, `_inert`-equivalent (predict `churn_bucket == 0` only) | does it beat what ships |
| distribution of back-off level used | is the specific key ever available |
| accuracy on the **first sighting** of a context | can it predict somewhere new |

The last row is the one that matters. Everything above it can be satisfied by
memorisation; only first-sighting accuracy speaks to behaviour on games the roster never
shows.

## Tests

- A synthetic trajectory with one deterministic rule reaches 100% after first sighting.
- A deliberately inconsistent rule does not exceed its own purity.
- Back-off fires at a novel colour rather than abstaining.
- The effect signature is invariant to chrome cells. This is the test most worth having:
  chrome has broken two mechanisms already.

## Success criteria for stage 2

Stage 2 (letting the model influence action ordering) is justified only if class-keyed
accuracy beats **both** the state-keyed variant and the `_inert`-equivalent baseline, and
first-sighting accuracy beats the **marginal baseline** — predicting the game's single
most common effect while ignoring the key entirely. That is the concrete definition of
"chance" here, and it is computed in the same pass. If first-sighting accuracy does not
beat it, the model memorises rather than generalises and stage 2 is not worth building.

Note what the real test is. Our 25-game roster is never scored: the Kaggle rerun plays
110 games, and `publicScore` is ~55 **semi-private** games. Local RHAE is therefore a
regression guard, not the objective — a 2.25x local gain moved that held-out score by
zero. Stage 2's verdict comes from `publicScore`, one out-of-distribution reading per day.

## Risks

- **The model is accurate but useless.** Effect purity is already 98–100%, so high
  accuracy may be unsurprising and carry no action saving — the same trap that killed
  class compression at 1.4x. Mitigation: the `_inert` baseline column makes this visible
  before any policy work.
- **Chrome leaks into the signature**, inflating `churn_bucket` and making every effect
  look distinct. Mitigation: the invariance test, plus computing on `_settled`.
- **Avatar delta is unavailable** on click games, where `_det.avatar` is unpinned. That
  is expected, not a bug; the field is `None` and the other four carry the signal.
