# Wiring a model grader, and calibrating it before you believe it

Everything shipped in this repository runs offline. That is deliberate — it is what lets CI run
the whole evaluation on every push, and what makes the demo produce the same numbers on your
machine as on mine. But the point of a calibration harness is to calibrate a real judge, so here
is the path from the recorded graders to a live one.

None of the code below ships in the package. It is the glue you write once.

---

## 1. The protocol

A grader is three members. There is no base class to inherit.

```python
class Grader(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_model(self) -> bool: ...

    def grade(self, item: Item) -> Verdict: ...
```

`is_model` is the one that matters. Setting it `True` is what arms the refusal: from that point
the report will not print this grader's estimate unless there are human labels to measure it
against. Set it honestly. A model grader labelled `is_model = False` gets its number printed
with nothing holding it up, which is the exact failure the package exists to prevent.

---

## 2. A live grader

```python
import json
import os

from anthropic import Anthropic
from legal_evals.items import Item, Verdict, check_label

VERDICT_TOOL = {
    "name": "record_verdict",
    "description": "Record the verdict on this candidate answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
            "rationale": {"type": "string"},
        },
        "required": ["label", "rationale"],
    },
}


class ClaudeGrader:
    name = "claude-judge"
    is_model = True

    def __init__(self, model: str = "claude-opus-5") -> None:
        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def grade(self, item: Item) -> Verdict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            tools=[VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict"},
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        question=item.question,
                        gold=item.gold,
                        candidate=item.candidate,
                    ),
                }
            ],
        )
        block = next(b for b in response.content if b.type == "tool_use")
        return Verdict(
            item_id=item.id,
            grader=self.name,
            label=check_label(block.input["label"]),
            rationale=str(block.input.get("rationale", "")),
        )
```

Two things about the prompt that are worth more than the plumbing:

**Give it the same three labels and the same strictness rule the humans had.** If your reviewers
were told a partial answer is not a correct one, the judge has to be told the same thing in the
same words. A good share of the disagreement people blame on the model is a judge working to a
different rubric than the humans it is being compared against.

**Do not show it the human verdict.** Obvious when stated; easy to do by accident when `Item`
carries `human` and you pass the whole object into a template.

---

## 3. Record it, then replay it

Calling a model on every item of every CI run is slow, costs money, and makes the reported
number drift for reasons nobody changed. Grade once, save the verdicts, and replay them.

```python
from legal_evals.graders import RecordedGrader
from legal_evals.items import EvalSet
from legal_evals.runner import run_graders

eval_set = EvalSet.load("fixtures/extraction.json")
result = run_graders(eval_set, [ClaudeGrader()])

RecordedGrader.from_run(result.run, "claude-judge").save("fixtures/graders/claude-judge.json")
```

A recorded grader file is the grader's name, whether it is a model, and one verdict per item:

```json
{
  "grader": "claude-judge",
  "is_model": true,
  "verdicts": [
    {"item_id": "x-001", "label": "partial", "rationale": "..."}
  ]
}
```

From then on `RecordedGrader.load(...)` replays it, and the evaluation is reproducible again.
It raises on an item it has no recording for rather than guessing — so adding items to the set
forces you to re-grade rather than silently evaluating a smaller set than you think.

---

## 4. Calibrate before you believe it

```bash
evals run --set fixtures/extraction.json --graders recorded:claude-judge
```

Read three numbers, in this order:

1. **Kappa.** Below 0.60 and the estimate is not a result. Raw agreement will look better than
   kappa does and it is the less informative of the two — a judge that says "correct" to
   everything agrees with a mostly-correct human most of the time.
2. **Too generous vs too harsh.** These are different problems. A consistently harsh judge still
   tracks a change over time; a generous one is not usable for anything, because it inflates
   every figure computed from it and the inflation is invisible in the figure.
3. **The denominator on the agreement column.** Kappa over twenty items is a noisy estimate of
   the judge's quality. The interval next to the agreement rate tells you how noisy.

Then look at `Where the graders and the human parted company`. The confusion table is where the
fix usually is: a judge that is only ever wrong in the `partial → correct` cell has a rubric
problem, not a reasoning problem, and rubric problems are cheap to fix.

---

## 5. The trap at the end

Once you start editing the judge's prompt and re-reading kappa, **you are fitting the judge to
your labelled set**, and kappa on that set stops being an honest estimate of the judge's quality.
It becomes a training score. This is the same overfitting everyone recognises in a model and
routinely misses in an evaluation, because it does not look like training.

Label a second set and do not touch it while you iterate. Confirm there. If the kappa you tuned
to and the kappa you confirm with are far apart, the first number was the prompt memorising your
twenty items.

The harness cannot detect this for you — it cannot see how many times you ran it. It is the one
discipline here that has to stay in your head.
