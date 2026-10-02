"""Running a set through its graders.

Short, because the interesting parts are elsewhere. Two decisions worth stating.

A grader that raises on one item does not abandon the run: the failure is recorded against that
item and the rest proceeds, because losing fifty-nine results to one bad row is a worse outcome
than an incomplete set. The report then refuses to compute agreement over a partial set, so the
incompleteness cannot be quietly averaged away.

Order is the set's order throughout. Everything downstream — kappa, the paired comparison —
lines two lists up positionally, so a grader that returned its verdicts in a different order
would silently compare the wrong pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graders import Grader
from .items import EvalSet, Run, Verdict


@dataclass
class RunReport:
    run: Run
    failures: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def run_graders(
    eval_set: EvalSet, graders: list[Grader], label: str = "", model: str = ""
) -> RunReport:
    """Grades every item with every grader, in the set's order."""
    run = Run(eval_set=eval_set.name, item_ids=eval_set.ids, label=label, model=model)
    report = RunReport(run=run)

    for grader in graders:
        for item in eval_set.items:
            try:
                verdict = grader.grade(item)
            except Exception as error:  # noqa: BLE001 - one bad item must not lose the run
                report.failures.append((grader.name, item.id, str(error)))
                continue
            if verdict.item_id != item.id:
                raise ValueError(
                    f"grader {grader.name!r} returned a verdict for {verdict.item_id!r} "
                    f"while grading {item.id!r}"
                )
            run.add(
                Verdict(
                    item_id=item.id,
                    grader=grader.name,
                    label=verdict.label,
                    rationale=verdict.rationale,
                )
            )

    return report


def model_grader_names(graders: list[Grader]) -> frozenset[str]:
    return frozenset(g.name for g in graders if g.is_model)
