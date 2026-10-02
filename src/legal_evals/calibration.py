"""Measuring the grader before believing anything it says.

The usual shape of an LLM evaluation is: run the system, have a model judge the output, report
the model's verdict as the result. The step that is missing is the one that decides whether any
of it means anything — **how often is the judge right?**

A judge that agrees with a person 70% of the time, used on a system that is right 85% of the
time, produces a number that is wrong in a direction nobody can predict from the number itself.
And the failure is invisible: the report looks exactly like a report from a judge that agrees
99% of the time.

So this module answers three questions about each grader, and the report prints all three
before it prints anything about the system:

1. How often does it agree with the human? With an interval, because the calibration set is
   usually small.
2. Is that agreement better than chance? Kappa, because a grader that answers "correct" to
   everything agrees with a mostly-correct human most of the time.
3. Which way does it err? A grader that is too generous and one that is too harsh are different
   problems, and the average hides which you have.
"""

from __future__ import annotations

from dataclasses import dataclass

from .items import LABELS, EvalSet, Label, Run
from .stats import Agreement, Interval, cohens_kappa, wilson


@dataclass(frozen=True)
class Calibration:
    """What one grader is worth, measured against the human labels."""

    grader: str
    is_model: bool
    agreement_with_human: Interval
    kappa: Agreement
    #: ``confusion[human][grader]`` — how often the human said one thing and the grader another.
    confusion: dict[Label, dict[Label, int]]

    @property
    def too_generous(self) -> int:
        """Items the human did not call correct and the grader did.

        The dangerous direction. These are the failures an evaluation will not show you, and
        they inflate every number computed from this grader.
        """
        return sum(self.confusion[human]["correct"] for human in LABELS if human != "correct")

    @property
    def too_harsh(self) -> int:
        """Items the human called correct and the grader did not.

        Wasteful rather than dangerous: it understates the system and sends somebody to
        investigate a success.
        """
        return sum(self.confusion["correct"][grader] for grader in LABELS if grader != "correct")

    @property
    def trustworthy(self) -> bool:
        """Whether a rate computed from this grader is worth printing without a caveat.

        The threshold is a judgement call and is stated rather than hidden: substantial
        agreement on Landis and Koch's scale, which is kappa at or above 0.6. The report never
        suppresses a grader for failing it — it prints the warning next to the number, because
        a reader who can see the calibration can decide for themselves.
        """
        return self.kappa.kappa >= 0.6

    def describe(self) -> str:
        return (
            f"{self.grader}: agrees with the human {self.agreement_with_human.as_percent()}, "
            f"{self.kappa.describe()}; "
            f"{self.too_generous} too generous, {self.too_harsh} too harsh"
        )


def calibrate(eval_set: EvalSet, run: Run, grader: str, is_model: bool) -> Calibration:
    """Measures one grader against the human labels, over the items that have one.

    Items nobody has judged are skipped rather than counted as agreement or disagreement. That
    is what makes a holdout workable: label sixty of five hundred, and the calibration is an
    honest figure over sixty rather than a flattering one over five hundred.
    """
    by_id = dict(zip(run.item_ids, run.labels_from(grader), strict=True))

    human_labels: list[Label] = []
    grader_labels: list[Label] = []
    for item in eval_set.items:
        if item.human is None:
            continue
        if item.id not in by_id:
            raise ValueError(f"grader {grader!r} has no verdict for labelled item {item.id!r}")
        human_labels.append(item.human)
        grader_labels.append(by_id[item.id])

    matches = sum(1 for h, g in zip(human_labels, grader_labels, strict=True) if h == g)

    confusion: dict[Label, dict[Label, int]] = {h: dict.fromkeys(LABELS, 0) for h in LABELS}
    for h, g in zip(human_labels, grader_labels, strict=True):
        confusion[h][g] += 1

    return Calibration(
        grader=grader,
        is_model=is_model,
        agreement_with_human=wilson(matches, len(human_labels)),
        kappa=(
            cohens_kappa(list(human_labels), list(grader_labels))
            if human_labels
            else Agreement(raw=0.0, expected_by_chance=0.0, kappa=0.0, n=0, raters=2)
        ),
        confusion=confusion,
    )


def estimate_from(run: Run, grader: str) -> Interval:
    """The system's success rate as this grader sees it."""
    labels = run.labels_from(grader)
    return wilson(sum(1 for label in labels if label == "correct"), len(labels))


def human_rate(eval_set: EvalSet) -> Interval:
    """The system's success rate according to the people. The only ground truth there is.

    The denominator is the number of items a person actually judged, not the size of the set.
    Dividing by the whole set would count every unlabelled item as a failure.
    """
    labelled = eval_set.labelled_only()
    return wilson(sum(1 for item in labelled.items if item.human_says_correct), len(labelled))
