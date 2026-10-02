"""The report, and the two things it refuses to do.

**It will not print a rate without an interval.** Every percentage in the output carries its
Wilson bounds and its denominator, because "92%" over twenty-five items and "92%" over two
thousand are different claims and look identical.

**It will not print a model grader's estimate without that grader's calibration above it.** The
structure is deliberate: the human's figure first, because it is the only ground truth; then
what each grader would have told you, next to how often that grader agrees with the human. A
reader sees the gap rather than being told about it.

Both rules are enforced in code rather than left as conventions, because a convention survives
exactly as long as nobody is in a hurry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import Calibration, calibrate, estimate_from, human_rate
from .items import LABELS, EvalSet, Run
from .stats import Agreement, Interval, PairedComparison, fleiss_kappa


class UncalibratedGraderError(RuntimeError):
    """Raised when a report would present a model grader's number as a result.

    The whole package exists to make this impossible by accident.
    """


@dataclass(frozen=True)
class Report:
    eval_set: str
    n: int
    #: How many of those items a person actually judged. Below ``n`` whenever the set is run
    #: as a holdout, which is the normal case once a set outgrows what anyone will label.
    n_labelled: int
    human: Interval
    calibrations: tuple[Calibration, ...]
    estimates: dict[str, Interval]
    consensus: Agreement | None = None


def build(eval_set: EvalSet, run: Run, model_graders: frozenset[str] = frozenset()) -> Report:
    """Computes everything the report needs, and refuses to omit the calibration."""
    calibrations = tuple(
        calibrate(eval_set, run, grader, is_model=grader in model_graders) for grader in run.graders
    )
    estimates = {grader: estimate_from(run, grader) for grader in run.graders}

    consensus = None
    if len(run.graders) >= 2:
        # Agreement among the graders themselves, which is a different question from agreement
        # with the human: graders can be consistent with each other and consistently wrong.
        consensus = fleiss_kappa(
            [[run.labels_from(g)[i] for g in run.graders] for i in range(len(eval_set))]
        )

    return Report(
        eval_set=eval_set.name,
        n=len(eval_set),
        n_labelled=eval_set.labelled_count,
        human=human_rate(eval_set),
        calibrations=calibrations,
        estimates=estimates,
        consensus=consensus,
    )


def render(report: Report, eval_set: EvalSet | None = None) -> str:
    """Markdown. Every rate carries its interval; every model grader carries its calibration."""
    uncalibrated = [c.grader for c in report.calibrations if c.is_model and c.kappa.n == 0]
    if uncalibrated:
        raise UncalibratedGraderError(
            f"refusing to render: {uncalibrated} are model graders with no human labels to "
            "measure them against. Add human verdicts to the set, or do not report a number."
        )

    lines = [
        f"# {report.eval_set}",
        "",
        f"{report.n} items. Every rate below carries a 95% Wilson interval and its denominator,",
        "because a percentage without those is not a measurement.",
        "",
        "## What the system actually scored",
        "",
        "From the human labels, which are the only ground truth here. A partial answer counts as",
        "not correct — the strict convention, stated because the other one is the easiest way to",
        "make an evaluation look better than the system is.",
        "",
        f"**{report.human.as_percent()}**",
        "",
    ]

    if report.n_labelled < report.n:
        lines += [
            f"A person judged {report.n_labelled} of the {report.n} items. Everything above rests",
            f"on those {report.n_labelled}; the graders ran on all {report.n}. Read the two",
            "denominators in the table below as the different claims they are — the estimate",
            "column covers the whole set, the agreement column covers only what was labelled.",
            "",
        ]

    if report.human.width > 0.25:
        lines += [
            f"That interval is {report.human.width * 100:.0f} points wide. The set is too small to",
            "distinguish this system from a noticeably better or worse one, and no amount of",
            "reporting discipline fixes that — only more items do.",
            "",
        ]

    lines += [
        "## What each grader would have told you",
        "",
        "| Grader | Its estimate | Agreement with the human | Kappa | Too generous | Too harsh |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for calibration in report.calibrations:
        estimate = report.estimates[calibration.grader]
        marker = "" if calibration.trustworthy else " (!)"
        lines.append(
            f"| `{calibration.grader}`{' (model)' if calibration.is_model else ''} "
            f"| {estimate.as_percent()} "
            f"| {calibration.agreement_with_human.as_percent()} "
            f"| {calibration.kappa.kappa:.2f} ({calibration.kappa.reading}){marker} "
            f"| {calibration.too_generous} | {calibration.too_harsh} |"
        )
    lines.append("")

    weak = [c for c in report.calibrations if not c.trustworthy]
    if weak:
        names = ", ".join(f"`{c.grader}`" for c in weak)
        lines += [
            f"(!) {names} agree with the human less than substantially (kappa below 0.60). Their",
            "estimates above are reported, not suppressed — a reader who can see the calibration",
            "can decide what to do with the number. What they are not is a result.",
            "",
        ]

    generous = [c for c in report.calibrations if c.too_generous > c.too_harsh]
    if generous:
        names = ", ".join(f"`{c.grader}`" for c in generous)
        lines += [
            f"{names} err towards saying yes. That is the dangerous direction: it inflates every",
            "figure computed from them, and the inflation is invisible in the figure itself. A",
            "consistently harsh grader is still usable for tracking a change over time; a",
            "generous one is not usable for anything.",
            "",
        ]

    if report.consensus is not None:
        lines += [
            "## How much the graders agree with each other",
            "",
            report.consensus.describe(),
            "",
            "Worth reading next to the column above and not instead of it: graders can agree with",
            "each other and be wrong together, which is what happens when they share a prompt.",
            "",
        ]

    if eval_set is not None and eval_set.all_tags:
        lines += [
            "## By kind of item",
            "",
            "Slices are where small samples get smaller. The intervals widen accordingly, which is",
            "the point of showing them.",
            "",
            "| Slice | Items | The system, per the humans |",
            "| --- | --- | --- |",
        ]
        for tag in eval_set.all_tags:
            sliced = eval_set.tagged(tag)
            lines.append(f"| {tag} | {len(sliced)} | {human_rate(sliced).as_percent()} |")
        lines.append("")

    disagreements = _disagreement_table(report, eval_set)
    if disagreements:
        lines += ["## Where the graders and the human parted company", "", *disagreements, ""]

    return "\n".join(lines)


def _disagreement_table(report: Report, eval_set: EvalSet | None) -> list[str]:
    """The confusion, written out, because the aggregate hides which way each grader leans."""
    if eval_set is None:
        return []
    rows = ["| Grader | Human said | Grader said | Items |", "| --- | --- | --- | --- |"]
    found = False
    for calibration in report.calibrations:
        for human_label in LABELS:
            for grader_label in LABELS:
                if human_label == grader_label:
                    continue
                count = calibration.confusion[human_label][grader_label]
                if count:
                    found = True
                    rows.append(
                        f"| `{calibration.grader}` | {human_label} | {grader_label} | {count} |"
                    )
    return rows if found else []


def render_comparison(comparison: PairedComparison, before_label: str, after_label: str) -> str:
    """Two runs over the same items, compared as the paired data they are."""
    return "\n".join(
        [
            f"# {before_label} -> {after_label}",
            "",
            "The same items under both, so this is a paired comparison and only the items the two",
            "runs disagreed on carry any information. Comparing the two accuracy figures as if",
            "they were independent samples throws the pairing away, and most of the power with it.",
            "",
            "| | Count |",
            "| --- | --- |",
            f"| Both correct | {comparison.both_correct} |",
            f"| Fixed by the change | {comparison.only_after} |",
            f"| Broken by the change | {comparison.only_before} |",
            f"| Both wrong | {comparison.both_wrong} |",
            "",
            f"**{comparison.describe()}**",
            "",
            "The p-value is an exact binomial over the disagreements, not the chi-squared",
            "approximation: a real evaluation usually has fewer than twenty-five of them, which is",
            "exactly where the approximation stops being trustworthy.",
            "",
            (
                "Not significant does not mean no difference. It means this set is too small to"
                " tell one from chance."
                if not comparison.significant
                else "A difference this size is unlikely to be chance on these items."
            ),
        ]
    )
