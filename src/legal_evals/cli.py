"""The command line.

``evals demo`` is the one to run first: it grades the shipped set with four graders, two of
them recorded model judges that fail in opposite directions, and prints the report. No key, no
network, and the same numbers every time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import report as reporting
from .calibration import human_rate
from .graders import ExactGrader, Grader, RecordedGrader, RuleGrader
from .items import EvalSet, Run
from .runner import model_grader_names, run_graders
from .stats import mcnemar

DEFAULT_SET = "fixtures/extraction.json"
DEFAULT_GRADERS_DIR = "fixtures/graders"


# The set is Portuguese and the report is Markdown; neither survives a terminal that defaults
# to a single-byte code page, which is what Windows still does. Reconfiguring here rather than
# asking the user to is the difference between a tool that runs and one that raises on them.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def out(line: str = "") -> None:
    print(line, flush=True)


def load_graders(spec: str | None, graders_dir: str = DEFAULT_GRADERS_DIR) -> list[Grader]:
    """``exact,rules,recorded:judge-strict`` — or everything available, which is the default."""
    deterministic: dict[str, Grader] = {"exact": ExactGrader(), "rules": RuleGrader()}

    if spec is None:
        graders: list[Grader] = list(deterministic.values())
        for path in sorted(Path(graders_dir).glob("*.json")):
            graders.append(RecordedGrader.load(path))
        return graders

    chosen: list[Grader] = []
    for name in (part.strip() for part in spec.split(",") if part.strip()):
        if name in deterministic:
            chosen.append(deterministic[name])
        elif name.startswith("recorded:"):
            chosen.append(RecordedGrader.load(Path(graders_dir) / f"{name[9:]}.json"))
        else:
            raise SystemExit(f"unknown grader {name!r}. Try exact, rules, or recorded:<name>.")
    return chosen


def cmd_run(args: argparse.Namespace) -> int:
    eval_set = EvalSet.load(args.set)
    graders = load_graders(args.graders, args.graders_dir)

    result = run_graders(eval_set, graders, label=args.label or "run")
    for grader, item_id, error in result.failures:
        out(f"! {grader} could not grade {item_id}: {error}")

    if args.out:
        result.run.save(args.out)
        out(f"wrote {args.out}")

    out(
        reporting.render(
            reporting.build(eval_set, result.run, model_grader_names(graders)), eval_set
        )
    )
    return 0 if result.ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    eval_set = EvalSet.load(args.set)
    run = Run.load(args.run)
    model_graders = frozenset(args.model_graders.split(",")) if args.model_graders else frozenset()
    out(reporting.render(reporting.build(eval_set, run, model_graders), eval_set))
    return 0


def _correct_flags(path: str, grader: str) -> tuple[tuple[str, ...], list[bool], str]:
    """The per-item correctness of one side of a comparison.

    ``human`` reads an evaluation set and uses its human labels, which is the ground truth and
    therefore the default. Any other name reads a saved run and uses that grader's labels —
    useful, and only as trustworthy as the calibration says that grader is.
    """
    if grader == "human":
        eval_set = EvalSet.load(path)
        if not eval_set.fully_labelled:
            # A paired comparison needs a verdict on both sides of every item it counts.
            # Silently dropping the unlabelled ones would change the denominator between the
            # two sides and quietly stop the comparison being paired at all.
            raise SystemExit(
                f"refusing to compare: {eval_set.labelled_count} of {len(eval_set)} items in "
                f"{path} carry a human verdict. Pair on a fully labelled set, or compare a "
                "grader's labels with --grader <name> and read its calibration first."
            )
        return (
            eval_set.ids,
            [item.human_says_correct for item in eval_set.items],
            eval_set.name,
        )
    run = Run.load(path)
    if grader not in run.graders:
        raise SystemExit(f"grader {grader!r} is not in {path}")
    return (
        run.item_ids,
        [label == "correct" for label in run.labels_from(grader)],
        run.label or path,
    )


def cmd_compare(args: argparse.Namespace) -> int:
    """Two versions of a system over the same items, compared as the paired data they are."""
    before_ids, before, before_label = _correct_flags(args.before, args.grader)
    after_ids, after, after_label = _correct_flags(args.after, args.grader)

    if before_ids != after_ids:
        raise SystemExit(
            "refusing to compare: the two sides cover different items, so the comparison would "
            "not be paired and the number would be meaningless. Evaluate both over the same set."
        )

    comparison = mcnemar(before, after)
    out(reporting.render_comparison(comparison, before_label, after_label))

    if comparison.discordant == 0 and args.grader != "human":
        out("")
        out(
            f"Nothing moved, which is worth a second look: `{args.grader}` may be blind to "
            "whatever changed. Compare the human labels (--grader human) before concluding "
            "that the change did nothing."
        )
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    """What is in the set, before any grading happens."""
    eval_set = EvalSet.load(args.set)
    out(f"{eval_set.name} — {len(eval_set)} items")
    out(eval_set.description)
    out()
    if eval_set.fully_labelled:
        out(f"all {len(eval_set)} carry a human verdict")
    else:
        out(
            f"{eval_set.labelled_count} of {len(eval_set)} carry a human verdict; "
            "the rest can be graded but not measured"
        )
    out(f"the system, per the humans: {human_rate(eval_set).as_percent()}")
    out()
    out("| Slice | Items | Per the humans |")
    out("| --- | --- | --- |")
    for tag in eval_set.all_tags:
        sliced = eval_set.tagged(tag)
        out(f"| {tag} | {len(sliced)} | {human_rate(sliced).as_percent()} |")
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    """Everything, offline, on the shipped set."""
    out("DEMO — the shipped set, four graders, no key and no network call.")
    out()
    eval_set = EvalSet.load(DEFAULT_SET)
    graders = load_graders(None)
    result = run_graders(eval_set, graders, label="demo")
    out(
        reporting.render(
            reporting.build(eval_set, result.run, model_grader_names(graders)), eval_set
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evals",
        description="An evaluation harness that will not print a rate without an interval, "
        "nor a model grader's number without that grader's calibration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def with_set(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--set", default=DEFAULT_SET)
        return p

    p = with_set(sub.add_parser("demo", help="the whole thing offline on the shipped set"))
    p.set_defaults(func=cmd_demo)

    p = with_set(sub.add_parser("describe", help="what is in the set, before grading"))
    p.set_defaults(func=cmd_describe)

    p = with_set(sub.add_parser("run", help="grade a set and report"))
    p.add_argument("--graders", help="exact,rules,recorded:<name>; default is all available")
    p.add_argument("--graders-dir", default=DEFAULT_GRADERS_DIR)
    p.add_argument("--out", help="write the run to this path")
    p.add_argument("--label", help="a name for this run, shown in comparisons")
    p.set_defaults(func=cmd_run)

    p = with_set(sub.add_parser("report", help="report on a saved run"))
    p.add_argument("--run", required=True)
    p.add_argument("--model-graders", help="comma-separated names that are language models")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="two runs over the same items, paired")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument(
        "--grader",
        default="human",
        help="'human' compares two evaluation sets by their human labels (the default, and the "
        "only ground truth); any other name compares two saved runs by that grader",
    )
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, reporting.UncalibratedGraderError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
