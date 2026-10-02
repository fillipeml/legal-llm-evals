"""The command line, and the holdout it exists to report honestly.

Every test here runs the real commands against the shipped fixtures. There is no network call
and no key anywhere in this file, which is the property that lets the whole evaluation run in
CI on every push.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_evals.calibration import calibrate, estimate_from, human_rate
from legal_evals.cli import main
from legal_evals.graders import RecordedGrader, RuleGrader
from legal_evals.items import EvalSet
from legal_evals.report import UncalibratedGraderError, build, render
from legal_evals.runner import run_graders

SET = "fixtures/extraction.json"
V2 = "fixtures/extraction-v2.json"
HOLDOUT = "fixtures/extraction-holdout.json"


class TestHoldout:
    """A set where twenty of sixty items carry a human verdict — the normal case."""

    def test_only_the_labelled_items_count_as_measured(self) -> None:
        eval_set = EvalSet.load(HOLDOUT)
        assert len(eval_set) == 60
        assert eval_set.labelled_count == 20
        assert not eval_set.fully_labelled
        # Not 20/60. An item nobody judged is not a failure.
        assert human_rate(eval_set).n == 20

    def test_the_estimate_covers_the_set_and_the_calibration_covers_the_holdout(self) -> None:
        eval_set = EvalSet.load(HOLDOUT)
        grader = RecordedGrader.load("fixtures/graders/judge-strict.json")
        result = run_graders(eval_set, [grader])

        assert estimate_from(result.run, "judge-strict").n == 60
        assert calibrate(eval_set, result.run, "judge-strict", is_model=True).kappa.n == 20

    def test_the_report_states_both_denominators(self) -> None:
        eval_set = EvalSet.load(HOLDOUT)
        result = run_graders(eval_set, [RecordedGrader.load("fixtures/graders/judge-strict.json")])
        rendered = render(build(eval_set, result.run, frozenset({"judge-strict"})), eval_set)
        assert "A person judged 20 of the 60 items" in rendered
        assert "n=60" in rendered and "n=20" in rendered

    def test_an_unlabelled_item_refuses_to_answer_the_strict_question(self) -> None:
        eval_set = EvalSet.load(HOLDOUT)
        unlabelled = next(i for i in eval_set.items if not i.labelled)
        # Returning False here would count "nobody looked" as "the system got it wrong", which
        # is the quiet version of the mistake this package is about.
        with pytest.raises(ValueError, match="nothing to be strict about"):
            _ = unlabelled.human_says_correct

    def test_a_model_grader_with_nothing_labelled_is_refused(self) -> None:
        # The realistic shape of the refusal: a full set, a model judge, and nobody who
        # checked any of it.
        eval_set = EvalSet.load(HOLDOUT)
        nothing_labelled = EvalSet(
            name=eval_set.name,
            description=eval_set.description,
            items=tuple(replace(item, human=None) for item in eval_set.items),
        )
        result = run_graders(
            nothing_labelled, [RecordedGrader.load("fixtures/graders/judge-strict.json")]
        )
        report = build(nothing_labelled, result.run, frozenset({"judge-strict"}))
        with pytest.raises(UncalibratedGraderError, match="no human labels"):
            render(report)


class TestCommands:
    def test_demo_runs_offline_and_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["demo"]) == 0
        printed = capsys.readouterr().out
        assert "judgment-field-extraction" in printed
        assert "What the system actually scored" in printed

    def test_describe_names_the_holdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["describe", "--set", HOLDOUT]) == 0
        assert "20 of 60 carry a human verdict" in capsys.readouterr().out

    def test_describe_says_so_when_everything_is_labelled(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["describe", "--set", SET]) == 0
        assert "all 60 carry a human verdict" in capsys.readouterr().out

    def test_run_writes_a_run_that_report_can_read_back(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "run.json"
        assert main(["run", "--set", SET, "--graders", "rules", "--out", str(out)]) == 0
        capsys.readouterr()

        assert main(["report", "--set", SET, "--run", str(out)]) == 0
        assert "What each grader would have told you" in capsys.readouterr().out

    def test_an_unknown_grader_is_named_rather_than_ignored(self) -> None:
        with pytest.raises(SystemExit, match="unknown grader"):
            main(["run", "--set", SET, "--graders", "gpt-vibes"])

    def test_compare_finds_the_improvement_in_the_human_labels(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["compare", "--before", SET, "--after", V2]) == 0
        printed = capsys.readouterr().out
        assert "Fixed by the change" in printed
        assert "significant" in printed

    def test_compare_refuses_two_sets_that_are_not_the_same_items(self, tmp_path: Path) -> None:
        small = tmp_path / "small.json"
        small.write_text(
            Path(SET).read_text(encoding="utf-8").replace('"items": [', '"items": [', 1),
            encoding="utf-8",
        )
        import json

        payload = json.loads(small.read_text(encoding="utf-8"))
        payload["items"] = payload["items"][:10]
        small.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(SystemExit, match="not be paired"):
            main(["compare", "--before", SET, "--after", str(small)])

    def test_compare_refuses_a_partially_labelled_set(self) -> None:
        # Dropping the unlabelled items would change the denominator on one side only, which
        # is precisely how a paired comparison stops being paired.
        with pytest.raises(SystemExit, match="refusing to compare"):
            main(["compare", "--before", HOLDOUT, "--after", V2])

    def test_a_grader_blind_to_the_change_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The rules grader only looks at figures, and the v2 improvement was mostly prose. A
        # harness that reported "no change" and stopped there would be worse than useless.
        before, after = tmp_path / "b.json", tmp_path / "a.json"
        for path, source in ((before, SET), (after, V2)):
            eval_set = EvalSet.load(source)
            run_graders(eval_set, [RuleGrader()], label=path.stem).run.save(path)

        assert (
            main(["compare", "--before", str(before), "--after", str(after), "--grader", "rules"])
            == 0
        )
        printed = capsys.readouterr().out
        assert "may be blind to whatever changed" in printed

    def test_the_demo_is_reproducible(self, capsys: pytest.CaptureFixture[str]) -> None:
        # An evaluation whose number moves when nobody changed anything is not a measurement.
        # The bootstrap in the consensus figure is the part most likely to drift, so it is
        # seeded, and this is the test that keeps it seeded.
        main(["demo"])
        first = capsys.readouterr().out
        main(["demo"])
        assert capsys.readouterr().out == first
