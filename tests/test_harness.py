"""The harness: items, graders, calibration, and the two things the report refuses to do.

The tests that matter most here are the refusals. A measurement tool whose guard rails are
conventions has no guard rails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_evals.calibration import calibrate, estimate_from, human_rate
from legal_evals.graders import ExactGrader, RecordedGrader, RuleGrader, numbers_in
from legal_evals.items import EvalSet, Item, Run, Verdict
from legal_evals.report import UncalibratedGraderError, build, render
from legal_evals.runner import model_grader_names, run_graders

SET = "fixtures/extraction.json"
GRADERS = Path("fixtures/graders")


def item(**over: object) -> Item:
    base = {
        "id": "x1",
        "question": "q",
        "source": "s",
        "gold": "Retenção: 10%. Índice: INPC.",
        "candidate": "Retenção: 10%. Índice: INPC.",
        "human": "correct",
    }
    base.update(over)
    return Item(**base)  # type: ignore[arg-type]


class TestItem:
    def test_a_partial_answer_is_not_a_correct_one(self) -> None:
        # The strict convention, pinned. Counting partials as successes is the easiest way to
        # make an evaluation look better than the system is, and it is usually accidental.
        assert item(human="correct").human_says_correct
        assert not item(human="partial").human_says_correct
        assert not item(human="incorrect").human_says_correct

    def test_a_label_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            item(human="mostly fine")

    def test_an_item_without_an_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs an id"):
            item(id="")


class TestEvalSet:
    def test_the_shipped_set_loads(self) -> None:
        eval_set = EvalSet.load(SET)
        assert len(eval_set) == 60
        assert eval_set.all_tags

    def test_duplicate_ids_are_refused(self, tmp_path: Path) -> None:
        # A duplicate silently double-counts, and breaks the pairing a comparison depends on.
        payload = {
            "name": "dupes",
            "items": [
                {
                    "id": "a",
                    "question": "q",
                    "source": "s",
                    "gold": "g",
                    "candidate": "c",
                    "human": "correct",
                },
                {
                    "id": "a",
                    "question": "q",
                    "source": "s",
                    "gold": "g",
                    "candidate": "c",
                    "human": "correct",
                },
            ],
        }
        path = tmp_path / "dupes.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate item ids"):
            EvalSet.load(path)

    def test_a_slice_keeps_only_its_tag(self) -> None:
        eval_set = EvalSet.load(SET)
        sliced = eval_set.tagged("hedged")
        assert 0 < len(sliced) < len(eval_set)
        assert all("hedged" in i.tags for i in sliced.items)

    def test_the_shipped_set_is_fictional_by_construction(self) -> None:
        # Every case number is dated 2099, which no real case can be.
        eval_set = EvalSet.load(SET)
        for row in eval_set.items:
            assert ".2099." in row.source, row.id

    def test_no_case_number_could_verify_as_a_real_one(self) -> None:
        """A CNJ number verifies by ISO 7064 mod 97-10. None of these do.

        The first version of the generator picked its check digits from an arithmetic
        sequence and one number in sixty landed on the correct value by accident, which a
        scanner duly flagged. "Obviously fake" is not a property you get to assume.
        """
        import re

        eval_set = EvalSet.load(SET)
        pattern = re.compile(r"(\d{7})-(\d{2})\.(\d{4})\.(\d)\.(\d{2})\.(\d{4})")
        found = 0
        for row in eval_set.items:
            for sequential, check, year, segment, court, origin in pattern.findall(row.source):
                found += 1
                base = int(sequential + year + segment + court + origin)
                assert int(check) != 98 - (base * 100) % 97, (
                    f"{row.id} carries a case number whose check digits verify"
                )
        assert found == len(eval_set)


class TestGraders:
    def test_numbers_survive_brazilian_decimal_notation(self) -> None:
        # A full stop groups thousands and a comma separates the decimal, which is the opposite
        # of what most parsers assume. Getting it wrong is a thousandfold error.
        assert numbers_in("R$ 1.000,50") == ["1000.5"]
        assert numbers_in("retenção de 10%") == ["10%"]
        assert numbers_in("10% e 1.234,00") == ["10%", "1234"]

    def test_the_rule_grader_accepts_a_paraphrase_that_keeps_every_figure(self) -> None:
        verdict = RuleGrader().grade(
            item(candidate="O pedido foi julgado com retenção de 10%, corrigida pelo INPC.")
        )
        assert verdict.label == "correct"

    def test_the_rule_grader_rejects_a_changed_figure(self) -> None:
        verdict = RuleGrader().grade(item(candidate="Retenção: 20%. Índice: INPC."))
        assert verdict.label == "incorrect"
        assert "10%" in verdict.rationale

    def test_an_invented_figure_is_partial_rather_than_wrong(self) -> None:
        # It has contradicted nothing and asserted something nobody checked. Collapsing the two
        # into "incorrect" loses a distinction a reviewer needs.
        verdict = RuleGrader().grade(item(candidate="Retenção: 10%. Índice: INPC. Multa: 2%."))
        assert verdict.label == "partial"
        assert "2%" in verdict.rationale

    def test_the_exact_grader_forgives_accents_and_case_and_nothing_else(self) -> None:
        # If this grader scores well, the task did not need a model and the report should say
        # so. Accents and case are forgiven because a missing cedilla is not a wrong answer in
        # Portuguese; any change to the wording is not forgiven, which is what makes it a floor.
        assert ExactGrader().grade(item()).label == "correct"
        assert (
            ExactGrader().grade(item(candidate="retencao: 10%. indice: inpc.")).label == "correct"
        )
        assert (
            ExactGrader().grade(item(candidate="A retenção é de 10%, pelo INPC.")).label
            == "incorrect"
        )

    def test_a_recorded_grader_never_guesses(self) -> None:
        grader = RecordedGrader.load(GRADERS / "judge-strict.json")
        with pytest.raises(KeyError, match="never guesses"):
            grader.grade(item(id="not-in-the-recording"))

    def test_a_run_round_trips_into_a_replayable_recording(self, tmp_path: Path) -> None:
        # The documented workflow: grade once with a live judge, record, replay from then on.
        # If this breaks, every evaluation that depends on it silently starts calling a model.
        eval_set = EvalSet.load(SET)
        original = RecordedGrader.load(GRADERS / "judge-strict.json")
        run = run_graders(eval_set, [original]).run

        path = tmp_path / "replay.json"
        RecordedGrader.from_run(run, "judge-strict").save(path)
        replayed = RecordedGrader.load(path)

        assert replayed.is_model
        assert [replayed.grade(i).label for i in eval_set.items] == [
            original.grade(i).label for i in eval_set.items
        ]

    def test_recording_a_grader_that_did_not_run_is_refused(self) -> None:
        run = run_graders(EvalSet.load(SET), [RuleGrader()]).run
        with pytest.raises(KeyError, match="no verdicts from"):
            RecordedGrader.from_run(run, "judge-strict")

    def test_only_model_graders_are_flagged_as_models(self) -> None:
        graders = [ExactGrader(), RuleGrader(), RecordedGrader.load(GRADERS / "judge-strict.json")]
        assert model_grader_names(graders) == frozenset({"judge-strict"})


class TestRunner:
    def test_it_grades_every_item_with_every_grader(self) -> None:
        eval_set = EvalSet.load(SET)
        result = run_graders(eval_set, [ExactGrader(), RuleGrader()])
        assert result.ok
        assert set(result.run.graders) == {"exact", "rules"}
        assert len(result.run.labels_from("rules")) == 60

    def test_one_bad_item_does_not_lose_the_run(self) -> None:
        class Flaky:
            name = "flaky"
            is_model = False

            def grade(self, row: Item) -> Verdict:
                if row.id.endswith("007"):
                    raise RuntimeError("deliberate")
                return Verdict(item_id=row.id, grader="flaky", label="correct")

        result = run_graders(EvalSet.load(SET), [Flaky()])
        assert not result.ok
        assert len(result.failures) == 1
        # And the incompleteness cannot then be quietly averaged away.
        with pytest.raises(ValueError, match="did not label"):
            result.run.labels_from("flaky")


class TestCalibration:
    def test_it_measures_the_grader_against_the_human(self) -> None:
        eval_set = EvalSet.load(SET)
        result = run_graders(eval_set, [RecordedGrader.load(GRADERS / "judge-strict.json")])
        calibration = calibrate(eval_set, result.run, "judge-strict", is_model=True)
        assert calibration.agreement_with_human.n == 60
        assert 0 < calibration.kappa.kappa <= 1

    def test_a_generous_grader_is_identified_as_generous(self) -> None:
        # The dangerous direction: it inflates every figure computed from it, invisibly.
        eval_set = EvalSet.load(SET)
        result = run_graders(eval_set, [RecordedGrader.load(GRADERS / "judge-lenient.json")])
        calibration = calibrate(eval_set, result.run, "judge-lenient", is_model=True)
        assert calibration.too_generous > calibration.too_harsh
        assert not calibration.trustworthy

    def test_a_generous_grader_overstates_the_system(self) -> None:
        eval_set = EvalSet.load(SET)
        result = run_graders(eval_set, [RecordedGrader.load(GRADERS / "judge-lenient.json")])
        truth = human_rate(eval_set)
        estimate = estimate_from(result.run, "judge-lenient")
        # The whole argument of the package, as an assertion.
        assert estimate.point > truth.point + 0.15

    def test_the_trustworthy_threshold_is_the_stated_one(self) -> None:
        eval_set = EvalSet.load(SET)
        result = run_graders(eval_set, [RecordedGrader.load(GRADERS / "judge-strict.json")])
        calibration = calibrate(eval_set, result.run, "judge-strict", is_model=True)
        assert calibration.trustworthy == (calibration.kappa.kappa >= 0.6)


class TestReport:
    def test_no_rate_is_printed_without_its_interval(self) -> None:
        eval_set = EvalSet.load(SET)
        graders = [ExactGrader(), RuleGrader(), RecordedGrader.load(GRADERS / "judge-lenient.json")]
        result = run_graders(eval_set, graders)
        rendered = render(build(eval_set, result.run, model_grader_names(graders)), eval_set)

        # Every percentage in the output is immediately followed by its bounds. The scan is
        # over the whole document rather than the tables alone, because the prose is exactly
        # where an uncaveated number gets in.
        import re

        for match in re.finditer(r"\d+\.\d%", rendered):
            tail = rendered[match.end() : match.end() + 3]
            assert tail.startswith(" ["), (
                f"a bare percentage at {match.group(0)} in: "
                f"{rendered[max(0, match.start() - 60) : match.end() + 30]}"
            )

    def test_the_human_figure_comes_before_any_grader_estimate(self) -> None:
        eval_set = EvalSet.load(SET)
        result = run_graders(eval_set, [RuleGrader()])
        rendered = render(build(eval_set, result.run), eval_set)
        assert rendered.index("What the system actually scored") < rendered.index(
            "What each grader would have told you"
        )

    def test_a_weak_grader_is_warned_about_rather_than_hidden(self) -> None:
        eval_set = EvalSet.load(SET)
        graders = [RecordedGrader.load(GRADERS / "judge-lenient.json")]
        result = run_graders(eval_set, graders)
        rendered = render(build(eval_set, result.run, model_grader_names(graders)), eval_set)
        assert "judge-lenient" in rendered
        assert "less than substantially" in rendered
        assert "err towards saying yes" in rendered

    def test_a_model_grader_with_nothing_to_calibrate_against_is_refused(self) -> None:
        # The refusal the whole package exists for.
        empty = EvalSet(name="empty", description="", items=())
        run = Run(eval_set="empty", item_ids=())
        run.verdicts["judge"] = []
        report = build(empty, run, frozenset({"judge"}))
        with pytest.raises(UncalibratedGraderError, match="no human labels"):
            render(report)

    def test_a_wide_interval_says_the_set_is_too_small(self) -> None:
        small = EvalSet(
            name="tiny",
            description="",
            items=tuple(
                item(id=f"i{n}", human="correct" if n % 2 else "incorrect") for n in range(6)
            ),
        )
        result = run_graders(small, [RuleGrader()])
        rendered = render(build(small, result.run), small)
        assert "too small" in rendered
