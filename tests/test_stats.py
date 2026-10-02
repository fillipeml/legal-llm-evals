"""The arithmetic, checked against values worked out by hand.

Every expected number in this file was computed independently of the implementation — from the
published formula, or from a textbook example — because a test that asserts whatever the code
currently returns tests nothing at all.
"""

from __future__ import annotations

import math

import pytest

from legal_evals.stats import (
    bootstrap_mean,
    cohens_kappa,
    fleiss_kappa,
    mcnemar,
    wilson,
)


class TestWilson:
    def test_a_rate_near_one_does_not_exceed_one(self) -> None:
        # The reason Wilson rather than the normal approximation: at 24/25 the naive interval
        # is 0.96 ± 1.96·√(0.96·0.04/25) = [0.883, 1.037], and a probability of 1.037 is not a
        # thing. Hand-computed Wilson: centre 0.89873, margin 0.09418.
        interval = wilson(24, 25)
        assert interval.point == pytest.approx(0.96)
        assert interval.low == pytest.approx(0.80455, abs=1e-4)
        assert interval.high == pytest.approx(0.99291, abs=1e-4)
        assert interval.high < 1.0

    def test_a_rate_of_zero_does_not_go_below_zero(self) -> None:
        interval = wilson(0, 20)
        assert interval.point == 0.0
        assert interval.low == 0.0
        assert interval.high > 0.0  # zero successes is not proof of a zero rate

    def test_a_perfect_score_still_carries_doubt(self) -> None:
        # The most important case for an evaluation report: 20/20 is not 100%, it is "we did
        # not see a failure in twenty tries", and the lower bound says how little that proves.
        interval = wilson(20, 20)
        assert interval.point == 1.0
        assert interval.low < 0.9
        assert interval.high == pytest.approx(1.0)

    def test_the_interval_narrows_as_the_sample_grows(self) -> None:
        small = wilson(9, 10)
        large = wilson(90, 100)
        assert small.point == pytest.approx(large.point)
        assert small.width > large.width * 2

    def test_half_of_a_hundred_is_centred_on_a_half(self) -> None:
        # Hand-computed: centre 0.5, margin 1.96/1.0384 · √(0.0025 + 0.0000096) ≈ 0.0950.
        interval = wilson(50, 100)
        assert interval.low == pytest.approx(0.4038, abs=1e-3)
        assert interval.high == pytest.approx(0.5962, abs=1e-3)

    def test_no_data_is_not_a_rate_of_zero(self) -> None:
        interval = wilson(0, 0)
        assert math.isnan(interval.point)
        assert (interval.low, interval.high) == (0.0, 1.0)

    def test_impossible_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError):
            wilson(11, 10)
        with pytest.raises(ValueError):
            wilson(-1, 10)

    def test_a_rate_is_never_printed_without_its_bounds(self) -> None:
        # The format is the point of the type: there is no way to render a bare percentage.
        rendered = wilson(35, 60).as_percent()
        assert "58.3%" in rendered
        assert "[" in rendered and "]" in rendered
        assert "n=60" in rendered

    def test_overlap_is_offered_but_not_as_a_test(self) -> None:
        assert wilson(50, 100).overlaps(wilson(55, 100))
        assert not wilson(10, 100).overlaps(wilson(90, 100))


class TestCohensKappa:
    def test_chance_agreement_scores_zero(self) -> None:
        # Two graders agreeing on exactly half, with balanced marginals: observed 0.5,
        # expected 0.5, so kappa is 0 — they agree exactly as much as coin flips would.
        result = cohens_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"])
        assert result.raw == pytest.approx(0.5)
        assert result.expected_by_chance == pytest.approx(0.5)
        assert result.kappa == pytest.approx(0.0)

    def test_high_raw_agreement_can_still_be_worthless(self) -> None:
        # The case the whole statistic exists for. Both graders say "correct" nine times in ten
        # and agree on 82% of items, while sharing no judgement whatever.
        a = ["correct"] * 9 + ["wrong"]
        b = ["correct"] * 8 + ["wrong"] + ["correct"]
        result = cohens_kappa(a, b)
        assert result.raw == pytest.approx(0.8)
        assert result.kappa < 0.2
        assert result.reading in {"slight", "fair", "worse than chance"}

    def test_perfect_agreement_scores_one(self) -> None:
        result = cohens_kappa(["a", "b", "c"], ["a", "b", "c"])
        assert result.kappa == pytest.approx(1.0)

    def test_total_disagreement_scores_below_zero(self) -> None:
        result = cohens_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"])
        assert result.kappa < 0
        assert result.reading == "worse than chance"

    def test_one_label_for_everything_is_reported_as_perfect_with_the_caveat_visible(self) -> None:
        # Kappa is undefined when there is no variance. Returning 1.0 is the least misleading
        # answer, and `raw` and `expected_by_chance` both being 1.0 is the tell.
        result = cohens_kappa(["ok"] * 5, ["ok"] * 5)
        assert result.kappa == 1.0
        assert result.expected_by_chance == pytest.approx(1.0)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same items"):
            cohens_kappa(["a"], ["a", "b"])


class TestFleissKappa:
    def test_unanimous_raters_score_one(self) -> None:
        items = [["a", "a", "a"], ["b", "b", "b"], ["a", "a", "a"]]
        assert fleiss_kappa(items).kappa == pytest.approx(1.0)

    def test_evenly_split_raters_score_near_zero(self) -> None:
        # Four items, two raters each way: per-item agreement is 1/3, and with balanced
        # marginals chance agreement is 0.5, so kappa is negative.
        items = [["a", "a", "b", "b"], ["b", "b", "a", "a"]] * 2
        result = fleiss_kappa(items)
        assert result.raw == pytest.approx(1 / 3)
        assert result.kappa < 0

    def test_it_reports_how_many_raters_it_used(self) -> None:
        result = fleiss_kappa([["a", "a", "b"]] * 4)
        assert result.raters == 3
        assert result.n == 4

    def test_a_rater_who_skipped_an_item_is_refused(self) -> None:
        with pytest.raises(ValueError, match="every rater"):
            fleiss_kappa([["a", "b"], ["a"]])

    def test_one_rater_is_not_agreement(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            fleiss_kappa([["a"], ["b"]])


class TestMcNemar:
    def test_only_the_disagreements_count(self) -> None:
        # Nine fixed, one broken, out of forty. Two-sided exact binomial over the ten
        # discordant pairs: 2·(C(10,0)+C(10,1))/2^10 = 22/1024 = 0.021484.
        #        20 both right | 9 the change fixed | 1 it broke | 10 both wrong
        before = [True] * 20 + [False] * 9 + [True] * 1 + [False] * 10
        after = [True] * 20 + [True] * 9 + [False] * 1 + [False] * 10
        result = mcnemar(before, after)
        assert result.discordant == 10
        assert result.p_value == pytest.approx(22 / 1024, abs=1e-6)
        assert result.significant

    def test_a_small_improvement_is_not_significant(self) -> None:
        # Three fixed, two broken: 2·(C(5,0)+C(5,1)+C(5,2))/2^5 = 2·16/32 = 1.0.
        #        10 both right | 3 fixed | 2 broken
        before = [True] * 10 + [False] * 3 + [True] * 2
        after = [True] * 10 + [True] * 3 + [False] * 2
        result = mcnemar(before, after)
        assert result.discordant == 5
        assert result.p_value == pytest.approx(1.0)
        assert not result.significant

    def test_two_identical_runs_have_nothing_to_compare(self) -> None:
        run = [True, False, True, True]
        result = mcnemar(run, run)
        assert result.discordant == 0
        assert result.p_value == 1.0
        assert "nothing to compare" in result.describe()

    def test_the_description_names_the_direction(self) -> None:
        before = [False] * 10 + [True] * 10
        after = [True] * 10 + [True] * 10
        assert "better" in mcnemar(before, after).describe()
        assert "worse" in mcnemar(after, before).describe()

    def test_mismatched_runs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same items"):
            mcnemar([True], [True, False])


class TestBootstrap:
    def test_it_brackets_the_mean(self) -> None:
        values = [float(v) for v in range(1, 21)]
        interval = bootstrap_mean(values, resamples=2000)
        assert interval.point == pytest.approx(10.5)
        assert interval.low < 10.5 < interval.high

    def test_it_is_reproducible(self) -> None:
        # An interval that moves between runs invites the obvious abuse, so the seed is fixed.
        values = [1.0, 4.0, 9.0, 2.0, 7.0]
        first = bootstrap_mean(values, resamples=1000)
        second = bootstrap_mean(values, resamples=1000)
        assert (first.low, first.high) == (second.low, second.high)

    def test_identical_values_give_a_zero_width_interval(self) -> None:
        interval = bootstrap_mean([5.0] * 10, resamples=500)
        assert interval.width == pytest.approx(0.0)

    def test_a_single_value_is_not_an_estimate(self) -> None:
        interval = bootstrap_mean([3.0])
        assert (interval.low, interval.point, interval.high) == (3.0, 3.0, 3.0)
        assert interval.n == 1

    def test_no_values_is_refused(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_mean([])


def test_an_even_trade_is_not_reported_as_worse() -> None:
    """Equal fixes and breakages is not a regression, and the word is printed in bold.

    The ternary had no equality branch, so `better if after > before else "worse"` called an
    even trade worse. It is the one case where the reader most needs the harness to be exact,
    because the counts are the same and only the label tells them which way to read it.
    """
    before = [True] * 5 + [False] * 5 + [True] * 10
    after = [False] * 5 + [True] * 5 + [True] * 10
    result = mcnemar(before, after)
    assert result.only_before == result.only_after == 5
    assert "unchanged on balance" in result.describe()
    assert "worse" not in result.describe()
