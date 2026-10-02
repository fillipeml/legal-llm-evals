"""The arithmetic, which is the part of an evaluation that is easiest to get wrong quietly.

Four things live here, and each exists because of a specific way evaluation reports mislead.

**Wilson intervals**, because "92% accuracy" over 25 items is not a measurement. The naive
interval (p ± 1.96·√(p(1-p)/n)) is the one everyone reaches for, and it is wrong in exactly the
cases evaluations live in: small samples, and rates near 0 or 1, where it happily reports a
bound above 100% or below 0%. Wilson is barely harder and does not do that.

**Cohen's and Fleiss' kappa**, because raw agreement between two graders is inflated by however
much they would have agreed by chance. Two graders who both answer "correct" 90% of the time
agree 82% of the time while knowing nothing about each other's reasoning.

**McNemar's test**, because two prompts evaluated on the same 60 items are *paired*, and
comparing their accuracies as if they were independent samples throws away the pairing and most
of the statistical power with it. What matters is only the items where they disagreed.

**A bootstrap**, for statistics with no closed form — a mean score, a median — where an interval
is still owed.

No dependency on scipy or numpy: everything here is a dozen lines of standard library, and a
harness that is hard to install is a harness nobody runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

#: Two-sided 95%, the default everywhere. Stated once rather than scattered as 1.96.
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate that cannot be printed without its bounds."""

    point: float
    low: float
    high: float
    n: int
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_percent(self, places: int = 1) -> str:
        """``58.3% [45.1, 70.5], n=60`` — the only way this package prints a rate.

        The interval is not decoration. At n=25 it is wide enough that a reader stops
        treating the point estimate as a fact, which is the correct response.
        """
        return (
            f"{self.point * 100:.{places}f}% "
            f"[{self.low * 100:.{places}f}, {self.high * 100:.{places}f}], n={self.n}"
        )

    def overlaps(self, other: Interval) -> bool:
        """True when the two intervals overlap at all.

        Useful for a quick read, and **not** a significance test: non-overlapping intervals
        imply a difference, but overlapping ones do not imply its absence. For a real answer
        on paired data use :func:`mcnemar`.
        """
        return self.low <= other.high and other.low <= self.high


def wilson(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """The Wilson score interval for a proportion.

    Chosen over the normal approximation because evaluation sets are small and rates sit near
    the ends. At 24/25 the naive interval runs past 100%; Wilson stops short of it, which is
    the behaviour a reader needs when the question is "could this be worse than it looks".
    """
    if n < 0:
        raise ValueError("n cannot be negative")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be between 0 and n; got {successes} of {n}")
    if n == 0:
        # No data is not a rate of zero. The caller has to decide what to do about it.
        return Interval(point=float("nan"), low=0.0, high=1.0, n=0, confidence=confidence)

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Interval(
        point=p,
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        n=n,
        confidence=confidence,
    )


def rate(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """Alias for :func:`wilson`, named for what the caller is computing."""
    return wilson(successes, n, confidence)


# --------------------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    """How much two or more graders agree, beyond chance."""

    raw: float
    expected_by_chance: float
    kappa: float
    n: int
    raters: int
    #: An interval for ``raw``, by the method that is valid for this number of raters: Wilson
    #: for two (where raw agreement is a binomial proportion over items) and a percentile
    #: bootstrap for three or more (where it is the mean of per-item agreement proportions and
    #: Wilson would be wrong). The distinction is small and reporting it is the whole job.
    raw_interval: Interval | None = None

    @property
    def reading(self) -> str:
        """Landis and Koch's labels, with the caveat that matters.

        These bands are conventional, widely quoted, and arbitrary — the original paper says
        so. They are here because a reader wants a word, and the word is reported alongside
        the number rather than instead of it.
        """
        k = self.kappa
        if k < 0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"

    def describe(self) -> str:
        """One line, with an interval on the rate and the chance baseline as a proportion.

        ``expected_by_chance`` is written as a decimal rather than a percentage on purpose. It
        is not a measured rate — it is what these graders' marginals imply they would hit while
        sharing no judgement at all — and dressing it as a percentage invites the reader to
        compare it with the one above as though the two were the same kind of number. Kappa is
        that comparison, done properly, and it is the figure that leads the line.
        """
        raw = (
            self.raw_interval.as_percent()
            if self.raw_interval is not None
            else f"{self.raw * 100:.1f}% (n={self.n})"
        )
        return (
            f"kappa {self.kappa:.2f} ({self.reading}) across {self.raters} graders; "
            f"raw agreement {raw}, "
            f"against {self.expected_by_chance:.3f} expected by chance alone"
        )


def cohens_kappa(a: list[str], b: list[str]) -> Agreement:
    """Agreement between exactly two graders over the same items, in order.

    The number worth looking at is not ``raw`` — two graders who each say "correct" nine times
    out of ten agree 82% of the time while sharing no judgement at all. Kappa removes that.
    """
    if len(a) != len(b):
        raise ValueError(f"both graders must label the same items; got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("no items to compare")

    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)

    # Perfect agreement with no variance at all: every item got the same label from both. Kappa
    # is undefined there (0/0), and the honest answer is 1.0 with the caveat visible in `raw`.
    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    return Agreement(
        raw=observed,
        expected_by_chance=expected,
        kappa=kappa,
        n=n,
        raters=2,
        # With two raters each item either matches or does not, so raw agreement is a binomial
        # proportion and Wilson applies directly.
        raw_interval=wilson(round(observed * n), n),
    )


def fleiss_kappa(labels_per_item: list[list[str]]) -> Agreement:
    """Agreement among three or more graders, each labelling every item.

    Fleiss rather than an average of pairwise Cohens: the pairwise average answers a different
    question and is not comparable across different numbers of raters.
    """
    if not labels_per_item:
        raise ValueError("no items to compare")
    raters = len(labels_per_item[0])
    if raters < 2:
        raise ValueError("fleiss_kappa needs at least two raters per item")
    if any(len(row) != raters for row in labels_per_item):
        raise ValueError("every item must be labelled by every rater")

    n = len(labels_per_item)
    categories = sorted({label for row in labels_per_item for label in row})

    # Agreement within each item: the proportion of rater pairs that match.
    per_item = []
    for row in labels_per_item:
        counts = [row.count(c) for c in categories]
        agreeing_pairs = sum(c * (c - 1) for c in counts)
        per_item.append(agreeing_pairs / (raters * (raters - 1)))
    observed = sum(per_item) / n

    # Chance agreement from the overall distribution of each category.
    totals = [sum(row.count(c) for row in labels_per_item) for c in categories]
    proportions = [t / (n * raters) for t in totals]
    expected = sum(p**2 for p in proportions)

    kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    return Agreement(
        raw=observed,
        expected_by_chance=expected,
        kappa=kappa,
        n=n,
        raters=raters,
        # With three or more raters an item is not simply agreed or not: it carries a
        # proportion of agreeing pairs, and `observed` is the mean of those. That is not a
        # binomial proportion, so Wilson would be the wrong interval and a percentile
        # bootstrap over the per-item values is the right one.
        raw_interval=bootstrap_mean(per_item, resamples=2_000),
    )


# --------------------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedComparison:
    """Two systems on the same items. Only the disagreements carry information."""

    both_correct: int
    only_before: int
    only_after: int
    both_wrong: int
    p_value: float
    n: int

    @property
    def discordant(self) -> int:
        return self.only_before + self.only_after

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def describe(self) -> str:
        if self.discordant == 0:
            return f"the two runs agreed on all {self.n} items, so there is nothing to compare"
        if self.only_after > self.only_before:
            direction = "better"
        elif self.only_before > self.only_after:
            direction = "worse"
        else:
            # The ternary this replaces had no equality branch, so five fixes against five
            # breakages was reported, in bold, as "the change looks worse".
            direction = "unchanged on balance"
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.only_after} item(s) fixed, {self.only_before} broken, "
            f"{self.discordant} disagreements out of {self.n}; "
            f"the change looks {direction}, p={self.p_value:.3f} ({verdict} at 0.05)"
        )


def mcnemar(before: list[bool], after: list[bool]) -> PairedComparison:
    """Exact McNemar test on two runs over the same items, in order.

    The exact binomial form rather than the chi-squared approximation, because the number of
    disagreements in a real evaluation is often under twenty-five, which is exactly where the
    approximation stops being trustworthy.

    The items both runs got right, and the items both got wrong, contribute nothing: the whole
    question is whether the fixes outnumber the breakages by more than a coin flip would
    explain. That is why comparing two accuracy figures as if they were independent samples
    throws away most of the power — it ignores the pairing entirely.
    """
    if len(before) != len(after):
        raise ValueError(f"both runs must cover the same items; got {len(before)} and {len(after)}")
    n = len(before)
    if n == 0:
        raise ValueError("no items to compare")

    both_correct = sum(1 for x, y in zip(before, after, strict=True) if x and y)
    only_before = sum(1 for x, y in zip(before, after, strict=True) if x and not y)
    only_after = sum(1 for x, y in zip(before, after, strict=True) if y and not x)
    both_wrong = sum(1 for x, y in zip(before, after, strict=True) if not x and not y)

    discordant = only_before + only_after
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(only_before, only_after)
        # Two-sided exact binomial at p=0.5 over the discordant pairs.
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)

    return PairedComparison(
        both_correct=both_correct,
        only_before=only_before,
        only_after=only_after,
        both_wrong=both_wrong,
        p_value=p_value,
        n=n,
    )


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------


def bootstrap_mean(
    values: list[float],
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = 10_000,
    seed: int = 20260101,
) -> Interval:
    """A percentile bootstrap interval for a mean, for scores with no closed form.

    The seed is fixed and explicit. An evaluation whose reported interval moves between runs
    invites the obvious abuse, and a resampling procedure is exactly the place that can happen
    without anybody intending it.
    """
    if not values:
        raise ValueError("no values to resample")
    if len(values) == 1:
        only = values[0]
        return Interval(point=only, low=only, high=only, n=1, confidence=confidence)

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = (1 - confidence) / 2
    low = means[int(alpha * resamples)]
    high = means[min(resamples - 1, int((1 - alpha) * resamples))]
    return Interval(point=sum(values) / n, low=low, high=high, n=n, confidence=confidence)
