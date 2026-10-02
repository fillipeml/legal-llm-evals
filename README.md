# legal-llm-evals

An evaluation harness for LLM systems that **refuses to print a success rate without a
confidence interval, and refuses to print a model grader's score without that grader's
calibration against human labels.**

Both rules are enforced in code. A convention survives exactly as long as nobody is in a hurry.

The worked example is extraction from Brazilian court decisions — pull the outcome, the
retention percentage, the correction index and the fee from a judgment — but nothing in the
harness is specific to law.

---

## The problem, in one table

Four graders judge the same sixty items. A person has also judged all sixty. Here is what each
grader would have told you about the system:

| Grader | Its estimate | Agreement with the human | Kappa | Too generous | Too harsh |
| --- | --- | --- | --- | --- | --- |
| `exact` | 23.3% [14.4, 35.4], n=60 | 36.7% [25.6, 49.3], n=60 | 0.18 (slight) (!) | 0 | 18 |
| `judge-lenient` (model) | 85.0% [73.9, 91.9], n=60 | 61.7% [49.0, 72.9], n=60 | 0.25 (fair) (!) | 20 | 1 |
| `judge-strict` (model) | 45.0% [33.1, 57.5], n=60 | 88.3% [77.8, 94.2], n=60 | 0.81 (almost perfect) | 0 | 5 |
| `rules` | 81.7% [70.1, 89.4], n=60 | 68.3% [55.8, 78.7], n=60 | 0.39 (fair) (!) | 18 | 1 |

The humans score the system at **53.3% [40.9, 65.4], n=60**.

`judge-lenient` would have reported **85%**. It is not a broken grader — it agrees with the
human on 62% of items, it produces a fluent rationale for every verdict, and its output is
indistinguishable in shape from `judge-strict`, which is trustworthy. The only thing separating
a 32-point overstatement from a usable number is the calibration column, and that column is
absent from most evaluation harnesses.

That is the entire argument. Everything else here is the machinery for making that column
impossible to leave out.

```bash
pip install -e .
evals demo        # the table above, with no API key and no network call
```

---

## What it enforces

**No rate without an interval.** Every percentage the harness prints carries 95% Wilson bounds
and its denominator. `Interval.as_percent()` is the only code path that renders a rate, so
there is no way to print a bare one by accident. 92% over twenty-five items and 92% over two
thousand are different claims that look identical, and a test walks the rendered report
asserting that no percentage escapes without its bounds.

Wilson rather than the normal approximation because evaluation sets are small and rates sit near
the ends — at 24/25 the naive interval runs past 100%.

**No model grader without its calibration.** `render()` raises `UncalibratedGraderError` if a
grader marked `is_model` has no human labels to measure against. The report is laid out so the
human figure comes first and each grader's estimate sits in the same row as its agreement with
the human, so a reader sees the gap rather than being told about it.

**Generosity is called out by direction, not just by magnitude.** A grader that is consistently
harsh still tracks a change over time. A generous one is not usable for anything, because it
inflates every figure computed from it and the inflation is invisible in the figure. The report
says which you have.

---

## Kappa, and why raw agreement is not enough

`judge-lenient` agrees with the human 61.7% of the time, which sounds like most of the way
there. Its kappa is 0.25.

Raw agreement rewards a grader for matching the base rate. If a human calls 53% of items correct
and a grader says "correct" to almost everything, they will agree about half the time while
sharing no judgement at all. Kappa subtracts what the two graders' marginals would have produced
by chance and reports what is left.

The harness reports both, with Landis and Koch's labels attached — and with the caveat that
those bands are conventional and arbitrary, which the original paper says itself. The
trustworthiness threshold is kappa ≥ 0.60, stated in code rather than hidden, and a grader that
fails it is **warned about, not suppressed**: a reader who can see the calibration can decide
what to do with the number.

It also reports Fleiss' kappa across all graders at once:

```
kappa 0.10 (slight) across 4 graders; raw agreement 49.7% [41.7, 58.1], n=60,
against 0.439 expected by chance alone
```

Which is a different question from agreement with the human, and worth reading next to it rather
than instead of it. Graders can agree with each other and be wrong together — that is what
happens when they share a prompt.

Note the two intervals are computed differently and deliberately so. With two raters an item
either matches or does not, so raw agreement is a binomial proportion and Wilson applies. With
three or more, each item carries a *proportion* of agreeing rater pairs and the figure is the
mean of those — not a binomial proportion, so Wilson would be the wrong interval and a seeded
percentile bootstrap is the right one. `expected_by_chance` is printed as a decimal rather than
a percentage because it is not a measured rate, and dressing it as one invites exactly the
comparison that kappa already does properly.

---

## The holdout

Nobody hand-labels five hundred items to ship one change. The realistic shape is a holdout: a
few dozen items carry a human verdict, the graders run over everything, and the report has to
keep the two apart.

```bash
evals run --set fixtures/extraction-holdout.json
```

```
**50.0% [29.9, 70.1], n=20**

A person judged 20 of the 60 items. Everything above rests on those 20; the graders ran
on all 60.

| Grader | Its estimate | Agreement with the human | ... |
| `judge-strict` (model) | 45.0% [33.1, 57.5], n=60 | 90.0% [69.9, 97.2], n=20 | ... |
```

The two denominators in that row are the whole point: the estimate covers sixty items, the
calibration rests on twenty. An item nobody judged is not a failure and is not counted as one.
`Item.human_says_correct` raises on an unlabelled item rather than returning `False`, because
returning `False` would quietly record "nobody looked" as "the system got it wrong".

What is refused is the degenerate case — a model grader with *nothing* to calibrate against.

---

## Comparing two versions

Two runs over the same items are paired data, and only the items they disagreed on carry any
information. Comparing two accuracy figures as though they were independent samples throws the
pairing away along with most of the statistical power.

```bash
evals compare --before fixtures/extraction.json --after fixtures/extraction-v2.json
```

```
| Both correct | 30 |
| Fixed by the change | 10 |
| Broken by the change | 2 |
| Both wrong | 18 |

**10 item(s) fixed, 2 broken, 12 disagreements out of 60; the change looks better,
p=0.039 (significant at 0.05)**
```

McNemar's test, computed as an exact two-sided binomial over the discordant pairs rather than
the chi-squared approximation — a real evaluation usually has fewer than twenty-five
disagreements, which is exactly where the approximation stops being trustworthy.

When the result is not significant the harness says so in words that do not invite the usual
misreading: *not significant does not mean no difference; it means this set is too small to tell
one from chance.*

### A finding that came out of building this

Running the same comparison through the `rules` grader instead of the human labels reports
nothing at all — *"the two runs agreed on all 60 items, so there is nothing to compare"*. The
improvement in v2 was an added index name and removed hedging, and `rules` only checks figures.
It was blind to the entire change.

A harness that reported "no change" and stopped there would be worse than useless, so `compare`
now defaults to `--grader human` and, when a grader finds zero disagreements, prints:

> Nothing moved, which is worth a second look: `rules` may be blind to whatever changed. Compare
> the human labels (`--grader human`) before concluding that the change did nothing.

---

## The graders

| Grader | Model? | What it is for |
| --- | --- | --- |
| `exact` | no | The floor. Normalised string match, forgiving accents and case and nothing else. If this scores well the task did not need a model, and the evaluation should say so rather than crediting one. |
| `rules` | no | Every figure in the gold answer must appear in the candidate and none may be invented. The honest baseline a model grader has to beat to justify its cost. |
| `recorded` | yes | Replays verdicts captured earlier. What makes the whole evaluation reproducible offline and in CI. |

`rules` scores an invented figure as `partial` rather than `incorrect`: the candidate has
contradicted nothing and asserted something nobody checked, which is a different failure, and
collapsing the two loses a distinction the reviewer needs.

It also handles Brazilian decimal notation, where a full stop groups thousands and a comma
separates the decimal — the opposite of what most parsers assume. Getting it wrong turns a
correct answer into a failed one by a factor of a thousand.

A `RecordedGrader` raises on an item it has no recording for. It never guesses, because a grader
that silently scores an unknown item as correct is the worst failure a measurement tool can
have.

To plug in a live model grader, implement three members — `name`, `is_model`, `grade(item)` —
and pass it to `run_graders`. Marking it `is_model = True` is what arms the refusal.
[docs/wiring-a-model-grader.md](docs/wiring-a-model-grader.md) walks through a live Claude
judge, the record-and-replay step that keeps CI offline, how to read the calibration, and the
overfitting trap waiting at the end of it.

---

## Statistics, with no numpy and no scipy

The package has **zero runtime dependencies**. Wilson intervals, Cohen's and Fleiss' kappa,
exact McNemar and the percentile bootstrap are three hundred and fifty lines of `statistics`
and `math`, and every expected value in the tests was computed independently rather than by
running the code and pasting the answer.

A harness that is hard to install is a harness nobody runs, and an evaluation nobody runs is
worth less than no evaluation at all because it still appears in the README.

The bootstrap seed is fixed and explicit. An evaluation whose reported interval moves between
runs invites the obvious abuse, and a resampling procedure is exactly where that can happen
without anybody intending it. A test runs the demo twice and asserts the output is byte-identical.

---

## The fixtures are invented, and provably so

Sixty synthetic Brazilian court decisions, generated by `scripts/make_fixtures.py` from a fixed
seed. Invented court, invented municipality, invented state. Every case number is dated **2099**,
which no real case can be.

CI regenerates the fixtures from the seed and fails on any diff. That is the load-bearing check
behind "every item is invented": a file that came from somewhere else could not survive it, and
neither could an edit made by hand to flatter a number. A second check greps for
CPF- and CNPJ-shaped strings and for any case number outside the 2099 convention.

Two things in the generator are worth calling out:

**The human verdict is not a function of the construction kind.** An earlier version tagged each
item by how it was built (`paraphrased`, `hedged`, `wrong_retention`) and derived the human label
from the same tag. Every slice then came out at 0% or 100%, because the slice *was* the label.
`REVIEWER_DEPARTURES` makes a seeded fraction of reviewers disagree with the construction, which
is what a real reviewer does.

**The two model graders fail in opposite directions on purpose.** `judge-lenient` waves partials
through; `judge-strict` is harsh on hedging. A harness demonstrated only against a grader that
works demonstrates nothing.

---

## Commands

| | |
| --- | --- |
| `evals demo` | The whole thing offline on the shipped set. Start here. |
| `evals describe` | What is in the set, and how much of it a person has judged, before any grading. |
| `evals run --set S --graders exact,rules --out run.json` | Grade and report. |
| `evals report --set S --run run.json --model-graders judge-x` | Report on a saved run. |
| `evals compare --before A --after B` | Two versions over the same items, paired. |

---

## What this does not do

- **It does not call a model.** There is a `claude` extra and a `Grader` protocol; wiring a live
  judge is yours to do. Everything shipped runs offline so CI can run the whole evaluation on
  every push.
- **It does not tell you the human labels are right.** Inter-rater agreement between *people* is
  the next thing to measure, and `fleiss_kappa` will take their labels when you have them.
- **It does not fix a small set.** It tells you the set is small, in the only way that stays
  honest: an interval forty points wide, next to the number.
- **It does not choose your threshold.** Kappa ≥ 0.60 is a judgement call, stated rather than
  hidden, and easy to change.

---

## Licence

MIT. See [LICENSE](LICENSE).
