# CLAUDE.md

Working rules for AI-assisted changes in this repository. They mirror the README; the README
wins on conflict.

## Non-negotiable rules

1. **No rate is rendered without its interval.** `Interval.as_percent()` is the only code path
   that turns a proportion into a printed percentage. A new f-string that formats one directly
   is a defect, and `tests/test_harness.py` walks the rendered report asserting none exists.
2. **No model grader's estimate is rendered without its calibration.** `render()` raises
   `UncalibratedGraderError` when a grader marked `is_model` has nothing to measure against.
   Do not add a flag to bypass it. A weak grader is *warned about*, never suppressed — the
   reader decides.
3. **The human figure comes first, and each grader's estimate sits in the same row as its
   agreement with the human.** The layout is the argument. Do not reorder it so that a grader's
   number can be read on its own.
4. **An unlabelled item is not a failed item.** `Item.human_says_correct` raises rather than
   returning `False`, and `human_rate` divides by the labelled count, not the set size.
   Counting "nobody looked" as "the system got it wrong" is the quiet version of the mistake
   this repository is about.
5. **A grader never guesses.** `RecordedGrader` raises on an item it has no recording for.
6. **The interval must match the statistic.** Wilson for a binomial proportion, a seeded
   percentile bootstrap for a mean of per-item proportions, the exact binomial for McNemar.
   Using Wilson on Fleiss' raw agreement would be wrong and would look fine.
7. **Everything stays reproducible.** Fixed seeds, no clock, no network. CI regenerates the
   fixtures from the seed and fails on any diff, and a test runs the demo twice and compares
   the output byte for byte.
8. **Fictional data only.** Invented court, municipality and state; every case number dated
   2099. Never a real case number, name, CPF or CNPJ, in a fixture or a docstring or a test.

## Conventions

- Python 3.12+, **zero runtime dependencies**. A harness that is hard to install is a harness
  nobody runs. `pytest` and `ruff` are dev-only; the `claude` extra is optional and nothing
  shipped imports it.
- `ruff check` and `ruff format --check` both pass. Line length 100.
- Expected values in `tests/test_stats.py` are computed **independently** — by hand or from a
  published worked example — never by running the implementation and pasting the result. A test
  that only asserts the code does what the code does tests nothing.
- Docstrings explain *why*, not what. This repository is read more than it is run, and the
  reasoning is the deliverable.
- Tests are named as the sentence they assert. The refusals get the most attention, because a
  measurement tool whose guard rails are conventions has no guard rails.
- Commits: English, Conventional Commits, one logical change each, no AI attribution trailers.
