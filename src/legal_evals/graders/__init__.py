"""Graders: the things that decide whether a candidate answer is right.

The protocol is three lines. What matters is that there are several implementations and that
the harness treats them identically — the only way to find out whether a model grader can be
trusted is to run it beside graders that need no trusting, and beside a person.
"""

from .base import ExactGrader, Grader, RecordedGrader, RuleGrader, normalise, numbers_in

__all__ = [
    "ExactGrader",
    "Grader",
    "RecordedGrader",
    "RuleGrader",
    "normalise",
    "numbers_in",
]
