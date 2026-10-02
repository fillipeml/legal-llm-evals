"""What an evaluation is made of.

The shape here encodes the argument the whole package makes. An item carries four things, and
the fourth is the one most harnesses leave out:

* ``gold`` — the correct answer, written by a person.
* ``candidate`` — what the system under test produced.
* ``human`` — **a person's verdict on that candidate**, which is the only ground truth in the
  building.
* and everything else is metadata for slicing.

Without ``human`` you can measure whether a model grader likes the system's output. You cannot
measure whether the grader is right, and so you cannot say anything about the system at all.
That is the gap this package exists to close.

``human`` is nonetheless optional, because hand-labelling is expensive and nobody labels five
hundred items by hand to ship one change. The realistic shape is a **holdout**: a few dozen
items carry a human verdict, the graders run on all of them, and the calibration rests on the
labelled part while the estimate covers the whole. The report prints both denominators rather
than letting the larger one stand in for the smaller.

What is refused is the degenerate case: a model grader with *nothing* to calibrate against,
which is a number with no backing at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

#: A closed vocabulary, deliberately three-valued. Two values force an honest "partly right"
#: into one of the extremes; more than three is a scale nobody applies consistently, which
#: shows up immediately as a collapsed kappa.
Label = Literal["correct", "partial", "incorrect"]
LABELS: tuple[Label, ...] = ("correct", "partial", "incorrect")


def check_label(value: str) -> Label:
    if value not in LABELS:
        raise ValueError(f"{value!r} is not one of {LABELS}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class Item:
    """One thing a system was asked to do, with a human's verdict on how it did."""

    id: str
    question: str
    source: str
    gold: str
    candidate: str
    #: A person's verdict, or None if nobody has looked at this item yet. Only the labelled
    #: items can calibrate a grader, and only they can measure the system.
    human: Label | None = None
    tags: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.human is not None:
            check_label(self.human)
        if not self.id:
            raise ValueError("every item needs an id")

    @property
    def labelled(self) -> bool:
        """Whether a person has judged this item."""
        return self.human is not None

    @property
    def human_says_correct(self) -> bool:
        """Strict by default: a partial answer is not a correct one.

        The alternative — counting partials as successes — is the single easiest way to make an
        evaluation look better than the system is, and it is usually done by accident. The
        report states which convention it used, every time.

        Raises on an unlabelled item rather than returning False, which would quietly count
        "nobody looked" as "the system got it wrong".
        """
        if self.human is None:
            raise ValueError(
                f"item {self.id!r} has no human verdict, so there is nothing to be strict about. "
                "Filter to EvalSet.labelled_only() before asking."
            )
        return self.human == "correct"

    @classmethod
    def from_dict(cls, payload: dict) -> Item:
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            source=str(payload["source"]),
            gold=str(payload["gold"]),
            candidate=str(payload["candidate"]),
            human=check_label(str(payload["human"])) if payload.get("human") else None,
            tags=tuple(payload.get("tags", ())),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class Verdict:
    """One grader's judgement of one item."""

    item_id: str
    grader: str
    label: Label
    rationale: str = ""

    def __post_init__(self) -> None:
        check_label(self.label)

    @property
    def says_correct(self) -> bool:
        return self.label == "correct"


@dataclass(frozen=True)
class EvalSet:
    """A named set of items, loaded once and never mutated."""

    name: str
    description: str
    items: tuple[Item, ...]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.items)

    def tagged(self, tag: str) -> EvalSet:
        """A slice, for reporting a rate per kind of item.

        Slicing is where small samples get smaller, which is exactly why every rate in the
        report carries its interval: a slice of eight items produces a number that looks like a
        finding and is not.
        """
        return EvalSet(
            name=f"{self.name}[{tag}]",
            description=self.description,
            items=tuple(item for item in self.items if tag in item.tags),
        )

    def labelled_only(self) -> EvalSet:
        """The items a person has judged — the only ones that can calibrate anything."""
        return EvalSet(
            name=self.name,
            description=self.description,
            items=tuple(item for item in self.items if item.labelled),
        )

    @property
    def labelled_count(self) -> int:
        return sum(1 for item in self.items if item.labelled)

    @property
    def fully_labelled(self) -> bool:
        return self.labelled_count == len(self.items)

    @property
    def all_tags(self) -> tuple[str, ...]:
        return tuple(sorted({tag for item in self.items for tag in item.tags}))

    @classmethod
    def load(cls, path: str | Path) -> EvalSet:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        items = tuple(Item.from_dict(row) for row in payload["items"])

        duplicates = {i for i in (item.id for item in items) if [x.id for x in items].count(i) > 1}
        if duplicates:
            # A duplicated id silently double-counts an item and, worse, breaks the pairing
            # that a paired comparison depends on.
            raise ValueError(f"duplicate item ids in {path}: {sorted(duplicates)}")

        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            items=items,
        )


@dataclass
class Run:
    """Every verdict from every grader over one set, plus what produced them."""

    eval_set: str
    item_ids: tuple[str, ...]
    verdicts: dict[str, list[Verdict]] = field(default_factory=dict)
    model: str = ""
    label: str = ""

    def add(self, verdict: Verdict) -> None:
        self.verdicts.setdefault(verdict.grader, []).append(verdict)

    @property
    def graders(self) -> tuple[str, ...]:
        return tuple(sorted(self.verdicts))

    def labels_from(self, grader: str) -> list[Label]:
        """One label per item, in the set's order, so two graders can be compared positionally."""
        by_id = {v.item_id: v.label for v in self.verdicts.get(grader, [])}
        missing = [i for i in self.item_ids if i not in by_id]
        if missing:
            raise ValueError(
                f"grader {grader!r} did not label {len(missing)} item(s): {missing[:3]}… "
                "An agreement figure over a partial set is not comparable with one over the whole."
            )
        return [check_label(by_id[i]) for i in self.item_ids]

    def to_dict(self) -> dict:
        return {
            "eval_set": self.eval_set,
            "label": self.label,
            "model": self.model,
            "item_ids": list(self.item_ids),
            "verdicts": {
                grader: [
                    {"item_id": v.item_id, "label": v.label, "rationale": v.rationale}
                    for v in verdicts
                ]
                for grader, verdicts in self.verdicts.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Run:
        run = cls(
            eval_set=str(payload["eval_set"]),
            item_ids=tuple(payload["item_ids"]),
            model=str(payload.get("model", "")),
            label=str(payload.get("label", "")),
        )
        for grader, verdicts in payload.get("verdicts", {}).items():
            for row in verdicts:
                run.add(
                    Verdict(
                        item_id=str(row["item_id"]),
                        grader=grader,
                        label=check_label(str(row["label"])),
                        rationale=str(row.get("rationale", "")),
                    )
                )
        return run

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Run:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
