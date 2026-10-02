"""The grader protocol and the three that need no network.

Each exists for a different reason.

``ExactGrader`` is the floor. If a normalised string comparison already scores well, the task
did not need a model and the evaluation should say so rather than crediting one.

``RuleGrader`` is the one that usually wins on an extraction task: numbers must match exactly,
because a retention percentage is not approximately anything, while the prose around them may
vary freely. It is also the honest baseline a model grader has to beat to justify its cost.

``RecordedGrader`` replays verdicts captured earlier, which is what makes a whole evaluation
reproducible offline and in CI. An evaluation whose number moves when nobody changed anything
is not a measurement.
"""

from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from ..items import Item, Label, Run, Verdict, check_label


class Grader(Protocol):
    """Anything that can judge one item."""

    @property
    def name(self) -> str: ...

    @property
    def is_model(self) -> bool:
        """True when this grader is itself a language model.

        The report keys off this: a run whose only graders are models, with no human
        calibration, is refused rather than printed.
        """
        ...

    def grade(self, item: Item) -> Verdict: ...


# --------------------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lowercased, accent-free, punctuation collapsed.

    Deliberately crude. The point of ExactGrader is to be the floor, so making this cleverer
    would raise the floor and hide the thing it exists to reveal.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9%.,/-]+", " ", without_accents.lower()).split())


#: Numbers, percentages and money, which are the values that have to match exactly.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s*%?")

#: A full stop used to group thousands rather than to mark a decimal: exactly three digits
#: after every stop. "1.000" and "1.234.567" match; "1.5" and "1.0000" do not.
_GROUPED = re.compile(r"\d{1,3}(?:\.\d{3})+")


def numbers_in(text: str) -> list[str]:
    """Every number in the text, canonicalised so 1.000,50 and 1000.5 compare equal.

    Brazilian decimal notation groups thousands with a full stop and separates the decimal with
    a comma, which is the opposite of the convention most parsers assume. Getting this wrong
    turns a correct answer into a failed one, a thousandfold.

    The first version of this function did it wrong in exactly the way the paragraph above
    warns about: it applied the rule only when a comma was present, so "R$ 1.000" came back as
    1 and a grader scored "R$ 1,00" as a correct extraction of a thousand reais. A round
    thousand is the commonest figure in a judgment, so the case that broke was the likely one.

    Grouping is recognised by shape: a full stop followed by exactly three digits, repeated.
    That leaves "1.5" alone, which no Brazilian document means as fifteen hundred.
    """
    found: list[str] = []
    for raw in _NUMBER.findall(text):
        token = raw.replace(" ", "")
        is_percent = token.endswith("%")
        digits = token.rstrip("%")
        if "," in digits:
            digits = digits.replace(".", "").replace(",", ".")
        elif _GROUPED.fullmatch(digits):
            digits = digits.replace(".", "")
        value = digits.rstrip(".")
        try:
            # Decimal, not float: f"{float(x):g}" caps at six significant digits, so two
            # amounts in the millions that differ in the last digits compared equal.
            canonical = format(Decimal(value).normalize(), "f")
        except (InvalidOperation, ValueError):
            canonical = value
        found.append(canonical + ("%" if is_percent else ""))
    return found


class ExactGrader:
    """Correct only if the candidate normalises to exactly the gold answer."""

    name = "exact"
    is_model = False

    def grade(self, item: Item) -> Verdict:
        same = normalise(item.candidate) == normalise(item.gold)
        return Verdict(
            item_id=item.id,
            grader=self.name,
            label="correct" if same else "incorrect",
            rationale="normalised strings are identical" if same else "normalised strings differ",
        )


class RuleGrader:
    """Every figure in the gold answer must appear in the candidate, and none may be invented.

    This is the rule that matters on an extraction task. A summary that paraphrases freely but
    reproduces every figure is right; one that reads beautifully and changes a percentage is
    wrong, and no amount of fluency compensates.

    A candidate carrying a figure the gold answer does not have is scored ``partial`` rather
    than ``incorrect``: it has contradicted nothing, and it has asserted something nobody
    checked. Those are different failures and collapsing them loses the distinction.
    """

    name = "rules"
    is_model = False

    def grade(self, item: Item) -> Verdict:
        gold_numbers = numbers_in(item.gold)
        candidate_numbers = numbers_in(item.candidate)

        missing = [n for n in gold_numbers if n not in candidate_numbers]
        extra = [n for n in candidate_numbers if n not in gold_numbers]

        if missing:
            label: Label = "incorrect"
            rationale = f"the candidate is missing {', '.join(missing)}"
        elif extra:
            label = "partial"
            rationale = f"the candidate asserts {', '.join(extra)}, which the gold answer does not"
        else:
            label = "correct"
            rationale = (
                f"all {len(gold_numbers)} figure(s) match"
                if gold_numbers
                else "there were no figures to check"
            )
        return Verdict(item_id=item.id, grader=self.name, label=label, rationale=rationale)


class RecordedGrader:
    """Replays verdicts captured from an earlier run.

    How the demo, the tests and CI evaluate a model grader with no key and no network call. An
    item with no recording raises rather than defaulting, because a grader that silently scores
    an unknown item as correct is the worst failure a measurement tool can have.
    """

    def __init__(self, name: str, verdicts: dict[str, dict], is_model: bool = True) -> None:
        self._name = name
        self._verdicts = verdicts
        self._is_model = is_model

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_model(self) -> bool:
        return self._is_model

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> RecordedGrader:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=name or str(payload["grader"]),
            verdicts={str(row["item_id"]): row for row in payload["verdicts"]},
            is_model=bool(payload.get("is_model", True)),
        )

    @classmethod
    def from_run(cls, run: Run, grader: str, is_model: bool = True) -> RecordedGrader:
        """Captures one grader's verdicts out of a run, so they can be replayed.

        A run holds every grader's verdicts together with the items they covered; a recording
        holds one grader's, which is the unit you replay. Calling a model on every item of
        every CI run is slow, costs money, and makes the reported number drift for reasons
        nobody changed — so you grade once, record, and replay from then on.
        """
        verdicts = run.verdicts.get(grader)
        if not verdicts:
            raise KeyError(f"run {run.label or run.eval_set!r} has no verdicts from {grader!r}")
        return cls(
            name=grader,
            verdicts={
                v.item_id: {"item_id": v.item_id, "label": v.label, "rationale": v.rationale}
                for v in verdicts
            },
            is_model=is_model,
        )

    def save(self, path: str | Path) -> None:
        """Writes the recording in the format :meth:`load` reads."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "grader": self._name,
                    "is_model": self._is_model,
                    "verdicts": list(self._verdicts.values()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + chr(10),
            encoding="utf-8",
        )

    def grade(self, item: Item) -> Verdict:
        row = self._verdicts.get(item.id)
        if row is None:
            raise KeyError(
                f"no recorded verdict for item {item.id!r} from grader {self._name!r}. "
                "A recorded grader answers only for what it recorded, and never guesses."
            )
        return Verdict(
            item_id=item.id,
            grader=self._name,
            label=check_label(str(row["label"])),
            rationale=str(row.get("rationale", "")),
        )
