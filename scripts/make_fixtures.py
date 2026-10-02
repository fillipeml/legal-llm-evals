"""Builds the evaluation set and the recorded grader verdicts.

Generated rather than hand-written, so anyone can see exactly what is in it and regenerate it.
Everything is invented: fictional parties, a fictional state, and case numbers dated 2099 whose
check digits do not compute, so none of them can be a real case.

Two things this generator is designed to produce, because a fixture set where every grader
agrees teaches nothing:

**A system that is neither good nor terrible.** Around 60% correct, so the interval is wide
enough at n=60 to make the point that sixty items is not many.

**Graders that fail in opposite directions.** ``judge-lenient`` waves through partial answers
and some wrong ones; ``judge-strict`` is harsh on anything hedged. Both look plausible, their
raw agreement with the human is similar, and only the kappa and the generous/harsh split tell
them apart — which is the entire argument of the package.

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("fixtures")
SEED = 20260102

OUTCOMES = [
    ("JULGADO PARCIALMENTE PROCEDENTE", "parcialmente procedente"),
    ("JULGADO PROCEDENTE", "procedente"),
    ("JULGADO IMPROCEDENTE", "improcedente"),
    ("JULGADO EXTINTO", "extinto"),
]
INDICES = ["INPC", "IGP-M", "IPCA"]
RETENTIONS = [5, 10, 15, 20, 25]
FEES = [10, 12, 15, 20]

QUESTION = (
    "Extract the final outcome, the retention percentage, the indexation index and the fee "
    "percentage from this decision."
)


def invalid_check_digits(sequential: int) -> str:
    """Two digits that are deliberately *not* the CNJ check digits for this number.

    A CNJ number verifies by ISO 7064 mod 97-10: the check digits are
    ``98 - (NNNNNNN AAAA J TR OOOO * 100) mod 97``. An earlier version of this script picked
    them from an arithmetic sequence, and one number in sixty landed on the correct value by
    accident — enough for a scanner to flag it as a real case number, and a reminder that
    "obviously fake" is not a property you get to assume.

    So compute the right answer and return something else.
    """
    base = int(f"{sequential}20998260100")
    correct = 98 - (base * 100) % 97
    wrong = (correct % 89) + 10 if (correct % 89) + 10 != correct else correct + 1
    assert wrong != correct and 10 <= wrong <= 99
    return f"{wrong:02d}"


def decision_text(i: int, outcome: str, retention: int, index: str, fee: int) -> str:
    """A short, synthetic operative part. Portuguese, because the documents would be."""
    case = f"{1000000 + i}-{invalid_check_digits(1000000 + i)}.2099.8.26.0100"
    return (
        f"Processo nº {case}. 2ª Vara Cível da Comarca de Porto das Águas, Estado de "
        f"Zetalândia. Ante o exposto, {outcome} o pedido para condenar a ré à restituição "
        f"dos valores pagos, com retenção de {retention}% a título de despesas "
        f"administrativas, corrigidos pelo {index} desde cada desembolso e acrescidos de "
        f"juros de mora de 1% ao mês a contar da citação. Custas processuais e honorários "
        f"advocatícios fixados em {fee}% sobre o valor da condenação, a cargo da ré."
    )


def gold_answer(outcome_label: str, retention: int, index: str, fee: int) -> str:
    return (
        f"Resultado: {outcome_label}. Retenção: {retention}%. Índice: {index}. Honorários: {fee}%."
    )


#: Each kind produces a candidate and the verdict a careful person would give it. The weights
#: are what set the overall success rate, and they are chosen so it lands near 60%.
KINDS = [
    ("verbatim", 14),
    ("paraphrased", 13),
    ("reordered", 9),
    ("missing_index", 6),
    ("extra_figure", 5),
    ("hedged", 5),
    ("wrong_retention", 4),
    ("wrong_fee", 2),
    ("wrong_outcome", 2),
]


def candidate_for(
    kind: str, outcome_label: str, retention: int, index: str, fee: int, rng: random.Random
) -> tuple[str, str]:
    """Returns the candidate answer and the human verdict on it."""
    if kind == "verbatim":
        return gold_answer(outcome_label, retention, index, fee), "correct"

    if kind == "paraphrased":
        return (
            f"O pedido foi julgado {outcome_label}, com retenção de {retention}% sobre os "
            f"valores pagos, correção pelo {index} e honorários de {fee}%.",
            "correct",
        )

    if kind == "reordered":
        return (
            f"Honorários: {fee}%. Índice: {index}. Retenção: {retention}%. "
            f"Resultado: {outcome_label}.",
            "correct",
        )

    if kind == "missing_index":
        return (
            f"Resultado: {outcome_label}. Retenção: {retention}%. Honorários: {fee}%.",
            "partial",
        )

    if kind == "extra_figure":
        invented = rng.choice([2, 3, 8])
        return (
            f"Resultado: {outcome_label}. Retenção: {retention}%. Índice: {index}. "
            f"Honorários: {fee}%. Multa contratual: {invented}%.",
            "partial",
        )

    if kind == "hedged":
        return (
            f"Aparentemente o pedido foi julgado {outcome_label}; a retenção parece ser de "
            f"{retention}%, com correção pelo {index} e honorários de {fee}%, embora a "
            f"redação não seja inteiramente clara.",
            "partial",
        )

    if kind == "wrong_retention":
        wrong = rng.choice([r for r in RETENTIONS if r != retention])
        return (
            f"Resultado: {outcome_label}. Retenção: {wrong}%. Índice: {index}. Honorários: {fee}%.",
            "incorrect",
        )

    if kind == "wrong_fee":
        wrong = rng.choice([f for f in FEES if f != fee])
        return (
            f"Resultado: {outcome_label}. Retenção: {retention}%. Índice: {index}. "
            f"Honorários: {wrong}%.",
            "incorrect",
        )

    if kind == "wrong_outcome":
        wrong = rng.choice([o for _, o in OUTCOMES if o != outcome_label])
        return (
            f"Resultado: {wrong}. Retenção: {retention}%. Índice: {index}. Honorários: {fee}%.",
            "incorrect",
        )

    raise ValueError(f"unknown kind {kind!r}")


#: How often a reviewer departs from what the construction kind would imply.
#:
#: Without this, every slice of the report is 100% or 0% by construction — the tag IS the
#: label — and the per-slice table teaches nothing. Real review is not a pure function of how
#: an answer was generated: a paraphrase can lose a nuance, and an extra figure can turn out to
#: have been in the source and harmless.
REVIEWER_DEPARTURES = {
    "paraphrased": (0.20, "partial", "the paraphrase lost the basis of the index"),
    "reordered": (0.10, "partial", "reordering obscured which percentage was which"),
    "extra_figure": (0.25, "correct", "the extra figure was in the source and is harmless"),
    "hedged": (0.20, "correct", "the hedging is stylistic; every figure is right"),
}


def reviewer_judgement(kind: str, implied: str, rng: random.Random) -> tuple[str, str]:
    """The human verdict, which is not a pure function of how the candidate was made."""
    departure = REVIEWER_DEPARTURES.get(kind)
    if departure is None:
        return implied, ""
    probability, label, reason = departure
    if rng.random() < probability:
        return label, reason
    return implied, ""


def build_items() -> list[dict]:
    rng = random.Random(SEED)
    kinds: list[str] = []
    for kind, count in KINDS:
        kinds += [kind] * count
    rng.shuffle(kinds)

    items = []
    for i, kind in enumerate(kinds, start=1):
        outcome_text, outcome_label = rng.choice(OUTCOMES)
        retention = rng.choice(RETENTIONS)
        index = rng.choice(INDICES)
        fee = rng.choice(FEES)

        candidate, human = candidate_for(kind, outcome_label, retention, index, fee, rng)
        human, note = reviewer_judgement(kind, human, rng)
        items.append(
            {
                "id": f"item-{i:03d}",
                "question": QUESTION,
                "source": decision_text(i, outcome_text, retention, index, fee),
                "gold": gold_answer(outcome_label, retention, index, fee),
                "candidate": candidate,
                "human": human,
                "tags": [kind],
                "note": note,
            }
        )
    return items


#: How each recorded grader departs from the human, as a probability per human label.
#: Chosen so the two look similar on raw agreement and differ sharply on kappa and on which
#: way they err — which is exactly the pair of graders a real project ends up choosing between.
PROFILES = {
    "judge-lenient": {
        # Waves partials through, and misses a third of the genuinely wrong answers.
        "correct": {"correct": 0.98, "partial": 0.02, "incorrect": 0.0},
        "partial": {"correct": 0.80, "partial": 0.15, "incorrect": 0.05},
        "incorrect": {"correct": 0.30, "partial": 0.20, "incorrect": 0.50},
    },
    "judge-strict": {
        # Reliable on wrong answers, but punishes anything hedged or reordered.
        "correct": {"correct": 0.80, "partial": 0.18, "incorrect": 0.02},
        "partial": {"correct": 0.05, "partial": 0.80, "incorrect": 0.15},
        "incorrect": {"correct": 0.02, "partial": 0.08, "incorrect": 0.90},
    },
}

RATIONALES = {
    "correct": "every figure in the gold answer is present and none is contradicted",
    "partial": "the answer is incomplete or asserts something the gold answer does not",
    "incorrect": "at least one figure contradicts the gold answer",
}


def build_grader(name: str, items: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    profile = PROFILES[name]
    verdicts = []
    for item in items:
        distribution = profile[item["human"]]
        roll = rng.random()
        cumulative = 0.0
        chosen = "correct"
        for label, probability in distribution.items():
            cumulative += probability
            if roll <= cumulative:
                chosen = label
                break
        verdicts.append({"item_id": item["id"], "label": chosen, "rationale": RATIONALES[chosen]})
    return {"grader": name, "is_model": True, "verdicts": verdicts}


#: What a prompt change fixed and what it broke, for the paired-comparison demonstration.
#: Deliberately modest: six fixes and two regressions out of sixty is the shape of a real
#: iteration, and it is right at the edge of what sixty items can distinguish from chance —
#: which is the lesson.
V2_FIXES = {"missing_index", "hedged"}
V2_BREAKS = 2


def build_v2(items: list[dict]) -> list[dict]:
    """The same questions and the same gold answers, with a later system's candidates.

    Same ids, because a paired comparison is only valid over the same items. The CLI refuses to
    compare two runs whose item lists differ, and this is what a legitimate pair looks like.
    """
    rng = random.Random(SEED + 100)
    improved = []
    broken = 0
    for item in items:
        kind = item["tags"][0]
        new_item = dict(item)

        if kind in V2_FIXES and item["human"] != "correct":
            # The change taught the system to carry every field through.
            new_item["candidate"] = item["gold"]
            new_item["human"] = "correct"
            new_item["note"] = "fixed by the prompt change"
        elif kind == "verbatim" and broken < V2_BREAKS and rng.random() < 0.3:
            # And it introduced a hedge the earlier version did not have.
            new_item["candidate"] = (
                "Provavelmente " + item["candidate"][0].lower() + item["candidate"][1:]
            )
            new_item["human"] = "partial"
            new_item["note"] = "regressed: the change introduced hedging"
            broken += 1
        improved.append(new_item)
    return improved


HOLDOUT_LABELLED = 20


def build_holdout(items: list[dict]) -> list[dict]:
    """The same sixty items with a human verdict on only twenty of them.

    This is what an evaluation set looks like in practice. Hand-labelling is the expensive
    part, so a team labels what it can afford and runs the graders over everything. The point
    of shipping it as a fixture is that the report then has to show two different denominators
    and a reader can see the difference between what was measured and what was merely graded.
    """
    rng = random.Random(SEED + 99)
    keep = set(rng.sample(range(len(items)), HOLDOUT_LABELLED))
    out = []
    for position, item in enumerate(items):
        copy = dict(item)
        if position not in keep:
            copy.pop("human", None)
            copy["note"] = "not yet reviewed"
        out.append(copy)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "graders").mkdir(exist_ok=True)

    items = build_items()
    counts = {
        label: sum(1 for i in items if i["human"] == label)
        for label in ("correct", "partial", "incorrect")
    }

    (OUT / "extraction.json").write_text(
        json.dumps(
            {
                "name": "judgment-field-extraction",
                "description": (
                    "Sixty synthetic Brazilian court decisions, each with a gold extraction of "
                    "outcome, retention, index and fee, a candidate answer, and a human verdict "
                    "on that candidate. Every case number is dated 2099 with check digits that "
                    "do not compute."
                ),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"extraction.json      {len(items)} items  {counts}")

    v2 = build_v2(items)
    v2_counts = {
        label: sum(1 for i in v2 if i["human"] == label)
        for label in ("correct", "partial", "incorrect")
    }
    (OUT / "extraction-v2.json").write_text(
        json.dumps(
            {
                "name": "judgment-field-extraction-v2",
                "description": (
                    "The same sixty questions and gold answers, with the candidates a later "
                    "version of the system produced. Same item ids, so the two can be compared "
                    "as paired data."
                ),
                "items": v2,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"extraction-v2.json   {len(v2)} items  {v2_counts}")

    holdout = build_holdout(items)
    (OUT / "extraction-holdout.json").write_text(
        json.dumps(
            {
                "name": "judgment-field-extraction-holdout",
                "description": (
                    "The same sixty items, with a human verdict on only twenty of them. What an "
                    "evaluation set looks like once it outgrows what anyone will label by hand: "
                    "the graders run on all sixty, the calibration rests on twenty, and the "
                    "report has to say which number is which."
                ),
                "items": holdout,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    labelled = sum(1 for i in holdout if "human" in i)
    print(f"extraction-holdout.json  {len(holdout)} items, {labelled} labelled")

    for offset, name in enumerate(PROFILES):
        grader = build_grader(name, items, seed=SEED + 1 + offset)
        (OUT / "graders" / f"{name}.json").write_text(
            json.dumps(grader, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        agreement = sum(
            1
            for item, verdict in zip(items, grader["verdicts"], strict=True)
            if item["human"] == verdict["label"]
        )
        print(f"graders/{name}.json  agrees with the human on {agreement}/{len(items)}")


if __name__ == "__main__":
    main()
