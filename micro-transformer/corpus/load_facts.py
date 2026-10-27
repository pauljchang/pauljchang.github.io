"""
load_facts.py
=============
Reads the editable CSVs under facts/ (matrices for fears/hunts/eats/
attribution/sleeps/lives, plus a long-format table for the smaller
relations: drinks/likes/has/temporal) and reconstructs the same FACTS list
shape that generate_corpus_staged.py originally used inline.

Matrix cell encoding:
    blank      -> undefined (no fact either way -- important for the
                  closed-world constraint on stage 5's yes/no questions:
                  we only ever ask about a cell that's actually filled in)
    "N"        -> TRUE, introduced at stage N   (e.g. "3")
    "-N"       -> FALSE (a stated negation), introduced at stage N (e.g. "-3")

Editing a CSV cell and rerunning generate_corpus_staged.py (which imports
this loader) regenerates the whole corpus from the edited matrix -- no
Python editing required for ordinary fact changes.
"""

import csv
from pathlib import Path

FACTS_DIR = Path(__file__).parent / "facts"


def _parse_cell(raw):
    """Return (stage, negation) or None if the cell is blank/undefined."""
    raw = raw.strip()
    if raw == "":
        return None
    stage = int(raw)
    if stage < 0:
        return (-stage, True)
    return (stage, False)


def _read_matrix(filename):
    """Yield (row_label, col_label, stage, negation) for every filled cell."""
    path = FACTS_DIR / filename
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = header[1:]
        for row in reader:
            if not row:
                continue
            r_label, cells = row[0], row[1:]
            for c_label, raw in zip(cols, cells):
                parsed = _parse_cell(raw)
                if parsed is not None:
                    stage, neg = parsed
                    yield r_label, c_label, stage, neg


def load_facts():
    facts = []

    for r, c, stage, neg in _read_matrix("fears.csv"):
        facts.append(dict(stage=stage, shape="transitive", s=r, v="fears", o=c, **({"negation": True} if neg else {})))

    for r, c, stage, neg in _read_matrix("hunts.csv"):
        facts.append(dict(stage=stage, shape="transitive", s=r, v="hunts", o=c, **({"negation": True} if neg else {})))

    for r, c, stage, neg in _read_matrix("eats.csv"):
        facts.append(dict(stage=stage, shape="transitive", s=r, v="eats", o=c, **({"negation": True} if neg else {})))

    for r, c, stage, neg in _read_matrix("attribution.csv"):
        facts.append(dict(stage=stage, shape="attribution", s=r, adj=c, **({"negation": True} if neg else {})))

    for r, c, stage, neg in _read_matrix("sleeps.csv"):
        facts.append(dict(stage=stage, shape="locative", s=r, v="sleeps", loc=c, **({"negation": True} if neg else {})))

    for r, c, stage, neg in _read_matrix("lives.csv"):
        facts.append(dict(stage=stage, shape="locative", s=r, v="lives", loc=c, **({"negation": True} if neg else {})))

    with open(FACTS_DIR / "simple_facts.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relation = row["relation"]
            stage = int(row["stage"])
            neg = row["negation"].strip() in ("True", "true", "1")
            if relation == "temporal":
                facts.append(dict(stage=stage, shape="temporal", s=row["subject"], answer=row["value"].split(" ")))
            else:
                facts.append(dict(stage=stage, shape="transitive", s=row["subject"], v=relation, o=row["value"],
                                   **({"negation": True} if neg else {})))

    return facts
