"""
generate_corpus_staged.py
==========================
Staged, cumulative training-corpus generator for the "Island Tribe"
Micro-Transformer curriculum. Not tied to a fixed calendar -- stages track
CONCEPTS, and each stage's corpus is a strict superset of the previous one
(same world model throughout; vocabulary and sentence shapes only grow).

Stage map
---------
0  Embeddings           - one sentence shape (Subject Verb Object.), tiny vocab.
1  Position             - adds yes/no questions. "wolf hunts deer ?" -> yes and
                          "deer hunts wolf ?" -> no use the same words in a
                          different order with different answers, so a
                          position-blind (averaging) model provably cannot
                          get both right.
2  Attention / routing  - adds a 2nd and 3rd sentence shape (locative "in X",
                          attribution "is Adj") + "what"/"?" so the verb must
                          be inspected to know what kind of token comes next.
3  Training/generalize  - scales up fact count, adds "not"-negation, enables
                          a by-FACT train/held-out split.
4  Chatbot              - adds temporal shape ("when"), full prompt+response
                          pairs ready for teacher forcing + autoregressive
                          generation.
5  Judgment vs retrieval- currently the same data as stage 4 (yes/no moved to
                          stage 1). Lesson 6 will test judgment on claims held
                          out of training; yes/no answers are always built
                          ONLY from committed facts/negations (closed-world).

Run:  python generate_corpus_staged.py            (builds all stages)
      python generate_corpus_staged.py --stage 2  (builds through stage 2 only)

Editing facts
-------------
Facts are no longer hardcoded here -- they live in editable CSVs under
facts/, loaded by load_facts.py:
  facts/fears.csv, hunts.csv, eats.csv   -- subject x object matrices
  facts/attribution.csv                  -- subject x adjective matrix
  facts/sleeps.csv, lives.csv            -- subject x location matrices
  facts/simple_facts.csv                 -- long-format table for the
                                             smaller relations: drinks,
                                             likes, has, temporal (shines)

Matrix cell encoding: blank = undefined (no fact either way -- this
matters for stage 5's closed-world yes/no questions, which only ever ask
about a cell that's actually filled in). "N" = true, introduced at stage
N. "-N" = false (a stated negation), introduced at stage N. Open any
matrix CSV in a spreadsheet, edit cells, rerun this script -- no Python
editing needed for ordinary fact changes. See load_facts.py for the
loader that reconstructs the same fact structure this file used to
define inline.
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# 1. FACT REGISTRY -- loaded from facts/*.csv via load_facts.py (see above).
#    A fact used at stage N is automatically included in every stage >= N.
# ---------------------------------------------------------------------------
# shape one of: transitive | locative | temporal | attribution
#   ("locative" covers both "sleeps in X" and "lives in X" -- any verb whose
#   object slot is a place. "is" is reserved exclusively for attribution now.)
# negation=True means this is a stated FALSE version of the relation
#   (used from stage 3 onward for "not", and stage 5 for yes/no false claims).

from load_facts import load_facts
FACTS = load_facts()

# Stage 5 draws its yes/no claims from facts already on the books (stage<=4,
# non-negated -> true claims; negation=True -> false claims). No new facts
# needed, only a new question TASK over old facts -- see build_stage().

STRUCTURAL_BY_STAGE = {
    0: ["<PAD>", "."],
    1: ["<PAD>", "yes", "no", ".", "?"],
    2: ["<PAD>", "what", "in", "yes", "no", ".", "?"],
    3: ["<PAD>", "what", "in", "not", "yes", "no", ".", "?"],
    4: ["<PAD>", "what", "when", "in", "not", "yes", "no", ".", "?"],
    5: ["<PAD>", "what", "when", "in", "not", "yes", "no", ".", "?"],
}

# Semantic groups: NOT read by the generator's grammar logic at all -- this is
# instructor-side ground truth for the vector-comparison exercise ("did wolf
# and bear end up closer together than wolf and berry after training?").
SEMANTIC_GROUPS = {
    "predators": ["wolf", "bear", "hunter"],
    "prey": ["deer", "fish"],
    "companion": ["dog", "hunter"],
    "brave": ["hunter", "dog", "wolf", "bear"],
    "elements": ["fire", "water", "sun", "moon"],
    "places": ["forest", "cave", "river"],
    "positive_states": ["brave"],
    "negative_states": ["scared", "scary", "cold", "hungry"],
}


def toks(*words):
    return " ".join(str(w) for w in words)


def build_stage(n: int):
    """Return (statements, qa_pairs, vocab_tokens) for everything unlocked by stage n."""
    active_facts = [f for f in FACTS if f["stage"] <= n]

    statements, qa_pairs = [], []

    for f in active_facts:
        shape = f["shape"]
        neg = f.get("negation", False)

        if shape == "transitive":
            s, v, o = f["s"], f["v"], f["o"]
            if neg:
                # "not" statements only from stage 3; before that a negated
                # fact appears only as a "no" answer to a yes/no question.
                if n >= 3:
                    statements.append(toks(s, v, "not", o, "."))  # "not" goes right after the verb
                continue
            stmt = toks(s, v, o, ".")
            statements.append(stmt)
            if n >= 2:  # "what" questions unlock at stage 2
                qa_pairs.append({"prompt": toks(s, v, "what", "?"), "response": stmt})

        elif shape == "locative":
            s, v, loc = f["s"], f["v"], f["loc"]
            stmt = toks(s, v, "in", loc, ".")
            statements.append(stmt)
            if n >= 2:
                qa_pairs.append({"prompt": toks(s, v, "in", "what", "?"), "response": stmt})

        elif shape == "attribution":
            s, adj = f["s"], f["adj"]
            if neg:
                if n >= 3:
                    statements.append(toks(s, "is", "not", adj, "."))
                continue
            stmt = toks(s, "is", adj, ".")
            statements.append(stmt)
            if n >= 2:
                qa_pairs.append({"prompt": toks(s, "is", "what", "?"), "response": stmt})

        elif shape == "temporal":
            s, answer = f["s"], f["answer"]
            stmt = toks(s, "shines", ".")
            statements.append(stmt)
            if n >= 4:
                qa_pairs.append({"prompt": toks(s, "shines", "when", "?"), "response": toks(*answer, ".")})

    # --- Yes/no questions, from stage 1 on, built ONLY from committed facts --
    # Stage 1 needs them: "wolf hunts deer ?" -> yes vs "deer hunts wolf ?"
    # -> no is the smallest real case where word order alone changes the
    # right answer, which is what the position lesson has to demonstrate.
    if n >= 1:
        for f in active_facts:
            shape, neg = f["shape"], f.get("negation", False)
            if shape == "transitive":
                claim = toks(f["s"], f["v"], f["o"], "?")
            elif shape == "attribution":
                claim = toks(f["s"], "is", f["adj"], "?")
            else:
                continue
            answer = "no ." if neg else "yes ."
            qa_pairs.append({"prompt": claim, "response": answer})

    vocab_tokens = set(STRUCTURAL_BY_STAGE[n])
    for stmt in statements:
        vocab_tokens.update(stmt.split(" "))
    for pair in qa_pairs:
        vocab_tokens.update(pair["prompt"].split(" "))
        vocab_tokens.update(pair["response"].split(" "))

    random.shuffle(statements)
    random.shuffle(qa_pairs)
    return statements, qa_pairs, sorted(vocab_tokens)


def write_stage(n: int, out_root: Path):
    statements, qa_pairs, vocab_tokens = build_stage(n)
    stage_dir = out_root / f"stage{n}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    token_to_id = {tok: i for i, tok in enumerate(["<PAD>"] + [t for t in vocab_tokens if t != "<PAD>"])}
    with open(stage_dir / "vocab.json", "w") as f:
        json.dump(token_to_id, f, indent=2)

    with open(stage_dir / "statements.txt", "w") as f:
        f.write("\n".join(statements) + ("\n" if statements else ""))

    with open(stage_dir / "qa_pairs.jsonl", "w") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")

    # answer_key.json: prompt -> every valid response seen for it. Some
    # prompts genuinely have more than one true answer (e.g. "hunter is
    # what ?" -> tired / safe / brave, "hunter hunts what ?" -> deer / bear).
    # qa_pairs.jsonl keeps these as separate training examples on purpose --
    # a next-token model should learn a real probability distribution over
    # them, not collapse to one. This file is for EVALUATION: check a
    # generated answer against the full set of valid answers, not just
    # whichever single one happened to be sampled into a given training row.
    answer_key = {}
    for pair in qa_pairs:
        answer_key.setdefault(pair["prompt"], set()).add(pair["response"])
    answer_key = {p: sorted(r) for p, r in answer_key.items()}
    with open(stage_dir / "answer_key.json", "w") as f:
        json.dump(answer_key, f, indent=2)
    ambiguous_prompts = sum(1 for r in answer_key.values() if len(r) > 1)

    manifest = {
        "stage": n,
        "vocab_size": len(token_to_id),
        "sentence_shapes": sorted({f["shape"] for f in FACTS if f["stage"] <= n}),
        "num_statements": len(statements),
        "num_qa_pairs": len(qa_pairs),
        "total_examples": len(statements) + len(qa_pairs),
        "unique_prompts": len(answer_key),
        "ambiguous_prompts": ambiguous_prompts,
    }
    with open(stage_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=None, help="Build through this stage only (default: all 0-5)")
    args = parser.parse_args()

    out_root = Path("corpus_out_staged")
    out_root.mkdir(exist_ok=True)

    with open(out_root / "semantic_groups.json", "w") as f:
        json.dump(SEMANTIC_GROUPS, f, indent=2)

    stages = [args.stage] if args.stage is not None else list(range(6))
    manifests = [write_stage(n, out_root) for n in stages]

    print(f"{'stage':>5} {'vocab':>6} {'shapes':<45} {'stmts':>6} {'qa':>6} {'total':>6} {'ambig':>6}")
    for m in manifests:
        print(f"{m['stage']:>5} {m['vocab_size']:>6} {', '.join(m['sentence_shapes']):<45} "
              f"{m['num_statements']:>6} {m['num_qa_pairs']:>6} {m['total_examples']:>6} {m['ambiguous_prompts']:>6}")
    print(f"\nWritten to {out_root.resolve()}/")


if __name__ == "__main__":
    main()
