# Micro-Transformer training corpus

The training data for the Micro-Transformer lessons is generated from the
fact tables in `facts/`, not written by hand.

## Run it

```
cd micro-transformer/corpus
python generate_corpus_staged.py      # builds all stages into corpus_out_staged/
python export_readable.py             # optional: all_stages.csv / all_stages.txt for browsing
```

`python generate_corpus_staged.py --stage 2` builds a single stage.

## Edit facts

Each file in `facts/` is a spreadsheet you can edit directly:

- `fears.csv`, `hunts.csv`, `eats.csv`: subject (row) x object (column)
- `attribution.csv`: subject x adjective ("fire is hot")
- `sleeps.csv`, `lives.csv`: subject x location
- `simple_facts.csv`: smaller relations (drinks, likes, has, shines)

Matrix cells:

| Cell  | Meaning |
|-------|---------|
| blank | no fact either way (never asked about) |
| `N`   | true, introduced at stage N |
| `-N`  | false, introduced at stage N (answered "no"; stated with "not" from stage 3) |

Each stage includes everything from earlier stages. Stage 1, for example,
adds yes/no questions: `wolf hunts deer ?` is yes and `deer hunts wolf ?` is
no, which is what the position lesson needs.
