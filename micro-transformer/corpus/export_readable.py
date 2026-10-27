import json, csv
from pathlib import Path

src = Path("corpus_out_staged")
out_csv = Path("corpus_out_staged/all_stages.csv")
out_txt = Path("corpus_out_staged/all_stages.txt")

rows = []
txt_lines = []

for stage_num in range(6):
    sdir = src / f"stage{stage_num}"
    manifest = json.loads((sdir / "manifest.json").read_text())

    txt_lines.append(f"{'='*70}")
    txt_lines.append(f"STAGE {stage_num}  |  vocab={manifest['vocab_size']}  "
                      f"shapes={', '.join(manifest['sentence_shapes'])}  "
                      f"statements={manifest['num_statements']}  qa_pairs={manifest['num_qa_pairs']}")
    txt_lines.append(f"{'='*70}")

    txt_lines.append("-- statements --")
    statements = (sdir / "statements.txt").read_text().splitlines()
    for s in statements:
        if not s.strip():
            continue
        rows.append({"stage": stage_num, "type": "statement", "prompt": "", "response": s})
        txt_lines.append(f"  {s}")

    txt_lines.append("-- qa pairs --")
    qa_lines = (sdir / "qa_pairs.jsonl").read_text().splitlines()
    for line in qa_lines:
        if not line.strip():
            continue
        pair = json.loads(line)
        rows.append({"stage": stage_num, "type": "qa", "prompt": pair["prompt"], "response": pair["response"]})
        txt_lines.append(f"  Q: {pair['prompt']}")
        txt_lines.append(f"  A: {pair['response']}")

    txt_lines.append("")

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["stage", "type", "prompt", "response"])
    writer.writeheader()
    writer.writerows(rows)

out_txt.write_text("\n".join(txt_lines) + "\n")

print(f"CSV rows: {len(rows)}")
print(f"Written: {out_csv}")
print(f"Written: {out_txt}")
