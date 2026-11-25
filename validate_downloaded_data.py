import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent / "assets/datasets"
source = ROOT / "raw/DAPO-Math-17k/data/dapo-math-17k.parquet"
h = hashlib.sha256()
with source.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        h.update(chunk)
assert h.hexdigest() == "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
counts = {}
for path in [ROOT / "processed/train.parquet", *sorted((ROOT / "benchmarks").glob("*.parquet"))]:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        assert row["prompt"][1]["content"].strip()
        assert row["reward_model"]["ground_truth"].strip()
    counts[str(path.relative_to(ROOT))] = len(rows)
train = pq.read_table(ROOT / "processed/train.parquet").to_pylist()
normalize = lambda s: "".join(s.lower().split())
train_questions = {normalize(r["prompt"][1]["content"]) for r in train}
overlaps = {}
eval_questions = set()
for path in sorted((ROOT / "benchmarks").glob("*.parquet")):
    rows = pq.read_table(path).to_pylist()
    eval_questions.update(normalize(r["prompt"][1]["content"]) for r in rows)
    overlaps[path.name] = sum(normalize(r["prompt"][1]["content"]) in train_questions for r in rows)
clean = [r for r in train if normalize(r["prompt"][1]["content"]) not in eval_questions]
import pyarrow as pa
pq.write_table(pa.Table.from_pylist(clean), ROOT / "processed/train_no_eval_overlap.parquet")
report = {"raw_train_sha256": h.hexdigest(), "schema_and_answers_valid": True,
          "train_no_eval_overlap_rows": len(clean),
          "row_counts": counts, "exact_normalized_question_overlap_with_train": overlaps,
          "overlap_note": "Exact normalized matching only; paraphrase contamination is not ruled out."}
(ROOT / "data-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
