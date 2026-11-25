"""Deduplicate DAPO, remove original answer-format wrapper, select English candidates.

The paper does not publish its exact language filter. This explicit script is a
reproducible approximation, not a claim of byte-identical paper data.
"""
import collections
import hashlib
import json
import re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_benchmarks import SYSTEM

ROOT = Path(__file__).resolve().parent / "assets/datasets"
PREFIX = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def main():
    source = ROOT / "raw/DAPO-Math-17k/data/dapo-math-17k.parquet"
    reader = pq.ParquetFile(source)
    unique = {}
    repeats = collections.Counter()
    for batch in reader.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            index = row["extra_info"]["index"]
            signature = (json.dumps(row["prompt"], ensure_ascii=False), row["reward_model"]["ground_truth"])
            if index in unique and unique[index][0] != signature:
                raise ValueError(f"Same ID with differing content: {index}")
            if index not in unique:
                unique[index] = (signature, row)
            repeats[index] += 1
    output = []
    excluded = []
    for index, (_, row) in unique.items():
        assert len(row["prompt"]) == 1 and row["prompt"][0]["role"] == "user"
        problem = row["prompt"][0]["content"]
        assert problem.startswith(PREFIX) and problem.endswith(SUFFIX), index
        problem = problem[len(PREFIX):-len(SUFFIX)].strip()
        # Reject Han/Cyrillic/Kana/Hangul text; retain Latin prose and math.
        # Latin-script non-English text may remain: this is only a heuristic.
        if re.search(r"[\u3400-\u9fff\u0400-\u052f\u3040-\u30ff\uac00-\ud7af]", problem):
            excluded.append({"index": index, "problem": problem})
            continue
        output.append({"data_source": "math_dapo", "ability": "math",
                       "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem}],
                       "reward_model": {"style": "rule", "ground_truth": str(row["reward_model"]["ground_truth"])},
                       "extra_info": {"index": index, "split": "train"}})
    dest = ROOT / "processed"
    dest.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output), dest / "train.parquet")
    (dest / "excluded_language.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"source_repo": "BytedTsinghua-SIA/DAPO-Math-17k", "revision": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
              "raw_rows": reader.metadata.num_rows, "unique_ids": len(unique),
              "repetition_counts": dict(collections.Counter(repeats.values())),
              "excluded_language": len(excluded), "train_rows": len(output),
              "language_filter": "Reject Han/Cyrillic/Kana/Hangul; heuristic English candidates, not author's exact subset",
              "prompt_change": "Remove DAPO Answer: wrapper; apply paper system prompt with boxed answer"}
    (dest / "preprocessing-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
