"""Convert downloaded benchmarks to the repository's verl parquet schema."""
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent / "assets" / "datasets"
SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."


def convert(rows, name):
    result = []
    for index, row in enumerate(rows):
        assert row["problem"].strip() and row["answer"] is not None
        result.append({
            "data_source": name,
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": row["problem"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(row["answer"])},
            "extra_info": {"split": "test", "index": index},
        })
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    for suffix in ["", "_verl"]:
        pq.write_table(pa.Table.from_pylist(result), out / f"{name}{suffix}.parquet")
    print(name, len(result))


if __name__ == "__main__":
    aime = pq.read_table(ROOT / "raw/aimo-validation-aime/data/train-00000-of-00001.parquet").to_pylist()
    aime = [r for r in aime if "/2024_AIME_" in r["url"]]
    assert len(aime) == 30
    convert(aime, "aime-24")
    convert(pq.read_table(ROOT / "raw/aime_2025/data/train-00000-of-00001.parquet").to_pylist(), "aime-2025")
    with (ROOT / "raw/MATH-500/test.jsonl").open(encoding="utf-8") as stream:
        convert([json.loads(line) for line in stream if line.strip()], "math500")
