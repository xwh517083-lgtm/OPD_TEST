"""Download pinned model snapshots into the portable project directory."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".download-cache"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download


def main():
    manifest_path = ROOT / "assets" / "models-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    api = HfApi()
    for name in ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B-Instruct-2507"]:
        repo = "Qwen/" + name
        revision = records.get(repo, {}).get("revision") or api.model_info(repo).sha
        records[repo] = {"revision": revision, "complete": False}
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Downloading {repo} revision={revision}", flush=True)
        snapshot_download(repo, revision=revision, local_dir=ROOT / "assets" / "models" / name,
                          max_workers=3)
        records[repo]["complete"] = True
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Completed {repo}", flush=True)


if __name__ == "__main__":
    main()
