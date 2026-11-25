"""Resumable model downloads, pinned revisions and official LFS SHA256 checks."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def read_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch_file(url, path, size, sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        if sha is None or digest(path) == sha:
            return
    partial = path.with_name(path.name + ".part")
    if not partial.exists() and sha:
        candidates = list((path.parent / ".cache/huggingface/download").glob(f"*.{sha}.incomplete"))
        if len(candidates) == 1 and candidates[0].stat().st_size <= size:
            candidates[0].replace(partial)
    for attempt in range(12):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ValueError(f"Oversized partial file: {partial}")
            if offset < size:
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Incorrect server byte range")
                    elif offset:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as stream:
                        last = time.monotonic()
                        while True:
                            chunk = response.read(256 * 1024)
                            if not chunk:
                                break
                            stream.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last >= 30:
                                stream.flush()
                                print(f"{path.parent.name}/{path.name}: {offset / 1e6:.1f}/{size / 1e6:.1f} MB", flush=True)
                                last = time.monotonic()
            if partial.stat().st_size != size:
                raise ValueError(f"Incomplete size for {path.name}")
            if sha and digest(partial) != sha:
                raise ValueError(f"SHA256 mismatch for {partial}")
            partial.replace(path)
            return
        except Exception as exc:
            print(f"Retry {attempt + 1}/12 {path.name}: {exc}", flush=True)
            if attempt == 11 or "SHA256 mismatch" in str(exc):
                raise
            time.sleep(5)


def download_model(name):
    repo = "Qwen/" + name
    manifest_path = ASSETS / "models" / name / "download-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = previous.get("revision")
    if not revision:
        old_path = ASSETS / "models-manifest.json"
        if old_path.exists():
            revision = json.loads(old_path.read_text()).get(repo, {}).get("revision")
    api_url = f"https://huggingface.co/api/models/{repo}"
    info = read_json(api_url + (f"/revision/{revision}" if revision else "") + "?blobs=true")
    revision = info["sha"]
    record = {"repo": repo, "revision": revision, "complete": False, "files": info["siblings"]}
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    for item in info["siblings"]:
        filename = item["rfilename"]
        path = manifest_path.parent / filename
        if manifest_path.parent.resolve() not in path.resolve().parents:
            raise ValueError("Unsafe repository path")
        print(f"Downloading {repo}/{filename}", flush=True)
        fetch_file(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true",
                   path, item["size"], item.get("lfs", {}).get("sha256"))
    record["complete"] = True
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Completed and verified {repo}", flush=True)
    return repo, {"revision": revision, "complete": True}


def main():
    names = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B-Instruct-2507"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = dict(pool.map(download_model, names))
    (ASSETS / "models-manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
