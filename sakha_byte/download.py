"""Explicit downloads for the remote machine; no code execution from repositories."""

import fnmatch
import hashlib
import json
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_model(name: str, spec: dict, root: Path, weights: bool = False) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("Model revisions must be immutable commit SHAs")
    cache = root / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
    patterns = ["*.json", "*.py", "*.txt", "*.model", "*.tiktoken", "merges.txt", "README.md"]
    if weights:
        patterns += ["*.safetensors"]
    selected = [s for s in info.siblings if "/" not in s.rfilename and
                any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
    # Conservative full selected-size requirement, including a margin for temporary files.
    required = sum(s.size or 0 for s in selected) + 2 * 1024**3
    if shutil.disk_usage(cache).free < required:
        raise RuntimeError(f"Not enough space: require at least {required / 1024**3:.1f} GiB free")
    path = snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                             cache_dir=str(cache), allow_patterns=[s.rfilename for s in selected])
    result = {"name": name, "repo_id": spec["repo_id"], "revision": info.sha,
              "path": str(Path(path).resolve()), "weights": weights,
              "files": [{"name": s.rfilename, "size": s.size} for s in selected]}
    if spec.get("tokenizer_dependency"):
        result["tokenizer_dependency"] = prepare_model(
            name + "-tokenizer", spec["tokenizer_dependency"], root, weights=False)
    write_json(root / "models" / f"{name}.json", result)
    return result


def fetch_data(source: dict, destination: Path):
    """Resolve main once, then reuse data.lock.json unless explicitly removed."""
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / "data.lock.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock["github_repo"] != source["github_repo"]:
            raise ValueError("Source differs from existing data lock")
        revision = lock["revision"]
    else:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{source['github_repo']}/commits/{source['ref']}",
            headers={"User-Agent": "sakha-byte/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            revision = json.load(response)["sha"]
        lock = {"github_repo": source["github_repo"], "revision": revision,
                "resolved_at": datetime.now(timezone.utc).isoformat(), "files": []}
        write_json(lock_path, lock)
    old_hashes = {x["name"]: x["sha256"] for x in lock.get("files", [])}
    entries = []
    for name in source["files"]:
        if Path(name).name != name:
            raise ValueError("Only flat source paths are supported")
        url = f"https://raw.githubusercontent.com/{source['github_repo']}/{revision}/{name}"
        target = destination / name
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if name in old_hashes and old_hashes[name] != sha:
            temporary.unlink()
            raise ValueError(f"Hash mismatch for {name}")
        temporary.replace(target)
        entries.append({"name": name, "url": url, "sha256": sha, "bytes": target.stat().st_size})
    lock["files"] = entries
    lock["license_note"] = source["license_note"]
    write_json(lock_path, lock)
    return lock
