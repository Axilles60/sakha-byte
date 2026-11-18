"""Data acquisition and schema inspection. All entry points require remote opt-in.

Repositories supply data only: no dataset scripts, notebooks or pickle execution.
"""

import csv
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from .download import read_json, write_json


def require_remote():
    if platform.system() != "Linux" or os.environ.get("SAKHA_REMOTE_EXECUTION") != "immers":
        raise RuntimeError("Run only on IMMERS Linux: export SAKHA_REMOTE_EXECUTION=immers")


def file_sha(path: Path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(root: Path, name: str):
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe repository path: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path leaves dataset root")
    return path


def api_json(url):
    request = urllib.request.Request(url, headers={"User-Agent":"sakha-byte/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def verify_source(directory: Path):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "complete" or not manifest.get("files"):
        raise ValueError(f"Incomplete source: {directory}")
    for entry in manifest["files"]:
        path = safe_path(directory / "files", entry["name"])
        if not path.is_file() or path.stat().st_size != entry["size"] or file_sha(path) != entry["sha256"]:
            raise ValueError(f"Source missing or changed: {entry['name']}")
    return manifest


def download_source(name, spec, root, accept_gated=False, acknowledge_license=False):
    require_remote()
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Invalid source key")
    if spec.get("gated") and not accept_gated:
        raise ValueError("Accept upstream conditions yourself, then pass --gated-access-granted")
    if (not spec.get("license") or "NC" in spec["license"]) and not acknowledge_license:
        raise ValueError("Check source terms for your use, then pass --license-reviewed")
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if path.exists():
        manifest = read_json(path)
        if manifest["spec_sha256"] != spec_hash:
            raise ValueError("Source config changed: use a different --raw-root for a new version")
        if manifest["status"] == "complete":
            verify_source(directory)
            return manifest
    else:
        if spec["provider"] == "hf":
            from huggingface_hub import HfApi
            # HF_TOKEN is read by the library on the server; never copied into manifests.
            info = HfApi().dataset_info(spec["repo_id"], revision=spec["ref"], files_metadata=True)
            revision = info.sha
            entries = [{"name":s.rfilename, "size":s.size} for s in info.siblings]
        elif spec["provider"] == "github":
            base = "https://api.github.com/repos/" + spec["repo_id"]
            revision = api_json(base + "/commits/" + urllib.parse.quote(spec["ref"], safe=""))["sha"]
            tree = api_json(base + "/git/trees/" + revision + "?recursive=1")
            if tree.get("truncated"):
                raise ValueError("GitHub tree is truncated; explicit file list needed")
            entries = [{"name":x["path"], "size":x.get("size")} for x in tree["tree"]
                       if x["type"] == "blob" and x.get("mode") != "120000"]
        else:
            raise ValueError("Unknown provider")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Expected immutable full commit SHA")
        entries = [x for x in entries if any(fnmatch.fnmatchcase(x["name"], p) for p in spec["patterns"])]
        if not entries or any(x["size"] is None for x in entries):
            raise ValueError("No matching files or missing sizes; inspect source configuration")
        for entry in entries:
            safe_path(directory / "files", entry["name"])
        total = sum(x["size"] for x in entries)
        if total > spec["max_gib"] * 1024**3:
            raise ValueError("Source exceeds configured max_gib download budget")
        manifest = {"status":"locked", "source":name, "provider":spec["provider"],
                    "repo_id":spec["repo_id"], "revision":revision, "spec_sha256":spec_hash,
                    "declared_license":spec.get("license"), "files":entries}
        write_json(path, manifest)
    pending = [x for x in manifest["files"] if not x.get("sha256")]
    if shutil.disk_usage(directory).free < sum(x["size"] for x in pending) + 256*1024**2:
        raise RuntimeError("Insufficient space for selected files plus 256 MiB margin")
    for entry in manifest["files"]:
        target = safe_path(directory / "files", entry["name"])
        if entry.get("sha256"):
            if not target.is_file() or file_sha(target) != entry["sha256"]:
                raise ValueError("Previously downloaded file changed; use a fresh raw root")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if spec["provider"] == "hf":
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(spec["repo_id"], filename=entry["name"], repo_type="dataset",
                                         revision=manifest["revision"], local_dir=str(directory/"files"))
            if Path(downloaded).resolve() != target:
                raise ValueError("Unexpected Hugging Face destination")
        else:
            url = (f"https://raw.githubusercontent.com/{spec['repo_id']}/{manifest['revision']}/"
                   + urllib.parse.quote(entry["name"]))
            partial = target.with_name(target.name + ".part")
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            partial.replace(target)
        if target.stat().st_size != entry["size"]:
            raise ValueError(f"Size mismatch: {entry['name']}")
        entry["sha256"] = file_sha(target)
        write_json(path, manifest)
    manifest["status"] = "complete"
    write_json(path, manifest)
    verify_source(directory)
    return manifest


def select_files(directory, patterns):
    manifest = verify_source(directory)
    files = [safe_path(directory/"files", x["name"]) for x in manifest["files"]
             if any(fnmatch.fnmatchcase(x["name"], p) for p in patterns)]
    if not files:
        raise ValueError(f"No files match {patterns}")
    return sorted(files)


def field(row, key, optional=False):
    if key is None:
        return None
    value = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            if optional:
                return None
            raise ValueError(f"Missing field {key}; run inspect-source and correct the mapping")
        value = value[part]
    return value


def records(path, mapping):
    kind = mapping.get("format", "auto")
    kind = path.suffix.lower().lstrip(".") if kind == "auto" else kind
    if kind == "parquet":
        import pyarrow.parquet as parquet
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    elif kind == "jsonl":
        with path.open(encoding="utf-8-sig") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Expected JSONL objects")
                    yield row
    elif kind == "json":
        value = read_json(path)
        if mapping.get("records_path"):
            value = field(value, mapping["records_path"])
        if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
            raise ValueError("Expected JSON object array; configure records_path explicitly")
        yield from value
    elif kind in ("csv", "tsv"):
        with path.open(encoding="utf-8-sig", newline="") as source:
            yield from csv.DictReader(source, delimiter=mapping.get("delimiter") or ("\t" if kind == "tsv" else ","))
    elif kind == "xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            if mapping.get("sheet") is None and len(book.sheetnames) != 1:
                raise ValueError("Workbook has multiple sheets; configure sheet explicitly")
            sheet = book[mapping["sheet"]] if mapping.get("sheet") else book.active
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows)
            if any(not isinstance(x, str) or not x for x in headers) or len(set(headers)) != len(headers):
                raise ValueError("Invalid XLSX headers")
            for values in rows:
                if any(x is not None for x in values):
                    yield dict(zip(headers, values))
        finally:
            book.close()
    else:
        raise ValueError(f"Unsupported data format {kind}: {path.name}")


def inspect_source(directory):
    require_remote()
    manifest = verify_source(directory)
    result = {"source":manifest["source"], "revision":manifest["revision"], "files":[]}
    for entry in manifest["files"]:
        path = safe_path(directory/"files", entry["name"])
        info = {"name":entry["name"], "bytes":entry["size"]}
        if path.suffix == ".parquet":
            import pyarrow.parquet as parquet
            data = parquet.ParquetFile(path)
            info.update(rows=data.metadata.num_rows, schema=str(data.schema_arrow))
        elif path.suffix in (".csv", ".tsv", ".jsonl"):
            first = next(records(path, {}), None)
            info["columns"] = {k:type(v).__name__ for k,v in first.items()} if first else {}
        elif path.suffix == ".json":
            value = read_json(path)
            info["root_type"] = type(value).__name__
            info["root_keys"] = list(value) if isinstance(value, dict) else None
            if isinstance(value, list) and value and isinstance(value[0], dict):
                info["columns"] = {k:type(v).__name__ for k,v in value[0].items()}
        elif path.suffix == ".xlsx":
            from openpyxl import load_workbook
            book = load_workbook(path,read_only=True,data_only=True)
            try:
                info["sheets"] = {sheet.title:{"rows_including_header":sheet.max_row,
                                  "columns":list(next(sheet.iter_rows(values_only=True),[]))}
                                  for sheet in book.worksheets}
            finally:
                book.close()
        result["files"].append(info)
    return result
