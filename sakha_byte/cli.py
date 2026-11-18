"""Run on a remote Linux GPU server. No action occurs at import time."""

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import PROMPT_VERSION, combine_rotations, digest, load_items, make_prompt, summarize
from .download import fetch_data, prepare_model, read_json, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def evaluate(args, models):
    from .runtime import HFScorer, software_versions
    import torch

    items = load_items(args.benchmark, args.allow_draft)
    if args.limit:
        items = items[:args.limit]
    for item in items:
        make_prompt(item, args.language)  # fail before model load for missing controls
        if args.rotations not in (0, 1, len(item["choices"])):
            raise ValueError("Use --rotations 0 for all positions or 1 for a single position")
    if args.output.exists():
        raise FileExistsError("Use a new output path; results are never overwritten")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint_hashes = None
    if args.checkpoint:
        # Full weight hashes distinguish adapted runs even after local files change.
        files = sorted(args.checkpoint.glob("*.safetensors"))
        if not files:
            raise ValueError("--checkpoint requires a full merged safetensors checkpoint")
        checkpoint_hashes = {}
        for file in files + sorted(args.checkpoint.glob("*.json")):
            sha = hashlib.sha256()
            with file.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024**2), b""):
                    sha.update(block)
            checkpoint_hashes[file.name] = sha.hexdigest()
    metadata = {"status": "running", "created_at": utc_now(), "model_key": args.model,
                "model": models[args.model], "stage": args.stage,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "checkpoint_sha256": checkpoint_hashes,
                "benchmark_sha256": digest(args.benchmark), "n": len(items),
                "prompt_version": PROMPT_VERSION, "language": args.language,
                "rotations": "all" if args.rotations == 0 else args.rotations, "seed": args.seed,
                "choice_counts":sorted({len(x["choices"]) for x in items}),
                "draft": any(x["review_status"] != "approved" for x in items),
                "software": software_versions(), "dtype": "bfloat16", "quantization": None,
                "scoring": "next-label-conditional-logprob-v2", "ensemble": False}
    write_json(args.output / "metadata.json", metadata)
    try:
        scorer = HFScorer(args.model, models[args.model], args.artifacts,
                          args.checkpoint, args.max_input_units)
        metadata["resolved_model"] = scorer.manifest
        metadata["gpu"] = torch.cuda.get_device_name()
        rows = []
        with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as out:
            for item in items:
                runs, details = [], []
                n_choices = len(item["choices"])
                for rotation in range(args.rotations or n_choices):
                    prompt, order = make_prompt(item, args.language, rotation)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    log_probs, units = scorer.score(prompt, n_choices)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    runs.append((log_probs, order))
                    details.append({"rotation": rotation, "order": order, "log_probs": log_probs,
                                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                    "input_units": units, "prompt_bytes": len(prompt.encode()),
                                    "seconds": elapsed})
                scores = combine_rotations(runs)
                prediction = max(range(n_choices), key=scores.__getitem__)
                row = {"id": item["id"], "category": item["category"], "scores": scores,
                       "answer_index": item["answer_index"], "prediction": prediction, "n_choices":n_choices,
                       "group_id":item.get("group_id",item.get("pair_id",item["id"])),
                       "correct": prediction == item["answer_index"],
                       "tied": scores.count(max(scores)) > 1, "rotations": details}
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{item['id']}: {'correct' if row['correct'] else 'wrong'}", flush=True)
        write_json(args.output / "summary.json", summarize(rows))
        metadata.update(status="complete", completed_at=utc_now())
    except BaseException as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output / "metadata.json", metadata)


def ensemble(args):
    if not 0 <= args.bwen_weight <= 1:
        raise ValueError("Weight must be between zero and one")
    left, right = [read_json(p / "metadata.json") for p in (args.bwen_run, args.evabyte_run)]
    for meta, expected in ((left, "bwen"), (right, "evabyte")):
        if meta["status"] != "complete" or meta["model_key"] != expected:
            raise ValueError(f"Expected completed {expected} run")
    keys = ("benchmark_sha256", "prompt_version", "language", "rotations", "scoring", "n", "draft")
    if any(left[k] != right[k] for k in keys):
        raise ValueError("Runs use different benchmark/prompt/protocol")
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    source_rows = [[json.loads(line) for line in (p / "predictions.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()] for p in (args.bwen_run, args.evabyte_run)]
    if len(source_rows[0]) != left["n"] or len(source_rows[1]) != right["n"]:
        raise ValueError("Incomplete predictions")
    if any(len({r["id"] for r in rows}) != len(rows) for rows in source_rows):
        raise ValueError("Duplicate prediction IDs")
    rows = []
    for a, b in zip(*source_rows):
        if any(a[k] != b[k] for k in ("id", "answer_index", "category")):
            raise ValueError("Item order or gold labels differ")
        # Verify actual rendered prompts, not only metadata.
        if [r["prompt_sha256"] for r in a["rotations"]] != [r["prompt_sha256"] for r in b["rotations"]]:
            raise ValueError("Prompt hashes differ")
        if len(a["scores"]) != len(b["scores"]):
            raise ValueError("Choice counts differ")
        scores = [args.bwen_weight*x + (1-args.bwen_weight)*y
                  for x, y in zip(a["scores"], b["scores"])]
        if not 2 <= len(scores) <= 4 or not all(math.isfinite(x) for x in scores):
            raise ValueError("Invalid ensemble scores")
        prediction = max(range(len(scores)), key=scores.__getitem__)
        rows.append({"id": a["id"], "category": a["category"], "answer_index": a["answer_index"],
                     "scores": scores, "prediction": prediction, "n_choices":len(scores),
                     "group_id":a.get("group_id",a["id"]),
                     "correct": prediction == a["answer_index"],
                     "tied": scores.count(max(scores)) > 1})
    metadata = {k: left[k] for k in keys}
    metadata.update(status="complete", ensemble=True, model_key="bwen+evabyte", draft=left["draft"],
                    created_at=utc_now(), method="weighted-mean-logprob",
                    bwen_weight=args.bwen_weight,
                    source_metadata_sha256=[digest(p / "metadata.json") for p in (args.bwen_run, args.evabyte_run)],
                    source_predictions_sha256=[digest(p / "predictions.jsonl") for p in (args.bwen_run, args.evabyte_run)],
                    source_runs=[str(p.resolve()) for p in (args.bwen_run, args.evabyte_run)])
    args.output.mkdir(parents=True)
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "summary.json", summarize(rows))
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def report(paths):
    """Print a descriptive table only for runs of the same benchmark/protocol."""
    runs = [(read_json(p / "metadata.json"), read_json(p / "summary.json")) for p in paths]
    reference = runs[0][0]
    for meta, summary in runs:
        if meta["status"] != "complete":
            raise ValueError("Report requires completed runs")
        for key in ("benchmark_sha256", "prompt_version", "language", "rotations", "scoring", "n"):
            if meta[key] != reference[key]:
                raise ValueError(f"Incompatible runs: {key} differs")
        if summary["n"] != meta["n"]:
            raise ValueError("Summary count differs from metadata")
    print("DRAFT: native-speaker review pending" if any(m["draft"] for m, _ in runs) else "Reviewed pilot")
    print("Descriptive results; small paired pilot does not establish statistical superiority.\n")
    print("| System | Stage | Correct | Accuracy |")
    print("|---|---|---:|---:|")
    for meta, summary in runs:
        print(f"| {meta['model_key']} | {meta.get('stage', 'ensemble')} | "
              f"{summary['correct']}/{summary['n']} | {summary['accuracy']:.1%} |")
    for category in sorted({k for _,s in runs for k in s.get("by_category",{})}):
        if not any("macro_f1" in s.get("by_category",{}).get(category,{}) for _,s in runs):
            continue
        print(f"\n{category}: macro-F1 (NER type classification uses gold spans; not extraction F1)")
        for meta,summary in runs:
            score=summary.get("by_category",{}).get(category,{})
            if "macro_f1" in score:
                print(f"{meta['model_key']}: {score['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("download-model")
    prepare.add_argument("--model", choices=["bwen", "qwen", "evabyte"], required=True)
    prepare.add_argument("--weights", action="store_true", help="Without this flag download metadata/tokenizers only")
    commands.add_parser("download-data")
    pilot = commands.add_parser("create-pilot")
    pilot.add_argument("--output", type=Path, default=Path("data/eval/pilot.jsonl"))
    for command in ("validate", "plan", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--benchmark", type=Path, default=Path("data/eval/pilot.jsonl"))
        sub.add_argument("--allow-draft", action="store_true")
        if command == "evaluate":
            sub.add_argument("--model", choices=["bwen", "qwen", "evabyte"], required=True)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--language", choices=["sah", "ru"], default="sah")
            sub.add_argument("--rotations", type=int, choices=[0, 1, 2, 3, 4], default=0,
                             help="0: all cyclic positions (default); 1: single position")
            sub.add_argument("--stage", choices=["base", "adapted"], default="base")
            sub.add_argument("--checkpoint", type=Path)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--limit", type=int)
            sub.add_argument("--max-input-units", type=int, default=8192)
    joined = commands.add_parser("ensemble")
    joined.add_argument("--bwen-run", type=Path, required=True)
    joined.add_argument("--evabyte-run", type=Path, required=True)
    joined.add_argument("--bwen-weight", type=float, default=0.5)
    joined.add_argument("--output", type=Path, required=True)
    table = commands.add_parser("report")
    table.add_argument("runs", nargs="+", type=Path)
    fetch = commands.add_parser("fetch-source")
    fetch.add_argument("--source", required=True)
    fetch.add_argument("--config", type=Path, default=Path("configs/dataset_downloads.json"))
    fetch.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    fetch.add_argument("--gated-access-granted", action="store_true")
    fetch.add_argument("--license-reviewed", action="store_true")
    inspect = commands.add_parser("inspect-source")
    inspect.add_argument("--source", required=True)
    inspect.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    corpus = commands.add_parser("prepare-corpus")
    corpus.add_argument("--profiles", nargs="+", required=True)
    corpus.add_argument("--mappings", type=Path, default=Path("configs/corpus_mappings.json"))
    corpus.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    corpus.add_argument("--tokenizer-path", type=Path, required=True)
    corpus.add_argument("--exclude", type=Path, nargs="+", required=True)
    corpus.add_argument("--output", type=Path, required=True)
    corpus.add_argument("--seed", type=int, default=42)
    corpus.add_argument("--validation-fraction", type=float, default=0.05)
    corpus.add_argument("--test-fraction", type=float, default=0.05)
    corpus.add_argument("--max-tokens", type=int, default=2047)
    corpus.add_argument("--min-chars", type=int, default=20)
    corpus.add_argument("--workers", type=int, default=1, help="Parallel document tokenizers")
    corpus.add_argument("--split-overlong-words", action="store_true",
                        help="Split an overlong word at Unicode character boundaries")
    corpus.add_argument("--allow-content-groups", action="store_true")
    converted = commands.add_parser("prepare-eval")
    converted.add_argument("--profile", required=True)
    converted.add_argument("--mappings", type=Path, default=Path("configs/eval_mappings.json"))
    converted.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    converted.add_argument("--output", type=Path, required=True)
    converted.add_argument("--ids-file", type=Path)
    converted.add_argument("--limit", type=int)
    args = parser.parse_args()
    # No CLI data/model operation may run accidentally on the user's Windows device.
    from .datasets import require_remote
    require_remote()
    if args.command == "create-pilot":
        from .pilot import create_pilot
        create_pilot(args.output)
    elif args.command == "download-model":
        models = read_json(args.models_config)
        print(json.dumps(prepare_model(args.model, models[args.model], args.artifacts, args.weights), indent=2))
    elif args.command == "download-data":
        source = read_json(Path("configs/data_sources.json"))["unimorph_sah"]
        print(json.dumps(fetch_data(source, Path("data/raw/unimorph_sah")), indent=2))
    elif args.command in ("validate", "plan"):
        items = load_items(args.benchmark, args.allow_draft)
        result = {"items": len(items), "sha256": digest(args.benchmark),
                  "draft_items": sum(x["review_status"] != "approved" for x in items),
                  "prompt_version": PROMPT_VERSION}
        if args.command == "plan":
            result["first_prompt"], _ = make_prompt(items[0], "sah")
            result["forward_passes_per_model_per_language"] = sum(len(x["choices"]) for x in items)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if (args.stage == "adapted") != bool(args.checkpoint):
            parser.error("An adapted run requires --checkpoint; a base run must omit it")
        evaluate(args, read_json(args.models_config))
    elif args.command == "ensemble":
        ensemble(args)
    elif args.command == "report":
        report(args.runs)
    elif args.command == "fetch-source":
        from .datasets import download_source
        config = read_json(args.config)
        print(json.dumps(download_source(args.source,config[args.source],args.raw_root,
                         args.gated_access_granted,args.license_reviewed),ensure_ascii=False,indent=2))
    elif args.command == "inspect-source":
        from .datasets import inspect_source
        print(json.dumps(inspect_source(args.raw_root/args.source),ensure_ascii=False,indent=2))
    elif args.command == "prepare-corpus":
        from .convert import prepare_corpus
        result = prepare_corpus(args)
        print(json.dumps({"status":result["status"],"counts":result["counts"]},indent=2))
    elif args.command == "prepare-eval":
        from .convert import prepare_eval
        print(json.dumps(prepare_eval(args),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
