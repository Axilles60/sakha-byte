"""Remote-only LoRA baseline: continued LM training on supplied Sakha JSONL documents.

Input schema: {"text": "...", "source": "...", "document_id": "..."}.
Split documents/sources before providing these files. Never train on pilot fixtures.
"""

import argparse
import hashlib
import json
import random
import unicodedata
from itertools import chain
from pathlib import Path

from .benchmark import load_items
from .download import read_json, write_json


def read_documents(path: Path):
    documents, hashes = [], set()
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) and row[k].strip()
                       for k in ("text", "source", "document_id")):
                raise ValueError("Each document needs nonempty text/source/document_id")
            text = unicodedata.normalize("NFC", row["text"]).strip()
            key = hashlib.sha256(text.encode()).hexdigest()
            if key not in hashes:
                hashes.add(key)
                documents.append({**row, "text": text})
    if not documents:
        raise ValueError("No documents")
    return documents, hashes


def sample_rows(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return rows
    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


def main():
    from .convert import make_exclusion_matcher
    from .datasets import file_sha, require_remote
    require_remote()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen_lora.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--corpus-manifest", type=Path,
                        help="Verify prepare-corpus output hashes and retain preprocessing provenance")
    parser.add_argument("--smoke-test", action="store_true",
                        help="One optimizer step with small splits; still save and merge the adapter")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Training output must be a new directory")
    config = read_json(args.config)
    if args.smoke_test:
        config.update(max_steps=1, gradient_accumulation_steps=1, eval_steps=1, save_steps=1,
                      max_validation_rows=8)
    validation_limit = config.get("max_validation_rows")
    if validation_limit is not None and (type(validation_limit) is not int or validation_limit < 1):
        raise ValueError("max_validation_rows must be a positive integer or null")
    train_sha256 = file_sha(args.train_file)
    validation_sha256 = file_sha(args.validation_file)
    corpus_provenance = None
    if args.corpus_manifest:
        corpus_provenance = read_json(args.corpus_manifest)
        if corpus_provenance.get("status") != "complete":
            raise ValueError("Corpus preparation is incomplete")
        for key,sha256 in (("train",train_sha256),("validation",validation_sha256)):
            if corpus_provenance["outputs"][key] != sha256:
                raise ValueError(f"Prepared {key} data changed after conversion")
    train, train_hashes = read_documents(args.train_file)
    validation, validation_hashes = read_documents(args.validation_file)
    train_ids = {(x["source"], x["document_id"]) for x in train}
    validation_ids = {(x["source"], x["document_id"]) for x in validation}
    if train_hashes & validation_hashes or train_ids & validation_ids:
        raise ValueError("Train/validation document overlap")
    load_items(args.benchmark, allow_draft=True)
    excluded = make_exclusion_matcher([args.benchmark])
    if any(excluded(row["text"]) for row in chain(train, validation)):
        raise ValueError("Benchmark context found in corpus; inspect contamination")
    available_train_rows = len(train)
    available_validation_rows = len(validation)
    if args.smoke_test:
        train = sample_rows(train, 4, config["seed"])
    validation = sample_rows(validation, validation_limit, config["seed"])

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments, set_seed)
    from .runtime import software_versions

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    set_seed(config["seed"])
    manifest = read_json(args.artifacts / "models/qwen.json")
    expected = read_json(Path("configs/models.json"))["qwen"]
    if not manifest["weights"] or manifest["revision"] != expected["revision"]:
        raise ValueError("Prepare pinned Qwen weights first")
    model_path = Path(manifest["path"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    def encode(row):
        ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(ids) > config["max_length"]:
            raise ValueError("Document too long: prepare shorter, source-grouped passages; no silent truncation")
        return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": ids.copy()}

    prepared_tokenizer = (corpus_provenance or {}).get("tokenizer_files", {})
    lazy_tokenization = (
        {"tokenizer.json", "tokenizer_config.json"} <= prepared_tokenizer.keys()
        and corpus_provenance["max_tokens_without_eos"] + 1 <= config["max_length"]
        and all((model_path / name).is_file() and file_sha(model_path / name) == expected_hash
                for name, expected_hash in prepared_tokenizer.items())
    )
    if lazy_tokenization:
        class PreparedDataset(torch.utils.data.Dataset):
            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return encode(self.rows[index])

        train_dataset = PreparedDataset(train)
        validation_dataset = PreparedDataset(validation)
    else:
        train_dataset = Dataset.from_list(train).map(encode, remove_columns=list(train[0]))
        validation_dataset = Dataset.from_list(validation).map(encode, remove_columns=list(validation[0]))
    model = AutoModelForCausalLM.from_pretrained(
        manifest["path"], local_files_only=True, torch_dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    args.output.mkdir(parents=True)
    write_json(args.output / "training_manifest.json", {
        "status": "configured", "config": config, "base_model": expected, "smoke_test": args.smoke_test,
        "train_sha256": train_sha256, "validation_sha256": validation_sha256,
        "benchmark_excluded_sha256": file_sha(args.benchmark), "software": software_versions(),
        "train_documents": len(train), "validation_documents": len(validation),
        "available_train_rows": available_train_rows, "available_validation_rows": available_validation_rows,
        "lazy_tokenization": lazy_tokenization,
        "corpus_preparation":corpus_provenance})
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"), max_steps=config["max_steps"],
            learning_rate=config["learning_rate"], bf16=True, per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps", eval_steps=config["eval_steps"],
            save_steps=config.get("save_steps", config["eval_steps"]), save_total_limit=2, logging_steps=10,
            report_to="none", seed=config["seed"]),
        train_dataset=train_dataset, eval_dataset=validation_dataset,
        # Pad labels with -100, preserving actual EOS targets despite pad_id == eos_id.
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(args.output / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(args.output / "merged", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "merged")
    manifest_out = read_json(args.output / "training_manifest.json")
    manifest_out["status"] = "complete"
    write_json(args.output / "training_manifest.json", manifest_out)


if __name__ == "__main__":
    main()
