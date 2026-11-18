"""Run paired SAAS evaluations and transfer a fixed Qwen LoRA into Bwen on IMMERS."""

import hashlib
import json
import re
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

from sakha_byte.benchmark import combine_rotations, load_items, make_prompt
from sakha_byte.datasets import file_sha, require_remote
from sakha_byte.download import read_json
from sakha_byte.runtime import HFScorer, software_versions


BENCHMARK = Path("data/eval/saas-clean-v1/benchmark.jsonl")
BENCHMARK_SHA = "a0c97877fe6f9cc292372f99119da0be3d33dca61e390c37380f648676088b4c"
PRIOR = Path("results/saas-definitions-582")
ARTIFACTS = Path("artifacts")
TRAINED = ARTIFACTS / "qwen-sah-full-200"
TRANSFERRED = ARTIFACTS / "bwen-sah-qwen-lora-200"
PROJECTIONS = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def save(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_index(path):
    return read_json(path / "model.safetensors.index.json")["weight_map"]


def open_checkpoint(stack, path):
    from safetensors import safe_open

    index = read_index(path)
    shards = {name: stack.enter_context(safe_open(path / name, framework="pt"))
              for name in sorted(set(index.values()))}
    return {key: shards[name] for key, name in index.items()}


def inspect_transfer(output):
    import torch
    from safetensors import safe_open

    qwen = read_json(ARTIFACTS / "models/qwen.json")
    bwen = read_json(ARTIFACTS / "models/bwen.json")
    training_file = TRAINED / "training_manifest.json"
    training = read_json(training_file)
    adapter_dir = TRAINED / "adapter"
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    config = read_json(adapter_config)
    trainer_state = read_json(TRAINED / "checkpoints/checkpoint-200/trainer_state.json")
    assert trainer_state["global_step"] == 200
    assert Path(config["base_model_name_or_path"]).resolve() == Path(qwen["path"]).resolve()
    assert training["base_model"]["revision"] == qwen["revision"]
    expected_config = {"peft_type": "LORA", "bias": "none", "lora_bias": False,
                       "use_dora": False, "use_rslora": False, "fan_in_fan_out": False,
                       "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": None,
                       "target_parameters": None, "trainable_token_indices": None,
                       "layer_replication": None}
    assert all(config[key] == value for key, value in expected_config.items())
    assert set(config["target_modules"]) == {p for names in PROJECTIONS.values() for p in names}
    rank, scale = config["r"], config["lora_alpha"] / config["r"]
    q_config = read_json(Path(qwen["path"]) / "config.json")
    b_config = read_json(Path(bwen["path"]) / "config.json")
    shared_fields = ("hidden_size", "intermediate_size", "num_hidden_layers", "head_dim",
                     "num_attention_heads", "num_key_value_heads", "attention_bias",
                     "hidden_act", "rms_norm_eps", "rope_theta", "sliding_window")
    shared = {key: q_config[key] for key in shared_fields}
    assert all(b_config[key] == value for key, value in shared.items())
    assert b_config["use_qk_norm"] and b_config["use_head_qk_norm"]
    assert b_config["norm_type"] == "qwen_rms"
    assert b_config["layer_types"] == ["full_attention"] * shared["num_hidden_layers"]
    assert not q_config["use_sliding_window"]

    mapping = []
    with ExitStack() as stack:
        source = open_checkpoint(stack, Path(qwen["path"]))
        target = open_checkpoint(stack, Path(bwen["path"]))
        merged = open_checkpoint(stack, TRAINED / "merged")
        adapter = stack.enter_context(safe_open(adapter_file, framework="pt"))
        expected_keys = set()
        for layer in range(shared["num_hidden_layers"]):
            for block, projections in PROJECTIONS.items():
                for proj in projections:
                    module = f"model.layers.{layer}.{block}.{proj}"
                    weight = f"{module}.weight"
                    a, b = (f"base_model.model.{module}.lora_{side}.weight" for side in "AB")
                    expected_keys.update((a, b))
                    dims = source[weight].get_slice(weight).get_shape()
                    assert dims == target[weight].get_slice(weight).get_shape()
                    assert dims == merged[weight].get_slice(weight).get_shape()
                    assert adapter.get_slice(a).get_shape() == [rank, dims[1]]
                    assert adapter.get_slice(b).get_shape() == [dims[0], rank]
                    mapping.append({"weight": weight, "shape": dims, "a": a, "b": b})
        assert expected_keys == set(adapter.keys())

        # Check the direction and scale against the saved Qwen merge, sampling every matrix.
        sampled = exact = within_one_bf16_step = 0
        for entry in mapping:
            key = entry["weight"]
            row_count, column_count = entry["shape"]
            for row in (0, row_count // 2, row_count - 1):
                start = column_count // 3
                a = adapter.get_slice(entry["a"])[:, start:start + 32].float()
                b = adapter.get_slice(entry["b"])[row:row + 1, :].float()
                base = source[key].get_slice(key)[row:row + 1, start:start + 32].float()
                saved = merged[key].get_slice(key)[row:row + 1, start:start + 32].float()
                expected = (base + scale * (b @ a)).to(torch.bfloat16)
                low = torch.nextafter(expected, torch.full_like(expected, -torch.inf)).float()
                high = torch.nextafter(expected, torch.full_like(expected, torch.inf)).float()
                exact += (expected.float() == saved).sum().item()
                within_one_bf16_step += ((saved >= low) & (saved <= high)).sum().item()
                sampled += saved.numel()
        assert exact / sampled > 0.95 and within_one_bf16_step == sampled, (exact, sampled, within_one_bf16_step)

    evidence = {"qwen": qwen, "bwen": bwen, "shared_config": shared, "step": 200,
                "rank": rank, "alpha": config["lora_alpha"], "scale": scale,
                "adapter_sha256": file_sha(adapter_file), "adapter_config_sha256": file_sha(adapter_config),
                "training_manifest_sha256": file_sha(training_file),
                "qwen_merge_samples": {"total": sampled, "exact": exact,
                                       "within_one_bf16_step": within_one_bf16_step},
                "mapping": mapping}
    save(output / "compatibility.json", evidence)
    print(f"Compatible: {len(mapping)} global matrices; Qwen merge sample exact {exact}/{sampled}", flush=True)
    return evidence


def apply_transfer(scorer, evidence, out):
    import torch
    from safetensors import safe_open

    adapter_file = TRAINED / "adapter/adapter_model.safetensors"
    assert file_sha(adapter_file) == evidence["adapter_sha256"]
    model = scorer.model
    stats = []
    with ExitStack() as stack, torch.no_grad():
        target = open_checkpoint(stack, Path(evidence["bwen"]["path"]))
        adapter = stack.enter_context(safe_open(adapter_file, framework="pt"))
        for entry in evidence["mapping"]:
            key = entry["weight"]
            parameter = model.get_parameter(key)
            assert list(parameter.shape) == entry["shape"] and parameter.dtype == torch.bfloat16
            a = adapter.get_tensor(entry["a"]).to(parameter.device, torch.float32)
            b = adapter.get_tensor(entry["b"]).to(parameter.device, torch.float32)
            original = target[key].get_tensor(key).to(parameter.device, torch.float32)
            assert torch.equal(original.to(parameter.dtype), parameter), key
            delta = (b @ a).mul_(evidence["scale"])
            updated = (original + delta).to(parameter.dtype)
            changed = torch.count_nonzero(updated != parameter).item()
            assert changed > 0 and torch.isfinite(updated).all().item(), key
            stats.append({"weight": key, "changed_after_bf16": changed, "elements": parameter.numel(),
                          "relative_frobenius_norm": (delta.norm() / original.norm()).item()})
            parameter.copy_(updated)
    save(out / "bwen-update.json", stats)
    print(f"Transferred all {len(stats)} matrices at unit transfer strength", flush=True)


def save_transferred_model(scorer, evidence, out, prompt):
    import torch
    from transformers import AutoTokenizer

    final = TRANSFERRED.resolve()
    temporary = final.with_name(final.name + ".partial")
    if final.exists():
        raise FileExistsError(final)
    temporary.mkdir()
    scorer.model.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
    scorer.tokenizer.save_pretrained(temporary)
    for path in Path(evidence["bwen"]["path"]).glob("*.py"):
        shutil.copy2(path, temporary / path.name)
    tokenizer = AutoTokenizer.from_pretrained(temporary, trust_remote_code=True, local_files_only=True)
    assert tokenizer.encode(prompt, add_special_tokens=False) == scorer.tokenizer.encode(prompt, add_special_tokens=False)
    with ExitStack() as stack:
        saved = open_checkpoint(stack, temporary)
        assert saved.keys() == scorer.model.state_dict().keys()
        for entry in evidence["mapping"]:
            key = entry["weight"]
            sample = saved[key].get_slice(key)[:4, :16]
            assert torch.equal(sample, scorer.model.get_parameter(key)[:4, :16].detach().cpu()), key
    temporary.rename(final)
    save(out / "transferred-checkpoint.json", {"path": str(final), "dtype": "bfloat16", "verified_matrices": len(evidence["mapping"]),
                                             "adapter_sha256": evidence["adapter_sha256"], "bwen_revision": evidence["bwen"]["revision"]})
    print(f"Saved and verified merged Bwen on IMMERS at {final}", flush=True)


def dataset():
    assert file_sha(BENCHMARK) == BENCHMARK_SHA
    items = load_items(BENCHMARK, allow_draft=True)
    rules = read_json(Path("results/saas-error-audit/comparison.json"))["rules"]
    assert rules == read_json(PRIOR / "bwen.json")["rules"]
    assert len(items) == 582 and all(len(item["choices"]) == 3 for item in items)
    prompts = []
    for item in items:
        assert re.fullmatch(r"saas:(forum|kyym):(train|val|test):\d+", item["id"]), item["id"]
        choices = []
        for rot in range(3):
            original, order = make_prompt(item, "sah", rot)
            start, end = original.split("\n\n", 1)
            choices.append((f"{start}\n{rules}\n{end}", order))
        prompts.append(choices)
    return items, prompts, rules


def score(out, name, scorer, data, prior_name):
    items, prompts, rules = data
    prior = read_json(PRIOR / f"{prior_name}.json")
    assert prior["benchmark_sha256"] == BENCHMARK_SHA and prior["rules"] == rules
    assert len(prior["rows"]) == len(items)
    rows = []
    for i, (item, rotations, old) in enumerate(zip(items, prompts, prior["rows"], strict=True), 1):
        reference_id, gold = item["id"], item["answer_index"]
        hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt, _ in rotations]
        assert (reference_id, gold, hashes) == (old["id"], old["answer_index"], old["prompt_sha256"])
        values = combine_rotations([(scorer.score(prompt, 3)[0], order) for prompt, order in rotations])
        prediction = max(range(3), key=values.__getitem__)
        rows.append({"id": reference_id, "domain": reference_id.split(":")[1], "category": "sentiment",
                     "answer_index": gold, "prediction": prediction, "correct": prediction == gold,
                     "scores": values, "n_choices": 3, "prompt_sha256": hashes})
        if i == 1 or i % 100 == 0:
            print(f"{name}: {i}/{len(items)}", flush=True)
    agreements = sum(row["prediction"] == old["prediction"] for row, old in zip(rows, prior["rows"], strict=True))
    save(out / f"{name}.json", {"name": name, "benchmark_sha256": BENCHMARK_SHA, "rules": rules,
                                 "checkpoint": scorer.model_path, "manifest": scorer.manifest,
                                 "software": software_versions(), "prior_prompt_hashes_verified": 3 * len(rows),
                                 "matches_historical_predictions": agreements, "rows": rows})
    print(f"{name}: {len(rows)}/{len(rows)} complete, {sum(row['correct'] for row in rows)} correct", flush=True)


def main():
    require_remote()
    phase, directory = sys.argv[1:]
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    if phase == "inspect":
        inspect_transfer(out)
        dataset()
        return
    data = dataset()
    models = read_json(Path("configs/models.json"))
    if phase == "qwen":
        score(out, "qwen-base", HFScorer("qwen", models["qwen"], ARTIFACTS), data, "qwen-base")
        scorer = HFScorer("qwen", models["qwen"], ARTIFACTS, TRAINED / "merged")
        score(out, "qwen-lora-200", scorer, data, "qwen-lora-200")
    elif phase == "bwen":
        evidence = read_json(out / "compatibility.json")
        scorer = HFScorer("bwen", models["bwen"], ARTIFACTS)
        assert scorer.manifest == evidence["bwen"]
        score(out, "bwen-base", scorer, data, "bwen")
        apply_transfer(scorer, evidence, out)
        score(out, "bwen-lora-transfer", scorer, data, "bwen")
        save_transferred_model(scorer, evidence, out, data[1][0][0][0])
    else:
        raise ValueError(phase)


if __name__ == "__main__":
    main()
