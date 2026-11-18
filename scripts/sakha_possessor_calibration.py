"""Score matched suffixless controls for the existing possessor prompts on IMMERS."""

import hashlib
import sys
from pathlib import Path

from sakha_byte.datasets import file_sha, require_remote
from sakha_byte.runtime import HFScorer, software_versions
from sakha_possessor_dialogue import CONDITIONS, SOURCE, get_pairs, variants
from sakha_suffix_pairs import TRANSFER
from saas_lora_transfer import ARTIFACTS, TRAINED, apply_transfer, read_json, save


PRIOR = Path("results/sakha-possessor-dialogue")
PROMPTS = tuple(condition for condition in CONDITIONS if condition != "stem_probe")


def controls(pairs):
    for condition in PROMPTS:
        for idx, p in enumerate(pairs):
            # Render every original question, role assignment and option order with the bare stem.
            bare = {**p, "forms": [p["stem"], p["stem"]]}
            yield condition, p, variants(bare, idx, 0, condition)


def score(out, name, model, pairs, base_tokenizer=None):
    source = PRIOR / f"{name}.json"
    prior = read_json(source)
    assert prior["benchmark_sha256"] == file_sha(SOURCE / "benchmark.json")
    assert prior["manifest"] == model.manifest and prior["model_path"] == model.model_path
    original = {r["pair_id"]: r for r in prior["rows"] if r["condition"] == "stem_probe"}
    max_original_error = 0.0
    rows = []
    for condition, p, renders in controls(pairs):
        hashes, values = [], []
        for text, order in renders:
            hashes.append(hashlib.sha256(text.encode()).hexdigest())
            if base_tokenizer is not None:
                assert base_tokenizer.encode(text, add_special_tokens=False) == model.tokenizer.encode(
                    text, add_special_tokens=False)
            rotation, _ = model.score(text, 2)
            values.append([rotation[order.index(i)] for i in range(2)])
        if condition == "original":
            before = original[p["id"]]
            assert hashes == before["prompt_sha256"]
            max_original_error = max(max_original_error, *(abs(x - y)
                for a, b in zip(values, before["rotation_scores"], strict=True)
                for x, y in zip(a, b, strict=True)))
        rows.append({"condition": condition, "pair_id": p["id"], "prompt_sha256": hashes,
                     "rotation_scores": values})
        if p is pairs[-1]:
            print(f"{name}: {condition}, {len(pairs)} controls scored", flush=True)
    assert max_original_error <= 1e-6, max_original_error
    save(out / f"{name}-controls.json", {
        "benchmark_sha256": prior["benchmark_sha256"], "prior_scores_sha256": file_sha(source),
        "manifest": model.manifest, "software": software_versions(),
        "model_path": model.model_path, "max_original_control_score_difference": max_original_error,
        "qwen_base_tokenizer_verified": base_tokenizer is not None, "rows": rows})
    print(f"{name}: all 37 original controls reproduced; max score error {max_original_error:.8g}", flush=True)


def main():
    require_remote()
    phase, destination = sys.argv[1:]
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    pairs = get_pairs()
    specs = read_json(Path("configs/models.json"))
    if phase == "qwen":
        base = HFScorer("qwen", specs["qwen"], ARTIFACTS)
        score(out, "qwen-base", base, pairs)
        tokenizer = base.tokenizer
        del base
        tuned = HFScorer("qwen", specs["qwen"], ARTIFACTS, TRAINED / "merged")
        score(out, "qwen-lora-200", tuned, pairs, tokenizer)
    elif phase == "bwen":
        model = HFScorer("bwen", specs["bwen"], ARTIFACTS)
        evidence = read_json(TRANSFER)
        assert evidence["bwen"] == model.manifest
        score(out, "bwen-base", model, pairs)
        apply_transfer(model, evidence, out)
        score(out, "bwen-lora-transfer", model, pairs)
    else:
        raise ValueError(phase)


if __name__ == "__main__":
    main()
