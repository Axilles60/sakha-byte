"""Rerun the 101 Sakha pairs and matched calibration controls on IMMERS."""

import hashlib
import sys
from collections import Counter
from pathlib import Path

from sakha_byte.benchmark import combine_rotations
from sakha_byte.datasets import file_sha, require_remote
from sakha_byte.runtime import HFScorer, software_versions
from sakha_suffix_pairs import FEATURES, TRANSFER, prompts
from saas_lora_transfer import ARTIFACTS, TRAINED, apply_transfer, read_json, save


SOURCE = Path("results/sakha-suffix-pairs-v2")
POSSESSOR = Path("results/sakha-possessor-calibration")
MASKS = {"mask": "[…]", "alternate": "□"}


def stimulus(p, form):
    return prompts({**p, "forms": [form, form]}, 0)


def render_pair(p):
    forms = {f"form{i}": stimulus(p, word) for i, word in enumerate(p["forms"])}
    forms.update({name: stimulus(p, marker) for name, marker in MASKS.items()})
    if p["feature"] != "number":
        forms["stem"] = stimulus(p, p["stem"])
    else:
        assert p["stem"] == p["forms"][0]
    return forms


def score(out, name, model, benchmark, base_tokenizer=None):
    path = SOURCE / f"{name}.json"
    prior = read_json(path)
    controls = read_json(POSSESSOR / f"{name}-controls.json")
    assert prior["benchmark_sha256"] == controls["benchmark_sha256"] == benchmark
    assert prior["manifest"] == controls["manifest"] == model.manifest
    assert prior["model_path"] == controls["model_path"] == model.model_path
    old_forms = {(r["pair_id"], r["answer"]): r for r in prior["rows"]}
    old_stems = {r["pair_id"]: r for r in controls["rows"] if r["condition"] == "original"}
    cache, rows = {}, []
    max_form_delta = max_stem_delta = 0.0
    prediction_matches = 0
    pairs = read_json(SOURCE / "benchmark.json")["pairs"]
    assert Counter(p["feature"] for p in pairs) == {"number": 32, "direction": 32, "possessor": 37}
    for p in pairs:
        pair_id = p["id"]
        data = {}
        for label, renders in render_pair(p).items():
            per_order, raw = [], []
            hashes = [hashlib.sha256(text.encode()).hexdigest() for text, _ in renders]
            for text, order in renders:
                if text not in cache:
                    if base_tokenizer is not None:
                        assert base_tokenizer.encode(text, add_special_tokens=False) == model.tokenizer.encode(
                            text, add_special_tokens=False)
                    cache[text] = model.score(text, 2)[0]
                values = cache[text]
                per_order.append([values[order.index(i)] for i in range(2)])
                raw.append((values, order))
            data[label] = {"rotation_scores": per_order, "prompt_sha256": hashes}
            if label.startswith("form"):
                i = int(label[-1])
                old = old_forms[pair_id, i]
                assert hashes == old["prompt_sha256"]
                values = combine_rotations(raw)
                max_form_delta = max(max_form_delta, *(abs(x-y) for x, y in zip(values, old["scores"], strict=True)))
                prediction_matches += max(range(2), key=values.__getitem__) == old["prediction"]
            if label == "stem" and p["feature"] == "possessor":
                old = old_stems[pair_id]
                assert hashes == old["prompt_sha256"]
                max_stem_delta = max(max_stem_delta, *(abs(x-y)
                    for a, b in zip(per_order, old["rotation_scores"], strict=True) for x, y in zip(a, b, strict=True)))
        rows.append({"pair_id": pair_id, "stimuli": data})
        if len(rows) % 25 == 0:
            print(f"{name}: {len(rows)}/{len(pairs)} pairs", flush=True)
    save(out / f"{name}.json", {"benchmark_sha256": benchmark, "software": software_versions(),
                                "model_path": model.model_path, "manifest": model.manifest,
                                "old_form_scores_sha256": file_sha(path), "old_possessor_scores_sha256": file_sha(
                                    POSSESSOR / f"{name}-controls.json"),
                                "max_form_score_delta": max_form_delta, "max_possessor_control_score_delta": max_stem_delta,
                                "historical_predictions_matched": prediction_matches, "distinct_model_calls": len(cache),
                                "rows": rows})
    print(f"{name}: {len(rows)} pairs complete, {len(cache)} prompt calls, original matches "
          f"{prediction_matches}/202; max form/control deltas {max_form_delta:.8g}/{max_stem_delta:.8g}", flush=True)


def main():
    require_remote()
    phase, dest = sys.argv[1:]
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    benchmark = file_sha(SOURCE / "benchmark.json")
    specs = read_json(Path("configs/models.json"))
    if phase == "qwen":
        base = HFScorer("qwen", specs["qwen"], ARTIFACTS)
        score(out, "qwen-base", base, benchmark)
        tokenizer = base.tokenizer
        del base
        tuned = HFScorer("qwen", specs["qwen"], ARTIFACTS, TRAINED / "merged")
        score(out, "qwen-lora-200", tuned, benchmark, tokenizer)
    elif phase == "bwen":
        model = HFScorer("bwen", specs["bwen"], ARTIFACTS)
        evidence = read_json(TRANSFER)
        assert evidence["bwen"] == model.manifest
        score(out, "bwen-base", model, benchmark)
        apply_transfer(model, evidence, out)
        score(out, "bwen-lora-transfer", model, benchmark)
    else:
        raise ValueError(phase)


if __name__ == "__main__":
    main()
