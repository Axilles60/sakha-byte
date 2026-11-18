"""Evaluate role and answer-wording changes on the existing Sakha possessor pairs."""

import hashlib
import sys
from pathlib import Path

from sakha_byte.benchmark import combine_rotations, make_prompt
from sakha_byte.datasets import file_sha, require_remote
from sakha_byte.runtime import HFScorer, software_versions
from sakha_suffix_pairs import FEATURES, TRANSFER, prompts
from saas_lora_transfer import ARTIFACTS, TRAINED, apply_transfer, read_json, save


SOURCE = Path("results/sakha-suffix-pairs-v2")
CONDITIONS = ("original", "context", "role_options", "dialogue_roles", "dialogue_names", "stem_probe")
ROLE_OPTIONS = ["Тому, кто произносит реплику", "Тому, к кому обращена реплика"]
PEOPLE = (("Айаана", "Айаане"), ("Сардана", "Сардане"))


def get_pairs():
    benchmark = SOURCE / "benchmark.json"
    pairs = [p for p in read_json(benchmark)["pairs"] if p["feature"] == "possessor"]
    assert len(pairs) == 37
    assert sum(p["cohort"] != "lexical_seen" for p in pairs) == 29
    for p in pairs:
        assert p["forms"] == [p["stem"] + s for s in ("м", "ҥ")]
    return pairs


def variants(p, idx, form_index, condition):
    if condition == "original":
        return prompts(p, form_index)

    template, question, choices = FEATURES["possessor"]
    form = p["stem"] if condition == "stem_probe" else p["forms"][form_index]
    context = template.format(form)
    speaker, listener = PEOPLE[idx % 2], PEOPLE[1 - idx % 2]
    if condition in ("context", "dialogue_roles", "dialogue_names"):
        context = (f"{speaker[0]} произносит реплику. Она обращается к {listener[1]}.\n"
                   f"Реплика на якутском: «{context}»")
    if condition in ("role_options", "dialogue_roles"):
        choices = ROLE_OPTIONS
    if condition == "dialogue_names":
        question = "Кто владелец предмета или существа, названного в реплике?"
        choices = [speaker[0], listener[0]]
    item = {"context_sah": context, "question_ru": question, "choices": choices}
    return [make_prompt(item, "sah", rotation) for rotation in range(2)]


def tasks(pairs):
    for condition in CONDITIONS:
        for idx, pair in enumerate(pairs):
            for form_index in range(1 if condition == "stem_probe" else 2):
                yield condition, pair, form_index, variants(pair, idx, form_index, condition)


def score(out, name, model, pairs, base_tokenizer=None):
    benchmark_sha = file_sha(SOURCE / "benchmark.json")
    old = read_json(SOURCE / f"{name}.json")
    assert old["benchmark_sha256"] == benchmark_sha
    assert old["manifest"] == model.manifest
    historical = {(row["pair_id"], row["answer"]): row for row in old["rows"]}
    rows = []
    for condition, p, form, rotations in tasks(pairs):
        hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt, _ in rotations]
        if condition == "original":
            assert hashes == historical[p["id"], form]["prompt_sha256"]
        scores, by_rotation = [], []
        for prompt, order in rotations:
            if base_tokenizer is not None:
                assert base_tokenizer.encode(prompt, add_special_tokens=False) == model.tokenizer.encode(
                    prompt, add_special_tokens=False)
            values, _ = model.score(prompt, 2)
            scores.append((values, order))
            canonical = [0.0, 0.0]
            for label, answer in enumerate(order):
                canonical[answer] = values[label]
            by_rotation.append(canonical)
        values = combine_rotations(scores)
        prediction = max(range(2), key=values.__getitem__)
        rows.append({"condition": condition, "pair_id": p["id"], "form_index": form, "prediction": prediction,
                     "scores": values, "rotation_scores": by_rotation, "prompt_sha256": hashes})
        last_form = 0 if condition == "stem_probe" else 1
        if p["id"] == pairs[-1]["id"] and form == last_form:
            print(f"{name}: {condition} complete", flush=True)
    matches = sum(row["prediction"] == historical[row["pair_id"], row["form_index"]]["prediction"]
                  for row in rows if row["condition"] == "original")
    save(out / f"{name}.json", {"benchmark_sha256": benchmark_sha, "model": name, "model_path": model.model_path,
                                "manifest": model.manifest, "software": software_versions(), "rows": rows,
                                "original_historical_matches": matches, "tokenizer_matches_qwen_base": base_tokenizer is not None})
    print(f"{name}: historical original predictions match {matches}/74", flush=True)


def main():
    require_remote()
    phase, dest = sys.argv[1:]
    out = Path(dest)
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
