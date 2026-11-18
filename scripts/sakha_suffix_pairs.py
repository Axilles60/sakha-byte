"""Build and score Sakha minimal pairs on IMMERS."""

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from sakha_byte.benchmark import combine_rotations, make_prompt
from sakha_byte.datasets import file_sha, require_remote
from sakha_byte.runtime import HFScorer, software_versions
from saas_lora_transfer import ARTIFACTS, TRAINED, apply_transfer, read_json, save


CORPUS = Path("data/corpus/qwen-full-repaired/train.jsonl")
UD = Path("data/raw/ud_yakut/files/sah_yktdt-ud-test.conllu")
TRANSFER = Path("results/saas-lora-transfer-582/compatibility.json")
VOWELS = "аоэө"
FEATURES = {
    "direction": ("Кини {} айаннаата.", "Какое значение передаёт форма названия места в предложении?",
                  ["Место, куда направились", "Место, откуда направились"]),
    "possessor": ("Бу {}.", "Кому принадлежит предмет или существо?",
                  ["Говорящему, «мой»", "Собеседнику, «твой»"]),
    "number": ("Көр: {}.", "Каково грамматическое число показанной словоформы?",
               ["Единственное число", "Множественное число"]),
}
# Regular vowel-final allomorphs described by Torotoev and Torotoeva (2022).
SUFFIXES = {
    "direction": dict(zip(VOWELS, [("ҕа", "ттан"), ("ҕо", "ттон"), ("ҕэ", "ттэн"), ("ҕө", "ттөн")])),
    "possessor": {v: ("м", "ҥ") for v in VOWELS},
    "number": dict(zip(VOWELS, [("", "лар"), ("", "лор"), ("", "лэр"), ("", "лөр")])),
}
PLACES = ("Амма", "Маайа", "Ньурба", "Таатта", "Томпо", "Усуйаана", "Чыаппара",
          "Табаҕа", "Суола", "Хоро", "Марха")
IRREGULAR_NUMBER = {"оҕо", "ийэ", "аҕа", "эбэ", "эһэ", "нуучча"}
NOUNS_EXCLUDED = {"яблоко", "хойчоонно", "күөгүрээпийэ", "олоҥхо", "сыа", "баҕа", "мүнүүтэ", "сөкүүндэ"}
ONSETS = ("чаҥ", "кэр", "сом", "мөл", "дьаҥ", "бөл", "тэр", "нор")
ENDINGS = {"а": ("нара", "лара", "таха", "мача"), "о": ("норо", "лоро", "тохо", "мочо"),
           "э": ("нэрэ", "лэрэ", "тэхэ", "мэчэ"), "ө": ("нөрө", "лөрө", "төхө", "мөчө")}


def norm(text):
    return unicodedata.normalize("NFC", text.casefold())


def pair(feature, stem, origin):
    suffixes = SUFFIXES[feature][stem[-1]]
    return {"feature": feature, "stem": stem, "origin": origin, "suffixes": suffixes,
            "forms": [stem + suffix for suffix in suffixes]}


def candidates():
    nouns = set()
    for line in UD.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) == 10 and fields[0].isdigit() and fields[3] == "NOUN":
            stem = norm(fields[2])
            if stem.isalpha() and stem[-1] in VOWELS and stem not in NOUNS_EXCLUDED:
                nouns.add(stem)
    for feature in FEATURES:
        real = sorted(nouns - (IRREGULAR_NUMBER if feature == "number" else set()))
        if feature == "direction":
            # Single-word, vowel-final place names with a transparent final vowel.
            real = sorted(stem for stem in PLACES if stem.isalpha() and stem[-1] in VOWELS)
        for stem in real:
            yield pair(feature, stem, "lexical")
        for vowel in VOWELS:
            prefixes = [p for p in ONSETS if vowel in p]
            for i, onset in enumerate(prefixes):
                for j, ending in enumerate(ENDINGS[vowel]):
                    # Different stems for each feature, matched for the final vowel.
                    bridge = {"direction": "б", "possessor": "ҕ" if vowel in "ао" else "г", "number": "д"}[feature]
                    stem = onset + vowel + bridge + vowel + ending
                    if feature == "direction":
                        stem = stem.capitalize()
                    yield pair(feature, stem, "nonce")


def build(out):
    inventory = list(candidates())
    corpus_hash = file_sha(CORPUS)
    assert corpus_hash == read_json(TRAINED / "training_manifest.json")["train_sha256"]
    targets = {norm(form) for p in inventory for form in [p["stem"], *p["forms"]]}
    token = re.compile(r"[^\W\d_]+")
    counts = Counter()
    n = 0
    with CORPUS.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            text = norm(json.loads(line)["text"])
            counts.update(targets.intersection(token.findall(text)))
    assert n == read_json(TRAINED / "training_manifest.json")["train_documents"]
    ranked = {}
    for p in inventory:
        counts_here = [counts[norm(form)] for form in p["forms"]]
        stem_count = counts[norm(p["stem"])]
        if p["origin"] == "nonce" and not any(counts_here) and not stem_count:
            cohort = "nonce_unseen"
        elif p["origin"] == "lexical" and not any(counts_here):
            cohort = "lexical_unseen"
        elif p["origin"] == "lexical" and all(counts_here):
            cohort = "lexical_seen"
        else:
            continue
        key = (cohort, p["feature"])
        ranked.setdefault(key, []).append({**p, "cohort": cohort, "stem_docs": stem_count, "form_docs": counts_here})

    selected = []
    for (cohort, feature), pool in sorted(ranked.items()):
        if cohort == "nonce_unseen":
            for vowel in VOWELS:
                by_vowel = [p for p in pool if p["stem"][-1] == vowel]
                assert len(by_vowel) >= 6
                selected.extend(by_vowel[:6])
        else:
            # Alternate final vowels to retain different allomorphs with a fixed cap.
            buckets = [[p for p in pool if p["stem"][-1] == vowel] for vowel in VOWELS]
            order = [bucket[i] for i in range(max(map(len, buckets))) for bucket in buckets if i < len(bucket)]
            selected.extend(order[:12 if cohort == "lexical_unseen" else 8])
    assert sum(p["cohort"] == "nonce_unseen" for p in selected) == 72
    assert len({(p["feature"], *p["forms"]) for p in selected}) == len(selected)
    for i, p in enumerate(selected, 1):
        p["id"] = f"sakha-pair-{i:03d}"
        for suffix, form in zip(p["suffixes"], p["forms"], strict=True):
            assert form == p["stem"] + suffix
    save(out / "benchmark.json", {"corpus_sha256": corpus_hash, "training_documents": n,
                                  "ud_sha256": file_sha(UD), "pairs": selected, "review_status": "draft"})
    print(Counter((p["cohort"], p["feature"]) for p in selected), flush=True)


def prompts(p, answer):
    template, question, choices = FEATURES[p["feature"]]
    item = {"context_sah": template.format(p["forms"][answer]), "question_ru": question, "choices": choices}
    return [make_prompt(item, "sah", r) for r in range(2)]


def score(out, name, model, pairs):
    rows = []
    for p in pairs:
        for answer in range(2):
            rotations = prompts(p, answer)
            values = combine_rotations([(model.score(prompt, 2)[0], order) for prompt, order in rotations])
            prediction = max(range(2), key=values.__getitem__)
            rows.append({"pair_id": p["id"], "answer": answer, "prediction": prediction,
                         "scores": values, "prompt_sha256": [hashlib.sha256(s.encode()).hexdigest() for s, _ in rotations]})
    save(out / f"{name}.json", {"benchmark_sha256": file_sha(out / "benchmark.json"), "manifest": model.manifest,
                                "model_path": model.model_path, "software": software_versions(), "rows": rows})
    print(f"{name}: {len(rows)} items, {sum(r['answer'] == r['prediction'] for r in rows)} correct", flush=True)


def main():
    require_remote()
    phase, destination = sys.argv[1:]
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    if phase == "build":
        build(out)
        return
    pairs = read_json(out / "benchmark.json")["pairs"]
    specs = read_json(Path("configs/models.json"))
    if phase == "qwen":
        base = HFScorer("qwen", specs["qwen"], ARTIFACTS)
        score(out, "qwen-base", base, pairs)
        tokenizer = base.tokenizer
        del base
        adapted = HFScorer("qwen", specs["qwen"], ARTIFACTS, TRAINED / "merged")
        for p in pairs:
            for answer in range(2):
                for prompt, _ in prompts(p, answer):
                    assert tokenizer.encode(prompt, add_special_tokens=False) == adapted.tokenizer.encode(
                        prompt, add_special_tokens=False)
        print(f"Qwen base and merged tokenizer IDs identical on {4 * len(pairs)} prompts", flush=True)
        score(out, "qwen-lora-200", adapted, pairs)
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
