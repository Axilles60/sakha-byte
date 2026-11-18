"""Verify four scores per pair and report paired differences on IMMERS."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from sakha_byte.datasets import file_sha, require_remote
from sakha_suffix_pairs import FEATURES, norm, prompts, read_json, save


MODELS = {
    "qwen-base": "Qwen3‑8B Base", "qwen-lora-200": "Qwen3‑8B + LoRA",
    "bwen-base": "Bwen‑8B Base", "bwen-lora-transfer": "Bwen‑8B + дельта LoRA",
}
COHORTS = {
    "nonce_unseen": "Искусственные основы, обе формы отсутствуют в корпусе LoRA",
    "lexical_unseen": "Словарные основы, обе формы отсутствуют в корпусе LoRA",
    "lexical_seen": "Контроль, обе словоформы найдены в корпусе LoRA",
}
NAMES = {"direction": "Направление", "possessor": "Владелец", "number": "Число"}


def read_predictions(out, pairs):
    import hashlib

    data = []
    for name in MODELS:
        result = read_json(out / f"{name}.json")
        assert result["benchmark_sha256"] == file_sha(out / "benchmark.json")
        assert len(result["rows"]) == 2 * len(pairs)
        predictions = []
        for p, a, b in zip(pairs, result["rows"][::2], result["rows"][1::2], strict=True):
            for expected, row in enumerate((a, b)):
                assert (row["pair_id"], row["answer"]) == (p["id"], expected)
                hashes = [hashlib.sha256(text.encode()).hexdigest() for text, _ in prompts(p, expected)]
                assert row["prompt_sha256"] == hashes
                assert row["prediction"] == max(range(2), key=row["scores"].__getitem__)
            predictions.append({"first": a["prediction"], "second": b["prediction"],
                                "response_shift": (b["scores"][1] - b["scores"][0])
                                - (a["scores"][1] - a["scores"][0])})
        data.append(predictions)
    return data


def summary(rows):
    n = len(rows)
    item_correct = sum((row["first"] == 0) + (row["second"] == 1) for row in rows)
    both_correct = sum(row["first"] == 0 and row["second"] == 1 for row in rows)
    return {"n_pairs": n, "n_correct_items": item_correct, "n_correct_pairs": both_correct,
            "n_flips": sum(row["first"] != row["second"] for row in rows),
            "n_both_first": sum(row["first"] == row["second"] == 0 for row in rows),
            "n_score_shifts_correct_direction": sum(row["response_shift"] > 0 for row in rows)}


def compare(pairs, data, selected):
    ok = np.array([[row["first"] == 0 and row["second"] == 1 for row in run] for run in data], dtype=int)
    clusters = defaultdict(list)
    for i in selected:
        clusters[norm(pairs[i]["stem"])].append(i)
    totals = np.array([ok[:, ids].sum(axis=1) for ids in clusters.values()])
    sizes = np.array([len(ids) for ids in clusters.values()])
    sample = np.random.default_rng(42).integers(len(clusters), size=(10000, len(clusters)))
    boots = totals[sample].sum(axis=1) / sizes[sample].sum(axis=1, keepdims=True)
    samples = (boots[:, 1] - boots[:, 0], boots[:, 3] - boots[:, 2],
               (boots[:, 3] - boots[:, 2]) - (boots[:, 1] - boots[:, 0]),
               boots[:, 2] - boots[:, 0], boots[:, 3] - boots[:, 1])
    base = ok[:, selected].mean(axis=1)
    points = (base[1] - base[0], base[3] - base[2], (base[3] - base[2]) - (base[1] - base[0]),
              base[2] - base[0], base[3] - base[1])
    out = {name: {"delta": float(point), "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}
           for name, point, draws in zip(("Qwen после − до", "Bwen после − до",
                   "Прирост Bwen − прирост Qwen", "Базовая Bwen − базовая Qwen",
                   "Адаптированная Bwen − адаптированная Qwen"), points, samples, strict=True)}
    for name, a, b in (("Qwen после − до", 0, 1), ("Bwen после − до", 2, 3)):
        change = ok[b, selected] - ok[a, selected]
        out[name].update(fixed=int((change == 1).sum()), broken=int((change == -1).sum()))
    return out


def row_scores(r):
    n = r["n_pairs"]
    items, pairs = r["n_correct_items"], r["n_correct_pairs"]
    shift = r["n_score_shifts_correct_direction"]
    return f"| {items}/{2*n} ({items/(2*n):.2%}) | {pairs}/{n} ({pairs/n:.2%}) | {shift}/{n} |"


def main(out):
    bench = read_json(out / "benchmark.json")
    pairs = bench["pairs"]
    data = read_predictions(out, pairs)
    for p in pairs:
        assert all(stem + suffix == form for stem, suffix, form in zip(
            [p["stem"]]*2, p["suffixes"], p["forms"], strict=True))
        assert all(p["form_docs"]) if p["cohort"] == "lexical_seen" else not any(p["form_docs"])
    selected = [i for i, p in enumerate(pairs) if p["cohort"] != "lexical_seen"]
    groups = {"unseen": selected, **{c: [i for i, p in enumerate(pairs) if p["cohort"] == c] for c in COHORTS}}
    for f in FEATURES:
        groups[f"unseen_{f}"] = [i for i in selected if pairs[i]["feature"] == f]
    groups["unseen_suffix_replacement"] = [i for i in selected if all(pairs[i]["suffixes"])]
    groups["all_possessor"] = [i for i, p in enumerate(pairs) if p["feature"] == "possessor"]
    agg = {group: {m: summary([run[i] for i in ids]) for m, run in zip(MODELS, data, strict=True)}
           for group, ids in groups.items() if ids}
    changes = compare(pairs, data, selected)
    task_rows = []
    for p in pairs:
        template, question, choices = FEATURES[p["feature"]]
        for idx, form in enumerate(p["forms"]):
            task_rows.append({"id": f"{p['id']}-{idx+1}", "pair_id": p["id"], "cohort": p["cohort"],
                              "feature": p["feature"], "context_sah": template.format(form),
                              "question_ru": question, "choices": choices, "answer_index": idx})
    save(out / "tasks.json", task_rows)
    save(out / "comparison.json", {"benchmark_sha256": file_sha(out / "benchmark.json"),
                                   "metrics": agg, "paired_changes_unseen": changes})

    lines = ["# Минимальные пары якутских словоформ", "",
             f"На IMMERS H100 четыре модели решили один и тот же тест из {len(pairs)} пар ({2*len(pairs)} задания). "
             f"В основной срез вошли {len(selected)} пар, у которых обе словоформы отсутствуют среди "
             f"{format(bench['training_documents'], ',').replace(',', ' ')} документов входного корпуса LoRA. Совпадения проверены по целым "
             "словам после нормализации NFC и приведения регистра. Состав предобучения Qwen и Bwen не проверялся.", "",
             "Основная метрика — доля пар, где верны **оба ответа**. Для каждого задания проверены оба порядка вариантов. "
             "Ответ A/B выбирали по средней логарифмической оценке условной вероятности. "
             "Последний столбец показывает, сколько раз смена суффикса сдвинула отношение оценок в правильную сторону, "
             "даже если по двум формам победил один и тот же вариант.", "",
             "## Новые словоформы", "",
             "| Модель | Задания верно | Обе формы верно | Сдвиг оценки в нужную сторону |",
             "|---|---:|---:|---:|"]
    lines.extend(f"| {MODELS[name]} {row_scores(r)}" for name, r in agg["unseen"].items())
    lines.extend(["", "Парные 95%-е интервалы получены бутстрэпом по основам (10 000 выборок). "
                  "Одна и та же основа попадает в выборку целиком во всех четырёх моделях.", "",
                  "| Сравнение по доле полностью решённых пар | Разница, п. п. [95% ДИ] | Исправлено / испорчено пар |",
                  "|---|---:|---:|"])
    for name, value in changes.items():
        low, high = value["ci95"]
        transition = f"{value['fixed']} / {value['broken']}" if "fixed" in value else "—"
        lines.append(f"| {name} | {100*value['delta']:+.2f} [{100*low:+.2f}, {100*high:+.2f}] | {transition} |")
    lines += ["", "## По происхождению форм и грамматическому контрасту", "",
              "| Срез | Qwen Base | Qwen + LoRA | Bwen Base | Bwen + дельта |",
              "|---|---:|---:|---:|---:|"]
    for subset, desc in [*(COHORTS.items()), *((f"unseen_{k}", v) for k, v in NAMES.items()),
                         ("unseen_suffix_replacement", "Новые формы с заменой одного ненулевого суффикса другим")]:
        if subset in agg:
            values = [f"{r['n_correct_pairs']}/{r['n_pairs']}" for r in agg[subset].values()]
            lines.append(f"| {desc} | {' | '.join(values)} |")
    owner = agg["all_possessor"]
    if all(r["n_both_first"] == r["n_pairs"] for r in owner.values()):
        n_owner = next(iter(owner.values()))["n_pairs"]
        lines += ["", f"Во всех {n_owner} парах на владельца четыре модели выбрали «мой» для обеих форм, "
                  "включая контрольные пары. Порядок вариантов был сбалансирован. Этот результат отражает "
                  "перекос ответов в данном формате задания."]
    lines += ["", "## Примеры и границы теста", "",
              "| Пара | Словоформы | Правильные интерпретации |",
              "|---|---|---|"]
    for feature in FEATURES:
        examples = [p for p in pairs if p["feature"] == feature and p["cohort"] == "lexical_unseen"]
        p = examples[0] if examples else next(p for p in pairs if p["feature"] == feature and p["cohort"] == "nonce_unseen")
        lines.append(f"| {p['id']} | {p['forms'][0]} / {p['forms'][1]} | {' / '.join(FEATURES[feature][2])} |")
    lines.extend(["", "Внутри пары сохраняются основа, предложение, вопрос и варианты ответа. "
                  "Меняется суффикс направления или владельца. В парах числа суффикс множественного числа добавляется "
                  "к форме единственного числа с нулевым показателем. Правила алломорфии взяты из описания имён "
                  "в якутском. Словарные основы отобраны из корпуса UD и списка названий мест. "
                  "[Описание имён в якутском](https://journal.sfu-kras.ru/en/article/109205) · "
                  "[Yakut Universal Dependencies](https://universaldependencies.org/sah/index.html) · "
                  "[Якутские названия улусов](https://sah.wikisource.org/wiki/Саха_Өрөспүүбүлүкэтин_Төрүт_Сокуона) · "
                  "[Чыаппара в Национальной библиотеке РС(Я)](https://nlrs.ru/to-readers/segments/iyakutia/open/95931)", "",
                  "Это пилот на коротких шаблонных фразах с инструкцией на русском. Искусственные основы "
                  "проверяют перенос правила на новую строку. Проверка носителем ещё не проводилась. "
                  "Контрольные пары с засвидетельствованными формами могут отличаться по лексике и частотности. "
                  "Для Bwen использована та же дельта 200 шагов из предыдущего опыта, применённая к 252 матрицам. "
                  "В журнале зафиксировано совпадение токенизации всех промптов между двумя Qwen. "
                  "Сравнение отражает поведение этих чекпойнтов и не изолирует влияние BPE от остальных различий.", "",
                  "Ранний прогон проверял формат. Затем были сверены топонимы и уточнён вопрос о грамматическом числе. "
                  "Значения в таблицах рассчитаны только для итогового набора. "
                  "[Журнал раннего прогона](preliminary/logs/01-evaluation.log)", "",
                  "[Задания с ключами](tasks.json) · [Пары и счётчики корпуса](benchmark.json) · [Полные метрики](comparison.json) · "
                  "[Журнал оценки](logs/01-evaluation.log)"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified four model results on {len(pairs)} pairs and all prompt hashes", flush=True)


if __name__ == "__main__":
    require_remote()
    main(Path(sys.argv[1]))
