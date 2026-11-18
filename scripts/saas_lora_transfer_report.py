"""Summarize the paired SAAS transfer run on IMMERS without copying source texts."""

import json
import math
import sys
from pathlib import Path

import numpy as np

from sakha_byte.datasets import require_remote
from saas_lora_transfer import BENCHMARK_SHA, save


MODELS = {
    "qwen-base": "Qwen3‑8B Base",
    "qwen-lora-200": "Qwen3‑8B + LoRA 200",
    "bwen-base": "Bwen‑8B Base",
    "bwen-lora-transfer": "Bwen‑8B + дельта LoRA",
}
DOMAINS = {"forum": "Forum YKT", "kyym": "Kyym"}
PAIRS = {"Qwen": (0, 1), "Bwen": (2, 3)}


def f1(confusion):
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    total = confusion.sum(axis=-2) + confusion.sum(axis=-1)
    return np.divide(2 * tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0).mean(axis=-1)


def summarize(gold, prediction):
    confusion = np.bincount(gold * 3 + prediction, minlength=9).reshape(3, 3)
    correct = int(confusion.trace())
    return {"n": len(gold), "correct": correct, "accuracy": correct / len(gold),
            "macro_f1": float(f1(confusion)), "confusion": confusion.tolist()}


def transition(gold, before, after):
    prior, current = before == gold, after == gold
    fixed = int((~prior & current).sum())
    broken = int((prior & ~current).sum())
    n = fixed + broken
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixed, broken) + 1)) / 2**n)
    return {"fixed": fixed, "broken": broken, "predictions_changed": int((before != after).sum()),
            "mcnemar_exact_two_sided_p_accuracy": p}


def intervals(gold, predictions, domains, rng):
    # Preserve the two domain sizes and resample the same examples for all four models.
    samples = np.concatenate([rng.choice(ids, (10000, len(ids)), replace=True) for ids in domains], axis=1)
    metrics = []
    for pred in predictions:
        cells = (gold * 3 + pred)[samples]
        confusion = np.stack([(cells == k).sum(axis=1) for k in range(9)], axis=1).reshape(-1, 3, 3)
        metrics.append(np.stack((np.trace(confusion, axis1=1, axis2=2) / len(gold), f1(confusion)), axis=1))
    qwen, bwen = metrics[1] - metrics[0], metrics[3] - metrics[2]
    return {name: {metric: np.percentile(delta[:, i], [2.5, 97.5]).tolist()
                   for i, metric in enumerate(("accuracy", "macro_f1"))}
            for name, delta in (("Qwen", qwen), ("Bwen", bwen), ("Bwen−Qwen", bwen - qwen))}


def main(out):
    inputs = [json.loads((out / f"{key}.json").read_text(encoding="utf-8")) for key in MODELS]
    reference = inputs[0]["rows"]
    fields = ("id", "domain", "answer_index", "prompt_sha256")
    assert len(reference) == len({row["id"] for row in reference}) == 582
    for result in inputs:
        assert result["benchmark_sha256"] == BENCHMARK_SHA and result["rules"] == inputs[0]["rules"]
        assert result["prior_prompt_hashes_verified"] == 1746
        for expected, actual in zip(reference, result["rows"], strict=True):
            assert all(expected[key] == actual[key] for key in fields)
            assert actual["prediction"] == max(range(3), key=actual["scores"].__getitem__)
            assert actual["correct"] == (actual["prediction"] == actual["answer_index"])

    gold = np.asarray([row["answer_index"] for row in reference])
    pred = np.asarray([[row["prediction"] for row in result["rows"]] for result in inputs])
    labels = np.asarray([row["domain"] for row in reference])
    domains = {name: np.flatnonzero(labels == name) for name in DOMAINS}
    assert {name: len(ids) for name, ids in domains.items()} == {"forum": 146, "kyym": 436}
    groups = {"all": np.arange(len(gold)), **domains}
    aggregates, transitions = {}, {}
    for group, ids in groups.items():
        actual, predictions = gold[ids], pred[:, ids]
        aggregates[group] = {key: summarize(actual, p) for key, p in zip(MODELS, predictions, strict=True)}
        transitions[group] = {name: transition(actual, predictions[before], predictions[after])
                              for name, (before, after) in PAIRS.items()}

    scores = list(aggregates["all"].values())
    metrics = ("accuracy", "macro_f1")
    differences = {name: {metric: scores[after][metric] - scores[before][metric] for metric in metrics}
                   for name, (before, after) in PAIRS.items()}
    differences["Bwen−Qwen"] = {metric: differences["Bwen"][metric] - differences["Qwen"][metric]
                               for metric in metrics}
    ci = intervals(gold, pred, list(domains.values()), np.random.default_rng(42))
    diffs = {name: {metric: {"delta": values[metric], "paired_bootstrap_95": ci[name][metric]}
                    for metric in metrics} for name, values in differences.items()}
    neutral = summarize(gold, np.full_like(gold, 2))
    evidence = json.loads((out / "compatibility.json").read_text(encoding="utf-8"))
    update = json.loads((out / "bwen-update.json").read_text(encoding="utf-8"))
    assert [entry["weight"] for entry in evidence["mapping"]] == [entry["weight"] for entry in update]
    checkpoint = json.loads((out / "transferred-checkpoint.json").read_text(encoding="utf-8"))
    assert Path(checkpoint["path"]).is_dir() and checkpoint["verified_matrices"] == len(update)
    changed = sum(row["changed_after_bf16"] for row in update)
    total = sum(row["elements"] for row in update)
    reproducibility = {key: result["matches_historical_predictions"]
                       for key, result in zip(MODELS, inputs, strict=True) if key != "bwen-lora-transfer"}
    save(out / "comparison.json", {"benchmark_sha256": BENCHMARK_SHA, "class_order": ["positive", "negative", "neutral"],
                                   "metrics": aggregates, "adaptation_deltas": diffs, "paired_transitions": transitions,
                                   "neutral_baseline": neutral, "historical_prediction_matches": reproducibility,
                                   "bwen_weight_elements_changed": changed, "bwen_weight_elements_targeted": total,
                                   "adapted_models_disagreements": int((pred[1] != pred[3]).sum())})

    lines = ["# SAAS — перенос якутской LoRA из Qwen3‑8B в Bwen‑8B", "",
             "Четыре модели оценены заново на IMMERS H100 на одних и тех же 582 заданиях SAAS "
             "после аудита пересечений с обучающим корпусом. Для каждого задания использовали неизменный промпт "
             "с определениями классов и три циклические перестановки вариантов. У всех моделей сверены "
             "1 746 хешей промптов с предыдущим прогоном.", "", f"> {inputs[0]['rules']}", "",
             "## Точность и Macro‑F1", "", "| Модель | Верно из 582 | Точность | Macro‑F1 |",
             "|---|---:|---:|---:|"]
    for name, summary in zip(MODELS.values(), scores, strict=True):
        lines.append(f"| {name} | {summary['correct']} | {summary['accuracy']:.2%} | {summary['macro_f1']:.4f} |")
    lines.extend([f"| Всегда нейтральная | {neutral['correct']} | {neutral['accuracy']:.2%} | {neutral['macro_f1']:.4f} |", "",
                  "95%-е интервалы рассчитаны парным бутстрэпом по заданиям с сохранением размеров доменов "
                  "(10 000 выборок). Macro‑F1 одинаково взвешивает все три класса. Интервалы отражают "
                  "вариативность по заданиям для этой пары чекпойнтов.", "",
                  "| Сравнение | Δ точности, п. п. [95% ДИ] | Δ Macro‑F1 [95% ДИ] |",
                  "|---|---:|---:|"])
    for name, values in diffs.items():
        formatted = []
        for metric, scale, digits in (("accuracy", 100, 2), ("macro_f1", 1, 4)):
            value = values[metric]
            lower, upper = value["paired_bootstrap_95"]
            delta, lower, upper = (round(scale * x, digits) or 0.0 for x in (value["delta"], lower, upper))
            formatted.append(f"{delta:+.{digits}f} [{lower:+.{digits}f}, {upper:+.{digits}f}]")
        label = "Разница приростов Bwen и Qwen" if name == "Bwen−Qwen" else f"{name} после − до"
        lines.append(f"| {label} | {formatted[0]} | {formatted[1]} |")
    lines += ["", "## Ошибки по доменам", "",
              "| Домен | Модель | Точность до → после | Macro‑F1 до → после | Ошибки до → после | Исправлено | Испорчено |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        for family, (first, second) in PAIRS.items():
            before, after = [aggregates[group][list(MODELS)[i]] for i in (first, second)]
            pair = transitions[group][family]
            n = before["n"]
            lines.append(f"| {DOMAINS.get(group, 'Все')} ({n}) | {family} | {before['accuracy']:.2%} → {after['accuracy']:.2%} | "
                         f"{before['macro_f1']:.4f} → {after['macro_f1']:.4f} | {n - before['correct']} → {n - after['correct']} | "
                         f"{pair['fixed']} | {pair['broken']} |")
    lines += ["", "## Что было перенесено", "",
              f"В чекпойнте после {evidence['step']} шагов LoRA ранг равен {evidence['rank']}, α = {evidence['alpha']}. "
              f"Совпадение имён, форм и назначения проверено для всех {len(update)} матриц внимания и MLP "
              f"в {evidence['shared_config']['num_hidden_layers']} глобальных слоях. Для каждой матрицы Bwen "
              f"вычислили W + (α/r)·BA в FP32 и привели к BF16 для инференса. После округления изменилось "
              f"{changed / total:.2%} элементов целевых матриц. Локальные байтовые блоки и выходная голова Bwen "
              "сохранили исходные веса. Базовую и адаптированную Bwen оценили в одном процессе. "
              "Для Qwen использовали сохранённое слияние той же LoRA с её родительским чекпойнтом.", "",
              f"Проверка слияния Qwen дала точное совпадение {evidence['qwen_merge_samples']['exact']} из "
              f"{evidence['qwen_merge_samples']['total']} проверенных значений. Количество ответов, совпавших "
              f"с предыдущей оценкой — {', '.join(f'{MODELS[key]} {count}/582' for key, count in reproducibility.items())}.", "",
              "Базовый и объединённый токенизаторы Qwen выдали одинаковые последовательности для всех 1 746 промптов "
              "и трёх букв ответа. Слитый чекпойнт Bwen сохранён на IMMERS. Путь записан в "
              "[манифесте чекпойнта](transferred-checkpoint.json).", "",
              "LoRA обучалась предсказывать текст на якутском корпусе. SAAS измеряет, как эта языковая адаптация "
              "повлияла на классификацию тональности. Сравнение охватывает одну адаптацию на 200 шагов и один бенчмарк.", "",
              "[Полные метрики и матрицы ошибок](comparison.json) · [Карта соответствия слоёв](compatibility.json) · "
              "[Статистика изменений весов](bwen-update.json) · [Журнал оценки](logs/02-evaluation.log)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Verified all 4 × 582 predictions and 1,746 prompts per model. Report complete.", flush=True)


if __name__ == "__main__":
    require_remote()
    main(Path(sys.argv[1]))
