"""Apply matched control subtraction and report paired Qwen–Bwen contrasts."""

import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from sakha_byte.datasets import file_sha, require_remote
from sakha_possessor_calibration_report import TIE_TOLERANCE, action, margins
from sakha_suffix_calibrated import POSSESSOR, SOURCE, render_pair
from sakha_suffix_pairs import norm
from sakha_suffix_pairs_report import MODELS
from saas_lora_transfer import read_json, save


MODES = {
    "raw": "Без калибровки",
    "primary": "Основной контроль",
    "alternate": "Другой символ маски",
    "direction_stem": "Основа для направления",
    "all_mask": "Маска для всех трёх категорий",
}
FEATURE_LABELS = {"number": "Число", "direction": "Направление", "possessor": "Владелец"}
CONTRASTS = {"base": ("qwen-base", "bwen-base"), "adapted": ("qwen-lora-200", "bwen-lora-transfer")}


def control_for(mode, feature):
    if mode == "all_mask":
        return "mask"
    if feature == "possessor" or (feature == "direction" and mode == "direction_stem"):
        return "stem"
    return "alternate" if mode == "alternate" else "mask"


def checked_scores(out, pairs, renders):
    data = {}
    for name in MODELS:
        bundle = read_json(out / f"{name}.json")
        assert bundle["benchmark_sha256"] == file_sha(SOURCE / "benchmark.json")
        assert bundle["historical_predictions_matched"] == 2*len(pairs)
        assert bundle["max_form_score_delta"] < 1e-6 and bundle["max_possessor_control_score_delta"] < 1e-6
        scores = []
        for p, expected, row in zip(pairs, renders, bundle["rows"], strict=True):
            assert p["id"] == row["pair_id"] and expected.keys() == row["stimuli"].keys()
            scores.append({})
            for label, rotations in expected.items():
                observed = row["stimuli"][label]
                assert observed["prompt_sha256"] == [hashlib.sha256(prompt.encode()).hexdigest() for prompt, _ in rotations]
                assert [order for _, order in rotations] == [[0, 1], [1, 0]]
                scores[-1][label] = margins(observed)
        data[name] = scores
    return data


def adjust(p, stimuli, mode):
    threshold = [0.0, 0.0] if mode == "raw" else stimuli[control_for(mode, p["feature"])]
    raw = [sum(stimuli[f"form{i}"]) / 2 for i in range(2)]
    adjusted = [sum(a-b for a, b in zip(stimuli[f"form{i}"], threshold, strict=True)) / 2 for i in range(2)]
    assert math.isclose(raw[1] - raw[0], adjusted[1] - adjusted[0], abs_tol=1e-9)
    prediction = [int(v > 0) if mode == "raw" else action(v) for v in adjusted]
    return {"pair_id": p["id"], "feature": p["feature"], "cohort": p["cohort"], "predictions": prediction,
            "scores": adjusted, "control_scores": threshold,
            "items_correct": sum(value == gold for gold, value in enumerate(prediction)),
            "pair_correct": prediction == [0, 1], "ties": prediction.count(None),
            "separation_correct": adjusted[1] - adjusted[0] > TIE_TOLERANCE}


def counts(rows):
    return {"n_pairs": len(rows), "items_correct": sum(r["items_correct"] for r in rows),
            "pairs_correct": sum(r["pair_correct"] for r in rows), "ties": sum(r["ties"] for r in rows),
            "separation_correct": sum(r["separation_correct"] for r in rows),
            "correct_ids": [r["pair_id"] for r in rows if r["pair_correct"]]}


def gap(qwen, bwen, indices, pairs):
    clusters = defaultdict(list)
    for idx in indices:
        clusters[norm(pairs[idx]["stem"])].append(idx)
    pair_values = np.asarray([int(q["pair_correct"]) - int(b["pair_correct"])
                              for q, b in zip(qwen, bwen, strict=True)])
    n = np.asarray([len(v) for v in clusters.values()])
    totals = np.asarray([pair_values[v].sum() for v in clusters.values()])
    samples = np.random.default_rng(42).integers(len(n), size=(10000, len(n)))
    draws = totals[samples].sum(axis=1) / n[samples].sum(axis=1)
    return {"delta": float(pair_values[indices].mean()),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist() if np.unique(draws).size > 1 else None,
            "qwen_only": int((pair_values[indices] == 1).sum()),
            "bwen_only": int((pair_values[indices] == -1).sum())}


def parse_gap(data):
    interval = data["ci95"]
    values = [] if interval is None else [f"{x*100:+.1f}".replace(".", ",") for x in interval]
    label = f"{data['delta']*100:+.1f}".replace(".", ",")
    return f"{label} [{'; '.join(values)}]" if interval is not None else f"{label} [—]"


def main(out):
    pairs = read_json(SOURCE / "benchmark.json")["pairs"]
    renders = [render_pair(p) for p in pairs]
    data = checked_scores(out, pairs, renders)
    groups = {"all": list(range(len(pairs))),
              "unseen": [i for i, p in enumerate(pairs) if p["cohort"] != "lexical_seen"],
              **{f: [i for i, p in enumerate(pairs) if p["feature"] == f] for f in FEATURE_LABELS}}
    results, metric, gaps = {}, {}, {}
    for mode in MODES:
        results[mode] = {name: [adjust(p, r, mode) for p, r in zip(pairs, scores, strict=True)]
                         for name, scores in data.items()}
        metric[mode] = {subset: {name: counts([rows[i] for i in ids]) for name, rows in results[mode].items()}
                        for subset, ids in groups.items()}
        gaps[mode] = {
            subset: {stage: gap(results[mode][q], results[mode][b], ids, pairs) for stage, (q, b) in CONTRASTS.items()}
            for subset, ids in groups.items()
        }
    for name in MODELS:
        for subset in groups:
            assert len({metric[m][subset][name]["separation_correct"] for m in MODES}) == 1
    previous_possessor = read_json(POSSESSOR / "comparison.json")
    for name in MODELS:
        assert sorted(metric["primary"]["possessor"][name]["correct_ids"]) == previous_possessor[
            "metrics"]["all"]["original"][name]["calibrated_correct_pair_ids"]
    save(out / "predictions.json", results)
    save(out / "comparison.json", {"benchmark_sha256": file_sha(SOURCE / "benchmark.json"),
                                   "tie_tolerance": TIE_TOLERANCE, "metrics": metric, "qwen_minus_bwen": gaps})
    unique = {hashlib.sha256(text.encode()).hexdigest(): {"prompt": text, "order": order}
              for pair in renders for label, rotations in pair.items() if not label.startswith("form")
              for text, order in rotations}
    save(out / "control-prompts.json", unique)

    lines = ["# Якутские минимальные пары — сохраняется ли преимущество Qwen после калибровки?", "",
             "На IMMERS H100 для каждого из четырёх чекпойнтов заново оценили обе формы **всех 101 пар** "
             "и контрольные стимулы в обоих порядках A/B. В составе теста 32 пары на число, 32 на направление "
             "и 37 на владельца. Проверены хеши промптов и совпадение 202 исходных предсказаний каждого чекпойнта с архивом.", "",
             "Основное сравнение сохраняет прежний контроль для владельца — та же основа без притяжательного суффикса. "
             "На числе и направлении целевую словоформу заменили на `[…]`. Для числа голая основа совпадает "
             "с формой единственного числа, поэтому этот контроль сделал бы первый ответ ничьей по построению. "
             "Для направления дополнительно проверили название места без падежного суффикса. Оно сохраняет лексику, "
             "но может звучать неестественно в предложении. Также проверили другой символ маски `□` на числе и "
             "направлении и режим с одной маской для всех трёх категорий.", "",
             "Из логарифмического перевеса второго ответа над первым, отдельно для каждой перестановки, вычли "
             "такой же перевес на контрольном промпте. Затем усреднили два значения. Абсолютный перевес "
             "до 0,000001 считали ничьей. Пара засчитана, если обе формы распознаны верно.", "",
             "## Полностью решённые пары из 101", "",
             "| Контроль | Qwen Base | Qwen + LoRA | Bwen Base | Bwen + дельта | Qwen − Bwen, базы | Qwen − Bwen, после адаптации |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, title in MODES.items():
        scores = metric[mode]["all"]
        cells = [str(scores[m]["pairs_correct"]) for m in MODELS]
        cells += [parse_gap(gaps[mode]["all"][stage]) for stage in CONTRASTS]
        lines.append(f"| {title} | {' | '.join(cells)} |")
    lines += ["", "Разница указана в процентных пунктах с парным 95%-м бутстрэп-интервалом "
              "по основам. Повторяющаяся основа перевыбиралась целиком для обеих моделей. "
              "Интервалы описывают задания данного пилота.", "",
              "На основном контроле Qwen решила больше пар в обеих конфигурациях, но оба интервала включают ноль. "
              "После адаптации порядок моделей меняется при контроле с названием места без падежного суффикса. "
              "На срезе новых словоформ с основным контролем адаптированная Bwen решила на одну пару больше. "
              "Следовательно, преимущество по числу полностью решённых пар зависит от контроля и среза.", "",
              "## Основной контроль по грамматическим контрастам", "",
              "| Набор | Qwen Base | Qwen + LoRA | Bwen Base | Bwen + дельта |",
              "|---|---:|---:|---:|---:|"]
    for subset, title in (*FEATURE_LABELS.items(), ("unseen", "Обе формы отсутствовали в корпусе LoRA")):
        n = len(groups[subset])
        cells = [f"{metric['raw'][subset][name]['pairs_correct']} → {metric['primary'][subset][name]['pairs_correct']}"
                 for name in MODELS]
        lines.append(f"| {title} ({n}) | {' | '.join(cells)} |")
    lines.extend(["", "Показано число верных пар до и после калибровки.", "",
                  "У каждой пары вычитание одной и той же оценки из обеих форм сохраняет относительный сдвиг между "
                  "суффиксами. Меняется порог между вариантами. Это позволяет отличить чувствительность относительной "
                  "оценки к форме от способности правильно классифицировать обе формы при заданном пороге. "
                  "На 101 паре правильное направление сдвига наблюдалось в "
                  + ", ".join(f"{metric['raw']['all'][name]['separation_correct']} случаях у {title}"
                              for name, title in MODELS.items()) + ". В JSON сохранены точность отдельных "
                  "ответов, ничьи и ID пар.", "",
                  "Контроли со скрытым словом оценивают предпочтение в шаблоне. Символ маски сам может менять "
                  "это предпочтение и разбивается токенизаторами по-разному. Тест состоит из коротких шаблонных фраз "
                  "и сравнивает эти чекпойнты целиком. Разметка остаётся пилотной, без проверки носителем.", "",
                  "[Сравнение по моделям и категориям](comparison.json) · [Оценки по парам](predictions.json) · "
                  "[Контрольные промпты](control-prompts.json) · [Журнал](logs/01-evaluation.log)"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified {len(pairs)} pairs and all control hashes for {len(MODELS)} models", flush=True)


if __name__ == "__main__":
    require_remote()
    main(Path(sys.argv[1]))
