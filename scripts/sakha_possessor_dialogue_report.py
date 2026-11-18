"""Verify and compare the dialogue and response wording conditions on IMMERS."""

import hashlib
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from sakha_byte.datasets import file_sha, require_remote
from sakha_possessor_dialogue import CONDITIONS, SOURCE, get_pairs, tasks
from sakha_suffix_pairs_report import MODELS
from saas_lora_transfer import read_json, save


TITLES = {
    "original": "Исходный промпт",
    "context": "Только контекст",
    "role_options": "Только формулировки ролей",
    "dialogue_roles": "Диалог + роли",
    "dialogue_names": "Диалог + имена",
}


def summary(pairs):
    n = len(pairs)
    first = sum(a["prediction"] == 0 for a, _ in pairs)
    second = sum(b["prediction"] == 1 for _, b in pairs)
    both = sum(a["prediction"] == 0 and b["prediction"] == 1 for a, b in pairs)
    odds_first = [a["scores"][1] - a["scores"][0] for a, _ in pairs]
    odds_second = [b["scores"][1] - b["scores"][0] for _, b in pairs]
    shifts = [b - a for a, b in zip(odds_first, odds_second, strict=True)]
    raw = Counter("speaker" if p[0] > p[1] else "listener" if p[1] > p[0] else "tie"
                  for pair in pairs for row in pair for p in row["rotation_scores"])
    by_order = [dict(Counter("speaker" if p[0] > p[1] else "listener" if p[1] > p[0] else "tie"
                             for pair in pairs for row in pair for p in [row["rotation_scores"][i]]))
                for i in range(2)]
    return {"n_pairs": n, "n_correct_items": first + second, "n_correct_pairs": both,
            "speaker_forms_correct": first, "listener_forms_correct": second,
            "speaker_predictions": first + (n - second),
            "both_chose_speaker": sum(a["prediction"] == b["prediction"] == 0 for a, b in pairs),
            "both_chose_listener": sum(a["prediction"] == b["prediction"] == 1 for a, b in pairs),
            "reversed_pairs": sum(a["prediction"] == 1 and b["prediction"] == 0 for a, b in pairs),
            "median_log_odds_listener_on_speaker_form": median(odds_first),
            "median_log_odds_listener_on_listener_form": median(odds_second),
            "median_suffix_shift": median(shifts), "positive_suffix_shifts": sum(s > 0 for s in shifts),
            "unaveraged_rotation_choices": dict(raw), "choices_by_order": by_order}


def paired_gain(reference, condition):
    def outcomes(pairs):
        return np.array([a["prediction"] == 0 and b["prediction"] == 1 for a, b in pairs], dtype=int)

    change = outcomes(condition) - outcomes(reference)
    sampled = np.random.default_rng(42).integers(len(change), size=(10000, len(change)))
    boots = change[sampled].mean(axis=1)
    return {"delta": float(change.mean()), "ci95": np.percentile(boots, [2.5, 97.5]).tolist()
            if np.unique(change).size > 1 else None,
            "fixed": int((change == 1).sum()), "broken": int((change == -1).sum())}


def main(out):
    pairs = get_pairs()
    benchmark_sha = file_sha(SOURCE / "benchmark.json")
    expected = list(tasks(pairs))
    prompts = []
    for cond, p, form, rotations in expected:
        prompts.append({"condition": cond, "pair_id": p["id"], "cohort": p["cohort"],
                        "form": p["stem"] if cond == "stem_probe" else p["forms"][form],
                        "gold_role": None if cond == "stem_probe" else ("speaker", "listener")[form],
                        "rotations": [{"prompt": prompt, "role_order": [("speaker", "listener")[i] for i in order],
                                       "gold_label": None if cond == "stem_probe" else "AB"[order.index(form)]}
                                      for prompt, order in rotations]})
    save(out / "prompts.json", prompts)
    save(out / "pairs.json", {"source_benchmark_sha256": benchmark_sha, "pairs": pairs})
    groups = {"all": pairs,
              "unseen": [p for p in pairs if p["cohort"] != "lexical_seen"],
              **{cohort: [p for p in pairs if p["cohort"] == cohort]
                 for cohort in ("nonce_unseen", "lexical_unseen", "lexical_seen")}}
    metrics, changes, probes, reruns = {}, {}, {}, {}
    for name in MODELS:
        result = read_json(out / f"{name}.json")
        assert result["benchmark_sha256"] == benchmark_sha
        scores = {}
        for (cond, p, form, rotations), row in zip(expected, result["rows"], strict=True):
            key = cond, p["id"], form
            assert key == (row["condition"], row["pair_id"], row["form_index"])
            assert row["prompt_sha256"] == [hashlib.sha256(text.encode()).hexdigest() for text, _ in rotations]
            assert row["prediction"] == max(range(2), key=row["scores"].__getitem__)
            for i, value in enumerate(row["scores"]):
                assert abs(value - sum(r[i] for r in row["rotation_scores"]) / 2) < 1e-9
            scores[key] = row
        reruns[name] = result["original_historical_matches"]
        metrics[name], changes[name] = {}, {}
        for group, items in groups.items():
            slices = {cond: [(scores[cond, p["id"], 0], scores[cond, p["id"], 1]) for p in items]
                      for cond in TITLES}
            metrics[name][group] = {cond: summary(values) for cond, values in slices.items()}
            changes[name][group] = {cond: paired_gain(slices["original"], values)
                                    for cond, values in slices.items() if cond != "original"}
        masked = [scores["stem_probe", p["id"], 0] for p in pairs]
        probes[name] = {"speaker_predictions": sum(row["prediction"] == 0 for row in masked),
                        "median_log_odds_listener": median(row["scores"][1] - row["scores"][0] for row in masked)}
    save(out / "comparison.json", {"benchmark_sha256": benchmark_sha, "conditions": TITLES, "metrics": metrics,
                                   "paired_changes": changes, "stem_probe": probes, "original_rerun_matches": reruns})

    lines = ["# Якутские пары на владельца — роль контекста и вариантов ответа", "",
             "На IMMERS H100 повторно проверены все 37 пар из исходного теста. В каждой паре "
             "меняется только форма с суффиксом «мой» или «твой». У 29 пар обе словоформы отсутствовали "
             "в корпусе LoRA, у восьми обе встречались. Модели и способ выбора между A/B совпадают с исходным прогоном.", "",
             "| Вариант | Контекст | Варианты ответа |", "|---|---|---|",
             "| Исходный промпт | Реплика «Бу …» | «Говорящему, “мой”» / «Собеседнику, “твой”» |",
             "| Только контекст | Кто произнёс реплику и к кому обратился | Те же |",
             "| Только формулировки ролей | Исходная реплика | «Тому, кто произносит реплику» / «Тому, к кому обращена реплика» |",
             "| Диалог + роли | Кто произнёс реплику и к кому обратился | Формулировки ролей |",
             "| Диалог + имена | Тот же диалог | Имена участников. Вопрос «Кто владелец предмета или существа, названного в реплике?» |",
             "", "Во всех диалогах Айаана и Сардана чередуются в роли говорящей между парами. Обе формы каждой пары "
             "проверены с одинаковыми ролями и обоими порядками A/B. В отдельном контрольном промпте оставлена основа "
             "без притяжательного суффикса. Для него измеряли предпочтение ответа без подсчёта правильности.", "",
             "## Сравнение с исходным промптом", "",
             "Пара засчитана, если правильно распознаны обе формы. В столбце «твой» указано, сколько форм "
             "второго лица получили правильный ответ.", "",
             "| Модель | Промпт | Верных ответов из 74 | Пары из 37 | Верно на «твой» из 37 | Всего выборов говорящего из 74 | Пары с новыми формами из 29 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, title in MODELS.items():
        for cond, label in TITLES.items():
            row = metrics[name]["all"][cond]
            lines.append(f"| {title} | {label} | {row['n_correct_items']} | {row['n_correct_pairs']} | "
                         f"{row['listener_forms_correct']} | {row['speaker_predictions']} | "
                         f"{metrics[name]['unseen'][cond]['n_correct_pairs']} |")
    lines += ["", "При добавлении одного контекста все модели решили часть пар. Замена формулировок без контекста "
              "помогла сильнее обеим Qwen, но Bwen с дельтой продолжила выбирать говорящего на всех заданиях. "
              "Сочетание того же диалога с описаниями ролей перевернуло предпочтение у всех четырёх моделей. "
              "Они выбирали собеседницу для обеих форм каждой пары. Варианты с именами дали умеренный прирост.", "",
              "Парный 95%-й интервал для разницы долей решённых пар рассчитан по 10 000 бутстрэп-выборок "
              "из тех же 37 пар. Интервалы отражают вариативность заданий внутри пилота. Когда правильность совпала "
              "на каждой паре, перевыборка дала бы вырожденный интервал, поэтому он опущен.", "",
              "| Модель | Диалог + роли, Δ п. п. [95% ДИ] | Диалог + имена, Δ п. п. [95% ДИ] |",
              "|---|---:|---:|"]
    for name, title in MODELS.items():
        cells = []
        for cond in ("dialogue_roles", "dialogue_names"):
            d = changes[name]["all"][cond]
            if d["ci95"] is None:
                cells.append(f"{100*d['delta']:+.1f} [—]")
            else:
                low, high = d["ci95"]
                cells.append(f"{100*d['delta']:+.1f} [{100*low:+.1f}, {100*high:+.1f}]")
        lines.append(f"| {title} | {' | '.join(cells)} |")
    lines += ["", "## Почему исходная модель выбирала «мой»", "",
              "Оценка в таблице — логарифмический перевес ответа «собеседнику» над ответом «говорящему» "
              "после усреднения по двум порядкам. Отрицательное значение означает, что побеждает «говорящему». "
              "Сдвиг сравнивает две формы одной основы при замене суффикса первого лица суффиксом второго.", "",
              "| Модель | Медиана на форме «мой» | Медиана на форме «твой» | Медиана без суффикса | "
              "Без суффикса выбрано «мой», из 37 | Сдвиг в сторону «твой», из 37 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, title in MODELS.items():
        r, probe = metrics[name]["all"]["original"], probes[name]
        lines.append(f"| {title} | {r['median_log_odds_listener_on_speaker_form']:+.3f} | "
                     f"{r['median_log_odds_listener_on_listener_form']:+.3f} | {probe['median_log_odds_listener']:+.3f} | "
                     f"{probe['speaker_predictions']} | {r['positive_suffix_shifts']} |")
    lines += ["", "### Выбор говорящего в каждом порядке по отдельности", "",
              "| Модель | Говорящий под A, выборов из 74 | Говорящий под B, выборов из 74 | Всего выборов B из 148 |",
              "|---|---:|---:|---:|"]
    for name, title in MODELS.items():
        a, b = metrics[name]["all"]["original"]["choices_by_order"]
        lines.append(f"| {title} | {a.get('speaker', 0)} | {b.get('speaker', 0)} | "
                     f"{a.get('listener', 0) + b.get('speaker', 0)} |")
    lines += ["",
              "У Qwen выбор сильно зависел от позиции B. После усреднения оценок по перестановкам говорящий "
              "побеждал на каждом задании. У Bwen говорящий обычно побеждал в обоих порядках. "
              "Сдвиг оценки внутри пары показывает чувствительность скорера к форме. Он сам по себе не доказывает "
              "знание правила. Контроль с основой измеряет предпочтение варианта в вынужденном выборе, "
              "поскольку у такой формы нет ключа «мой» или «твой». Сравнение «только контекст» и «только формулировки» "
              "оценивает эффекты выбранных шаблонов. При замене слов меняются и словесные ассоциации, "
              "и способ выражения ролей, поэтому эффекты нельзя приписать одному слову.", "",
              "## Пример", "",
              "Айаана произносит реплику и обращается к Сардане. «Бу дуоскам» относится к Айаане, "
              "а «Бу дуоскаҥ» — к Сардане. В выгрузке промптов сохранены обе перестановки A/B.", "",
              "Совпадения повторного исходного прогона с архивом — "
              + ", ".join(f"{MODELS[m]} {n}/74" for m, n in reruns.items()) + ". "
              "Разметка исходного теста остаётся пилотной, без проверки носителем.", "",
              "[Метрики и частоты по перестановкам](comparison.json) · [Тексты всех промптов](prompts.json) · "
              "[Журнал оценки](logs/01-evaluation.log)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified {len(expected)} conditions/items per model with both rotations", flush=True)


if __name__ == "__main__":
    require_remote()
    main(Path(sys.argv[1]))
