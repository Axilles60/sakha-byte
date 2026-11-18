"""Subtract matched stem log-odds and compare exact-pair success."""

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

from sakha_byte.datasets import file_sha, require_remote
from sakha_possessor_calibration import PRIOR, PROMPTS, controls
from sakha_possessor_dialogue import SOURCE, get_pairs, variants
from sakha_possessor_dialogue_report import TITLES
from sakha_suffix_pairs_report import MODELS
from saas_lora_transfer import read_json, save


TIE_TOLERANCE = 1e-6


def margins(row):
    scores = row["rotation_scores"]
    assert len(scores) == 2 and all(len(r) == 2 and all(map(math.isfinite, r)) for r in scores)
    return [r[1] - r[0] for r in scores]


def action(value):
    if abs(value) <= TIE_TOLERANCE:
        return None
    return int(value > 0)


def evaluate(out, name, expected, indices, benchmark_sha):
    source = PRIOR / f"{name}.json"
    old = read_json(source)
    control = read_json(out / f"{name}-controls.json")
    assert old["benchmark_sha256"] == control["benchmark_sha256"] == benchmark_sha
    assert control["prior_scores_sha256"] == file_sha(source)
    assert old["manifest"] == control["manifest"] and old["model_path"] == control["model_path"]
    assert control["max_original_control_score_difference"] <= TIE_TOLERANCE
    assert all(len(r["prompt_sha256"]) == 2 for r in old["rows"])
    forms = {(r["condition"], r["pair_id"], r["form_index"]): r for r in old["rows"]
             if r["condition"] in PROMPTS}
    assert len(forms) == 2 * len(expected)
    rows = []
    for (condition, p, renders), probe in zip(expected, control["rows"], strict=True):
        pair_id = p["id"]
        assert (condition, pair_id) == (probe["condition"], probe["pair_id"])
        assert [hashlib.sha256(text.encode()).hexdigest() for text, _ in renders] == probe["prompt_sha256"]
        baseline = margins(probe)
        row_forms = []
        for form in range(2):
            archived = forms[condition, pair_id, form]
            original = variants(p, indices[pair_id], form, condition)
            assert [hashlib.sha256(text.encode()).hexdigest() for text, _ in original] == archived["prompt_sha256"]
            for (before, order), (bare, bare_order) in zip(original, renders, strict=True):
                signal = f"Бу {p['forms'][form]}."
                assert order == bare_order and before.count(signal) == 1
                assert before.replace(signal, f"Бу {p['stem']}.", 1) == bare
            raw = margins(archived)
            adjusted = [(v - b) for v, b in zip(raw, baseline, strict=True)]
            raw_average = sum(raw) / 2
            calibrated_average = sum(adjusted) / 2
            assert abs(raw_average - (archived["scores"][1] - archived["scores"][0])) < 1e-9
            row_forms.append({"gold": form, "form": p["forms"][form], "raw_prediction": archived["prediction"],
                              "calibrated_prediction": action(calibrated_average), "raw_log_odds": raw_average,
                              "calibrated_log_odds": calibrated_average, "calibrated_by_rotation": adjusted})
        assert math.isclose(row_forms[1]["raw_log_odds"] - row_forms[0]["raw_log_odds"],
                            row_forms[1]["calibrated_log_odds"] - row_forms[0]["calibrated_log_odds"], abs_tol=1e-9)
        rows.append({"pair_id": pair_id, "cohort": p["cohort"], "condition": condition, "stem": p["stem"],
                     "control_by_rotation": baseline, "forms": row_forms})
    save(out / f"{name}-calibrated.json", {"benchmark_sha256": benchmark_sha,
                                         "tie_tolerance": TIE_TOLERANCE, "rows": rows})
    return rows


def stats(rows):
    ids = {}
    for mode in ("raw", "calibrated"):
        key = f"{mode}_prediction"
        ids[mode] = sorted(row["pair_id"] for row in rows if all(f[key] == f["gold"] for f in row["forms"]))
    labels = Counter(f["calibrated_prediction"] for row in rows for f in row["forms"])
    return {"n_pairs": len(rows), "raw_correct_pair_ids": ids["raw"], "calibrated_correct_pair_ids": ids["calibrated"],
            "fixed_pair_ids": sorted(set(ids["calibrated"]) - set(ids["raw"])),
            "broken_pair_ids": sorted(set(ids["raw"]) - set(ids["calibrated"])),
            "raw_correct_items": sum(f["raw_prediction"] == f["gold"] for row in rows for f in row["forms"]),
            "calibrated_correct_items": sum(f["calibrated_prediction"] == f["gold"] for row in rows for f in row["forms"]),
            "calibrated_speaker": labels.get(0, 0), "calibrated_listener": labels.get(1, 0),
            "calibrated_ties": labels.get(None, 0)}


def main(out):
    pairs = get_pairs()
    benchmark_sha = file_sha(SOURCE / "benchmark.json")
    indices = {p["id"]: i for i, p in enumerate(pairs)}
    expected = list(controls(pairs))
    save(out / "control-prompts.json", [{"condition": c, "pair_id": p["id"], "stem": p["stem"],
                                       "rotations": [{"prompt": text, "role_order": order} for text, order in renders]}
                                      for c, p, renders in expected])
    all_models = {name: evaluate(out, name, expected, indices, benchmark_sha) for name in MODELS}
    groups = {"all": {p["id"] for p in pairs}, "unseen": {p["id"] for p in pairs if p["cohort"] != "lexical_seen"}}
    summaries, common = {}, {}
    for subset, ids in groups.items():
        summaries[subset], common[subset] = {}, {}
        for condition in PROMPTS:
            scores = {name: stats([r for r in rows if r["condition"] == condition and r["pair_id"] in ids])
                      for name, rows in all_models.items()}
            summaries[subset][condition] = scores
            common[subset][condition] = {f"{mode}_correct_pair_ids": sorted(set.intersection(*(
                set(r[f"{mode}_correct_pair_ids"]) for r in scores.values()))) for mode in ("raw", "calibrated")}
    joint = {
        subset: {
            c: sorted(set(groups_by_prompt["original"]["calibrated_correct_pair_ids"])
                      & set(groups_by_prompt[c]["calibrated_correct_pair_ids"]))
            for c in ("context", "dialogue_roles", "dialogue_names")
        }
        for subset, groups_by_prompt in common.items()
    }
    save(out / "comparison.json", {"benchmark_sha256": benchmark_sha, "conditions": TITLES, "metrics": summaries,
                                   "all_four": common, "all_four_original_and_dialogue": joint,
                                   "tie_tolerance": TIE_TOLERANCE})

    lines = ["# Минимальные пары на владельца — калибровка по основе без суффикса", "",
             "На IMMERS H100 для каждой из четырёх моделей заново оценили 185 контрольных форм "
             "(37 основ × 5 промптов) в двух порядках A/B. Оценки 74 форм с суффиксами на каждый промпт взяли "
             "из предыдущего прогона на той же H100. Сверили контрольные суммы входных файлов и хеши промптов. "
             "Свежие оценки исходных контролей совпали с архивом в пределах 0,000001.", "",
             "Для каждой основы, модели, промпта и порядка A/B вычислили логарифмический перевес «твой» над «мой» "
             "на форме с суффиксом и вычли такой же перевес на форме без суффикса. В диалогах имена и роли сохранялись. "
             "Две скорректированные оценки усреднили. Положительный результат даёт «твой», отрицательный — «мой». "
             "Разницу по модулю до 0,000001 считали ничьей и не засчитывали как верный ответ.", "",
             "## Верно распознаны обе формы, из 37 пар", "",
             "| Промпт | Qwen Base | Qwen + LoRA | Bwen Base | Bwen + дельта | Те же пары решены всеми четырьмя |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition, scores in summaries["all"].items():
        def counts(r):
            return f"{len(r['raw_correct_pair_ids'])} → {len(r['calibrated_correct_pair_ids'])}"
        cells = [counts(r) for r in scores.values()] + [counts(common["all"][condition])]
        lines.append(f"| {TITLES[condition]} | {' | '.join(cells)} |")
    lines += ["", "В каждой ячейке указано число пар до и после калибровки. «Только контекст» — исходные варианты «мой» "
              "и «твой» с добавленными именами и явными ролями в реплике. В колонке «всеми четырьмя» подсчитано "
              "пересечение по ID пар.", "",
              "## Срез из 29 пар, обе формы отсутствуют в корпусе LoRA", "",
              "| Промпт | Qwen Base | Qwen + LoRA | Bwen Base | Bwen + дельта | Все четыре |",
              "|---|---:|---:|---:|---:|---:|"]
    for condition, scores in summaries["unseen"].items():
        cells = [str(len(r["calibrated_correct_pair_ids"])) for r in scores.values()]
        cells.append(str(len(common["unseen"][condition]["calibrated_correct_pair_ids"])))
        lines.append(f"| {TITLES[condition]} | {' | '.join(cells)} |")
    lines += ["", "## Верные ответы и ничьи после калибровки", "",
              "В ячейке указаны число верных отдельных форм из 74 и число ничьих.", "",
              "| Промпт | Qwen Base | Qwen + LoRA | Bwen Base | Bwen + дельта |", "|---|---:|---:|---:|---:|"]
    for condition, scores in summaries["all"].items():
        cells = [f"{r['calibrated_correct_items']} / {r['calibrated_ties']}" for r in scores.values()]
        lines.append(f"| {TITLES[condition]} | {' | '.join(cells)} |")
    lines += ["", "## Какие пары решили все четыре модели", "",
              "| Промпт | ID и словоформы |", "|---|---|"]
    for condition, r in common["all"].items():
        values = []
        for pair_id in r["calibrated_correct_pair_ids"]:
            p = pairs[indices[pair_id]]
            values.append(f"`{pair_id}` ({' / '.join(p['forms'])})")
        lines.append(f"| {TITLES[condition]} | {', '.join(values) or '—'} |")
    both = ", ".join(f"«{TITLES[c]}» — {len(ids)}" for c, ids in joint["all"].items())
    lines.extend(["", "Число пар, которые все четыре модели решили и с исходным промптом, и с каждым "
                      f"вариантом диалога после калибровки — {both}."])
    lines += ["", "Вычитание меняет порог выбора и сохраняет разность оценок между двумя суффиксами внутри каждой пары. "
              "Калибровка проверяет, лежат ли две формы по разные стороны оценки основы. Изменение числа верных пар "
              "отражает выбор этого порога. Основа без суффикса служит контрольным стимулом без эталонного ответа. "
              "Задания короткие и шаблонные. Разметка остаётся пилотной.", "",
              "[Числа и ID по каждой модели](comparison.json) · [Контрольные промпты](control-prompts.json) · "
              "[Журнал инференса](logs/01-controls.log)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified {len(expected)} control items per model and calibrated both forms for each pair", flush=True)


if __name__ == "__main__":
    require_remote()
    main(Path(sys.argv[1]))
