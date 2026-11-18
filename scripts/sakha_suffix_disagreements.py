"""Explain score-shift disagreements on the 77 new Sakha pairs on IMMERS."""

import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

from sakha_byte.datasets import file_sha, require_remote
from sakha_possessor_calibration_report import TIE_TOLERANCE, margins
from sakha_suffix_calibrated_report import FEATURE_LABELS
from sakha_suffix_direction_report import INPUT, SOURCE
from sakha_suffix_pairs_report import MODELS
from saas_lora_transfer import read_json, save


DIRECTION = Path("results/sakha-suffix-direction-101")
Q, QA, B, BA = MODELS
TUNED_GAP = "Адаптированная Qwen − адаптированная Bwen"
SIGNS = {-1: "−", 0: "0", 1: "+"}


def sign(value):
    return int(value > TIE_TOLERANCE) - int(value < -TIE_TOLERANCE)


def fmt(value, digits=4):
    if sign(value) == 0:
        return "0"
    return f"{value:+.{digits}f}".replace("-", "−").replace(".", ",")


def load_rows():
    benchmark = SOURCE / "benchmark.json"
    sha = file_sha(benchmark)
    pairs = read_json(benchmark)["pairs"]
    prior = read_json(DIRECTION / "comparison.json")
    assert prior["benchmark_sha256"] == sha and prior["tie_tolerance"] == TIE_TOLERANCE
    effects = read_json(DIRECTION / "pair-effects.json")
    data = {}
    for name in MODELS:
        bundle = read_json(INPUT / f"{name}.json")
        assert bundle["benchmark_sha256"] == sha and bundle["historical_predictions_matched"] == 202
        data[name] = bundle["rows"]
    rows = []
    for idx, (p, effect) in enumerate(zip(pairs, effects, strict=True)):
        assert p["id"] == effect["id"] and p["forms"] == effect["forms"]
        if p["cohort"] == "lexical_seen":
            continue
        assert not any(p["form_docs"])
        values = {}
        for name, scores in data.items():
            row = scores[idx]
            assert row["pair_id"] == p["id"]
            a, b = [margins(row["stimuli"][f"form{i}"]) for i in range(2)]
            by_rotation = [y-x for x, y in zip(a, b, strict=True)]
            effect_value = mean(by_rotation)
            assert math.isclose(effect_value, effect["signed_shifts"][name], abs_tol=1e-9)
            values[name] = {"shift": effect_value, "sign": sign(effect_value), "shift_by_rotation": by_rotation,
                            "form_odds": [mean(a), mean(b)], "form_odds_by_rotation": [a, b]}
        signs = (values[QA]["sign"], values[BA]["sign"])
        category = "same" if signs[0] == signs[1] else "opposite" if signs[0]*signs[1] == -1 else "with_zero"
        rows.append({**{k: p[k] for k in ("id", "feature", "cohort", "forms", "suffixes")},
                     "category": category, "models": values})
    assert len(rows) == 77
    return rows, prior["subsets"]["unseen"], sha


def aggregate(rows):
    angles = Counter(tuple(r["models"][m]["sign"] for m in (QA, BA)) for r in rows)
    models = {}
    for name in MODELS:
        values = [r["models"][name] for r in rows]
        models[name] = {"positive": sum(v["sign"] == 1 for v in values),
                        "both_rotations_positive": sum(all(sign(x) == 1 for x in v["shift_by_rotation"]) for v in values),
                        "median_shift": median(v["shift"] for v in values)}
    calibration = {}
    for base, tuned in ((Q, QA), (B, BA)):
        changes = [[r["models"][tuned]["form_odds"][i] - r["models"][base]["form_odds"][i]
                    for i in range(2)] for r in rows]
        first, second = (mean(col) for col in zip(*changes, strict=True))
        calibration[tuned] = {"mean_change_first_form": first, "mean_change_second_form": second,
                              "mean_change_shift": second-first,
                              "resolved": sum(r["models"][tuned]["sign"] == 1 and r["models"][base]["sign"] != 1 for r in rows),
                              "lost": sum(r["models"][tuned]["sign"] != 1 and r["models"][base]["sign"] == 1 for r in rows)}
    return {"n": len(rows), "sign_counts": {"/".join(SIGNS[x] for x in k): v for k, v in angles.items()},
            "models": models, "adaptation": calibration}


def table_rows(rows, category):
    lines = ["| ID | Тип | Формы | Qwen до → после | Bwen до → после |", "|---|---|---|---:|---:|"]
    for r in (item for item in rows if item["category"] == category):
        cells = [f"{fmt(r['models'][base]['shift'])} → {fmt(r['models'][tuned]['shift'])}"
                 for base, tuned in ((Q, QA), (B, BA))]
        lines.append(f"| `{r['id']}` | {FEATURE_LABELS[r['feature']]} | {' / '.join(r['forms'])} | {' | '.join(cells)} |")
    return lines


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    rows, prior, sha = load_rows()
    groups = {"all": rows, **{f: [r for r in rows if r["feature"] == f] for f in FEATURE_LABELS}}
    summary = {f: aggregate(group) for f, group in groups.items()}
    suffix_groups = {suffix: aggregate([r for r in groups["number"] if r["suffixes"][1] == suffix])
                     for suffix in sorted({r["suffixes"][1] for r in groups["number"]})}
    for feature, values in summary.items():
        reference = prior["total" if feature == "all" else feature]
        assert all(r["positive"] == reference["models"][m]["correct"] for m, r in values["models"].items())
        values["tuned_qwen_minus_bwen"] = reference["contrasts"][TUNED_GAP]
    save(out / "comparison.json", {"benchmark_sha256": sha, "tie_tolerance": TIE_TOLERANCE,
                                   "by_type": summary, "number_allomorphs": suffix_groups})
    save(out / "pair-details.json", rows)

    lines = ["# Новые якутские словоформы — где расходятся Qwen и Bwen после адаптации", "",
             "Сверены 77 пар, обе формы которых отсутствовали во входном корпусе LoRA. Анализ выполнен "
             "на IMMERS по сохранённым оценкам обоих порядков A/B. Положительный сдвиг означает, что вторая "
             "форма повысила перевес второго ответа над первым относительно первой формы. Нулём считается "
             "сдвиг по модулю до 0,000001. Это сравнение относительных оценок, даже когда итоговый ответ "
             "в обеих формах одинаков. Все оценки указаны в единицах натурального логарифма.", "",
             "## Где модели расходятся", "",
             "| Тип | Обе в верную сторону | Qwen +, Bwen − | Qwen −, Bwen + | Qwen 0, Bwen + | Обе в обратную сторону |",
             "|---|---:|---:|---:|---:|---:|"]
    keys = ("+/+", "+/−", "−/+", "0/+", "−/−")
    for f, r in summary.items():
        assert sum(r["sign_counts"].get(k, 0) for k in keys) == r["n"]
        name = FEATURE_LABELS.get(f, "Всего")
        cells = " | ".join(str(r["sign_counts"].get(k, 0)) for k in keys)
        lines.append(f"| {name} ({r['n']}) | {cells} |")
    lines += ["", "### Противоположные знаки", "", *table_rows(rows, "opposite"), "",
              "### У одной модели нулевой сдвиг", "", *table_rows(rows, "with_zero"), "",
              "## Владелец", "",
              "Bwen после переноса дала верный знак на 26 из 29 пар, а Qwen после LoRA — на 24 из 29. "
              "Разница в две пары существовала и до адаптации (19 против 17). Обе модели прибавили по семь "
              "в чистом счёте. Qwen исправила семь знаков. Bwen исправила восемь и потеряла один. "
              "В обоих порядках A/B сдвиг был положительным на 24 парах у адаптированной Bwen и на 17 у "
              "адаптированной Qwen. Парный 95%-й ДИ перевеса Bwen на 29 парах — от 0 до 17,2 п. п.", "",
              "- `sakha-pair-027`, **көхөм / көхөҥ**. После переноса знак Bwen поменялся с −0,1075 на +0,0462. "
              "У Qwen он остался отрицательным (−0,0313 после LoRA). В Bwen на форме «твой» перевес второго ответа "
              "был −0,541 против −0,587 на форме «мой». Второй ответ стал относительно вероятнее, хотя обе оценки "
              "по-прежнему отдавали предпочтение «мой». В Qwen перестановки дали сдвиги −0,1250 и +0,0625.", "",
              "- `sakha-pair-099`, **мөлөгөмөчөм / мөлөгөмөчөҥ**. У Bwen сдвиг был положительным ещё до переноса "
              "и сохранился после него в обоих порядках (+0,0954 и +0,2160). У обеих Qwen средний знак отрицательный. "
              "Вместе эти две пары дают весь перевес Bwen в этом срезе.", "",
              "## Число", "",
              "Базовые модели начинали с одинакового счёта 20 из 24. LoRA исправила у Qwen три знака без потерь. "
              "У Bwen перенос исправил один и испортил три. После адаптации счёт 23 против 18. Из шести "
              "строго противоположных знаков все шесть в пользу Qwen, а на одной дополнительной паре Bwen "
              "показала положительный сдвиг при нуле у Qwen. Парный 95%-й ДИ перевеса Qwen — от 0 до 41,7 п. п.", "",
              "| Суффикс множественного | Qwen до → после | Bwen до → после |",
              "|---|---:|---:|"]
    for suffix, r in suffix_groups.items():
        m, n = r["models"], r["n"]
        lines.append(f"| -{suffix} ({n} пар) | {m[Q]['positive']} → {m[QA]['positive']} | {m[B]['positive']} → {m[BA]['positive']} |")
    lines += ["", "Четыре из шести строгих расхождений приходятся на **-лэр**. В среднем LoRA снизила у Qwen "
              "перевес множественного на единственном числе на 0,0625, а на множественном — лишь на 0,0169. "
              "Тем самым она увеличила разделение между формами. Перенос у Bwen снизил обе оценки примерно "
              "одинаково, на 0,0364 и 0,0403 соответственно.", "",
              "- `sakha-pair-065`, **нородолоро / нородолоролор**. Сдвиг Qwen изменился с −0,0313 на +0,1875 "
              "и стал положительным в обоих порядках. У Bwen средний знак сменился с +0,0155 на −0,0154.", "",
              "- `sakha-pair-068`, **кэрэдэтэхэ / кэрэдэтэхэлэр**. У адаптированной Qwen добавление -лэр "
              "повышает относительную оценку множественного в обоих порядках (+0,2500 и +0,3750). "
              "У адаптированной Bwen порядки дают разные знаки (−0,1246 и +0,0933).", "",
              "- `sakha-pair-069`, **кэрэдэмэчэ / кэрэдэмэчэлэр**. Qwen дала +0,3750 в обоих порядках. "
              "У Bwen два сдвига почти погасили друг друга, и среднее −0,0003 даёт отрицательный знак "
              "по принятому критерию. На паре `sakha-pair-071` сдвиг Bwen около +0,0002 тоже близок к нулю.", "",
              "## Как читать эти различия", "",
              "На направлении нашлись два строгих расхождения в пользу Qwen. Ещё на одной паре Bwen даёт "
              "положительный сдвиг при нуле у Qwen. Сводная модельная разница на направлении составляет "
              "одну пару. Интервалы по каждому типу включают ноль. Исследование показывает, какие словоформы, "
              "изменения оценок и эффекты порядка дали разные итоги. Для вывода о причинной роли токенизации "
              "нужен отдельный контроль. Языковая разметка остаётся пилотной, а 72 из 77 пар используют "
              "искусственные основы; все 24 пары на число относятся к этому набору.", "",
              "[Полные оценки обеих форм и перестановок](pair-details.json) · [Агрегаты](comparison.json) · "
              "[Журнал анализа](logs/01-analysis.log)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} new pairs; {sum(r['category']=='opposite' for r in rows)} strict opposites and "
          f"{sum(r['category']=='with_zero' for r in rows)} with one zero.", flush=True)


if __name__ == "__main__":
    require_remote()
    main(Path(sys.argv[1]))
