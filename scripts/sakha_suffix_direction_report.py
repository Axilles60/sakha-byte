"""Compare threshold-free suffix effects from the archived IMMERS logits."""

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from sakha_byte.datasets import file_sha, require_remote
from sakha_possessor_calibration_report import TIE_TOLERANCE, margins
from sakha_suffix_calibrated import SOURCE
from sakha_suffix_calibrated_report import FEATURE_LABELS
from sakha_suffix_pairs import norm
from sakha_suffix_pairs_report import MODELS
from saas_lora_transfer import read_json, save


INPUT = Path("results/sakha-suffix-calibrated-101")
BOOTSTRAP_DRAWS = 20000
SEED = 42
CONTRASTS = {
    "Базовая Qwen − базовая Bwen": ("qwen-base", "bwen-base"),
    "Адаптированная Qwen − адаптированная Bwen": ("qwen-lora-200", "bwen-lora-transfer"),
    "Qwen после LoRA − до": ("qwen-lora-200", "qwen-base"),
    "Bwen после переноса − до": ("bwen-lora-transfer", "bwen-base"),
}


def load_effects(pairs, benchmark_sha):
    calibrated = read_json(INPUT / "predictions.json")
    summary = read_json(INPUT / "comparison.json")
    assert summary["benchmark_sha256"] == benchmark_sha
    effects = {}
    for name in MODELS:
        original = read_json(INPUT / f"{name}.json")
        assert original["benchmark_sha256"] == benchmark_sha
        assert original["historical_predictions_matched"] == 2 * len(pairs)
        values = []
        for idx, (p, row) in enumerate(zip(pairs, original["rows"], strict=True)):
            assert row["pair_id"] == p["id"]
            a, b = (margins(row["stimuli"][f"form{i}"]) for i in range(2))
            value = sum(y - x for x, y in zip(a, b, strict=True)) / 2
            for mode in calibrated.values():
                check = mode[name][idx]
                assert check["pair_id"] == p["id"]
                observed = check["scores"][1] - check["scores"][0]
                assert math.isclose(value, observed, rel_tol=0, abs_tol=1e-9)
            values.append(value)
        effects[name] = np.asarray(values)
        expected = summary["metrics"]["raw"]["all"][name]["separation_correct"]
        assert int((effects[name] > TIE_TOLERANCE).sum()) == expected
    return effects


def confidence(draws):
    if np.all(draws == draws[0]):
        return None
    return np.percentile(draws, [2.5, 97.5]).tolist()


def summarize(pairs, effects, selected):
    clusters = defaultdict(list)
    for i in selected:
        clusters[norm(pairs[i]["stem"])].append(i)
    correct = np.stack([values > TIE_TOLERANCE for values in effects.values()])
    sizes = np.asarray([len(ids) for ids in clusters.values()])
    totals = np.stack([correct[:, ids].sum(axis=1) for ids in clusters.values()])
    rng = np.random.default_rng(SEED)
    sample = rng.integers(len(sizes), size=(BOOTSTRAP_DRAWS, len(sizes)))
    replicates = totals[sample].sum(axis=1) / sizes[sample].sum(axis=1, keepdims=True)
    models = {}
    for idx, (name, values) in enumerate(effects.items()):
        sub = values[selected]
        n_correct = int(correct[idx, selected].sum())
        n_ties = int((np.abs(sub) <= TIE_TOLERANCE).sum())
        models[name] = {"correct": n_correct, "opposite": len(sub) - n_correct - n_ties, "ties": n_ties,
                        "rate": n_correct / len(sub), "ci95": confidence(replicates[:, idx])}
    contrasts = {}
    names = list(effects)
    for label, (left, right) in CONTRASTS.items():
        a, b = names.index(left), names.index(right)
        differences = correct[a, selected].astype(int) - correct[b, selected].astype(int)
        contrasts[label] = {"gap": float(differences.mean()),
                            "ci95": confidence(replicates[:, a] - replicates[:, b]),
                            "left_only": int((differences == 1).sum()),
                            "right_only": int((differences == -1).sum())}
    return {"n_pairs": len(selected), "n_stems": len(clusters), "models": models, "contrasts": contrasts}


def percent(value, signed=False):
    return format(value * 100, "+.1f" if signed else ".1f").replace(".", ",")


def interval(bounds, signed=False):
    return "—" if bounds is None else " до ".join(percent(v, signed) for v in bounds)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    benchmark = SOURCE / "benchmark.json"
    benchmark_sha = file_sha(benchmark)
    pairs = read_json(benchmark)["pairs"]
    assert len(pairs) == len({p["id"] for p in pairs}) == 101
    effects = load_effects(pairs, benchmark_sha)
    all_pairs = list(range(len(pairs)))
    new = [i for i in all_pairs if pairs[i]["cohort"] != "lexical_seen"]
    assert len(new) == 77 and all(not any(pairs[i]["form_docs"]) for i in new)
    subsets = {group: {
        "total": summarize(pairs, effects, selected),
        **{f: summarize(pairs, effects, [i for i in selected if pairs[i]["feature"] == f]) for f in FEATURE_LABELS}
    } for group, selected in (("all", all_pairs), ("unseen", new))}
    save(out / "comparison.json", {"benchmark_sha256": benchmark_sha, "tie_tolerance": TIE_TOLERANCE,
                                   "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": SEED,
                                   "source_checksum_sha256": file_sha(INPUT / "SHA256SUMS.txt"),
                                   "subsets": subsets})
    save(out / "pair-effects.json", [{**{k: p[k] for k in ("id", "feature", "cohort", "forms")},
                                     "signed_shifts": {m: float(v[i]) for m, v in effects.items()}}
                                    for i, p in enumerate(pairs)])

    lines = ["# Якутские минимальные пары — сдвиг оценки без порога выбора", "",
             "Взяли сохранённые оценки четырёх моделей на IMMERS H100. Анализ выполнен на IMMERS без нового "
             "инференса. Для каждой формы посчитали логарифмическую оценку второго варианта минус оценку первого. "
             "Затем из перевеса на второй форме вычли перевес на первой и усреднили по двум порядкам A/B. "
             "Положительный сдвиг означает, что при смене суффикса модель сдвинула относительную оценку "
             "к правильному для второй формы ответу. Например, при переходе с формы «мой» на форму «твой» "
             "оценивается рост перевеса «твой». Правильность итогового выбора в метрику не входит.", "",
             "Значения по модулю до 0,000001 считаются нулевым сдвигом и остаются в знаменателе. Проверено, "
             "что на всех парах разность оценок совпадает до и после каждой ранее проверенной калибровки. "
             "Парные 95%-е бутстрэп-интервалы получены на 20 000 выборках по нормализованным основам. "
             "Одни и те же пары и повторяющиеся основы перевыбирались совместно у четырёх моделей. "
             "Интервалы описывают вариативность внутри этого пилота.", ""]
    for subset, label in (("all", "Все 101 пары"), ("unseen", "77 новых пар")):
        lines += [f"## {label}", "",
                  "| Срез | " + " | ".join(MODELS.values()) + " |", "|---|---:|---:|---:|---:|"]
        for group, r in subsets[subset].items():
            n = r["n_pairs"]
            cells = [f"{m['correct']}/{n} ({percent(m['rate'])}%, ДИ {interval(m['ci95'])}%)"
                     for m in r["models"].values()]
            lines.append(f"| {FEATURE_LABELS.get(group, 'Все')} | {' | '.join(cells)} |")
        lines.append("")
    lines += ["Новые пары определены по полному входному корпусу LoRA. В нём отсутствовали обе словоформы.", "",
              "## Парные различия между моделями", "",
              "| Сравнение | Все 101, п. п. [95% ДИ] | 77 новых, п. п. [95% ДИ] |",
              "|---|---:|---:|"]
    for contrast in CONTRASTS:
        cells = []
        for subset in subsets.values():
            r = subset["total"]["contrasts"][contrast]
            cells.append(f"{percent(r['gap'], True)} [{interval(r['ci95'], True)}]")
        lines.append(f"| {contrast} | {' | '.join(cells)} |")
    total_ties = ", ".join(f"{MODELS[m]} — {d['ties']}"
                           for m, d in subsets["all"]["total"]["models"].items())
    lines += ["", f"Нулевые сдвиги среди 101 пары — {total_ties}.", "",
              "Тест состоит из коротких шаблонов, большинство новых пар построено на искусственных основах. "
              "Разметка остаётся пилотной. Метрика измеряет порядок относительных оценок и не устанавливает "
              "причину различий между чекпойнтами.", "",
              "[Числа и интервалы по всем срезам](comparison.json) · [Сдвиги по каждой паре](pair-effects.json)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified {len(pairs)} score shifts for each model and all prior calibration modes; bootstrapped {BOOTSTRAP_DRAWS} samples per slice.", flush=True)


if __name__ == "__main__":
    require_remote()
    main(Path(sys.argv[1]))
