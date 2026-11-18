"""Pure-Python benchmark validation and prompt construction."""

import hashlib
import json
import math
import unicodedata
from pathlib import Path

LABELS = "ABCD"
PROMPT_VERSION = "ru-instruction-sah-context-v2-variable-choices"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(path: Path, allow_draft: bool = False) -> list[dict]:
    items, seen = [], set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {"id", "category", "context_sah", "question_ru",
                    "choices", "answer_index", "review_status", "rationale_ru"}
        if not required <= item.keys():
            raise ValueError(f"Line {line_no}: missing fields {required - item.keys()}")
        if not isinstance(item["id"], str) or item["id"] in seen:
            raise ValueError(f"Line {line_no}: invalid/duplicate id")
        seen.add(item["id"])
        if (not isinstance(item["choices"], list) or not 2 <= len(item["choices"]) <= 4
                or any(not isinstance(x, str) or not x.strip() for x in item["choices"])
                or len(set(item["choices"])) != len(item["choices"])):
            raise ValueError(f"{item['id']}: expected 2..4 distinct nonempty choices")
        if type(item["answer_index"]) is not int or item["answer_index"] not in range(len(item["choices"])):
            raise ValueError(f"{item['id']}: answer_index outside choices")
        if item["category"] in ("sentiment","gold_span_entity_type") and len(item["choices"]) != 3:
            raise ValueError(f"{item['id']}: this category requires exactly three canonical classes")
        for key in ("category", "context_sah", "question_ru", "rationale_ru"):
            value = item[key]
            if not isinstance(value, str) or not value.strip() or value != unicodedata.normalize("NFC", value):
                raise ValueError(f"{item['id']}: {key} must be nonempty NFC text")
        if item.get("context_ru") is not None:
            value = item["context_ru"]
            if not isinstance(value,str) or not value.strip() or value != unicodedata.normalize("NFC",value):
                raise ValueError(f"{item['id']}: invalid Russian control")
        if item["review_status"] not in ("draft", "approved"):
            raise ValueError(f"{item['id']}: invalid review status")
        if item["review_status"] == "approved" and not item.get("reviewer"):
            raise ValueError(f"{item['id']}: approved item needs reviewer")
        if item["review_status"] != "approved" and not allow_draft:
            raise ValueError("Draft items require --allow-draft; native-speaker review is pending")
        items.append(item)
    if not items:
        raise ValueError("Empty benchmark")
    return items


def make_prompt(item: dict, language: str, rotation: int = 0) -> tuple[str, list[int]]:
    n = len(item["choices"])
    if language not in ("sah", "ru") or rotation not in range(n):
        raise ValueError("Invalid language or rotation")
    if not item.get("context_"+language):
        raise ValueError(f"{item['id']}: no {language} context; controls must be supplied, not invented")
    order = [(i + rotation) % n for i in range(n)]
    choices = "\n".join(f"{LABELS[i]}. {item['choices'][j]}" for i, j in enumerate(order))
    prompt = ("Прочитай текст и выбери один ответ на вопрос. "
              f"Ответь только одной латинской буквой: {', '.join(LABELS[:n])}.\n\n"
              f"Текст: {item['context_' + language]}\n"
              f"Вопрос: {item['question_ru']}\n{choices}\nОтвет: ")
    return prompt, order


def combine_rotations(runs: list[tuple[list[float], list[int]]]) -> list[float]:
    """Mean log probability for each original choice across cyclic rotations."""
    if not runs:
        raise ValueError("No scores")
    n = len(runs[0][0])
    if not 2 <= n <= 4:
        raise ValueError("Expected 2..4 scores")
    scores = [0.0] * n
    for log_probs, order in runs:
        if len(log_probs) != n or sorted(order) != list(range(n)):
            raise ValueError("Malformed scores/order")
        for label, original in enumerate(order):
            if not math.isfinite(log_probs[label]):
                raise ValueError("Non-finite label score")
            scores[original] += log_probs[label] / len(runs)
    return scores


def wilson(correct: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No completed predictions")
    groups = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    correct = sum(r["correct"] for r in rows)
    by_category = {k:{"n":len(v), "accuracy":sum(r["correct"] for r in v)/len(v)}
                   for k,v in sorted(groups.items())}
    for category in ("sentiment", "gold_span_entity_type"):
        if category not in groups:
            continue
        confusion = [[0]*3 for _ in range(3)]
        for row in groups[category]:
            confusion[row["answer_index"]][row["prediction"]] += 1
        f1s = []
        for label in range(3):
            tp = confusion[label][label]
            denominator = sum(confusion[label]) + sum(r[label] for r in confusion)
            f1s.append(2*tp/denominator if denominator else 0.0)
        by_category[category].update(macro_f1=sum(f1s)/3, confusion_matrix=confusion,
                                    class_order=["positive","negative","neutral"] if category=="sentiment" else ["PER","LOC","ORG"])
    return {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "accuracy_wilson95": wilson(correct, len(rows)),
            "chance_accuracy": sum(1/r.get("n_choices",len(r.get("scores",[0]*4))) for r in rows)/len(rows),
            "by_category":by_category}
