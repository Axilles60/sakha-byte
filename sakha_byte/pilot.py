"""Original AI-drafted fixtures. Writes a data file ONLY when called on the server.

Sakha text and Russian controls are unreviewed; no claim of native-speaker validation.
The test measures reading Sakha with Russian questions and Russian choices.
"""

import json
from pathlib import Path


def create_pilot(path: Path):
    # category, sah, ru control, question, correct answer, distractors, explanation
    examples = [
        ("roles", "Маша Петяны көрдө.", "Маша увидела Петю.", "Кого увидела Маша?",
         "Петю", ["Машу", "Обоих", "Никого"], "Объект действия — Петя."),
        ("roles", "Петя Машаны көрдө.", "Петя увидел Машу.", "Кто увидел Машу?",
         "Петя", ["Маша", "Оба", "Никто"], "Действие совершает Петя."),
        ("negation", "Мин бардым.", "Я пошёл.", "Говорящий пошёл?",
         "Да", ["Нет", "Пойдёт завтра", "В тексте нет действия"], "Утверждается совершённое действие."),
        ("negation", "Мин барбатым.", "Я не пошёл.", "Говорящий пошёл?",
         "Нет", ["Да", "Только собирается идти", "В тексте нет действия"], "Отрицается совершённое действие."),
        ("number", "Үс оҕо кэллэ.", "Пришли три ребёнка.", "Сколько детей пришло?",
         "Три", ["Один", "Два", "Пять"], "Үс обозначает три."),
        ("number", "Биэс оҕо кэллэ.", "Пришли пять детей.", "Сколько детей пришло?",
         "Пять", ["Три", "Два", "Один"], "Биэс обозначает пять."),
        ("time", "Бэҕэһээ мин кэллим.", "Вчера я пришёл.", "Когда пришёл говорящий?",
         "Вчера", ["Сегодня", "Завтра", "Каждый день"], "Бэҕэһээ обозначает вчера."),
        ("time", "Бүгүн мин кэллим.", "Сегодня я пришёл.", "Когда пришёл говорящий?",
         "Сегодня", ["Вчера", "Завтра", "Каждый день"], "Бүгүн обозначает сегодня."),
        ("possession", "Бу мин кинигэм.", "Это моя книга.", "Кому принадлежит книга?",
         "Говорящему", ["Собеседнику", "Маше", "Пете"], "В тексте обозначено первое лицо."),
        ("possession", "Бу эн кинигэҥ.", "Это твоя книга.", "Кому принадлежит книга?",
         "Собеседнику", ["Говорящему", "Маше", "Пете"], "В тексте обозначено второе лицо."),
        ("lexical", "Бу улахан дьиэ.", "Это большой дом.", "Какой дом описан?",
         "Большой", ["Маленький", "Красный", "Старый"], "Улахан обозначает большой."),
        ("lexical", "Бу кыра дьиэ.", "Это маленький дом.", "Какой дом описан?",
         "Маленький", ["Большой", "Красный", "Старый"], "Кыра обозначает маленький."),
        ("location", "Кини дьиэҕэ баар.", "Он или она находится дома.", "Где находится человек?",
         "Дома", ["В школе", "На улице", "В магазине"], "Дьиэҕэ указывает на нахождение в доме."),
        ("location", "Кини оскуолаҕа баар.", "Он или она находится в школе.", "Где находится человек?",
         "В школе", ["Дома", "На улице", "В магазине"], "Оскуолаҕа указывает на нахождение в школе."),
        ("pronouns", "Биһиги кэллибит.", "Мы пришли.", "Кто пришёл?",
         "Мы", ["Я", "Вы", "Они"], "Биһиги обозначает мы."),
        ("pronouns", "Эһиги кэллигит.", "Вы пришли.", "Кто пришёл?",
         "Вы", ["Мы", "Я", "Они"], "Эһиги обозначает вы."),
    ]
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite reviewed data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for n, (category, sah, ru, question, answer, wrong, rationale) in enumerate(examples):
            index = n % 4
            choices = list(wrong)
            choices.insert(index, answer)
            item = {"id": f"pilot-{n+1:03d}", "pair_id": f"pair-{n//2+1:02d}",
                    "category": category, "context_sah": sah, "context_ru": ru,
                    "question_ru": question, "choices": choices, "answer_index": index,
                    "rationale_ru": rationale, "review_status": "draft", "reviewer": None,
                    "provenance": "AI-authored original draft; native-speaker review pending"}
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
