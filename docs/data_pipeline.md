# Серверный маршрут данных

Выполнять только на IMMERS. На устройстве пользователя созданы только исходники,
конфигурации и эта инструкция. Ни команды, ни тесты нового кода здесь не выполнялись.

## Поддержанные источники и форматы

`fetch-source` использует `configs/dataset_downloads.json` и поддерживает:

- HF: `sakha_mono`, `sakha_parallel`, `wmt_yakut`, `averoo_parallel`, `sakha_judge`;
- GitHub: `ud_yakut`, `sakha_ner_saas`.

Для каждого источника ветка разрешается в полный SHA один раз. Записывается
`data/raw/<source>/manifest.json`, затем скачиваются только выбранные файлы в
подкаталог `files/`. Контролируются размер, место и SHA-256. Повторный запуск
использует тот же SHA, продолжает незавершённую загрузку и проверяет файлы.
Изменение источника/конфигурации требует нового raw-root; молчаливого обновления нет.
Загрузчик не запускает сторонние dataset scripts, pickle, notebooks или код репозитория.

У WMT выбран экспорт `refs/convert/parquet`, затем фиксируется его SHA. Если ветка
недоступна, команда завершится ошибкой: нужно явно выбрать доступную версию, а не
переключаться на выполнение старого dataset script.

Читатели поддерживают Parquet, JSONL, массивы JSON, CSV/TSV и XLSX. Схемы gated-корпусов
нельзя достоверно установить без доступа; профили колонок являются явными начальными
настройками. Неизвестная колонка/метка вызывает ошибку, а не угадывание.

## 1. Установка и загрузка на ВМ

Окружения моделей создаются по README. В SSH-сессии:

```bash
export SAKHA_REMOTE_EXECUTION=immers
uv pip install --python .venv-bwen/bin/python -r requirements/data.txt
.venv-bwen/bin/python -m unittest discover -s tests -v

# HF_TOKEN задаётся на сервере через хранилище секретов/окружение.
# Сначала самостоятельно принять условия доступа двух gated-корпусов на HF.
.venv-bwen/bin/sakha-byte fetch-source --source sakha_mono --gated-access-granted
.venv-bwen/bin/sakha-byte fetch-source --source sakha_parallel --gated-access-granted
.venv-bwen/bin/sakha-byte fetch-source --source ud_yakut
.venv-bwen/bin/sakha-byte fetch-source --source sakha_ner_saas --license-reviewed
.venv-bwen/bin/sakha-byte fetch-source --source sakha_judge
```

`--gated-access-granted` фиксирует, что доступ уже оформлен; сам код условия не принимает.
`--license-reviewed` требуется для неизвестной или некоммерческой лицензии: это
подтверждение проверки условий для предполагаемого использования, не изменение лицензии.
WMT/averoo с неустановленной лицензией тоже требуют этот флаг. Нет автоматической
загрузки всех источников сразу, аренды ВМ или расширения диска.

```bash
.venv-bwen/bin/sakha-byte inspect-source --source sakha_mono
.venv-bwen/bin/sakha-byte inspect-source --source sakha_parallel
.venv-bwen/bin/sakha-byte inspect-source --source sakha_ner_saas
```

Инспекция выводит имена файлов, столбцы/типы, строки Parquet и листы XLSX без печати
образцов текста. По результату скорректировать `configs/corpus_mappings.json` и
`configs/eval_mappings.json` **на сервере**. Поля задаются dotted path, например
`translation.sah`, без eval/выражений Python. В SAAS явно выбрать один набор файлов,
чтобы не включить CSV и XLSX копии одного корпуса одновременно.

## 2. Подготовить тесты и запретить их попадание в обучение

```bash
.venv-bwen/bin/sakha-byte prepare-eval --profile ud_lemma --output data/eval/ud
.venv-bwen/bin/sakha-byte prepare-eval --profile ner_type --output data/eval/ner
```

Каждый каталог содержит:

- `benchmark.jsonl`: задания для единых Qwen/Bwen/EvaByte оценщиков;
- `exclusions.jsonl`: исходные контексты для исключения из обучения;
- `manifest.json`: источник, SHA, профиль, хеши и число примеров.

UD: CoNLL-U превращается в вопрос о лемме выделенного слова, с тремя детерминированно
выбранными отвлекающими леммами той же части речи. Это диагностический MCQ, не
стандартная точность свободной лемматизации. Контексты всех UD-предложений исключаются
из обучения, даже при `--limit`.

NER: Label Studio JSON читается по исходным символьным offset, до нормализации текста.
Берётся последняя неотменённая разметка; используется опубликованный test split.
Ключи документов вычисляются тем же алгоритмом, что в upstream notebook (data-поля,
иначе SHA-1 исходного UTF-8 текста). Ошибки split и offset не игнорируются.
Задание — выбрать PER/LOC/ORG для **уже выделенной** сущности. Отрицательные предложения
не дают таких заданий. Это gold-span type classification, а не извлечение NER:
нельзя сравнивать полученный балл с опубликованным entity extraction F1.
Все исходные NER-контексты остаются в exclusions.

SAAS: три настоящих варианта — Positive, Negative, Neutral. В mapping явно
сопоставить фактические метки индексам 0, 1, 2; Skip исключается. Неизвестные метки
вызывают ошибку. Есть два способа определить held-out выборку:

1. Указать в профиле `split_field` и `split_value` существующего разбиения.
2. Создать на сервере JSON-массив строковых ID и передать `--ids-file`.
   Если `id_field` не задан, ID имеет вид `<путь-файла>:<номер-записи-с-нуля>`.
   Это не номер строки CSV с заголовком. Для нескольких файлов ID содержит относительный путь.

```bash
# Только после проверки mapping и выбора независимых ID на сервере:
.venv-bwen/bin/sakha-byte prepare-eval --profile saas --ids-file data/splits/saas-test-ids.json --output data/eval/saas
```

SAAS сам не получает случайный split. Отбор ID и выбор доменов нужно определить до
измерений. `exclusions.jsonl` содержит все контексты выбранных SAAS-файлов.

Все новые задания помечаются `draft`: перенос готовой разметки не подтверждает
качество новых формулировок. Для технического прогона допустим `--allow-draft`.
Для итогового теста носитель проверяет вопросы/ключи и задаёт approved/reviewer.
Русские переводы контекстов не выдумываются: `context_ru=null`, поэтому `--language ru`
до добавления проверенных переводов завершится ошибкой ещё до загрузки модели.

## 3. Подготовить корпус Qwen

В mono mapping обязательно установить настоящий `document_field`, если он есть.
Разбиение выполняется по идентичности документа/источника до нарезки на отрывки.
Для проверенной схемы `sakha_mono` с колонками `source` и `text` используйте
`mono_by_source`. Одна метка `source` целиком попадает в один набор. Доли задают
пороги для распределения групп, а фактические доли строк зависят от размеров
источников. Для других корпусов без этих меток `--allow-content-groups` позволяет
явно выбрать группировку по содержимому отдельных записей.

`--tokenizer-path` — путь к уже загруженному на сервере снимку токенизатора
Qwen3-8B-Base из `artifacts/models/qwen.json`. Не указывать токенизатор Bwen.

```bash
qwen_snapshot=$(.venv-bwen/bin/python -c 'import json; from pathlib import Path; print(json.loads(Path("artifacts/models/qwen.json").read_text())["path"])')
.venv-bwen/bin/python -m sakha_byte.cli prepare-corpus \
  --profiles mono_by_source \
  --tokenizer-path "$qwen_snapshot" \
  --exclude data/eval/pilot.jsonl data/eval/ud/exclusions.jsonl \
  --seed 112 --validation-fraction 0.1 --test-fraction 0.1 \
  --output data/corpus/qwen-v1
```

Добавьте NER, SAAS и любой тест перевода/понимания к `--exclude`, если они участвуют в
эксперименте. Список exclusions обязателен. Из исходных записей удаляются точные
нормализованные дубли; полное совпадение текста/перевода контекста с фрагментом
обучающей записи приводит к исключению этой записи. Обе стороны параллельной пары
проверяются отдельно. Повторяющиеся компоненты между разными split отклоняются;
идентичные итоговые отрывки не включаются повторно.

Нормализация: Unicode NFC; для сравнения дублей — регистр и пробелы. Якутские буквы
не заменяются русскими. Строки с U+FFFD и слишком короткие записи отбрасываются.
Это минимальные проверки, а не полноценный LID, исправление OCR, удаление близких
дублей или поиск переводных/семантических утечек. Дополнительный аудит остаётся нужен.

Выход — `train.jsonl`, `validation.jsonl`, `test.jsonl` и `manifest.json` с числом строк
и групп по сплитам, хешами входов и выходов и снимком конфигурации. Test здесь — отложенный
языковой корпус, не автоматический тест понимания. Пустой split считается ошибкой.
Статус failed запрещает использование результата как готового корпуса.

Длинные документы режутся по границам слов под реальный токенизатор, без скрытой
обрезки и без декодирования отдельных байтов. По умолчанию максимум 2047 токенов
плюс EOS в тренере. Все отрывки одного документа остаются в одном split.
Для полной конвертации на многопроцессорной ВМ задайте `--workers 10`.
Параллельные токенизаторы сохраняют исходный порядок записей и проверяют
фактическую длину каждого отрывка.

Режимы:

- `mono`, `mono_by_source`: обычный якутский текст;
- `parallel`, `wmt`, `averoo`: aligned LM-текст `Русский: … / Якутский: …`;
- `judge_alpaca`: инструктивный текст с ответом; синтетическое происхождение
  сохраняется, человеческие оценки из Sakha-Judge в обучение не включаются.

Последний режим является causal language modeling всего текста, включая инструкцию,
а не отдельным SFT с loss только на ответе. По умолчанию начните с mono; влияние
parallel/instruction компонентов проверяйте отдельными конфигурациями обучения.

```bash
timeout -k 10s 10m .venv-bwen/bin/python -m sakha_byte.train_qwen \
  --train-file data/corpus/qwen-v1/train.jsonl \
  --validation-file data/corpus/qwen-v1/validation.jsonl \
  --corpus-manifest data/corpus/qwen-v1/manifest.json \
  --benchmark data/eval/pilot.jsonl \
  --smoke-test --output artifacts/qwen-sah-smoke

timeout -k 10s 5m .venv-bwen/bin/python -m sakha_byte.cli evaluate \
  --model qwen --stage adapted --checkpoint artifacts/qwen-sah-smoke/merged \
  --benchmark data/eval/pilot.jsonl --allow-draft --limit 1 \
  --output results/qwen-adapted-smoke
```

Перед тестом проверьте `status=complete` в манифесте корпуса. При повторе после
прерванной попытки выберите новый каталог вывода. Тест делает один шаг оптимизатора
на выборке до четырёх строк train и проверяет до восьми строк validation. Он сохраняет
адаптер и полный объединённый чекпойнт. Проверьте `status=complete` в манифестах
обучения и оценки перед полным пилотом.

Для 200 шагов уберите `--smoke-test` и задайте новый каталог `artifacts/qwen-sah-v1`.
Лимит времени выберите по фактическому времени подготовки, шагов и сохранения на ВМ.
Полный рецепт детерминированно выбирает до 256 строк validation до токенизации.
Для проверки всего сплита задайте `max_validation_rows: null` в конфигурации LoRA.
Тренер проверяет хеши train и validation против `--corpus-manifest`, а также попадание
контекста указанного `--benchmark` с учётом Unicode, регистра и пробелов.

## 4. Оценка и ансамбль

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

.venv-bwen/bin/sakha-byte evaluate --model bwen --benchmark data/eval/ud/benchmark.jsonl --allow-draft --output results/bwen-ud
.venv-eva/bin/sakha-byte evaluate --model evabyte --benchmark data/eval/ud/benchmark.jsonl --allow-draft --output results/eva-ud
.venv-bwen/bin/sakha-byte ensemble --bwen-run results/bwen-ud --evabyte-run results/eva-ud --output results/ensemble-ud
.venv-bwen/bin/sakha-byte evaluate --model qwen --benchmark data/eval/ud/benchmark.jsonl --allow-draft --output results/qwen-base-ud
.venv-bwen/bin/sakha-byte evaluate --model qwen --stage adapted --checkpoint artifacts/qwen-sah-v1/merged --benchmark data/eval/ud/benchmark.jsonl --allow-draft --output results/qwen-adapted-ud
.venv-bwen/bin/sakha-byte report results/qwen-base-ud results/qwen-adapted-ud results/bwen-ud results/eva-ud results/ensemble-ud
```

Повторить с каталогами SAAS и NER, сохраняя задачи раздельно. Новый протокол v2
поддерживает 2–4 ответа, все циклические перестановки (`--rotations 0`, по умолчанию),
размер-зависимый chance и macro-F1/confusion matrix для трёх фиксированных классов.
Ансамбль объединяет одинаковые вопросы с одинаковыми порядками и хешами промптов.
Для сравнения используйте одну версию протокола и один файл теста внутри каждой задачи.

## Первичная проверка на сервере

Сначала unit-тесты, инспекция схем и небольшие конвертации. Затем по одному
заданию на каждой модели, полный пилот, короткий тренировочный шаг и только после
успеха полный эксперимент. Проверки в этом репозитории написаны, но локально
не выполнялись по ограничению пользователя.
