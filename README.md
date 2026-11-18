# sakha_byte

Код и конфигурации для оценки якутского языка на удалённой GPU-ВМ.
Проект сравнивает Qwen3‑8B и Bwen‑8B до и после якутской адаптации.
Ансамбль Bwen‑8B + EvaByte оценивается отдельно.

Команды ниже предназначены для арендованной Linux GPU-ВМ. Обучение, загрузки
данных и инференс выполняются на сервере. Репозиторий содержит код, конфигурации
и документацию.

**Статус.** На IMMERS H100 завершён первый раунд. Проведены 200 шагов LoRA на Qwen,
перенос дельты в Bwen и оценки на UD, SAAS и минимальных парах. На минимальных
парах адаптация улучшила долю правильных сдвигов оценок у обеих моделей. Вклад BPE
отдельно пока не установлен.

[Итоговый отчёт первого раунда](docs/round-1-report.md) · [План второго раунда](docs/round-2-plan.md)

## Что входит

- `configs/models.json`: модели и зафиксированные SHA ревизий Hugging Face.
- `sakha_byte/download.py`: загрузка с проверкой места, манифестами и хешами данных.
- `sakha_byte/pilot.py`: генератор 16 черновых заданий; создаёт JSONL на сервере.
- `sakha_byte/benchmark.py`: проверка структуры, промпты, перестановки и метрики.
- `sakha_byte/runtime.py`: отдельные адаптеры Bwen, Qwen и EvaByte.
- `sakha_byte/cli.py`: оценка моделей и объединение результатов Bwen + EvaByte.
- `sakha_byte/train_qwen.py`: пилотное языковое дообучение Qwen через LoRA.
- `scripts/saas_lora_transfer.py`: проверка соответствия слоёв Qwen и Bwen и перенос LoRA.
- `scripts/sakha_suffix_pairs.py`: построение и оценка минимальных пар на якутском.
- `scripts/sakha_suffix_calibrated.py`: оценка минимальных пар с контролями для калибровки.
- `tests/test_protocol.py`: тесты протокола для запуска на сервере.
- `docs/immers.md`: исследование инфраструктуры и порядок удалённого запуска.
- `docs/protocol.md`: ограничения эксперимента и проверка носителем.
- `configs/dataset_registry.json` и `docs/data_plan.md`: источники и их роли.
- `sakha_byte/datasets.py`, `sakha_byte/convert.py`: загрузчики HF/GitHub и конвертеры.
- `docs/data_pipeline.md`: полный серверный маршрут от загрузки до обучения и ансамбля.

В базовом режиме нет chat templates, SFT-моделей и сгенерированных свободных ответов.
Все модели оценивают один и тот же промпт с 2–4 вариантами A–D. Это позволяет проверить
первую связку без попытки складывать несовместимые веса или словари моделей.

## 1. Удалённые окружения

Скопируйте этот каталог на сервер и откройте SSH-сессию. Дальнейшие пути относительны
корню проекта на сервере. Рекомендуемая первая машина: Ubuntu 24.04, A100 80GB,
64GB RAM, диск 240GB. Модели запускаются по очереди на одной GPU.

Python 3.11 или 3.12 и драйвер NVIDIA должны быть установлены на сервере.
`uv` используется только как удобный менеджер окружений. При его отсутствии
сначала установите его на сервере по официальной документации.

```bash
# Выполнять только в SSH-сессии на удалённой ВМ!
export SAKHA_REMOTE_EXECUTION=immers
uv venv --python 3.11 .venv-bwen
uv pip install --python .venv-bwen/bin/python -e .
uv pip install --python .venv-bwen/bin/python -r requirements/bwen-qwen.txt

uv venv --python 3.11 .venv-eva
uv pip install --python .venv-eva/bin/python -e .
uv pip install --python .venv-eva/bin/python -r requirements/evabyte.txt
```

Проверьте, что установлена CUDA-сборка PyTorch, соответствующая драйверу и GPU.
Окружения разделены: Bwen использует xLSTM и современный Transformers, EvaByte
использует более старую реализацию Cache. Точные версии в requirements — начальные
конфигурации для серверной проверки, а не проверенный lock-файл.
Для Bwen на H100 с PyTorch `2.8.0+cu128` нужны компилятор и заголовки CUDA 12.8.
После подключения [репозитория NVIDIA для Ubuntu 24.04](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)
установите `cuda-nvcc-12-8` и `cuda-cudart-dev-12-8`. Перед запуском Bwen задайте
`CUDA_HOME=/usr/local/cuda-12.8` и добавьте `$CUDA_HOME/bin` к `PATH`.
Для RTX 5090 эта старая конфигурация EvaByte/PyTorch не подходит без отдельной
проверки поддержки Blackwell; для первого запуска предпочтительнее A100.

## 2. Тест и внешние данные

```bash
.venv-bwen/bin/python -m unittest discover -s tests -v
.venv-bwen/bin/sakha-byte create-pilot
.venv-bwen/bin/sakha-byte validate --allow-draft
.venv-bwen/bin/sakha-byte plan --allow-draft
.venv-bwen/bin/sakha-byte download-data
```

`create-pilot` создаёт `data/eval/pilot.jsonl`. Это оригинальный AI-черновик:
16 заданий, 8 тематических пар, по 4 варианта ответа. Носитель должен проверить
якутский текст, контрольный перевод, ключ и отсутствие двусмысленности. После
проверки задайте `review_status: "approved"` и `reviewer` для каждого задания.
Без `--allow-draft` оценка непроверенных заданий запрещена.

`download-data` скачивает внешний морфологический ресурс UniMorph Sakha
в `data/raw/unimorph_sah/`, фиксирует Git commit и SHA-256 файлов.
Последующие скачивания используют тот же commit из `data.lock.json`.
UniMorph не является корпусом понимания текста и автоматически не включается
в обучение. Исходные условия использования нужно проверить до перераспространения.

## 3. Загрузка моделей на сервер

```bash
# Сначала небольшие конфигурации, код моделей и токенизаторы.
.venv-bwen/bin/sakha-byte download-model --model bwen
.venv-bwen/bin/sakha-byte download-model --model qwen
.venv-eva/bin/sakha-byte download-model --model evabyte

# Затем полные веса; команды явно требуют --weights.
.venv-bwen/bin/sakha-byte download-model --model bwen --weights
.venv-bwen/bin/sakha-byte download-model --model qwen --weights
.venv-eva/bin/sakha-byte download-model --model evabyte --weights
```

Загрузчик хранит снимки в `artifacts/hf/`, манифесты в `artifacts/models/`.
Он выбирает только корневые файлы моделей, исключая дополнительные training
checkpoints в подкаталогах. У Bwen дополнительно скачивается исходный токенизатор
Qwen3-8B по отдельному SHA: upstream открывает его внутри своего токенизатора.
Во время инференса ссылка перенаправляется на этот локальный снимок.

Полные веса занимают десятки GB. Нужен запас для двух окружений, LoRA-чекпойнтов
и итогового merged Qwen. Загрузка не исполняет remote code; при оценке собственный
код Bwen/EvaByte загружается через `trust_remote_code=True` из закреплённого снимка.

## 4. Первичная проверка инференса

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Проверка одного задания; это НЕ итоговый прогон.
.venv-bwen/bin/sakha-byte evaluate --model qwen --allow-draft --limit 1 --output results/check-qwen
.venv-bwen/bin/sakha-byte evaluate --model bwen --allow-draft --limit 1 --output results/check-bwen
.venv-eva/bin/sakha-byte evaluate --model evabyte --allow-draft --limit 1 --output results/check-eva
```

В случае несовместимости библиотек сначала исправьте окружение на сервере и
зафиксируйте версии. Сценарий не подменяет модель, не переключается скрыто на CPU,
не квантует веса и не обрезает промпт.

## 5. Полный пилот и связка

```bash
.venv-bwen/bin/sakha-byte evaluate --model qwen --allow-draft --output results/qwen-base-sah
.venv-bwen/bin/sakha-byte evaluate --model bwen --allow-draft --output results/bwen-base-sah
.venv-eva/bin/sakha-byte evaluate --model evabyte --allow-draft --output results/eva-base-sah
.venv-bwen/bin/sakha-byte ensemble --bwen-run results/bwen-base-sah --evabyte-run results/eva-base-sah --output results/ensemble-sah
```

По умолчанию каждая модель оценивает все циклические порядки вариантов ответа:
четыре для пилота и UD, три для SAAS и классификации сущностей.
Связка усредняет логарифмы условных вероятностей ответов двух моделей с весами
0.5/0.5; это геометрическое объединение. Вес можно изменить через `--bwen-weight`,
но выбирать его следует на отдельной development-выборке, до итогового теста.
Для каждого нового вопроса ансамбль требует оценки обеих моделей. Здесь их можно
получать последовательно и затем объединять; одновременная загрузка не нужна.

В каждом каталоге результата: `metadata.json`, `predictions.jsonl`, `summary.json`.
Сохраняются хеш теста, SHA модели, версии библиотек, GPU, оценки всех перестановок,
длительность forward pass и признак черновых данных. Совпадение протокола и хешей
промптов проверяется перед объединением результатов.

Повторите три одиночных прогона с `--language ru` и новыми каталогами, затем
соберите отдельный русский ансамбль. Это контроль понимания инструкции и задания.
Он не заменяет якутскую проверку и не является обучением.

## 6. Контрольная Qwen после дообучения

Для `sakha_mono` используйте профиль `mono_by_source` и маршрут подготовки
с [корпусным манифестом и коротким тестом LoRA](docs/data_pipeline.md#3-подготовить-корпус-qwen).
Сначала проверьте один шаг оптимизатора и загрузку объединённого чекпойнта.

Нужны внешние документы на якутском, подготовленные отдельно на сервере.
Формат строк `data/corpus/train.jsonl` и `data/corpus/validation.jsonl`:

```json
{"document_id":"source-document-001","source":"licensed-source","text":"Якутский текст документа или отрывка"}
```

Разделите по исходным документам и по возможности источникам до нарезки отрывков.
Не помещайте пилот, его переводы или варианты ответов в корпус. Код проверяет
совпадение документов, точные дубли и вхождение контекстов пилота, но не выявляет
все перефразы и близкие дубли. В текущем рецепте отрывок должен умещаться в
2048 токенов; слишком длинный текст вызовет ошибку вместо скрытой обрезки.

```bash
uv pip install --python .venv-bwen/bin/python -r requirements/train-qwen.txt
.venv-bwen/bin/python -m sakha_byte.train_qwen --train-file data/corpus/train.jsonl --validation-file data/corpus/validation.jsonl --benchmark data/eval/pilot.jsonl --output artifacts/qwen-sah-lora
.venv-bwen/bin/sakha-byte evaluate --model qwen --stage adapted --checkpoint artifacts/qwen-sah-lora/merged --allow-draft --output results/qwen-adapted-sah
.venv-bwen/bin/sakha-byte report results/qwen-base-sah results/qwen-adapted-sah results/bwen-base-sah results/eva-base-sah results/ensemble-sah
```

Это continued language modeling через LoRA в BF16, без квантизации, с последующим
слиянием адаптера. 200 шагов — технический пилот, а не обещание достаточного
обучения языку. Рецепт проверяет до 256 строк валидации; настройка
`max_validation_rows` определяет размер выборки.
Измерения Qwen до адаптации, одиночных Bwen/EvaByte и ансамбля нужно сохранить:
без них нельзя понять, откуда взялось изменение качества.

В первом раунде для адаптации использован монокорпус `sakha_mono`. Загрузчик
и конвертер описаны в [маршруте подготовки данных](docs/data_pipeline.md).

## Лицензия

© 2026 Axilles60. Оригинальный код, конфигурации и документация Sakha Byte
публикуются под [Creative Commons Attribution–NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).
[Полный текст лицензии](LICENSE).

При некоммерческом распространении или переработке укажите Sakha Byte и Axilles60,
дайте ссылку на исходный репозиторий и лицензию, отметьте внесённые изменения.
Подключаемые модели, датасеты и библиотеки распространяются их правообладателями
на собственных условиях. Ссылки на модели приведены ниже, ссылки на датасеты —
в [реестре источников](configs/dataset_registry.json).

## Источники

- [Byteification / Bolmo, Nature](https://pmc.ncbi.nlm.nih.gov/articles/PMC13686933/)
- [bolmo-core, реализация и оценка](https://github.com/allenai/bolmo-core)
- [Bwen-8B](https://huggingface.co/allenai/Bwen-8B)
- [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)
- [EvaByte, описание метода](https://hkunlp.github.io/blog/2025/evabyte/)
- [EvaByte, код и особенности запуска](https://github.com/OpenEvaByte/evabyte)
- [UniMorph Sakha](https://github.com/unimorph/sah)
