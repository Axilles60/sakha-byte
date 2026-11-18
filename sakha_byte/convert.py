"""Deterministic data conversion; writes outputs only in explicit server commands."""

import hashlib
import json
import random
import re
import sys
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import local

from .benchmark import load_items
from .datasets import field, file_sha, records, require_remote, safe_path, select_files, verify_source
from .download import read_json, write_json


def clean(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty text; correct the field mapping")
    return unicodedata.normalize("NFC", value).strip()


def canonical(value):
    return " ".join(clean(value).casefold().split())


def text_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def split_for(group, seed=42, validation_fraction=0.05, test_fraction=0.05):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value < test_fraction else "validation" if value < test_fraction+validation_fraction else "train"


def input_rows(raw_root, profile):
    directory = raw_root / profile["source"]
    for path in select_files(directory, profile["files"]):
        relative = path.relative_to((directory/"files").resolve()).as_posix()
        for index, row in enumerate(records(path, profile)):
            yield relative, index, row


def training_record(profile, row, row_key, allow_content_groups=False):
    mode = profile["mode"]
    if mode == "mono":
        language = field(row, profile.get("language_field"))
        if language is not None and language != profile["language_value"]:
            return None
        components = [clean(field(row, profile["text_field"]))]
        text = components[0]
    elif mode == "parallel":
        sah = clean(field(row, profile["sah_field"]))
        ru = clean(field(row, profile["ru_field"]))
        components = [sah, ru]
        # Language modeling of aligned bilingual text, not masked instruction SFT.
        text = f"Русский: {ru}\nЯкутский: {sah}"
    elif mode == "instruction":
        instruction = clean(field(row, profile["instruction_field"]))
        extra = field(row, profile.get("input_field"))
        answer = clean(field(row, profile["output_field"]))
        components = [instruction, answer]
        if extra:
            components.append(clean(extra))
        text = f"Задание: {instruction}" + (f"\nВход: {clean(extra)}" if extra else "") + f"\nОтвет: {answer}"
    else:
        raise ValueError("Unknown training mode")
    origin = field(row, profile.get("source_field"))
    origin = str(origin) if origin is not None else profile["source"]
    document_field = profile.get("document_field")
    document = field(row, document_field)
    if document_field and document_field == profile.get("source_field"):
        if document is None or not str(document).strip():
            raise ValueError("Source-grouped split requires a nonempty source field")
        grouping = "source_only"
    elif document is None or str(document).strip() == "":
        if not allow_content_groups:
            raise ValueError("No original document ID. Configure document_field or explicitly allow content groups")
        document = "content:" + text_sha(text)
        grouping = "content_only"
    else:
        grouping = "source_document"
    return {"text":text, "source":profile["source"]+":"+origin,
            "document_id":str(document), "grouping":grouping, "components":components,
            "source_record":row_key, "synthetic":bool(profile.get("synthetic")), "mode":mode}


def make_exclusion_matcher(paths):
    texts = set()
    for path in paths:
        for row in records(path, {"format":"jsonl"}):
            candidates = row.get("texts") or [row.get("context_sah"), row.get("context_ru"), row.get("text")]
            for text in candidates:
                if isinstance(text, str) and text.strip():
                    texts.add(canonical(text))
    if not texts:
        raise ValueError("Exclusion files contain no text")
    unindexed = []
    by_anchor = defaultdict(list)
    for text in sorted(texts):
        interior_words = text.split()[1:-1]
        if interior_words:
            by_anchor[max(interior_words, key=len)].append(text)
        else:
            unindexed.append(text)

    def matches(value):
        value = canonical(value)
        if any(text in value for text in unindexed):
            return True
        # An interior word is present even when the match starts or ends inside a word.
        return any(text in value for word in set(value.split()) for text in by_anchor.get(word, ()))

    return matches


def split_word(word, tokenizer, max_tokens):
    pending = [word]
    chunks = []
    while pending:
        text = pending.pop()
        if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
            chunks.append(text)
        elif len(text) == 1:
            raise ValueError("One Unicode character exceeds token budget")
        else:
            middle = len(text) // 2
            pending.extend((text[middle:], text[:middle]))
    return chunks


def chunk_document(text, tokenizer, max_tokens, split_overlong_words=False):
    """Split at whitespace boundaries; never decode isolated UTF-8 fragments."""
    token_starts = None
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        token_count = len(encoded["input_ids"])
        token_starts = [start for start, _ in encoded["offset_mapping"]]
        if any(a > b for a, b in zip(token_starts, token_starts[1:])):
            token_starts = None
    else:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count <= max_tokens:
        return [text]
    words = list(re.finditer(r"\S+",text))
    word_ends = [word.end() for word in words]
    chunks, left = [], 0
    while left < len(words):
        start = words[left].start()
        if token_starts is not None:
            next_token = bisect_left(token_starts, start) + max_tokens
            boundary = token_starts[next_token] if next_token < len(token_starts) else len(text)
            right = bisect_right(word_ends, boundary, lo=left)
            if right > left:
                candidate = text[start:word_ends[right - 1]]
                if len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_tokens:
                    chunks.append(candidate)
                    left = right
                    continue
        low, high, best = 1, min(max_tokens,len(words)-left), 0
        while low <= high:
            middle = (low+high)//2
            trial = text[start:word_ends[left+middle-1]]
            if len(tokenizer.encode(trial,add_special_tokens=False)) <= max_tokens:
                best, low = middle, middle+1
            else:
                high = middle-1
        if not best:
            if not split_overlong_words:
                raise ValueError("One word exceeds token budget; inspect OCR/data instead of truncating")
            chunks.extend(split_word(text[start:word_ends[left]], tokenizer, max_tokens))
            left += 1
            continue
        chunk = text[start:word_ends[left+best-1]]
        if len(tokenizer.encode(chunk,add_special_tokens=False)) > max_tokens:
            raise ValueError("Chunk exceeds token budget")
        chunks.append(chunk)
        left += best
    return chunks


def prepare_corpus(args):
    require_remote()
    profiles = read_json(args.mappings)["profiles"]
    chosen = [profiles[name] for name in args.profiles]
    if len(set(args.profiles)) != len(args.profiles):
        raise ValueError("Duplicate profile")
    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1 or args.validation_fraction+args.test_fraction >= 1:
        raise ValueError("Invalid split proportions")
    workers = getattr(args, "workers", 1)
    split_overlong_words = getattr(args, "split_overlong_words", False)
    if args.max_tokens < 1 or args.min_chars < 1 or workers < 1:
        raise ValueError("Token/character limits and workers must be positive")
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    sources = {p["source"]:verify_source(args.raw_root/p["source"]) for p in chosen}
    excluded = make_exclusion_matcher(args.exclude)
    from transformers import AutoTokenizer
    def load_tokenizer():
        return AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True, trust_remote_code=False)

    tokenizer = load_tokenizer()
    thread_state = local()

    def chunk_record(text, key):
        active_tokenizer = tokenizer
        if workers > 1:
            if not hasattr(thread_state, "tokenizer"):
                thread_state.tokenizer = load_tokenizer()
            active_tokenizer = thread_state.tokenizer
        try:
            return chunk_document(text, active_tokenizer, args.max_tokens, split_overlong_words)
        except ValueError as error:
            raise ValueError(f"Token chunking failed at {key}: {error}") from error

    args.output.mkdir(parents=True)
    manifest = {"status":"running", "profiles":args.profiles, "mappings_sha256":file_sha(args.mappings),
                "profile_configs":{name:profiles[name] for name in args.profiles}, "min_chars":args.min_chars,
                "sources":sources, "seed":args.seed, "max_tokens_without_eos":args.max_tokens,
                "allow_content_groups":args.allow_content_groups,
                "workers":workers,
                "split_overlong_words":split_overlong_words,
                "validation_fraction":args.validation_fraction, "test_fraction":args.test_fraction,
                "exclusions":{str(p):file_sha(p) for p in args.exclude},
                "tokenizer_files":{p.name:file_sha(p) for p in sorted(args.tokenizer_path.glob("*"))
                                   if p.is_file() and p.suffix in (".json", ".txt", ".model")},
                "quality_filter":"NFC, whitespace/case normalized dedup, reject replacement character, min_chars",
                "language_quality":"Only optional explicit source-language field; no automatic LID or semantic OCR correction",
                "mode":"continued_language_modeling_including_prompt_tokens",
                "chunking":"whitespace boundaries, fast tokenizer offset estimate with exact token-count verification"}
    write_json(args.output/"manifest.json", manifest)
    counts, seen, seen_chunks, component_splits = Counter(), set(), set(), {}
    split_groups = defaultdict(set)

    def write_chunks(outputs, record, group, split, chunks):
        for chunk_index, text in enumerate(chunks):
            chunk_hash = text_sha(text)
            if chunk_hash in seen_chunks:
                counts["duplicate_chunks"] += 1
                continue
            seen_chunks.add(chunk_hash)
            output = {**record, "text":text, "chunk_index":chunk_index,
                      "text_sha256":chunk_hash, "split":split}
            outputs[split].write(json.dumps(output, ensure_ascii=False)+"\n")
            counts[split] += 1
            split_groups[split].add(group)

    try:
        with ExitStack() as stack:
            outputs = {key:stack.enter_context((args.output/f"{key}.jsonl").open("w", encoding="utf-8"))
                       for key in ("train","validation","test")}
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers)) if workers > 1 else None
            pending = deque()
            for profile in chosen:
                for filename, index, row in input_rows(args.raw_root, profile):
                    counts["input_records"] += 1
                    if counts["input_records"] % 10000 == 0:
                        print(f"Corpus progress: {dict(counts)}", file=sys.stderr, flush=True)
                    key = f"{profile['source']}:{filename}:{index}"
                    if profile["mode"] == "mono":
                        raw_text = field(row, profile["text_field"])
                        if raw_text is None or (isinstance(raw_text, str) and not raw_text.strip()):
                            counts["quality_rejected"] += 1
                            continue
                    record = training_record(profile, row, key, args.allow_content_groups)
                    if record is None:
                        counts["language_rejected"] += 1
                        continue
                    if "\ufffd" in record["text"] or len(record["text"]) < args.min_chars:
                        counts["quality_rejected"] += 1
                        continue
                    # Catch held-out text on either side of bilingual/instruction records.
                    if any(excluded(x) for x in record["components"]):
                        counts["heldout_rejected"] += 1
                        continue
                    fingerprint = text_sha(record["text"])
                    if fingerprint in seen:
                        counts["duplicate_records"] += 1
                        continue
                    seen.add(fingerprint)
                    group = record["source"]+":"+record["document_id"]
                    split = split_for(group, args.seed, args.validation_fraction, args.test_fraction)
                    component_keys = [text_sha(x) for x in record["components"]]
                    if any(k in component_splits and component_splits[k] != split for k in component_keys):
                        counts["component_cross_split_rejected"] += 1
                        continue
                    for k in component_keys:
                        component_splits[k] = split
                    # Check before chunking: exclusion boundaries must not be lost.
                    del record["components"]
                    if pool is None:
                        write_chunks(outputs, record, group, split, chunk_record(record["text"], key))
                        continue
                    pending.append((record, group, split, pool.submit(chunk_record, record["text"], key)))
                    if len(pending) >= 2 * workers:
                        item, item_group, item_split, future = pending.popleft()
                        write_chunks(outputs, item, item_group, item_split, future.result())
            for item, item_group, item_split, future in pending:
                write_chunks(outputs, item, item_group, item_split, future.result())
        if any(counts[k] == 0 for k in ("train","validation","test")):
            raise ValueError("An output split is empty; provide more documents or change split fractions")
        manifest.update(status="complete", counts=dict(counts),
                         group_counts={key:len(groups) for key,groups in split_groups.items()},
                        outputs={key:file_sha(args.output/f"{key}.jsonl") for key in ("train","validation","test")})
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, counts=dict(counts))
        raise
    finally:
        write_json(args.output/"manifest.json", manifest)
    return manifest


def conllu_sentences(path):
    text, sent_id, tokens = None, None, []
    lines = path.read_text(encoding="utf-8-sig").splitlines() + [""]
    for line in lines:
        if not line.strip():
            if tokens:
                if not text or not sent_id:
                    raise ValueError("CoNLL-U sentence needs # text and # sent_id")
                yield {"id":sent_id, "text":text, "tokens":tokens}
            text, sent_id, tokens = None, None, []
        elif line.startswith("# text = "):
            text = line[len("# text = "):]
        elif line.startswith("# sent_id = "):
            sent_id = line[len("# sent_id = "):]
        elif not line.startswith("#"):
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError("CoNLL-U requires ten columns")
            if fields[0].isdigit():
                tokens.append({"id":fields[0], "form":fields[1], "lemma":fields[2], "upos":fields[3]})


def base_item(key, context, question, choices, answer, category, provenance):
    return {"id":key, "category":category, "context_sah":clean(context), "context_ru":None,
            "question_ru":question, "choices":[clean(x) for x in choices], "answer_index":answer,
            "review_status":"draft", "reviewer":None, "rationale_ru":"Ключ перенесён из исходной разметки; новое задание требует проверки.",
            "provenance":provenance}


def ud_items(sentences, source, seed=42):
    pools = defaultdict(set)
    for sentence in sentences:
        for token in sentence["tokens"]:
            if token["lemma"] != "_":
                pools[token["upos"]].add(clean(token["lemma"]))
    for sentence in sentences:
        for token in sentence["tokens"]:
            if token["lemma"] == "_":
                continue
            gold = clean(token["lemma"])
            candidates = sorted(pools[token["upos"]] - {gold})
            if len(candidates) < 3:
                continue
            key = f"ud:{sentence['id']}:{token['id']}"
            rng = random.Random(f"{seed}:{key}")
            choices = [gold] + rng.sample(candidates, 3)
            rng.shuffle(choices)
            item = base_item(key, sentence["text"],
                             f"Какова начальная форма слова «{clean(token['form'])}» (позиция {token['id']}) в тексте?",
                             choices, choices.index(gold), "ud_lemma_mcq", source)
            item["group_id"] = sentence["id"]
            yield item


def ner_doc_key(task):
    # Exact upstream get_doc_key algorithm, before any NFC/whitespace normalization.
    data = task["data"]
    for key in ("file_upload", "document_id", "doc_id", "source", "file"):
        if data.get(key):
            return f"{key}:{data[key]}"
    return "sha1:" + hashlib.sha1(data["text"].encode("utf-8")).hexdigest()


def ner_type_items(tasks, split, source):
    types = ["PER", "LOC", "ORG"]
    choices = ["Человек (PER)", "Место (LOC)", "Организация (ORG)"]
    for task in tasks:
        if ner_doc_key(task) not in split:
            continue
        annotations = task.get("annotations", [])
        if not annotations or annotations[-1].get("was_cancelled"):
            continue
        text = task["data"]["text"]
        for index, result in enumerate(annotations[-1].get("result", [])):
            value = result.get("value", {})
            labels = value.get("labels", [])
            if result.get("type") != "labels" or not labels:
                continue
            if len(labels) != 1 or labels[0] not in types:
                raise ValueError("Unexpected entity label")
            start, end = value["start"], value["end"]
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
                raise ValueError("Invalid NER character offsets")
            span = text[start:end]
            if value.get("text") is not None and value["text"] != span:
                raise ValueError("NER span text differs from offsets")
            item = base_item(f"ner-type:{task['id']}:{index}", text,
                             f"Какой тип у выделенной сущности «{clean(span)}» в тексте?",
                             choices, types.index(labels[0]), "gold_span_entity_type", source)
            item.update(group_id=ner_doc_key(task), source_span={"start":start,"end":end,"text":span})
            yield item


def sentiment_items(rows, profile, source, heldout_ids=None):
    if not profile.get("split_field") and heldout_ids is None:
        raise ValueError("SAAS requires split_field/split_value or explicit held-out --ids-file")
    if set(profile["label_map"].values()) != {0,1,2}:
        raise ValueError("SAAS label_map must use indices 0=positive, 1=negative, 2=neutral")
    used_ids = set()
    for filename, index, row in rows:
        record_id = field(row, profile.get("id_field"))
        record_id = str(record_id) if record_id is not None else f"{filename}:{index}"
        text = field(row, profile["text_field"])
        label = str(field(row, profile["label_field"]))
        if label in profile.get("skip_labels", []):
            continue
        if label not in profile["label_map"]:
            raise ValueError(f"Unmapped SAAS label {label!r}; configure explicit label_map")
        if profile.get("split_field") and field(row, profile["split_field"]) != profile["split_value"]:
            continue
        if heldout_ids is not None and record_id not in heldout_ids:
            continue
        if record_id in used_ids:
            raise ValueError("Duplicate sentiment record ID")
        used_ids.add(record_id)
        yield base_item("saas:"+record_id, text, "Какова тональность текста?",
                        ["Положительная", "Отрицательная", "Нейтральная"],
                        profile["label_map"][label], "sentiment", source)
    if heldout_ids is not None and used_ids != heldout_ids:
        raise ValueError("Held-out IDs absent, skipped, or outside configured source split")


def prepare_eval(args):
    require_remote()
    profile = read_json(args.mappings)["profiles"][args.profile]
    directory = args.raw_root/profile["source"]
    manifest = verify_source(directory)
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    provenance = {"source":profile["source"], "revision":manifest["revision"],
                  "manifest_sha256":file_sha(directory/"manifest.json"), "adapter":profile["adapter"]}
    files = select_files(directory, profile["files"])
    exclusions = []
    adapter = profile["adapter"]
    if adapter == "ud_lemma":
        sentences = [row for file in files for row in conllu_sentences(file)]
        items = list(ud_items(sentences, provenance, profile.get("seed", 42)))
        exclusions = [{"texts":[s["text"]], "source_id":s["id"]} for s in sentences]
    elif adapter == "labelstudio_ner_type":
        tasks = [row for file in files for row in records(file, {"format":"json"})]
        split_path = safe_path(directory/"files", profile["split_file"])
        split_data = read_json(split_path)
        splits = {k:set(split_data[k+"_doc_keys"]) for k in ("train","dev","test")}
        if any(splits[a] & splits[b] for a,b in (("train","dev"),("train","test"),("dev","test"))):
            raise ValueError("Published NER split overlaps")
        all_keys = set().union(*splits.values())
        valid_tasks = [t for t in tasks if t.get("annotations") and not t["annotations"][-1].get("was_cancelled")]
        if any(ner_doc_key(t) not in all_keys for t in valid_tasks):
            raise ValueError("NER file differs from published split")
        items = list(ner_type_items(tasks, splits[profile["split"]], provenance))
        exclusions = [{"texts":[t["data"]["text"]], "source_id":str(t["id"])} for t in tasks]
    elif adapter == "sentiment":
        rows = list(input_rows(args.raw_root, profile))
        raw_ids = read_json(args.ids_file) if args.ids_file else None
        if raw_ids is not None and (not isinstance(raw_ids,list) or not all(isinstance(x,str) for x in raw_ids)
                                    or len(set(raw_ids)) != len(raw_ids)):
            raise ValueError("ids-file must be a JSON array of unique strings")
        heldout_ids = set(raw_ids) if raw_ids is not None else None
        items = list(sentiment_items(rows, profile, provenance, heldout_ids))
        exclusions = [{"texts":[clean(field(r, profile["text_field"]))], "source_id":f"{f}:{i}"} for f,i,r in rows]
    else:
        raise ValueError("Unknown evaluation adapter")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Limit must be positive")
        # All source contexts stay excluded even when only a small pilot is emitted.
        items = items[:args.limit]
    if not items:
        raise ValueError("No evaluation items; inspect source schemas/splits")
    args.output.mkdir(parents=True)
    for filename, rows in (("benchmark.jsonl",items),("exclusions.jsonl",exclusions)):
        (args.output/filename).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")
    load_items(args.output/"benchmark.jsonl", allow_draft=True)
    write_json(args.output/"manifest.json", {"status":"complete", **provenance, "profile":args.profile,
               "profile_config":profile, "ids_sha256":file_sha(args.ids_file) if args.ids_file else None,
               "n":len(items), "excluded_contexts":len(exclusions), "review_status":"draft",
               "outputs":{key:file_sha(args.output/key) for key in ("benchmark.jsonl","exclusions.jsonl")}})
    return {"items":len(items), "excluded_contexts":len(exclusions), "status":"draft"}
