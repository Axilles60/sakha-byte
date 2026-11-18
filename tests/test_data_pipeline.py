"""Server-only verification fixtures. No external data or downloads required."""

import hashlib
import json
import re
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sakha_byte.benchmark import combine_rotations, load_items, make_prompt, summarize
from sakha_byte.cli import ensemble
from sakha_byte.convert import (base_item, canonical, chunk_document, conllu_sentences,
                                make_exclusion_matcher, ner_doc_key, ner_type_items,
                                prepare_corpus, sentiment_items, split_for, training_record)
from sakha_byte.datasets import download_source, file_sha, records, safe_path, verify_source
from sakha_byte.download import write_json
from sakha_byte.train_qwen import sample_rows


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FastWordTokenizer(WordTokenizer):
    is_fast = True

    def __call__(self, text, **kwargs):
        words = list(re.finditer(r"\S+", text))
        return {"input_ids": [word.group() for word in words],
                "offset_mapping": [word.span() for word in words]}


def fake_source(root, name, filename, rows):
    directory = root/name
    path = directory/"files"/filename
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    write_json(directory/"manifest.json", {"status":"complete", "source":name,
               "provider":"fixture", "repo_id":"fixture/repo", "revision":"a"*40,
               "files":[{"name":filename,"size":path.stat().st_size,"sha256":file_sha(path)}]})
    return path


class DataPipelineTests(unittest.TestCase):
    def test_unsafe_paths_and_modified_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../escape", "/absolute", "folder\\bad", "C:/bad"):
                with self.assertRaises(ValueError):
                    safe_path(root,name)
            path = fake_source(root,"source","data.jsonl",[{"text":"Оҕо"}])
            verify_source(root/"source")
            path.write_text("changed",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"changed"):
                verify_source(root/"source")

    def test_gate_is_checked_before_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sakha_byte.datasets.require_remote"):
            with self.assertRaisesRegex(ValueError,"conditions"):
                download_source("test",{"gated":True},Path(tmp))

    def test_three_class_rotation_and_no_fake_russian_control(self):
        item = base_item("x","Оҕо", "Вопрос?", ["Первый","Второй","Третий"],1,"sentiment",{})
        runs=[]
        for rotation in range(3):
            prompt, order = make_prompt(item,"sah",rotation)
            self.assertNotIn("D.",prompt)
            scores=[-10.0]*3
            scores[order.index(1)] = -0.1
            runs.append((scores,order))
        combined=combine_rotations(runs)
        self.assertEqual(max(range(3),key=combined.__getitem__),1)
        with self.assertRaisesRegex(ValueError,"no ru context"):
            make_prompt(item,"ru")
        summary=summarize([{"category":"sentiment","correct":True,"answer_index":1,
                            "prediction":1,"n_choices":3}])
        self.assertAlmostEqual(summary["chance_accuracy"],1/3)
        self.assertAlmostEqual(summary["by_category"]["sentiment"]["macro_f1"],1/3)

    def test_unicode_and_bilingual_holdout_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"exclusions.jsonl"
            path.write_text(json.dumps({"texts":["Оҕо   кэллэ.","Ребёнок пришёл."]},ensure_ascii=False)+"\n",encoding="utf-8")
            blocked=make_exclusion_matcher([path])
            self.assertTrue(blocked("Заголовок. ОҔО\nкэллэ. Конец"))
            self.assertTrue(blocked("Сегодня ребёнок пришёл. Потом ушёл."))
            self.assertFalse(blocked("Другая строка"))

    def test_heldout_match_can_start_and_end_inside_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exclusions.jsonl"
            path.write_text('{"texts":["нужный контекст здесь", "два слова"]}\n', encoding="utf-8")
            blocked = make_exclusion_matcher([path])
            self.assertTrue(blocked("Ненужный   контекст\nздесь-то"))
            self.assertTrue(blocked("Передва слова-то"))
            self.assertFalse(blocked("Нужный другой контекст находится здесь"))

    def test_document_identity_requires_explicit_fallback(self):
        profile={"source":"fixture","mode":"mono","text_field":"text"}
        with self.assertRaisesRegex(ValueError,"original document"):
            training_record(profile,{"text":"Оҕо кэллэ."},"row")
        output=training_record(profile,{"text":"Оҕо кэллэ."},"row",True)
        self.assertEqual(output["grouping"],"content_only")
        profile["document_field"]="book"
        a=training_record(profile,{"text":"Оҕо кэллэ.","book":"same-book"},"a")
        b=training_record(profile,{"text":"Оҕо барда.","book":"same-book"},"b")
        self.assertEqual(split_for(a["source"]+a["document_id"]),split_for(b["source"]+b["document_id"]))

    def test_source_group_does_not_claim_document_identity(self):
        profile = {"source": "fixture", "mode": "mono", "text_field": "text",
                   "source_field": "source", "document_field": "source"}
        first = training_record(profile, {"text": "Оҕо кэллэ.", "source": "paper"}, "a")
        second = training_record(profile, {"text": "Оҕо барда.", "source": "paper"}, "b")
        self.assertEqual(first["grouping"], "source_only")
        self.assertEqual(first["document_id"], second["document_id"])
        with self.assertRaisesRegex(ValueError, "nonempty source"):
            training_record(profile, {"text": "Оҕо кэллэ.", "source": ""}, "c", True)

    def test_training_subset_is_bounded_deterministic_and_unique(self):
        rows = list(range(20))
        selected = sample_rows(rows, 4, 42)
        self.assertEqual(len(set(selected)), 4)
        self.assertEqual(selected, sample_rows(rows, 4, 42))
        self.assertEqual(sample_rows(rows, None, 42), rows)

    def test_chunking_preserves_words_and_token_limit(self):
        text="Задание: Оҕо\nЯкутский: Үтүө күн"
        for tokenizer in (WordTokenizer(), FastWordTokenizer()):
            chunks=chunk_document(text,tokenizer,2)
            self.assertTrue(all(len(x.split())<=2 for x in chunks))
            self.assertEqual(" ".join(text.split())," ".join(" ".join(chunks).split()))
            self.assertNotIn("\ufffd","".join(chunks))

    def test_opt_in_word_split_preserves_every_character(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
        word = "Үтүөҥҕ"
        with self.assertRaisesRegex(ValueError, "One word"):
            chunk_document(word, tokenizer, 2)
        chunks = chunk_document(word, tokenizer, 2, split_overlong_words=True)
        self.assertEqual("".join(chunks), word)
        self.assertTrue(all(len(chunk) <= 2 for chunk in chunks))

    def test_ner_split_key_matches_upstream_bytes_and_offsets(self):
        task={"id":1,"data":{"text":"Өрө\u00a0Аня"},"annotations":[{"result":[
            {"type":"labels","value":{"start":4,"end":7,"text":"Аня","labels":["PER"]}}
        ]}]}
        key="sha1:"+hashlib.sha1(task["data"]["text"].encode()).hexdigest()
        self.assertEqual(ner_doc_key(task),key)
        self.assertEqual(list(ner_type_items([task],{key},{}))[0]["answer_index"],0)
        self.assertEqual(list(ner_type_items([task],{"other"},{})),[])
        task["annotations"][0]["result"][0]["value"]["end"]=6
        with self.assertRaisesRegex(ValueError,"span text"):
            list(ner_type_items([task],{key},{}))

    def test_saas_needs_heldout_and_explicit_label_map(self):
        profile={"text_field":"text","label_field":"sentiment","label_map":{"Positive":0,"Negative":1,"Neutral":2},"skip_labels":["Skip"]}
        rows=[("saas.csv",0,{"text":"Оҕо","sentiment":"Positive"})]
        with self.assertRaisesRegex(ValueError,"held-out"):
            list(sentiment_items(rows,profile,{}))
        items=list(sentiment_items(rows,profile,{}, {"saas.csv:0"}))
        self.assertEqual(len(items[0]["choices"]),3)
        with self.assertRaisesRegex(ValueError,"absent"):
            list(sentiment_items(rows,profile,{}, {"missing"}))

    def test_three_class_ensemble_retains_macro_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta={"status":"complete","benchmark_sha256":"x","prompt_version":"v2",
                  "language":"sah","rotations":"all","scoring":"v2","n":1,"draft":True}
            row={"id":"item","category":"sentiment","answer_index":1,
                 "scores":[-4.0,-0.1,-3.0],"rotations":[{"prompt_sha256":"same"}],"group_id":"doc"}
            for name,key in (("bwen","bwen"),("eva","evabyte")):
                write_json(root/name/"metadata.json",{**meta,"model_key":key})
                (root/name/"predictions.jsonl").write_text(json.dumps(row)+"\n",encoding="utf-8")
            ensemble(Namespace(bwen_run=root/"bwen",evabyte_run=root/"eva",bwen_weight=0.5,output=root/"out"))
            summary=json.loads((root/"out/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["correct"],1)
            self.assertAlmostEqual(summary["chance_accuracy"],1/3)
            self.assertIn("macro_f1",summary["by_category"]["sentiment"])

    def test_conllu_skips_multitokens_and_empty_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.conllu"
            path.write_text("# sent_id = s1\n# text = Оҕо.\n"
                            "1-2\tОҕо\t_\t_\t_\t_\t_\t_\t_\t_\n"
                            "1\tОҕо\tоҕо\tNOUN\t_\t_\t0\troot\t_\t_\n"
                            "1.1\t_\t_\t_\t_\t_\t_\t_\t_\t_\n\n",encoding="utf-8")
            self.assertEqual(len(list(conllu_sentences(path))[0]["tokens"]),1)

    def test_corpus_pipeline_hashes_splits_dedup_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[{"text":f"Оҕо кэллэ. Документ {i} мазмуна.","book":f"book-{i}"} for i in range(200)]
            rows += [rows[1].copy(), {"text":"Тестовый отрывок нельзя обучать.","book":"heldout"}]
            fake_source(root/"raw","fixture","data.jsonl",rows)
            mappings=root/"mapping.json"
            write_json(mappings,{"profiles":{"mono":{"source":"fixture","files":["*.jsonl"],"format":"jsonl",
                       "mode":"mono","text_field":"text","document_field":"book"}}})
            exclusion=root/"exclude.jsonl"
            exclusion.write_text(json.dumps({"texts":["Тестовый отрывок"]},ensure_ascii=False)+"\n",encoding="utf-8")
            args=Namespace(mappings=mappings,profiles=["mono"],raw_root=root/"raw",output=root/"out",
                           exclude=[exclusion],tokenizer_path=root/"tokenizer",seed=42,validation_fraction=0.2,
                           test_fraction=0.2,max_tokens=3,min_chars=1,allow_content_groups=False)
            fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:WordTokenizer()))
            with patch("sakha_byte.convert.require_remote"),patch.dict("sys.modules",{"transformers":fake}):
                manifest=prepare_corpus(args)
                args.output = root / "parallel"
                args.workers = 2
                parallel = prepare_corpus(args)
            self.assertEqual(manifest["outputs"], parallel["outputs"])
            self.assertEqual(manifest["counts"], parallel["counts"])
            self.assertEqual(manifest["status"],"complete")
            self.assertEqual(manifest["counts"]["heldout_rejected"],1)
            self.assertEqual(manifest["counts"]["duplicate_records"],1)
            documents={key:{r["document_id"] for r in records(root/"out"/f"{key}.jsonl",{})}
                       for key in ("train","validation","test")}
            self.assertFalse(documents["train"] & documents["validation"])
            self.assertFalse(documents["train"] & documents["test"])
            self.assertFalse(documents["validation"] & documents["test"])
            for key in documents:
                self.assertEqual(manifest["outputs"][key],file_sha(root/"out"/f"{key}.jsonl"))


if __name__ == "__main__":
    unittest.main()
