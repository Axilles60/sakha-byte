"""Run these tests on the remote server; no weights, CUDA or network needed."""

import json
import tempfile
import unittest
from pathlib import Path

from sakha_byte.benchmark import combine_rotations, load_items, make_prompt, summarize
from sakha_byte.cli import ensemble
from sakha_byte.download import write_json
from sakha_byte.pilot import create_pilot
from argparse import Namespace


class ProtocolTests(unittest.TestCase):
    def test_pilot_requires_explicit_draft_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.jsonl"
            create_pilot(path)
            with self.assertRaisesRegex(ValueError, "Draft"):
                load_items(path)
            items = load_items(path, allow_draft=True)
            self.assertEqual(len(items), 16)
            self.assertEqual([sum(x["answer_index"] == i for x in items) for i in range(4)], [4]*4)
            with self.assertRaises(FileExistsError):
                create_pilot(path)

    def test_rotation_preserves_original_answer_and_prompt_hides_gold(self):
        item = {"context_sah": "Текст", "context_ru": "Text", "question_ru": "Вопрос?",
                "choices": ["zero", "one", "two", "three"], "answer_index": 2}
        runs = []
        for rotation in range(4):
            prompt, order = make_prompt(item, "sah", rotation)
            scores = [-10.0]*4
            scores[order.index(2)] = -0.1
            runs.append((scores, order))
            changed = {**item, "answer_index": 1}
            self.assertEqual(prompt, make_prompt(changed, "sah", rotation)[0])
        combined = combine_rotations(runs)
        self.assertEqual(max(range(4), key=combined.__getitem__), 2)

    def test_ensemble_refuses_mismatched_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"status": "complete", "benchmark_sha256": "x", "prompt_version": "v1",
                      "language": "sah", "rotations": 4, "scoring": "test", "n": 1, "draft": True}
            write_json(root / "bwen/metadata.json", {**common, "model_key": "bwen"})
            write_json(root / "eva/metadata.json", {**common, "model_key": "evabyte", "language": "ru"})
            with self.assertRaisesRegex(ValueError, "different"):
                ensemble(Namespace(bwen_run=root/"bwen", evabyte_run=root/"eva",
                                   output=root/"out", bwen_weight=0.5))

    def test_uncertainty_interval_is_not_point_estimate(self):
        result = summarize([{"category": "a", "correct": True}] * 4)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertLess(result["accuracy_wilson95"][0], 0.6)
        self.assertAlmostEqual(result["accuracy_wilson95"][1], 1.0)

    def test_ensemble_uses_both_models_and_rejects_different_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"status": "complete", "benchmark_sha256": "x", "prompt_version": "v1",
                      "language": "sah", "rotations": 1, "scoring": "test", "n": 1, "draft": True}
            base_row = {"id": "item", "category": "roles", "answer_index": 1,
                        "rotations": [{"prompt_sha256": "same"}]}
            write_json(root / "bwen/metadata.json", {**common, "model_key": "bwen"})
            write_json(root / "eva/metadata.json", {**common, "model_key": "evabyte"})
            a = {**base_row, "scores": [-0.1, -1.0, -5.0, -5.0]}
            b = {**base_row, "scores": [-4.0, -0.1, -5.0, -5.0]}
            for folder, row in (("bwen", a), ("eva", b)):
                (root / folder / "predictions.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
            args = Namespace(bwen_run=root/"bwen", evabyte_run=root/"eva",
                             output=root/"out", bwen_weight=0.5)
            ensemble(args)
            result = json.loads((root / "out/predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["prediction"], 1)
            self.assertTrue(result["correct"])
            b["rotations"] = [{"prompt_sha256": "different"}]
            (root / "eva/predictions.jsonl").write_text(json.dumps(b)+"\n", encoding="utf-8")
            args.output = root / "other-out"
            with self.assertRaisesRegex(ValueError, "Prompt hashes"):
                ensemble(args)


if __name__ == "__main__":
    unittest.main()
