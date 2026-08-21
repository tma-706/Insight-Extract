from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate_checklists import _process_task, main
from src.checklist_generator import (
    ChecklistGenerator,
    ChecklistPromptBuilder,
    ChecklistValidationError,
    parse_and_validate_checklist,
)
from src.llm_client import OpenRouterConfig
from src.utils import sha256_file


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[tuple[str, tuple]] = []

    def complete(self, prompt: str, images: tuple = ()) -> str:
        self.calls.append((prompt, images))
        return next(self.responses)


class ChecklistGeneratorTests(unittest.TestCase):
    def test_prompt_substitutes_query_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.txt"
            prompt_path.write_text("Before\n{query}\nAfter", encoding="utf-8")
            query = "  Compare {A} and B.\n"
            prompt = ChecklistPromptBuilder(prompt_path).build(query)
            self.assertEqual(prompt, "Before\n  Compare {A} and B.\n\nAfter")

    def test_valid_fenced_checklist_is_parsed(self) -> None:
        raw = """```json
{"checklist":[{"id":1,"requirement":"Describe A","category":"content"}]}
```"""
        payload, normalized = parse_and_validate_checklist(raw)
        self.assertFalse(normalized)
        self.assertEqual(payload["checklist"][0]["requirement"], "Describe A")

    def test_invalid_ids_are_normalized_without_changing_content(self) -> None:
        raw = json.dumps(
            {
                "checklist": [
                    {"id": 4, "requirement": "Describe A", "category": "content"},
                    {
                        "id": 4,
                        "requirement": "Compare B",
                        "category": "comparison",
                    },
                ]
            }
        )
        payload, normalized = parse_and_validate_checklist(raw)
        self.assertTrue(normalized)
        self.assertEqual([item["id"] for item in payload["checklist"]], [1, 2])
        self.assertEqual(payload["checklist"][1]["requirement"], "Compare B")
        self.assertEqual(payload["checklist"][1]["category"], "comparison")

    def test_empty_checklist_is_rejected(self) -> None:
        with self.assertRaises(ChecklistValidationError):
            parse_and_validate_checklist('{"checklist": []}')

    def test_malformed_json_gets_one_text_only_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.txt"
            prompt_path.write_text("Query: {query}", encoding="utf-8")
            repaired = json.dumps(
                {
                    "checklist": [
                        {
                            "id": 1,
                            "requirement": "Describe A",
                            "category": "content",
                        }
                    ]
                }
            )
            client = FakeClient(["not json", repaired])
            result = ChecklistGenerator(
                client, ChecklistPromptBuilder(prompt_path)
            ).generate("Compare A")
            self.assertEqual(len(client.calls), 2)
            self.assertTrue(all(images == () for _, images in client.calls))
            self.assertEqual(result.payload["checklist"][0]["id"], 1)

    def test_validate_only_reads_query_and_writes_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "data" / "001"
            task.mkdir(parents=True)
            (task / "query.txt").write_text("Describe A", encoding="utf-8")
            prompt = root / "checklist_prompt.txt"
            prompt.write_text("Query: {query}", encoding="utf-8")
            output = root / "output"
            exit_code = main(
                [
                    "--task",
                    "001",
                    "--data-dir",
                    str(root / "data"),
                    "--output-dir",
                    str(output),
                    "--prompt",
                    str(prompt),
                    "--validate-only",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse(output.exists())

    def test_task_output_is_independent_from_insight_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "data" / "001"
            user_files = task / "user_files"
            user_files.mkdir(parents=True)
            (task / "query.txt").write_text("Describe A", encoding="utf-8")
            (user_files / "must-not-be-read.pdf").write_bytes(b"not a real PDF")

            prompt = root / "checklist_prompt.txt"
            prompt.write_text("Query: {query}", encoding="utf-8")
            prompt_builder = ChecklistPromptBuilder(prompt)
            output_task = root / "output" / "001"
            output_task.mkdir(parents=True)
            old_run_metadata = '{"insight_pipeline":"unchanged"}'
            old_insights = '{"gold_insights":[{"insight":"unchanged"}]}'
            (output_task / "run_metadata.json").write_text(
                old_run_metadata, encoding="utf-8"
            )
            (output_task / "generated_insights.json").write_text(
                old_insights, encoding="utf-8"
            )

            raw = json.dumps(
                {
                    "checklist": [
                        {
                            "id": 1,
                            "requirement": "Describe A",
                            "category": "content",
                        }
                    ]
                }
            )
            failure_count = _process_task(
                task_dir=task,
                output_root=root / "output",
                prompt_builder=prompt_builder,
                prompt_hash=sha256_file(prompt),
                client=FakeClient([raw]),
                config=OpenRouterConfig(api_key="not-used"),
                validate_only=False,
            )

            self.assertEqual(failure_count, 0)
            self.assertEqual(
                (output_task / "run_metadata.json").read_text(encoding="utf-8"),
                old_run_metadata,
            )
            self.assertEqual(
                (output_task / "generated_insights.json").read_text(encoding="utf-8"),
                old_insights,
            )
            self.assertEqual(
                (output_task / "raw" / "checklist_raw.txt").read_text(encoding="utf-8"),
                raw,
            )
            checklist = json.loads(
                (output_task / "checklist.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (output_task / "checklist_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checklist["checklist"][0]["requirement"], "Describe A")
            self.assertEqual(metadata["status"], "success")


if __name__ == "__main__":
    unittest.main()
