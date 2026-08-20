from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.extractor import (
    ExtractionSettings,
    InsightExtractor,
    InsightValidationError,
    parse_and_validate_insights,
)
from src.loaders import load_source
from src.prompt_builder import PromptBuilder


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls: list[tuple[str, int]] = []

    def complete(self, prompt: str, images=()) -> str:
        self.calls.append((prompt, len(images)))
        return next(self.responses)


class ChunkClient:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def complete(self, prompt: str, images=()) -> str:
        self.calls.append((prompt, len(images)))
        if prompt.startswith("Consolidate candidate insights"):
            return json.dumps(
                [{"insight": "Consolidated source fact", "source": "long.txt"}]
            )
        return json.dumps([{"insight": "Chunk source fact", "source": "long.txt"}])


class PipelineTests(unittest.TestCase):
    def test_prompt_replacement_does_not_interpret_source_braces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "prompt.txt"
            template.write_text(
                "Q={query}\nS={source_name}\nC={content_text}", encoding="utf-8"
            )
            prompt = PromptBuilder(template).build(
                query="query", source_name="a.txt", content_text='{"value": 1}'
            )
            self.assertEqual(prompt, 'Q=query\nS=a.txt\nC={"value": 1}')

    def test_json_fence_is_cleaned_and_source_is_normalized(self) -> None:
        raw = '```json\n[{"insight":"Specific method", "source":"REPORT.PDF"}]\n```'
        self.assertEqual(
            parse_and_validate_insights(raw, "report.pdf"),
            [{"insight": "Specific method", "source": "report.pdf"}],
        )

    def test_wrong_source_is_rejected(self) -> None:
        with self.assertRaises(InsightValidationError):
            parse_and_validate_insights(
                '[{"insight":"Specific method", "source":"other.pdf"}]',
                "report.pdf",
            )

    def test_quoted_single_column_csv_is_unwrapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["name,value"])
                writer.writerow(["alpha,10"])
            prepared = load_source(path)
            self.assertEqual(prepared.loader_metadata["column_count"], 2)
            self.assertIn('["name", "value"]', prepared.content_text)
            self.assertIn('["alpha", "10"]', prepared.content_text)

    def test_malformed_json_gets_one_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "prompt.txt"
            template.write_text(
                "{query}\n{source_name}\n{content_text}", encoding="utf-8"
            )
            source_path = Path(directory) / "sample.txt"
            source_path.write_text("Specific source material", encoding="utf-8")
            client = FakeClient(
                [
                    "not json",
                    json.dumps(
                        [
                            {
                                "insight": "Specific source material",
                                "source": "sample.txt",
                            }
                        ]
                    ),
                ]
            )
            result = InsightExtractor(
                client,
                PromptBuilder(template),
                ExtractionSettings(),
            ).extract("query", load_source(source_path))
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(result.insights[0]["source"], "sample.txt")

    def test_long_source_candidates_are_consolidated_to_original_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "prompt.txt"
            template.write_text(
                "{query}\n{source_name}\n{content_text}", encoding="utf-8"
            )
            source_path = Path(directory) / "long.txt"
            source_path.write_text("source detail\n" * 20, encoding="utf-8")
            client = ChunkClient()
            result = InsightExtractor(
                client,
                PromptBuilder(template),
                ExtractionSettings(
                    max_source_chars=20,
                    chunk_target_chars=80,
                    max_images_per_request=8,
                ),
            ).extract("query", load_source(source_path))
            self.assertTrue(result.chunked)
            self.assertGreater(result.chunk_count, 1)
            self.assertEqual(
                result.insights,
                [{"insight": "Consolidated source fact", "source": "long.txt"}],
            )
            self.assertEqual(len(client.calls), result.chunk_count + 1)


if __name__ == "__main__":
    unittest.main()
