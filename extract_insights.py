from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.extractor import ExtractionFailure, ExtractionSettings, InsightExtractor
from src.llm_client import OpenRouterClient, OpenRouterConfig
from src.loaders import SUPPORTED_EXTENSIONS, SourceLoadError, load_source
from src.prompt_builder import PromptBuilder, PromptTemplateError
from src.utils import safe_output_name, sha256_file, write_json_atomic

LOGGER = logging.getLogger("extract_insights")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate candidate DR3 user-file insights through OpenRouter."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", help="Task folder name, for example 008")
    selection.add_argument(
        "--all", action="store_true", help="Process all valid task folders"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--prompt", type=Path, default=Path("prompts/user_file_insight_prompt.txt")
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Load and inspect sources without calling the API or writing outputs",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv(override=False)

    try:
        prompt_builder = PromptBuilder(args.prompt)
    except (OSError, PromptTemplateError) as exc:
        LOGGER.error("Cannot load prompt template: %s", exc)
        return 2

    tasks = _select_tasks(args.data_dir, args.task, args.all)
    if not tasks:
        LOGGER.error("No valid task folders found")
        return 2

    client = None
    config = None
    settings = _settings_from_env()
    if not args.parse_only:
        try:
            config = OpenRouterConfig.from_env()
            client = OpenRouterClient(config)
        except (ValueError, RuntimeError) as exc:
            LOGGER.error("Cannot initialize OpenRouter client: %s", exc)
            return 2

    total_failures = 0
    for task_dir in tasks:
        failures = _process_task(
            task_dir=task_dir,
            output_root=args.output_dir,
            prompt_builder=prompt_builder,
            prompt_hash=sha256_file(args.prompt),
            settings=settings,
            client=client,
            config=config,
            parse_only=args.parse_only,
        )
        total_failures += failures
    return 1 if total_failures else 0


def _settings_from_env() -> ExtractionSettings:
    return ExtractionSettings(
        max_source_chars=int(os.getenv("MAX_SOURCE_CHARS", "300000")),
        chunk_target_chars=int(os.getenv("CHUNK_TARGET_CHARS", "120000")),
        max_images_per_request=int(os.getenv("MAX_IMAGES_PER_REQUEST", "8")),
    )


def _select_tasks(data_dir: Path, task: str | None, process_all: bool) -> list[Path]:
    if not data_dir.is_dir():
        return []
    if task is not None:
        candidate = data_dir / task
        return [candidate] if _valid_task_dir(candidate) else []
    if process_all:
        return sorted(
            (path for path in data_dir.iterdir() if _valid_task_dir(path)),
            key=lambda path: path.name,
        )
    return []


def _valid_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "query.txt").is_file()
        and (path / "user_files").is_dir()
    )


def _process_task(
    *,
    task_dir: Path,
    output_root: Path,
    prompt_builder: PromptBuilder,
    prompt_hash: str,
    settings: ExtractionSettings,
    client: OpenRouterClient | None,
    config: OpenRouterConfig | None,
    parse_only: bool,
) -> int:
    task_id = task_dir.name
    try:
        query = (task_dir / "query.txt").read_text(encoding="utf-8").strip()
    except OSError as exc:
        LOGGER.error("Task %s: cannot read query.txt: %s", task_id, exc)
        return 1
    if not query:
        LOGGER.error("Task %s: query.txt is empty", task_id)
        return 1

    source_paths = sorted(
        (path for path in (task_dir / "user_files").iterdir() if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    names = _raw_names(source_paths)
    started = datetime.now(timezone.utc)
    source_records: list[dict[str, Any]] = []
    combined: list[dict[str, str]] = []
    failures = 0

    extractor = (
        InsightExtractor(client, prompt_builder, settings)
        if client is not None
        else None
    )
    for source_path in source_paths:
        record: dict[str, Any] = {
            "source_filename": source_path.name,
            "source_type": source_path.suffix.lower().lstrip("."),
        }
        trace: list[dict[str, Any]] = []
        raw_path = output_root / task_id / "raw" / names[source_path.name]
        try:
            if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                record.update(status="skipped", error="unsupported format")
                LOGGER.warning(
                    "Task %s: skipped unsupported file %s", task_id, source_path.name
                )
                failures += 1
                continue
            prepared = load_source(source_path)
            chunked = prepared.needs_chunking(
                settings.max_source_chars, settings.max_images_per_request
            )
            chunks = (
                len(
                    prepared.chunks(
                        settings.chunk_target_chars, settings.max_images_per_request
                    )
                )
                if chunked
                else 1
            )
            record.update(
                processing_path=prepared.processing_path,
                chunked=chunked,
                chunk_count=chunks,
                loader_metadata=prepared.loader_metadata,
            )
            if parse_only:
                record["status"] = "parsed"
                LOGGER.info(
                    "Task %s: parsed %s (%s, %s, %d chars, %d image(s), chunked=%s)",
                    task_id,
                    source_path.name,
                    prepared.source_type,
                    prepared.processing_path,
                    len(prepared.content_text),
                    len(prepared.attachments),
                    chunked,
                )
                continue
            assert extractor is not None
            result = extractor.extract(query, prepared)
            trace = result.trace
            combined.extend(result.insights)
            record.update(
                status="success",
                insight_count=len(result.insights),
                raw_file=str(raw_path),
            )
            LOGGER.info(
                "Task %s: extracted %d insight(s) from %s",
                task_id,
                len(result.insights),
                source_path.name,
            )
        except ExtractionFailure as exc:
            failures += 1
            trace = exc.trace
            record.update(status="failed", error=str(exc), raw_file=str(raw_path))
            LOGGER.error(
                "Task %s: extraction failed for %s: %s", task_id, source_path.name, exc
            )
        except (SourceLoadError, OSError, ValueError) as exc:
            failures += 1
            record.update(status="failed", error=str(exc))
            LOGGER.error(
                "Task %s: parse failed for %s: %s", task_id, source_path.name, exc
            )
        finally:
            source_records.append(record)
            if not parse_only and (trace or record.get("status") == "failed"):
                write_json_atomic(
                    raw_path,
                    {
                        "task_id": task_id,
                        "source_filename": source_path.name,
                        "status": record.get("status"),
                        "error": record.get("error"),
                        "model_responses": trace,
                    },
                )

    if parse_only:
        return failures

    assert config is not None
    finished = datetime.now(timezone.utc)
    task_output = output_root / task_id
    write_json_atomic(
        task_output / "generated_insights.json", {"gold_insights": combined}
    )
    write_json_atomic(
        task_output / "run_metadata.json",
        {
            "task_id": task_id,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "model": config.model,
            "temperature": config.temperature,
            "prompt_file": str(prompt_builder.template_path),
            "prompt_sha256": prompt_hash,
            "candidate_data_notice": "Machine-generated candidates; manual verification required.",
            "settings": {
                "max_source_chars": settings.max_source_chars,
                "chunk_target_chars": settings.chunk_target_chars,
                "max_images_per_request": settings.max_images_per_request,
            },
            "sources": source_records,
            "failure_count": failures,
        },
    )
    LOGGER.info("Task %s: wrote %s", task_id, task_output / "generated_insights.json")
    return failures


def _raw_names(paths: list[Path]) -> dict[str, str]:
    stem_counts: dict[str, int] = {}
    for path in paths:
        key = path.stem.casefold()
        stem_counts[key] = stem_counts.get(key, 0) + 1
    result: dict[str, str] = {}
    for path in paths:
        base = path.stem if stem_counts[path.stem.casefold()] == 1 else path.name
        result[path.name] = safe_output_name(base) + ".json"
    return result


if __name__ == "__main__":
    sys.exit(main())
