from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.checklist_generator import (
    ChecklistGenerationFailure,
    ChecklistGenerator,
    ChecklistPromptBuilder,
    ChecklistPromptError,
)
from src.llm_client import OpenRouterClient, OpenRouterConfig
from src.utils import sha256_file, write_json_atomic, write_text_atomic

LOGGER = logging.getLogger("generate_checklists")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate candidate DR3 instruction-following checklists."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", help="Task folder name, for example 008")
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process all task folders containing query.txt",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--prompt", type=Path, default=Path("prompts/checklist_prompt.txt")
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate task queries and prompt substitution without API calls or output",
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
        prompt_builder = ChecklistPromptBuilder(args.prompt)
    except (OSError, ChecklistPromptError) as exc:
        LOGGER.error("Cannot load checklist prompt: %s", exc)
        return 2

    tasks = _select_tasks(args.data_dir, args.task, args.all)
    if not tasks:
        LOGGER.error("No task folders containing query.txt were found")
        return 2

    config = None
    client = None
    if not args.validate_only:
        try:
            config = OpenRouterConfig.from_env()
            client = OpenRouterClient(config)
        except (ValueError, RuntimeError) as exc:
            LOGGER.error("Cannot initialize OpenRouter client: %s", exc)
            return 2

    prompt_hash = sha256_file(args.prompt)
    failures = 0
    for task_dir in tasks:
        failures += _process_task(
            task_dir=task_dir,
            output_root=args.output_dir,
            prompt_builder=prompt_builder,
            prompt_hash=prompt_hash,
            client=client,
            config=config,
            validate_only=args.validate_only,
        )
    return 1 if failures else 0


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
    return path.is_dir() and (path / "query.txt").is_file()


def _process_task(
    *,
    task_dir: Path,
    output_root: Path,
    prompt_builder: ChecklistPromptBuilder,
    prompt_hash: str,
    client: OpenRouterClient | None,
    config: OpenRouterConfig | None,
    validate_only: bool,
) -> int:
    task_id = task_dir.name
    query_path = task_dir / "query.txt"
    try:
        query = query_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        LOGGER.error("Task %s: cannot read query.txt: %s", task_id, exc)
        return 1
    if not query.strip():
        LOGGER.error("Task %s: query.txt is empty", task_id)
        return 1

    if validate_only:
        prompt_builder.build(query)
        LOGGER.info(
            "Task %s: validated query-only prompt (%d query chars)", task_id, len(query)
        )
        return 0

    assert client is not None
    assert config is not None
    started = datetime.now(timezone.utc)
    output_dir = output_root / task_id
    raw_dir = output_dir / "raw"
    trace: list[dict[str, Any]] = []
    raw_files: list[str] = []
    metadata: dict[str, Any] = {
        "task_id": task_id,
        "started_at": started.isoformat(),
        "model": config.model,
        "temperature": config.temperature,
        "prompt_file": str(prompt_builder.template_path),
        "prompt_sha256": prompt_hash,
        "query_file": str(query_path),
        "query_sha256": sha256_file(query_path),
        "candidate_data_notice": (
            "Machine-generated checklist candidate; manual human review required."
        ),
    }

    try:
        result = ChecklistGenerator(client, prompt_builder).generate(query)
        trace = result.trace
        raw_files = _write_raw_responses(raw_dir, trace)
        write_json_atomic(output_dir / "checklist.json", result.payload)
        metadata.update(
            status="success",
            checklist_item_count=len(result.payload["checklist"]),
            ids_normalized=result.ids_normalized,
            raw_files=raw_files,
        )
        LOGGER.info(
            "Task %s: wrote %d checklist item(s) to %s",
            task_id,
            len(result.payload["checklist"]),
            output_dir / "checklist.json",
        )
        failures = 0
    except ChecklistGenerationFailure as exc:
        trace = exc.trace
        raw_files = _write_raw_responses(raw_dir, trace)
        metadata.update(status="failed", error=str(exc), raw_files=raw_files)
        LOGGER.error("Task %s: checklist generation failed: %s", task_id, exc)
        failures = 1
    except OSError as exc:
        metadata.update(status="failed", error=str(exc), raw_files=raw_files)
        LOGGER.error("Task %s: checklist output failed: %s", task_id, exc)
        failures = 1

    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["model_attempts"] = [
        {
            key: record[key]
            for key in ("attempt", "api_error", "validation_error")
            if key in record
        }
        for record in trace
    ]
    try:
        write_json_atomic(output_dir / "checklist_metadata.json", metadata)
    except OSError as exc:
        LOGGER.error("Task %s: cannot write checklist metadata: %s", task_id, exc)
        return 1
    return failures


def _write_raw_responses(raw_dir: Path, trace: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    for record in trace:
        response = record.get("response")
        if not isinstance(response, str):
            continue
        filename = (
            "checklist_raw.txt"
            if record.get("attempt") == "initial"
            else "checklist_repair_raw.txt"
        )
        path = raw_dir / filename
        write_text_atomic(path, response)
        written.append(str(path))
    return written


if __name__ == "__main__":
    sys.exit(main())
