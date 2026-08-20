from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .loaders import ImageAttachment, PreparedSource
from .prompt_builder import PromptBuilder


class CompletionClient(Protocol):
    def complete(
        self, prompt: str, images: tuple[ImageAttachment, ...] = ()
    ) -> str: ...


class InsightValidationError(ValueError):
    pass


class ExtractionFailure(RuntimeError):
    def __init__(self, message: str, trace: list[dict[str, Any]]):
        super().__init__(message)
        self.trace = trace


@dataclass(frozen=True)
class ExtractionSettings:
    max_source_chars: int = 300_000
    chunk_target_chars: int = 120_000
    max_images_per_request: int = 8


@dataclass(frozen=True)
class ExtractionResult:
    insights: list[dict[str, str]]
    chunked: bool
    chunk_count: int
    trace: list[dict[str, Any]]


class InsightExtractor:
    def __init__(
        self,
        client: CompletionClient,
        prompt_builder: PromptBuilder,
        settings: ExtractionSettings,
    ):
        self.client = client
        self.prompt_builder = prompt_builder
        self.settings = settings

    def extract(self, query: str, source: PreparedSource) -> ExtractionResult:
        trace: list[dict[str, Any]] = []
        try:
            chunked = source.needs_chunking(
                self.settings.max_source_chars,
                self.settings.max_images_per_request,
            )
            if not chunked:
                prompt = self.prompt_builder.build(
                    query=query,
                    source_name=source.source_name,
                    content_text=source.content_text,
                )
                insights = self._request_validated(
                    prompt=prompt,
                    images=tuple(source.attachments),
                    source_name=source.source_name,
                    stage="direct",
                    trace=trace,
                )
                return ExtractionResult(insights, False, 1, trace)

            chunks = source.chunks(
                self.settings.chunk_target_chars,
                self.settings.max_images_per_request,
            )
            candidates: list[dict[str, str]] = []
            for index, chunk in enumerate(chunks, start=1):
                chunk_content = (
                    f"[Structural chunk {index} of {len(chunks)} from the same original "
                    f"source]\n{chunk.text}"
                )
                prompt = self.prompt_builder.build(
                    query=query,
                    source_name=source.source_name,
                    content_text=chunk_content,
                )
                candidates.extend(
                    self._request_validated(
                        prompt=prompt,
                        images=chunk.attachments,
                        source_name=source.source_name,
                        stage=f"chunk-{index}",
                        trace=trace,
                    )
                )

            consolidation_prompt = _build_consolidation_prompt(
                query=query,
                source_name=source.source_name,
                candidates=candidates,
            )
            insights = self._request_validated(
                prompt=consolidation_prompt,
                images=(),
                source_name=source.source_name,
                stage="consolidation",
                trace=trace,
            )
            return ExtractionResult(insights, True, len(chunks), trace)
        except ExtractionFailure:
            raise
        except Exception as exc:
            raise ExtractionFailure(str(exc), trace) from exc

    def _request_validated(
        self,
        *,
        prompt: str,
        images: tuple[ImageAttachment, ...],
        source_name: str,
        stage: str,
        trace: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        try:
            raw = self.client.complete(prompt, images)
        except Exception as exc:
            trace.append({"stage": stage, "attempt": "initial", "api_error": str(exc)})
            raise ExtractionFailure(f"API error during {stage}: {exc}", trace) from exc

        record: dict[str, Any] = {
            "stage": stage,
            "attempt": "initial",
            "response": raw,
        }
        trace.append(record)
        try:
            return parse_and_validate_insights(raw, source_name)
        except InsightValidationError as exc:
            record["validation_error"] = str(exc)

        repair_prompt = _build_repair_prompt(raw, source_name)
        try:
            repaired = self.client.complete(repair_prompt, ())
        except Exception as exc:
            trace.append({"stage": stage, "attempt": "repair", "api_error": str(exc)})
            raise ExtractionFailure(
                f"API error during JSON repair for {stage}: {exc}", trace
            ) from exc
        repair_record: dict[str, Any] = {
            "stage": stage,
            "attempt": "repair",
            "response": repaired,
        }
        trace.append(repair_record)
        try:
            return parse_and_validate_insights(repaired, source_name)
        except InsightValidationError as exc:
            repair_record["validation_error"] = str(exc)
            raise ExtractionFailure(
                f"Malformed model JSON after one repair attempt during {stage}: {exc}",
                trace,
            ) from exc


def parse_and_validate_insights(raw: str, source_name: str) -> list[dict[str, str]]:
    cleaned = _remove_markdown_fence(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InsightValidationError(f"Response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise InsightValidationError("Response must be a JSON array")
    if len(payload) > 2:
        raise InsightValidationError(
            "Response must contain at most 2 insights per source"
        )

    validated: list[dict[str, str]] = []
    expected = Path(source_name).name
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise InsightValidationError(f"Item {index} must be a JSON object")
        insight = item.get("insight")
        source = item.get("source")
        if not isinstance(insight, str) or not insight.strip():
            raise InsightValidationError(
                f"Item {index} has an empty or missing insight"
            )
        if not isinstance(source, str) or not source.strip():
            raise InsightValidationError(f"Item {index} has an empty or missing source")
        if Path(source.strip()).name.casefold() != expected.casefold():
            raise InsightValidationError(
                f"Item {index} source must resolve to original filename {expected!r}"
            )
        validated.append({"insight": insight.strip(), "source": expected})
    return validated


def _remove_markdown_fence(raw: str) -> str:
    stripped = raw.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip() == "```":
        raise InsightValidationError("JSON markdown fence is not safely removable")
    if lines[0].strip().lower() not in {"```", "```json"}:
        raise InsightValidationError("Unexpected markdown fence language")
    return "\n".join(lines[1:-1]).strip()


def _build_repair_prompt(raw: str, source_name: str) -> str:
    return (
        "Reformat the previous response as valid JSON only. Return a JSON array with at "
        'most 2 items. Every item must contain non-empty string fields "insight" and '
        f'"source"; source must be exactly {json.dumps(source_name)}. Do not add, infer, '
        "or invent any insight. Remove markdown fences and all commentary.\n\n"
        "Previous response:\n" + raw
    )


def _build_consolidation_prompt(
    *,
    query: str,
    source_name: str,
    candidates: list[dict[str, str]],
) -> str:
    return (
        "Consolidate candidate insights extracted from structural chunks of one original "
        "source. Keep only insights relevant to the task query, remove duplicates, keep "
        "each insight atomic (1-12 English words), and retain source-specific information. "
        "Return 1-2 core insights for the original source, or fewer if specific information "
        "is lacking. Do not introduce facts absent from the candidates. Output JSON array "
        'only, using objects with "insight" and "source". The source field must be '
        f"exactly {json.dumps(source_name)}.\n\n"
        f"Task query:\n{query}\n\n"
        "Candidate insights:\n" + json.dumps(candidates, ensure_ascii=False, indent=2)
    )
