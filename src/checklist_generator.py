from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .utils import strip_json_markdown_fence

QUERY_PLACEHOLDER = "{query}"


class ChecklistPromptError(ValueError):
    pass


class ChecklistValidationError(ValueError):
    pass


class ChecklistGenerationFailure(RuntimeError):
    def __init__(self, message: str, trace: list[dict[str, Any]]):
        super().__init__(message)
        self.trace = trace


class TextCompletionClient(Protocol):
    def complete(self, prompt: str, images: tuple = ()) -> str: ...


@dataclass(frozen=True)
class ChecklistGenerationResult:
    payload: dict[str, list[dict[str, Any]]]
    trace: list[dict[str, Any]]
    ids_normalized: bool


class ChecklistPromptBuilder:
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.template = template_path.read_text(encoding="utf-8")
        if QUERY_PLACEHOLDER not in self.template:
            raise ChecklistPromptError(
                f"Checklist prompt must contain placeholder {QUERY_PLACEHOLDER}"
            )

    def build(self, query: str) -> str:
        return self.template.replace(QUERY_PLACEHOLDER, query)


class ChecklistGenerator:
    def __init__(
        self,
        client: TextCompletionClient,
        prompt_builder: ChecklistPromptBuilder,
    ):
        self.client = client
        self.prompt_builder = prompt_builder

    def generate(self, query: str) -> ChecklistGenerationResult:
        trace: list[dict[str, Any]] = []
        prompt = self.prompt_builder.build(query)
        try:
            raw = self.client.complete(prompt)
        except Exception as exc:
            trace.append({"attempt": "initial", "api_error": str(exc)})
            raise ChecklistGenerationFailure(
                f"Checklist API request failed: {exc}", trace
            ) from exc

        initial_record: dict[str, Any] = {"attempt": "initial", "response": raw}
        trace.append(initial_record)
        try:
            payload, ids_normalized = parse_and_validate_checklist(raw)
            return ChecklistGenerationResult(payload, trace, ids_normalized)
        except ChecklistValidationError as exc:
            initial_record["validation_error"] = str(exc)

        repair_prompt = _build_repair_prompt(raw)
        try:
            repaired = self.client.complete(repair_prompt)
        except Exception as exc:
            trace.append({"attempt": "repair", "api_error": str(exc)})
            raise ChecklistGenerationFailure(
                f"Checklist JSON repair request failed: {exc}", trace
            ) from exc

        repair_record: dict[str, Any] = {"attempt": "repair", "response": repaired}
        trace.append(repair_record)
        try:
            payload, ids_normalized = parse_and_validate_checklist(repaired)
            return ChecklistGenerationResult(payload, trace, ids_normalized)
        except ChecklistValidationError as exc:
            repair_record["validation_error"] = str(exc)
            raise ChecklistGenerationFailure(
                f"Malformed checklist JSON after one repair attempt: {exc}", trace
            ) from exc


def parse_and_validate_checklist(
    raw: str,
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    try:
        cleaned = strip_json_markdown_fence(raw)
    except ValueError as exc:
        raise ChecklistValidationError(str(exc)) from exc
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ChecklistValidationError(
            f"Response is not valid JSON: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise ChecklistValidationError("Response root must be a JSON object")
    if "checklist" not in payload:
        raise ChecklistValidationError('Response must contain key "checklist"')
    checklist = payload["checklist"]
    if not isinstance(checklist, list):
        raise ChecklistValidationError('"checklist" must be an array')
    if not checklist:
        raise ChecklistValidationError('"checklist" must not be empty')

    normalized: list[dict[str, Any]] = []
    model_ids: list[Any] = []
    for index, item in enumerate(checklist):
        if not isinstance(item, dict):
            raise ChecklistValidationError(f"Checklist item {index} must be an object")
        requirement = item.get("requirement")
        category = item.get("category")
        if not isinstance(requirement, str) or not requirement.strip():
            raise ChecklistValidationError(
                f"Checklist item {index} has an empty or missing requirement"
            )
        if not isinstance(category, str) or not category.strip():
            raise ChecklistValidationError(
                f"Checklist item {index} has an empty or missing category"
            )
        model_ids.append(item.get("id"))
        normalized.append(
            {
                "id": item.get("id"),
                "requirement": requirement,
                "category": category,
            }
        )

    expected_ids = list(range(1, len(normalized) + 1))
    ids_are_valid = all(type(item_id) is int for item_id in model_ids) and (
        model_ids == expected_ids
    )
    if not ids_are_valid:
        for new_id, item in enumerate(normalized, start=1):
            item["id"] = new_id
    return {"checklist": normalized}, not ids_are_valid


def _build_repair_prompt(raw: str) -> str:
    return (
        "Reformat the previous response as valid JSON only. The root must be an object "
        'with a non-empty "checklist" array. Every checklist item must retain its existing '
        'requirement and category and contain fields "id", "requirement", and "category". '
        "Do not add, infer, merge, split, or invent requirements. Remove markdown fences "
        "and all commentary.\n\nPrevious response:\n" + raw
    )
