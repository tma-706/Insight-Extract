from __future__ import annotations

from pathlib import Path

PLACEHOLDERS = ("{query}", "{source_name}", "{content_text}")


class PromptTemplateError(ValueError):
    pass


class PromptBuilder:
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.template = template_path.read_text(encoding="utf-8")
        missing = [token for token in PLACEHOLDERS if token not in self.template]
        if missing:
            raise PromptTemplateError(
                f"Prompt template is missing placeholders: {', '.join(missing)}"
            )

    def build(self, *, query: str, source_name: str, content_text: str) -> str:
        # Plain replacement is intentional: braces inside source content must remain literal.
        prompt = self.template
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{source_name}", source_name)
        prompt = prompt.replace("{content_text}", content_text)
        return prompt
