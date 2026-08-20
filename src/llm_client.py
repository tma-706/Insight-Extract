from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

from .loaders import ImageAttachment


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen3.7-flash"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    api_max_retries: int = 2
    max_output_tokens: int = 2048
    http_referer: str | None = None
    app_title: str | None = None

    @classmethod
    def from_env(cls) -> OpenRouterConfig:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        return cls(
            api_key=api_key,
            base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).strip(),
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.7-flash").strip(),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            api_max_retries=int(os.getenv("LLM_API_MAX_RETRIES", "2")),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048")),
            http_referer=os.getenv("OPENROUTER_HTTP_REFERER") or None,
            app_title=os.getenv("OPENROUTER_APP_TITLE") or None,
        )


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for OpenRouter calls"
            ) from exc

        headers: dict[str, str] = {}
        if config.http_referer:
            headers["HTTP-Referer"] = config.http_referer
        if config.app_title:
            headers["X-Title"] = config.app_title
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.api_max_retries,
            default_headers=headers or None,
        )

    def complete(self, prompt: str, images: tuple[ImageAttachment, ...] = ()) -> str:
        content: str | list[dict[str, Any]]
        if images:
            multimodal: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for attachment in images:
                encoded = base64.b64encode(attachment.data).decode("ascii")
                multimodal.append(
                    {"type": "text", "text": f"Attached image: {attachment.label}"}
                )
                multimodal.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{attachment.mime_type};base64,{encoded}"
                        },
                    }
                )
            content = multimodal
        else:
            content = prompt

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        message_content = response.choices[0].message.content
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts: list[str] = []
            for part in message_content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif hasattr(part, "text"):
                    parts.append(str(part.text))
            return "".join(parts)
        raise RuntimeError("Model returned no textual response")
