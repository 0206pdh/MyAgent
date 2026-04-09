import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Add the project root so local_assistant/assistant imports work in GUI mode.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import local_assistant


SUPPORTED_EXTENSIONS = tuple(local_assistant.COMMON_EXTENSIONS)


@dataclass
class SearchIntent:
    query: str
    recent: bool = False
    extension: str | None = None
    folder_only: bool = False
    open_top_result: bool = False
    limit: int = 5


def _format_search_results(matches: list[local_assistant.Match]) -> str:
    if not matches:
        return "찾는 파일이나 폴더를 찾지 못했어요."

    lines = []
    for index, item in enumerate(matches[:5], start=1):
        icon = "[DIR]" if item.kind == "folder" else "[FILE]"
        lines.append(f"{index}. {icon} {item.path.name}\n   {item.path.parent}")
    return "\n".join(lines)


def _format_open_result(match: local_assistant.Match) -> str:
    return f"열었어요.\n[FILE] {match.path.name}"


def _search_from_intent(intent: SearchIntent) -> list[local_assistant.Match]:
    if intent.recent:
        return local_assistant.list_recent_files(
            limit=intent.limit,
            extension=intent.extension,
        )

    query_parts = [intent.query.strip()]
    if intent.extension and intent.extension not in intent.query.lower().split():
        query_parts.append(intent.extension)
    if intent.folder_only:
        query_parts.append("folder")

    query = " ".join(part for part in query_parts if part).strip()
    return local_assistant.search_files(query or intent.query, limit=intent.limit)


class LocalBackend:
    """Rule-based backend with no external model dependency."""

    def parse_intent(self, user_input: str) -> SearchIntent:
        normalized = local_assistant.normalize_query(user_input)
        extension = local_assistant.extract_extension(normalized)

        return SearchIntent(
            query=user_input,
            recent=local_assistant.contains_any(normalized, local_assistant.RECENT_HINTS),
            extension=extension,
            folder_only=local_assistant.contains_any(normalized, local_assistant.FOLDER_HINTS),
            open_top_result=local_assistant.contains_any(normalized, local_assistant.OPEN_HINTS),
        )

    def query(self, user_input: str) -> str:
        intent = self.parse_intent(user_input)
        matches = _search_from_intent(intent)
        if not matches:
            return "찾는 파일이나 폴더를 찾지 못했어요."

        if intent.open_top_result:
            local_assistant.open_path(matches[0].path)
            return _format_open_result(matches[0])

        return _format_search_results(matches)


class OllamaBackend:
    """Local LLM backend that parses intent, then uses deterministic file tools."""

    def __init__(self) -> None:
        self._model = os.getenv("OLLAMA_MODEL", "").strip()
        if not self._model:
            raise RuntimeError("OLLAMA_MODEL is not set.")

        self._host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self._timeout = float(os.getenv("OLLAMA_TIMEOUT", "20"))
        self._fallback = LocalBackend()

    def _prompt(self, user_input: str) -> str:
        extension_list = ", ".join(SUPPORTED_EXTENSIONS)
        return (
            "You convert a user's file-search request into JSON.\n"
            "Return JSON only. No markdown. No explanation.\n"
            'Schema: {"query": string, "recent": boolean, "extension": string|null, '
            '"folder_only": boolean, "open_top_result": boolean, "limit": integer}\n'
            "Rules:\n"
            "- Keep query concise and preserve the user's core search words.\n"
            "- Set recent=true for recent/latest/newest requests.\n"
            "- Set open_top_result=true only if the user clearly wants to open/launch the result.\n"
            "- Set folder_only=true only if the user is asking for folders/directories.\n"
            f"- extension must be one of: {extension_list}; otherwise use null.\n"
            "- limit should usually be 5.\n"
            f"User request: {user_input}"
        )

    def parse_intent(self, user_input: str) -> SearchIntent:
        payload = {
            "model": self._model,
            "prompt": self._prompt(user_input),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
            },
        }

        request = urllib.request.Request(
            f"{self._host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        content = body.get("response", "").strip()
        parsed = json.loads(content)
        extension = parsed.get("extension")
        if extension not in SUPPORTED_EXTENSIONS:
            extension = None

        limit = parsed.get("limit", 5)
        if not isinstance(limit, int):
            limit = 5

        query = str(parsed.get("query") or user_input).strip()
        if not query:
            query = user_input

        return SearchIntent(
            query=query,
            recent=bool(parsed.get("recent")),
            extension=extension,
            folder_only=bool(parsed.get("folder_only")),
            open_top_result=bool(parsed.get("open_top_result")),
            limit=max(1, min(limit, 10)),
        )

    def query(self, user_input: str) -> str:
        try:
            intent = self.parse_intent(user_input)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
            return self._fallback.query(user_input)

        matches = _search_from_intent(intent)
        if not matches:
            return "찾는 파일이나 폴더를 찾지 못했어요."

        if intent.open_top_result:
            local_assistant.open_path(matches[0].path)
            return _format_open_result(matches[0])

        return _format_search_results(matches)


class OpenAIBackend:
    """OpenAI Responses API backend."""

    def __init__(self) -> None:
        from openai import OpenAI
        import assistant

        self._client = OpenAI()
        self._assistant = assistant

    def query(self, user_input: str) -> str:
        response = self._client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            instructions=(
                "You are a friendly local Windows folder assistant. "
                "Use tools whenever you need to search or open files. "
                "Answer in Korean. Keep responses concise."
            ),
            input=user_input,
            tools=self._assistant.TOOLS,
        )

        while True:
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break

            tool_outputs = []
            for call in calls:
                args = json.loads(call.arguments)
                result = self._assistant.run_tool(call.name, args)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            response = self._client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                previous_response_id=response.id,
                input=tool_outputs,
                tools=self._assistant.TOOLS,
            )

        return self._assistant.extract_text(response) or "응답이 비어 있어요."


def create_backend():
    """Choose the best available backend for the local assistant."""
    if os.getenv("OLLAMA_MODEL"):
        try:
            return OllamaBackend()
        except Exception:
            pass

    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIBackend()
        except Exception:
            pass

    return LocalBackend()
