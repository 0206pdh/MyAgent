import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")


def load_roots() -> list[Path]:
    raw = os.getenv("ASSISTANT_ROOTS", "").strip()
    if not raw:
        raise RuntimeError("ASSISTANT_ROOTS is not set.")
    roots: list[Path] = []
    for item in raw.split(";"):
        candidate = Path(item.strip()).expanduser()
        if candidate.exists():
            roots.append(candidate)
    if not roots:
        raise RuntimeError("No valid paths found in ASSISTANT_ROOTS.")
    return roots


@dataclass
class Match:
    path: str
    kind: str
    score: float
    modified_ts: float


def score_path(path: Path, query: str) -> float:
    q = query.lower().strip()
    name = path.name.lower()
    full = str(path).lower()
    if not q:
        return 0.0

    token_hits = sum(1 for token in q.split() if token and token in full)
    similarity = SequenceMatcher(None, q, name).ratio()
    prefix_bonus = 0.3 if name.startswith(q) else 0.0
    contains_bonus = 0.5 if q in name else 0.0
    return token_hits + similarity + prefix_bonus + contains_bonus


def search_files(query: str, limit: int = 10) -> dict[str, Any]:
    roots = load_roots()
    matches: list[Match] = []

    for root in roots:
        for path in root.rglob("*"):
            try:
                stat = path.stat()
            except OSError:
                continue

            score = score_path(path, query)
            if score < 0.65:
                continue

            matches.append(
                Match(
                    path=str(path),
                    kind="folder" if path.is_dir() else "file",
                    score=round(score, 3),
                    modified_ts=stat.st_mtime,
                )
            )

    matches.sort(key=lambda item: (item.score, item.modified_ts), reverse=True)
    top = matches[: max(1, min(limit, 20))]
    return {
        "query": query,
        "roots": [str(root) for root in roots],
        "results": [
            {
                "path": item.path,
                "kind": item.kind,
                "score": item.score,
            }
            for item in top
        ],
    }


def list_recent_files(limit: int = 10) -> dict[str, Any]:
    roots = load_roots()
    items: list[Match] = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append(
                Match(
                    path=str(path),
                    kind="file",
                    score=0.0,
                    modified_ts=stat.st_mtime,
                )
            )

    items.sort(key=lambda item: item.modified_ts, reverse=True)
    top = items[: max(1, min(limit, 20))]
    return {"results": [{"path": item.path} for item in top]}


def open_path(path: str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"ok": False, "error": "Path not found", "path": path}

    os.startfile(str(target))  # type: ignore[attr-defined]
    return {"ok": True, "opened": str(target)}


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_files",
        "description": "Search local files and folders by natural language query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_recent_files",
        "description": "List recently modified files from configured roots.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "open_path",
        "description": "Open a local file or folder in Windows.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "search_files":
        return search_files(**arguments)
    if name == "list_recent_files":
        return list_recent_files(**arguments)
    if name == "open_path":
        return open_path(**arguments)
    return {"ok": False, "error": f"Unknown tool: {name}"}


def extract_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return text
    return "응답 텍스트가 비어 있습니다."


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    print("Local Folder Assistant")
    print("종료하려면 'exit' 입력")

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        response = client.responses.create(
            model=DEFAULT_MODEL,
            instructions=(
                "You are a local Windows folder assistant. "
                "Use tools whenever you need to search or open files. "
                "Before opening a file, make sure the search result clearly matches the user's intent. "
                "If multiple results are ambiguous, ask a short clarification question."
            ),
            input=user_input,
            tools=TOOLS,
        )

        while True:
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                print(f"Assistant> {extract_text(response)}")
                break

            tool_outputs = []
            for call in calls:
                args = json.loads(call.arguments)
                result = run_tool(call.name, args)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=True),
                    }
                )

            response = client.responses.create(
                model=DEFAULT_MODEL,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=TOOLS,
            )


if __name__ == "__main__":
    main()
