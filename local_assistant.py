import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))


OPEN_HINTS = ("open", "launch", "열어", "실행")
RECENT_HINTS = ("recent", "latest", "newest", "최근", "최신", "방금")
FOLDER_HINTS = ("folder", "dir", "directory", "폴더")
COMMON_EXTENSIONS = (
    "pdf",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "ppt",
    "pptx",
    "txt",
    "md",
    "jpg",
    "jpeg",
    "png",
    "gif",
    "zip",
    "csv",
    "py",
)


@dataclass
class Match:
    path: Path
    kind: str
    score: float
    modified_ts: float


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


def normalize_query(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^\w\s.-]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(item in text for item in needles)


def extract_extension(query: str) -> str | None:
    for ext in COMMON_EXTENSIONS:
        if re.search(rf"(^|\s){re.escape(ext)}($|\s)", query):
            return ext
    return None


def strip_control_words(query: str) -> str:
    control_words = {
        "find",
        "search",
        "open",
        "show",
        "launch",
        "file",
        "folder",
        "recent",
        "latest",
        "newest",
        "찾아",
        "찾기",
        "검색",
        "열어",
        "보여",
        "파일",
        "폴더",
        "최근",
        "최신",
        "문서",
    }
    parts = [token for token in query.split() if token not in control_words]
    return " ".join(parts).strip()


def score_path(path: Path, query: str, folder_only: bool, extension: str | None) -> float:
    if folder_only and not path.is_dir():
        return 0.0
    if extension and path.is_file() and path.suffix.lower() != f".{extension}":
        return 0.0

    name = path.name.lower()
    full = str(path).lower()
    keyword_query = strip_control_words(query)
    if not keyword_query and extension:
        keyword_query = extension

    token_hits = 0
    for token in keyword_query.split():
        if token and token in full:
            token_hits += 1

    similarity = SequenceMatcher(None, keyword_query, name).ratio() if keyword_query else 0.0
    contains_bonus = 0.7 if keyword_query and keyword_query in name else 0.0
    extension_bonus = 0.6 if extension and path.suffix.lower() == f".{extension}" else 0.0
    folder_bonus = 0.25 if path.is_dir() else 0.0
    return token_hits + similarity + contains_bonus + extension_bonus + folder_bonus


def iter_paths(roots: list[Path]) -> Iterable[Path]:
    for root in roots:
        yield root
        for path in root.rglob("*"):
            yield path


def search_files(query: str, limit: int = 10) -> list[Match]:
    roots = load_roots()
    normalized = normalize_query(query)
    folder_only = contains_any(normalized, FOLDER_HINTS)
    extension = extract_extension(normalized)

    matches: list[Match] = []
    for path in iter_paths(roots):
        try:
            stat = path.stat()
        except OSError:
            continue

        score = score_path(path, normalized, folder_only, extension)
        if score < 0.65:
            continue

        matches.append(
            Match(
                path=path,
                kind="folder" if path.is_dir() else "file",
                score=round(score, 3),
                modified_ts=stat.st_mtime,
            )
        )

    matches.sort(key=lambda item: (item.score, item.modified_ts), reverse=True)
    return matches[: max(1, min(limit, 20))]


def list_recent_files(limit: int = 10, extension: str | None = None) -> list[Match]:
    roots = load_roots()
    items: list[Match] = []

    for path in iter_paths(roots):
        if not path.is_file():
            continue
        if extension and path.suffix.lower() != f".{extension}":
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            Match(
                path=path,
                kind="file",
                score=0.0,
                modified_ts=stat.st_mtime,
            )
        )

    items.sort(key=lambda item: item.modified_ts, reverse=True)
    return items[: max(1, min(limit, 20))]


def open_path(path: Path) -> None:
    os.startfile(str(path))  # type: ignore[attr-defined]


def print_matches(matches: list[Match]) -> None:
    if not matches:
        print("Assistant> 결과를 찾지 못했습니다.")
        return

    print("Assistant> 후보 결과:")
    for index, item in enumerate(matches, start=1):
        print(f"  {index}. [{item.kind}] {item.path}")


def main() -> None:
    print("Local Folder Assistant")
    print("종료하려면 'exit' 입력")

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        normalized = normalize_query(user_input)
        extension = extract_extension(normalized)

        if contains_any(normalized, RECENT_HINTS):
            matches = list_recent_files(limit=10, extension=extension)
        else:
            matches = search_files(user_input, limit=10)

        print_matches(matches)
        if not matches:
            continue

        should_open = contains_any(normalized, OPEN_HINTS)
        if should_open and len(matches) == 1:
            open_path(matches[0].path)
            print(f"Assistant> 열었습니다: {matches[0].path}")
            continue

        if should_open:
            top = matches[0]
            open_path(top.path)
            print(f"Assistant> 가장 유력한 결과를 열었습니다: {top.path}")
            continue

        print("Assistant> 번호를 입력하면 열고, Enter를 누르면 다음으로 넘어갑니다.")
        choice = input("Open> ").strip()
        if not choice:
            continue
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(matches):
                open_path(matches[index].path)
                print(f"Assistant> 열었습니다: {matches[index].path}")
            else:
                print("Assistant> 잘못된 번호입니다.")
        else:
            print("Assistant> 번호만 입력해 주세요.")


if __name__ == "__main__":
    main()
