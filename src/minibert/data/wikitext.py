"""Convert WikiText article files into document strings."""

from __future__ import annotations

import re
from collections.abc import Iterator, Iterable

ARTICLE_TITLE_PATTERN = re.compile(r"^= (.+) =$")


def _article_title(line: str) -> str | None:
    """Return a top-level WikiText article title, if this line is one."""
    stripped = line.strip()

    if stripped.startswith("= ="):
        return None
    
    match = ARTICLE_TITLE_PATTERN.fullmatch(stripped)
    if match is None:
        return None

    return match.group(1)

def iter_wikitext_documents(lines: Iterable[str]) -> Iterator[str]:
    """Group WikiText text lines into complete article documents."""
    title: str | None = None
    body_lines: list[str] = []

    def flush() -> str | None:
        parts = [part for part in [title, *body_lines] if part]
        return " ".join(parts) if parts else None
    
    for line in lines:
        candidate_title = _article_title(line)

        if candidate_title is not None:
            document = flush()
            if document is not None:
                yield document

            title = candidate_title
            body_lines = []
            continue
        
        text = line.strip()
        if text:
            body_lines.append(text)
    
    document = flush()
    if document is not None:
        yield document