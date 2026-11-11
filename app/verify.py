"""Check that citation quotes appear word for word in the cited section.

The contract's checker ignores Markdown symbols (`**`, `#`, `-`) and extra spaces, so both the
quote and the source are normalised the same way before comparing. Matching is case-sensitive.
When a quote matches, the citation is returned with the exact source text for that span, so the
submitted quote is always copied from the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.documents import Section

_IGNORED = set("*#-")  # exactly the symbols the contract says its checker ignores


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Drop ignored symbols, collapse whitespace; return the string and a map to original indices."""
    chars: list[str] = []
    index_map: list[int] = []
    previous_space = True  # also strips leading whitespace
    for i, ch in enumerate(text):
        if ch in _IGNORED:
            continue
        if ch.isspace():
            if previous_space:
                continue
            chars.append(" ")
            index_map.append(i)
            previous_space = True
            continue
        chars.append(ch)
        index_map.append(i)
        previous_space = False
    # strip trailing space
    while chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map


def normalise(text: str) -> str:
    return _normalise_with_map(text)[0]


@dataclass(frozen=True)
class Citation:
    doc_id: str
    section: str
    quote: str


def verify_citation(raw: dict, sections_by_id: dict[str, Section]) -> Citation | None:
    """Return a verified Citation (quote replaced by exact source text), or None if invalid."""
    doc_id = str(raw.get("doc_id", "")).strip().removesuffix(".md")
    section_no = str(raw.get("section", "")).strip().rstrip(".")
    quote = str(raw.get("quote", ""))
    section = sections_by_id.get(f"{doc_id}#{section_no}")
    if section is None:
        return None
    needle = normalise(quote)
    if not needle:
        return None
    haystack, index_map = _normalise_with_map(section.raw_text)
    start = haystack.find(needle)
    if start == -1:
        return None
    end = start + len(needle) - 1
    exact = section.raw_text[index_map[start] : index_map[end] + 1].strip()
    return Citation(doc_id=doc_id, section=section_no, quote=exact)


def split_citations(
    raw_citations: list[dict], sections_by_id: dict[str, Section]
) -> tuple[list[Citation], list[dict]]:
    """Split model citations into (verified, invalid). Duplicates are removed."""
    verified: list[Citation] = []
    invalid: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_citations:
        citation = verify_citation(raw, sections_by_id)
        if citation is None:
            invalid.append(raw)
            continue
        key = (citation.doc_id, citation.section, citation.quote)
        if key not in seen:
            seen.add(key)
            verified.append(citation)
    return verified, invalid
