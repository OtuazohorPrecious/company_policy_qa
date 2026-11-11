"""Load policy Markdown files and split them into one chunk per top-level `## N.` section."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SECTION_HEADING = re.compile(r"^##\s+(\d+)\.\s*(.*)$")
TITLE_HEADING = re.compile(r"^#\s+(.*)$")


@dataclass(frozen=True)
class Section:
    doc_id: str          # file name without .md, e.g. "04_business_travel_policy"
    title: str           # document title, e.g. "Business Travel Policy"
    section: str         # top-level section number as a string, e.g. "4"
    heading: str         # e.g. "4. Accommodation"
    body: str            # section text below the heading (original Markdown)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.section}"

    @property
    def raw_text(self) -> str:
        """Heading plus body exactly as in the file; used for quote verification."""
        return f"## {self.heading}\n{self.body}"

    @property
    def index_text(self) -> str:
        """Text that is embedded and keyword-indexed: title in front of the section (decision A)."""
        return f"{self.title} · {self.heading}\n{self.body}"


def parse_policy(path: Path) -> list[Section]:
    doc_id = path.stem
    title = doc_id
    sections: list[Section] = []
    current_number: str | None = None
    current_heading = ""
    body_lines: list[str] = []

    def flush() -> None:
        if current_number is not None:
            sections.append(
                Section(doc_id, title, current_number, current_heading, "\n".join(body_lines).strip())
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        heading_match = SECTION_HEADING.match(line)
        if heading_match:
            flush()
            current_number = heading_match.group(1)
            current_heading = f"{heading_match.group(1)}. {heading_match.group(2).strip()}"
            body_lines = []
            continue
        if current_number is None:
            # Preamble (disclaimer, title, metadata line) is not indexed, but the title is kept.
            title_match = TITLE_HEADING.match(line)
            if title_match:
                title = title_match.group(1).strip()
            continue
        body_lines.append(line)
    flush()
    return sections


def load_sections(policies_dir: Path) -> list[Section]:
    files = sorted(policies_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No policy files found in {policies_dir}")
    sections: list[Section] = []
    for path in files:
        sections.extend(parse_policy(path))
    return sections
