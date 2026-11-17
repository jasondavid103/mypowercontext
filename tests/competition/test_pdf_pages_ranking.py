"""赛题四检索排序补测:覆盖稀有词加权与章节标题匹配。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q tests/competition/test_pdf_pages_ranking.py
"""

from __future__ import annotations

import asyncio

from powercontext.artifacts.models import ArtifactRef
from powercontext.sources.pdf_pages import search_pdf_pages


class _FakeHit:
    """One retrieved entry of the fake scoped Memory."""

    def __init__(self, entry_id: str, text: str) -> None:
        self.entry_id = entry_id
        self.entry_version_id = f"version-{entry_id}"
        self.memory_ref = ArtifactRef(family="memory", artifact_id="memory-1", revision=1)
        self.text = text


class _FakeSource:
    """One source reference of a fake entry."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id


class _FakeEntry:
    """One stored entry of the fake scoped Memory."""

    def __init__(self, source_id: str) -> None:
        self.sources = (_FakeSource(source_id),)


class _FakeSearchPage:
    """One search result page of the fake scoped Memory."""

    def __init__(self, hits: list[_FakeHit]) -> None:
        self.hits = tuple(hits)


class _FakeEntryPage:
    """One entry lookup result of the fake scoped Memory."""

    def __init__(self, entry: _FakeEntry) -> None:
        self.entry = entry


class _FakeMemory:
    """Scoped Memory stub that serves fixed pages for every query."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.queries: list[str] = []
        self.hits = tuple(_FakeHit(entry_id, text) for entry_id, text in pages.items())

    async def search(self, request):
        self.queries.append(request.query)
        return _FakeSearchPage(list(self.hits))

    async def get(self, request):
        entry_id = request.citation.entry_id
        return _FakeEntryPage(_FakeEntry(f"source-{entry_id}"))


def _search(memory: _FakeMemory, question: str, limit: int = 20):
    citations = asyncio.run(search_pdf_pages(memory, question, limit=limit))
    return [citation.entry_id for citation in citations]


def test_question_word_expansion_finds_the_section_page() -> None:
    pages = {
        "entry-1": "CONSOLIDATED STATEMENTS OF CASH FLOWS\n2022\nNet cash provided by operating activities",
        "entry-2": (
            "CONSOLIDATED BALANCE SHEETS\n"
            "December 31 (Millions) 2022 2021\n"
            "Assets\nCash and cash equivalents 5,510\n"
            "Total assets 100,000\n"
            "Total liabilities 80,000"
        ),
        "entry-3": "ITEM 1. BUSINESS\nThe company files a report with the government every year for 2022.",
    }
    memory = _FakeMemory(pages)
    ordered = _search(memory, "Does the company have a healthy liquidity profile based on its quick ratio for 2022?")
    assert ordered[0] == "entry-2"


def test_geography_question_reaches_the_geographic_section() -> None:
    pages = {
        "entry-1": "ITEM 1. BUSINESS\nThe company files a report with the government every year for 2022.",
        "entry-2": (
            "GEOGRAPHIC OPERATIONS\nTotal revenues net of interest expense by region\nUnited States 41,396\nEMEA 4,871"
        ),
        "entry-3": "CONSOLIDATED BALANCE SHEETS\nTotal assets 100,000 for 2022",
    }
    memory = _FakeMemory(pages)
    ordered = _search(memory, "What are the geographies that the company operates in as of 2022?")
    assert ordered[0] == "entry-2"


def test_search_uses_several_query_variants_and_returns_distinct_pages() -> None:
    pages = {f"entry-{number}": f"Statement {number} of the company report for 2022." for number in range(1, 6)}
    memory = _FakeMemory(pages)
    ordered = _search(memory, "What are the revenues of the company in 2022?", limit=3)
    assert len(ordered) == 3
    assert len(set(ordered)) == 3
    assert len(memory.queries) > 1
