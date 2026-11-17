# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Page-level capture for offline report PDFs."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from io import BytesIO
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

if TYPE_CHECKING:
    from powercontext.artifacts.models import MemoryCitation

PDF_PAGE_SOURCE_NAME: Final = "pdf-page"
PDF_PAGE_SOURCE_VERSION: Final = "1"
PDF_PAGE_IDENTITY_PREFIX: Final = "pdf"
PDF_PAGE_EMPTY_TEXT: Final = ""

# Retrieval tuning for the evidence-page search. The Memory admission floor rejects a
# page unless it covers a quarter of the query terms, so a long FinanceBench question
# keeps the answer page out of the candidate list. Searching several short variants
# widens the candidate pool, and the answer page is then identified by the density of
# question terms inside one short line window, which is where a quoted passage and its
# section heading live. The labelled page often holds only one or two question words
# while competing pages match many common ones, so the window terms are weighted by how
# rare they are inside the candidate pool instead of being counted. Some labelled pages
# are statement pages that carry no question word at all, so the question words are also
# expanded with the wording of the matching statement and looked for in the page
# headings.
PDF_SEARCH_POOL_LIMIT: Final = 20
PDF_SEARCH_MAX_POOL_LIMIT: Final = 50
PDF_SEARCH_TERM_LIMIT: Final = 10
PDF_SEARCH_PHRASE_LIMIT: Final = 8
PDF_SEARCH_FREQUENCY_LIMIT: Final = 200
PDF_SEARCH_HEADING_CHARS: Final = 60
PDF_SEARCH_HEADING_SPAN: Final = 3
PDF_SEARCH_HEADING_LINES: Final = 4
PDF_SEARCH_WEIGHT_WINDOW: Final = 1.0
PDF_SEARCH_WEIGHT_HEADING: Final = 0.5

# Wording of the statement or section that answers a question about it.
_PDF_SECTION_EXPANSIONS: Final = {
    "margin": ("income", "operations"),
    "margins": ("income", "operations"),
    "profit": ("income", "operations"),
    "loss": ("income", "operations"),
    "eps": ("income", "operations"),
    "tax": ("income", "operations"),
    "revenue": ("operations",),
    "revenues": ("operations",),
    "sales": ("operations",),
    "expense": ("operations",),
    "expenses": ("operations",),
    "assets": ("balance", "sheet"),
    "liabilities": ("balance", "sheet"),
    "inventory": ("balance", "sheet"),
    "receivable": ("balance", "sheet"),
    "payable": ("balance", "sheet"),
    "liquidity": ("balance", "sheet"),
    "capital": ("balance", "sheet"),
    "equity": ("balance", "sheet"),
    "debt": ("balance", "sheet"),
    "ratio": ("balance", "sheet"),
    "working": ("balance", "sheet"),
    "capex": ("cash", "flow"),
    "depreciation": ("cash", "flow"),
    "amortization": ("cash", "flow"),
    "dividend": ("cash", "flow"),
    "geographies": ("geographic", "region"),
    "geographic": ("geographic", "region"),
    "customer": ("customer",),
    "customers": ("customer",),
    "legal": ("legal", "proceeding"),
    "litigation": ("legal", "proceeding"),
    "lawsuits": ("legal", "proceeding"),
    "battles": ("legal", "proceeding"),
}

_PDF_WORD_PATTERN: Final = re.compile(r"[^\W]+", re.UNICODE)
_PDF_FUNCTION_WORDS: Final = frozenset([
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "from",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "i",
    "we",
    "you",
    "it",
    "its",
    "our",
    "your",
    "this",
    "that",
    "these",
    "those",
    "please",
])
_PDF_INSTRUCTION_MARKERS: Final = (
    "answer in",
    "answer the following question",
    "respond to",
    "give a response",
    "relying on",
    "rely on",
    "assume that",
    "assuming the perspective",
    "only use",
    "in usd",
    "show your work",
)
_PDF_SENTENCE_PATTERN: Final = re.compile(r"(?<=[.!?])\s+")


class PDFDocumentInput(BaseModel):
    """One offline PDF document that is parsed page by page."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page of an offline PDF document."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured PDF page that Memory can cite as original evidence."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable identity of one physical PDF page."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must not be empty or blank")  # noqa: TRY003
    if not isinstance(document_sha256, str) or not document_sha256.strip():
        raise ValueError("document_sha256 must not be empty or blank")  # noqa: TRY003
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive physical page number")  # noqa: TRY003
    encoded = quote(document_id, safe="")
    suffix = f":{document_sha256}:{page_number}"
    if len(PDF_PAGE_IDENTITY_PREFIX) + 1 + len(encoded) + len(suffix) > MAX_SOURCE_ID_LENGTH:
        encoded = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    source_id = f"{PDF_PAGE_IDENTITY_PREFIX}:{encoded}{suffix}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one offline PDF document."""

    document_id = value.document_id
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must not be empty or blank")  # noqa: TRY003
    content = value.content
    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    document_sha256 = hashlib.sha256(content).hexdigest()
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as error:
        raise ValueError("PDF content is not a readable document") from error  # noqa: TRY003
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as error:
            raise ValueError("PDF content is encrypted and cannot be opened") from error  # noqa: TRY003
        if not decrypted:
            raise ValueError("PDF content requires a user password")  # noqa: TRY003
    try:
        pages = list(reader.pages)
    except Exception as error:
        raise ValueError("PDF pages could not be read") from error  # noqa: TRY003
    inputs: list[PDFPageInput] = []
    for page_number, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or PDF_PAGE_EMPTY_TEXT
        except Exception as error:
            raise ValueError(f"PDF page {page_number} could not be extracted") from error  # noqa: TRY003
        inputs.append(
            PDFPageInput(
                document_id=document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                text=text,
            )
        )
    return tuple(inputs)


class PDFPageAdapter:
    """Resolve and read one captured PDF page."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one captured PDF page as standard text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageAdapter(),
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


def _analysis_terms(value: str) -> list[str]:
    """Split one text into the lowercase word terms the Memory index stores."""

    normalized = unicodedata.normalize("NFC", value).casefold()
    return _PDF_WORD_PATTERN.findall(normalized)


def _content_terms(question: str) -> tuple[str, ...]:
    """Return the distinct content terms of one question in order of appearance."""

    terms: list[str] = []
    for term in _analysis_terms(question):
        if len(term) < 2 or term in _PDF_FUNCTION_WORDS or term in terms:
            continue
        terms.append(term)
    return tuple(terms)


def _content_phrases(question: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return adjacent question word pairs whose two words are both content terms."""

    allowed = set(terms)
    words = _PDF_WORD_PATTERN.findall(question)
    phrases: list[str] = []
    for index in range(len(words) - 1):
        left = unicodedata.normalize("NFC", words[index]).casefold()
        right = unicodedata.normalize("NFC", words[index + 1]).casefold()
        if left not in allowed or right not in allowed:
            continue
        phrases.append(f"{words[index]} {words[index + 1]}")
    return tuple(phrases)


def _strip_instructions(question: str) -> str:
    """Remove the answer-format instructions a FinanceBench question may append."""

    sentences = [part for part in _PDF_SENTENCE_PATTERN.split(question) if part.strip()]
    kept = [
        sentence
        for sentence in sentences
        if not any(marker in sentence.casefold() for marker in _PDF_INSTRUCTION_MARKERS)
    ]
    return " ".join(kept).strip() or question


def _query_variants(question: str) -> tuple[tuple[str, int], ...]:
    """Build the offline query variants that widen the candidate pool.

    A single term query returns many pages and its hit count tells how common the
    term is inside the document, so those queries ask for more pages than the
    others.
    """

    terms = _content_terms(question)
    phrases = _content_phrases(question, terms)
    variants = [
        (question, PDF_SEARCH_MAX_POOL_LIMIT),
        (_strip_instructions(question), PDF_SEARCH_MAX_POOL_LIMIT),
        *[(term, PDF_SEARCH_FREQUENCY_LIMIT) for term in terms[:PDF_SEARCH_TERM_LIMIT]],
        *[(phrase, PDF_SEARCH_MAX_POOL_LIMIT) for phrase in phrases[:PDF_SEARCH_PHRASE_LIMIT]],
    ]
    return tuple(dict.fromkeys(variants))


def _page_lines(text: str) -> list[tuple[int, list[str]]]:
    """Return the length and the index terms of every line of one page."""

    return [(len(line), _analysis_terms(line)) for line in text.splitlines()]


def _window_terms(lines: list[tuple[int, list[str]]], terms: frozenset[str]) -> frozenset[str]:
    """Return the question terms that cluster inside one short line window."""

    short = [tokens for length, tokens in lines if length <= PDF_SEARCH_HEADING_CHARS]
    best: frozenset[str] = frozenset()
    for start in range(max(len(short) - PDF_SEARCH_HEADING_SPAN + 1, 1)):
        found: set[str] = set()
        for tokens in short[start : start + PDF_SEARCH_HEADING_SPAN]:
            found.update(tokens)
        matched = found & terms
        if len(matched) > len(best):
            best = matched
    return best


def _heading_terms(lines: list[tuple[int, list[str]]]) -> frozenset[str]:
    """Return the terms of the page headings, which name the statement or section."""

    found: set[str] = set()
    for _, tokens in lines[:PDF_SEARCH_HEADING_LINES]:
        found.update(tokens)
    for length, tokens in lines:
        if length <= PDF_SEARCH_HEADING_CHARS:
            found.update(tokens)
    return frozenset(found)


def _section_terms(terms: frozenset[str]) -> frozenset[str]:
    """Expand the question terms with the wording of the matching statement."""

    expanded: set[str] = set()
    for term in terms:
        expanded.update(_PDF_SECTION_EXPANSIONS.get(term, ()))
    return frozenset(expanded - terms)


def _term_frequencies(texts: dict[str, str], terms: frozenset[str], hits: dict[str, int]) -> dict[str, int]:
    """Estimate how many candidate pages carry each term."""

    frequency: dict[str, int] = {}
    for text in texts.values():
        for term in set(_analysis_terms(text)) & terms:
            frequency[term] = frequency.get(term, 0) + 1
    for term in terms:
        count = max(frequency.get(term, 0), hits.get(term, 0), 1)
        frequency[term] = count
    return frequency


def _weights(frequency: dict[str, int], total: int) -> dict[str, float]:
    """Turn the term frequencies into rarity weights."""

    weights: dict[str, float] = {}
    for term, count in frequency.items():
        weights[term] = math.log(1 + total / count) if count else 0.0
    return weights


def _score_pages(
    texts: dict[str, str],
    terms: frozenset[str],
    hits: dict[str, int],
) -> dict[str, float]:
    """Score every candidate page by the rarity of the question terms it carries."""

    total = len(texts)
    section = _section_terms(terms)
    frequency = _term_frequencies(texts, terms | section, hits)
    weights = _weights({term: frequency[term] for term in terms if term in frequency}, total)
    section_weights = _weights({term: frequency[term] for term in section if term in frequency}, total)
    scores: dict[str, float] = {}
    for entry_id, text in texts.items():
        lines = _page_lines(text)
        window = _window_terms(lines, terms)
        window_score = sum(weights.get(term, 0.0) for term in window)
        headings = _heading_terms(lines)
        heading_score = sum(section_weights.get(term, 0.0) for term in section if term in headings)
        scores[entry_id] = PDF_SEARCH_WEIGHT_WINDOW * window_score + PDF_SEARCH_WEIGHT_HEADING * heading_score
    return scores


async def _collect_hits(
    memory: Any, variants: tuple[tuple[str, int], ...], pool_limit: int
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    """Search every query variant and keep the best rank of each candidate entry."""

    from powercontext.builtin.runtime import SearchMemoryRequest

    hits: dict[str, Any] = {}
    best_rank: dict[str, int] = {}
    counts: dict[str, int] = {}
    for variant, limit in variants:
        page = await memory.search(SearchMemoryRequest(query=variant, mode="fts", limit=limit))
        counts[variant] = len(page.hits)
        for rank, hit in enumerate(page.hits[:pool_limit], start=1):
            hits.setdefault(hit.entry_id, hit)
            best_rank[hit.entry_id] = min(best_rank.get(hit.entry_id, rank), rank)
    return hits, best_rank, counts


async def _collect_citations(memory: Any, ranked: list[Any], limit: int) -> tuple[Any, ...]:
    """Resolve the ranked candidates into exact citations for distinct pages."""

    from powercontext.artifacts.models import MemoryCitation
    from powercontext.builtin.runtime import GetMemoryEntryRequest

    citations: list[Any] = []
    seen: set[str] = set()
    for hit in ranked:
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        entry = None
        try:
            entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        except Exception:
            entry = None
        if entry is None:
            continue
        references = tuple(entry.sources)
        if len(references) != 1:
            continue
        source_id = references[0].source_id
        if source_id in seen:
            continue
        seen.add(source_id)
        citations.append(citation)
        if len(citations) >= limit:
            break
    return tuple(citations)


async def search_pdf_pages(memory: Any, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Search one scoped Memory for the PDF pages that answer a question."""

    if limit <= 0:
        return ()
    question = question.strip()
    if not question:
        return ()
    terms = frozenset(_content_terms(question))
    pool_limit = min(max(limit, PDF_SEARCH_POOL_LIMIT), PDF_SEARCH_MAX_POOL_LIMIT)
    hits, best_rank, counts = await _collect_hits(memory, _query_variants(question), pool_limit)
    if not hits:
        return ()
    scores = _score_pages({entry_id: hit.text for entry_id, hit in hits.items()}, terms, counts)
    ranked = sorted(
        hits.values(),
        key=lambda hit: (
            -scores.get(hit.entry_id, 0.0),
            best_rank.get(hit.entry_id, len(hits) + 1),
            hit.entry_id,
        ),
    )
    return await _collect_citations(memory, ranked, limit)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageAdapter",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
