"""赛题四契约边界补测:覆盖题面 2.3 / 2.4 中公开测试未覆盖的约定。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q tests/competition/test_pdf_pages_edge_cases.py
"""

from __future__ import annotations

import asyncio
import hashlib
import io

import pytest
from pypdf import PdfWriter

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import SourceDefinitionRegistry, TextEvidence
from powercontext.sources.models import SourceMaterialization
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    PDFPageSource,
    parse_pdf_document,
    pdf_page_source_id,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _write_pdf(pages: int, metadata: dict[str, str] | None = None) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=200)
    if metadata:
        writer.add_metadata(metadata)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _parse(content: bytes, document_id: str = "doc"):
    return parse_pdf_document(PDFDocumentInput(document_id=document_id, content=content))


def test_document_id_must_not_be_empty_or_blank() -> None:
    for document_id in ("", "   ", "\t\n"):
        with pytest.raises(ValueError):
            _parse(_write_pdf(1), document_id=document_id)


def test_document_sha256_is_lowercase_hex_of_the_whole_input() -> None:
    content = _write_pdf(2)
    pages = _parse(content)
    expected = hashlib.sha256(content).hexdigest()
    assert all(page.document_sha256 == expected for page in pages)
    assert expected == expected.lower()
    assert len(expected) == 64


def test_all_physical_pages_are_returned_in_order() -> None:
    pages = _parse(_write_pdf(4))
    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    assert len(pages) == 4


def test_blank_pages_keep_empty_text() -> None:
    pages = _parse(_write_pdf(2))
    assert [page.text for page in pages] == ["", ""]


def test_definition_identity_fields() -> None:
    definition = PDF_PAGE_SOURCE_DEFINITION
    assert definition.name == "pdf-page"
    assert definition.version == "1"
    assert definition.input_class is PDFPageInput
    assert definition.source_class is PDFPageSource
    assert [(p.name, p.version) for p in definition.projections] == [
        (TEXT_EVIDENCE_PROJECTION_KEY.name, TEXT_EVIDENCE_PROJECTION_KEY.version)
    ]
    assert definition.projections[0].output_class is TextEvidence


def test_resolve_produces_captured_page_source() -> None:
    pages = _parse(_write_pdf(1))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(pages[0]))
    assert isinstance(source, PDFPageSource)
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.name == source.source_id
    assert asyncio.run(registry.read(source)) == pages[0]


def test_identity_is_stable_and_page_specific() -> None:
    content = _write_pdf(3)
    pages = _parse(content)
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    first = asyncio.run(registry.resolve(pages[0]))
    again = asyncio.run(registry.resolve(pages[0]))
    second = asyncio.run(registry.resolve(pages[1]))
    assert first.name == again.name
    assert first.name != second.name


def test_revision_changes_identity_but_keeps_the_old_page_readable() -> None:
    original = _parse(_write_pdf(1))
    revised = _parse(_write_pdf(1, metadata={"/Title": "revised"}))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    old_source = asyncio.run(registry.resolve(original[0]))
    new_source = asyncio.run(registry.resolve(revised[0]))
    assert old_source.name != new_source.name
    assert old_source.document_sha256 != new_source.document_sha256
    assert asyncio.run(registry.read(old_source)) == original[0]


def test_identity_stays_within_the_length_limit_for_long_document_ids() -> None:
    document_id = "x" * 400
    pages = _parse(_write_pdf(1), document_id=document_id)
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(pages[0]))
    assert len(source.source_id) <= MAX_SOURCE_ID_LENGTH
    assert source.document_id == document_id


def test_identity_components_cannot_collide() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert pdf_page_source_id("a:b", digest, 1) != pdf_page_source_id("a", digest, 1)
    assert pdf_page_source_id("doc", digest, 1) != pdf_page_source_id("doc", digest, 2)
    assert pdf_page_source_id("doc", digest, 1) == pdf_page_source_id("doc", digest, 1)


def test_projection_content_and_metadata_shape() -> None:
    pages = _parse(_write_pdf(1))
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(pages[0]))
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "pdf-page"
    assert evidence.source_id == source.name
    assert evidence.content == f"{pages[0].document_id}\n{pages[0].text}"
    assert evidence.metadata == {
        "document_id": pages[0].document_id,
        "document_sha256": pages[0].document_sha256,
        "page_number": pages[0].page_number,
    }


def test_corrupt_documents_fail_without_partial_pages() -> None:
    for raw in (b"", b"not a PDF", b"%PDF-1.7\ntruncated"):
        with pytest.raises(ValueError):
            _parse(raw)


def test_search_with_non_positive_limit_returns_nothing() -> None:
    assert asyncio.run(search_pdf_pages(object(), "question", limit=0)) == ()
    assert asyncio.run(search_pdf_pages(object(), "question", limit=-1)) == ()
