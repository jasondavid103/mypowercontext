"""赛题十契约边界补测:选择模式下的原文区间、引用对应与预算约束。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q <本文件>
"""

from __future__ import annotations

import json

import pytest

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    PrepareContextRequest,
    SearchMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.prepared_context import PreparedContextBuilder, PreparedMemoryCandidates
from powercontext.builtin.scope import ScopeDraft
from powercontext.builtin.sources import ContentSource

PREFIX = "General administrative notes. " * 40


class LiteralPipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(kind="document", text=source.content, sources=(source,))
            for source in request.sources
            if isinstance(source, ContentSource)
        )


async def _prepare(query: str, documents: list[str], max_bytes: int, selection: bool) -> tuple[object, list[str]]:
    async with open_builtin_runtime(BuiltinConfig(), candidate_pipeline=LiteralPipeline()) as runtime:
        scope = await runtime.scopes.create(
            ScopeDraft(
                title="Selection",
                summary="Contract test",
                idempotency_key=f"selection-{max_bytes}-{selection}",
            )
        )
        sources = runtime.sources.for_scope(scope.scope_id)
        memory = runtime.memory.for_scope(scope.scope_id)
        for index, document in enumerate(documents):
            await sources.capture(
                CaptureSource(source_id=f"document-{index}", content=document, metadata={"index": index})
            )
        while (await memory.flush()).processed:
            pass
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=16))
        groups = (PreparedMemoryCandidates(scope_id=scope.scope_id, memory_ref=page.memory_ref, hits=page.hits),)
        request = PrepareContextRequest(query=query, max_bytes=max_bytes, evidence_selection=selection)
        build = PreparedContextBuilder().build_scopes_result(
            request=request, current_scope_id=scope.scope_id, memory_candidates=groups
        )
        texts = []
        for fragment in build.fragments:
            entry = await memory.get(
                GetMemoryEntryRequest(
                    citation=MemoryCitation(
                        memory_ref=fragment.address.memory.artifact,
                        entry_id=fragment.address.entry_id,
                        entry_version_id=fragment.address.entry_version_id,
                    )
                )
            )
            texts.append(entry.entry.text)
        return build, texts


@pytest.mark.asyncio
async def test_fragments_quote_the_entry_and_keep_exact_ranges() -> None:
    documents = [
        PREFIX + "The cobalt sensor is calibrated at 37 degrees. ",
        PREFIX + "The amber valve opens at 19 degrees. ",
    ]
    build, texts = await _prepare("cobalt sensor calibrated", documents, 1200, True)
    assert build.fragments
    for fragment, text in zip(build.fragments, texts, strict=True):
        assert type(fragment.start) is int
        assert type(fragment.end) is int
        assert 0 <= fragment.start < fragment.end <= len(text)
        assert fragment.text == text[fragment.start : fragment.end]


@pytest.mark.asyncio
async def test_envelope_carries_every_fragment_and_respects_the_budget() -> None:
    documents = [PREFIX + "The cobalt sensor is calibrated at 37 degrees. "]
    build, _ = await _prepare("cobalt sensor calibrated", documents, 1200, True)
    content = build.context.content
    assert build.context.content_bytes == len(content.encode("utf-8"))
    assert build.context.content_bytes <= 1200
    items = json.loads(content.splitlines()[-2])["items"]
    assert len(items) == len(build.fragments) == len(build.origins)
    for item, fragment in zip(items, build.fragments, strict=True):
        assert item["content"] == fragment.text
        assert item["start"] == fragment.start
        assert item["end"] == fragment.end
        assert item["citation"]["entry_id"] == fragment.address.entry_id
        assert item["citation"]["entry_version_id"] == fragment.address.entry_version_id


@pytest.mark.asyncio
async def test_default_prepare_keeps_the_plain_behaviour() -> None:
    documents = [PREFIX + "The cobalt sensor is calibrated at 37 degrees. "]
    build, _ = await _prepare("cobalt sensor", documents, 1200, False)
    assert not build.fragments
    items = json.loads(build.context.content.splitlines()[-2])["items"]
    assert items
    assert items[0]["content"].startswith("General administrative notes.")
