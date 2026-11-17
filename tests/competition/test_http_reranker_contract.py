"""赛题五契约补测:HTTP 重排的批次覆盖、全局排序与响应校验。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q <本文件>
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="m", revision=1),
        entry_id=f"e{rank}",
        entry_version_id=f"v{rank}",
        text=f"candidate {rank}",
        score=1.0 / rank,
        matched_by=("fts",),
    )


def _reranker(handler, batch_size: int = 2) -> tuple[HTTPMemoryReranker, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HTTPMemoryReranker("https://reranker.invalid/score", client=client, batch_size=batch_size), client


def test_batches_cover_every_rank_once_and_report_requests() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": 0.5} for item in body["candidates"]]},
        )

    async def scenario() -> None:
        reranker, client = _reranker(handler)
        async with client:
            decision = await reranker.rerank("q", tuple(_hit(rank) for rank in range(1, 6)), 3)
        assert decision.usage.requests == 3
        assert [len(body["candidates"]) for body in seen] == [2, 2, 1]
        assert sorted(item["rank"] for body in seen for item in body["candidates"]) == [1, 2, 3, 4, 5]

    asyncio.run(scenario())


def test_global_order_breaks_ties_by_original_rank() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        scores = {1: 0.2, 2: 0.9, 3: 0.9, 4: 0.1}
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": scores[item["rank"]]} for item in body["candidates"]]},
        )

    async def scenario() -> None:
        reranker, client = _reranker(handler)
        async with client:
            decision = await reranker.rerank("q", tuple(_hit(rank) for rank in range(1, 5)), 3)
        assert decision.selected_ranks == (2, 3, 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"scores": [{"rank": 2, "score": 0.5}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.4}]},
        {"scores": [{"rank": 1, "score": float("nan")}]},
        {"scores": [{"rank": 1, "score": float("inf")}]},
        {"scores": [{"rank": 1, "score": 0.5, "extra": True}]},
        {"scores": [{"rank": 1, "score": 0.5}], "extra": True},
        {"scores": [{"rank": "1", "score": 0.5}]},
        {"scores": [{"rank": 1, "score": "0.5"}]},
    ],
)
def test_invalid_response_schema_is_rejected(payload) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client)
            with pytest.raises(ValueError):
                await reranker.rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


def test_invalid_bounds_are_rejected() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"scores": []}))
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client)
            with pytest.raises(ValueError):
                await reranker.rerank("q", (), 1)
            with pytest.raises(ValueError):
                await reranker.rerank("q", (_hit(1),), 2)
            with pytest.raises(ValueError):
                HTTPMemoryReranker("https://invalid", client=client, batch_size=0)

    asyncio.run(scenario())


def test_transport_and_status_errors_are_preserved() -> None:
    def transport_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport_error)) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client)
            with pytest.raises(httpx.ConnectError):
                await reranker.rerank("q", (_hit(1),), 1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(503))) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client)
            with pytest.raises(httpx.HTTPStatusError):
                await reranker.rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())
