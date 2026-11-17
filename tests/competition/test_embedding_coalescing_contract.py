"""赛题八契约补测:Embedding 请求合并、LRU 缓存与取消隔离。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q <本文件>
"""

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel


class RecordingEmbedding:
    profile = EmbeddingProfile(profile_id="count-contract", model="local", dimension=1)

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
        self.calls.append(texts)
        return EmbeddingResult(vectors=tuple((float(len(text)),) for text in texts), usage=InferenceUsage(requests=1))


def test_concurrent_calls_share_one_request_and_keep_order() -> None:
    async def scenario() -> None:
        base = RecordingEmbedding()
        model = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=20)
        first, second = await asyncio.gather(
            model.embed(("alpha", "beta", "alpha")),
            model.embed(("beta", "gamma")),
        )
        assert first.vectors == ((5.0,), (4.0,), (5.0,))
        assert second.vectors == ((4.0,), (5.0,))
        assert sorted(text for call in base.calls for text in call) == ["alpha", "beta", "gamma"]
        assert all(len(call) <= 2 for call in base.calls)
        assert first.usage.requests == 1
        assert second.usage.requests == 2

    asyncio.run(scenario())


def test_cache_hit_costs_no_request_and_evicts_the_least_recently_used() -> None:
    async def scenario() -> None:
        base = RecordingEmbedding()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=2)
        await model.embed(("alpha",))
        await model.embed(("beta",))
        cached = await model.embed(("alpha",))
        assert cached.usage.requests == 0
        assert cached.usage.input_tokens is None
        assert sum(call.count("alpha") for call in base.calls) == 1
        await model.embed(("gamma",))
        await model.embed(("alpha",))
        assert sum(call.count("alpha") for call in base.calls) == 1
        await model.embed(("beta",))
        assert sum(call.count("beta") for call in base.calls) == 2

    asyncio.run(scenario())


def test_cancelling_one_waiter_keeps_the_shared_request_alive() -> None:
    class Blocked(RecordingEmbedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
            self.calls.append(texts)
            self.started.set()
            await self.release.wait()
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts), usage=InferenceUsage(requests=1))

    async def scenario() -> None:
        base = Blocked()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        cancelled = asyncio.create_task(model.embed(("shared",)))
        survivor = asyncio.create_task(model.embed(("shared",)))
        await base.started.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        base.release.set()
        assert (await survivor).vectors == ((1.0,),)
        assert base.calls == [("shared",)]

    asyncio.run(scenario())


def test_failed_batch_is_not_cached_and_can_be_retried() -> None:
    class Flaky(RecordingEmbedding):
        async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
            self.calls.append(texts)
            if len(self.calls) == 1:
                raise RuntimeError("temporary")
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts), usage=InferenceUsage(requests=1))

    async def scenario() -> None:
        model = CoalescingEmbeddingModel(Flaky(), coalesce_ms=0)
        with pytest.raises(RuntimeError):
            await model.embed(("retry",))
        assert (await model.embed(("retry",))).vectors == ((1.0,),)

    asyncio.run(scenario())


def test_empty_input_and_invalid_vector_count_are_rejected() -> None:
    class Invalid(RecordingEmbedding):
        async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
            return EmbeddingResult(vectors=())

    async def scenario() -> None:
        base = RecordingEmbedding()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        empty = await model.embed(())
        assert empty.vectors == ()
        assert empty.usage.requests == 0
        assert base.calls == []
        with pytest.raises(ValueError):
            await CoalescingEmbeddingModel(Invalid(), coalesce_ms=0).embed(("broken",))
        with pytest.raises(ValueError):
            CoalescingEmbeddingModel(base, max_batch_size=0)
        with pytest.raises(ValueError):
            CoalescingEmbeddingModel(base, coalesce_ms=-1)
        with pytest.raises(ValueError):
            CoalescingEmbeddingModel(base, cache_size=-1)

    asyncio.run(scenario())
