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

"""Coalesced embedding requests with a bounded vector cache.

Two callers that need the same text inside one merge window wait for a single
underlying request instead of asking the model twice. A successful vector is
kept in a bounded LRU cache, so a later call reuses it without any request at
all. Only successful vectors are cached: a failed batch is not remembered and
can be retried.

The shared request runs as its own task. Cancelling one caller therefore ends
that call only: the request keeps running for the other waiters, and their
vectors arrive unchanged.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile


class _Batch:
    """One underlying embedding request that several callers may wait for."""

    __slots__ = ("task", "texts", "usage", "vectors")

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.task: asyncio.Task[EmbeddingResult] | None = None
        self.usage: InferenceUsage | None = None
        self.vectors: dict[str, EmbeddingVector] = {}


class CoalescingEmbeddingModel:
    """Share one embedding request between callers that need the same text."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("embedding batch size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, (int, float)) or coalesce_ms < 0:
            raise ValueError("embedding coalescing window must not be negative")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("embedding cache size must not be negative")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._batches: dict[str, _Batch] = {}
        self._open: list[_Batch] = []

    @property
    def profile(self) -> EmbeddingProfile:
        """Expose the wrapped deployment profile unchanged."""

        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one ordered vector per input text.

        Repeated positions keep their vector, and every caller receives the
        vectors in its own input order. The reported usage covers only the
        batches this call waited for, so a cached text costs nothing.
        """

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        unique = tuple(dict.fromkeys(texts))
        vectors: dict[str, EmbeddingVector] = {}
        batches: list[_Batch] = []
        for text in unique:
            cached = self._cached_vector(text)
            if cached is not None:
                vectors[text] = cached
                continue
            batch = self._batches.get(text)
            if batch is None:
                batch = self._open_batch(text)
            if not any(batch is known for known in batches):
                batches.append(batch)
        if batches:
            outcomes = await asyncio.gather(
                *(asyncio.shield(self._task(batch)) for batch in batches),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            for batch in batches:
                vectors.update(batch.vectors)
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=_sum_usage([batch.usage for batch in batches if batch.usage is not None]),
        )

    def _open_batch(self, text: str) -> _Batch:
        """Register one text in an open batch, or open a new batch for it."""

        target: _Batch | None = None
        for candidate in self._open:
            if len(candidate.texts) < self._max_batch_size:
                target = candidate
                break
        if target is None:
            target = _Batch()
            self._open.append(target)
        target.texts.append(text)
        self._batches[text] = target
        if target.task is None:
            target.task = asyncio.create_task(self._run(target), name="powercontext-embedding-batch")
        return target

    async def _run(self, batch: _Batch) -> EmbeddingResult:
        """Send one batch to the wrapped model and publish its vectors."""

        try:
            await asyncio.sleep(self._coalesce_seconds)
            texts = tuple(batch.texts)
            self._close(batch)
            result = await self._model.embed(texts)
            _require_vector_count(result.vectors, texts)
        except BaseException:
            for text in batch.texts:
                self._batches.pop(text, None)
            self._close(batch)
            raise
        else:
            batch.usage = result.usage
            batch.vectors = dict(zip(texts, result.vectors, strict=True))
            for text, vector in batch.vectors.items():
                self._batches.pop(text, None)
                self._store(text, vector)
            return result

    def _close(self, batch: _Batch) -> None:
        """Stop accepting new texts for one dispatched batch."""

        if batch in self._open:
            self._open.remove(batch)

    @staticmethod
    def _task(batch: _Batch) -> asyncio.Task[EmbeddingResult]:
        """Return the shared request task of one batch."""

        task = batch.task
        if task is None:  # pragma: no cover - a batch always owns its task
            raise RuntimeError("embedding batch has no request")  # noqa: TRY003
        return task

    def _cached_vector(self, text: str) -> EmbeddingVector | None:
        """Return a cached vector and mark it as the most recently used."""

        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _store(self, text: str, vector: EmbeddingVector) -> None:
        """Keep one successful vector within the bounded LRU cache."""

        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _require_vector_count(vectors: tuple[EmbeddingVector, ...], texts: tuple[str, ...]) -> None:
    """Reject a batch whose vectors do not match the texts it was asked for."""

    if len(vectors) != len(texts):
        raise ValueError("embedding model returned a different number of vectors")  # noqa: TRY003


def _sum_usage(usages: list[InferenceUsage]) -> InferenceUsage:
    """Aggregate the usage of the batches one call waited for."""

    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=_sum_optional([usage.input_tokens for usage in usages]),
        output_tokens=_sum_optional([usage.output_tokens for usage in usages]),
    )


def _sum_optional(values: list[int | None]) -> int | None:
    """Sum the reported values and keep an unreported total unreported."""

    present = [value for value in values if value is not None]
    return sum(present) if present else None


__all__ = ["CoalescingEmbeddingModel"]
