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

"""HTTP scoring service adapter for Memory reranking.

A coarse retrieval pool is split into bounded batches, every batch is scored by
one HTTP request, and the scores of all batches take part in one global
selection. The original coarse rank stays the identity of a candidate: a
selected rank still points at the same Memory entry and entry version, and the
text sent to the service is the text of that very candidate.

The service answers one request with exactly ``{"scores": [{"rank", "score"}]}``.
A response that misses a rank, repeats one, names a rank of another batch, or
carries any extra field is rejected, because a partial score set would silently
change the global order.
"""

from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ScoreRecord(_StrictModel):
    """One scored candidate of one request batch."""

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)

    @field_validator("rank", mode="before")
    @classmethod
    def _require_integer_rank(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("score rank must be an integer")  # noqa: TRY003, TRY004
        return value

    @field_validator("score", mode="before")
    @classmethod
    def _require_number_score(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be a number")  # noqa: TRY003, TRY004
        return value


class _ScoreResponse(_StrictModel):
    """The exact response shape of the scoring service."""

    scores: tuple[_ScoreRecord, ...]


class HTTPMemoryReranker:
    """Select final Memory hits with an external HTTP scoring service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rerank endpoint must be a nonempty URL")  # noqa: TRY003
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("rerank batch size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("rerank policy id must be a nonempty identifier")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate and select the globally best original ranks.

        Every candidate is sent exactly once, in batches of at most
        ``batch_size``. The scores of all batches are ordered together, so a
        later batch can outrank an earlier one, and equal scores keep the
        original coarse order.
        """

        _validate_bounds(len(candidates), limit)
        scored: list[tuple[int, float]] = []
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(range(start + 1, start + 1 + len(batch)))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)
                    ],
                },
            )
            requests += 1
            response.raise_for_status()
            scored.extend(_batch_scores(response, ranks))
        ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
        selected = tuple(rank for rank, _ in ordered[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=len(candidates) - len(selected),
        )


def _batch_scores(response: httpx.Response, ranks: tuple[int, ...]) -> list[tuple[int, float]]:
    """Read one batch response and require it to cover exactly its own ranks."""

    try:
        payload = json.loads(response.content)
    except ValueError as error:
        raise ValueError("rerank response must be valid JSON") from error  # noqa: TRY003
    try:
        parsed = _ScoreResponse.model_validate(payload)
    except ValidationError as error:
        raise ValueError("rerank response does not match the scoring schema") from error  # noqa: TRY003
    observed = tuple(record.rank for record in parsed.scores)
    if sorted(observed) != sorted(ranks):
        raise ValueError("rerank response must cover every requested rank exactly once")  # noqa: TRY003
    return [(record.rank, record.score) for record in parsed.scores]


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["DEFAULT_HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
