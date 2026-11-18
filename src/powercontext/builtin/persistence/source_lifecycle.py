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

"""Durable evidence dependency graph for Source withdrawal propagation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.codec import dump_model, load_model
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.supervision import database_utc_now
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_NODES_TABLE,
    SOURCES_TABLE,
)
from powercontext.sources import SourceRef

SourceLifecycleKind = Literal["source", "memory", "topic"]

_SOURCE_LIFECYCLE_KINDS: frozenset[str] = frozenset({"source", "memory", "topic"})


@dataclass(frozen=True, slots=True)
class SourceLifecycleNode:
    """One registered evidence node and the availability it currently carries."""

    node_id: str
    kind: SourceLifecycleKind
    available: bool
    revision: int | None
    subject_id: str
    reference: ArtifactRef | MemoryCitation | SourceRef
    parents: tuple[str, ...]
    policy: str | None
    registered_order: int
    last_sequence: int | None
    last_reason: str | None


def require_source_lifecycle_kind(value: str) -> SourceLifecycleKind:
    """Accept only the three registered evidence kinds."""

    if value not in _SOURCE_LIFECYCLE_KINDS:
        raise TypeError(f"unsupported Source lifecycle kind: {value}")  # noqa: TRY003
    return value  # type: ignore[return-value]


async def withdrawn_topic_artifact_ids(connection: AsyncConnection, scope_id: str, /) -> frozenset[str]:
    """List the Topic artifacts this Scope withdrew from native retrieval."""

    rows = await connection.execute(
        select(SOURCE_LIFECYCLE_NODES_TABLE.c.subject_id).where(
            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
            SOURCE_LIFECYCLE_NODES_TABLE.c.kind == "topic",
            SOURCE_LIFECYCLE_NODES_TABLE.c.available.is_(False),
        )
    )
    return frozenset(str(row[0]) for row in rows)


class SourceLifecycleRepository:
    """Store lifecycle registrations, events, and derived availability per Scope."""

    def __init__(self, database: AsyncDatabase) -> None:
        self._database = database

    async def load(self, scope_id: str, /) -> tuple[SourceLifecycleNode, ...]:
        """Read every registered node of one Scope in registration order."""

        async with self._database.transaction() as connection:
            return await self._load(connection, scope_id)

    async def _load(self, connection: AsyncConnection, scope_id: str, /) -> tuple[SourceLifecycleNode, ...]:
        rows = await connection.execute(
            select(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
            .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.registered_order)
        )
        return tuple(_decode_node(row) for row in rows.mappings())

    async def register(self, scope_id: str, node: SourceLifecycleNode, /) -> None:
        """Persist one registration; an existing identity keeps its first binding."""

        async with self._database.transaction() as connection:
            created_at = await database_utc_now(connection)
            await connection.execute(
                insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                    scope_id=scope_id,
                    node_id=node.node_id,
                    kind=node.kind,
                    available=node.available,
                    revision=node.revision,
                    subject_id=node.subject_id,
                    reference=_encode_reference(node),
                    parents=json.dumps(list(node.parents)),
                    policy=node.policy,
                    registered_order=node.registered_order,
                    last_sequence=node.last_sequence,
                    last_reason=node.last_reason,
                    created_at=created_at,
                )
            )

    async def save_availability(
        self,
        scope_id: str,
        states: Mapping[str, bool],
        /,
        *,
        event_node_id: str | None = None,
        event_sequence: int | None = None,
        event_reason: str | None = None,
    ) -> None:
        """Persist derived availability and, optionally, the applied Source event."""

        async with self._database.transaction() as connection:
            for node_id, available in states.items():
                await connection.execute(
                    update(SOURCE_LIFECYCLE_NODES_TABLE)
                    .where(
                        SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                        SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                    )
                    .values(available=available)
                )
            if event_node_id is not None and event_sequence is not None:
                await connection.execute(
                    update(SOURCE_LIFECYCLE_NODES_TABLE)
                    .where(
                        SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                        SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == event_node_id,
                    )
                    .values(
                        available=states.get(event_node_id, True),
                        last_sequence=event_sequence,
                        last_reason=event_reason,
                    )
                )

    async def source_exists(self, scope_id: str, ref: SourceRef, /) -> bool:
        """Report whether one captured Source exists and stays readable."""

        async with self._database.transaction() as connection:
            row = await connection.execute(
                select(SOURCES_TABLE.c.source_id).where(
                    SOURCES_TABLE.c.scope_id == scope_id,
                    SOURCES_TABLE.c.source_type == ref.source_type,
                    SOURCES_TABLE.c.source_id == ref.source_id,
                )
            )
            return row.first() is not None


def _encode_reference(node: SourceLifecycleNode) -> bytes:
    return dump_model(node.reference, kind="source-lifecycle", name=node.node_id)


def _decode_node(row: Mapping[str, object]) -> SourceLifecycleNode:
    node_id = str(row["node_id"])
    kind = require_source_lifecycle_kind(str(row["kind"]))
    payload = row["reference"]
    if not isinstance(payload, bytes):
        raise TypeError("Source lifecycle reference payload is not bytes")  # noqa: TRY003
    return SourceLifecycleNode(
        node_id=node_id,
        kind=kind,
        available=bool(row["available"]),
        revision=None if row["revision"] is None else int(row["revision"]),
        subject_id=str(row["subject_id"]),
        reference=_decode_reference(kind, payload, node_id),
        parents=tuple(json.loads(str(row["parents"]))),
        policy=None if row["policy"] is None else str(row["policy"]),
        registered_order=int(row["registered_order"]),
        last_sequence=None if row["last_sequence"] is None else int(row["last_sequence"]),
        last_reason=None if row["last_reason"] is None else str(row["last_reason"]),
    )


def _decode_reference(
    kind: SourceLifecycleKind, payload: bytes, node_id: str
) -> ArtifactRef | MemoryCitation | SourceRef:
    if kind == "source":
        return load_model(SourceRef, payload, kind="source-lifecycle", name=node_id)
    if kind == "memory":
        return load_model(MemoryCitation, payload, kind="source-lifecycle", name=node_id)
    return load_model(ArtifactRef, payload, kind="source-lifecycle", name=node_id)
