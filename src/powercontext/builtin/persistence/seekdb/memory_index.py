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

"""seekdb Memory search indexes using native FULLTEXT and vector indexes.

One candidate retrieval issues exactly one seekdb statement. A hybrid retrieval
keeps the full-text and the vector candidates in that single statement, so both
native indexes stay in the database plan instead of being merged in Python.
"""

from __future__ import annotations

from pyobvector import VECTOR, VectorIndex
from sqlalchemy import (
    BigInteger,
    Column,
    MetaData,
    Table,
    UniqueConstraint,
    bindparam,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemoryCapabilities,
    MemoryProjection,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import embedding_content_hash, validate_embedding
from powercontext.builtin.persistence.memory_index import memory_channel_hits
from powercontext.builtin.persistence.tables import (
    ARTIFACT_HEADS_TABLE,
    MAX_MEMORY_ENTRY_ID_LENGTH,
    MAX_MEMORY_HASH_LENGTH,
    MEMORY_ENTRY_HEADS_TABLE,
    identity_string,
)
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql
from powercontext.limits import MAX_ARTIFACT_ID_LENGTH, MAX_SCOPE_ID_LENGTH

_SEEKDB_FTS_INDEX_NAME = "ix_pc_memory_entry_heads_fts"
_SEEKDB_CREATE_FTS_SQL = f"""
CREATE FULLTEXT INDEX {_SEEKDB_FTS_INDEX_NAME}
ON pc_memory_entry_heads (searchable_text) WITH PARSER SPACE
"""
_SEEKDB_FTS_INDEX_EXISTS_SQL = text(
    """
    SELECT COUNT(*)
    FROM information_schema.statistics
    WHERE table_schema = DATABASE()
      AND table_name = 'pc_memory_entry_heads'
      AND index_name = :index_name
    """
)
_SEEKDB_VECTOR_TABLE_NAME = "pc_memory_vector_entries"
_SEEKDB_VECTOR_INDEX_NAME = "ix_pc_memory_vector_entries_embedding"
_SEEKDB_VECTOR_TYPE_SQL = text(
    """
    SELECT data_type
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = :table_name
      AND column_name = 'embedding'
    """
)

# The full-text branch keeps the native relevance ordering of the relational
# active-head projection; the vector branch keeps the native distance ordering
# of the rebuildable embedding projection.
_SEEKDB_FTS_CANDIDATES_SQL = """
        SELECT h.memory_artifact_id AS memory_artifact_id,
               h.head_revision AS head_revision,
               h.entry_id AS entry_id,
               h.entry_version_id AS entry_version_id,
               v.text AS text,
               NULL AS distance,
               'fts' AS channel
        FROM pc_memory_entry_heads AS h
        JOIN pc_memory_entry_versions AS v
          ON v.scope_id = h.scope_id
         AND v.memory_artifact_id = h.memory_artifact_id
         AND v.entry_version_id = h.entry_version_id
        WHERE h.scope_id = :scope_id
          AND h.memory_artifact_id IN :memory_artifact_ids
          AND MATCH(h.searchable_text) AGAINST (:analyzed_query)
          {tag_filter}
        ORDER BY MATCH(h.searchable_text) AGAINST (:analyzed_query) DESC,
                 h.memory_artifact_id,
                 h.entry_id,
                 h.entry_version_id
        LIMIT :candidate_limit
"""
_SEEKDB_VECTOR_CANDIDATES_SQL = """
        SELECT m.memory_artifact_id AS memory_artifact_id,
               m.head_revision AS head_revision,
               m.entry_id AS entry_id,
               m.entry_version_id AS entry_version_id,
               v.text AS text,
               l2_distance(m.embedding, :query_vector) AS distance,
               'vector' AS channel
        FROM pc_memory_vector_entries AS m
        JOIN pc_memory_entry_versions AS v
          ON v.scope_id = m.scope_id
         AND v.memory_artifact_id = m.memory_artifact_id
         AND v.entry_version_id = m.entry_version_id
        WHERE m.scope_id = :scope_id
          AND m.memory_artifact_id IN :memory_artifact_ids
          {tag_filter}
        ORDER BY l2_distance(m.embedding, :query_vector) {approximate}
        LIMIT :candidate_limit
"""


class SeekDBMemoryIndex:
    """seekdb candidate retrieval over the native FULLTEXT and vector indexes."""

    def __init__(self, profile: EmbeddingProfile | None = None) -> None:
        if profile is not None and (
            profile.dimension < 1 or profile.distance != "l2" or profile.normalization != "unit"
        ):
            raise CapabilityNotSupportedError(
                "vector",
                "seekdb requires a positive unit-normalized L2 embedding profile",
            )
        self.profile = profile
        self.capabilities = MemoryCapabilities(
            fts=True,
            vector=profile is not None,
            hybrid=profile is not None,
            tag_filter=True,
            embedding_profile=profile,
        )
        self.tables: tuple[Table, ...] = () if profile is None else (self._vector_table(profile),)
        self._vector_index = (
            None
            if profile is None
            else VectorIndex(
                _SEEKDB_VECTOR_INDEX_NAME,
                self.tables[0].c.embedding,
                params="distance=l2,type=hnsw",
            )
        )

    @staticmethod
    def _vector_table(profile: EmbeddingProfile) -> Table:
        metadata = MetaData()
        return Table(
            _SEEKDB_VECTOR_TABLE_NAME,
            metadata,
            Column("vector_id", BigInteger, primary_key=True, autoincrement=True),
            Column("scope_id", identity_string(MAX_SCOPE_ID_LENGTH), nullable=False),
            Column("memory_artifact_id", identity_string(MAX_ARTIFACT_ID_LENGTH), nullable=False),
            Column("head_revision", BigInteger, nullable=False),
            Column("entry_id", identity_string(MAX_MEMORY_ENTRY_ID_LENGTH), nullable=False),
            Column("entry_version_id", identity_string(MAX_MEMORY_ENTRY_ID_LENGTH), nullable=False),
            Column("entry_content_hash", identity_string(MAX_MEMORY_HASH_LENGTH), nullable=False),
            Column("embedding_content_hash", identity_string(MAX_MEMORY_HASH_LENGTH), nullable=False),
            Column("embedding", VECTOR(profile.dimension), nullable=False),
            UniqueConstraint(
                "scope_id",
                "memory_artifact_id",
                "entry_id",
                name="uq_pc_memory_vector_entries_head",
            ),
        )

    async def initialize(self, connection: AsyncConnection, /) -> None:
        count = await connection.scalar(
            _SEEKDB_FTS_INDEX_EXISTS_SQL,
            {"index_name": _SEEKDB_FTS_INDEX_NAME},
        )
        if count == 0:
            await connection.exec_driver_sql(_SEEKDB_CREATE_FTS_SQL)
        if self.profile is None or self._vector_index is None:
            return
        column_type = await connection.scalar(
            _SEEKDB_VECTOR_TYPE_SQL,
            {"table_name": _SEEKDB_VECTOR_TABLE_NAME},
        )
        expected = f"VECTOR({self.profile.dimension})"
        if str(column_type).upper() != expected:
            raise CapabilityNotSupportedError(
                "vector",
                f"seekdb projection uses {column_type!r}; expected {expected}",
            )
        await connection.run_sync(lambda sync_connection: self._vector_index.create(sync_connection, checkfirst=True))

    async def replace(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        table = self._vector_table_or_none()
        if table is None:
            return
        await connection.execute(
            delete(table).where(
                table.c.scope_id == scope_id,
                table.c.memory_artifact_id == memory_ref.artifact_id,
            )
        )
        await self.upsert(connection, scope_id, memory_ref, projections)

    async def delete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        entry_ids: tuple[str, ...],
        /,
    ) -> None:
        table = self._vector_table_or_none()
        if table is None or not entry_ids:
            return
        await connection.execute(
            delete(table).where(
                table.c.scope_id == scope_id,
                table.c.memory_artifact_id == memory_ref.artifact_id,
                table.c.entry_id.in_(entry_ids),
            )
        )

    async def upsert(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memory_ref: ArtifactRef,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> None:
        table = self._vector_table_or_none()
        if table is None:
            return
        values = []
        for projection in projections:
            if projection.embedding is None or projection.embedding_content_hash is None:
                continue
            entry = projection.entry_version
            values.append({
                "scope_id": scope_id,
                "memory_artifact_id": memory_ref.artifact_id,
                "head_revision": memory_ref.revision,
                "entry_id": entry.entry_id,
                "entry_version_id": entry.entry_version_id,
                "entry_content_hash": entry.entry_content_hash,
                "embedding_content_hash": projection.embedding_content_hash,
                "embedding": validate_embedding(
                    projection.embedding,
                    dimension=self.profile.dimension if self.profile is not None else 0,
                ),
            })
        if values:
            await connection.execute(insert(table), values)

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        if request.mode == "fts":
            return MemorySearchChannels(fts=await self._fts_candidates(connection, scope_id, request))
        if request.mode == "vector":
            return MemorySearchChannels(vector=await self._vector_candidates(connection, scope_id, request))
        if self.profile is None:
            raise CapabilityNotSupportedError("hybrid")
        if request.embedding_profile != self.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self.vector_complete(connection, scope_id, request.memories, self.profile):
            raise CapabilityNotSupportedError("vector")
        # Both channels travel in one statement so the database plan keeps the
        # native full-text scan and the native vector index in one execution.
        statement = (
            f"SELECT * FROM ({self._fts_sql(request.tag_filter)}) AS fts_candidates "  # noqa: S608 - fixed clauses; values are bound.
            "UNION ALL "
            f"SELECT * FROM ({self._vector_sql(request.tag_filter, approximate=True)}) AS vector_candidates"
        )
        rows = await self._execute(connection, scope_id, request, statement, vector=True)
        fts = [row for row in rows if row["channel"] == "fts"]
        vector = [row for row in rows if row["channel"] == "vector"]
        return MemorySearchChannels(
            fts=memory_channel_hits(fts, request.memories),
            vector=memory_channel_hits(vector, request.memories),
        )

    async def _fts_candidates(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
    ) -> tuple:
        rows = await self._execute(connection, scope_id, request, self._fts_sql(request.tag_filter))
        return memory_channel_hits(rows, request.memories)

    async def _vector_candidates(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
    ) -> tuple:
        if self.profile is None:
            raise CapabilityNotSupportedError("vector")
        if request.embedding_profile != self.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self.vector_complete(connection, scope_id, request.memories, self.profile):
            raise CapabilityNotSupportedError("vector")
        rows = await self._execute(
            connection,
            scope_id,
            request,
            self._vector_sql(request.tag_filter, approximate=request.tag_filter is None),
            vector=True,
        )
        return memory_channel_hits(rows, request.memories)

    @staticmethod
    def _fts_sql(tag_filter) -> str:
        return _SEEKDB_FTS_CANDIDATES_SQL.format(
            tag_filter="" if tag_filter is None else memory_tag_sql("h"),
        )

    @staticmethod
    def _vector_sql(tag_filter, *, approximate: bool) -> str:
        return _SEEKDB_VECTOR_CANDIDATES_SQL.format(
            tag_filter="" if tag_filter is None else memory_tag_sql("m"),
            approximate="APPROXIMATE" if approximate else "",
        )

    async def _execute(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        statement: str,
        *,
        vector: bool = False,
    ) -> list:
        parameters: dict[str, object] = {
            "scope_id": scope_id,
            "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
            "analyzed_query": request.analyzed_query,
            "candidate_limit": request.candidate_limit,
            **memory_tag_parameters(request.tag_filter),
        }
        bindparams = [bindparam("memory_artifact_ids", expanding=True)]
        if vector and self.profile is not None:
            parameters["query_vector"] = validate_embedding(
                request.query_vector if request.query_vector is not None else (),
                dimension=self.profile.dimension,
            )
            bindparams.append(bindparam("query_vector", type_=VECTOR(self.profile.dimension)))
        rows = (
            await connection.execute(
                text(statement).bindparams(*bindparams),
                parameters,
            )
        ).mappings()
        return list(rows)

    async def vector_complete(
        self,
        connection: AsyncConnection,
        scope_id: str,
        memories: tuple[ArtifactRef, ...],
        profile: EmbeddingProfile,
        /,
    ) -> bool:
        table = self._vector_table_or_none()
        if table is None or profile != self.profile:
            return False
        for memory in memories:
            current = await connection.scalar(
                select(ARTIFACT_HEADS_TABLE.c.revision).where(
                    ARTIFACT_HEADS_TABLE.c.scope_id == scope_id,
                    ARTIFACT_HEADS_TABLE.c.family == "memory",
                    ARTIFACT_HEADS_TABLE.c.artifact_id == memory.artifact_id,
                )
            )
            if current is None or int(current) != memory.revision:
                # The caller validated this head before the search. If it moved while
                # this query ran, report a stale head so the runtime can retry with it.
                raise CapabilityNotSupportedError("head")
            heads = (
                await connection.execute(
                    select(
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_version_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.entry_content_hash,
                    ).where(
                        MEMORY_ENTRY_HEADS_TABLE.c.scope_id == scope_id,
                        MEMORY_ENTRY_HEADS_TABLE.c.memory_artifact_id == memory.artifact_id,
                    )
                )
            ).all()
            metadata = (
                await connection.execute(
                    select(
                        table.c.entry_id,
                        table.c.entry_version_id,
                        table.c.entry_content_hash,
                        table.c.embedding_content_hash,
                    ).where(
                        table.c.scope_id == scope_id,
                        table.c.memory_artifact_id == memory.artifact_id,
                    )
                )
            ).all()
            expected = {(str(row[0]), str(row[1]), str(row[2])) for row in heads}
            actual = {(str(row[0]), str(row[1]), str(row[2])) for row in metadata}
            if actual != expected:
                return False
            if any(str(row[3]) != _embedding_hash(self.profile, str(row[2])) for row in metadata):
                return False
        return True

    async def hydrate(
        self,
        connection: AsyncConnection,
        scope_id: str,
        projections: tuple[MemoryProjection, ...],
        /,
    ) -> tuple[MemoryProjection, ...]:
        table = self._vector_table_or_none()
        if table is None or self.profile is None:
            return projections
        hydrated: list[MemoryProjection] = []
        for projection in projections:
            entry = projection.entry_version
            row = (
                await connection.execute(
                    select(
                        table.c.entry_content_hash,
                        table.c.embedding_content_hash,
                        table.c.embedding,
                    ).where(
                        table.c.scope_id == scope_id,
                        table.c.memory_artifact_id == entry.memory_artifact_id,
                        table.c.entry_id == entry.entry_id,
                        table.c.entry_version_id == entry.entry_version_id,
                    )
                )
            ).one_or_none()
            if (
                row is None
                or str(row[0]) != entry.entry_content_hash
                or str(row[1]) != _embedding_hash(self.profile, entry.entry_content_hash)
            ):
                hydrated.append(projection)
                continue
            embedding = validate_embedding(
                tuple(float(value) for value in row[2]),
                dimension=self.profile.dimension,
            )
            hydrated.append(
                MemoryProjection(
                    entry_version=entry,
                    searchable_text=projection.searchable_text,
                    embedding=embedding,
                    embedding_content_hash=str(row[1]),
                )
            )
        return tuple(hydrated)

    def _vector_table_or_none(self) -> Table | None:
        return self.tables[0] if self.tables else None


def _embedding_hash(profile: EmbeddingProfile, entry_hash: str) -> str:
    return embedding_content_hash(
        profile_id=profile.profile_id,
        model=profile.model,
        dimension=profile.dimension,
        distance=profile.distance,
        normalization=profile.normalization,
        entry_content_hash=entry_hash,
    )
