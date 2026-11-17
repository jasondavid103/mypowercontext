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

"""Start one Family's processing when enough new Source entries wait for it.

A time condition starts a Family's processing on a schedule, and an explicit
flush starts it on demand. A count threshold adds a third condition: once the
new Source entries behind one Family cursor reach the configured number, the
Runtime accepts the same processing call that a flush would accept.

The count follows the persisted journal identity. Every Source identity keeps
one journal position, so the number of journal rows behind the cursor is the
number of distinct entries that wait for the Family. A replayed identity keeps
its first position and therefore never counts twice, and a lineage-only Source
occupies one journal entry while staying outside the generated evidence.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.artifacts.topic_memory import TOPIC_MEMORY_SOURCE_WINDOW_BINDING
from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.processing import ArtifactProcessingPendingRepository
from powercontext.builtin.persistence.processing_intents import (
    ArtifactProcessingIntentRepository,
    StoredArtifactProcessingIntent,
)
from powercontext.builtin.persistence.tables import SOURCES_TABLE


async def pending_source_count(
    connection: AsyncConnection,
    scope_id: str,
    binding_name: str,
    /,
    *,
    after: int | None = None,
) -> int:
    """Count the Source entries that wait behind one Family cursor.

    The caller may pass the cursor position it already read. Without it the
    cursor is loaded, and a missing cursor counts from the journal start.
    """

    if after is None:
        cursor = await SourceCursorRepository().load(connection, scope_id, binding_name)
        after = 0 if cursor is None else cursor.cursor.sequence
    statement = (
        select(func.count())
        .select_from(SOURCES_TABLE)
        .where(
            SOURCES_TABLE.c.scope_id == scope_id,
            SOURCES_TABLE.c.journal_position > after,
        )
    )
    return int(await connection.scalar(statement) or 0)


async def request_count_triggered_processing(
    connection: AsyncConnection,
    scope_id: str,
    binding_name: str,
    threshold: int | None,
    /,
    *,
    after: int | None = None,
    intent: StoredArtifactProcessingIntent | None = None,
) -> bool:
    """Accept one Family's processing when its pending Source count reached the threshold.

    The request is the same durable request that an explicit flush records, so
    an accepted call survives a restart and keeps its own retry semantics. A
    Family that already holds an accepted request keeps it: the count only
    decides when the next call starts.
    """

    if threshold is None:
        return False
    intents = ArtifactProcessingIntentRepository()
    if intent is None:
        intent = await intents.load(connection, scope_id, binding_name, for_update=True)
    if intent is not None and intent.requested_generation > intent.handled_generation:
        return False
    if await pending_source_count(connection, scope_id, binding_name, after=after) < threshold:
        return False
    if binding_name == TOPIC_MEMORY_SOURCE_WINDOW_BINDING:
        # Topic Memory consumes a wave, so accept the published head as the
        # target of that wave before the Worker asks for it.
        accepted = await ArtifactProcessingPendingRepository().request_flush(connection, scope_id, binding_name)
        if accepted is None:
            return False
    await intents.request(connection, scope_id, binding_name)
    return True
