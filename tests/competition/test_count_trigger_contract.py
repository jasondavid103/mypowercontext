"""赛题七契约补测:数量门槛的配置严格性、journal 计数口径与持久化请求。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q <本文件>
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.schema import create_tables
from powercontext.builtin.persistence.tables import (
    SHARED_TABLES,
    SOURCE_JOURNAL_HEADS_TABLE,
    SOURCES_TABLE,
)
from powercontext.builtin.runtime.config import RuntimeConfig
from powercontext.builtin.runtime.source_count_trigger import (
    pending_source_count,
    request_count_triggered_processing,
)
from powercontext.builtin.sources import SourceCursor
from powercontext.builtin.triggers import SOURCE_WINDOW_TRIGGER_NAME

MEMORY_BINDING = SOURCE_WINDOW_TRIGGER_NAME
SCOPE = "scope-count"


def test_threshold_configuration_is_strict() -> None:
    for field in ("memory_source_count_threshold", "topic_memory_source_count_threshold"):
        assert field in RuntimeConfig.model_fields
        assert getattr(RuntimeConfig(), field) is None
        for value in (None, 1, 3, 100):
            assert getattr(RuntimeConfig(**{field: value}), field) == value
        for value in (True, False, 0, -1, 1.0, "3"):
            with pytest.raises(ValidationError):
                RuntimeConfig(**{field: value})


async def _database(tmp_path, name: str = "count.db") -> AsyncDatabase:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}")
    database = AsyncDatabase.own(engine)
    async with database.transaction() as connection:
        await create_tables(connection, SHARED_TABLES)
    return database


async def _publish(database: AsyncDatabase, source_ids: tuple[str, ...]) -> None:
    async with database.transaction() as connection:
        for position, source_id in enumerate(source_ids, start=1):
            await connection.execute(
                insert(SOURCES_TABLE).values(
                    scope_id=SCOPE,
                    source_type="content",
                    source_id=source_id,
                    payload=b"{}",
                    journal_position=position,
                )
            )
        await connection.execute(insert(SOURCE_JOURNAL_HEADS_TABLE).values(scope_id=SCOPE, position=len(source_ids)))


async def _count_follows_cursor(tmp_path) -> None:
    database = await _database(tmp_path)
    try:
        await _publish(database, ("one", "two", "three"))
        async with database.transaction() as connection:
            assert await pending_source_count(connection, SCOPE, MEMORY_BINDING) == 3
            await SourceCursorRepository().save(
                connection,
                SCOPE,
                MEMORY_BINDING,
                SourceCursor(sequence=2),
                expected_generation=None,
            )
            assert await pending_source_count(connection, SCOPE, MEMORY_BINDING) == 1
    finally:
        await database.close()


async def _one_row_per_identity(tmp_path) -> None:
    database = await _database(tmp_path)
    try:
        await _publish(database, ("one", "two"))
        with pytest.raises(IntegrityError):
            async with database.transaction() as connection:
                await connection.execute(
                    insert(SOURCES_TABLE).values(
                        scope_id=SCOPE,
                        source_type="content",
                        source_id="one",
                        payload=b"{}",
                        journal_position=3,
                    )
                )
        async with database.transaction() as connection:
            assert await pending_source_count(connection, SCOPE, MEMORY_BINDING) == 2
    finally:
        await database.close()


async def _request_is_durable(tmp_path) -> None:
    database = await _database(tmp_path)
    try:
        await _publish(database, ("one", "two", "three"))
        intents = ArtifactProcessingIntentRepository()
        async with database.transaction() as connection:
            assert await request_count_triggered_processing(connection, SCOPE, MEMORY_BINDING, 4) is False
            assert await intents.load(connection, SCOPE, MEMORY_BINDING) is None
            assert await request_count_triggered_processing(connection, SCOPE, MEMORY_BINDING, 3) is True
            accepted = await intents.load(connection, SCOPE, MEMORY_BINDING)
            assert accepted is not None
            assert accepted.requested_generation == 1
            # An outstanding request is kept instead of being counted twice.
            assert await request_count_triggered_processing(connection, SCOPE, MEMORY_BINDING, 1) is False
            kept = await intents.load(connection, SCOPE, MEMORY_BINDING)
            assert kept is not None
            assert kept.requested_generation == 1
        async with database.transaction() as connection:
            # The accepted request survives the caller's transaction.
            durable = await intents.load(connection, SCOPE, MEMORY_BINDING)
            assert durable is not None
            assert durable.requested_generation == 1
    finally:
        await database.close()


def test_pending_count_follows_journal_identity_and_cursor(tmp_path) -> None:
    asyncio.run(_count_follows_cursor(tmp_path))


def test_one_journal_row_per_source_identity(tmp_path) -> None:
    asyncio.run(_one_row_per_identity(tmp_path))


def test_count_request_is_durable_and_never_duplicated(tmp_path) -> None:
    asyncio.run(_request_is_durable(tmp_path))
