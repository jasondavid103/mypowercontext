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

"""Contract tests for the Source lifecycle service beyond the published checks."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from pydantic import ValidationError

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft


def _event(node_id: str, sequence: int, available: bool, *, revision: int = 3) -> SourceLifecycleEvent:
    return SourceLifecycleEvent(
        node_id=node_id,
        revision=revision,
        sequence=sequence,
        available=available,
        reason="contract lifecycle",
    )


async def _seed(runtime, *, suffix: str):
    """Register one Source root and one Memory entry that depends on it."""

    scope = await runtime.scopes.create(
        ScopeDraft(title="Lifecycle", summary="Contract fixture", idempotency_key=suffix)
    )
    service = runtime.lifecycle.for_scope(scope.scope_id)
    receipt = await runtime.sources.for_scope(scope.scope_id).capture(
        CaptureSource(source_id="root", content="Origin root", metadata={})
    )
    await service.register_source("root", receipt.source_ref, revision=3)
    remembered = await runtime.memory.for_scope(scope.scope_id).remember(
        RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text="root fixture evidence"),))
    )
    citation = remembered.entry.citation
    await service.register_memory("leaf", citation, parents=("root",), policy="all")
    return scope.scope_id, service, citation


def test_registration_is_idempotent_and_conflicts_are_rejected(tmp_path):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
        async with open_builtin_runtime(config) as runtime:
            scope, service, citation = await _seed(runtime, suffix="idempotent")
            remembered = await runtime.memory.for_scope(scope).remember(
                RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text="other evidence"),))
            )
            other = remembered.entry.citation
            await service.register_memory("leaf", citation, parents=("root",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("leaf", other, parents=("root",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("leaf", citation, parents=("root",), policy="any")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("loop", citation, parents=("loop",), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("duplicated", citation, parents=("root", "root"), policy="all")
            with pytest.raises(SourceLifecycleError):
                await service.register_memory("unknown", citation, parents=("missing",), policy="all")

    asyncio.run(run())


def test_event_model_rejects_extra_and_blank_fields():
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="root", revision=3, sequence=1, available=False, reason="ok", extra=True)
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="  ", revision=3, sequence=1, available=False, reason="ok")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="root", revision=3, sequence=1, available=False, reason="   ")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="root", revision=0, sequence=1, available=False, reason="ok")
    with pytest.raises(ValidationError):
        SourceLifecycleEvent(node_id="root", revision=3, sequence=0, available=False, reason="ok")


def test_successful_replay_creates_no_new_memory_revision(tmp_path):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
        async with open_builtin_runtime(config) as runtime:
            scope, service, _citation = await _seed(runtime, suffix="replay")
            await service.apply(_event("root", 1, False))
            before = (await runtime.memory.for_scope(scope).list(include_inactive=True)).memory_ref
            await service.apply(_event("root", 1, False))
            after = (await runtime.memory.for_scope(scope).list(include_inactive=True)).memory_ref
            assert before == after
            with pytest.raises(SourceLifecycleError):
                await service.apply(_event("root", 1, True))
            await service.apply(_event("root", 2, True))
            with pytest.raises(SourceLifecycleError):
                await service.apply(_event("root", 1, True))
            with pytest.raises(SourceLifecycleError):
                await service.apply(_event("root", 3, True, revision=2))
            with pytest.raises(SourceLifecycleError):
                await service.apply(_event("unknown", 1, True))
            with pytest.raises(SourceLifecycleError):
                await service.resolve("leaf", purpose="unknown")
            with pytest.raises(SourceLifecycleError):
                await service.resolve("missing", purpose="audit")

    asyncio.run(run())


def test_interrupted_native_write_completes_on_retry(tmp_path):
    async def run():
        database = tmp_path / "lifecycle.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{database}"))
        async with open_builtin_runtime(config) as runtime:
            scope, service, citation = await _seed(runtime, suffix="interrupted")
            connection = await aiosqlite.connect(database)
            try:
                await connection.execute(
                    "CREATE TRIGGER block_memory_revision BEFORE INSERT ON pc_artifacts "
                    "WHEN NEW.family = 'memory' BEGIN SELECT RAISE(ABORT, 'blocked'); END"
                )
                await connection.commit()
                with pytest.raises(Exception):  # noqa: B017 - the driver error type varies
                    await service.apply(_event("root", 1, False))
                await connection.execute("DROP TRIGGER block_memory_revision")
                await connection.commit()
            finally:
                await connection.close()
            await service.apply(_event("root", 1, False))
            entries = await runtime.memory.for_scope(scope).list(include_inactive=True)
            states = {record.entry.text: record.state for record in entries.entries}
            assert states["root fixture evidence"] == "inactive"
            page = await runtime.memory.for_scope(scope).search(SearchMemoryRequest(query="fixture", mode="fts"))
            assert not any(hit.text == "root fixture evidence" for hit in page.hits)
            original = await runtime.memory.for_scope(scope).get(GetMemoryEntryRequest(citation=citation))
            assert original.entry.text == "root fixture evidence"

    asyncio.run(run())


def test_scopes_keep_the_same_node_identity_independent(tmp_path):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
        async with open_builtin_runtime(config) as runtime:
            first, first_service, _ = await _seed(runtime, suffix="scope-one")
            second, second_service, _ = await _seed(runtime, suffix="scope-two")
            await first_service.apply(_event("root", 1, False))
            first_entries = await runtime.memory.for_scope(first).list(include_inactive=True)
            second_entries = await runtime.memory.for_scope(second).list(include_inactive=True)
            assert {record.entry.text: record.state for record in first_entries.entries} == {
                "root fixture evidence": "inactive"
            }
            assert {record.entry.text: record.state for record in second_entries.entries} == {
                "root fixture evidence": "active"
            }
            await second_service.apply(_event("root", 1, False))
            assert (await second_service.apply(_event("root", 1, False))) == []

    asyncio.run(run())


def test_withdrawn_memory_leaves_generation_and_audit_purposes_split(tmp_path):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
        async with open_builtin_runtime(config) as runtime:
            _, service, citation = await _seed(runtime, suffix="purposes")
            assert await service.resolve("leaf", purpose="generation") == citation
            await service.apply(_event("root", 1, False))
            with pytest.raises(SourceLifecycleError):
                await service.resolve("leaf", purpose="generation")
            assert await service.resolve("leaf", purpose="audit") == citation

    asyncio.run(run())


def test_lifecycle_state_survives_a_restart(tmp_path):
    async def run():
        database = tmp_path / "lifecycle.db"
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{database}"))
        async with open_builtin_runtime(config) as runtime:
            scope, service, citation = await _seed(runtime, suffix="restart")
            await service.apply(_event("root", 1, False))
        async with open_builtin_runtime(config) as runtime:
            service = runtime.lifecycle.for_scope(scope)
            assert await service.resolve("leaf", purpose="audit") == citation
            with pytest.raises(SourceLifecycleError):
                await service.resolve("leaf", purpose="generation")
            assert await service.apply(_event("root", 1, False)) == []
            await service.apply(_event("root", 2, True))
            assert await service.resolve("leaf", purpose="generation") == citation

    asyncio.run(run())


def test_source_registration_requires_a_readable_source(tmp_path):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}"))
        async with open_builtin_runtime(config) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Lifecycle", summary="Missing source", idempotency_key="missing")
            )
            service = runtime.lifecycle.for_scope(scope.scope_id)
            with pytest.raises(SourceLifecycleError):
                await service.register_source("absent", _missing_ref(), revision=3)

    asyncio.run(run())


def _missing_ref():
    from powercontext.sources import SourceRef

    return SourceRef(source_type="content", source_id="absent")
