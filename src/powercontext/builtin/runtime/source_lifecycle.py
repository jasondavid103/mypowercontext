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

"""Propagate Source withdrawal through the evidence dependencies of one Scope."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.artifacts.topic_memory import TopicMemory
from powercontext.builtin.persistence.source_lifecycle import (
    SourceLifecycleNode,
    SourceLifecycleRepository,
)
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.statistics import ModelUsagePurpose
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.service import MemoryService
    from powercontext.builtin.runtime.application import BuiltinRuntime

LifecyclePurpose = Literal["generation", "audit"]

_LIFECYCLE_PURPOSES: frozenset[str] = frozenset({"generation", "audit"})
_LIFECYCLE_POLICIES: frozenset[str] = frozenset({"all", "any"})

_LIFECYCLE_MESSAGES: dict[str, str] = {
    "binding": "Lifecycle node is already bound differently",
    "citation": "Lifecycle node has no Memory citation",
    "event-type": "Lifecycle events must be SourceLifecycleEvent values",
    "identity": "Lifecycle node ids must not be blank",
    "memory-missing": "Memory entry is not readable",
    "memory-unreadable": "Memory entry is not readable",
    "no-memory": "The Scope has no Memory to deactivate",
    "node-unknown": "Lifecycle node is not registered",
    "parents": "Lifecycle node requires at least one parent",
    "parents-self": "Lifecycle node must not depend on itself",
    "parents-unique": "Lifecycle parents must be unique",
    "parents-unregistered": "Lifecycle parent is not registered",
    "policy": "Unsupported dependency policy",
    "purpose": "Unknown lifecycle purpose",
    "repository": "The Runtime has no Source lifecycle repository",
    "revision": "Source lifecycle revision must be positive",
    "revision-mismatch": "Source lifecycle revision does not match",
    "sequence": "Source event sequence is not newer",
    "sequence-conflict": "Source event sequence conflicts with a recorded event",
    "source-unknown": "Source is not registered",
    "source-unreadable": "Source is not readable",
    "topic-family": "Artifact is not a Topic Memory revision",
    "topic-unreadable": "Topic Memory revision is not readable",
    "unavailable": "Lifecycle node is not available",
}


class SourceLifecycleError(ValueError):
    """Raised when a registration or event breaks the evidence dependency contract."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        message = _LIFECYCLE_MESSAGES.get(code, code)
        super().__init__(message if detail is None else f"{message}: {detail}")


class SourceLifecycleEvent(BaseModel):
    """One availability transition of a registered Source."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_visible_text(cls, value: str, info) -> str:
        """Reject empty and blank-only identities and reasons."""

        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")  # noqa: TRY003
        return value


class SourceLifecycleService:
    """Register evidence dependencies and propagate Source availability in one Scope."""

    def __init__(
        self,
        runtime: BuiltinRuntime,
        scope_id: str,
        repository: SourceLifecycleRepository | None,
    ) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._repository = repository

    async def register_source(self, node_id: str, source_ref: SourceRef, revision: int = 1) -> None:
        """Register one captured Source as an evidence root."""

        identity = _require_identity(node_id)
        if revision < 1:
            raise SourceLifecycleError("revision", identity)
        async with (
            self._runtime._scope_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
        ):
            nodes = await self._load()
            existing = _find(nodes, identity)
            if existing is not None:
                _require_same_binding(existing, kind="source", reference=source_ref, revision=revision)
                return
            if not await self._require_readable_source(source_ref):
                raise SourceLifecycleError("source-unreadable", source_ref.source_id)
            await self._register(
                SourceLifecycleNode(
                    node_id=identity,
                    kind="source",
                    available=True,
                    revision=revision,
                    subject_id=source_ref.source_id,
                    reference=source_ref,
                    parents=(),
                    policy=None,
                    registered_order=_next_order(nodes),
                    last_sequence=None,
                    last_reason=None,
                )
            )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        parents: tuple[str, ...] = (),
        policy: str = "all",
    ) -> None:
        """Register one Memory entry and the evidence it depends on."""

        identity = _require_identity(node_id)
        parent_ids = _require_parents(identity, parents)
        selected = _require_policy(policy)
        async with (
            self._runtime._scope_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
        ):
            nodes = await self._load()
            existing = _find(nodes, identity)
            if existing is not None:
                _require_same_derived_binding(existing, parent_ids, selected, citation)
                return
            _require_registered_parents(nodes, parent_ids)
            await self._require_readable_memory(citation)
            available = _derive(nodes, parent_ids, selected)
            registered = SourceLifecycleNode(
                node_id=identity,
                kind="memory",
                available=available,
                revision=None,
                subject_id=citation.memory_ref.artifact_id,
                reference=citation,
                parents=parent_ids,
                policy=selected,
                registered_order=_next_order(nodes),
                last_sequence=None,
                last_reason=None,
            )
            await self._register(registered)
            await self._propagate_memory((*nodes, registered), {identity: available})

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        parents: tuple[str, ...] = (),
        policy: str = "all",
    ) -> None:
        """Register one Topic Memory artifact and the evidence it depends on."""

        identity = _require_identity(node_id)
        parent_ids = _require_parents(identity, parents)
        selected = _require_policy(policy)
        async with (
            self._runtime._scope_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
        ):
            nodes = await self._load()
            existing = _find(nodes, identity)
            if existing is not None:
                _require_same_derived_binding(existing, parent_ids, selected, artifact_ref)
                return
            _require_registered_parents(nodes, parent_ids)
            await self._require_readable_topic(artifact_ref)
            available = _derive(nodes, parent_ids, selected)
            await self._register(
                SourceLifecycleNode(
                    node_id=identity,
                    kind="topic",
                    available=available,
                    revision=None,
                    subject_id=artifact_ref.artifact_id,
                    reference=artifact_ref,
                    parents=parent_ids,
                    policy=selected,
                    registered_order=_next_order(nodes),
                    last_sequence=None,
                    last_reason=None,
                )
            )

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source event and return every currently available node."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event-type")
        async with (
            self._runtime._scope_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
        ):
            nodes = await self._load()
            node = _find(nodes, event.node_id)
            if node is None or node.kind != "source":
                raise SourceLifecycleError("source-unknown", event.node_id)
            if node.revision != event.revision:
                raise SourceLifecycleError("revision-mismatch", event.node_id)
            replay = _require_event_order(node, event)
            if replay:
                desired = _availability(nodes)
            else:
                desired = _availability(nodes, override=(event.node_id, event.available))
                await self._repository.save_availability(  # type: ignore[union-attr]
                    self.scope_id,
                    desired,
                    event_node_id=event.node_id,
                    event_sequence=event.sequence,
                    event_reason=event.reason,
                )
            await self._propagate_memory(nodes, desired)
            return sorted(node_id for node_id, available in desired.items() if available)

    async def resolve(
        self,
        node_id: str,
        purpose: str = "generation",
    ) -> ArtifactRef | MemoryCitation | SourceRef:
        """Return the original reference of one node for the requested purpose."""

        if purpose not in _LIFECYCLE_PURPOSES:
            raise SourceLifecycleError("purpose", purpose)
        async with self._runtime._scope_operation(self.scope_id):
            nodes = await self._load()
            node = _find(nodes, node_id)
            if node is None:
                raise SourceLifecycleError("node-unknown", node_id)
            if purpose == "generation" and not node.available:
                raise SourceLifecycleError("unavailable", node_id)
            return node.reference

    async def _load(self) -> tuple[SourceLifecycleNode, ...]:
        if self._repository is None:
            raise SourceLifecycleError("repository")
        return await self._repository.load(self.scope_id)

    async def _register(self, node: SourceLifecycleNode) -> None:
        if self._repository is None:
            raise SourceLifecycleError("repository")
        await self._repository.register(self.scope_id, node)

    async def _require_readable_source(self, source_ref: SourceRef) -> bool:
        if self._repository is None:
            raise SourceLifecycleError("repository")
        return await self._repository.source_exists(self.scope_id, source_ref)

    async def _require_readable_memory(self, citation: MemoryCitation) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except Exception as error:
            raise SourceLifecycleError("memory-unreadable", citation.entry_id) from error

    async def _require_readable_topic(self, artifact_ref: ArtifactRef) -> None:
        if artifact_ref.family != TopicMemory.family:
            raise SourceLifecycleError("topic-family", artifact_ref.family)
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                artifact_ref.family,
                artifact_ref.artifact_id,
                artifact_ref.revision,
            )
        except Exception as error:
            raise SourceLifecycleError("topic-unreadable", artifact_ref.artifact_id) from error

    async def _propagate_memory(
        self,
        nodes: tuple[SourceLifecycleNode, ...],
        desired: Mapping[str, bool],
    ) -> None:
        """Reconcile native Memory entry states with the computed availability."""

        memory_nodes = tuple(node for node in nodes if node.kind == "memory")
        if not memory_nodes:
            return
        async with self._runtime._context(
            self.scope_id,
            embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING,
        ) as context:
            service = context.artifacts.memory
            current = await _head_or_none(service, context.artifacts.memory_artifact_id)
            if current is None:
                raise SourceLifecycleError("no-memory")
            entries = {item.entry_id: item for item in await service.entries(current)}
            states = {item.entry_id: item.state for item in current.content.manifest.entries}
            for node in memory_nodes:
                citation = node.reference
                if not isinstance(citation, MemoryCitation):
                    raise SourceLifecycleError("citation", node.node_id)
                entry = entries.get(citation.entry_id)
                state = states.get(citation.entry_id)
                if entry is None or state is None:
                    raise SourceLifecycleError("memory-missing", citation.entry_id)
                target = desired.get(node.node_id, node.available)
                if target and state == "inactive":
                    current = await service.reactivate(
                        current,
                        entries=(entry,),
                        reason="source lifecycle restored",
                    )
                    states = {item.entry_id: item.state for item in current.content.manifest.entries}
                elif not target and state == "active":
                    current = await service.forget(
                        current,
                        entries=(entry,),
                        reason="source lifecycle withdrawn",
                    )
                    states = {item.entry_id: item.state for item in current.content.manifest.entries}


class LifecycleApplication:
    """Select the scoped Source lifecycle service."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> SourceLifecycleService:
        """Return the lifecycle service of one Scope."""

        return SourceLifecycleService(self._runtime, scope_id, self._runtime._lifecycle_repository)


async def _head_or_none(service: MemoryService, artifact_id: str):  # type: ignore[no-untyped-def]
    """Read one Memory head without failing on an absent Memory."""

    try:
        return await service.head(artifact_id)
    except Exception:
        return None


def _require_identity(node_id: str) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("identity")
    return node_id


def _require_policy(policy: str) -> str:
    if policy not in _LIFECYCLE_POLICIES:
        raise SourceLifecycleError("policy", policy)
    return policy


def _require_parents(node_id: str, parents: tuple[str, ...]) -> tuple[str, ...]:
    if not parents:
        raise SourceLifecycleError("parents", node_id)
    selected = tuple(parents)
    for parent in selected:
        if not isinstance(parent, str) or not parent.strip():
            raise SourceLifecycleError("identity", node_id)
    if len(set(selected)) != len(selected):
        raise SourceLifecycleError("parents-unique", node_id)
    if node_id in selected:
        raise SourceLifecycleError("parents-self", node_id)
    return selected


def _require_registered_parents(nodes: tuple[SourceLifecycleNode, ...], parents: tuple[str, ...]) -> None:
    known = {node.node_id for node in nodes}
    for parent in parents:
        if parent not in known:
            raise SourceLifecycleError("parents-unregistered", parent)


def _find(nodes: tuple[SourceLifecycleNode, ...], node_id: str) -> SourceLifecycleNode | None:
    return next((node for node in nodes if node.node_id == node_id), None)


def _next_order(nodes: tuple[SourceLifecycleNode, ...]) -> int:
    return max((node.registered_order for node in nodes), default=0) + 1


def _require_same_binding(
    node: SourceLifecycleNode,
    *,
    kind: str,
    reference: object,
    revision: int | None = None,
) -> None:
    if node.kind != kind or node.reference != reference or node.revision != revision:
        raise SourceLifecycleError("binding", node.node_id)


def _require_same_derived_binding(
    node: SourceLifecycleNode,
    parents: tuple[str, ...],
    policy: str,
    reference: object,
) -> None:
    if node.parents != parents or node.policy != policy or node.reference != reference:
        raise SourceLifecycleError("binding", node.node_id)


def _derive(
    nodes: tuple[SourceLifecycleNode, ...],
    parents: tuple[str, ...],
    policy: str,
) -> bool:
    """Compute one node's availability from the current availability of its parents."""

    available = {node.node_id: node.available for node in nodes}
    supports = [available.get(parent, False) for parent in parents]
    return all(supports) if policy == "all" else any(supports)


def _availability(
    nodes: tuple[SourceLifecycleNode, ...],
    override: tuple[str, bool] | None = None,
) -> dict[str, bool]:
    """Compute every node's availability in registration order."""

    available: dict[str, bool] = {}
    for node in nodes:
        if node.kind == "source":
            if override is not None and node.node_id == override[0]:
                available[node.node_id] = override[1]
            else:
                available[node.node_id] = node.available
            continue
        supports = [available.get(parent, False) for parent in node.parents]
        available[node.node_id] = all(supports) if node.policy == "all" else any(supports)
    return available


def _require_event_order(node: SourceLifecycleNode, event: SourceLifecycleEvent) -> bool:
    """Accept a newer event, an identical replay, and reject anything else."""

    if node.last_sequence is None:
        return False
    if event.sequence < node.last_sequence:
        raise SourceLifecycleError("sequence", node.node_id)
    if event.sequence > node.last_sequence:
        return False
    if node.last_reason != event.reason or node.available != event.available:
        raise SourceLifecycleError("sequence-conflict", node.node_id)
    return True
