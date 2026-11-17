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

"""Raw Pi session event capture for one selected branch of a session tree."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME: Final = "pi-event"
PI_EVENT_SOURCE_VERSION: Final = "1"
PI_EVENT_IDENTITY_PREFIX: Final = "pi"
PI_SESSION_CONNECTOR_NAME: Final = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION: Final = "1"
PI_SESSION_FORMAT_VERSION: Final = 3
PI_CHECKPOINT_VERSION: Final = 1

# Entry types that carry a visible body. Every other entry type only links the
# session tree and therefore never produces text evidence.
_PI_VISIBLE_TYPES: Final = frozenset({"message", "compaction", "branch_summary", "custom_message"})
_PI_SUMMARY_TYPES: Final = frozenset({"compaction", "branch_summary"})


class PiSessionInput(BaseModel):
    """One raw Pi session archive together with the branch it selects."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event of a Pi session branch."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """Captured Pi session event that Memory can cite as original evidence."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SessionNode:
    """One parsed archive record with its position in the session tree."""

    event_id: str
    parent_id: str | None
    event_type: str
    record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ParsedSession:
    """One parsed session archive with the branch it selects."""

    session_id: str
    nodes: dict[str, _SessionNode]
    branch: tuple[_SessionNode, ...]
    events: tuple[PiEventInput, ...]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable identity of one Pi session event."""

    if not isinstance(session_id, str) or not session_id or session_id != session_id.strip():
        raise ValueError("session_id must be a non-empty string without surrounding whitespace")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id or event_id != event_id.strip():
        raise ValueError("event_id must be a non-empty string without surrounding whitespace")  # noqa: TRY003
    identity = f"{PI_EVENT_IDENTITY_PREFIX}:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
    return f"{PI_EVENT_IDENTITY_PREFIX}:{digest}"


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the visible events of the branch selected by one Pi session archive."""

    return _parse_session(value).events


def _parse_session(value: PiSessionInput) -> _ParsedSession:
    """Parse one archive into its tree and the visible events of the selected branch."""

    session_id, nodes, ordered = _read_nodes(value)
    branch = _branch_nodes(nodes, ordered, value.leaf_id)
    events: list[PiEventInput] = []
    for node in branch:
        event = _build_event(session_id, node)
        if event is not None:
            events.append(event)
    return _ParsedSession(session_id=session_id, nodes=nodes, branch=branch, events=tuple(events))


def _read_nodes(value: PiSessionInput) -> tuple[str, dict[str, _SessionNode], tuple[_SessionNode, ...]]:
    """Read every record of one archive and keep the order that links the tree."""

    lines = _decode_lines(value.content)
    if not lines:
        raise ValueError("Pi session content must contain a session header")  # noqa: TRY003
    records = [_decode_record(line) for line in lines]
    session_id = _read_header(records[0])
    nodes: dict[str, _SessionNode] = {}
    ordered: list[_SessionNode] = []
    for record in records[1:]:
        node = _read_node(record, nodes)
        nodes[node.event_id] = node
        ordered.append(node)
    return session_id, nodes, tuple(ordered)


def _decode_lines(content: bytes) -> list[str]:
    """Decode one archive into its non-empty text lines."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error  # noqa: TRY003
    return [line for line in text.splitlines() if line.strip()]


def _decode_record(line: str) -> dict[str, Any]:
    """Decode one archive line into a JSON object."""

    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("Pi session content must contain valid JSON lines") from error  # noqa: TRY003
    if not isinstance(record, dict):
        raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003, TRY004
    return record


def _read_header(record: dict[str, Any]) -> str:
    """Validate the session header and return the session identity."""

    if record.get("type") != "session":
        raise ValueError("The first Pi session record must be the session header")  # noqa: TRY003
    if record.get("version") != PI_SESSION_FORMAT_VERSION:
        raise ValueError("Pi session version must be 3")  # noqa: TRY003
    session_id = record.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must carry a non-empty id")  # noqa: TRY003
    return session_id


def _read_node(record: dict[str, Any], seen: Mapping[str, _SessionNode]) -> _SessionNode:
    """Validate one archive record against the nodes that already appeared."""

    event_id = record.get("id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("Pi session records must carry a non-empty id")  # noqa: TRY003
    if event_id in seen:
        raise ValueError(f"Pi session records must use unique ids: {event_id!r}")  # noqa: TRY003
    parent_id = record.get("parentId")
    if parent_id is None or parent_id == "":
        parent = None
    elif isinstance(parent_id, str) and parent_id in seen:
        parent = parent_id
    else:
        raise ValueError(f"Pi session record {event_id!r} must reference an earlier parent")  # noqa: TRY003
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("Pi session records must carry a non-empty type")  # noqa: TRY003
    return _SessionNode(event_id=event_id, parent_id=parent, event_type=event_type, record=record)


def _branch_nodes(
    nodes: Mapping[str, _SessionNode],
    ordered: tuple[_SessionNode, ...],
    leaf_id: str | None,
) -> tuple[_SessionNode, ...]:
    """Return the ancestors of the selected leaf from the root down to the leaf."""

    if leaf_id is None:
        if not ordered:
            raise ValueError("Pi session content must contain at least one event")  # noqa: TRY003
        leaf = ordered[-1]
    else:
        candidate = nodes.get(leaf_id)
        if candidate is None:
            raise ValueError(f"Pi session does not contain the selected leaf: {leaf_id!r}")  # noqa: TRY003
        leaf = candidate
    chain: list[_SessionNode] = []
    visited: set[str] = set()
    current: _SessionNode | None = leaf
    while current is not None:
        if current.event_id in visited:
            raise ValueError("Pi session records must not contain a cycle")  # noqa: TRY003
        visited.add(current.event_id)
        chain.append(current)
        current = nodes.get(current.parent_id) if current.parent_id else None
    chain.reverse()
    return tuple(chain)


def _build_event(session_id: str, node: _SessionNode) -> PiEventInput | None:
    """Build the visible event of one record, or nothing when it carries no body."""

    role = _event_role(node)
    body = _event_body(node)
    if not role or not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=node.event_id,
        parent_id=node.parent_id,
        event_type=node.event_type,
        role=role,
        content=f"{role}: {body}",
        record=node.record,
    )


def _event_role(node: _SessionNode) -> str:
    """Return the role that one record is reported under."""

    if node.event_type == "message":
        message = node.record.get("message")
        role = message.get("role") if isinstance(message, dict) else None
        return role if isinstance(role, str) else ""
    if node.event_type in _PI_SUMMARY_TYPES:
        return node.event_type
    if node.event_type == "custom_message":
        return "custom"
    return ""


def _event_body(node: _SessionNode) -> str:
    """Return the visible body of one record."""

    if node.event_type == "message":
        message = node.record.get("message")
        return _message_body(message) if isinstance(message, dict) else ""
    if node.event_type in _PI_SUMMARY_TYPES:
        return _text_value(node.record.get("summary"))
    if node.event_type == "custom_message":
        return _content_text(node.record.get("content"))
    return ""


def _message_body(message: Mapping[str, Any]) -> str:
    """Return the visible body of one message record."""

    role = message.get("role")
    if role == "toolResult":
        name = _text_value(message.get("toolName"))
        call_id = _text_value(message.get("toolCallId"))
        return f"[tool_result {name} {call_id}] {_content_text(message.get('content'))}"
    if role == "bashExecution":
        command = _text_value(message.get("command"))
        return f"$ {command}\n{_text_value(message.get('output'))}"
    return _content_text(message.get("content"))


def _content_text(content: Any) -> str:
    """Return the visible text of one message content value."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [_block_text(block) for block in content]
    return "\n".join(part for part in parts if part)


def _block_text(block: Any) -> str:
    """Return the visible text of one content block."""

    if not isinstance(block, dict):
        return ""
    kind = block.get("type")
    if kind == "text":
        return _text_value(block.get("text"))
    if kind == "toolCall":
        return _tool_call_text(block)
    return ""


def _tool_call_text(block: Mapping[str, Any]) -> str:
    """Return the visible text of one tool call block."""

    name = _text_value(block.get("name"))
    call_id = _text_value(block.get("id"))
    arguments = block.get("arguments")
    encoded = json.dumps(
        arguments if isinstance(arguments, dict) else {},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"[tool_call {name} {call_id}] {encoded}"


def _text_value(value: Any) -> str:
    """Return one value when it is text and nothing otherwise."""

    return value if isinstance(value, str) else ""


class PiEventAdapter:
    """Resolve and read one captured Pi session event."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured Pi session event as standard text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import the visible events of one Pi session branch as durable Sources."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Import every visible event of the selected branch that is still missing."""

        content = await asyncio.to_thread(self._read_bytes)
        parsed = _parse_session(PiSessionInput(content=content, leaf_id=self._leaf_id))
        imported = _load_imported(session.checkpoint, parsed.session_id)
        _verify_records(parsed, imported)
        collected = dict(imported)
        for event in parsed.events:
            if event.event_id in imported:
                continue
            outcome = await session.submit(
                pi_event_source_id(event.session_id, event.event_id),
                PI_EVENT_SOURCE_NAME,
                event,
            )
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                collected[event.event_id] = _record_digest(event.record)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=_build_checkpoint(parsed.session_id, collected),
        )

    def _read_bytes(self) -> bytes:
        return self._path.read_bytes()


def _load_imported(checkpoint: JsonValue | None, session_id: str) -> dict[str, str]:
    """Read the imported event identities and record digests of one checkpoint."""

    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")  # noqa: TRY003, TRY004
    stored_session = checkpoint.get("session_id")
    if isinstance(stored_session, str) and stored_session != session_id:
        return {}
    imported = checkpoint.get("imported")
    if imported is None:
        return {}
    if not isinstance(imported, dict):
        raise ValueError("Pi session checkpoint must carry an imported event map")  # noqa: TRY003, TRY004
    result: dict[str, str] = {}
    for event_id, digest in imported.items():
        if not isinstance(digest, str):
            raise ValueError("Pi session checkpoint must map event ids to record digests")  # noqa: TRY003, TRY004
        result[event_id] = digest
    return result


def _verify_records(parsed: _ParsedSession, imported: Mapping[str, str]) -> None:
    """Reject an archive that rewrites an event that was already imported."""

    for event_id, digest in imported.items():
        node = parsed.nodes.get(event_id)
        if node is None:
            continue
        if _record_digest(node.record) != digest:
            raise ValueError(f"Pi session event {event_id!r} was rewritten after it was imported")  # noqa: TRY003


def _record_digest(record: Mapping[str, Any]) -> str:
    """Return a stable digest of one original session record."""

    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_checkpoint(session_id: str, imported: Mapping[str, str]) -> JsonValue:
    """Build the checkpoint that records the imported events of one session."""

    return {
        "version": PI_CHECKPOINT_VERSION,
        "session_id": session_id,
        "imported": dict(sorted(imported.items())),
    }


__all__ = [
    "PI_EVENT_IDENTITY_PREFIX",
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventAdapter",
    "PiEventInput",
    "PiEventSource",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
