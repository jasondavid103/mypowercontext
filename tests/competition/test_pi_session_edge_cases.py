"""赛题三契约边界补测:覆盖题面 2.3 / 2.4 / 2.5 中公开测试未覆盖的约定。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q tests/competition/test_pi_session_edge_cases.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import ConnectorBinding, SourceDefinitionRegistry, project_source_for_transport
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.models import SourceRef
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
    pi_event_source_id,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY

HEADER = {"type": "session", "version": 3, "id": "edge"}


def _archive(records: list[dict[str, Any]], *, header: dict[str, Any] | None = None, leaf: str | None = None):
    rows = [header or HEADER, *records]
    raw = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
    return parse_pi_session(PiSessionInput(content=raw, leaf_id=leaf))


def _message(event_id: str, parent: str | None, role: str, content: Any) -> dict[str, Any]:
    return {"type": "message", "id": event_id, "parentId": parent, "message": {"role": role, "content": content}}


def test_tool_call_arguments_are_sorted_compact_and_unicode() -> None:
    events = _archive(
        [
            _message(
                "a",
                None,
                "assistant",
                [
                    {
                        "type": "toolCall",
                        "name": "write",
                        "id": "call-1",
                        "arguments": {"路径": "文件.txt", "mode": "overwrite", "count": 3},
                    }
                ],
            )
        ]
    )
    assert events[0].content == 'assistant: [tool_call write call-1] {"count":3,"mode":"overwrite","路径":"文件.txt"}'


def test_control_nodes_link_the_branch_without_text() -> None:
    records = [
        _message("a", None, "user", "start"),
        {"type": "model_change", "id": "control", "parentId": "a"},
        _message("b", "control", "assistant", "after control"),
    ]
    events = _archive(records)
    assert [event.event_id for event in events] == ["a", "b"]
    assert [event.parent_id for event in events] == [None, "control"]
    assert [event.event_type for event in events] == ["message", "message"]


def test_control_leaf_selects_only_its_visible_ancestors() -> None:
    records = [
        _message("a", None, "user", "start"),
        {"type": "usage", "id": "control", "parentId": "a", "kind": "cache_warm"},
    ]
    assert [event.event_id for event in _archive(records, leaf="control")] == ["a"]


def test_empty_parent_is_treated_as_a_root() -> None:
    records = [_message("a", "", "user", "start"), _message("b", "a", "assistant", "next")]
    events = _archive(records)
    assert [event.event_id for event in events] == ["a", "b"]
    assert events[0].parent_id is None


def test_multiple_roots_keep_only_the_selected_branch() -> None:
    records = [
        _message("a", None, "user", "first root"),
        _message("b", None, "user", "second root"),
        _message("c", "a", "assistant", "under first root"),
    ]
    assert [event.event_id for event in _archive(records, leaf="c")] == ["a", "c"]
    assert [event.event_id for event in _archive(records, leaf="b")] == ["b"]


def test_identity_is_stable_bounded_and_collision_free() -> None:
    first = pi_event_source_id("session", "event")
    assert first == pi_event_source_id("session", "event")
    assert first != pi_event_source_id("session", "other")
    assert first != pi_event_source_id("other", "event")
    long_identity = pi_event_source_id("s" * 400, "e" * 400)
    assert len(long_identity) <= MAX_SOURCE_ID_LENGTH
    assert long_identity != pi_event_source_id("s" * 400, "f" * 400)


def test_large_tool_result_is_preserved_in_full() -> None:
    payload = "x" * 20000
    events = _archive([_message("a", None, "toolResult", [{"type": "text", "text": payload}])])
    assert events[0].content.endswith(payload)
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(events[0]))
    evidence = registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY)
    assert evidence["content"] == events[0].content


def test_read_returns_the_original_event() -> None:
    event = _archive([_message("a", None, "user", "start")])[0]
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(event))
    assert asyncio.run(registry.read(source)) == event
    assert source.name == pi_event_source_id(event.session_id, event.event_id)


def test_projection_keeps_the_event_identity() -> None:
    records = [
        _message("a", None, "user", "start"),
        {"type": "compaction", "id": "c", "parentId": "a", "summary": "summary text"},
    ]
    events = _archive(records)
    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    for event in events:
        source = asyncio.run(registry.resolve(event))
        observation = project_source_for_transport(registry, source)
        evidence = observation.projection(TEXT_EVIDENCE_PROJECTION_KEY)
        assert evidence["source_type"] == "pi-event"
        assert evidence["source_id"] == source.name
        assert evidence["content"] == event.content
        assert evidence["metadata"]["event_id"] == event.event_id
        assert evidence["metadata"]["event_type"] == event.event_type
        assert evidence["metadata"]["role"] == event.role


class _Sink:
    """Collect submitted items instead of persisting them."""

    def __init__(self) -> None:
        self.items: list[tuple[str, object]] = []

    async def submit(self, binding, item_id, definition_name, value, /) -> SourceRef:
        self.items.append((item_id, value))
        return SourceRef(source_type=definition_name, source_id=item_id)


class _Checkpoints:
    """Keep one connector checkpoint in memory."""

    def __init__(self) -> None:
        self.value: JsonValue | None = None

    async def load(self, binding, /) -> JsonValue | None:
        return self.value

    async def save(self, binding, checkpoint, /, *, expected) -> None:
        self.value = checkpoint


def _binding() -> ConnectorBinding:
    return ConnectorBinding(
        scope_id="scope",
        binding_id="pi",
        connector_name="pi-session-history",
        connector_version="1",
    )


def _run(path: Path, leaf: str | None, sink: _Sink, checkpoints: _Checkpoints):
    connector = PiSessionConnector(path, leaf)
    return asyncio.run(ConnectorLifecycle(sink=sink, checkpoints=checkpoints).run(connector, _binding()))


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [HEADER, *records]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_branch_switch_imports_only_the_new_nodes(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    records = [
        _message("root", None, "user", "Shared request"),
        _message("left", "root", "assistant", "Left evidence"),
        {"type": "model_change", "id": "control", "parentId": "root"},
        _message("right", "control", "assistant", "Right evidence"),
    ]
    _write(path, records)
    sink, checkpoints = _Sink(), _Checkpoints()

    first = _run(path, "left", sink, checkpoints)
    assert len(first.items) == 2
    second = _run(path, "right", sink, checkpoints)
    assert len(second.items) == 1
    assert second.previous_checkpoint == first.committed_checkpoint
    assert len(sink.items) == 3


def test_replay_after_restart_imports_nothing(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    _write(path, [_message("root", None, "user", "Shared request")])
    sink, checkpoints = _Sink(), _Checkpoints()
    first = _run(path, None, sink, checkpoints)
    replay = _run(path, None, sink, checkpoints)
    assert replay.items == ()
    assert replay.committed_checkpoint == first.committed_checkpoint
    assert len(sink.items) == 1


def test_appended_events_extend_the_same_progress(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    records = [_message("root", None, "user", "Shared request")]
    _write(path, records)
    sink, checkpoints = _Sink(), _Checkpoints()
    first = _run(path, None, sink, checkpoints)

    records.append(_message("tail", "root", "assistant", "Later evidence"))
    _write(path, records)
    second = _run(path, "tail", sink, checkpoints)
    assert len(second.items) == 1
    assert second.previous_checkpoint == first.committed_checkpoint


def test_corrupt_input_keeps_the_committed_progress(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    records = [_message("root", None, "user", "Shared request")]
    _write(path, records)
    sink, checkpoints = _Sink(), _Checkpoints()
    first = _run(path, None, sink, checkpoints)

    path.write_text(path.read_text(encoding="utf-8") + "{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _run(path, None, sink, checkpoints)
    assert checkpoints.value == first.committed_checkpoint


def test_rewritten_event_is_rejected_without_losing_progress(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    records = [_message("root", None, "user", "Shared request")]
    _write(path, records)
    sink, checkpoints = _Sink(), _Checkpoints()
    first = _run(path, None, sink, checkpoints)

    records[0]["message"]["content"] = "Changed request"
    _write(path, records)
    with pytest.raises(ValueError):
        _run(path, None, sink, checkpoints)
    assert checkpoints.value == first.committed_checkpoint

    records[0]["message"]["content"] = "Shared request"
    _write(path, records)
    repaired = _run(path, None, sink, checkpoints)
    assert repaired.previous_checkpoint == first.committed_checkpoint
    assert repaired.items == ()
