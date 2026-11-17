"""赛题二契约边界补测:逐条覆盖题面 2.3 / 2.4 中公开测试未覆盖的约定。

运行方式(WSL,仓库根目录下):
    cd /mnt/c/Users/Administrator/test1/mypowercontext
    PYTHONPATH=src competition/.venv/bin/python -m pytest -q <本文件>
"""

from __future__ import annotations

import json

import pytest

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    SourceDefinitionRegistry,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.models import SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _parse(content: bytes, primary_key: str = "id", document_id: str = "doc"):
    return parse_csv_document(CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content))


# ---------------------------------------------------------------- 2.3 CSV 解析


def test_values_keep_raw_strings_without_type_conversion() -> None:
    rows = _parse(b"id,n,empty\n1,11,\n")
    assert rows[0].values == ("1", "11", "")
    assert all(isinstance(value, str) for value in rows[0].values)


def test_non_ascii_content_is_not_escaped() -> None:
    rows = _parse("id,name\n1,北京首都机场\n".encode())
    evidence_content = json.dumps({"id": "1", "name": "北京首都机场"}, ensure_ascii=False, separators=(",", ":"))
    assert evidence_content == '{"id":"1","name":"北京首都机场"}'
    assert "\\u" not in evidence_content
    assert rows[0].values == ("1", "北京首都机场")


def test_primary_key_value_zero_is_not_treated_as_empty() -> None:
    rows = _parse(b"id,name\n0,zero\n")
    assert rows[0].primary_key_value == "0"


def test_header_columns_are_compared_exactly_not_trimmed() -> None:
    rows = _parse(b" id ,name\n001,one\n", primary_key=" id ")
    assert rows[0].columns == (" id ", "name")
    assert rows[0].primary_key == " id "


def test_columns_differing_only_by_spaces_are_distinct() -> None:
    rows = _parse(b"id, id\n1,2\n")
    assert rows[0].columns == ("id", " id")
    assert rows[0].values == ("1", "2")


def test_single_column_header_is_valid() -> None:
    rows = _parse(b"id\n1\n2\n")
    assert [row.primary_key_value for row in rows] == ["1", "2"]


def test_bom_with_header_only_document_is_empty_tuple() -> None:
    assert _parse(b"\xef\xbb\xbfid,name\n") == ()


def test_embedded_crlf_and_quotes_survive_in_values() -> None:
    rows = _parse(b'id,note\r\n1,"a\r\nb ""c"""\r\n')
    assert rows[0].values == ("1", 'a\r\nb "c"')


def test_embedded_commas_and_json_specials_round_trip() -> None:
    # CSV 内嵌引号必须写成 "" (反斜杠不是 CSV 转义符)
    rows = _parse(b'id,note\n1,"x,y {""k"":1}"\n')
    assert rows[0].values == ("1", 'x,y {"k":1}')
    row = {"id": rows[0].values[0], "note": rows[0].values[1]}
    assert json.loads(json.dumps(row)) == row


def test_parse_failure_returns_no_partial_rows() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name\n1,one\n2,two\n3\n")


def test_invalid_utf8_is_rejected() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name\n\xff,one\n")


def test_unclosed_quote_is_rejected() -> None:
    with pytest.raises(ValueError):
        _parse(b'id,name\n1,"unclosed\n')


def test_blank_document_id_and_primary_key_are_rejected() -> None:
    with pytest.raises(ValueError):
        _parse(b"id,name\n1,one\n", document_id="")
    with pytest.raises(ValueError):
        _parse(b"id,name\n1,one\n", document_id=" doc ")
    with pytest.raises(ValueError):
        _parse(b"id,name\n1,one\n", primary_key="")


def test_header_only_document_yields_empty_tuple() -> None:
    assert _parse(b"id,name\n") == ()


# ---------------------------------------------------------------- 2.4 身份与投影


def test_identity_format_and_percent_encoding() -> None:
    assert csv_row_source_id("doc:one", "key/one") == "csv:doc%3Aone:key%2Fone"
    assert csv_row_source_id("a b", "c+d") == "csv:a%20b:c%2Bd"
    # quote 不编码 A-Za-z0-9_.-~,其余百分号编码
    assert csv_row_source_id("a.b-c_d~e", "9") == "csv:a.b-c_d~e:9"


def test_identity_components_cannot_collide() -> None:
    assert csv_row_source_id("a:b", "c") != csv_row_source_id("a", "b:c")
    assert csv_row_source_id("ab", "c") != csv_row_source_id("a", "bc")


def test_identity_length_boundary() -> None:
    prefix = "csv:doc:"
    room = MAX_SOURCE_ID_LENGTH - len(prefix)
    assert len(csv_row_source_id("doc", "x" * room)) == MAX_SOURCE_ID_LENGTH
    with pytest.raises(ValueError):
        csv_row_source_id("doc", "x" * (room + 1))


def test_definition_identity_fields() -> None:
    definition = CSV_ROW_SOURCE_DEFINITION
    assert definition.name == "csv-row"
    assert definition.version == "1"
    assert definition.input_class is CSVRowInput
    assert definition.source_class is CSVRowSource
    assert [(projection.name, projection.version) for projection in definition.projections] == [
        (TEXT_EVIDENCE_PROJECTION_KEY.name, TEXT_EVIDENCE_PROJECTION_KEY.version)
    ]
    assert definition.projections[0].output_class is TextEvidence


def test_resolve_produces_captured_source_with_matching_identity() -> None:
    rows = _parse(b"id,name\n01,Alpha\n")
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio_run(registry.resolve(rows[0]))
    assert isinstance(source, CSVRowSource)
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.name == source.source_id == "csv:doc:01"
    assert asyncio_run(registry.read(source)) == rows[0]


def test_projection_metadata_shape_and_column_order() -> None:
    rows = _parse(b"id,name,note\n01,Alpha,n\n")
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio_run(registry.resolve(rows[0]))
    evidence = TextEvidence.model_validate(registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY))
    assert evidence.source_type == "csv-row"
    assert evidence.source_id == "csv:doc:01"
    assert evidence.content == '{"id":"01","name":"Alpha","note":"n"}'
    assert set(evidence.metadata) == {
        "document_id",
        "primary_key",
        "primary_key_value",
        "columns",
        "row",
    }
    assert list(evidence.metadata["row"]) == ["id", "name", "note"]
    assert evidence.metadata["columns"] == ["id", "name", "note"]


def test_manifest_and_observation_are_produced() -> None:
    rows = _parse(b"id,name\n01,Alpha\n")
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio_run(registry.resolve(rows[0]))
    manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
    assert manifest.name == "csv-row" and manifest.version == "1"
    observation = project_source_for_transport(registry, source)
    assert observation.source_type == "csv-row"
    assert observation.payload["name"] == observation.payload["source_id"] == "csv:doc:01"


def asyncio_run(coroutine):
    import asyncio

    return asyncio.run(coroutine)
