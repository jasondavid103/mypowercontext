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

"""Context Pack wire protocol for incremental context transfer.

A Context Pack carries the whole context of one turn as a list of chunks. A
Context Pack Delta carries only the chunks that the receiver does not own yet:
the prefix of the base pack it keeps and the suffix it has to append. Both
objects serialize to canonical JSON so that a receiver can rebuild the target
context from the bytes alone, without any state of the sending process.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

CONTEXT_PACK_SCHEMA: Final = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA: Final = "powercontext.context-pack-delta.v1"
CONTEXT_PACK_DEFAULT_CHUNK_BYTES: Final = 512
CONTEXT_PACK_CONTEXT_CHUNK_BYTES: Final = 512
CONTEXT_PACK_DIGEST_PREFIX: Final = "sha256:"
CONTEXT_PACK_DIGEST_LENGTH: Final = len("sha256:") + 64


class ContextPackError(ValueError):
    """Raised when a Context Pack or delta violates the wire contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context payload split into transferable chunks."""

    items: tuple[bytes, ...] = ()

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a pack from the chunks of one context."""

        return cls(items=tuple(items))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = CONTEXT_PACK_DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split one context into chunks that keep the original bytes."""

        return cls(items=tuple(_chunk_content(content, chunk_bytes)))

    @property
    def digest(self) -> str:
        """Return the digest of the chunk list and the schema."""

        return _digest({"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(self.items)})

    @property
    def byte_count(self) -> int:
        """Return the number of bytes the serialized pack occupies."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the pack into its canonical wire form."""

        payload: dict[str, Any] = {
            "schema": CONTEXT_PACK_SCHEMA,
            "items": _encode_items(self.items),
            "digest": self.digest,
        }
        count = len(_canonical({**payload, "byte_count": 0}))
        while True:
            encoded = _canonical({**payload, "byte_count": count})
            if len(encoded) == count:
                return encoded
            count = len(encoded)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and verify one serialized pack."""

        payload = _decode_object(data)
        _require_fields(payload, {"schema", "items", "digest", "byte_count"}, "pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("pack schema is not supported")  # noqa: TRY003
        items = _decode_items(payload["items"])
        pack = cls(items=items)
        if payload["digest"] != pack.digest:
            raise ContextPackError("pack digest does not match its chunks")  # noqa: TRY003
        if payload["byte_count"] != len(data):
            raise ContextPackError("pack byte count does not match its wire form")  # noqa: TRY003
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """The chunks one receiver has to append to a base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...] = ()

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Build the delta that turns the base pack into the target pack."""

        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=tuple(target.items[prefix:]),
        )

    @property
    def byte_count(self) -> int:
        """Return the number of bytes the serialized delta occupies."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta into its canonical wire form."""

        return _canonical({
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        })

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and verify one serialized delta."""

        payload = _decode_object(data)
        _require_fields(
            payload,
            {"schema", "base_digest", "target_digest", "prefix_items", "appended"},
            "delta",
        )
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("delta schema is not supported")  # noqa: TRY003
        prefix = payload["prefix_items"]
        if type(prefix) is not int or prefix < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not _is_digest(base_digest):
            raise ContextPackError("base_digest must be a sha256 digest")  # noqa: TRY003
        if not isinstance(target_digest, str) or not _is_digest(target_digest):
            raise ContextPackError("target_digest must be a sha256 digest")  # noqa: TRY003
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix,
            appended=_decode_items(payload["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from the base pack and one delta."""

    if base.digest != delta.base_digest:
        raise ContextPackError("delta does not belong to the base pack")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("prefix_items exceeds the base pack")  # noqa: TRY003
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("delta does not rebuild the target pack")  # noqa: TRY003
    return target


def _canonical(payload: dict[str, Any]) -> bytes:
    """Encode one wire object as canonical JSON."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: dict[str, Any]) -> str:
    """Return the digest of one wire object."""

    return CONTEXT_PACK_DIGEST_PREFIX + hashlib.sha256(_canonical(payload)).hexdigest()


def _is_digest(value: str) -> bool:
    """Report whether one value is a sha256 digest."""

    if len(value) != CONTEXT_PACK_DIGEST_LENGTH or not value.startswith(CONTEXT_PACK_DIGEST_PREFIX):
        return False
    return all(character in "0123456789abcdef" for character in value[len(CONTEXT_PACK_DIGEST_PREFIX) :])


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    """Encode the chunks of one pack as base64 strings."""

    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: Any) -> tuple[bytes, ...]:
    """Decode the base64 chunks of one pack."""

    if not isinstance(value, list):
        raise ContextPackError("items must be a list of base64 strings")  # noqa: TRY003
    decoded: list[bytes] = []
    for item in value:
        if not isinstance(item, str):
            raise ContextPackError("items must be a list of base64 strings")  # noqa: TRY003
        try:
            decoded.append(base64.b64decode(item, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("items must be a list of base64 strings") from error  # noqa: TRY003
    return tuple(decoded)


def _decode_object(data: bytes) -> dict[str, Any]:
    """Decode one wire object and require that it is a JSON object."""

    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("wire form must be a JSON object") from error  # noqa: TRY003
    if not isinstance(payload, dict):
        raise ContextPackError("wire form must be a JSON object")  # noqa: TRY003
    return payload


def _require_fields(payload: dict[str, Any], fields: set[str], kind: str) -> None:
    """Require that one wire object carries exactly the declared fields."""

    if set(payload) != fields:
        raise ContextPackError(f"{kind} must carry exactly the declared fields")  # noqa: TRY003


def _chunk_content(content: str, chunk_bytes: int) -> list[bytes]:
    """Split one context into chunks of at most the given number of bytes."""

    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
    data = content.encode("utf-8")
    chunks: list[bytes] = []
    start = 0
    while start < len(data):
        end = min(start + chunk_bytes, len(data))
        while end > start:
            try:
                data[start:end].decode("utf-8")
            except UnicodeDecodeError:
                end -= 1
                continue
            break
        else:
            raise ContextPackError("chunk_bytes cannot hold a valid utf-8 boundary")  # noqa: TRY003
        chunks.append(data[start:end])
        start = end
    return chunks


__all__ = [
    "CONTEXT_PACK_DEFAULT_CHUNK_BYTES",
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
