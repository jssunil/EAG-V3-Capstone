"""Offline parsing for candidate MCP tools/call response envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .authorization import validate_json_schema_instance
from .client import (
    KeystoneClient,
    KeystoneProtocolError,
    KeystoneRpcError,
    KeystoneToolError,
)


class McpContentKind(str, Enum):
    STRUCTURED = "structured"
    TEXT = "text"
    MIXED = "mixed"
    OTHER = "other"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ParsedMcpToolCallResponse:
    request_id: int
    content_kind: McpContentKind
    structured_content: Mapping[str, Any] | None
    text_content: tuple[str, ...]
    output_schema_validated: bool
    pagination_ready: bool
    unresolved_contract: str | None = None


def parse_mcp_tool_call_response(
    message: Any,
    request_id: int,
    output_schema: Mapping[str, Any] | None,
) -> ParsedMcpToolCallResponse:
    """Classify a synthetic MCP response without assuming Keystone's wire shape.

    Structured data is schema-checked but withheld until Keystone's envelope
    contract and a live invocation path are separately reviewed.
    """

    if type(request_id) is not int:
        raise TypeError("request_id must be an integer")
    if (
        not isinstance(message, dict)
        or type(message.get("id")) is not int
        or message.get("id") != request_id
    ):
        raise KeystoneProtocolError("Keystone response ID does not match the request.")
    result = KeystoneClient._parse_response(message, request_id)
    if "isError" in result and type(result["isError"]) is not bool:
        raise KeystoneProtocolError("MCP tool isError flag is malformed.")
    if result.get("isError") is True:
        raise KeystoneToolError("Keystone MCP operation reported an error.")
    if "_meta" in result and not isinstance(result["_meta"], Mapping):
        raise KeystoneProtocolError("MCP tool result _meta is malformed.")

    allowed_keys = {"content", "structuredContent", "isError", "_meta"}
    unknown_keys = set(result) - allowed_keys
    if unknown_keys:
        return ParsedMcpToolCallResponse(
            request_id=request_id,
            content_kind=McpContentKind.AMBIGUOUS,
            structured_content=None,
            text_content=(),
            output_schema_validated=False,
            pagination_ready=False,
            unresolved_contract="MCP tool result contains unsupported fields.",
        )

    if "content" not in result:
        return ParsedMcpToolCallResponse(
            request_id=request_id,
            content_kind=McpContentKind.AMBIGUOUS,
            structured_content=None,
            text_content=(),
            output_schema_validated=False,
            pagination_ready=False,
            unresolved_contract="MCP tool result has no content array.",
        )
    if not isinstance(result["content"], list):
        raise KeystoneProtocolError("MCP tool result content must be an array.")

    text_content: list[str] = []
    has_other_content = False
    for item in result["content"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("type"), str):
            raise KeystoneProtocolError("MCP content block is malformed.")
        if item["type"] == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise KeystoneProtocolError("MCP text content block is malformed.")
            text_content.append(text)
        else:
            has_other_content = True

    structured_present = "structuredContent" in result
    if not structured_present:
        if text_content and not has_other_content:
            kind = McpContentKind.TEXT
            reason = "Text content is not validated against the resource output schema."
        elif has_other_content and not text_content:
            kind = McpContentKind.OTHER
            reason = "Non-text MCP content cannot be used as a resource page."
        elif text_content or has_other_content:
            kind = McpContentKind.MIXED
            reason = "MCP content has no verified structured resource result."
        else:
            kind = McpContentKind.EMPTY
            reason = "MCP tool result contains no usable content."
        return ParsedMcpToolCallResponse(
            request_id=request_id,
            content_kind=kind,
            structured_content=None,
            text_content=tuple(text_content),
            output_schema_validated=False,
            pagination_ready=False,
            unresolved_contract=reason,
        )

    candidate = result["structuredContent"]
    if not isinstance(candidate, Mapping):
        raise KeystoneProtocolError("MCP structuredContent must be an object.")
    if not isinstance(output_schema, Mapping):
        raise KeystoneProtocolError("A discovered outputSchema is required.")
    validate_json_schema_instance(
        output_schema,
        candidate,
        location="MCP structuredContent",
    )

    if has_other_content:
        return ParsedMcpToolCallResponse(
            request_id=request_id,
            content_kind=McpContentKind.AMBIGUOUS,
            structured_content=None,
            text_content=tuple(text_content),
            output_schema_validated=True,
            pagination_ready=False,
            unresolved_contract=(
                "Structured content is accompanied by unsupported content blocks."
            ),
        )

    kind = (
        McpContentKind.MIXED
        if text_content or has_other_content
        else McpContentKind.STRUCTURED
    )
    return ParsedMcpToolCallResponse(
        request_id=request_id,
        content_kind=kind,
        structured_content=None,
        text_content=tuple(text_content),
        output_schema_validated=True,
        pagination_ready=False,
        unresolved_contract=(
            "The Keystone tools/call structuredContent envelope has not been verified."
        ),
    )


def parse_calendar_list_pilot_response(
    message: Any,
    request_id: int,
    output_schema: Mapping[str, Any] | None,
) -> ParsedMcpToolCallResponse:
    """Validate the strict, unverified Calendar.list pilot envelope.

    Successful responses require an exact JSON-RPC result envelope, an empty
    ``content`` array, and structured page fields ``data``, ``total``, ``limit``,
    and ``offset``. Candidate content is validated and then withheld.
    """

    _validate_calendar_list_pilot_envelope(message, request_id)
    parsed = parse_mcp_tool_call_response(message, request_id, output_schema)
    if (
        parsed.content_kind is not McpContentKind.STRUCTURED
        or not parsed.output_schema_validated
        or parsed.structured_content is not None
        or parsed.pagination_ready
        or parsed.text_content
    ):
        raise KeystoneProtocolError(
            "Calendar.list pilot response is not an unambiguous structured result."
        )

    candidate = message["result"]["structuredContent"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "data",
        "total",
        "limit",
        "offset",
    }:
        raise KeystoneProtocolError(
            "Calendar.list pilot structured content has an unexpected shape."
        )
    data = candidate["data"]
    total = candidate["total"]
    if (
        not isinstance(data, list)
        or len(data) > 1
        or type(total) is not int
        or total < len(data)
        or type(candidate["limit"]) is not int
        or candidate["limit"] != 1
        or type(candidate["offset"]) is not int
        or candidate["offset"] != 0
    ):
        raise KeystoneProtocolError(
            "Calendar.list pilot response disagrees with the fixed one-record request."
        )
    return parsed


def _validate_calendar_list_pilot_envelope(message: Any, request_id: int) -> None:
    if (
        type(request_id) is not int
        or not isinstance(message, dict)
        or type(message.get("jsonrpc")) is not str
        or message.get("jsonrpc") != "2.0"
        or type(message.get("id")) is not int
        or message.get("id") != request_id
    ):
        raise KeystoneProtocolError(
            "Calendar.list response has an invalid JSON-RPC version or request ID."
        )
    if "error" in message:
        if (
            "result" in message
            or set(message) != {"jsonrpc", "id", "error"}
            or not isinstance(message["error"], dict)
            or set(message["error"]) - {"code", "message", "data"}
            or type(message["error"].get("code")) is not int
            or not isinstance(message["error"].get("message"), str)
        ):
            raise KeystoneProtocolError(
                "Calendar.list JSON-RPC error envelope is malformed."
            )
        return
    if set(message) != {"jsonrpc", "id", "result"} or not isinstance(
        message["result"], dict
    ):
        raise KeystoneProtocolError(
            "Calendar.list JSON-RPC result envelope is malformed."
        )

    result = message["result"]
    allowed_result_fields = {
        "content",
        "structuredContent",
        "isError",
        "_meta",
    }
    if (
        set(result) - allowed_result_fields
        or "content" not in result
        or not isinstance(result["content"], list)
        or (
            "isError" in result
            and type(result["isError"]) is not bool
        )
        or (
            "_meta" in result
            and (
                not isinstance(result["_meta"], Mapping)
                or bool(result["_meta"])
            )
        )
    ):
        raise KeystoneProtocolError(
            "Calendar.list MCP result does not match the assumed pilot envelope."
        )
    if result.get("isError") is True:
        return
    if (
        "structuredContent" not in result
        or result["content"] != []
    ):
        raise KeystoneProtocolError(
            "Calendar.list success must contain structuredContent and an empty content array."
        )
