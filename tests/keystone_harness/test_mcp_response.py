from __future__ import annotations

from typing import Any

import pytest

from harnesses.keystone_harness import (
    KeystoneProtocolError,
    KeystoneRpcError,
    KeystoneToolError,
    McpContentKind,
    parse_mcp_tool_call_response,
)


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string", "minLength": 1}},
                "required": ["id"],
                "additionalProperties": True,
            },
        },
        "total": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 0},
        "offset": {"type": "integer", "minimum": 0},
    },
    "required": ["data", "total", "limit", "offset"],
    "additionalProperties": True,
}


def _response(result: dict[str, Any], *, request_id: int = 7) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _structured_page() -> dict[str, Any]:
    return {
        "data": [{"id": "r1", "untyped_extra": "preserved"}],
        "total": 1,
        "limit": 1,
        "offset": 0,
    }


def test_structured_candidate_is_schema_checked_but_withheld_by_default() -> None:
    parsed = parse_mcp_tool_call_response(
        _response({
            "content": [{"type": "text", "text": "One record"}],
            "structuredContent": _structured_page(),
        }),
        7,
        OUTPUT_SCHEMA,
    )

    assert parsed.content_kind is McpContentKind.MIXED
    assert parsed.output_schema_validated
    assert parsed.structured_content is None
    assert not parsed.pagination_ready
    assert "not been verified" in (parsed.unresolved_contract or "")


def test_structured_candidate_never_becomes_pagination_data_without_contract() -> None:
    page = _structured_page()
    parsed = parse_mcp_tool_call_response(
        _response({
            "content": [{"type": "text", "text": "One record"}],
            "structuredContent": page,
        }),
        7,
        OUTPUT_SCHEMA,
    )

    assert parsed.content_kind is McpContentKind.MIXED
    assert parsed.structured_content is None
    assert parsed.output_schema_validated
    assert not parsed.pagination_ready
    assert "not been verified" in (parsed.unresolved_contract or "")


def test_text_only_result_is_not_parsed_as_resource_data() -> None:
    parsed = parse_mcp_tool_call_response(
        _response({"content": [{"type": "text", "text": "{\"data\":[]}"}]}),
        7,
        OUTPUT_SCHEMA,
    )

    assert parsed.content_kind is McpContentKind.TEXT
    assert parsed.structured_content is None
    assert not parsed.pagination_ready


def test_empty_result_is_ambiguous_and_not_pagination_ready() -> None:
    parsed = parse_mcp_tool_call_response(
        _response({"content": []}),
        7,
        OUTPUT_SCHEMA,
    )

    assert parsed.content_kind is McpContentKind.EMPTY
    assert parsed.structured_content is None
    assert not parsed.pagination_ready


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "1.0", "id": 7, "result": {"content": []}},
        {"jsonrpc": "2.0", "id": 8, "result": {"content": []}},
        {"jsonrpc": "2.0", "id": 7, "result": None},
    ],
)
def test_malformed_jsonrpc_envelopes_are_rejected(message: dict[str, Any]) -> None:
    with pytest.raises(KeystoneProtocolError):
        parse_mcp_tool_call_response(message, 7, OUTPUT_SCHEMA)


def test_jsonrpc_error_is_distinguished_without_echoing_server_text() -> None:
    secret_detail = "sensitive server detail"
    message = {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32000, "message": secret_detail},
    }

    with pytest.raises(KeystoneRpcError) as caught:
        parse_mcp_tool_call_response(message, 7, OUTPUT_SCHEMA)

    assert secret_detail not in str(caught.value)


def test_mcp_tool_error_is_distinguished_from_jsonrpc_error() -> None:
    with pytest.raises(KeystoneToolError):
        parse_mcp_tool_call_response(
            _response({"isError": True, "content": [{"type": "text", "text": "failed"}]}),
            7,
            OUTPUT_SCHEMA,
        )


def test_malformed_tool_result_flags_and_content_blocks_are_rejected() -> None:
    for result in (
        {"isError": "false", "content": []},
        {"content": "not-an-array"},
        {"content": [None]},
        {"content": [{"type": "text", "text": 12}]},
        {"content": [], "structuredContent": []},
    ):
        with pytest.raises(KeystoneProtocolError):
            parse_mcp_tool_call_response(_response(result), 7, OUTPUT_SCHEMA)


def test_unknown_result_fields_are_ambiguous_and_withheld() -> None:
    parsed = parse_mcp_tool_call_response(
        _response({
            "content": [],
            "structuredContent": _structured_page(),
            "unrecognized": "extension",
        }),
        7,
        OUTPUT_SCHEMA,
    )

    assert parsed.content_kind is McpContentKind.AMBIGUOUS
    assert parsed.structured_content is None
    assert not parsed.pagination_ready


def test_missing_content_array_does_not_guess_the_structured_envelope() -> None:
    parsed = parse_mcp_tool_call_response(
        _response({"structuredContent": _structured_page()}),
        7,
        OUTPUT_SCHEMA,
    )

    assert parsed.content_kind is McpContentKind.AMBIGUOUS
    assert parsed.structured_content is None
    assert not parsed.pagination_ready


def test_structured_content_must_match_the_discovered_output_schema() -> None:
    malformed_page = {
        "data": [{"id": "r1"}],
        "total": 1,
        "limit": 1,
    }

    with pytest.raises(KeystoneProtocolError, match="structuredContent"):
        parse_mcp_tool_call_response(
            _response({
                "content": [],
                "structuredContent": malformed_page,
            }),
            7,
            OUTPUT_SCHEMA,
        )


def test_unsupported_content_blocks_are_not_passed_to_pagination() -> None:
    parsed = parse_mcp_tool_call_response(
        _response({
            "content": [{"type": "image", "data": "opaque"}],
            "structuredContent": _structured_page(),
        }),
        7,
        OUTPUT_SCHEMA,
    )

    assert parsed.content_kind is McpContentKind.AMBIGUOUS
    assert parsed.structured_content is None
    assert not parsed.pagination_ready
