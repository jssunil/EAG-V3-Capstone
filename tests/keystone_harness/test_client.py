from __future__ import annotations

from typing import Any

import pytest
import requests

from harnesses.keystone_harness import (
    DEFAULT_BASE_URL,
    KeystoneAuthenticationError,
    KeystoneClient,
    KeystoneConfig,
    KeystoneConfigurationError,
    KeystoneHttpError,
    KeystoneIncompleteDiscoveryError,
    KeystoneJsonError,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
    KeystoneRpcError,
    KeystoneTimeoutError,
    KeystoneTransportError,
    KeystoneToolError,
    PRODUCTION_READ_ONLY_OPERATIONS,
)


SECRET = "keystone-test-secret-never-display"
TEST_OPERATION = "fixture.read"


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeSession:
    def __init__(self, response: FakeResponse | Exception | None = None) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is None:
            raise AssertionError("unexpected transport invocation")
        return self.response


def response(payload: object, *, request_id: int = 1) -> FakeResponse:
    return FakeResponse({"jsonrpc": "2.0", "id": request_id, "result": payload})


def test_default_keystone_origin_is_accepted() -> None:
    config = KeystoneConfig()

    assert config.base_url == DEFAULT_BASE_URL
    assert config.endpoint == f"{DEFAULT_BASE_URL}/api/mcp"


def test_hosted_runner_config_accepts_secure_platform_endpoint() -> None:
    config = KeystoneConfig.for_hosted_runner(
        base_url="https://runner.example.test/tenant/keystone",
        token=SECRET,
    )

    assert config.endpoint == "https://runner.example.test/tenant/keystone/api/mcp"
    assert SECRET not in repr(config)


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://runner.example.test",
        "http://runner.example.test",
        "https://user:password@runner.example.test",
        "https://runner.example.test?instance=keystone",
        "https://runner.example.test:99999",
    ],
)
def test_hosted_runner_config_rejects_unsafe_endpoint(base_url: str) -> None:
    with pytest.raises(KeystoneConfigurationError):
        KeystoneConfig.for_hosted_runner(base_url=base_url, token=SECRET)


def test_hosted_runner_config_preserves_platform_https_host_path_and_port() -> None:
    config = KeystoneConfig.for_hosted_runner(
        base_url="https://runner.internal:8443/tenant/keystone",
        token=SECRET,
    )

    assert config.endpoint == "https://runner.internal:8443/tenant/keystone/api/mcp"


def test_hosted_runner_config_preserves_platform_service_port() -> None:
    config = KeystoneConfig.for_hosted_runner(
        base_url="https://runner.internal:8080/keystone",
        token=SECRET,
    )

    assert config.endpoint == "https://runner.internal:8080/keystone/api/mcp"


@pytest.mark.parametrize(
    "destination",
    [
        "http://class.agentswitch.theschoolofai.in",
        "https://agentswitch.theschoolofai.in",
        "https://class.agentswitch.theschoolofai.in:444",
        "https://user:password@class.agentswitch.theschoolofai.in",
        "https://class.agentswitch.theschoolofai.in?mode=read",
        "https://class.agentswitch.theschoolofai.in#fragment",
        "https://class.agentswitch.theschoolofai.in/path",
        "https://class.agentswitch.theschoolofai.in:not-a-port",
        "https://",
        "not a URL",
    ],
)
def test_invalid_or_non_keystone_destinations_are_rejected(destination: str) -> None:
    with pytest.raises(KeystoneConfigurationError):
        KeystoneConfig(base_url=destination)


def test_standard_https_port_is_canonicalized_to_approved_origin() -> None:
    config = KeystoneConfig(base_url="https://class.agentswitch.theschoolofai.in:443/")

    assert config.base_url == DEFAULT_BASE_URL


def test_configuration_does_not_fall_back_to_agentswitch_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KEYSTONE_TOKEN", raising=False)
    monkeypatch.setenv("AGENTSWITCH_TOKEN", SECRET)

    config = KeystoneConfig.from_env()

    assert config.token is None
    assert SECRET not in repr(config)


def test_configuration_reads_only_the_keystone_token_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTSWITCH_TOKEN", raising=False)
    monkeypatch.setenv("KEYSTONE_TOKEN", SECRET)

    config = KeystoneConfig.from_env()

    assert config.token == SECRET
    assert SECRET not in repr(config)


def test_invalid_keystone_token_error_does_not_echo_the_secret() -> None:
    with pytest.raises(KeystoneConfigurationError) as caught:
        KeystoneConfig(token=f"{SECRET} contains whitespace")

    assert SECRET not in str(caught.value)


def test_client_construction_performs_no_request() -> None:
    session = FakeSession()

    KeystoneClient(KeystoneConfig(), session=session)  # type: ignore[arg-type]

    assert session.calls == []


def test_client_has_no_approved_transport_operation() -> None:
    session = FakeSession()
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    assert PRODUCTION_READ_ONLY_OPERATIONS == ()
    with pytest.raises(KeystoneOperationNotAllowedError, match="transport is disabled"):
        client.call_read(TEST_OPERATION, {"limit": 1})

    assert session.calls == []


@pytest.mark.parametrize("operation", ["tools/list", "CalendarEvent.create", "anything"])
def test_unapproved_operations_fail_before_transport(operation: str) -> None:
    session = FakeSession()
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    with pytest.raises(KeystoneOperationNotAllowedError):
        client.call_read(operation)

    assert session.calls == []


def test_config_and_client_diagnostics_do_not_represent_the_token() -> None:
    config = KeystoneConfig(token=SECRET)
    client = KeystoneClient(config, session=FakeSession())  # type: ignore[arg-type]

    assert SECRET not in repr(config)
    assert SECRET not in repr(client)


def test_arbitrary_operation_arguments_never_reach_transport() -> None:
    session = FakeSession()
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    with pytest.raises(KeystoneOperationNotAllowedError):
        client.call_read("tools/call", {"name": "anything"})

    assert session.calls == []


def test_response_parser_accepts_a_valid_json_rpc_result() -> None:
    assert KeystoneClient._parse_response(
        {"jsonrpc": "2.0", "id": 7, "result": {"read_only": True}},
        7,
    ) == {"read_only": True}


def test_list_tools_performs_only_protocol_discovery() -> None:
    session = FakeSession(response({
        "tools": [
            {
                "name": "Party.list",
                "description": "Synthetic read-only fixture tool.",
                "annotations": {"readOnlyHint": True, "custom": "kept"},
                "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
                "outputSchema": {
                    "type": "object",
                    "properties": {"data": {"type": "array"}},
                },
                "_meta": {"agentswitch": {"permission": "read", "risk": "READ"}},
            },
            {"name": "Party.create", "annotations": {"readOnlyHint": False}, "inputSchema": {}},
        ],
        "nextCursor": None,
    }))
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    discovery = client.list_tools()
    tools = discovery.tools

    assert [(tool.name, tool.read_only_hint) for tool in tools] == [
        ("Party.list", True),
        ("Party.create", False),
    ]
    assert tools[0].description == "Synthetic read-only fixture tool."
    assert tools[0].input_schema == {
        "type": "object",
        "properties": {"limit": {"type": "integer"}},
    }
    assert tools[0].annotations == {"readOnlyHint": True, "custom": "kept"}
    assert tools[0].output_schema == {
        "type": "object",
        "properties": {"data": {"type": "array"}},
    }
    assert tools[0].meta == {
        "agentswitch": {"permission": "read", "risk": "READ"},
    }
    assert tools[1].description is None
    assert tools[1].output_schema is None
    assert tools[1].meta is None
    assert discovery.completeness.status.value == "complete"
    assert discovery.collection_evidence.pages[0].continuation_metadata_present
    assert discovery.collection_evidence.pages[0].has_more is False
    assert discovery.completeness.observed_record_count == 2
    assert discovery.global_catalog_completeness_proven is False
    assert len(session.calls) == 1
    request = session.calls[0]
    assert request["url"] == f"{DEFAULT_BASE_URL}/api/mcp"
    assert request["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert request["json"] == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }


def test_list_tools_marks_absent_cursor_as_unknown_completeness() -> None:
    client = KeystoneClient(
        KeystoneConfig(token=SECRET),
        session=FakeSession(response({
            "tools": [{"name": "Party.list", "inputSchema": {}}],
        })),
    )  # type: ignore[arg-type]

    discovery = client.list_tools()

    page = discovery.collection_evidence.pages[0]
    assert discovery.completeness.status.value == "unknown"
    assert page.continuation_metadata_present is False
    assert page.has_more is None
    assert page.truncated is None
    assert discovery.global_catalog_completeness_proven is False


def test_list_tools_never_sends_a_masked_authorization_header() -> None:
    session = FakeSession(response({"tools": [], "nextCursor": None}))
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    client.list_tools()

    authorization = session.calls[0]["headers"]["Authorization"]
    assert authorization == f"Bearer {SECRET}"
    assert authorization != "******"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("outputSchema", None, "outputSchema"),
        ("outputSchema", [], "outputSchema"),
        ("_meta", None, "_meta"),
        ("_meta", [], "_meta"),
    ],
)
def test_list_tools_rejects_malformed_optional_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    tool = {"name": "Party.list", "inputSchema": {}, "annotations": {}}
    tool[field] = value
    client = KeystoneClient(
        KeystoneConfig(token=SECRET),
        session=FakeSession(response({"tools": [tool]})),
    )  # type: ignore[arg-type]

    with pytest.raises(KeystoneProtocolError, match=message):
        client.list_tools()


def test_discovery_errors_and_logs_do_not_expose_the_token(caplog: pytest.LogCaptureFixture) -> None:
    session = FakeSession(FakeResponse({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32000, "message": SECRET},
    }))
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    with pytest.raises(KeystoneRpcError) as caught:
        client.list_tools()

    assert SECRET not in str(caught.value)
    assert SECRET not in caplog.text


def test_list_tools_requires_token_without_transport() -> None:
    session = FakeSession()
    client = KeystoneClient(KeystoneConfig(), session=session)  # type: ignore[arg-type]

    with pytest.raises(KeystoneAuthenticationError):
        client.list_tools()

    assert session.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"jsonrpc": "2.0", "id": 1, "result": {}},
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": "Party.list"}},
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": [None]}},
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "", "annotations": {}}]}},
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": "Party.list", "annotations": {"readOnlyHint": "yes"}}]},
        },
    ],
)
def test_list_tools_rejects_malformed_catalog(payload: object) -> None:
    client = KeystoneClient(
        KeystoneConfig(token=SECRET),
        session=FakeSession(FakeResponse(payload)),
    )  # type: ignore[arg-type]

    with pytest.raises(KeystoneProtocolError):
        client.list_tools()


@pytest.mark.parametrize(
    "tool",
    [
        {"name": "Party.list", "annotations": {}},
        {"name": "Party.list", "description": 7, "annotations": {}, "inputSchema": {}},
        {"name": "Party.list", "annotations": [], "inputSchema": {}},
        {"name": "Party.list", "annotations": {}, "inputSchema": "object"},
    ],
)
def test_list_tools_rejects_invalid_metadata_field_shapes(tool: dict[str, Any]) -> None:
    client = KeystoneClient(
        KeystoneConfig(token=SECRET),
        session=FakeSession(response({"tools": [tool]})),
    )  # type: ignore[arg-type]

    with pytest.raises(KeystoneProtocolError):
        client.list_tools()


def test_list_tools_rejects_duplicate_names() -> None:
    client = KeystoneClient(
        KeystoneConfig(token=SECRET),
        session=FakeSession(response({
            "tools": [
                {"name": "Party.list", "annotations": {}, "inputSchema": {}},
                {"name": " Party.list ", "annotations": {}, "inputSchema": {}},
            ],
        })),
    )  # type: ignore[arg-type]

    with pytest.raises(KeystoneProtocolError, match="duplicate tool names"):
        client.list_tools()


def test_list_tools_reports_cursor_as_incomplete_without_following_it() -> None:
    session = FakeSession(response({
        "tools": [{"name": "Party.list", "annotations": {}, "inputSchema": {}}],
        "nextCursor": "synthetic-next-cursor",
    }))
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    with pytest.raises(KeystoneIncompleteDiscoveryError) as caught:
        client.list_tools()

    assert caught.value.completeness.status.value == "incomplete"
    page = caught.value.collection_evidence.pages[0]
    assert page.continuation_metadata_present is True
    assert page.has_more is True
    assert page.next_cursor == "synthetic-next-cursor"
    assert "continuation is not implemented" in str(caught.value)
    assert len(session.calls) == 1


@pytest.mark.parametrize("cursor", ["", 7, {}])
def test_list_tools_rejects_malformed_continuation_cursor(cursor: object) -> None:
    client = KeystoneClient(
        KeystoneConfig(token=SECRET),
        session=FakeSession(response({"tools": [], "nextCursor": cursor})),
    )  # type: ignore[arg-type]

    with pytest.raises(KeystoneProtocolError, match="cursor is malformed"):
        client.list_tools()


def test_list_tools_rejects_http_error_without_body() -> None:
    client = KeystoneClient(
        KeystoneConfig(token=SECRET),
        session=FakeSession(FakeResponse({"sensitive": SECRET}, status_code=403)),
    )  # type: ignore[arg-type]

    with pytest.raises(KeystoneHttpError) as caught:
        client.list_tools()

    assert SECRET not in str(caught.value)


def test_production_transport_allowlist_stays_empty() -> None:
    assert PRODUCTION_READ_ONLY_OPERATIONS == ()