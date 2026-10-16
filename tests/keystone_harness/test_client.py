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
        "http://runner.example.test",
        "https://user:password@runner.example.test",
        "https://runner.example.test?instance=keystone",
        "https://runner.example.test:444",
    ],
)
def test_hosted_runner_config_rejects_unsafe_endpoint(base_url: str) -> None:
    with pytest.raises(KeystoneConfigurationError):
        KeystoneConfig.for_hosted_runner(base_url=base_url, token=SECRET)


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
            {"name": "Party.list", "annotations": {"readOnlyHint": True}, "inputSchema": {}},
            {"name": "Party.create", "annotations": {"readOnlyHint": False}, "inputSchema": {}},
        ],
    }))
    client = KeystoneClient(KeystoneConfig(token=SECRET), session=session)  # type: ignore[arg-type]

    tools = client.list_tools()

    assert [(tool.name, tool.read_only_hint) for tool in tools] == [
        ("Party.list", True),
        ("Party.create", False),
    ]
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