"""Guarded MCP transport foundation; no Keystone operations are enabled yet."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from weakref import WeakSet

import requests

from .config import KeystoneConfig, KeystoneConfigurationError
from .discovery import (
    CollectionEvidence,
    CompletenessContract,
    CompletenessResult,
    CompletenessStatus,
    PageEvidence,
    validate_completeness,
)

if TYPE_CHECKING:
    from .calendar_list_transport import CalendarListPilotTransport
    from .mcp_response import ParsedMcpToolCallResponse


class KeystoneClientError(RuntimeError):
    """Base class for sanitized Keystone client failures."""


class KeystoneAuthenticationError(KeystoneClientError):
    """Raised when an authenticated operation has no Keystone token."""


class KeystoneOperationNotAllowedError(KeystoneClientError):
    """Raised when an operation is not explicitly registered as read-only."""


class KeystoneTimeoutError(KeystoneClientError):
    """Raised when the Keystone request exceeds its configured timeout."""


class KeystoneTransportError(KeystoneClientError):
    """Raised for transport failures without exposing request diagnostics."""


class KeystoneHttpError(KeystoneClientError):
    """Raised for non-success HTTP responses without exposing response bodies."""


class KeystoneJsonError(KeystoneClientError):
    """Raised when a response body is not valid JSON."""


class KeystoneProtocolError(KeystoneClientError):
    """Raised when the response does not satisfy the JSON-RPC response shape."""


class KeystoneIncompleteDiscoveryError(KeystoneClientError):
    """Raised when tool discovery indicates more data than this client can fetch."""

    def __init__(
        self,
        completeness: CompletenessResult,
        collection_evidence: CollectionEvidence,
    ) -> None:
        self.completeness = completeness
        self.collection_evidence = collection_evidence
        super().__init__(
            "Keystone tools/list discovery is incomplete; continuation is not implemented."
        )


class KeystoneRpcError(KeystoneClientError):
    """Raised for JSON-RPC errors without echoing server-provided text."""


class KeystoneToolError(KeystoneClientError):
    """Raised when an MCP result reports an operation error."""


@dataclass(frozen=True)
class DiscoveredTool:
    """Non-executable metadata returned by MCP tool discovery."""

    name: str
    read_only_hint: bool | None
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolDiscoveryResult:
    """One tools/list response and its response-scoped completeness evidence."""

    tools: tuple[DiscoveredTool, ...]
    collection_evidence: CollectionEvidence
    completeness: CompletenessResult
    global_catalog_completeness_proven: bool = field(default=False, init=False)


@dataclass(frozen=True, eq=False)
class KeystoneReadAuthorization:
    """Client-bound local policy decision; it is not an executable capability."""

    operation: str
    _arguments_json: str = field(repr=False)
    _client_identity: object = field(repr=False)
    _config: KeystoneConfig = field(repr=False)
    _session: object = field(repr=False)
    _discovery: ToolDiscoveryResult = field(repr=False)
    _descriptor_snapshot: DiscoveredTool = field(repr=False)

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self._arguments_json)


@dataclass(frozen=True)
class _CalendarListPilotRequest:
    """Client-claimed immutable request state passed only to the bound transport."""

    _client_identity: object = field(repr=False)
    _transport_identity: object = field(repr=False)
    endpoint: str
    timeout: tuple[float, float]
    request_id: int
    output_schema: dict[str, Any] | None = field(repr=False)


# Diagnostic declaration only. call_read has no transport execution path.
PRODUCTION_READ_ONLY_OPERATIONS: tuple[str, ...] = ()


class KeystoneClient:
    """Keystone client limited to protocol discovery; business tools stay disabled.

    ``tools/list`` discovers metadata but cannot invoke a business operation.
    ``call_read`` remains disabled until trusted operation approval exists.
    """

    def __init__(
        self,
        config: KeystoneConfig,
        session: requests.Session | None = None,
    ) -> None:
        if not isinstance(config, KeystoneConfig):
            raise TypeError("config must be a KeystoneConfig")
        self.config = config
        self._session = session if session is not None else requests.Session()
        self._authorization_identity = object()
        self._discovery_result: ToolDiscoveryResult | None = None
        self._discovery_tools_snapshot: tuple[DiscoveredTool, ...] | None = None
        self._discovery_config: KeystoneConfig | None = None
        self._discovery_session: object | None = None
        self._issued_read_authorizations: WeakSet[KeystoneReadAuthorization] = WeakSet()
        self._state_lock = threading.RLock()
        self._discovery_in_progress = False
        self._calendar_list_attempted = False
        self._next_request_id = 2
        self._calendar_list_transport: CalendarListPilotTransport | None = None

    def list_tools(self) -> ToolDiscoveryResult:
        """Fetch only the MCP tool catalog; never invoke a listed tool."""

        with self._state_lock:
            if self._calendar_list_attempted:
                raise KeystoneOperationNotAllowedError(
                    "Tool discovery cannot be refreshed after a Calendar.list pilot attempt."
                )
            if self._discovery_in_progress:
                raise KeystoneOperationNotAllowedError(
                    "Keystone tool discovery is already in progress."
                )
            self._discovery_in_progress = True
            self._clear_discovery_binding()

        discovery: ToolDiscoveryResult | None = None
        try:
            discovery = self._discover_tools()
            return discovery
        finally:
            with self._state_lock:
                if discovery is not None:
                    self._discovery_result = discovery
                    self._discovery_tools_snapshot = deepcopy(discovery.tools)
                    self._discovery_config = self.config
                    self._discovery_session = self._session
                self._discovery_in_progress = False

    def _discover_tools(self) -> ToolDiscoveryResult:
        try:
            token = self.config.require_token()
        except KeystoneConfigurationError:
            raise KeystoneAuthenticationError(
                "Set KEYSTONE_TOKEN to discover Keystone MCP tools."
            ) from None

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
        try:
            response = self._session.post(
                self.config.endpoint,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                json=payload,
                timeout=self.config.timeout,
            )
        except requests.Timeout as exc:
            raise KeystoneTimeoutError("Keystone tool discovery timed out.") from exc
        except requests.RequestException as exc:
            raise KeystoneTransportError("Keystone tool discovery failed.") from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 401:
                raise KeystoneAuthenticationError(
                    "Keystone rejected the authenticated tool-discovery request."
                ) from exc
            raise KeystoneHttpError(
                f"Keystone returned HTTP {response.status_code} for tool discovery."
            ) from exc

        try:
            message = response.json()
        except ValueError as exc:
            raise KeystoneJsonError("Keystone returned invalid JSON.") from exc

        result = self._parse_response(message, 1)
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise KeystoneProtocolError("Keystone tools/list result has no tools array.")

        discovered: list[DiscoveredTool] = []
        discovered_names: set[str] = set()
        for item in tools:
            if not isinstance(item, dict):
                raise KeystoneProtocolError("Keystone tools/list contains an invalid tool.")
            name = item.get("name")
            description = item.get("description")
            input_schema = item.get("inputSchema")
            annotations = item.get("annotations", {})
            output_schema = item.get("outputSchema")
            meta = item.get("_meta")
            if (
                not isinstance(name, str)
                or not name.strip()
                or (description is not None and not isinstance(description, str))
                or not isinstance(input_schema, dict)
                or not isinstance(annotations, dict)
            ):
                raise KeystoneProtocolError("Keystone tools/list contains malformed tool metadata.")
            if "outputSchema" in item and not isinstance(output_schema, dict):
                raise KeystoneProtocolError(
                    "Keystone tool outputSchema metadata is malformed."
                )
            if "_meta" in item and not isinstance(meta, dict):
                raise KeystoneProtocolError("Keystone tool _meta metadata is malformed.")
            normalized_name = name.strip()
            if normalized_name in discovered_names:
                raise KeystoneProtocolError("Keystone tools/list contains duplicate tool names.")
            discovered_names.add(normalized_name)
            read_only_hint = annotations.get("readOnlyHint")
            if read_only_hint is not None and type(read_only_hint) is not bool:
                raise KeystoneProtocolError("Keystone tool read-only metadata is malformed.")
            discovered.append(
                DiscoveredTool(
                    name=normalized_name,
                    read_only_hint=read_only_hint,
                    description=description,
                    input_schema=dict(input_schema),
                    annotations=dict(annotations),
                    output_schema=(
                        dict(output_schema) if output_schema is not None else None
                    ),
                    meta=dict(meta) if meta is not None else None,
                )
            )

        next_cursor_present = "nextCursor" in result
        next_cursor = result.get("nextCursor")
        if next_cursor_present and next_cursor is not None and (
            not isinstance(next_cursor, str) or not next_cursor.strip()
        ):
            raise KeystoneProtocolError("Keystone tools/list continuation cursor is malformed.")

        collection_evidence = CollectionEvidence(
            (
                PageEvidence(
                    continuation_metadata_present=next_cursor_present,
                    has_more=(
                        next_cursor is not None if next_cursor_present else None
                    ),
                    next_cursor=next_cursor,
                    observed_record_count=len(discovered),
                    record_ids=tuple(tool.name for tool in discovered),
                    truncated=None,
                ),
            ),
            collection_reference="tools/list response",
        )
        completeness = validate_completeness(
            collection_evidence,
            CompletenessContract(
                initial_position_known=True,
                require_truncation_metadata=False,
                max_iterations=1,
            ),
        )
        if completeness.status is not CompletenessStatus.COMPLETE:
            if completeness.status is CompletenessStatus.INCOMPLETE:
                raise KeystoneIncompleteDiscoveryError(
                    completeness, collection_evidence
                )
        discovery = ToolDiscoveryResult(
            tools=tuple(discovered),
            collection_evidence=collection_evidence,
            completeness=completeness,
        )
        return discovery

    def authorize_read_request(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> KeystoneReadAuthorization:
        """Bind a local policy decision to this client's observed catalog/session."""

        with self._state_lock:
            discovery = self._require_owned_discovery()
            from .authorization import validate_resource_read_policy

            descriptor = validate_resource_read_policy(discovery, operation, arguments)
            try:
                arguments_json = json.dumps(
                    dict(arguments),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (TypeError, ValueError):
                raise KeystoneProtocolError(
                    "Authorized arguments cannot be represented as JSON."
                ) from None
            authorization = KeystoneReadAuthorization(
                operation=operation,
                _arguments_json=arguments_json,
                _client_identity=self._authorization_identity,
                _config=self.config,
                _session=self._session,
                _discovery=discovery,
                _descriptor_snapshot=deepcopy(descriptor),
            )
            self._issued_read_authorizations.add(authorization)
            return authorization

    def validate_read_authorization(
        self,
        authorization: KeystoneReadAuthorization,
    ) -> DiscoveredTool:
        """Recheck a client-issued decision without invoking transport."""

        with self._state_lock:
            if (
                not isinstance(authorization, KeystoneReadAuthorization)
                or authorization not in self._issued_read_authorizations
                or authorization._client_identity is not self._authorization_identity
                or authorization._config is not self.config
                or authorization._session is not self._session
            ):
                raise KeystoneOperationNotAllowedError(
                    "Read authorization is not bound to this Keystone client."
                )
            discovery = self._require_owned_discovery()
            if authorization._discovery is not discovery:
                raise KeystoneOperationNotAllowedError(
                    "Read authorization belongs to a different discovery response."
                )
            from .authorization import validate_resource_read_policy

            descriptor = validate_resource_read_policy(
                discovery,
                authorization.operation,
                authorization.arguments,
            )
            if descriptor != authorization._descriptor_snapshot:
                raise KeystoneOperationNotAllowedError(
                    "The discovered descriptor changed after authorization."
                )
            return descriptor

    def _clear_discovery_binding(self) -> None:
        self._discovery_result = None
        self._discovery_tools_snapshot = None
        self._discovery_config = None
        self._discovery_session = None
        self._issued_read_authorizations.clear()

    def _require_owned_discovery(self) -> ToolDiscoveryResult:
        if self._discovery_in_progress:
            raise KeystoneOperationNotAllowedError(
                "Keystone discovery is being refreshed."
            )
        discovery = self._discovery_result
        if discovery is None or self._discovery_tools_snapshot is None:
            raise KeystoneOperationNotAllowedError(
                "This Keystone client has no successful tools/list discovery."
            )
        if self.config is not self._discovery_config or self._session is not self._discovery_session:
            raise KeystoneOperationNotAllowedError(
                "Client configuration or session changed after discovery."
            )
        if discovery.tools != self._discovery_tools_snapshot:
            raise KeystoneOperationNotAllowedError(
                "Discovered tool metadata changed after discovery."
            )
        return discovery

    def call_read(
        self,
        operation: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reject every operation until trusted approval and transport are designed."""

        raise KeystoneOperationNotAllowedError(
            "No Keystone operation has trusted approval; transport is disabled."
        )

    def call_authorized_read(
        self,
        authorization: KeystoneReadAuthorization,
    ) -> dict[str, Any]:
        """Validate a client-owned capability, then keep transport disabled."""

        self.validate_read_authorization(authorization)
        raise KeystoneOperationNotAllowedError(
            "Keystone read transport is disabled."
        )

    def _call_calendar_list_pilot(
        self,
        authorization: KeystoneReadAuthorization,
    ) -> ParsedMcpToolCallResponse:
        """Dispatch only through this client's explicitly configured pilot transport."""

        from .calendar_list_transport import CalendarListPilotTransport

        with self._state_lock:
            transport = self._calendar_list_transport
            if type(transport) is not CalendarListPilotTransport:
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list pilot transport is not configured."
                )
            if not transport._is_bound_to(self, self.config):
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list pilot transport is not bound to this client."
                )
        return transport.send_once(authorization)

    def _claim_calendar_list_pilot_request(
        self,
        authorization: KeystoneReadAuthorization,
        transport: CalendarListPilotTransport,
    ) -> _CalendarListPilotRequest:
        """Validate and consume client state before handing a request to its transport."""

        with self._state_lock:
            config = self.config
            session = self._session
            discovery = self._discovery_result
            if (
                transport is not self._calendar_list_transport
                or not transport._is_bound_to(self, config)
            ):
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list pilot transport is not bound to this client."
                )

            descriptor = self.validate_read_authorization(authorization)
            arguments = authorization.arguments
            if (
                self.config is not config
                or self._session is not session
                or self._discovery_result is not discovery
            ):
                raise KeystoneOperationNotAllowedError(
                    "Client state changed during Calendar.list authorization."
                )
            descriptor_snapshot = deepcopy(descriptor)
            authorized_descriptor = deepcopy(authorization._descriptor_snapshot)
            if descriptor_snapshot != authorized_descriptor:
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list descriptor changed during authorization."
                )
            if (
                authorization.operation != "Calendar.list"
                or set(arguments) != {"limit", "offset"}
                or type(arguments.get("limit")) is not int
                or arguments["limit"] != 1
                or type(arguments.get("offset")) is not int
                or arguments["offset"] != 0
                or descriptor_snapshot.name != "Calendar.list"
            ):
                raise KeystoneOperationNotAllowedError(
                    "Only the fixed Calendar.list pilot request is permitted."
                )
            if self._calendar_list_attempted:
                raise KeystoneOperationNotAllowedError(
                    "The Calendar.list pilot attempt has already been consumed."
                )
            try:
                config.require_token()
            except KeystoneConfigurationError:
                raise KeystoneAuthenticationError(
                    "Set KEYSTONE_TOKEN to invoke the approved Calendar.list pilot."
                ) from None

            request_id = self._next_request_id
            self._next_request_id += 1
            self._calendar_list_attempted = True
            return _CalendarListPilotRequest(
                _client_identity=self._authorization_identity,
                _transport_identity=transport._identity,
                endpoint=config.endpoint,
                timeout=config.timeout,
                request_id=request_id,
                output_schema=deepcopy(descriptor_snapshot.output_schema),
            )

    @staticmethod
    def _parse_response(message: Any, request_id: int) -> dict[str, Any]:
        """Validate a JSON-RPC result without exposing server-provided diagnostics."""

        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise KeystoneProtocolError("Keystone response is not a JSON-RPC 2.0 object.")
        if message.get("id") != request_id:
            raise KeystoneProtocolError("Keystone response ID does not match the request.")
        if "error" in message:
            if not isinstance(message["error"], dict):
                raise KeystoneProtocolError("Keystone JSON-RPC error is malformed.")
            raise KeystoneRpcError("Keystone returned a JSON-RPC error.")
        result = message.get("result")
        if not isinstance(result, dict):
            raise KeystoneProtocolError("Keystone JSON-RPC result is missing or malformed.")
        if result.get("isError") is True:
            raise KeystoneToolError("Keystone MCP operation reported an error.")
        return result