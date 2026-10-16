"""Guarded MCP transport foundation; no Keystone operations are enabled yet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

from .config import KeystoneConfig, KeystoneConfigurationError


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


class KeystoneRpcError(KeystoneClientError):
    """Raised for JSON-RPC errors without echoing server-provided text."""


class KeystoneToolError(KeystoneClientError):
    """Raised when an MCP result reports an operation error."""


@dataclass(frozen=True)
class DiscoveredTool:
    """Minimal, non-executable metadata returned by MCP tool discovery."""

    name: str
    read_only_hint: bool | None


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

    def list_tools(self) -> tuple[DiscoveredTool, ...]:
        """Fetch only the MCP tool catalog; never invoke a listed tool."""

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
        for item in tools:
            if not isinstance(item, dict):
                raise KeystoneProtocolError("Keystone tools/list contains an invalid tool.")
            name = item.get("name")
            annotations = item.get("annotations", {})
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(annotations, dict)
            ):
                raise KeystoneProtocolError("Keystone tools/list contains malformed tool metadata.")
            read_only_hint = annotations.get("readOnlyHint")
            if read_only_hint is not None and type(read_only_hint) is not bool:
                raise KeystoneProtocolError("Keystone tool read-only metadata is malformed.")
            discovered.append(DiscoveredTool(name=name.strip(), read_only_hint=read_only_hint))
        return tuple(discovered)

    def call_read(
        self,
        operation: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reject every operation until trusted approval and transport are designed."""

        raise KeystoneOperationNotAllowedError(
            "No Keystone operation has trusted approval; transport is disabled."
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