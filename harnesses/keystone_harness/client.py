"""Guarded MCP transport foundation; no Keystone operations are enabled yet."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import requests

from .config import KeystoneConfig


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


# Diagnostic declaration only. call_read has no transport execution path.
PRODUCTION_READ_ONLY_OPERATIONS: tuple[str, ...] = ()


class KeystoneClient:
    """Offline-only Keystone client shell; no operation is enabled for transport.

    Construction stores configuration/session objects only. A trusted operation
    policy and Keystone protocol contract must exist before transport is added.
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