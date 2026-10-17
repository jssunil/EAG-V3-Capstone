"""Offline authorization and response checks for a proposed Calendar.list pilot."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakSet

from .client import (
    KeystoneClient,
    DiscoveredTool,
    KeystoneOperationNotAllowedError,
    KeystoneReadAuthorization,
)
from .mcp_response import (
    ParsedMcpToolCallResponse,
    parse_calendar_list_pilot_response,
)


_PILOT_OPERATION = "Calendar.list"
_PILOT_ARGUMENTS = {"limit": 1, "offset": 0}


@dataclass(frozen=True, eq=False)
class CalendarListPilotAuthorization:
    """Non-executable authorization wrapper issued by one pilot instance."""

    _pilot_identity: object = field(repr=False)
    _client_authorization: KeystoneReadAuthorization = field(repr=False)


class CalendarListSmokePilot:
    """Offline-only gate for one assumed Calendar.list request/response contract.

    The assumed success envelope is JSON-RPC 2.0 with only ``jsonrpc``, integer
    ``id``, and ``result`` at the top level. The result requires an empty
    ``content`` array and ``structuredContent`` with exactly ``data``, ``total``,
    ``limit``, and ``offset``. Only optional ``isError: false`` and empty
    ``_meta`` are accepted. An error envelope contains only ``jsonrpc``, ``id``,
    and ``error``; ``error`` requires integer ``code`` and string ``message``,
    with optional ``data``. The page must contain at most one record, a
    nonnegative integer ``total`` no smaller than the returned row count, and
    exact integer ``limit=1`` and ``offset=0``. Errors are failures, never
    successful reads. This Keystone wire contract is unverified.
    """

    def __init__(self, client: KeystoneClient) -> None:
        if type(client) is not KeystoneClient:
            raise TypeError("client must be the concrete KeystoneClient")
        self._client = client
        self._identity = object()
        self._issued: WeakSet[CalendarListPilotAuthorization] = WeakSet()
        self._attempt_lock = threading.Lock()
        self._attempted = False

    def authorize(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> CalendarListPilotAuthorization:
        with self._attempt_lock:
            if operation != _PILOT_OPERATION:
                raise KeystoneOperationNotAllowedError(
                    "Only Calendar.list is allowed by the offline smoke-pilot policy."
                )
            self._validate_arguments(arguments)
            client_authorization = self._client.authorize_read_request(
                _PILOT_OPERATION,
                arguments,
            )
            self._client.validate_read_authorization(client_authorization)
            authorization = CalendarListPilotAuthorization(
                _pilot_identity=self._identity,
                _client_authorization=client_authorization,
            )
            self._issued.add(authorization)
            return authorization

    def invoke(
        self,
        authorization: CalendarListPilotAuthorization,
    ) -> ParsedMcpToolCallResponse:
        """Consume this pilot's sole attempt and return only a sanitized classification."""

        with self._attempt_lock:
            self._validate_authorization(authorization)
            if self._attempted:
                raise KeystoneOperationNotAllowedError(
                    "The Calendar.list pilot attempt has already been consumed."
                )
            self._attempted = True

        return self._client._call_calendar_list_pilot(
            authorization._client_authorization
        )

    def parse_response(
        self,
        authorization: CalendarListPilotAuthorization,
        message: Any,
        request_id: int,
    ) -> ParsedMcpToolCallResponse:
        descriptor = self._validate_authorization(authorization)
        return parse_calendar_list_pilot_response(
            message,
            request_id,
            descriptor.output_schema,
        )

    def _validate_authorization(
        self,
        authorization: CalendarListPilotAuthorization,
    ) -> DiscoveredTool:
        if (
            not isinstance(authorization, CalendarListPilotAuthorization)
            or authorization not in self._issued
            or authorization._pilot_identity is not self._identity
        ):
            raise KeystoneOperationNotAllowedError(
                "Pilot authorization was not issued by this Calendar.list pilot."
            )
        client_authorization = authorization._client_authorization
        if (
            client_authorization.operation != _PILOT_OPERATION
            or client_authorization.arguments != _PILOT_ARGUMENTS
        ):
            raise KeystoneOperationNotAllowedError(
                "Pilot authorization does not match the fixed Calendar.list request."
            )
        descriptor = self._client.validate_read_authorization(client_authorization)
        self._validate_arguments(client_authorization.arguments)
        return descriptor

    @staticmethod
    def _validate_arguments(arguments: Mapping[str, Any]) -> None:
        if (
            not isinstance(arguments, Mapping)
            or set(arguments) != set(_PILOT_ARGUMENTS)
            or type(arguments.get("limit")) is not int
            or arguments.get("limit") != 1
            or type(arguments.get("offset")) is not int
            or arguments.get("offset") != 0
        ):
            raise KeystoneOperationNotAllowedError(
                "Calendar.list pilot arguments must be exactly limit=1 and offset=0."
            )
