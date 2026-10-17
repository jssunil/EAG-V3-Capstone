"""Private, one-shot Requests transport for the Calendar.list pilot."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from typing import Any
from weakref import ref

import requests
from requests.adapters import HTTPAdapter, Retry

from .client import (
    _CalendarListPilotRequest,
    KeystoneClient,
    KeystoneReadAuthorization,
    KeystoneAuthenticationError,
    KeystoneHttpError,
    KeystoneJsonError,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
    KeystoneTimeoutError,
    KeystoneTransportError,
)
from .config import KeystoneConfig
from .mcp_response import (
    ParsedMcpToolCallResponse,
    parse_calendar_list_pilot_response,
)


_STANDARD_SESSION_PREPARE_REQUEST = requests.Session.prepare_request
_STANDARD_SESSION_GET_ADAPTER = requests.Session.get_adapter
_STANDARD_HTTP_ADAPTER_SEND = HTTPAdapter.send
# Tests replace this private seam with an in-memory adapter response.
_HTTP_ADAPTER_SEND = _STANDARD_HTTP_ADAPTER_SEND

_RETRY_PARAMETERS = inspect.signature(Retry).parameters
_REQUIRED_RETRY_FIELDS = ("total", "connect", "read", "redirect", "status")
if not all(name in _RETRY_PARAMETERS for name in _REQUIRED_RETRY_FIELDS):
    raise RuntimeError("Requests does not expose the retry controls required by the pilot.")


class CalendarListPilotTransport:
    """Own a private HTTPS-only transport for one fixed Calendar.list request."""

    def __init__(self, client: KeystoneClient) -> None:
        if type(client) is not KeystoneClient:
            raise TypeError("client must be the concrete KeystoneClient")
        config = client.config
        if type(config) is not KeystoneConfig:
            raise TypeError("client config must be the concrete KeystoneConfig")
        if not config.endpoint.startswith("https://"):
            raise KeystoneOperationNotAllowedError(
                "Calendar.list pilot transport requires HTTPS."
            )

        self._client = ref(client)
        self._client_identity = client._authorization_identity
        self._identity = object()
        self._config = config
        self._endpoint = config.endpoint
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.auth = None
        self._session.cookies.clear()
        self._session.proxies.clear()
        self._session.hooks = {"response": []}
        self._session.adapters.clear()

        retry_options: dict[str, Any] = {
            "total": 0,
            "connect": 0,
            "read": 0,
            "redirect": 0,
            "status": 0,
            "status_forcelist": (),
            "backoff_factor": 0,
        }
        if "other" in _RETRY_PARAMETERS:
            retry_options["other"] = 0
        retry_policy = Retry(**retry_options)
        self._adapter = HTTPAdapter(max_retries=retry_policy)
        self._session.mount("https://", self._adapter)

        self._attempt_lock = threading.Lock()
        self._attempted = False

    def _is_bound_to(self, client: KeystoneClient, config: KeystoneConfig) -> bool:
        return (
            self._client() is client
            and client._authorization_identity is self._client_identity
            and config is self._config
            and client.config is config
            and config.endpoint == self._endpoint
            and self._transport_configuration_is_safe()
        )

    def _transport_configuration_is_safe(self) -> bool:
        session = self._session
        adapter = self._adapter
        if (
            type(session) is not requests.Session
            or session.trust_env is not False
            or session.auth is not None
            or bool(session.cookies)
            or bool(session.proxies)
            or session.hooks != {"response": []}
            or session.verify is not True
            or session.cert is not None
            or set(session.adapters) != {"https://"}
            or type(adapter) is not HTTPAdapter
            or session.adapters.get("https://") is not adapter
            or not self._endpoint.startswith("https://")
            or requests.Session.prepare_request is not _STANDARD_SESSION_PREPARE_REQUEST
            or requests.Session.get_adapter is not _STANDARD_SESSION_GET_ADAPTER
            or HTTPAdapter.send is not _STANDARD_HTTP_ADAPTER_SEND
            or any(
                name in session.__dict__
                for name in (
                    "prepare_request",
                    "get_adapter",
                    "send",
                    "resolve_redirects",
                )
            )
            or "send" in adapter.__dict__
        ):
            return False

        retry_policy = adapter.max_retries
        if type(retry_policy) is not Retry:
            return False
        if any(getattr(retry_policy, name, None) != 0 for name in _REQUIRED_RETRY_FIELDS):
            return False
        if "other" in _RETRY_PARAMETERS and getattr(retry_policy, "other", None) != 0:
            return False
        if retry_policy.status_forcelist not in (None, frozenset(), ()):
            return False
        return retry_policy.backoff_factor == 0

    def send_once(
        self,
        authorization: KeystoneReadAuthorization,
    ) -> ParsedMcpToolCallResponse:
        client = self._client()
        if client is None:
            raise KeystoneOperationNotAllowedError(
                "Calendar.list pilot client no longer exists."
            )
        request = client._claim_calendar_list_pilot_request(
            authorization,
            self,
        )

        with self._attempt_lock:
            if self._attempted:
                raise KeystoneOperationNotAllowedError(
                    "The Calendar.list pilot transport attempt has already been consumed."
                )
            if not self._transport_configuration_is_safe():
                raise KeystoneOperationNotAllowedError(
                    "Calendar.list pilot transport configuration is not permitted."
                )
            try:
                token = self._config.require_token()
            except KeystoneConfigurationError:
                raise KeystoneAuthenticationError(
                    "Set KEYSTONE_TOKEN to invoke the approved Calendar.list pilot."
                ) from None
            self._attempted = True

        return self._dispatch_once(request, token)

    def _dispatch_once(
        self,
        request: _CalendarListPilotRequest,
        token: str,
    ) -> ParsedMcpToolCallResponse:
        if (
            request._client_identity is not self._client_identity
            or request._transport_identity is not self._identity
            or request.endpoint != self._endpoint
            or request.request_id < 2
            or not isinstance(request.output_schema, Mapping)
        ):
            raise KeystoneOperationNotAllowedError(
                "Calendar.list pilot request context is not bound to this transport."
            )

        payload = {
            "jsonrpc": "2.0",
            "id": request.request_id,
            "method": "tools/call",
            "params": {
                "name": "Calendar.list",
                "arguments": {"limit": 1, "offset": 0},
            },
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        prepared = _STANDARD_SESSION_PREPARE_REQUEST(
            self._session,
            requests.Request(
                "POST",
                self._endpoint,
                headers=headers,
                json=payload,
            ),
        )

        transport_failure: str | None = None
        response: Any = None
        try:
            # HTTPAdapter.send passes redirect=False and the zero-retry policy to urllib3.
            response = _HTTP_ADAPTER_SEND(
                self._adapter,
                prepared,
                stream=False,
                timeout=request.timeout,
                verify=True,
                cert=None,
                proxies={},
            )
        except requests.Timeout:
            transport_failure = "timeout"
        except requests.RequestException:
            transport_failure = "transport"

        if transport_failure == "timeout":
            raise KeystoneTimeoutError("Calendar.list pilot request timed out.")
        if transport_failure == "transport":
            raise KeystoneTransportError("Calendar.list pilot request failed.")

        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise KeystoneProtocolError(
                "Calendar.list pilot returned an invalid HTTP status."
            )
        if not 200 <= status_code < 300:
            if status_code == 401:
                raise KeystoneAuthenticationError(
                    "Keystone rejected the Calendar.list pilot request."
                )
            raise KeystoneHttpError(
                f"Keystone returned HTTP {status_code} for the Calendar.list pilot."
            )

        json_failure = False
        message: Any = None
        try:
            message = response.json()
        except ValueError:
            json_failure = True
        if json_failure:
            raise KeystoneJsonError("Calendar.list pilot returned invalid JSON.")

        return parse_calendar_list_pilot_response(
            message,
            request.request_id,
            request.output_schema,
        )
