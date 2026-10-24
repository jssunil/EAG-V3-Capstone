"""Keystone-only configuration; credentials are read only from KEYSTONE_TOKEN."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from collections.abc import Mapping
from urllib.parse import urlsplit


DEFAULT_BASE_URL = "https://class.agentswitch.theschoolofai.in"
_APPROVED_HOST = "class.agentswitch.theschoolofai.in"


class KeystoneConfigurationError(ValueError):
    """Raised when Keystone connection settings do not meet the safety boundary."""


def _validated_runner_base_url(value: object) -> str:
    """Validate the platform-injected instance URL without substituting an origin."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise KeystoneConfigurationError("AGENTSWITCH_BASE_URL must be a valid HTTPS URL.")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise KeystoneConfigurationError("AGENTSWITCH_BASE_URL must not contain whitespace or control characters.")
    if "?" in value or "#" in value:
        raise KeystoneConfigurationError("AGENTSWITCH_BASE_URL must not contain a query string or fragment.")

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise KeystoneConfigurationError("AGENTSWITCH_BASE_URL is malformed.") from None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise KeystoneConfigurationError(
            "AGENTSWITCH_BASE_URL must be an absolute HTTP(S) URL with a host."
        )
    if parsed.username is not None or parsed.password is not None:
        raise KeystoneConfigurationError("AGENTSWITCH_BASE_URL must not contain URL credentials.")
    if parsed.path.rstrip("/").casefold().endswith("/api/mcp"):
        raise KeystoneConfigurationError(
            "AGENTSWITCH_BASE_URL must be the instance base URL, not the /api/mcp endpoint."
        )
    return value.rstrip("/")


def _validated_base_url(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL must be a valid Keystone HTTPS origin.")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL must not contain whitespace or control characters.")
    if "?" in value or "#" in value:
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL must not contain a query string or fragment.")

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL is malformed.") from None

    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL must use HTTPS and include the approved host.")
    if parsed.username is not None or parsed.password is not None:
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL must not contain URL credentials.")
    if hostname != _APPROVED_HOST:
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL host must be class.agentswitch.theschoolofai.in.")
    if port not in (None, 443):
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL must use the standard HTTPS port.")
    if parsed.path not in ("", "/"):
        raise KeystoneConfigurationError("KEYSTONE_BASE_URL must be the approved origin without a path.")
    return DEFAULT_BASE_URL


def _positive_timeout(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise KeystoneConfigurationError(f"{name} must be a finite positive number.")
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError):
        raise KeystoneConfigurationError(f"{name} must be a finite positive number.") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise KeystoneConfigurationError(f"{name} must be a finite positive number.")
    return timeout


@dataclass(frozen=True)
class KeystoneConfig:
    """Validated Keystone origin and optional, Keystone-specific bearer token."""

    base_url: str = DEFAULT_BASE_URL
    token: str | None = field(default=None, repr=False)
    connect_timeout_seconds: float = 3.05
    read_timeout_seconds: float = 10.0
    _hosted_runner_origin: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validated_base_url = (
            _validated_runner_base_url(self.base_url)
            if self._hosted_runner_origin
            else _validated_base_url(self.base_url)
        )
        object.__setattr__(self, "base_url", validated_base_url)
        object.__setattr__(
            self,
            "connect_timeout_seconds",
            _positive_timeout(self.connect_timeout_seconds, "KEYSTONE_CONNECT_TIMEOUT_SECONDS"),
        )
        object.__setattr__(
            self,
            "read_timeout_seconds",
            _positive_timeout(self.read_timeout_seconds, "KEYSTONE_READ_TIMEOUT_SECONDS"),
        )
        if self.token is not None:
            if not isinstance(self.token, str):
                raise KeystoneConfigurationError("KEYSTONE_TOKEN must be text when provided.")
            if any(character.isspace() for character in self.token):
                raise KeystoneConfigurationError("KEYSTONE_TOKEN must not contain whitespace.")

    @classmethod
    def for_hosted_runner(cls, *, base_url: str, token: str) -> KeystoneConfig:
        """Build discovery-only settings from the AgentSwitch runner environment."""

        return cls(
            base_url=base_url,
            token=token,
            _hosted_runner_origin=True,
        )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> KeystoneConfig:
        """Read only KEYSTONE_* settings; no AgentSwitch fallback or login occurs."""

        values = os.environ if environ is None else environ
        base_url = values.get("KEYSTONE_BASE_URL", DEFAULT_BASE_URL)
        token = values.get("KEYSTONE_TOKEN") or None
        connect_timeout = values.get("KEYSTONE_CONNECT_TIMEOUT_SECONDS", "3.05")
        read_timeout = values.get("KEYSTONE_READ_TIMEOUT_SECONDS", "10")
        return cls(
            base_url=base_url,
            token=token,
            connect_timeout_seconds=connect_timeout,
            read_timeout_seconds=read_timeout,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/mcp"

    @property
    def is_approved_pilot_origin(self) -> bool:
        return (
            not self._hosted_runner_origin
            and self.base_url == DEFAULT_BASE_URL
            and self.endpoint == f"{DEFAULT_BASE_URL}/api/mcp"
        )

    @property
    def is_hosted_runner_config(self) -> bool:
        return self._hosted_runner_origin

    @property
    def timeout(self) -> tuple[float, float]:
        return self.connect_timeout_seconds, self.read_timeout_seconds

    def require_token(self) -> str:
        if not self.token:
            raise KeystoneConfigurationError("Set KEYSTONE_TOKEN to make an authenticated Keystone request.")
        return self.token