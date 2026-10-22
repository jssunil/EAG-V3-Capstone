from __future__ import annotations

import pytest

from calendar_agent import AgentSwitchConfig, ConfigurationError


def test_config_reads_token_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTSWITCH_TOKEN", "local-token")
    monkeypatch.setenv("AGENTSWITCH_BASE_URL", "https://example.test/")

    config = AgentSwitchConfig.from_env()

    assert config.endpoint == "https://example.test/api/mcp"
    assert config.auth_headers()["Authorization"] == "Bearer local-token"


def test_config_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTSWITCH_TOKEN", "local-token")

    for base_url in (None, "", " \t"):
        if base_url is None:
            monkeypatch.delenv("AGENTSWITCH_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("AGENTSWITCH_BASE_URL", base_url)

        with pytest.raises(ConfigurationError, match="AGENTSWITCH_BASE_URL"):
            AgentSwitchConfig.from_env()

    with pytest.raises(ConfigurationError, match="AGENTSWITCH_BASE_URL"):
        AgentSwitchConfig(base_url=" \t", token="local-token").validate()


def test_config_requires_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTSWITCH_TOKEN", raising=False)
    monkeypatch.setenv("AGENTSWITCH_BASE_URL", "https://example.test")

    with pytest.raises(ConfigurationError, match="AGENTSWITCH_TOKEN"):
        AgentSwitchConfig.from_env()


def test_config_does_not_accept_cookie_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTSWITCH_TOKEN", raising=False)
    monkeypatch.setenv("AGENTSWITCH_BASE_URL", "https://example.test")
    monkeypatch.setenv("AGENTSWITCH_COOKIE", "unverified-cookie")

    with pytest.raises(ConfigurationError, match="AGENTSWITCH_TOKEN"):
        AgentSwitchConfig.from_env()