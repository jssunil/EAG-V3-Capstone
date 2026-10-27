import pytest

from harnesses.keystone_harness import (
    KeystoneClient,
    KeystoneConfig,
    KeystoneOperationNotAllowedError,
)


def _client(base_url: str, approved_base_url: str | None) -> KeystoneClient:
    config = KeystoneConfig.for_hosted_runner(
        base_url=base_url,
        token="disposable-target-test-token",
    )
    arguments: dict[str, object] = {
        "enable_hosted_read_transport": True,
        "enable_hosted_mutation_transport": True,
    }
    if approved_base_url is not None:
        arguments["approved_disposable_base_url"] = approved_base_url
    return KeystoneClient(config, **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "base_url",
    [
        "https://class.agentswitch.theschoolofai.in",
        "https://agentswitch.theschoolofai.in",
        "https://tenant.agentswitch.theschoolofai.in",
        "https://class.agentswitch.theschoolofai.in.",
        "https://class.agentswitch.theschoolofai.in\u3002",
    ],
)
def test_disposable_mutations_reject_shared_live_domain_and_subdomains(
    base_url: str,
) -> None:
    with pytest.raises(KeystoneOperationNotAllowedError, match="shared AgentSwitch"):
        _client(base_url, base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://runner.example.test/copy",
        "https://127.0.0.1/copy",
        "https://[::1]/copy",
    ],
)
def test_disposable_mutations_reject_insecure_or_ip_targets(base_url: str) -> None:
    with pytest.raises(KeystoneOperationNotAllowedError):
        _client(base_url, base_url)


def test_disposable_mutations_require_exact_approved_target() -> None:
    base_url = "https://runner.example.test/copy/keystone"

    with pytest.raises(KeystoneOperationNotAllowedError, match="exact approved"):
        _client(base_url, None)
    with pytest.raises(KeystoneOperationNotAllowedError, match="exact approved"):
        _client(base_url, "https://other.example.test/copy/keystone")

    client = _client(base_url, base_url)

    assert client.hosted_mutation_transport_enabled
