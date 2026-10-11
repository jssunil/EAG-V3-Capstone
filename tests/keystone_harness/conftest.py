from __future__ import annotations

import pytest
import requests


@pytest.fixture(autouse=True)
def prevent_network_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block Requests entry points; this is not an OS-level network sandbox."""

    def fail_if_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline Keystone tests must not open network connections")

    monkeypatch.setattr(requests.sessions.Session, "request", fail_if_network)
    monkeypatch.setattr(requests.sessions.Session, "send", fail_if_network)