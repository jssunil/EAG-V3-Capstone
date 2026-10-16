from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import harnesses.suryodaya_harness.hosted_runner as hosted_runner
from calendar_agent.models import ToolCallRecord
from harnesses.keystone_harness import DiscoveredTool
from harnesses.suryodaya_harness.hosted_runner import (
    HostedConfigurationError,
    check_keystone_tool_discovery,
    check_suryodaya_goal_one,
    load_hosted_environment,
    run_hosted,
)
from harnesses.suryodaya_harness.results import parse_results
from tests.unit.fakes import FakeCalendarTools


NOW = datetime(2026, 10, 20, 8, tzinfo=timezone.utc)


def complete_goal_one_tools() -> FakeCalendarTools:
    return FakeCalendarTools(
        parties=[{
            "id": "party-1",
            "email": "plant@example.test",
            "job_title": "Plant Manager",
        }],
        calendars=[{"id": "calendar-1", "owner_party_id": "party-1"}],
        availability_rules=[{
            "id": "rule-1",
            "party_id": "party-1",
            "timezone": "UTC",
            "weekly_hours": [{
                "day": "tuesday",
                "start": "09:00",
                "end": "17:00",
                "enabled": True,
            }],
            "date_overrides": [],
        }],
        scheduling_preferences=[{
            "default_meeting_duration": 30,
            "min_notice_hours": 0,
        }],
        free_busy_result={
            "result": {
                "subjects": [{
                    "subject": "plant@example.test",
                    "visibility": "free_busy",
                    "busy": [],
                    "calendars": [{"access_level": "free_busy"}],
                }],
            },
        },
    )


def test_suryodaya_check_passes_only_for_one_verified_slot() -> None:
    tools = complete_goal_one_tools()

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert result.passed
    assert result.official_record()["passed"] is True
    assert "one structurally valid" in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_suryodaya_unresolved_party_is_non_passing_and_read_only() -> None:
    tools = FakeCalendarTools()

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert not result.passed
    assert result.official_record()["passed"] is False
    assert "planner status=needs_clarification" in result.evidence
    assert "mutation_attempts=0" in result.evidence
    assert all(not call.mutating for call in tools.trace)


class FakeKeystoneClient:
    def __init__(self, tools: tuple[DiscoveredTool, ...]) -> None:
        self.tools = tools
        self.called = False

    def list_tools(self) -> tuple[DiscoveredTool, ...]:
        self.called = True
        return self.tools


def test_keystone_check_reports_catalog_without_business_tool_calls() -> None:
    client = FakeKeystoneClient((DiscoveredTool("Party.list", True),))

    result = check_keystone_tool_discovery(client)  # type: ignore[arg-type]

    assert client.called
    assert result.passed
    assert result.official_record()["passed"] is True
    assert "no business tool was invoked" in result.evidence


def test_keystone_empty_catalog_is_non_passing() -> None:
    result = check_keystone_tool_discovery(FakeKeystoneClient(()))  # type: ignore[arg-type]

    assert not result.passed
    assert result.official_record()["passed"] is False


def test_hosted_environment_is_bound_to_selected_instance() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "keystone",
        "AGENTSWITCH_BASE_URL": "https://class.agentswitch.theschoolofai.in",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.instance == "keystone"
    assert config.base_url == "https://class.agentswitch.theschoolofai.in"
    assert "secret-token" not in repr(config)


def test_hosted_environment_uses_platform_injected_secure_instance_url() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://runner.example.test/tenant/suryodaya",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.base_url == "https://runner.example.test/tenant/suryodaya"


def test_hosted_environment_rejects_insecure_instance_url() -> None:
    try:
        load_hosted_environment({
            "AGENTSWITCH_INSTANCE": "keystone",
            "AGENTSWITCH_BASE_URL": "http://runner.example.test/tenant/keystone",
            "AGENTSWITCH_TOKEN": "secret-token",
        })
    except HostedConfigurationError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("The runner must not send its token over HTTP.")


def test_tool_trace_summary_does_not_include_party_data() -> None:
    from harnesses.suryodaya_harness.hosted_runner import _read_trace_summary

    trace = [ToolCallRecord("Party.list", {"job_title": "Plant Manager"})]

    assert _read_trace_summary(trace) == "read_calls=1;failed_reads=0;mutation_attempts=0"


def test_run_hosted_dispatches_suryodaya_check_and_writes_contract_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tools = complete_goal_one_tools()
    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", lambda config: object())
    monkeypatch.setattr(hosted_runner, "AgentSwitchCalendarTools", lambda client: tools)
    results_path = tmp_path / "results.json"

    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "https://agentswitch.theschoolofai.in",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=results_path,
        now=NOW,
    )

    document = parse_results(output_path.read_text(encoding="utf-8"))
    assert output_path == results_path
    assert checks[0].passed
    assert document["tasks"][0]["passed"] is True
    assert "secret-token" not in results_path.read_text(encoding="utf-8")


def test_run_hosted_dispatches_keystone_tool_discovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = FakeKeystoneClient((DiscoveredTool("Party.list", True),))
    monkeypatch.setattr(hosted_runner, "KeystoneClient", lambda config: client)
    results_path = tmp_path / "results.json"

    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "keystone",
            "AGENTSWITCH_BASE_URL": "https://class.agentswitch.theschoolofai.in",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=results_path,
    )

    document = parse_results(output_path.read_text(encoding="utf-8"))
    assert output_path == results_path
    assert client.called
    assert checks[0].passed
    assert document["tasks"][0]["id"] == "keystone.mcp_tool_discovery"
    assert document["tasks"][0]["passed"] is True
    assert "secret-token" not in results_path.read_text(encoding="utf-8")
