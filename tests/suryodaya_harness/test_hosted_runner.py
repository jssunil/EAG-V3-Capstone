from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import harnesses.suryodaya_harness.hosted_runner as hosted_runner
from calendar_agent import AgentSwitchError
from calendar_agent.models import ToolCallRecord
from harnesses.suryodaya_harness.hosted_runner import (
    HostedConfigurationError,
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


def test_suryodaya_candidate_is_unevaluated_even_when_structurally_valid() -> None:
    tools = complete_goal_one_tools()

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert not result.passed
    assert result.official_record()["passed"] is False
    assert "one structurally valid" in result.evidence
    assert result.evidence.startswith("UNEVALUATED:")
    assert "authoritative business-goal predicate" in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_suryodaya_unresolved_party_is_non_passing_and_read_only() -> None:
    tools = FakeCalendarTools()

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert not result.passed
    assert result.official_record()["passed"] is False
    assert "planner status=needs_clarification" in result.evidence
    assert "mutation_attempts=0" in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_agent_switch_read_failure_remains_unevaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = complete_goal_one_tools()

    def unavailable(self, task):
        raise AgentSwitchError("synthetic read failure")

    monkeypatch.setattr(hosted_runner.CalendarAgent, "run", unavailable)

    result = check_suryodaya_goal_one(tools, now=NOW)

    assert not result.passed
    assert result.evidence.startswith("UNEVALUATED:")
    assert "AgentSwitchError" in result.evidence
    assert all(not call.mutating for call in tools.trace)


def test_hosted_environment_is_bound_to_selected_instance() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://agentswitch.theschoolofai.in",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.instance == "suryodaya"
    assert config.base_url == "https://agentswitch.theschoolofai.in"
    assert "secret-token" not in repr(config)


def test_hosted_environment_uses_platform_injected_secure_instance_url() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://runner.example.test:8443/tenant/suryodaya",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.base_url == "https://runner.example.test:8443/tenant/suryodaya"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("AGENTSWITCH_BASE_URL", "", "must be set"),
        ("AGENTSWITCH_BASE_URL", "http://runner.example.test", "HTTPS"),
        ("AGENTSWITCH_BASE_URL", "https://user:password@runner.example.test", "credential-free"),
        ("AGENTSWITCH_BASE_URL", "https://runner.example.test/api/mcp", "instance base URL"),
        ("AGENTSWITCH_BASE_URL", "https://runner.example.test?tenant=suryodaya", "query or fragment"),
        ("AGENTSWITCH_INSTANCE", "keystone", "Only the Suryodaya"),
    ],
)
def test_hosted_environment_rejects_invalid_configuration(field, value, message) -> None:
    values = {
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://runner.example.test",
        "AGENTSWITCH_TOKEN": "secret-token",
    }
    values[field] = value

    with pytest.raises(HostedConfigurationError, match=message) as error:
        load_hosted_environment(values)

    assert "secret-token" not in str(error.value)


def test_hosted_environment_requires_each_runner_value() -> None:
    for missing in ("AGENTSWITCH_INSTANCE", "AGENTSWITCH_BASE_URL", "AGENTSWITCH_TOKEN"):
        values = {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "https://runner.example.test",
            "AGENTSWITCH_TOKEN": "secret-token",
        }
        del values[missing]
        with pytest.raises(HostedConfigurationError, match=missing):
            load_hosted_environment(values)


def test_hosted_environment_accepts_credential_free_https_instance_base() -> None:
    config = load_hosted_environment({
        "AGENTSWITCH_INSTANCE": "suryodaya",
        "AGENTSWITCH_BASE_URL": "https://runner.example.test/tenant/suryodaya/",
        "AGENTSWITCH_TOKEN": "secret-token",
    })

    assert config.base_url == "https://runner.example.test/tenant/suryodaya"


def test_hosted_configuration_error_is_unevaluated_and_secret_safe(tmp_path: Path) -> None:
    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "http://runner.example.test",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=tmp_path / "results.json",
    )

    serialized = output_path.read_text(encoding="utf-8")
    document = parse_results(serialized)
    assert checks[0].passed is False
    assert document["tasks"][0]["passed"] is False
    assert document["tasks"][0]["evidence"].startswith("UNEVALUATED:")
    assert "secret-token" not in serialized


def test_unexpected_hosted_initialization_error_is_not_masked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_initialization(config):
        raise RuntimeError("synthetic programming failure")

    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", fail_initialization)
    output_path = tmp_path / "results.json"

    with pytest.raises(RuntimeError, match="synthetic programming failure"):
        run_hosted(
            {
                "AGENTSWITCH_INSTANCE": "suryodaya",
                "AGENTSWITCH_BASE_URL": "https://runner.example.test:8443/tenant/suryodaya",
                "AGENTSWITCH_TOKEN": "synthetic-token",
            },
            output_path=output_path,
        )

    assert not output_path.exists()


def test_unexpected_planner_exception_is_not_masked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tools = complete_goal_one_tools()

    def fail_planning(self, task):
        raise RuntimeError("secret-token must not be serialized")

    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", lambda config: object())
    monkeypatch.setattr(hosted_runner, "AgentSwitchCalendarTools", lambda client: tools)
    monkeypatch.setattr(hosted_runner.CalendarAgent, "run", fail_planning)
    output_path = tmp_path / "results.json"

    with pytest.raises(RuntimeError, match="secret-token must not be serialized"):
        run_hosted(
            {
                "AGENTSWITCH_INSTANCE": "suryodaya",
                "AGENTSWITCH_BASE_URL": "https://runner.example.test",
                "AGENTSWITCH_TOKEN": "synthetic-token",
            },
            output_path=output_path,
            now=NOW,
        )

    assert not output_path.exists()


def test_suryodaya_candidate_does_not_serialize_as_predicate_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tools = complete_goal_one_tools()
    monkeypatch.setattr(hosted_runner, "AgentSwitchClient", lambda config: object())
    monkeypatch.setattr(hosted_runner, "AgentSwitchCalendarTools", lambda client: tools)

    output_path, checks = run_hosted(
        {
            "AGENTSWITCH_INSTANCE": "suryodaya",
            "AGENTSWITCH_BASE_URL": "https://runner.example.test",
            "AGENTSWITCH_TOKEN": "secret-token",
        },
        output_path=tmp_path / "results.json",
        now=NOW,
    )

    document = parse_results(output_path.read_text(encoding="utf-8"))
    task = document["tasks"][0]
    assert checks[0].passed is False
    assert task["passed"] is False
    assert task["score"] == 0.0
    assert task["evidence"].startswith("UNEVALUATED:")
    assert "business_goal_predicates=unevaluated" in document["summary"]


def test_hosted_toml_selects_suryodaya_only() -> None:
    config = Path("agentswitch-harness.toml").read_text(encoding="utf-8")

    assert 'instances = ["suryodaya"]' in config
    assert "keystone" not in config


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
    assert not checks[0].passed
    assert document["tasks"][0]["passed"] is False
    assert document["tasks"][0]["evidence"].startswith("UNEVALUATED:")
    assert "secret-token" not in results_path.read_text(encoding="utf-8")
