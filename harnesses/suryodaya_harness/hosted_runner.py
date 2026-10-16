"""Hosted, per-instance read-only checks for AgentSwitch Harness runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from calendar_agent import (
    AgentSwitchCalendarTools,
    AgentSwitchClient,
    AgentSwitchConfig,
    AgentSwitchError,
    AgentTask,
    CalendarTool,
    CalendarAgent,
    ConfigurationError,
    MutationDisabledError,
    ResultStatus,
)
from calendar_agent.models import ToolCallRecord
from calendar_agent.tools import ReadOnlyCalendarTool

from harnesses.keystone_harness import (
    DEFAULT_BASE_URL as KEYSTONE_BASE_URL,
    KeystoneClient,
    KeystoneClientError,
    KeystoneConfig,
    KeystoneConfigurationError,
)

from .assertions import verify_result
from .results import MAX_EVIDENCE_LENGTH, sanitize_evidence, validate_results
from .server_runner import ServerRunnerError, write_results_atomic


SURYOODAYA_BASE_URL = "https://agentswitch.theschoolofai.in"
SURYOODAYA_INSTANCE = "suryodaya"
KEYSTONE_INSTANCE = "keystone"
GOAL_ONE = "calendar.find_30_minutes"
GOAL_ONE_TITLE = "Find 30 minutes with the plant head"
_SEARCH_WINDOW = timedelta(days=14)


class HostedConfigurationError(ValueError):
    """Raised when the runner environment does not identify an approved instance."""


@dataclass(frozen=True)
class HostedEnvironment:
    instance: str
    base_url: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class HostedCheck:
    task_id: str
    title: str
    passed: bool
    evidence: str
    secret_values: tuple[str, ...] = ()

    def official_record(self) -> dict[str, object]:
        evidence = sanitize_evidence(self.evidence, self.secret_values)[:MAX_EVIDENCE_LENGTH]
        return {
            "id": self.task_id,
            "title": self.title,
            "passed": self.passed,
            "score": 1.0 if self.passed else 0.0,
            "evidence": evidence,
        }


def _task(
    *,
    task_id: str,
    title: str,
    passed: bool,
    evidence: str,
    secret_values: tuple[str, ...] = (),
) -> HostedCheck:
    return HostedCheck(
        task_id=task_id,
        title=title,
        passed=passed,
        evidence=evidence,
        secret_values=secret_values,
    )


def load_hosted_environment(values: Mapping[str, object]) -> HostedEnvironment:
    """Read the runner-provided target and enforce each instance's fixed origin."""

    instance_value = values.get("AGENTSWITCH_INSTANCE")
    base_url_value = values.get("AGENTSWITCH_BASE_URL")
    token_value = values.get("AGENTSWITCH_TOKEN")
    if not isinstance(instance_value, str) or not instance_value.strip():
        raise HostedConfigurationError("AGENTSWITCH_INSTANCE must be set.")
    if not isinstance(base_url_value, str) or not base_url_value.strip():
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL must be set.")
    if not isinstance(token_value, str) or not token_value.strip():
        raise HostedConfigurationError("AGENTSWITCH_TOKEN must be set.")

    instance = instance_value.strip().casefold()
    approved_origins = {
        SURYOODAYA_INSTANCE: SURYOODAYA_BASE_URL,
        KEYSTONE_INSTANCE: KEYSTONE_BASE_URL,
    }
    expected_origin = approved_origins.get(instance)
    if expected_origin is None:
        raise HostedConfigurationError("AGENTSWITCH_INSTANCE is not an enabled harness target.")

    try:
        supplied = urlsplit(base_url_value.strip())
        expected = urlsplit(expected_origin)
        port = supplied.port
    except ValueError:
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL is malformed.") from None
    if (
        supplied.scheme.casefold() != "https"
        or supplied.hostname != expected.hostname
        or port not in (None, 443)
        or supplied.path not in ("", "/")
        or supplied.query
        or supplied.fragment
        or supplied.username is not None
        or supplied.password is not None
    ):
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL does not match the selected instance.")

    return HostedEnvironment(
        instance=instance,
        base_url=expected_origin,
        token=token_value.strip(),
    )


def _read_trace_summary(calls: list[ToolCallRecord]) -> str:
    failed_reads = sum(not call.succeeded for call in calls if not call.mutating)
    mutation_attempts = sum(call.mutating for call in calls)
    return (
        f"read_calls={sum(not call.mutating for call in calls)};"
        f"failed_reads={failed_reads};mutation_attempts={mutation_attempts}"
    )


def check_suryodaya_goal_one(
    tools: CalendarTool,
    *,
    now: datetime | None = None,
    secret_values: tuple[str, ...] = (),
) -> HostedCheck:
    """Pass only when the read-only planner returns one structurally valid slot."""

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(timezone.utc).replace(microsecond=0)
    task = AgentTask(
        GOAL_ONE,
        GOAL_ONE_TITLE,
        {
            "now": current,
            "start": current,
            "end": current + _SEARCH_WINDOW,
        },
    )
    read_only_tools = ReadOnlyCalendarTool(tools)
    try:
        result = CalendarAgent(read_only_tools).run(task)
    except (AgentSwitchError, ConfigurationError, MutationDisabledError) as exc:
        trace = _read_trace_summary(tools.trace)
        return _task(
            task_id="suryodaya.goal1_read_only_candidate",
            title="Suryodaya read-only Goal 1 candidate check",
            passed=False,
            evidence=f"UNEVALUATED: planner stopped ({type(exc).__name__}); {trace}.",
            secret_values=secret_values,
        )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)
    if verification.passed:
        evidence = (
            "Read-only planner returned one structurally valid, availability-supported "
            f"30-minute candidate; {_read_trace_summary(result.tool_calls)}."
        )
        return _task(
            task_id="suryodaya.goal1_read_only_candidate",
            title="Suryodaya read-only Goal 1 candidate check",
            passed=True,
            evidence=evidence,
            secret_values=secret_values,
        )

    evidence = (
        f"UNEVALUATED: planner status={result.status.value}; "
        f"candidate_slots={len(result.candidate_slots)}; "
        f"clarifications={len(result.ambiguities)}; "
        f"{_read_trace_summary(result.tool_calls)}."
    )
    return _task(
        task_id="suryodaya.goal1_read_only_candidate",
        title="Suryodaya read-only Goal 1 candidate check",
        passed=False,
        evidence=evidence,
        secret_values=secret_values,
    )


def check_keystone_tool_discovery(
    client: KeystoneClient,
    *,
    secret_values: tuple[str, ...] = (),
) -> HostedCheck:
    """Pass only when Keystone returns a valid, non-empty MCP tool catalog."""

    try:
        tools = client.list_tools()
    except KeystoneClientError as exc:
        evidence = f"UNEVALUATED: Keystone tools/list was unavailable ({type(exc).__name__})."
        return _task(
            task_id="keystone.mcp_tool_discovery",
            title="Keystone MCP tool discovery",
            passed=False,
            evidence=evidence,
            secret_values=secret_values,
        )

    read_only_hints = sum(tool.read_only_hint is True for tool in tools)
    evidence = (
        f"Keystone tools/list returned {len(tools)} tools; "
        f"read_only_hints={read_only_hints}; no business tool was invoked."
    )
    return _task(
        task_id="keystone.mcp_tool_discovery",
        title="Keystone MCP tool discovery",
        passed=bool(tools),
        evidence=evidence if tools else f"UNEVALUATED: {evidence}",
        secret_values=secret_values,
    )


def run_hosted(
    environment: Mapping[str, object],
    *,
    output_path: str | Path,
    now: datetime | None = None,
) -> tuple[Path, tuple[HostedCheck, ...]]:
    """Run the single platform-selected instance and write its result document."""

    raw_token = environment.get("AGENTSWITCH_TOKEN")
    secret_values = (raw_token.strip(),) if isinstance(raw_token, str) and raw_token.strip() else ()
    try:
        config = load_hosted_environment(environment)
    except HostedConfigurationError as exc:
        tasks = (
            _task(
                task_id="harness.configuration",
                title="Hosted harness configuration",
                passed=False,
                evidence=f"UNEVALUATED: {type(exc).__name__}: {exc}",
                secret_values=secret_values,
            ),
        )
        summary = "Hosted harness configuration could not be validated."
    else:
        if config.instance == SURYOODAYA_INSTANCE:
            client_config = AgentSwitchConfig(base_url=config.base_url, token=config.token)
            tools = AgentSwitchCalendarTools(AgentSwitchClient(client_config))
            tasks = (
                check_suryodaya_goal_one(
                    tools,
                    now=now,
                    secret_values=secret_values,
                ),
            )
        else:
            try:
                keystone_config = KeystoneConfig(
                    base_url=config.base_url,
                    token=config.token,
                )
                client = KeystoneClient(keystone_config)
            except KeystoneConfigurationError as exc:
                tasks = (
                    _task(
                        task_id="keystone.mcp_tool_discovery",
                        title="Keystone MCP tool discovery",
                        passed=False,
                        evidence=f"UNEVALUATED: Keystone configuration failed ({type(exc).__name__}).",
                        secret_values=secret_values,
                    ),
                )
            else:
                tasks = (
                    check_keystone_tool_discovery(client, secret_values=secret_values),
                )
        passed_count = sum(task.passed for task in tasks)
        summary = f"{config.instance}: {passed_count}/{len(tasks)} harness checks passed."

    document = {
        "tasks": [task.official_record() for task in tasks],
        "summary": summary,
    }
    validate_results(document)
    results_json = json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False)
    written_path = write_results_atomic(output_path, results_json)
    return written_path, tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harnesses.suryodaya_harness.hosted_runner")
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    args = parser.parse_args(argv)
    try:
        output_path, tasks = run_hosted(os.environ, output_path=args.results)
    except (OSError, ServerRunnerError, TypeError, ValueError) as exc:
        print(f"Hosted harness error: {type(exc).__name__}.")
        return 1

    passed_count = sum(task.passed for task in tasks)
    print(f"Hosted harness wrote {output_path}; checks_passed={passed_count}/{len(tasks)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
