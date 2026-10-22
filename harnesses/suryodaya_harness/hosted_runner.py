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
    AgentSwitchError,
    AgentSwitchCalendarTools,
    AgentSwitchClient,
    AgentSwitchConfig,
    AgentTask,
    CalendarTool,
    CalendarAgent,
    ConfigurationError,
    MutationDisabledError,
    ResultStatus,
)
from calendar_agent.models import ToolCallRecord
from calendar_agent.tools import ReadOnlyCalendarTool

from .assertions import verify_result
from .results import MAX_EVIDENCE_LENGTH, sanitize_evidence, validate_results
from .server_runner import ServerRunnerError, write_results_atomic


SURYOODAYA_INSTANCE = "suryodaya"
GOAL_ONE = "calendar.find_30_minutes"
GOAL_ONE_TITLE = "Find 30 minutes with the plant head"
GOAL_TWO = "calendar.move_after_audit"
GOAL_TWO_TITLE = "Move events after the confirmed press shop and welding audit"
GOAL_TWO_AUDIT_NUMBER = "EVT-2026-00007"
GOAL_TWO_AUDIT_TITLE = "Internal quality audit — press shop and welding"
GOAL_TWO_DEFAULT_SHIFT = timedelta(minutes=30)
GOAL_TWO_TIMESTAMP_FALLBACK = "Asia/Kolkata"
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
    """Read the runner-provided target without substituting a local origin."""

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
    if instance != SURYOODAYA_INSTANCE:
        raise HostedConfigurationError("Only the Suryodaya harness target is enabled.")

    base_url = base_url_value.strip()
    if any(character.isspace() or ord(character) < 32 for character in base_url):
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL must not contain whitespace or control characters.")
    try:
        supplied = urlsplit(base_url)
        supplied.port
    except ValueError:
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL is malformed.") from None
    if supplied.scheme.casefold() != "https" or not supplied.hostname:
        raise HostedConfigurationError(
            "AGENTSWITCH_BASE_URL must be an absolute HTTPS instance base URL."
        )
    if supplied.username is not None or supplied.password is not None:
        raise HostedConfigurationError(
            "AGENTSWITCH_BASE_URL must be a credential-free HTTPS instance base URL."
        )
    if supplied.query or supplied.fragment:
        raise HostedConfigurationError("AGENTSWITCH_BASE_URL must not contain a query or fragment.")
    if supplied.path.rstrip("/").casefold().endswith("/api/mcp"):
        raise HostedConfigurationError(
            "AGENTSWITCH_BASE_URL must be the instance base URL, not the /api/mcp endpoint."
        )

    return HostedEnvironment(
        instance=instance,
        base_url=base_url.rstrip("/"),
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
    """Record a read-only candidate check without claiming a goal predicate pass."""

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
            passed=False,
            evidence=(
                "UNEVALUATED: "
                + evidence
                + " No authoritative business-goal predicate was available."
            ),
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


def check_suryodaya_goal_two(
    tools: CalendarTool,
    *,
    secret_values: tuple[str, ...] = (),
) -> HostedCheck:
    """Verify an exact-reference, read-only move plan that awaits confirmation."""

    task = AgentTask(
        GOAL_TWO,
        GOAL_TWO_TITLE,
        {
            "audit_number": GOAL_TWO_AUDIT_NUMBER,
            "audit_title": GOAL_TWO_AUDIT_TITLE,
            "shift": GOAL_TWO_DEFAULT_SHIFT,
            "naive_timestamp_timezone": GOAL_TWO_TIMESTAMP_FALLBACK,
        },
    )
    try:
        result = CalendarAgent(ReadOnlyCalendarTool(tools)).run(task)
    except (AgentSwitchError, ConfigurationError, MutationDisabledError) as exc:
        return _task(
            task_id="suryodaya.goal2_read_only_plan",
            title="Suryodaya read-only Goal 2 plan",
            passed=False,
            evidence=(
                "UNEVALUATED: Goal 2 read-only planning stopped "
                f"({type(exc).__name__}); {_read_trace_summary(tools.trace)}."
            ),
            secret_values=secret_values,
        )

    trace = _read_trace_summary(result.tool_calls)
    if result.status is not ResultStatus.PLANNED:
        details = " ".join((result.summary, *result.ambiguities))
        return _task(
            task_id="suryodaya.goal2_read_only_plan",
            title="Suryodaya read-only Goal 2 plan",
            passed=False,
            evidence=(
                f"Goal 2 did not produce a plan ({result.status.value}); "
                f"moves=0; {details}; {trace}."
            ),
            secret_values=secret_values,
        )

    verification = verify_result(result, expected_status=ResultStatus.PLANNED)
    reference = result.audit_reference
    exact_reference = (
        reference is not None
        and len(reference.candidates) == 1
        and reference.candidates[0].id == GOAL_TWO_AUDIT_NUMBER
        and reference.candidates[0].title == GOAL_TWO_AUDIT_TITLE
        and isinstance(reference.candidates[0].status, str)
        and reference.candidates[0].status.casefold() == "confirmed"
        and isinstance(reference.candidates[0].calendar_id, str)
        and bool(reference.candidates[0].calendar_id.strip())
    )
    outcome = result.claimed_outcome
    confirmation_required = (
        isinstance(outcome, dict)
        and outcome.get("confirmation_required") is True
        and isinstance(outcome.get("confirmation_prompt"), str)
        and bool(outcome["confirmation_prompt"].strip())
    )
    mutation_free = not any(call.mutating for call in result.tool_calls)
    move_summary = ";".join(
        f"{move.event_id}:{move.proposed_start_at}>{move.proposed_end_at}"
        for move in result.rescheduling_plan
    )
    evidence = (
        f"audit={GOAL_TWO_AUDIT_NUMBER};calendar="
        f"{reference.candidates[0].calendar_id if exact_reference else 'unverified'};"
        f"moves={len(result.rescheduling_plan)};shift=30m;"
        f"confirmation_required={str(confirmation_required).lower()};"
        "confirm_or_change_shift=true;"
        f"mutation_attempts={sum(call.mutating for call in result.tool_calls)};"
        f"plan={move_summary}"
    )
    passed = (
        verification.passed
        and exact_reference
        and confirmation_required
        and isinstance(outcome, dict)
        and outcome.get("moved") is False
        and outcome.get("shift") == GOAL_TWO_DEFAULT_SHIFT
        and mutation_free
        and bool(result.rescheduling_plan)
    )
    if len(evidence) > MAX_EVIDENCE_LENGTH:
        passed = False
        evidence = (
            "UNEVALUATED: Goal 2 plan is too large to fit the results evidence "
            "limit; a complete plan was not serialized."
        )
    return _task(
        task_id="suryodaya.goal2_read_only_plan",
        title="Suryodaya read-only Goal 2 plan",
        passed=passed,
        evidence=evidence,
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
        summary = (
            "Hosted harness configuration could not be validated; "
            "goal_tasks_not_run=true; official_evaluator=not_invoked."
        )
    else:
        if config.instance == SURYOODAYA_INSTANCE:
            try:
                client_config = AgentSwitchConfig(base_url=config.base_url, token=config.token)
                tools = AgentSwitchCalendarTools(AgentSwitchClient(client_config))
                tasks = (
                    check_suryodaya_goal_one(
                        tools,
                        now=now,
                        secret_values=secret_values,
                    ),
                    check_suryodaya_goal_two(
                        tools,
                        secret_values=secret_values,
                    ),
                )
            except (AgentSwitchError, ConfigurationError, MutationDisabledError) as exc:
                tasks = (
                    _task(
                        task_id="harness.execution",
                        title="Suryodaya harness execution",
                        passed=False,
                        evidence=(
                            "UNEVALUATED: hosted harness initialization or evaluation "
                            f"stopped ({type(exc).__name__})."
                        ),
                        secret_values=secret_values,
                    ),
                )
        else:
            raise AssertionError("load_hosted_environment must reject non-Suryodaya instances")
        passed_count = sum(task.passed for task in tasks)
        summary = (
            f"{config.instance}: {passed_count}/{len(tasks)} harness checks passed; "
            "official_evaluator=not_invoked."
        )

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
    print(
        f"Hosted harness wrote {output_path}; harness_checks_passed="
        f"{passed_count}/{len(tasks)}; official_evaluator=not_invoked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
