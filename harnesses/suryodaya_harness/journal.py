"""Pre-score task journals for hosted harness executions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from calendar_agent.models import AgentResult, ToolCallRecord

from .results import sanitize_evidence
from .server_runner import write_results_atomic


_OUTCOME_FIELDS = frozenset(
    {
        "alternative_shift_proposed",
        "confirmation_required",
        "moved",
        "planned_event_count",
        "requested_shift",
        "shift",
        "shift_was_proposed",
        "updates_attempted",
        "updates_completed",
        "updates_requested",
        "updates_verified",
        "verification_passed",
    }
)


class HostedRunJournal:
    """Write safe execution evidence before the harness scores each task."""

    def __init__(
        self,
        root: str | Path,
        *,
        instance: str,
        run_id: str | None = None,
        secret_values: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(instance, str) or not instance.strip():
            raise ValueError("instance must be non-empty text")
        self.run_id = run_id or str(uuid4())
        self.instance = instance
        self._root = Path(root) / self.run_id / instance
        self._secret_values = secret_values
        self._paths: dict[str, Path] = {}
        self._execution_recorded: set[str] = set()

    def begin(self, *, task_id: str, title: str, goal: str) -> Path:
        """Persist task identity before any predicate can score the task."""

        if not all(isinstance(value, str) and value.strip() for value in (task_id, title, goal)):
            raise ValueError("task_id, title, and goal must be non-empty text")
        if any(separator in task_id for separator in ("/", "\\")):
            raise ValueError("task_id must not contain path separators")
        if task_id in self._paths:
            raise ValueError(f"task journal already exists for {task_id}")

        path = self._root / f"{task_id}.json"
        self._paths[task_id] = path
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "instance": self.instance,
            "task_id": task_id,
            "title": title,
            "goal": goal,
            "phase": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(path, payload)
        return path

    def record_observation(
        self,
        task_id: str,
        *,
        observation: dict[str, object],
        tool_calls: tuple[ToolCallRecord, ...] | list[ToolCallRecord] = (),
    ) -> None:
        """Persist a compact backend observation before a discovery predicate."""

        payload = self._load(task_id)
        payload["observation"] = observation
        payload["tool_calls"] = self._tool_calls(tool_calls)
        payload["phase"] = "execution_recorded"
        payload["execution_recorded_at"] = datetime.now(timezone.utc).isoformat()
        self._execution_recorded.add(task_id)
        self._write(self._paths[task_id], payload)

    def record_execution(
        self,
        task_id: str,
        *,
        result: AgentResult | None,
        tool_calls: tuple[ToolCallRecord, ...] | list[ToolCallRecord],
        error_category: str | None = None,
    ) -> None:
        """Persist structured agent output and tool metadata before scoring."""

        payload = self._load(task_id)
        payload["agent_result"] = self._agent_result(result)
        payload["tool_calls"] = self._tool_calls(tool_calls)
        if error_category is not None:
            payload["execution_error_category"] = sanitize_evidence(
                error_category,
                self._secret_values,
            )
        payload["phase"] = "execution_recorded"
        payload["execution_recorded_at"] = datetime.now(timezone.utc).isoformat()
        self._execution_recorded.add(task_id)
        self._write(self._paths[task_id], payload)

    def finish(self, check: object) -> None:
        """Attach the separately computed task predicate to its execution journal."""

        task_id = getattr(check, "task_id", None)
        if not isinstance(task_id, str) or task_id not in self._paths:
            raise ValueError("check has no matching task journal")
        if task_id not in self._execution_recorded:
            raise RuntimeError(f"task {task_id} was scored before execution evidence was persisted")
        official_record = getattr(check, "official_record", None)
        if not callable(official_record):
            raise TypeError("check must provide official_record()")
        payload = self._load(task_id)
        payload["phase"] = "scored"
        payload["scored_at"] = datetime.now(timezone.utc).isoformat()
        payload["check"] = official_record()
        self._write(self._paths[task_id], payload)

    def path_for(self, task_id: str) -> Path:
        """Return the persisted path for one task in this run."""

        try:
            return self._paths[task_id]
        except KeyError:
            raise KeyError(f"no journal exists for task {task_id}") from None

    def has_execution_record(self, task_id: str) -> bool:
        return task_id in self._execution_recorded

    def _load(self, task_id: str) -> dict[str, Any]:
        path = self.path_for(task_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not reload task journal for {task_id}") from exc
        if not isinstance(payload, dict) or payload.get("task_id") != task_id:
            raise RuntimeError(f"task journal identity mismatch for {task_id}")
        return payload

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
        write_results_atomic(path, serialized)

    @staticmethod
    def _tool_calls(
        tool_calls: tuple[ToolCallRecord, ...] | list[ToolCallRecord],
    ) -> list[dict[str, object]]:
        return [
            {
                "tool": call.tool,
                "succeeded": call.succeeded,
                "mutating": call.mutating,
                "error_category": call.error,
            }
            for call in tool_calls
        ]

    @staticmethod
    def _agent_result(result: AgentResult | None) -> dict[str, object] | None:
        if result is None:
            return None
        outcome: dict[str, object] = {}
        for key in _OUTCOME_FIELDS:
            value = result.claimed_outcome.get(key)
            if isinstance(value, timedelta):
                outcome[f"{key}_seconds"] = value.total_seconds()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                if key in result.claimed_outcome:
                    outcome[key] = value
        reference = result.audit_reference
        return {
            "goal": result.goal,
            "status": result.status.value,
            "candidate_slots": [
                {
                    "start_at": slot.start_at,
                    "end_at": slot.end_at,
                    "timezone": slot.timezone,
                    "availability_confirmed": slot.availability_confirmed,
                }
                for slot in result.candidate_slots
            ],
            "audit_candidates": (
                [
                    {
                        "id": candidate.id,
                        "calendar_id": candidate.calendar_id,
                        "start_at": candidate.start_at,
                        "end_at": candidate.end_at,
                        "status": candidate.status,
                        "timezone": candidate.timezone,
                    }
                    for candidate in reference.candidates
                ]
                if reference is not None
                else []
            ),
            "moves": [
                {
                    "event_id": move.event_id,
                    "original_start_at": move.original_start_at,
                    "original_end_at": move.original_end_at,
                    "proposed_start_at": move.proposed_start_at,
                    "proposed_end_at": move.proposed_end_at,
                }
                for move in result.rescheduling_plan
            ],
            "claimed_outcome": outcome,
        }
