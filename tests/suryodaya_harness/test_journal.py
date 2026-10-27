import json

from calendar_agent.models import AgentResult, ResultStatus, ToolCallRecord
from harnesses.suryodaya_harness.hosted_runner import HostedCheck
from harnesses.suryodaya_harness.journal import HostedRunJournal


def test_hosted_journal_persists_execution_before_score_and_redacts_call_arguments(
    tmp_path,
) -> None:
    journal = HostedRunJournal(
        tmp_path,
        instance="suryodaya",
        run_id="run-1",
    )
    path = journal.begin(
        task_id="suryodaya.goal1_read_only_candidate",
        title="Goal 1 candidate",
        goal="calendar.find_30_minutes",
    )

    journal.record_execution(
        "suryodaya.goal1_read_only_candidate",
        result=AgentResult(
            goal="calendar.find_30_minutes",
            status=ResultStatus.NEEDS_CLARIFICATION,
            summary="The private detail is not persisted.",
        ),
        tool_calls=[
            ToolCallRecord(
                "Party.list",
                {"email": "plant-head@example.test"},
            )
        ],
    )

    execution_record = json.loads(path.read_text(encoding="utf-8"))
    serialized = path.read_text(encoding="utf-8")
    assert execution_record["phase"] == "execution_recorded"
    assert execution_record["agent_result"]["status"] == "needs_clarification"
    assert "summary" not in execution_record["agent_result"]
    assert "arguments" not in execution_record["tool_calls"][0]
    assert "plant-head@example.test" not in serialized
    assert journal.has_execution_record("suryodaya.goal1_read_only_candidate")

    journal.finish(
        HostedCheck(
            task_id="suryodaya.goal1_read_only_candidate",
            title="Goal 1 candidate",
            passed=False,
            evidence="UNEVALUATED: no candidate.",
        )
    )

    scored_record = json.loads(path.read_text(encoding="utf-8"))
    assert scored_record["phase"] == "scored"
    assert scored_record["check"]["passed"] is False
