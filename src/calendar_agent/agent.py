"""Calendar Agent orchestration without HTTP or mutation knowledge."""

from __future__ import annotations

from .models import AgentResult, AgentTask
from .planner import plan_task
from .tools import CalendarTool


class CalendarAgent:
    def __init__(
        self,
        tools: CalendarTool,
        *,
        allow_disposable_writes: bool = False,
    ) -> None:
        if type(allow_disposable_writes) is not bool:
            raise TypeError("allow_disposable_writes must be a boolean")
        self.tools = tools
        self._allow_disposable_writes = allow_disposable_writes

    def run(self, task: AgentTask) -> AgentResult:
        trace_start = len(self.tools.trace)
        result = plan_task(task, self.tools)
        result.tool_calls.extend(self.tools.trace[trace_start:])
        return result

    def confirm_move(
        self,
        task: AgentTask,
        proposal: AgentResult,
        confirmation_text: str,
    ) -> AgentResult:
        """Apply a previously returned move proposal after a separate confirmation."""

        from .goals.move_after_audit import apply_confirmed_move

        trace_start = len(self.tools.trace)
        result = apply_confirmed_move(
            task,
            proposal,
            self.tools,
            confirmation_text=confirmation_text,
            allow_disposable_writes=self._allow_disposable_writes,
        )
        result.tool_calls.extend(self.tools.trace[trace_start:])
        return result