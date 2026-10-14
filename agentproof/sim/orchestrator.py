"""Orchestration and simulation engine for ChemDesk dual-control benchmarks."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from agentproof.domain.db import DatabaseManager
from agentproof.sim.models import (
    EpisodeStatus,
    EpisodeTrajectory,
    SimulationMode,
    ToolCallRecord,
    TurnRecord,
    TurnRole,
)
from agentproof.tasks.task import Task
from agentproof.tools import AGENT_TOOLS, ALL_TOOLS, USER_TOOLS
from agentproof.tools.base import ToolResult

logger = logging.getLogger(__name__)

USER_TOOL_NAMES = {
    "check_site_access",
    "confirm_delivery_window",
    "report_receiver_on_site",
}
AGENT_TOOL_NAMES = set(AGENT_TOOLS.keys())
ALL_TOOL_NAMES = set(ALL_TOOLS.keys())

TERMINATION_TOKENS = ["###DONE###", "###TRANSFER###"]


class Orchestrator:
    """Manages multi-turn evaluation episodes across dual, no_user, and oracle_plan modes."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        output_dir: Union[str, Path] = "runs",
        seed_dir: Optional[str] = None,
        save_to_disk: bool = True,
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"run_{timestamp}_{uuid.uuid4().hex[:6]}"
        self.output_dir = Path(output_dir)
        self.seed_dir = seed_dir
        self.save_to_disk = save_to_disk

    def _apply_db_setup(self, db: DatabaseManager, task: Task) -> None:
        """Apply initial database configuration or test setup for the task."""
        if not task.initial_db_setup:
            return

        setup = task.initial_db_setup
        if callable(setup):
            setup(db)
        elif isinstance(setup, str):
            db.conn.executescript(setup)
            db.conn.commit()
        elif isinstance(setup, list):
            for item in setup:
                if isinstance(item, str):
                    db.conn.executescript(item)
                    db.conn.commit()
                elif isinstance(item, dict):
                    db.seed_from_data(item)
        elif isinstance(setup, dict):
            db.seed_from_data(setup)

    def _execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        caller: str,
        mode: SimulationMode,
        db: DatabaseManager,
        call_id: Optional[str] = None,
    ) -> ToolCallRecord:
        """Execute a tool with strict dual-control permission enforcement."""
        cid = call_id or f"call_{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()

        # Dual-control permission barrier checks
        if mode in (SimulationMode.DUAL, SimulationMode.ORACLE_PLAN):
            if caller == "agent":
                if tool_name in USER_TOOL_NAMES:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    denied_result = ToolResult(
                        is_error=True,
                        error_message="Permission Denied: User tool restricted to site personnel",
                    )
                    return ToolCallRecord(
                        call_id=cid,
                        tool_name=tool_name,
                        caller=caller,
                        arguments=arguments,
                        result=denied_result.to_dict(),
                        is_error=True,
                        latency_ms=latency_ms,
                    )
                if tool_name not in AGENT_TOOL_NAMES:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    denied_result = ToolResult(
                        is_error=True,
                        error_message=f"Permission Denied: Unknown or forbidden tool '{tool_name}' for agent",
                    )
                    return ToolCallRecord(
                        call_id=cid,
                        tool_name=tool_name,
                        caller=caller,
                        arguments=arguments,
                        result=denied_result.to_dict(),
                        is_error=True,
                        latency_ms=latency_ms,
                    )
            elif caller == "user":
                if tool_name in AGENT_TOOL_NAMES:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    denied_result = ToolResult(
                        is_error=True,
                        error_message="Permission Denied: Agent tool restricted to chemical order desk agent",
                    )
                    return ToolCallRecord(
                        call_id=cid,
                        tool_name=tool_name,
                        caller=caller,
                        arguments=arguments,
                        result=denied_result.to_dict(),
                        is_error=True,
                        latency_ms=latency_ms,
                    )
                if tool_name not in USER_TOOL_NAMES:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    denied_result = ToolResult(
                        is_error=True,
                        error_message=f"Permission Denied: Unknown or forbidden tool '{tool_name}' for user",
                    )
                    return ToolCallRecord(
                        call_id=cid,
                        tool_name=tool_name,
                        caller=caller,
                        arguments=arguments,
                        result=denied_result.to_dict(),
                        is_error=True,
                        latency_ms=latency_ms,
                    )
        elif mode == SimulationMode.NO_USER:
            # In no_user mode, agent is equipped with all 17 tools
            if tool_name not in ALL_TOOL_NAMES:
                latency_ms = (time.perf_counter() - t0) * 1000
                denied_result = ToolResult(
                    is_error=True,
                    error_message=f"Unknown tool '{tool_name}'",
                )
                return ToolCallRecord(
                    call_id=cid,
                    tool_name=tool_name,
                    caller=caller,
                    arguments=arguments,
                    result=denied_result.to_dict(),
                    is_error=True,
                    latency_ms=latency_ms,
                )

        tool_def = ALL_TOOLS.get(tool_name)
        if not tool_def:
            latency_ms = (time.perf_counter() - t0) * 1000
            err_result = ToolResult(is_error=True, error_message=f"Tool definition not found for '{tool_name}'")
            return ToolCallRecord(
                call_id=cid,
                tool_name=tool_name,
                caller=caller,
                arguments=arguments,
                result=err_result.to_dict(),
                is_error=True,
                latency_ms=latency_ms,
            )

        try:
            exec_result = tool_def.execute(db=db, **arguments)
            latency_ms = (time.perf_counter() - t0) * 1000
            return ToolCallRecord(
                call_id=cid,
                tool_name=tool_name,
                caller=caller,
                arguments=arguments,
                result=exec_result.to_dict(),
                is_error=bool(exec_result.is_error or not exec_result.success),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            err_result = ToolResult(is_error=True, error_message=str(exc))
            return ToolCallRecord(
                call_id=cid,
                tool_name=tool_name,
                caller=caller,
                arguments=arguments,
                result=err_result.to_dict(),
                is_error=True,
                latency_ms=latency_ms,
            )

    def _resolve_trace_id(self, task_id: str, mode: SimulationMode, episode_id: str) -> str:
        """Resolve OpenTelemetry trace ID if active or generate deterministic trace ID."""
        try:
            from opentelemetry import trace
            current_span = trace.get_current_span()
            if current_span:
                span_ctx = current_span.get_span_context()
                if span_ctx and span_ctx.is_valid:
                    return f"{span_ctx.trace_id:032x}"
        except Exception:
            pass

        return uuid.uuid5(uuid.NAMESPACE_DNS, f"{task_id}:{mode.value}:{episode_id}").hex

    def _save_trajectory(self, trajectory: EpisodeTrajectory) -> Path:
        """Save episode trajectory JSON to runs/<run_id>/<task_id>_<mode>.json."""
        target_dir = self.output_dir / self.run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{trajectory.task_id}_{trajectory.mode.value}.json"
        file_path.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
        return file_path

    def _contains_termination(self, text: str) -> bool:
        """Check if message content contains termination tokens."""
        if not text:
            return False
        return any(token in text for token in TERMINATION_TOKENS)

    def _invoke_entity(
        self,
        entity: Any,
        history: List[Dict[str, Any]],
        role: str,
        context: Dict[str, Any],
        db: DatabaseManager,
        mode: SimulationMode,
    ) -> Tuple[str, List[ToolCallRecord], float, int, int, bool]:
        """Invoke agent or user entity and extract response payload, tool calls, and metrics."""
        tool_records: List[ToolCallRecord] = []

        # Tool executor callback for agents that execute tools interactively
        def tool_executor(tool_name: str, **kwargs: Any) -> ToolResult:
            record = self._execute_tool_call(
                tool_name=tool_name,
                arguments=kwargs,
                caller=role,
                mode=mode,
                db=db,
            )
            tool_records.append(record)
            if record.is_error:
                err_msg = record.result.get("error") if isinstance(record.result, dict) else str(record.result)
                return ToolResult(is_error=True, error_message=err_msg)
            data_val = record.result.get("data") if isinstance(record.result, dict) else record.result
            return ToolResult(success=True, data=data_val)

        merged_context = {**context, "tool_executor": tool_executor}

        # Dispatch step invocation across different interface styles
        if hasattr(entity, "step"):
            step_fn = getattr(entity, "step")
        elif hasattr(entity, "act"):
            step_fn = getattr(entity, "act")
        elif callable(entity):
            step_fn = entity
        else:
            raise TypeError(f"Entity of type {type(entity)} does not implement step, act, or __call__")

        try:
            # Attempt invocation with history and context, falling back if not accepted
            try:
                response = step_fn(history=history, context=merged_context)
            except TypeError:
                try:
                    response = step_fn(history=history)
                except TypeError:
                    response = step_fn()
        except Exception as exc:
            logger.error("Error executing %s entity step: %s", role, exc)
            return f"Error: {exc}", tool_records, 0.0, 0, 0, True

        # Extract content, tool calls, tokens, and cost from response
        content = ""
        cost_usd = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        is_done = False
        declared_tool_calls: List[Dict[str, Any]] = []

        if isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            content = response.get("content", "")
            declared_tool_calls = response.get("tool_calls", [])
            cost_usd = float(response.get("cost_usd", 0.0))
            prompt_tokens = int(response.get("prompt_tokens", 0))
            completion_tokens = int(response.get("completion_tokens", 0))
            is_done = bool(response.get("is_done", False))
        else:
            content = getattr(response, "content", "")
            declared_tool_calls = getattr(response, "tool_calls", [])
            cost_usd = float(getattr(response, "cost_usd", 0.0))
            prompt_tokens = int(getattr(response, "prompt_tokens", 0))
            completion_tokens = int(getattr(response, "completion_tokens", 0))
            is_done = bool(getattr(response, "is_done", False))

        # Process any tool calls returned in the response object
        for call in declared_tool_calls:
            c_dict = dict(call)
            t_name = c_dict.get("tool_name") or c_dict.get("name") or ""
            t_args = c_dict.get("arguments") or c_dict.get("args") or {}
            if isinstance(t_args, str):
                try:
                    t_args = json.loads(t_args)
                except Exception:
                    t_args = {"raw": t_args}
            t_id = c_dict.get("call_id") or c_dict.get("id")

            record = self._execute_tool_call(
                tool_name=t_name,
                arguments=t_args,
                caller=role,
                mode=mode,
                db=db,
                call_id=t_id,
            )
            tool_records.append(record)

        return content, tool_records, cost_usd, prompt_tokens, completion_tokens, is_done

    def run_episode(
        self,
        task: Task,
        agent: Any,
        user_sim: Optional[Any] = None,
        mode: Union[SimulationMode, str] = SimulationMode.DUAL,
        max_turns: int = 15,
    ) -> EpisodeTrajectory:
        """Run an isolated evaluation episode and return the complete trajectory."""
        if isinstance(mode, str):
            mode = SimulationMode(mode)

        episode_start_time = time.perf_counter()
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        trace_id = self._resolve_trace_id(task.task_id, mode, episode_id)

        agent_name = (
            getattr(agent, "name", None)
            or getattr(agent, "agent_name", None)
            or agent.__class__.__name__
        )

        # 1. Initialize isolated database seeded from domain/seed
        db = DatabaseManager.create_isolated_db(seed_dir=self.seed_dir)
        try:
            self._apply_db_setup(db, task)
            initial_snapshot = db.get_snapshot()

            # 2. Context setup
            available_agent_tools = (
                list(ALL_TOOLS.keys())
                if mode == SimulationMode.NO_USER
                else list(AGENT_TOOLS.keys())
            )
            agent_context = {
                "task": task,
                "mode": mode,
                "tools": available_agent_tools,
                "gold_plan": task.gold_plan if mode == SimulationMode.ORACLE_PLAN else None,
            }
            user_context = {
                "task": task,
                "mode": mode,
                "tools": list(USER_TOOLS.keys()),
            }

            turns: List[TurnRecord] = []
            history: List[Dict[str, Any]] = []

            total_cost_usd = 0.0
            total_prompt_tokens = 0
            total_completion_tokens = 0
            terminal_status = EpisodeStatus.COMPLETED

            # 3. Turn-taking loop
            if mode == SimulationMode.NO_USER:
                # No user turns. Agent operates autonomously with all 17 tools and full task goal.
                history.append({"speaker": TurnRole.SYSTEM.value, "content": task.goal})

                for turn_idx in range(max_turns):
                    turn_t0 = time.perf_counter()
                    content, tool_recs, cost, p_tokens, c_tokens, is_done = self._invoke_entity(
                        entity=agent,
                        history=history,
                        role="agent",
                        context=agent_context,
                        db=db,
                        mode=mode,
                    )
                    turn_dur = (time.perf_counter() - turn_t0) * 1000
                    total_cost_usd += cost
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens

                    turn_record = TurnRecord(
                        turn_index=turn_idx,
                        speaker=TurnRole.AGENT.value,
                        content=content,
                        tool_calls=tool_recs,
                        duration_ms=turn_dur,
                    )
                    turns.append(turn_record)
                    history.append({"speaker": TurnRole.AGENT.value, "content": content})

                    if self._contains_termination(content) or is_done:
                        terminal_status = EpisodeStatus.COMPLETED
                        break
                else:
                    terminal_status = EpisodeStatus.MAX_TURNS_EXCEEDED

            else:
                # Dual or Oracle Plan mode: alternating Agent and User
                turn_idx = 0

                # User opening turn
                user_opening_content = task.initial_user_message or task.goal
                turn_t0 = time.perf_counter()

                if task.initial_user_message or user_sim is None:
                    # User states the initial message directly
                    turn_dur = (time.perf_counter() - turn_t0) * 1000
                    turn_record = TurnRecord(
                        turn_index=turn_idx,
                        speaker=TurnRole.USER.value,
                        content=user_opening_content,
                        tool_calls=[],
                        duration_ms=turn_dur,
                    )
                    turns.append(turn_record)
                    history.append({"speaker": TurnRole.USER.value, "content": user_opening_content})
                    turn_idx += 1
                else:
                    # Invoke user simulator for turn 0
                    content, tool_recs, cost, p_tokens, c_tokens, is_done = self._invoke_entity(
                        entity=user_sim,
                        history=history,
                        role="user",
                        context=user_context,
                        db=db,
                        mode=mode,
                    )
                    turn_dur = (time.perf_counter() - turn_t0) * 1000
                    total_cost_usd += cost
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens

                    turn_record = TurnRecord(
                        turn_index=turn_idx,
                        speaker=TurnRole.USER.value,
                        content=content,
                        tool_calls=tool_recs,
                        duration_ms=turn_dur,
                    )
                    turns.append(turn_record)
                    history.append({"speaker": TurnRole.USER.value, "content": content})
                    turn_idx += 1

                # Alternating conversation turns
                while turn_idx < max_turns:
                    # Next speaker: agent if previous was user, else user
                    last_speaker = turns[-1].speaker
                    current_speaker = "agent" if last_speaker == TurnRole.USER.value else "user"

                    turn_t0 = time.perf_counter()
                    if current_speaker == "agent":
                        content, tool_recs, cost, p_tokens, c_tokens, is_done = self._invoke_entity(
                            entity=agent,
                            history=history,
                            role="agent",
                            context=agent_context,
                            db=db,
                            mode=mode,
                        )
                    else:
                        if user_sim is not None:
                            content, tool_recs, cost, p_tokens, c_tokens, is_done = self._invoke_entity(
                                entity=user_sim,
                                history=history,
                                role="user",
                                context=user_context,
                                db=db,
                                mode=mode,
                            )
                        else:
                            content = "Acknowledged. ###DONE###"
                            tool_recs = []
                            cost, p_tokens, c_tokens = 0.0, 0, 0
                            is_done = True

                    turn_dur = (time.perf_counter() - turn_t0) * 1000
                    total_cost_usd += cost
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens

                    turn_record = TurnRecord(
                        turn_index=turn_idx,
                        speaker=current_speaker,
                        content=content,
                        tool_calls=tool_recs,
                        duration_ms=turn_dur,
                    )
                    turns.append(turn_record)
                    history.append({"speaker": current_speaker, "content": content})
                    turn_idx += 1

                    if self._contains_termination(content) or is_done:
                        terminal_status = EpisodeStatus.COMPLETED
                        break
                else:
                    terminal_status = EpisodeStatus.MAX_TURNS_EXCEEDED

            # 4. Final snapshot and DB diff
            final_snapshot = db.get_snapshot()
            db_diff = db.diff_snapshots(initial_snapshot, final_snapshot)

        finally:
            db.close()

        total_duration_ms = (time.perf_counter() - episode_start_time) * 1000

        trajectory = EpisodeTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            mode=mode,
            agent_name=agent_name,
            status=terminal_status,
            turns=turns,
            db_diff=db_diff,
            total_cost_usd=round(total_cost_usd, 6),
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            duration_ms=round(total_duration_ms, 2),
            trace_id=trace_id,
            metadata={
                **task.metadata,
                "category": task.category,
                "turns_count": len(turns),
            },
        )

        if self.save_to_disk:
            self._save_trajectory(trajectory)

        return trajectory


__all__ = ["Orchestrator"]
