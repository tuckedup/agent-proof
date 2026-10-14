"""Unit and integration tests for Orchestrator, Simulation Modes, and ScriptedAgent."""

import json
from pathlib import Path
from typing import Any, Dict

from agentproof.sim.mock_agent import ScriptedAction, ScriptedAgent
from agentproof.sim.models import (
    EpisodeStatus,
    EpisodeTrajectory,
    SimulationMode,
    Task,
    ToolCallRecord,
    TurnRecord,
    TurnRole,
)
from agentproof.sim.orchestrator import Orchestrator


def test_simulation_models_serialization() -> None:
    """Verify Pydantic v2 serialization and deserialization of simulation records."""
    tool_rec = ToolCallRecord(
        call_id="call_test_01",
        tool_name="find_account",
        caller="agent",
        arguments={"query": "Apex"},
        result={"success": True, "data": {"count": 1}},
        is_error=False,
        latency_ms=12.5,
    )
    assert tool_rec.tool_name == "find_account"
    assert tool_rec.is_error is False

    turn = TurnRecord(
        turn_index=0,
        speaker=TurnRole.AGENT,
        content="I am looking up the Apex account.",
        tool_calls=[tool_rec],
        duration_ms=45.0,
    )
    assert turn.turn_index == 0
    assert len(turn.tool_calls) == 1

    traj = EpisodeTrajectory(
        episode_id="ep_test_01",
        task_id="TASK-001",
        mode=SimulationMode.DUAL,
        agent_name="test_agent",
        status=EpisodeStatus.COMPLETED,
        turns=[turn],
        db_diff={"orders": {"added": [{"order_id": "ORD-NEW-01"}], "removed": []}},
        total_cost_usd=0.002,
        total_prompt_tokens=150,
        total_completion_tokens=60,
        duration_ms=120.0,
        trace_id="trace_test_123",
        metadata={"category": "order_creation"},
    )

    data = traj.model_dump()
    assert data["episode_id"] == "ep_test_01"
    assert data["mode"] == "dual"
    assert data["status"] == "completed"

    json_str = traj.model_dump_json()
    parsed = EpisodeTrajectory.model_validate_json(json_str)
    assert parsed.task_id == "TASK-001"
    assert len(parsed.turns) == 1
    assert parsed.turns[0].tool_calls[0].arguments == {"query": "Apex"}


def test_scripted_agent() -> None:
    """Verify ScriptedAgent deterministic steps, history tracking, and callable interface."""
    actions = [
        ScriptedAction(content="Turn 1 hello", tool_calls=[{"tool_name": "find_account", "arguments": {"query": "Apex"}}]),
        {"content": "Turn 2 done ###DONE###", "tool_calls": []},
    ]
    agent = ScriptedAgent(name="test_scripted", actions=actions)

    step1 = agent.step(history=[])
    assert step1.content == "Turn 1 hello"
    assert len(step1.tool_calls) == 1
    assert step1.tool_calls[0]["tool_name"] == "find_account"

    step2 = agent(history=[{"content": "Turn 1 hello"}])
    assert "###DONE###" in step2.content

    # Next step after exhausting actions
    step3 = agent.step()
    assert step3.is_done is True
    assert "###DONE###" in step3.content

    agent.reset()
    assert agent.current_step == 0


def test_orchestrator_dual_mode_success(tmp_path: Path) -> None:
    """Verify dual mode end-to-end turn-taking with agent and user actions."""
    task = Task(
        task_id="TASK-DUAL-01",
        goal="Place an order for Apex Energy Solutions at SITE-REE-001.",
        initial_user_message="Hello, I need to place an order for Apex Energy Solutions.",
    )

    agent_actions = [
        # Agent turn 1: find account
        ScriptedAction(
            content="Checking account records for Apex.",
            tool_calls=[{"tool_name": "find_account", "arguments": {"query": "Apex"}}],
        ),
        # Agent turn 2: wrap up and conclude
        ScriptedAction(
            content="Account confirmed. Delivery scheduled. ###DONE###",
            tool_calls=[],
        ),
    ]
    user_actions = [
        # User turn 2 (turn index 2): user checks site access
        ScriptedAction(
            content="I verified our pad road conditions.",
            tool_calls=[{"tool_name": "check_site_access", "arguments": {"site_code": "SITE-REE-001"}}],
        ),
    ]

    agent = ScriptedAgent(name="agent_dual", actions=agent_actions)
    user_sim = ScriptedAgent(name="user_dual", actions=user_actions)

    orchestrator = Orchestrator(
        run_id="test_dual_run",
        output_dir=tmp_path,
        save_to_disk=True,
    )

    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        user_sim=user_sim,
        mode=SimulationMode.DUAL,
        max_turns=10,
    )

    assert trajectory.status == EpisodeStatus.COMPLETED
    assert len(trajectory.turns) >= 3
    # Turn 0: user opening statement
    assert trajectory.turns[0].speaker == "user"
    assert "Apex Energy Solutions" in trajectory.turns[0].content
    # Turn 1: agent find_account
    assert trajectory.turns[1].speaker == "agent"
    assert len(trajectory.turns[1].tool_calls) == 1
    assert trajectory.turns[1].tool_calls[0].tool_name == "find_account"
    assert trajectory.turns[1].tool_calls[0].is_error is False
    # Turn 2: user check_site_access
    assert trajectory.turns[2].speaker == "user"
    assert len(trajectory.turns[2].tool_calls) == 1
    assert trajectory.turns[2].tool_calls[0].tool_name == "check_site_access"
    assert trajectory.turns[2].tool_calls[0].is_error is False

    # Check persistence
    saved_file = tmp_path / "test_dual_run" / "TASK-DUAL-01_dual.json"
    assert saved_file.is_file()
    saved_json = json.loads(saved_file.read_text(encoding="utf-8"))
    assert saved_json["episode_id"] == trajectory.episode_id


def test_orchestrator_permission_barrier_agent_calling_user_tool(tmp_path: Path) -> None:
    """Verify dual mode blocks agent from calling user tools with exact permission error."""
    task = Task(
        task_id="TASK-BARRIER-01",
        goal="Attempt unauthorized user tool invocation.",
        initial_user_message="Hello ChemDesk.",
    )

    agent_actions = [
        ScriptedAction(
            content="I will directly inspect your well pad road status.",
            tool_calls=[{"tool_name": "check_site_access", "arguments": {"site_code": "SITE-MID-001"}}],
        ),
        ScriptedAction(content="Closing episode. ###DONE###"),
    ]

    agent = ScriptedAgent(name="rogue_agent", actions=agent_actions)
    orchestrator = Orchestrator(output_dir=tmp_path, save_to_disk=False)

    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        mode=SimulationMode.DUAL,
        max_turns=5,
    )

    # Check the agent's tool call in turn 1
    agent_turn = trajectory.turns[1]
    assert agent_turn.speaker == "agent"
    assert len(agent_turn.tool_calls) == 1
    tool_call = agent_turn.tool_calls[0]
    assert tool_call.tool_name == "check_site_access"
    assert tool_call.is_error is True
    assert tool_call.result["is_error"] is True
    assert "Permission Denied: User tool restricted to site personnel" in tool_call.result["error_message"]


def test_orchestrator_permission_barrier_user_calling_agent_tool(tmp_path: Path) -> None:
    """Verify dual mode blocks user from calling agent tools."""
    task = Task(
        task_id="TASK-BARRIER-02",
        goal="Attempt unauthorized agent tool invocation from user.",
        initial_user_message=None,
    )

    user_actions = [
        ScriptedAction(
            content="I will directly create an order myself.",
            tool_calls=[{
                "tool_name": "create_order",
                "arguments": {"account_id": "ACC-001", "site_code": "SITE-MID-001", "delivery_date": "2026-10-20", "lines": []},
            }],
        ),
        ScriptedAction(content="Done. ###DONE###"),
    ]
    agent_actions = [
        ScriptedAction(content="Cannot do that. ###DONE###"),
    ]

    user_sim = ScriptedAgent(name="rogue_user", actions=user_actions)
    agent = ScriptedAgent(name="agent", actions=agent_actions)
    orchestrator = Orchestrator(output_dir=tmp_path, save_to_disk=False)

    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        user_sim=user_sim,
        mode=SimulationMode.DUAL,
        max_turns=5,
    )

    user_turn = trajectory.turns[0]
    assert user_turn.speaker == "user"
    assert len(user_turn.tool_calls) == 1
    assert user_turn.tool_calls[0].is_error is True
    assert "Permission Denied" in user_turn.tool_calls[0].result["error_message"]


def test_orchestrator_no_user_mode(tmp_path: Path) -> None:
    """Verify no_user mode equips agent with all 17 tools and executes without user turns."""
    task = Task(
        task_id="TASK-NOUSER-01",
        goal="Autonomous order placement with full environment observation.",
    )

    # In no_user mode, agent can call both agent tools AND user tools!
    agent_actions = [
        ScriptedAction(
            content="Checking site access autonomously.",
            tool_calls=[{"tool_name": "check_site_access", "arguments": {"site_code": "SITE-REE-001"}}],
        ),
        ScriptedAction(
            content="Finding account autonomously.",
            tool_calls=[{"tool_name": "find_account", "arguments": {"query": "Apex"}}],
        ),
        ScriptedAction(
            content="Autonomous workflow completed. ###DONE###",
        ),
    ]

    agent = ScriptedAgent(name="autonomous_agent", actions=agent_actions)
    orchestrator = Orchestrator(output_dir=tmp_path, save_to_disk=False)

    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        mode=SimulationMode.NO_USER,
        max_turns=10,
    )

    assert trajectory.status == EpisodeStatus.COMPLETED
    assert trajectory.mode == SimulationMode.NO_USER

    # Confirm all turns are agent turns
    for turn in trajectory.turns:
        assert turn.speaker == "agent"

    # Confirm check_site_access was permitted and executed successfully
    turn_0 = trajectory.turns[0]
    assert len(turn_0.tool_calls) == 1
    assert turn_0.tool_calls[0].tool_name == "check_site_access"
    assert turn_0.tool_calls[0].is_error is False


def test_orchestrator_oracle_plan_mode(tmp_path: Path) -> None:
    """Verify oracle_plan mode injects gold plan into agent context."""
    gold_plan = [
        {"action": "find_account", "query": "Apex"},
        {"action": "verify_caller", "caller_name": "John Doe"},
    ]
    task = Task(
        task_id="TASK-ORACLE-01",
        goal="Execute order with oracle guidance.",
        initial_user_message="Hello.",
        gold_plan=gold_plan,
    )

    received_context: Dict[str, Any] = {}

    def agent_step(history: Any = None, context: Any = None) -> ScriptedAction:
        if context:
            received_context.update(context)
        return ScriptedAction(content="Executing oracle plan. ###DONE###")

    orchestrator = Orchestrator(output_dir=tmp_path, save_to_disk=False)
    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent_step,
        mode=SimulationMode.ORACLE_PLAN,
        max_turns=5,
    )

    assert trajectory.status == EpisodeStatus.COMPLETED
    assert trajectory.mode == SimulationMode.ORACLE_PLAN
    assert received_context.get("gold_plan") == gold_plan


def test_orchestrator_max_turns_exceeded(tmp_path: Path) -> None:
    """Verify max_turns limit sets status to max_turns_exceeded."""
    task = Task(
        task_id="TASK-MAXTURNS-01",
        goal="Long conversation exceeding limit.",
        initial_user_message="Hello.",
    )

    # Agents that never emit completion tokens
    agent = ScriptedAgent(name="chattier_agent", actions=[ScriptedAction(content=f"Agent message {i}") for i in range(10)])
    user = ScriptedAgent(name="chattier_user", actions=[ScriptedAction(content=f"User message {i}") for i in range(10)])

    orchestrator = Orchestrator(output_dir=tmp_path, save_to_disk=False)
    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        user_sim=user,
        mode=SimulationMode.DUAL,
        max_turns=4,
    )

    assert trajectory.status == EpisodeStatus.MAX_TURNS_EXCEEDED
    assert len(trajectory.turns) == 4


def test_orchestrator_transfer_termination(tmp_path: Path) -> None:
    """Verify termination when agent or user emits ###TRANSFER###."""
    task = Task(
        task_id="TASK-TRANSFER-01",
        goal="Escalate to human desk.",
        initial_user_message="I demand a credit limit override right now.",
    )

    agent_actions = [
        ScriptedAction(content="I cannot override credit limit. Transferring to human desk. ###TRANSFER###"),
    ]
    agent = ScriptedAgent(name="agent", actions=agent_actions)

    orchestrator = Orchestrator(output_dir=tmp_path, save_to_disk=False)
    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        mode=SimulationMode.DUAL,
        max_turns=5,
    )

    assert trajectory.status == EpisodeStatus.COMPLETED
    assert any("###TRANSFER###" in t.content for t in trajectory.turns)


def test_orchestrator_initial_db_setup_and_diff(tmp_path: Path) -> None:
    """Verify initial_db_setup executes correctly and db_diff tracks mutations."""
    # Setup creates a custom account record
    setup_sql = """
    INSERT INTO accounts (account_id, company_name, status, credit_limit, payment_terms, created_at)
    VALUES ('ACC-CUSTOM-99', 'Custom Energy Inc', 'active', 50000.0, 'Net 30', '2026-10-14T00:00:00Z');
    """

    task = Task(
        task_id="TASK-SETUP-01",
        goal="Verify setup and mutation diff.",
        initial_db_setup=setup_sql,
        initial_user_message="Start.",
    )

    # Agent cancels an existing order to mutate the database
    agent_actions = [
        ScriptedAction(
            content="Cancelling order ORD-0001.",
            tool_calls=[{"tool_name": "cancel_order", "arguments": {"order_id": "ORD-0001", "reason": "Customer request"}}],
        ),
        ScriptedAction(content="Done. ###DONE###"),
    ]

    agent = ScriptedAgent(name="agent", actions=agent_actions)
    orchestrator = Orchestrator(output_dir=tmp_path, save_to_disk=False)

    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        mode=SimulationMode.DUAL,
        max_turns=5,
    )

    assert trajectory.status == EpisodeStatus.COMPLETED
    # Verify db_diff captured the order cancellation mutation
    assert "orders" in trajectory.db_diff
    order_diff = trajectory.db_diff["orders"]
    assert len(order_diff["added"]) > 0
    assert any(o["order_id"] == "ORD-0001" and o["status"] == "cancelled" for o in order_diff["added"])
