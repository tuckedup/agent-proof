"""Unit and integration tests for agent architectures in agentproof.agents."""

from pathlib import Path
from typing import Any, List

import pytest

from agentproof.agents.base import AgentAction, MockProvider
from agentproof.agents.langgraph_plan_execute import LangGraphPlanExecuteAgent
from agentproof.agents.langgraph_supervisor import (
    LOGISTICS_TOOLS,
    ORDERS_TOOLS,
    LangGraphSupervisorAgent,
)
from agentproof.agents.react import ReactBaselineAgent, load_chemdesk_policy
from agentproof.domain.db import DatabaseManager
from agentproof.sim.models import SimulationMode, Task
from agentproof.sim.orchestrator import Orchestrator
from agentproof.tools.agent_tools import AGENT_TOOLS
from agentproof.tools.base import ToolResult


@pytest.fixture
def test_db() -> DatabaseManager:
    """Provide an isolated seeded database for agent unit testing."""
    db = DatabaseManager(":memory:")
    seed_dir = Path(__file__).resolve().parent.parent / "domain" / "seed"
    db.seed_from_directory(str(seed_dir))
    return db


def test_agent_action_model() -> None:
    """Verify AgentAction validation, serialization, and orchestrator compatibility properties."""
    action = AgentAction(
        tool_name="find_account",
        arguments={"query": "Apex"},
        thought="Need to look up customer account.",
        is_done=False,
    )
    assert action.tool_name == "find_account"
    assert action.arguments == {"query": "Apex"}
    assert action.thought == "Need to look up customer account."
    assert action.content == "Need to look up customer account."
    assert len(action.tool_calls) == 1
    assert action.tool_calls[0]["tool_name"] == "find_account"

    final_action = AgentAction(
        thought="All actions concluded.",
        final_response="Order placed. ###DONE###",
        is_done=True,
    )
    assert final_action.final_response == "Order placed. ###DONE###"
    assert final_action.content == "Order placed. ###DONE###"
    assert final_action.tool_calls == []
    assert final_action.is_done is True


def test_mock_provider() -> None:
    """Verify MockProvider canned response and custom handler behaviors."""
    provider = MockProvider(
        canned_responses=[
            "Thought: Step 1.\nAction: find_account({\"query\": \"Apex\"})",
            "Final Response: Found account. ###DONE###",
        ]
    )
    res1 = provider.complete([{"role": "user", "content": "Hello"}])
    assert "find_account" in res1["choices"][0]["message"]["content"]

    res2 = provider.complete([{"role": "user", "content": "Continue"}])
    assert "###DONE###" in res2["choices"][0]["message"]["content"]
    assert provider.call_count == 2

    provider.reset()
    assert provider.call_count == 0


def test_react_baseline_policy_embedding() -> None:
    """Verify that ReactBaselineAgent embeds ChemDesk policy in system prompt."""
    agent = ReactBaselineAgent()
    policy_str = load_chemdesk_policy()
    assert len(policy_str) > 50
    assert "ChemDesk" in agent.system_prompt
    assert "Standard Operating Procedures" in agent.system_prompt or "Operating Policy" in agent.system_prompt
    assert "find_account" in agent.system_prompt or "verify_caller" in agent.system_prompt


def test_react_baseline_step_with_mock_provider(test_db: DatabaseManager) -> None:
    """Verify ReactBaselineAgent step execution using MockProvider."""
    mock_llm = MockProvider(
        canned_responses=[
            "Thought: I should search for the account first.\nAction: find_account({\"query\": \"Apex\"})",
            "Final Response: Located account for Apex. How else may I assist you? ###DONE###",
        ]
    )
    agent = ReactBaselineAgent(mock_provider=mock_llm)

    executed_tools: List[str] = []

    def tool_executor(name: str, **kwargs: Any) -> Any:
        executed_tools.append(name)
        tool_def = AGENT_TOOLS[name]
        return tool_def.execute(test_db, **kwargs)

    context = {"tool_executor": tool_executor, "db": test_db}
    messages = [{"role": "user", "content": "Hello, need to place an order for Apex."}]

    action = agent.step(messages=messages, context=context)

    assert "find_account" in executed_tools
    assert action.is_done is True
    assert "Apex" in (action.final_response or "")


def test_react_baseline_offline_heuristic_fallback(test_db: DatabaseManager) -> None:
    """Verify ReactBaselineAgent offline heuristic generator when no API key is available."""
    agent = ReactBaselineAgent(mock_provider=None)

    executed_tools: List[str] = []

    def tool_executor(name: str, **kwargs: Any) -> Any:
        executed_tools.append(name)
        tool_def = AGENT_TOOLS[name]
        return tool_def.execute(test_db, **kwargs)

    context = {"tool_executor": tool_executor, "db": test_db}
    messages = [{"role": "user", "content": "I need to order biocide for Apex at SITE-MID-001."}]

    action = agent.step(messages=messages, context=context)
    assert len(executed_tools) >= 1
    assert "find_account" in executed_tools
    assert action.final_response is not None


def test_langgraph_plan_execute_planner_and_executor(test_db: DatabaseManager) -> None:
    """Verify LangGraphPlanExecuteAgent planner decomposition and tool execution flow."""
    agent = LangGraphPlanExecuteAgent()

    executed_tools: List[str] = []

    def tool_executor(name: str, **kwargs: Any) -> Any:
        executed_tools.append(name)
        tool_def = AGENT_TOOLS[name]
        return tool_def.execute(test_db, **kwargs)

    context = {
        "tool_executor": tool_executor,
        "db": test_db,
        "receiver_on_site": True,
    }
    messages = [{"role": "user", "content": "Hank Callahan from Apex at SITE-MID-001 requesting quote."}]

    action = agent.step(messages=messages, context=context)
    assert action is not None
    assert "plan" in action.metadata
    assert len(action.metadata["plan"]) >= 4
    assert len(executed_tools) >= 1


def test_langgraph_plan_execute_policy_check_hazmat_block(test_db: DatabaseManager) -> None:
    """Verify policy_check halts hazmat write action if receiver certification is missing."""
    agent = LangGraphPlanExecuteAgent()

    # Ensure caller does not hold hazmat cert
    test_db.execute_write("UPDATE contacts SET has_hazmat_cert = 0 WHERE account_id = 'ACC-001'")

    executed_tools: List[str] = []

    def tool_executor(name: str, **kwargs: Any) -> Any:
        executed_tools.append(name)
        tool_def = AGENT_TOOLS[name]
        return tool_def.execute(test_db, **kwargs)

    context = {
        "tool_executor": tool_executor,
        "db": test_db,
        "receiver_on_site": False,  # Receiver not confirmed
    }

    state = {
        "current_action": {
            "tool_name": "create_order",
            "arguments": {
                "account_id": "ACC-001",
                "site_code": "SITE-MID-001",
                "lines": [{"product_id": "CHEM-B101", "quantity": 1000}],  # Hazmat Class 8
                "delivery_date": "2026-10-20",
            },
        },
        "context": context,
    }

    check_res = agent._policy_check_node(state)  # type: ignore[arg-type]
    assert check_res["policy_check_passed"] is False
    assert check_res["policy_violation"]["rule"] == "hazmat_certification"


def test_langgraph_plan_execute_policy_check_status_cutoff_modification(test_db: DatabaseManager) -> None:
    """Verify policy_check halts order modification once order reaches picked status."""
    agent = LangGraphPlanExecuteAgent()

    # Set existing order ORD-0001 to picked status
    test_db.execute_write("UPDATE orders SET status = 'picked' WHERE order_id = 'ORD-0001'")

    state = {
        "current_action": {
            "tool_name": "modify_order",
            "arguments": {"order_id": "ORD-0001", "lines": [{"product_id": "CHEM-B101", "quantity": 500}]},
        },
        "context": {"db": test_db},
    }

    check_res = agent._policy_check_node(state)  # type: ignore[arg-type]
    assert check_res["policy_check_passed"] is False
    assert check_res["policy_violation"]["rule"] == "status_cutoff"
    assert "picked" in check_res["policy_violation"]["message"]


def test_langgraph_plan_execute_policy_check_credit_limit(test_db: DatabaseManager) -> None:
    """Verify policy_check halts order creation when order exceeds available credit ceiling."""
    agent = LangGraphPlanExecuteAgent()

    # Set credit limit to negative to guarantee violation
    test_db.execute_write("UPDATE accounts SET credit_limit = -500000.0 WHERE account_id = 'ACC-001'")

    state = {
        "current_action": {
            "tool_name": "create_order",
            "arguments": {
                "account_id": "ACC-001",
                "site_code": "SITE-MID-001",
                "lines": [{"product_id": "CHEM-B101", "quantity": 2000}],  # 2000 * $12.50 = $25,000
                "delivery_date": "2026-10-20",
            },
        },
        "context": {"db": test_db},
    }

    check_res = agent._policy_check_node(state)  # type: ignore[arg-type]
    assert check_res["policy_check_passed"] is False
    assert check_res["policy_violation"]["rule"] == "credit_limit"
    assert check_res["policy_violation"]["reason"] == "credit_limit_exceeded"


def test_langgraph_plan_execute_rejects_incomplete_credit_result(
    test_db: DatabaseManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify policy checks fail closed when credit verification has no payload."""
    agent = LangGraphPlanExecuteAgent()

    def incomplete_credit_check(**_: Any) -> ToolResult:
        return ToolResult(success=True, data=None)

    monkeypatch.setattr(AGENT_TOOLS["check_credit"], "func", incomplete_credit_check)
    state = {
        "current_action": {
            "tool_name": "create_order",
            "arguments": {
                "account_id": "ACC-001",
                "site_code": "SITE-MID-001",
                "lines": [{"product_id": "CHEM-B101", "quantity": 1}],
                "delivery_date": "2026-10-20",
            },
        },
        "context": {"db": test_db},
    }

    check_res = agent._policy_check_node(state)  # type: ignore[arg-type]
    assert check_res["policy_check_passed"] is False
    assert check_res["policy_violation"]["reason"] == "credit_check_incomplete"


def test_langgraph_plan_execute_policy_check_road_access(test_db: DatabaseManager) -> None:
    """Verify policy_check halts delivery booking when caliche road access is closed or wet."""
    agent = LangGraphPlanExecuteAgent()

    test_db.execute_write(
        "INSERT OR REPLACE INTO site_access (site_code, road_status, updated_at) "
        "VALUES ('SITE-MID-001', 'restricted_wet', '2026-10-20T10:00:00Z')"
    )

    state = {
        "current_action": {
            "tool_name": "book_delivery",
            "arguments": {
                "order_id": "ORD-0001",
                "slot_id": "SLOT-MID-20261020-M",
                "site_code": "SITE-MID-001",
            },
        },
        "context": {"db": test_db},
    }

    check_res = agent._policy_check_node(state)  # type: ignore[arg-type]
    assert check_res["policy_check_passed"] is False
    assert check_res["policy_violation"]["rule"] == "road_access"


def test_langgraph_plan_execute_reflection_on_error(test_db: DatabaseManager) -> None:
    """Verify that reflection node triggers on tool execution error and updates plan."""
    agent = LangGraphPlanExecuteAgent()

    state = {
        "current_action": {"tool_name": "get_order", "arguments": {"order_id": "ORD-NONEXISTENT"}},
        "last_tool_result": {"is_error": True, "error": "Order ORD-NONEXISTENT not found"},
        "plan": ["get_order", "modify_order"],
        "plan_index": 0,
        "reflection_notes": [],
        "reflection_count": 0,
    }

    ref_res = agent._reflection_node(state)  # type: ignore[arg-type]
    assert len(ref_res["reflection_notes"]) == 1
    assert "ORD-NONEXISTENT" in ref_res["reflection_notes"][0]
    assert ref_res["reflection_count"] == 1


def test_langgraph_supervisor_routing_and_specialists(test_db: DatabaseManager) -> None:
    """Verify LangGraphSupervisorAgent coordinates Orders and Logistics specialists."""
    agent = LangGraphSupervisorAgent()

    executed_tools: List[str] = []

    def tool_executor(name: str, **kwargs: Any) -> Any:
        executed_tools.append(name)
        tool_def = AGENT_TOOLS[name]
        return tool_def.execute(test_db, **kwargs)

    context = {"tool_executor": tool_executor, "db": test_db}
    messages = [
        {
            "role": "user",
            "content": "Hank Callahan from Apex at SITE-MID-001. I want to check stock and order biocide with delivery.",
        }
    ]

    action = agent.step(messages=messages, context=context)

    # Verify both orders tools and logistics tools were utilized by specialists
    assert any(t in ORDERS_TOOLS for t in executed_tools)
    assert any(t in LOGISTICS_TOOLS for t in executed_tools)

    # Verify MAST telemetry logging
    assert "mast_telemetry" in action.metadata
    telemetry = action.metadata["mast_telemetry"]
    assert "inter_agent_messages" in telemetry
    assert len(telemetry["inter_agent_messages"]) >= 2
    assert "routing_decisions" in telemetry
    assert len(telemetry["routing_decisions"]) >= 2


def test_orchestrator_integration_with_react_baseline(test_db: DatabaseManager) -> None:
    """Verify end-to-end integration of ReactBaselineAgent with Orchestrator."""
    orchestrator = Orchestrator(save_to_disk=False)
    agent = ReactBaselineAgent()

    task = Task(
        task_id="TASK-TEST-REACT",
        title="Inquiry on Account",
        category="general_inquiry",
        difficulty="easy",
        goal="Locate Apex account details.",
        initial_user_message="Hello, can you verify account details for Apex?",
    )

    trajectory = orchestrator.run_episode(
        task=task,
        agent=agent,
        user_sim=None,
        mode=SimulationMode.DUAL,
        max_turns=3,
    )

    assert len(trajectory.turns) >= 2
    assert trajectory.status is not None
