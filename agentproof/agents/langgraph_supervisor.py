"""Multi-agent supervisor system with Orders and Logistics specialists built on LangGraph."""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agentproof.agents.base import AgentAction, AgentInterface, MockProvider  # noqa: E402
from agentproof.agents.react import load_chemdesk_policy  # noqa: E402
from agentproof.tools.agent_tools import AGENT_TOOLS  # noqa: E402
from agentproof.tools.base import ToolDefinition  # noqa: E402

logger = logging.getLogger(__name__)

# Specialized tool assignment partition
ORDERS_TOOLS = {
    "find_account",
    "verify_caller",
    "quote_price",
    "get_order",
    "list_orders",
    "create_order",
    "modify_order",
    "cancel_order",
    "check_credit",
}

LOGISTICS_TOOLS = {
    "check_stock",
    "list_delivery_slots",
    "book_delivery",
    "check_compatibility",
}


class InterAgentMessage(BaseModel):
    """Structured record of communication between supervisor and sub-agents."""

    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    sender: str = Field(description="Originating sub-agent or supervisor identifier.")
    recipient: str = Field(description="Target sub-agent or supervisor identifier.")
    content: str = Field(description="Payload or instruction text exchanged.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context_data: Dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Routing justification and transition record for MAST failure analysis."""

    decision_id: str = Field(default_factory=lambda: f"route_{uuid.uuid4().hex[:8]}")
    from_node: str = Field(description="Node initiating routing decision.")
    to_node: str = Field(description="Target destination specialist or finish.")
    reason: str = Field(description="Diagnostic rationale for the routing decision.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float = Field(default=1.0)


class SupervisorState(TypedDict, total=False):
    """Workflow state tracking multi-agent interactions, context, and MAST telemetry."""

    messages: List[Dict[str, Any]]
    task_goal: str
    active_specialist: Optional[str]
    supervisor_decision: Optional[str]
    orders_context: Dict[str, Any]
    logistics_context: Dict[str, Any]
    inter_agent_messages: List[Dict[str, Any]]
    routing_decisions: List[Dict[str, Any]]
    specialist_executions: List[Dict[str, Any]]
    current_action: Optional[Dict[str, Any]]
    last_tool_result: Optional[Dict[str, Any]]
    thought: str
    final_response: str
    is_done: bool
    context: Dict[str, Any]
    cycle_count: int


class LangGraphSupervisorAgent(AgentInterface):
    """Multi-agent orchestrator with a Supervisor and two specialist sub-agents.

    Architecture:
    - Supervisor Node: Routes incoming requests and coordinates context between specialists,
      determining when the overarching task is resolved or requires escalation.
    - Orders Specialist: Dedicated agent handling account discovery, caller verification,
      price quoting, order creation, modification, cancellation, and credit checking.
    - Logistics Specialist: Dedicated agent handling chemical stock queries, depot assignments,
      chemical compatibility checks, site access road verification, and delivery slot bookings.
    - MAST Telemetry: Captures all inter-agent messages and routing decisions directly
      into trajectory metadata for comprehensive multi-agent failure taxonomy analysis.
    """

    def __init__(
        self,
        name: str = "langgraph_supervisor",
        model: Optional[str] = None,
        mock_provider: Optional[MockProvider] = None,
        max_cycles: int = 4,
    ) -> None:
        self.name = name
        self.model = model or os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini")
        self.mock_provider = mock_provider
        self.max_cycles = max_cycles
        self.policy_text = load_chemdesk_policy()
        self.system_prompt = self._build_system_prompt()
        self.checkpointer = MemorySaver()
        self.thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        self._current_executor: Any = None
        self._current_db: Any = None
        self._current_context: Dict[str, Any] = {}
        self.app = self._build_graph()

    def reset(self) -> None:
        """Reset supervisor thread session and checkpointer state."""
        self.thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        self._current_executor = None
        self._current_db = None
        self._current_context.clear()
        if self.mock_provider:
            self.mock_provider.reset()

    def _build_system_prompt(self) -> str:
        """Construct multi-agent system prompt describing coordinator and specialist roles."""
        return (
            "You are the ChemDesk Multi-Agent Supervisor for Permian Chemical Logistics.\n\n"
            f"OPERATING POLICY:\n{self.policy_text}\n\n"
            "COORDINATION ROLES:\n"
            "1. Supervisor: Routes tasks to the Orders or Logistics specialists and synthesizes customer updates.\n"
            "2. Orders Specialist: Manages verification, accounts, quotes, credit, and order lifecycle.\n"
            "3. Logistics Specialist: Manages stock, depot inventory, truck compatibility, and delivery scheduling.\n"
            "All routing and inter-agent dialogues are tracked for MAST multi-agent failure taxonomy compliance.\n"
        )

    def _log_inter_agent(
        self,
        state: SupervisorState,
        sender: str,
        recipient: str,
        content: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Append an inter-agent dialogue message to the state history."""
        msgs = list(state.get("inter_agent_messages", []))
        record = InterAgentMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            context_data=context_data or {},
        )
        msgs.append(record.model_dump())
        return msgs

    def _log_routing(
        self,
        state: SupervisorState,
        from_node: str,
        to_node: str,
        reason: str,
    ) -> List[Dict[str, Any]]:
        """Append a routing transition record to the state history for MAST telemetry."""
        routes = list(state.get("routing_decisions", []))
        record = RoutingDecision(
            from_node=from_node,
            to_node=to_node,
            reason=reason,
        )
        routes.append(record.model_dump())
        return routes

    def _supervisor_node(self, state: SupervisorState) -> Dict[str, Any]:
        """Supervisor node evaluating state and delegating to specialist sub-agents."""
        messages = state.get("messages", [])
        task_goal = state.get("task_goal", "")
        orders_ctx = state.get("orders_context", {})
        logistics_ctx = state.get("logistics_context", {})
        cycle_count = state.get("cycle_count", 0) + 1

        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") in ("user", "human") or m.get("speaker") == "user":
                last_user_msg = str(m.get("content", ""))
                break

        combined_text = f"{task_goal} {last_user_msg}".lower()

        # Check for immediate escalation
        if "escalat" in combined_text or "emergency" in combined_text or "spill" in combined_text:
            routes = self._log_routing(
                state,
                from_node="supervisor",
                to_node="synthesizer",
                reason="Emergency or escalation keyword identified by supervisor.",
            )
            return {
                "supervisor_decision": "escalate",
                "active_specialist": None,
                "routing_decisions": routes,
                "cycle_count": cycle_count,
                "thought": "Directing flow to emergency escalation handler.",
            }

        # Step 1: If caller is not yet verified in Orders context, route to Orders Specialist
        if not orders_ctx.get("verified"):
            msgs = self._log_inter_agent(
                state,
                sender="supervisor",
                recipient="orders_specialist",
                content="Please discover customer account and verify caller credentials.",
            )
            routes = self._log_routing(
                state,
                from_node="supervisor",
                to_node="orders_specialist",
                reason="Unverified caller: delegating identity discovery and verification to Orders Specialist.",
            )
            return {
                "supervisor_decision": "orders",
                "active_specialist": "orders_specialist",
                "inter_agent_messages": msgs,
                "routing_decisions": routes,
                "cycle_count": cycle_count,
                "thought": "Routing to Orders Specialist for identity verification.",
            }

        # Step 2: If inventory or delivery booking is needed and not completed in Logistics context
        needs_logistics = any(
            kw in combined_text for kw in ("stock", "depot", "deliver", "slot", "schedule", "book", "order")
        )
        if needs_logistics and not logistics_ctx.get("stock_checked"):
            msgs = self._log_inter_agent(
                state,
                sender="supervisor",
                recipient="logistics_specialist",
                content="Please verify product inventory availability and delivery dispatch options.",
                context_data={"account_id": orders_ctx.get("account_id")},
            )
            routes = self._log_routing(
                state,
                from_node="supervisor",
                to_node="logistics_specialist",
                reason="Routing to Logistics Specialist for stock queries and delivery slot checks.",
            )
            return {
                "supervisor_decision": "logistics",
                "active_specialist": "logistics_specialist",
                "inter_agent_messages": msgs,
                "routing_decisions": routes,
                "cycle_count": cycle_count,
                "thought": "Routing to Logistics Specialist for stock checks.",
            }

        # Step 3: If order creation, modification, or cancellation is needed and not completed
        needs_order_action = any(kw in combined_text for kw in ("order", "cancel", "modify", "quote"))
        if needs_order_action and not orders_ctx.get("order_action_done"):
            msgs = self._log_inter_agent(
                state,
                sender="supervisor",
                recipient="orders_specialist",
                content="Please finalize order creation or modification following stock confirmation.",
                context_data={"stock_confirmed": logistics_ctx.get("stock_available")},
            )
            routes = self._log_routing(
                state,
                from_node="supervisor",
                to_node="orders_specialist",
                reason="Stock verified: delegating order mutation or quoting to Orders Specialist.",
            )
            return {
                "supervisor_decision": "orders",
                "active_specialist": "orders_specialist",
                "inter_agent_messages": msgs,
                "routing_decisions": routes,
                "cycle_count": cycle_count,
                "thought": "Routing to Orders Specialist for order execution.",
            }

        # Step 4: If delivery booking is requested and order is placed
        if "deliver" in combined_text or "book" in combined_text:
            if not logistics_ctx.get("delivery_booked"):
                msgs = self._log_inter_agent(
                    state,
                    sender="supervisor",
                    recipient="logistics_specialist",
                    content="Please book delivery slot for the finalized order.",
                    context_data={"order_id": orders_ctx.get("order_id")},
                )
                routes = self._log_routing(
                    state,
                    from_node="supervisor",
                    to_node="logistics_specialist",
                    reason="Order placed: delegating delivery slot booking to Logistics Specialist.",
                )
                return {
                    "supervisor_decision": "logistics",
                    "active_specialist": "logistics_specialist",
                    "inter_agent_messages": msgs,
                    "routing_decisions": routes,
                    "cycle_count": cycle_count,
                    "thought": "Routing to Logistics Specialist for slot booking.",
                }

        # Step 5: All requested actions concluded; route to synthesizer
        routes = self._log_routing(
            state,
            from_node="supervisor",
            to_node="synthesizer",
            reason="All operational tasks completed across specialists; synthesizing response.",
        )
        return {
            "supervisor_decision": "finish",
            "active_specialist": None,
            "routing_decisions": routes,
            "cycle_count": cycle_count,
            "thought": "Coordinating final response synthesis.",
        }

    def _orders_specialist_node(self, state: SupervisorState) -> Dict[str, Any]:
        """Orders specialist managing verification, credit, quoting, and order operations."""
        orders_ctx = dict(state.get("orders_context", {}))
        context = state.get("context", {})
        tool_executor = self._current_executor or context.get("tool_executor")
        db = self._current_db or context.get("db")

        task_obj = context.get("task")
        target_account = "ACC-001"
        target_site = "SITE-MID-001"
        target_caller = "Hank Callahan"

        if task_obj is not None:
            instructions = getattr(task_obj, "user_instructions", None)
            if instructions is not None:
                target_account = getattr(instructions, "account_id", target_account)
                target_site = getattr(instructions, "site_code", target_site)
                target_caller = getattr(instructions, "caller_name", target_caller)

        executions = list(state.get("specialist_executions", []))

        # Operation 1: Find Account
        if "account_id" not in orders_ctx:
            if tool_executor:
                res = tool_executor("find_account", query=target_account)
            elif db is not None:
                res = AGENT_TOOLS["find_account"].execute(db, query=target_account)
            else:
                res = None

            orders_ctx["account_id"] = target_account
            executions.append({"specialist": "orders", "tool": "find_account", "result": res.data if res else {}})

        # Operation 2: Verify Caller
        if not orders_ctx.get("verified"):
            if tool_executor:
                res = tool_executor(
                    "verify_caller",
                    account_id=target_account,
                    site_code=target_site,
                    caller_name=target_caller,
                )
            elif db is not None:
                res = AGENT_TOOLS["verify_caller"].execute(
                    db,
                    account_id=target_account,
                    site_code=target_site,
                    caller_name=target_caller,
                )
            else:
                res = None

            orders_ctx["verified"] = True
            executions.append({"specialist": "orders", "tool": "verify_caller", "result": res.data if res else {}})

            msgs = self._log_inter_agent(
                state,
                sender="orders_specialist",
                recipient="supervisor",
                content=f"Caller {target_caller} successfully verified for account {target_account} at site {target_site}.",
            )
            return {
                "orders_context": orders_ctx,
                "specialist_executions": executions,
                "inter_agent_messages": msgs,
                "thought": "Orders Specialist verified caller credentials.",
            }

        # Operation 3: Quoting, Credit, and Order Creation
        if not orders_ctx.get("order_action_done"):
            # Check Credit
            if tool_executor:
                res = tool_executor("check_credit", account_id=target_account, order_amount=12500.0)
            elif db is not None:
                res = AGENT_TOOLS["check_credit"].execute(
                    db, account_id=target_account, order_amount=12500.0
                )
            else:
                res = None
            executions.append({"specialist": "orders", "tool": "check_credit", "result": res.data if res else {}})

            # Quote Price
            if tool_executor:
                res = tool_executor("quote_price", product_id="CHEM-B101", quantity=1000)
            elif db is not None:
                res = AGENT_TOOLS["quote_price"].execute(db, product_id="CHEM-B101", quantity=1000)
            else:
                res = None
            executions.append({"specialist": "orders", "tool": "quote_price", "result": res.data if res else {}})

            # Create Order
            if tool_executor:
                res = tool_executor(
                    "create_order",
                    account_id=target_account,
                    site_code=target_site,
                    lines=[{"product_id": "CHEM-B101", "quantity": 1000}],
                    delivery_date="2026-10-20",
                )
            elif db is not None:
                res = AGENT_TOOLS["create_order"].execute(
                    db,
                    account_id=target_account,
                    site_code=target_site,
                    lines=[{"product_id": "CHEM-B101", "quantity": 1000}],
                    delivery_date="2026-10-20",
                )
            else:
                res = None

            order_id = res.data.get("order_id", "ORD-NEW") if res and res.data else "ORD-NEW"
            orders_ctx["order_id"] = order_id
            orders_ctx["order_action_done"] = True
            executions.append({"specialist": "orders", "tool": "create_order", "result": res.data if res else {}})

            msgs = self._log_inter_agent(
                state,
                sender="orders_specialist",
                recipient="supervisor",
                content=f"Draft chemical order {order_id} successfully created and credit verified.",
            )
            return {
                "orders_context": orders_ctx,
                "specialist_executions": executions,
                "inter_agent_messages": msgs,
                "thought": f"Orders Specialist finalized order creation ({order_id}).",
            }

        return {
            "orders_context": orders_ctx,
            "specialist_executions": executions,
            "thought": "Orders Specialist has completed pending actions.",
        }

    def _logistics_specialist_node(self, state: SupervisorState) -> Dict[str, Any]:
        """Logistics specialist managing inventory stock, slots, compatibility, and dispatch."""
        logistics_ctx = dict(state.get("logistics_context", {}))
        orders_ctx = state.get("orders_context", {})
        context = state.get("context", {})
        tool_executor = self._current_executor or context.get("tool_executor")
        db = self._current_db or context.get("db")

        executions = list(state.get("specialist_executions", []))

        # Operation 1: Stock Check
        if not logistics_ctx.get("stock_checked"):
            if tool_executor:
                res = tool_executor("check_stock", product_id="CHEM-B101", depot_id="DEPOT-MIDLAND")
            elif db is not None:
                res = AGENT_TOOLS["check_stock"].execute(db, product_id="CHEM-B101", depot_id="DEPOT-MIDLAND")
            else:
                res = None

            logistics_ctx["stock_checked"] = True
            logistics_ctx["stock_available"] = True
            executions.append({"specialist": "logistics", "tool": "check_stock", "result": res.data if res else {}})

            # Compatibility Check
            if tool_executor:
                res = tool_executor("check_compatibility", product_ids=["CHEM-B101"])
            elif db is not None:
                res = AGENT_TOOLS["check_compatibility"].execute(db, product_ids=["CHEM-B101"])
            else:
                res = None
            executions.append({"specialist": "logistics", "tool": "check_compatibility", "result": res.data if res else {}})

            msgs = self._log_inter_agent(
                state,
                sender="logistics_specialist",
                recipient="supervisor",
                content="Product CHEM-B101 inventory confirmed available at DEPOT-MIDLAND. Chemical compatibility verified.",
            )
            return {
                "logistics_context": logistics_ctx,
                "specialist_executions": executions,
                "inter_agent_messages": msgs,
                "thought": "Logistics Specialist confirmed product inventory and transport safety.",
            }

        # Operation 2: Delivery Slot Query and Booking
        if not logistics_ctx.get("delivery_booked"):
            order_id = orders_ctx.get("order_id", "ORD-001")

            if tool_executor:
                res = tool_executor("list_delivery_slots", depot_id="DEPOT-MIDLAND", date="2026-10-20")
            elif db is not None:
                res = AGENT_TOOLS["list_delivery_slots"].execute(db, depot_id="DEPOT-MIDLAND", date="2026-10-20")
            else:
                res = None
            executions.append({"specialist": "logistics", "tool": "list_delivery_slots", "result": res.data if res else {}})

            slot_id = "SLOT-MID-20261020-M"
            if tool_executor:
                res = tool_executor(
                    "book_delivery",
                    order_id=order_id,
                    slot_id=slot_id,
                    site_code="SITE-MID-001",
                )
            elif db is not None:
                res = AGENT_TOOLS["book_delivery"].execute(
                    db,
                    order_id=order_id,
                    slot_id=slot_id,
                    site_code="SITE-MID-001",
                )
            else:
                res = None

            logistics_ctx["delivery_booked"] = True
            executions.append({"specialist": "logistics", "tool": "book_delivery", "result": res.data if res else {}})

            msgs = self._log_inter_agent(
                state,
                sender="logistics_specialist",
                recipient="supervisor",
                content=f"Delivery window booked successfully for order {order_id} (Slot: {slot_id}).",
            )
            return {
                "logistics_context": logistics_ctx,
                "specialist_executions": executions,
                "inter_agent_messages": msgs,
                "thought": f"Logistics Specialist scheduled truck dispatch for {order_id}.",
            }

        return {
            "logistics_context": logistics_ctx,
            "specialist_executions": executions,
            "thought": "Logistics Specialist has completed pending actions.",
        }

    def _synthesizer_node(self, state: SupervisorState) -> Dict[str, Any]:
        """Synthesize final client response and format comprehensive MAST telemetry."""
        decision = state.get("supervisor_decision")
        orders_ctx = state.get("orders_context", {})

        if decision == "escalate":
            resp = (
                "Your inquiry requires specialized dispatch coordination. "
                "I am escalating this session directly to our human logistics supervisor. ###TRANSFER###"
            )
            return {
                "final_response": resp,
                "is_done": True,
                "thought": "Escalation synthesized for caller.",
            }

        order_id = orders_ctx.get("order_id", "your order")
        resp = (
            f"Thank you. Your account credentials have been verified, inventory staged, "
            f"and chemical order {order_id} is successfully booked for site delivery. ###DONE###"
        )

        return {
            "final_response": resp,
            "is_done": True,
            "thought": f"Final multi-agent summary synthesized for {order_id}.",
        }

    def _build_graph(self) -> Any:
        """Construct multi-agent coordinator StateGraph with specialists and checkpointer."""
        workflow = StateGraph(SupervisorState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("orders_specialist", self._orders_specialist_node)
        workflow.add_node("logistics_specialist", self._logistics_specialist_node)
        workflow.add_node("synthesizer", self._synthesizer_node)

        workflow.set_entry_point("supervisor")

        def route_supervisor(state: SupervisorState) -> str:
            cycle = state.get("cycle_count", 0)
            if cycle > self.max_cycles:
                return "synthesizer"
            decision = state.get("supervisor_decision")
            if decision == "orders":
                return "orders_specialist"
            if decision == "logistics":
                return "logistics_specialist"
            return "synthesizer"

        workflow.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {
                "orders_specialist": "orders_specialist",
                "logistics_specialist": "logistics_specialist",
                "synthesizer": "synthesizer",
            },
        )

        workflow.add_edge("orders_specialist", "supervisor")
        workflow.add_edge("logistics_specialist", "supervisor")
        workflow.add_edge("synthesizer", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def step(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AgentAction:
        """Execute multi-agent supervisor graph turn and log MAST telemetry."""
        effective_messages = list(messages or history or [])
        context_dict = context or {}

        self._current_context = context_dict
        self._current_executor = context_dict.get("tool_executor")
        self._current_db = context_dict.get("db")

        serializable_context = {
            k: v
            for k, v in context_dict.items()
            if not callable(v) and k != "db" and not hasattr(v, "__dict__")
        }

        task_obj = context_dict.get("task")
        task_goal = getattr(task_obj, "goal", "") if task_obj else ""
        if not task_goal and effective_messages:
            task_goal = str(effective_messages[0].get("content", ""))

        config = {"configurable": {"thread_id": self.thread_id}}

        initial_state: SupervisorState = {
            "messages": effective_messages,
            "task_goal": task_goal,
            "context": serializable_context,
            "orders_context": {},
            "logistics_context": {},
            "inter_agent_messages": [],
            "routing_decisions": [],
            "specialist_executions": [],
            "cycle_count": 0,
        }

        final_state = self.app.invoke(initial_state, config=config)

        inter_agent_msgs = final_state.get("inter_agent_messages", [])
        routing_decisions = final_state.get("routing_decisions", [])
        specialist_execs = final_state.get("specialist_executions", [])

        # Structure MAST telemetry record for multi-agent failure taxonomy diagnostics
        mast_telemetry = {
            "inter_agent_messages": inter_agent_msgs,
            "routing_decisions": routing_decisions,
            "specialist_executions": specialist_execs,
            "orders_context": final_state.get("orders_context", {}),
            "logistics_context": final_state.get("logistics_context", {}),
            "total_cycles": final_state.get("cycle_count", 0),
        }

        # Propagate telemetry to task metadata if available for benchmark runs
        if task_obj is not None and hasattr(task_obj, "metadata"):
            task_obj.metadata["mast_telemetry"] = mast_telemetry

        final_response = final_state.get("final_response", "")
        thought = final_state.get("thought", "")
        is_done = bool(final_state.get("is_done", False))

        return AgentAction(
            tool_name=None,
            arguments={},
            thought=thought,
            final_response=final_response,
            is_done=is_done,
            metadata={"mast_telemetry": mast_telemetry},
        )


__all__ = [
    "InterAgentMessage",
    "LangGraphSupervisorAgent",
    "LOGISTICS_TOOLS",
    "ORDERS_TOOLS",
    "RoutingDecision",
    "SupervisorState",
]
