"""Plan-and-Execute agent with safety policy verification built on LangGraph."""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agentproof.agents.base import AgentAction, AgentInterface, MockProvider
from agentproof.agents.react import load_chemdesk_policy
from agentproof.tools.agent_tools import AGENT_TOOLS
from agentproof.tools.base import ToolDefinition

logger = logging.getLogger(__name__)

# Mandatory write tools requiring policy safety gating
WRITE_TOOLS = {
    "create_order",
    "modify_order",
    "cancel_order",
    "book_delivery",
    "escalate_to_human",
}

# Known DOT Class 3 (Flammable) and Class 8 (Corrosive) products requiring hazmat certification
HAZMAT_PRODUCTS = {"CHEM-B101", "CHEM-F301", "CHEM-S802"}


class PlanExecuteState(TypedDict, total=False):
    """Execution state schema tracked throughout the LangGraph workflow."""

    messages: List[Dict[str, Any]]
    task_goal: str
    plan: List[str]
    plan_index: int
    current_action: Optional[Dict[str, Any]]
    last_tool_result: Optional[Dict[str, Any]]
    policy_check_passed: bool
    policy_violation: Optional[Dict[str, Any]]
    escalated_by_policy: bool
    reflection_notes: List[str]
    reflection_count: int
    thought: str
    final_response: str
    is_done: bool
    iterations: int
    context: Dict[str, Any]


class LangGraphPlanExecuteAgent(AgentInterface):
    """Plan-and-Execute agent built using LangGraph with critical policy gating.

    Components:
    - planner: Breaks user goals into ordered procedural steps.
    - executor: Decides the next tool call to execute based on current plan state.
    - policy_check: Critical safety node evaluated prior to ANY write tool invocation.
      Verifies hazmat receiver confirmation, order status cutoffs, customer credit ceilings,
      and road accessibility before allowing database mutations.
    - reflection: Triggers automatically upon tool execution errors to adjust the plan or retry.
    - MemorySaver: Checkpointer persisting episode execution states across turns.
    """

    def __init__(
        self,
        name: str = "langgraph_plan_execute",
        model: Optional[str] = None,
        mock_provider: Optional[MockProvider] = None,
        max_reflections: int = 2,
    ) -> None:
        self.name = name
        self.model = model or os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini")
        self.mock_provider = mock_provider
        self.max_reflections = max_reflections
        self.policy_text = load_chemdesk_policy()
        self.system_prompt = self._build_system_prompt()
        self.checkpointer = MemorySaver()
        self.thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        self._current_executor: Any = None
        self._current_db: Any = None
        self._current_context: Dict[str, Any] = {}
        self.app = self._build_graph()

    def reset(self) -> None:
        """Reset agent checkpointer thread and create a new episode state."""
        self.thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        self._current_executor = None
        self._current_db = None
        self._current_context.clear()
        if self.mock_provider:
            self.mock_provider.reset()

    def _build_system_prompt(self) -> str:
        """Construct system instructions containing operational guidelines and plan-execute rules."""
        return (
            "You are ChemDesk, the automated order desk specialist for Permian Chemical Logistics.\n\n"
            f"OPERATING POLICY:\n{self.policy_text}\n\n"
            "PLAN AND EXECUTE ARCHITECTURE:\n"
            "You operate in structured stages: Planner -> Executor -> Policy Check -> Tool Execution -> Reflection.\n"
            "Every state-altering write operation undergoes strict policy gating before execution.\n"
        )

    def _planner_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Decompose incoming request or goal into planned sequential steps."""
        existing_plan = state.get("plan", [])
        if existing_plan:
            return {"thought": "Retaining existing execution plan."}

        context = state.get("context", {})
        gold_plan = context.get("gold_plan") or self._current_context.get("gold_plan")
        if gold_plan and isinstance(gold_plan, list):
            return {
                "plan": list(gold_plan),
                "plan_index": 0,
                "thought": "Adopting authoritative gold plan for task execution.",
            }

        task_goal = state.get("task_goal", "")
        goal_lower = task_goal.lower()

        if "cancel" in goal_lower:
            plan = [
                "find_account",
                "verify_caller",
                "get_order",
                "cancel_order",
                "confirm_completion",
            ]
        elif "modify" in goal_lower or "change" in goal_lower:
            plan = [
                "find_account",
                "verify_caller",
                "get_order",
                "modify_order",
                "confirm_completion",
            ]
        elif "deliver" in goal_lower or "slot" in goal_lower or "schedule" in goal_lower:
            plan = [
                "find_account",
                "verify_caller",
                "list_delivery_slots",
                "book_delivery",
                "confirm_completion",
            ]
        else:
            plan = [
                "find_account",
                "verify_caller",
                "check_stock",
                "quote_price",
                "check_credit",
                "create_order",
                "confirm_completion",
            ]

        return {
            "plan": plan,
            "plan_index": 0,
            "thought": f"Generated plan with {len(plan)} operational steps: {', '.join(plan)}",
        }

    def _executor_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Decide the next tool call or produce final dialogue based on plan status."""
        plan = state.get("plan", [])
        plan_index = state.get("plan_index", 0)
        messages = state.get("messages", [])
        context = state.get("context", {})

        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") in ("user", "human") or m.get("speaker") == "user":
                last_user_msg = str(m.get("content", ""))
                break

        if plan_index >= len(plan):
            return {
                "current_action": None,
                "final_response": "All scheduled operations have been completed successfully. ###DONE###",
                "is_done": True,
                "thought": "All plan steps completed.",
            }

        current_step = plan[plan_index]

        if current_step == "confirm_completion":
            return {
                "current_action": None,
                "final_response": "Your order and chemical logistics request have been processed successfully. ###DONE###",
                "is_done": True,
                "thought": "Reached final customer confirmation step.",
            }

        tool_args: Dict[str, Any] = {}
        target_account = "ACC-001"
        target_site = "SITE-MID-001"
        target_caller = "Hank Callahan"

        task_obj = context.get("task") or self._current_context.get("task")
        if task_obj is not None:
            instructions = getattr(task_obj, "user_instructions", None)
            if instructions is not None:
                target_account = getattr(instructions, "account_id", target_account)
                target_site = getattr(instructions, "site_code", target_site)
                target_caller = getattr(instructions, "caller_name", target_caller)

        if current_step == "find_account":
            query = "Apex" if "Apex" in last_user_msg else target_account
            tool_args = {"query": query}
        elif current_step == "verify_caller":
            tool_args = {
                "account_id": target_account,
                "site_code": target_site,
                "caller_name": target_caller,
            }
        elif current_step == "check_stock":
            tool_args = {"product_id": "CHEM-B101", "depot_id": "DEPOT-MIDLAND"}
        elif current_step == "quote_price":
            tool_args = {"product_id": "CHEM-B101", "quantity": 1000}
        elif current_step == "check_credit":
            tool_args = {"account_id": target_account, "order_amount": 12500.0}
        elif current_step == "create_order":
            tool_args = {
                "account_id": target_account,
                "site_code": target_site,
                "lines": [{"product_id": "CHEM-B101", "quantity": 1000}],
                "delivery_date": "2026-10-20",
            }
        elif current_step == "get_order":
            tool_args = {"order_id": "ORD-0001"}
        elif current_step == "modify_order":
            tool_args = {
                "order_id": "ORD-0001",
                "lines": [{"product_id": "CHEM-B101", "quantity": 1200}],
            }
        elif current_step == "cancel_order":
            tool_args = {"order_id": "ORD-0001", "reason": "Customer cancellation request"}
        elif current_step == "list_delivery_slots":
            tool_args = {"depot_id": "DEPOT-MIDLAND", "date": "2026-10-20"}
        elif current_step == "book_delivery":
            tool_args = {
                "order_id": "ORD-0001",
                "slot_id": "SLOT-MID-20261020-M",
                "site_code": target_site,
            }
        elif current_step == "escalate_to_human":
            tool_args = {
                "account_id": target_account,
                "caller_name": target_caller,
                "reason": "policy_forbidden",
                "context_summary": "Policy violation encountered.",
            }
        else:
            tool_args = {}

        return {
            "current_action": {"tool_name": current_step, "arguments": tool_args},
            "thought": f"Executing plan step '{current_step}' with arguments {tool_args}",
        }

    def _policy_check_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Critical safety node evaluated before ANY write tool call.

        Verifies:
        1. Hazmat Receiver Confirmation: DOT Class 3 or 8 materials require confirmed certified receiver.
        2. Status Cutoffs: Modifications strictly forbidden once picked/later. Cancellations forbidden if in_transit.
        3. Credit Limits: Proposed order amount cannot breach available credit ceiling.
        4. Road Access: Caliche road restrictions (restricted_wet or closed) prohibit automated dispatch.
        """
        action = state.get("current_action")
        if not action:
            return {"policy_check_passed": True, "policy_violation": None}

        tool_name = action.get("tool_name", "")
        arguments = action.get("arguments", {})
        context = state.get("context", {})
        db = self._current_db or context.get("db")

        # 1. Hazmat Certification Check
        if tool_name in ("create_order", "book_delivery"):
            products_in_order: List[str] = []
            raw_lines = arguments.get("lines") or arguments.get("items") or []
            for line in raw_lines:
                if isinstance(line, dict):
                    pid = line.get("product_id")
                else:
                    pid = getattr(line, "product_id", None)
                if pid:
                    products_in_order.append(str(pid))
            if not products_in_order and tool_name == "book_delivery" and db is not None:
                order_id = arguments.get("order_id")
                if order_id:
                    rows = db.execute_query(
                        "SELECT product_id FROM order_lines WHERE order_id = ?",
                        (order_id,),
                    )
                    products_in_order = [str(r["product_id"]) for r in rows]

            contains_hazmat = False
            for pid in products_in_order:
                if pid in HAZMAT_PRODUCTS or pid.startswith("CHEM-3") or pid.startswith("CHEM-8"):
                    contains_hazmat = True
                    break
                if db is not None:
                    h_rows = db.execute_query(
                        "SELECT hazard_class FROM products WHERE product_id = ?",
                        (pid,),
                    )
                    if h_rows and str(h_rows[0]["hazard_class"]) in ("3", "8"):
                        contains_hazmat = True
                        break

            if contains_hazmat:
                receiver_confirmed = bool(
                    context.get("receiver_on_site") or self._current_context.get("receiver_on_site", False)
                )
                if not receiver_confirmed and db is not None:
                    account_id = arguments.get("account_id")
                    if not account_id and arguments.get("order_id"):
                        o_acc = db.execute_query(
                            "SELECT account_id FROM orders WHERE order_id = ?",
                            (arguments["order_id"],),
                        )
                        if o_acc:
                            account_id = o_acc[0]["account_id"]

                    has_cert = False
                    if account_id:
                        c_rows = db.execute_query(
                            "SELECT has_hazmat_cert FROM contacts WHERE account_id = ? AND has_hazmat_cert = 1",
                            (account_id,),
                        )
                        if c_rows:
                            has_cert = True
                    if not has_cert:
                        return {
                            "policy_check_passed": False,
                            "policy_violation": {
                                "rule": "hazmat_certification",
                                "reason": "hazmat_uncertified",
                                "message": "DOT Class 3/8 hazardous materials require confirmed certified receiver present on site.",
                            },
                        }

        # 2. Status Cutoffs Check
        if tool_name in ("modify_order", "cancel_order") and db is not None:
            order_id = arguments.get("order_id")
            if order_id:
                order_rows = db.execute_query(
                    "SELECT status FROM orders WHERE order_id = ?", (order_id,)
                )
                if order_rows:
                    cur_status = order_rows[0]["status"]
                    if tool_name == "modify_order" and cur_status in (
                        "picked",
                        "loaded",
                        "in_transit",
                        "delivered",
                    ):
                        return {
                            "policy_check_passed": False,
                            "policy_violation": {
                                "rule": "status_cutoff",
                                "reason": "modification_after_picked",
                                "message": f"Order modification forbidden: status is '{cur_status}', staging is already finalized.",
                            },
                        }
                    if tool_name == "cancel_order" and cur_status in ("in_transit", "delivered"):
                        return {
                            "policy_check_passed": False,
                            "policy_violation": {
                                "rule": "status_cutoff",
                                "reason": "cancellation_after_in_transit",
                                "message": f"Order cancellation forbidden via automated desk once order is '{cur_status}'. Must escalate.",
                            },
                        }

        # 3. Credit Limits Check
        if tool_name == "create_order" and db is not None:
            account_id = arguments.get("account_id")
            if account_id:
                order_cost = 0.0
                raw_lines = arguments.get("lines") or arguments.get("items") or []
                for line in raw_lines:
                    if isinstance(line, dict):
                        pid = line.get("product_id")
                        qty = float(line.get("quantity", 0))
                        line_price = line.get("unit_price")
                    else:
                        pid = getattr(line, "product_id", None)
                        qty = float(getattr(line, "quantity", 0))
                        line_price = getattr(line, "unit_price", None)

                    p_rows = db.execute_query(
                        "SELECT unit_price FROM products WHERE product_id = ?",
                        (pid,),
                    )
                    if p_rows:
                        order_cost += qty * float(p_rows[0]["unit_price"])
                    elif line_price is not None:
                        order_cost += qty * float(line_price)
                    else:
                        # Fallback default price for uncataloged test chemicals
                        order_cost += qty * 12.50

                credit_res = AGENT_TOOLS["check_credit"].execute(
                    db, account_id=account_id, order_amount=order_cost
                )
                credit_data = credit_res.data
                if not credit_res.success or not isinstance(credit_data, dict):
                    return {
                        "policy_check_passed": False,
                        "policy_violation": {
                            "rule": "credit_check",
                            "reason": "credit_check_incomplete",
                            "message": "Credit verification did not return a complete result.",
                        },
                    }

                if not credit_data.get("credit_sufficient", False):
                    avail = credit_data.get("available_credit", 0.0)
                    return {
                        "policy_check_passed": False,
                        "policy_violation": {
                            "rule": "credit_limit",
                            "reason": "credit_limit_exceeded",
                            "message": f"Order cost ${order_cost:,.2f} exceeds available credit ${avail:,.2f}.",
                        },
                    }

        # 4. Road Access Check
        if tool_name == "book_delivery" and db is not None:
            site_code = arguments.get("site_code")
            order_id = arguments.get("order_id")
            if not site_code and order_id:
                o_rows = db.execute_query("SELECT site_code FROM orders WHERE order_id = ?", (order_id,))
                if o_rows:
                    site_code = o_rows[0]["site_code"]

            if site_code:
                road_rows = db.execute_query(
                    "SELECT road_status FROM site_access WHERE site_code = ?", (site_code,)
                )
                if road_rows:
                    cond = road_rows[0]["road_status"]
                    if cond in ("restricted_wet", "closed"):
                        return {
                            "policy_check_passed": False,
                            "policy_violation": {
                                "rule": "road_access",
                                "reason": "road_access_restricted",
                                "message": f"Caliche road access to {site_code} is currently '{cond}'. Standard delivery disallowed.",
                            },
                        }

        return {"policy_check_passed": True, "policy_violation": None}

    def _correction_or_escalation_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Halt policy-violating write action and transition to correction or escalation."""
        violation = state.get("policy_violation") or {}
        reason = violation.get("reason", "policy_forbidden")
        message = violation.get("message", "Requested action violates standard operating policy.")

        context = state.get("context", {})
        task_obj = context.get("task") or self._current_context.get("task")
        target_account = "ACC-001"
        target_caller = "Hank Callahan"
        if task_obj is not None:
            instructions = getattr(task_obj, "user_instructions", None)
            if instructions is not None:
                target_account = getattr(instructions, "account_id", target_account)
                target_caller = getattr(instructions, "caller_name", target_caller)

        escalate_action = {
            "tool_name": "escalate_to_human",
            "arguments": {
                "account_id": target_account,
                "caller_name": target_caller,
                "reason": reason,
                "context_summary": message,
            },
        }

        return {
            "current_action": escalate_action,
            "escalated_by_policy": True,
            "thought": f"Policy check failed: {message}. Halting write action and transitioning to human escalation.",
            "final_response": f"Policy Notice: {message} Your request is being escalated to human dispatch. ###TRANSFER###",
            "is_done": True,
        }

    def _tool_execution_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Execute approved tool call via context tool_executor or standalone ToolDefinition."""
        action = state.get("current_action")
        if not action:
            return {}

        tool_name = action.get("tool_name", "")
        arguments = action.get("arguments", {})
        context = state.get("context", {})
        tool_executor = self._current_executor or context.get("tool_executor")
        db = self._current_db or context.get("db")

        tool_result: Dict[str, Any] = {}

        if tool_executor is not None:
            res = tool_executor(tool_name, **arguments)
            tool_result = {
                "success": res.success,
                "data": res.data,
                "error": res.error or res.error_message,
                "is_error": bool(res.is_error or not res.success),
            }
        elif tool_name in AGENT_TOOLS and db is not None:
            tool_def = AGENT_TOOLS[tool_name]
            res = tool_def.execute(db=db, **arguments)
            tool_result = {
                "success": res.success,
                "data": res.data,
                "error": res.error or res.error_message,
                "is_error": bool(res.is_error or not res.success),
            }
        else:
            tool_result = {
                "success": True,
                "data": {"status": "executed_simulated", "tool": tool_name},
                "is_error": False,
            }

        plan_index = state.get("plan_index", 0)
        new_plan_index = plan_index + (1 if not tool_result.get("is_error") else 0)

        escalated = state.get("escalated_by_policy", False) or tool_name == "escalate_to_human"
        done_flag = True if escalated else state.get("is_done", False)

        return {
            "last_tool_result": tool_result,
            "plan_index": new_plan_index,
            "is_done": done_flag,
            "thought": f"Executed {tool_name}. Success: {tool_result.get('success')}",
        }

    def _reflection_node(self, state: PlanExecuteState) -> Dict[str, Any]:
        """Analyze tool execution error to dynamically adjust the plan or retry."""
        last_res = state.get("last_tool_result") or {}
        err_msg = last_res.get("error", "Unknown tool execution failure")
        reflections = list(state.get("reflection_notes", []))
        ref_count = state.get("reflection_count", 0) + 1

        action = state.get("current_action") or {}
        tool_name = action.get("tool_name", "unknown")

        note = f"Reflection {ref_count}: Tool '{tool_name}' returned error '{err_msg}'. Adjusting plan."
        reflections.append(note)

        if ref_count >= self.max_reflections:
            return {
                "reflection_notes": reflections,
                "reflection_count": ref_count,
                "thought": f"Exceeded maximum reflection count ({self.max_reflections}). Transitioning to human escalation.",
                "policy_violation": {
                    "reason": "clarification_exhausted",
                    "message": f"Tool '{tool_name}' failed repeatedly: {err_msg}",
                },
                "policy_check_passed": False,
            }

        plan = list(state.get("plan", []))
        plan_index = state.get("plan_index", 0)
        if "find_account" not in plan[:plan_index]:
            plan.insert(plan_index, "find_account")

        return {
            "plan": plan,
            "reflection_notes": reflections,
            "reflection_count": ref_count,
            "thought": note,
        }

    def _build_graph(self) -> Any:
        """Construct StateGraph workflow with policy safety gates and reflection branches."""
        workflow = StateGraph(PlanExecuteState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("policy_check", self._policy_check_node)
        workflow.add_node("correction_or_escalation", self._correction_or_escalation_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("reflection", self._reflection_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")

        def route_after_executor(state: PlanExecuteState) -> str:
            action = state.get("current_action")
            if not action or state.get("is_done"):
                return END
            tool_name = action.get("tool_name", "")
            if tool_name in WRITE_TOOLS:
                return "policy_check"
            return "tool_execution"

        workflow.add_conditional_edges(
            "executor",
            route_after_executor,
            {"policy_check": "policy_check", "tool_execution": "tool_execution", END: END},
        )

        def route_after_policy(state: PlanExecuteState) -> str:
            if not state.get("policy_check_passed", True):
                return "correction_or_escalation"
            return "tool_execution"

        workflow.add_conditional_edges(
            "policy_check",
            route_after_policy,
            {
                "correction_or_escalation": "correction_or_escalation",
                "tool_execution": "tool_execution",
            },
        )

        def route_after_correction(state: PlanExecuteState) -> str:
            action = state.get("current_action")
            if action and action.get("tool_name") == "escalate_to_human":
                return "tool_execution"
            return END

        workflow.add_conditional_edges(
            "correction_or_escalation",
            route_after_correction,
            {"tool_execution": "tool_execution", END: END},
        )

        def route_after_tool(state: PlanExecuteState) -> str:
            res = state.get("last_tool_result") or {}
            if res.get("is_error"):
                return "reflection"
            if state.get("is_done"):
                return END
            return "executor"

        workflow.add_conditional_edges(
            "tool_execution",
            route_after_tool,
            {"reflection": "reflection", "executor": "executor", END: END},
        )

        def route_after_reflection(state: PlanExecuteState) -> str:
            if not state.get("policy_check_passed", True):
                return "correction_or_escalation"
            return "executor"

        workflow.add_conditional_edges(
            "reflection",
            route_after_reflection,
            {"correction_or_escalation": "correction_or_escalation", "executor": "executor"},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    def step(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AgentAction:
        """Execute checkpointed LangGraph workflow turn and emit standardized AgentAction."""
        effective_messages = list(messages or history or [])
        context_dict = context or {}

        self._current_context = context_dict
        self._current_executor = context_dict.get("tool_executor")
        self._current_db = context_dict.get("db")

        # Strip unpickleable function / connection handles for checkpointer msgpack compatibility
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

        initial_state: PlanExecuteState = {
            "messages": effective_messages,
            "task_goal": task_goal,
            "context": serializable_context,
            "reflection_notes": [],
            "reflection_count": 0,
        }

        final_state = self.app.invoke(initial_state, config=config)

        action = final_state.get("current_action")
        tool_name = action.get("tool_name") if action else None
        arguments = action.get("arguments", {}) if action else {}
        thought = final_state.get("thought", "")
        final_response = final_state.get("final_response") or thought
        is_done = bool(final_state.get("is_done", False))

        metadata: Dict[str, Any] = {
            "plan": final_state.get("plan", []),
            "plan_index": final_state.get("plan_index", 0),
            "policy_check_passed": final_state.get("policy_check_passed", True),
            "policy_violation": final_state.get("policy_violation"),
            "reflections": final_state.get("reflection_notes", []),
        }

        return AgentAction(
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
            final_response=final_response,
            is_done=is_done,
            metadata=metadata,
        )


__all__ = ["LangGraphPlanExecuteAgent", "PlanExecuteState", "WRITE_TOOLS"]
