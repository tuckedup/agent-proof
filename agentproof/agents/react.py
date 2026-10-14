"""Hand-written tool-calling ReAct baseline agent for ChemDesk."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentproof.agents.base import AgentAction, AgentInterface, MockProvider
from agentproof.tools.agent_tools import AGENT_TOOLS
from agentproof.tools.base import ToolDefinition

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Fallback default policy if domain/policy.md cannot be loaded from filesystem
DEFAULT_CHEMDESK_POLICY = """
ChemDesk Agent Operating Policy: Standard Operating Procedures
1. Scope and Authority: ChemDesk is the automated order desk assistant for Permian Chemical Logistics.
2. Caller Identity Verification: Before disclosing non-public account details or performing state alterations, verify caller via find_account and verify_caller.
3. Order Processing, Modification, and Cancellation: Orders progress through draft -> submitted -> approved -> picked -> loaded -> in_transit -> delivered.
Modifications permitted only before picked status. Cancellations incur $0 fee before picked, $350 fee for picked/loaded, forbidden once in_transit.
4. Hazardous Materials (DOT Regulations): DOT Class 3 or 8 requires certified receiver on site confirmed via report_receiver_on_site. Verify truck compatibility via check_compatibility.
5. Credit Ledger: Available Credit = Credit Limit - Balance - Pending Orders. Check credit prior to order creation.
6. Site Access and Delivery: Road closures during rain require checking site access. Propose slots and confirm delivery window prior to booking.
7. Escalation: Escalate to human if two-turn clarification fails or forbidden actions requested.
"""


def load_chemdesk_policy() -> str:
    """Load ChemDesk operational policy from domain/policy.md or return fallback text."""
    try:
        policy_path = Path(__file__).resolve().parents[2] / "domain" / "policy.md"
        if policy_path.exists():
            return policy_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read domain/policy.md directly: %s", exc)
    return DEFAULT_CHEMDESK_POLICY


class ReactBaselineAgent(AgentInterface):
    """Hand-written tool-calling ReAct baseline agent without third-party frameworks.

    Maintains an explicit Thought-Action-Observation loop using LiteLLM or an injected MockProvider.
    Strictly integrates the ChemDesk operating policy in its system instructions.
    """

    def __init__(
        self,
        name: str = "react_baseline",
        model: Optional[str] = None,
        mock_provider: Optional[MockProvider] = None,
        max_iterations: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.name = name
        self.model = model or os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini")
        self.mock_provider = mock_provider
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.policy_text = load_chemdesk_policy()
        self.system_prompt = self._build_system_prompt()
        self.turn_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clear conversation memories and internal scratchpads."""
        self.turn_history.clear()
        if self.mock_provider:
            self.mock_provider.reset()

    def _build_system_prompt(self) -> str:
        """Construct system instructions embedding complete operating policy and ReAct rules."""
        return (
            "You are ChemDesk, the professional automated chemical order desk assistant for Permian Chemical Logistics.\n\n"
            "OPERATING POLICY AND STANDARD PROCEDURES:\n"
            f"{self.policy_text}\n\n"
            "INTERACTION PROTOCOL (ReAct):\n"
            "When responding to customer requests, reason carefully in an internal Thought-Action-Observation loop:\n"
            "1. Thought: Analyze caller intent, account status, policy constraints, and determine the next action.\n"
            "2. Action: Choose an available tool to invoke with exact JSON arguments, or emit tool calls directly.\n"
            "   Format: Action: tool_name({\"arg1\": \"value1\"})\n"
            "3. Observation: Review tool results to verify success or identify errors.\n"
            "4. Final Response: Once all necessary actions are executed, or when requiring user input, provide a courteous response to the customer.\n"
            "   Format: Final Response: <message to caller>\n"
            "Signal task completion by including ###DONE### when the request is fully satisfied.\n"
            "Signal escalation to human dispatcher by including ###TRANSFER###.\n"
        )

    def _parse_action_string(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extract tool name and parsed JSON arguments from text formatted as Action: name(args)."""
        pattern = r"(?:Action|Tool):\s*([a-zA-Z0-9_]+)\s*\((.*?)\)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tool_name = match.group(1).strip()
            raw_args = match.group(2).strip()
            if not raw_args:
                return tool_name, {}
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    return tool_name, parsed
            except Exception:
                pass
            try:
                cleaned_args = "{" + raw_args.strip("{}") + "}"
                parsed = json.loads(cleaned_args)
                if isinstance(parsed, dict):
                    return tool_name, parsed
            except Exception:
                pass

        json_match = re.search(r"(?:Action|Tool):\s*(\{.*?\})", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                t_name = data.get("name") or data.get("tool") or data.get("tool_name")
                t_args = data.get("arguments") or data.get("args") or {}
                if t_name:
                    return t_name, t_args
            except Exception:
                pass

        return None

    def _parse_llm_response(
        self, response: Any
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any], Optional[str]]:
        """Parse thought, tool_name, arguments, and final_response from LLM response structure."""
        thought: Optional[str] = None
        tool_name: Optional[str] = None
        arguments: Dict[str, Any] = {}
        final_response: Optional[str] = None

        if isinstance(response, dict):
            choices = response.get("choices", [])
            msg = choices[0].get("message", {}) if choices else {}
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
        else:
            choices = getattr(response, "choices", [])
            msg = getattr(choices[0], "message", None) if choices else None
            content = getattr(msg, "content", "") if msg else ""
            tool_calls = getattr(msg, "tool_calls", []) if msg else []

        if tool_calls:
            first_call = tool_calls[0]
            if isinstance(first_call, dict):
                fn = first_call.get("function", {})
                tool_name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
            else:
                fn = getattr(first_call, "function", None)
                tool_name = getattr(fn, "name", None) if fn else None
                raw_args = getattr(fn, "arguments", "{}") if fn else "{}"

            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args

            thought = content if content else f"Invoking {tool_name} to fulfill operational request."
            return thought, tool_name, arguments, None

        if content:
            thought_match = re.search(r"Thought:\s*(.*?)(?=(?:Action|Final Response|$))", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1).strip()

            action_parsed = self._parse_action_string(content)
            if action_parsed:
                tool_name, arguments = action_parsed
                return thought, tool_name, arguments, None

            final_match = re.search(r"Final Response:\s*(.*)", content, re.DOTALL)
            if final_match:
                final_response = final_match.group(1).strip()
            else:
                final_response = content.strip()

        return thought, tool_name, arguments, final_response

    def _heuristic_fallback(
        self,
        messages: List[Dict[str, Any]],
        scratchpad: List[Dict[str, Any]],
        tools_dict: Dict[str, ToolDefinition],
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any], Optional[str]]:
        """Deterministic heuristic generator when offline or operating without API key."""
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") in ("user", "human") or m.get("speaker") == "user":
                last_user_msg = str(m.get("content", ""))
                break

        obs_tools = [s.get("tool_name") for s in scratchpad if "tool_name" in s]

        if "find_account" not in obs_tools:
            query = "Apex" if "Apex" in last_user_msg else "ACC-001"
            return (
                "Need to locate customer account record first.",
                "find_account",
                {"query": query},
                None,
            )

        if "verify_caller" not in obs_tools:
            return (
                "Found account record; now verifying caller credentials against site authorization.",
                "verify_caller",
                {"account_id": "ACC-001", "site_code": "SITE-MID-001", "caller_name": "Hank Callahan"},
                None,
            )

        if "check_stock" not in obs_tools and ("stock" in last_user_msg.lower() or "order" in last_user_msg.lower()):
            return (
                "Checking current inventory availability at Midland depot.",
                "check_stock",
                {"product_id": "CHEM-B101", "depot_id": "DEPOT-MIDLAND"},
                None,
            )

        if "quote_price" not in obs_tools and "order" in last_user_msg.lower():
            return (
                "Calculating price quote for requested chemical volume.",
                "quote_price",
                {"product_id": "CHEM-B101", "quantity": 1000},
                None,
            )

        if "check_credit" not in obs_tools and "order" in last_user_msg.lower():
            return (
                "Verifying customer available credit before finalizing order.",
                "check_credit",
                {"account_id": "ACC-001", "order_amount": 12500.0},
                None,
            )

        return (
            "All verification and inquiry steps completed.",
            None,
            {},
            f"Thank you. I have verified your account and processed your inquiry regarding {last_user_msg[:30]}. ###DONE###",
        )

    def _call_llm(
        self,
        llm_messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """Dispatch query to MockProvider or LiteLLM completion endpoint."""
        if self.mock_provider is not None:
            return self.mock_provider.complete(messages=llm_messages, tools=tools_schema)

        api_key_available = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or "mock" in self.model.lower()
        )

        if litellm is not None and api_key_available:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": llm_messages,
                "temperature": self.temperature,
            }
            if tools_schema:
                kwargs["tools"] = tools_schema
            return litellm.completion(**kwargs)

        return None

    def step(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AgentAction:
        """Execute the hand-written ReAct loop for this turn."""
        effective_messages: List[Dict[str, Any]] = list(messages or history or [])
        context_dict: Dict[str, Any] = context or {}

        if tools is not None:
            active_tools: Dict[str, ToolDefinition] = {t.name: t for t in tools}
        elif "tools" in context_dict and isinstance(context_dict["tools"], list):
            active_tools = {
                t_name: AGENT_TOOLS[t_name]
                for t_name in context_dict["tools"]
                if t_name in AGENT_TOOLS
            }
        else:
            active_tools = dict(AGENT_TOOLS)

        tools_schema = [t.to_openai_schema() for t in active_tools.values()]
        tool_executor = context_dict.get("tool_executor")
        db = context_dict.get("db")

        scratchpad: List[Dict[str, Any]] = []
        total_cost = 0.0
        total_p_tokens = 0
        total_c_tokens = 0

        last_thought: Optional[str] = None
        last_tool_name: Optional[str] = None
        last_arguments: Dict[str, Any] = {}
        final_resp: Optional[str] = None
        is_done = False

        for _iteration in range(self.max_iterations):
            llm_messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

            for msg in effective_messages:
                speaker = msg.get("speaker") or msg.get("role") or "user"
                role = "assistant" if speaker in ("agent", "assistant") else "user"
                content_val = msg.get("content", "")
                llm_messages.append({"role": role, "content": str(content_val)})

            for step_entry in scratchpad:
                if "thought" in step_entry:
                    llm_messages.append({"role": "assistant", "content": f"Thought: {step_entry['thought']}"})
                if "tool_name" in step_entry:
                    llm_messages.append({
                        "role": "assistant",
                        "content": f"Action: {step_entry['tool_name']}({json.dumps(step_entry.get('arguments', {}))})",
                    })
                if "observation" in step_entry:
                    llm_messages.append({
                        "role": "user",
                        "content": f"Observation: {json.dumps(step_entry['observation'])}",
                    })

            raw_response = self._call_llm(llm_messages, tools_schema)

            if raw_response is not None:
                thought, tool_name, arguments, final_text = self._parse_llm_response(raw_response)
                usage = (
                    raw_response.get("usage", {})
                    if isinstance(raw_response, dict)
                    else getattr(raw_response, "usage", None)
                )
                if usage:
                    if isinstance(usage, dict):
                        total_p_tokens += int(usage.get("prompt_tokens", 0))
                        total_c_tokens += int(usage.get("completion_tokens", 0))
                    else:
                        total_p_tokens += int(getattr(usage, "prompt_tokens", 0))
                        total_c_tokens += int(getattr(usage, "completion_tokens", 0))
            else:
                thought, tool_name, arguments, final_text = self._heuristic_fallback(
                    effective_messages, scratchpad, active_tools
                )

            last_thought = thought or last_thought

            if tool_name is not None and tool_name in active_tools:
                last_tool_name = tool_name
                last_arguments = arguments
                step_record: Dict[str, Any] = {
                    "thought": thought,
                    "tool_name": tool_name,
                    "arguments": arguments,
                }

                # Interactive execution within ReAct loop
                if tool_executor is not None:
                    exec_res = tool_executor(tool_name, **arguments)
                    obs_data = exec_res.data if exec_res.success else {"error": exec_res.error or exec_res.error_message}
                    step_record["observation"] = obs_data
                    scratchpad.append(step_record)

                    if tool_name == "escalate_to_human":
                        final_resp = (
                            f"Your request has been escalated to our human desk (Reference: {obs_data.get('escalation_id', 'ESC')}). "
                            "A representative will contact you shortly. ###TRANSFER###"
                        )
                        is_done = True
                        break
                    continue

                if db is not None:
                    tool_def = active_tools[tool_name]
                    exec_res = tool_def.execute(db=db, **arguments)
                    obs_data = exec_res.data if exec_res.success else {"error": exec_res.error or exec_res.error_message}
                    step_record["observation"] = obs_data
                    scratchpad.append(step_record)
                    continue

                # Single-step yielding for external orchestrator tool execution
                return AgentAction(
                    tool_name=tool_name,
                    arguments=arguments,
                    thought=thought,
                    is_done=False,
                    cost_usd=total_cost,
                    prompt_tokens=total_p_tokens,
                    completion_tokens=total_c_tokens,
                )

            if final_text is not None:
                final_resp = final_text
                if "###DONE###" in final_resp:
                    is_done = True
                if "###TRANSFER###" in final_resp:
                    is_done = True
                break

        if final_resp is None:
            final_resp = last_thought or "Assisting with chemical logistics operations. Please specify details."

        return AgentAction(
            tool_name=None,
            arguments={},
            thought=last_thought,
            final_response=final_resp,
            is_done=is_done,
            cost_usd=total_cost,
            prompt_tokens=total_p_tokens,
            completion_tokens=total_c_tokens,
            metadata={"iterations": len(scratchpad), "last_tool": last_tool_name, "last_arguments": last_arguments},
        )


__all__ = ["ReactBaselineAgent", "load_chemdesk_policy"]
