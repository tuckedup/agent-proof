"""Deterministic mock provider for offline testing and continuous integration.

Replays recorded responses matched by prompt hash, tool intents, or predefined
domain test scenarios without requiring real API keys or incurring financial costs.
Records live API invocations when AGENTPROOF_RECORD_MOCKS=1.
"""

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]

ModelResponse: Any = getattr(litellm, "ModelResponse", None) if litellm is not None else None


class MockToolCall(BaseModel):
    """Specification of a tool call within a simulated LLM completion."""

    id: str = Field(default_factory=lambda: f"call_mock_{uuid.uuid4().hex[:8]}")
    name: str = Field(description="Name of the tool to be called.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of arguments to pass to the tool.",
    )


class MockScenario(BaseModel):
    """Predefined domain scenario mapping intents to mock responses."""

    scenario_id: str = Field(description="Identifier for this test scenario.")
    category: str = Field(description="Domain category for this scenario.")
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords used to match this scenario against prompt messages.",
    )
    target_tool: Optional[str] = Field(
        default=None,
        description="Primary tool name associated with this scenario.",
    )
    tool_arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Predefined arguments for the target tool.",
    )
    text_content: Optional[str] = Field(
        default=None,
        description="Text content emitted if tool is not invoked or for direct answers.",
    )


def make_mock_response(
    content: Optional[str] = None,
    tool_calls: Optional[List[Union[Dict[str, Any], MockToolCall]]] = None,
    model: str = "mock-model",
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
) -> Any:
    """Construct a synthetic ModelResponse compatible with LiteLLM and OpenAI formats."""
    formatted_tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason = "stop"

    if tool_calls:
        finish_reason = "tool_calls"
        formatted_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, MockToolCall):
                formatted_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                })
            elif isinstance(tc, dict):
                if "function" in tc and "id" in tc:
                    formatted_tool_calls.append(tc)
                else:
                    tool_name = tc.get("name", tc.get("tool_name", "unknown_tool"))
                    args = tc.get("arguments", tc.get("args", {}))
                    arg_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    formatted_tool_calls.append({
                        "id": tc.get("id", f"call_mock_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arg_str,
                        },
                    })

    response_dict: Dict[str, Any] = {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": formatted_tool_calls,
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    if litellm is not None and ModelResponse is not None:
        try:
            return ModelResponse(**response_dict)
        except Exception:
            pass

    return response_dict


# Predefined domain scenarios for ChemDesk order desk operations
PREDEFINED_SCENARIOS: List[MockScenario] = [
    # 1. Account lookup and caller verification
    MockScenario(
        scenario_id="scenario_find_account",
        category="account_lookup",
        keywords=["find_account", "account", "lookup", "apex", "acc-001", "customer"],
        target_tool="find_account",
        tool_arguments={"query": "Apex"},
        text_content="Account ACC-001 for Apex Solutions is active with standard 30-day payment terms.",
    ),
    MockScenario(
        scenario_id="scenario_verify_caller",
        category="caller_verification",
        keywords=["verify_caller", "verify", "caller", "identity", "site-mid-001", "authorized"],
        target_tool="verify_caller",
        tool_arguments={
            "account_id": "ACC-001",
            "site_code": "SITE-MID-001",
            "caller_name": "John Doe",
        },
        text_content="Caller John Doe is verified as Site Supervisor for Apex Solutions at well site SITE-MID-001.",
    ),
    # 2. Stock check and quoting
    MockScenario(
        scenario_id="scenario_check_stock",
        category="stock_check",
        keywords=["check_stock", "stock", "inventory", "available", "chem-corr-01", "depot-001"],
        target_tool="check_stock",
        tool_arguments={"product_id": "CHEM-CORR-01", "depot_id": "DEPOT-001"},
        text_content="Depot DEPOT-001 currently holds 2,500 gallons of CHEM-CORR-01 in available stock.",
    ),
    MockScenario(
        scenario_id="scenario_quote_price",
        category="quoting",
        keywords=["quote_price", "quote", "price", "pricing", "cost", "gallon", "discount"],
        target_tool="quote_price",
        tool_arguments={"product_id": "CHEM-CORR-01", "quantity": 500.0},
        text_content="Quoted price for 500 gallons of CHEM-CORR-01 is $12.50 per gallon with volume discount, total $6,250.00.",
    ),
    # 3. Order creation, modification, cancellation
    MockScenario(
        scenario_id="scenario_create_order",
        category="order_creation",
        keywords=["create_order", "place order", "submit order", "new order"],
        target_tool="create_order",
        tool_arguments={
            "account_id": "ACC-001",
            "site_code": "SITE-MID-001",
            "items": [{"product_id": "CHEM-CORR-01", "quantity": 500.0}],
            "notes": "Standard delivery to Permian basin pad.",
        },
        text_content="Order ORD-0001 has been created for account ACC-001 at site SITE-MID-001.",
    ),
    MockScenario(
        scenario_id="scenario_modify_order",
        category="order_modification",
        keywords=["modify_order", "modify", "update order", "change quantity", "add item"],
        target_tool="modify_order",
        tool_arguments={
            "order_id": "ORD-0001",
            "action": "update_quantity",
            "product_id": "CHEM-CORR-01",
            "quantity": 600.0,
        },
        text_content="Order ORD-0001 quantity successfully updated to 600 gallons.",
    ),
    MockScenario(
        scenario_id="scenario_cancel_order",
        category="order_cancellation",
        keywords=["cancel_order", "cancel", "abort order", "cancellation"],
        target_tool="cancel_order",
        tool_arguments={
            "order_id": "ORD-0001",
            "reason": "Duplicate order placed by site supervisor.",
        },
        text_content="Order ORD-0001 has been canceled per request. Cancellation fee waived.",
    ),
    MockScenario(
        scenario_id="scenario_get_order",
        category="order_lookup",
        keywords=["get_order", "order status", "ord-0001", "lookup order"],
        target_tool="get_order",
        tool_arguments={"order_id": "ORD-0001"},
        text_content="Order ORD-0001 is currently scheduled for delivery to SITE-MID-001.",
    ),
    MockScenario(
        scenario_id="scenario_list_orders",
        category="order_listing",
        keywords=["list_orders", "all orders", "recent orders"],
        target_tool="list_orders",
        tool_arguments={"account_id": "ACC-001"},
        text_content="Found 2 active orders for account ACC-001.",
    ),
    # 4. Delivery slot search and booking
    MockScenario(
        scenario_id="scenario_list_delivery_slots",
        category="delivery_slot_search",
        keywords=["list_delivery_slots", "delivery slot", "slots", "dispatch window", "delivery date"],
        target_tool="list_delivery_slots",
        tool_arguments={"depot_id": "DEPOT-001", "delivery_date": "2026-10-15"},
        text_content="Available slots from Midland depot: SLOT-0001 (08:00 to 12:00) and SLOT-0002 (13:00 to 17:00).",
    ),
    MockScenario(
        scenario_id="scenario_book_delivery",
        category="delivery_booking",
        keywords=["book_delivery", "book delivery", "reserve slot", "confirm slot", "assign slot"],
        target_tool="book_delivery",
        tool_arguments={"order_id": "ORD-0001", "slot_id": "SLOT-0001"},
        text_content="Delivery slot SLOT-0001 booked for order ORD-0001 on 2026-10-15.",
    ),
    # 5. Hazmat compatibility check and certified receiver checks
    MockScenario(
        scenario_id="scenario_check_compatibility",
        category="hazmat_compatibility",
        keywords=["check_compatibility", "hazmat", "compatibility", "co-load", "incompatible", "segregation"],
        target_tool="check_compatibility",
        tool_arguments={"product_ids": ["CHEM-CORR-01", "CHEM-SCALE-02"]},
        text_content="Hazmat compatibility check passed: CHEM-CORR-01 and CHEM-SCALE-02 can safely share cargo compartments.",
    ),
    MockScenario(
        scenario_id="scenario_receiver_check",
        category="certified_receiver",
        keywords=["receiver", "certified receiver", "hazmat cert", "endorsement", "report_receiver_on_site"],
        target_tool="verify_caller",
        tool_arguments={
            "account_id": "ACC-001",
            "site_code": "SITE-MID-001",
            "caller_name": "Certified Lead",
        },
        text_content="Receiver certification confirmed on pad. Authorized recipient holds current Hazmat endorsement.",
    ),
    # 6. Credit checks and policy escalations
    MockScenario(
        scenario_id="scenario_check_credit",
        category="credit_check",
        keywords=["check_credit", "credit", "credit limit", "balance", "credit check"],
        target_tool="check_credit",
        tool_arguments={"account_id": "ACC-001", "order_amount": 6250.0},
        text_content="Credit check approved: Account has $35,000 available credit against a $50,000 limit.",
    ),
    MockScenario(
        scenario_id="scenario_escalate_to_human",
        category="policy_escalation",
        keywords=["escalate_to_human", "escalate", "human desk", "transfer", "policy exception", "supervisor approval"],
        target_tool="escalate_to_human",
        tool_arguments={
            "reason": "Credit limit exceeded by over $5,000.",
            "context_summary": "Order requested exceeds remaining credit balance and requires finance desk override.",
            "account_id": "ACC-001",
            "caller_name": "John Doe",
        },
        text_content="I am escalating your request to our human order desk lead due to credit policy limits. Transferring now. ###TRANSFER###",
    ),
    # 7. LangGraph supervisor agent routing responses
    MockScenario(
        scenario_id="scenario_supervisor_routing_order",
        category="supervisor_routing",
        keywords=["supervisor", "routing", "route", "next agent", "order desk specialist"],
        target_tool=None,
        tool_arguments={},
        text_content='{"next": "order_agent", "instruction": "Proceed with account verification and product ordering."}',
    ),
    MockScenario(
        scenario_id="scenario_supervisor_routing_compliance",
        category="supervisor_routing",
        keywords=["compliance", "hazmat review", "safety check"],
        target_tool=None,
        tool_arguments={},
        text_content='{"next": "compliance_agent", "instruction": "Verify hazmat compatibility and receiver certifications."}',
    ),
    MockScenario(
        scenario_id="scenario_supervisor_routing_finish",
        category="supervisor_routing",
        keywords=["done", "finish", "complete task"],
        target_tool=None,
        tool_arguments={},
        text_content='{"next": "FINISH", "instruction": "All order desk tasks completed successfully."}',
    ),
]


class MockProvider:
    """Mock LLM provider that replays recorded responses and simulates domain tool calls.

    Matches requests by prompt hash, tool intents, or predefined test scenarios.
    Records live API interactions when the AGENTPROOF_RECORD_MOCKS environment variable is set to 1.
    """

    def __init__(
        self,
        records_path: Optional[Union[str, Path]] = None,
        enable_recording: Optional[bool] = None,
    ) -> None:
        if records_path is not None:
            self._records_path = Path(records_path)
        else:
            self._records_path = Path(".cache") / "mock_records.json"

        if enable_recording is not None:
            self._enable_recording = enable_recording
        else:
            self._enable_recording = os.environ.get("AGENTPROOF_RECORD_MOCKS") == "1"

        self._lock = threading.Lock()
        self._recorded_mocks: Dict[str, Any] = {}
        self._custom_responses: Dict[str, Any] = {}
        self._custom_handlers: List[Callable[..., Optional[Any]]] = []
        self._load_recorded_mocks()

    @property
    def enable_recording(self) -> bool:
        """Whether live call recording is currently active."""
        return self._enable_recording

    @enable_recording.setter
    def enable_recording(self, value: bool) -> None:
        self._enable_recording = value

    def _load_recorded_mocks(self) -> None:
        """Load previously recorded mock calls from disk if available."""
        if self._records_path.is_file():
            try:
                with open(self._records_path, "r", encoding="utf-8") as f:
                    self._recorded_mocks = json.load(f)
            except Exception:
                self._recorded_mocks = {}

    def register_response(self, key_or_pattern: str, response: Any) -> None:
        """Register a custom deterministic response for a specific key or prompt pattern."""
        with self._lock:
            self._custom_responses[key_or_pattern] = response

    def register_handler(self, handler: Callable[..., Optional[Any]]) -> None:
        """Register a custom callback handler to resolve responses dynamically."""
        with self._lock:
            self._custom_handlers.append(handler)

    def record(self, key: str, request_data: Any, response_data: Any) -> None:
        """Persist a live request and response pair to the mock records store."""
        with self._lock:
            # Normalize response data
            if hasattr(response_data, "model_dump") and callable(getattr(response_data, "model_dump")):
                serialized_resp = response_data.model_dump()
            elif hasattr(response_data, "dict") and callable(getattr(response_data, "dict")):
                serialized_resp = response_data.dict()
            elif isinstance(response_data, dict):
                serialized_resp = response_data
            else:
                serialized_resp = str(response_data)

            self._recorded_mocks[key] = {
                "request": request_data,
                "response": serialized_resp,
            }

            try:
                self._records_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._records_path, "w", encoding="utf-8") as f:
                    json.dump(self._recorded_mocks, f, indent=2, sort_keys=True)
            except OSError:
                pass

    def _extract_available_tool_names(self, tools: Optional[List[Any]]) -> List[str]:
        """Extract tool function names from provided tool specifications."""
        if not tools:
            return []
        names = []
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function")
                if isinstance(fn, dict) and "name" in fn:
                    names.append(fn["name"])
                elif "name" in t:
                    names.append(t["name"])
                elif "tool_name" in t:
                    names.append(t["tool_name"])
            elif hasattr(t, "name"):
                names.append(getattr(t, "name"))
            elif isinstance(t, str):
                names.append(t)
        return names

    def _extract_conversation_text(self, messages: Optional[List[Dict[str, Any]]]) -> str:
        """Aggregate text content from conversation messages for intent analysis."""
        if not messages:
            return ""
        fragments = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                fragments.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        fragments.append(item.get("text", ""))
        return " ".join(fragments)

    def _match_scenario(
        self,
        text: str,
        available_tools: List[str],
    ) -> Optional[MockScenario]:
        """Match conversation text and available tools against predefined domain scenarios."""
        lower_text = text.lower()

        # Prioritize scenarios where the target tool is in available_tools
        matched_candidates: List[tuple[int, MockScenario]] = []
        for scenario in PREDEFINED_SCENARIOS:
            score = 0
            # Check target tool availability
            if scenario.target_tool and scenario.target_tool in available_tools:
                score += 5
            # Match keywords
            for kw in scenario.keywords:
                if kw in lower_text:
                    score += 2
            if score > 2:
                matched_candidates.append((score, scenario))

        if matched_candidates:
            matched_candidates.sort(key=lambda x: x[0], reverse=True)
            return matched_candidates[0][1]

        return None

    def get_response(
        self,
        model: str = "mock-model",
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        cache_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Produce a deterministic simulated response for the given parameters.

        Priority order:
        1. Custom handler callbacks.
        2. Recorded mock matches by cache_key or prompt string.
        3. Custom registered responses.
        4. Domain scenario matching based on tool intents and conversation keywords.
        5. Natural language conversational defaults.
        """
        # 1. Custom handlers
        for handler in self._custom_handlers:
            try:
                res = handler(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    cache_key=cache_key,
                    **kwargs,
                )
                if res is not None:
                    return res
            except Exception:
                pass

        # 2. Recorded mocks lookup
        if cache_key and cache_key in self._recorded_mocks:
            entry = self._recorded_mocks[cache_key]
            resp_data = entry.get("response", entry)
            if litellm is not None and ModelResponse is not None and isinstance(resp_data, dict):
                try:
                    return ModelResponse(**resp_data)
                except Exception:
                    pass
            return resp_data

        # 3. Custom response registered by prompt text or key
        text = self._extract_conversation_text(messages)
        if text:
            for pattern, custom_resp in self._custom_responses.items():
                if pattern in text or pattern == cache_key:
                    return custom_resp

        # 4. Domain scenario matching
        available_tools = self._extract_available_tool_names(tools)
        scenario = self._match_scenario(text, available_tools)

        if scenario is not None:
            if scenario.target_tool and scenario.target_tool in available_tools:
                # Generate tool call response
                tool_call = MockToolCall(
                    name=scenario.target_tool,
                    arguments=scenario.tool_arguments,
                )
                return make_mock_response(
                    content=None,
                    tool_calls=[tool_call],
                    model=model,
                    prompt_tokens=140,
                    completion_tokens=45,
                )
            elif scenario.text_content:
                # Generate text content response
                return make_mock_response(
                    content=scenario.text_content,
                    tool_calls=None,
                    model=model,
                    prompt_tokens=110,
                    completion_tokens=35,
                )

        # 5. Check if LangGraph supervisor routing is requested
        lower_text = text.lower()
        if "supervisor" in lower_text or "route" in lower_text or "routing" in lower_text:
            routing_json = json.dumps({"next": "order_agent", "instruction": "Process customer chemical order."})
            return make_mock_response(
                content=routing_json,
                tool_calls=None,
                model=model,
                prompt_tokens=90,
                completion_tokens=25,
            )

        # 6. Fallback tool call if tools exist
        if available_tools:
            first_tool = available_tools[0]
            dummy_args: Dict[str, Any] = {}
            if first_tool == "find_account":
                dummy_args = {"query": "Apex"}
            elif first_tool == "verify_caller":
                dummy_args = {"account_id": "ACC-001", "site_code": "SITE-MID-001", "caller_name": "John Doe"}
            elif first_tool == "check_stock":
                dummy_args = {"product_id": "CHEM-CORR-01"}
            elif first_tool == "quote_price":
                dummy_args = {"product_id": "CHEM-CORR-01", "quantity": 500.0}

            return make_mock_response(
                content=None,
                tool_calls=[MockToolCall(name=first_tool, arguments=dummy_args)],
                model=model,
                prompt_tokens=120,
                completion_tokens=35,
            )

        # 7. Fallback natural language assistant dialogue
        default_reply = (
            "ChemDesk Order Desk assistant ready. "
            "Please provide your account ID, well site code, and chemical product requirements."
        )
        return make_mock_response(
            content=default_reply,
            tool_calls=None,
            model=model,
            prompt_tokens=80,
            completion_tokens=25,
        )
