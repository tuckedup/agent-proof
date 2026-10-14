"""Cost tracking and budget enforcement for LLM provider calls.

Monitors prompt tokens, completion tokens, and dollar expenditures.
Enforces the MAX_RUN_USD budget ceiling and raises BudgetExceededError
when cumulative run costs reach or exceed the configured limit.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


# Fallback token pricing per 1,000,000 tokens (USD)
FALLBACK_TOKEN_PRICING: Dict[str, Dict[str, float]] = {
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "openai/gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "openai/gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "anthropic/claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "anthropic/claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    "anthropic/claude-3-opus-20240229": {"prompt": 15.00, "completion": 75.00},
    "default": {"prompt": 1.50, "completion": 5.00},
}


class BudgetExceededError(Exception):
    """Raised when total run cost reaches or exceeds the MAX_RUN_USD budget limit."""

    def __init__(
        self,
        current_cost: float,
        max_budget: float,
        message: Optional[str] = None,
    ) -> None:
        self.current_cost = current_cost
        self.max_budget = max_budget
        msg = (
            message
            or f"Budget ceiling exceeded: cumulative run cost ${current_cost:.6f} "
            f"reached or exceeded max budget ${max_budget:.6f}"
        )
        super().__init__(msg)


class CostRecord(BaseModel):
    """Individual record of token usage and dollar cost for a single LLM call."""

    model: str = Field(description="Name or identifier of the LLM model used.")
    prompt_tokens: int = Field(default=0, description="Number of prompt input tokens.")
    completion_tokens: int = Field(default=0, description="Number of completion output tokens.")
    total_tokens: int = Field(default=0, description="Sum of prompt and completion tokens.")
    cost_usd: float = Field(default=0.0, description="Calculated dollar cost for this call.")
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when this call was tracked.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution context such as task_id or agent_role.",
    )


class CostSummary(BaseModel):
    """Cumulative usage and financial summary for an episode or simulation run."""

    total_prompt_tokens: int = Field(default=0, description="Cumulative prompt tokens.")
    total_completion_tokens: int = Field(default=0, description="Cumulative completion tokens.")
    total_tokens: int = Field(default=0, description="Cumulative total tokens.")
    total_cost_usd: float = Field(default=0.0, description="Cumulative dollar cost.")
    max_run_usd: float = Field(default=5.0, description="Configured budget limit in USD.")
    remaining_budget_usd: float = Field(
        default=5.0,
        description="Remaining budget before BudgetExceededError.",
    )
    call_count: int = Field(default=0, description="Total number of tracked LLM calls.")


def calculate_fallback_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate approximate USD cost using fallback token pricing tables.

    Costs are computed per million tokens and rounded to 8 decimal places.
    """
    cleaned_model = model.lower().strip()
    pricing = FALLBACK_TOKEN_PRICING.get(cleaned_model)
    if not pricing:
        for key, candidate in FALLBACK_TOKEN_PRICING.items():
            if key != "default" and (key in cleaned_model or cleaned_model in key):
                pricing = candidate
                break
    if not pricing:
        pricing = FALLBACK_TOKEN_PRICING["default"]

    prompt_cost = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000.0) * pricing["completion"]
    return round(prompt_cost + completion_cost, 8)


class CostTracker:
    """Tracks token usage and dollar expenditures across LLM calls in a thread-safe manner.

    Enforces the MAX_RUN_USD environment variable (default 5.0) and raises BudgetExceededError
    when total run spending reaches or exceeds the budget limit.
    """

    def __init__(self, max_run_usd: Optional[float] = None) -> None:
        if max_run_usd is not None:
            self._max_run_usd = float(max_run_usd)
        else:
            raw_env = os.environ.get("MAX_RUN_USD", "5.0")
            try:
                self._max_run_usd = float(raw_env)
            except ValueError:
                self._max_run_usd = 5.0

        self._lock = threading.Lock()
        self._records: List[CostRecord] = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0
        self._total_cost_usd = 0.0

    @property
    def max_run_usd(self) -> float:
        """Configured budget ceiling in USD."""
        return self._max_run_usd

    @property
    def total_prompt_tokens(self) -> int:
        """Cumulative count of prompt tokens across all tracked calls."""
        with self._lock:
            return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """Cumulative count of completion tokens across all tracked calls."""
        with self._lock:
            return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        """Cumulative count of prompt and completion tokens across all tracked calls."""
        with self._lock:
            return self._total_tokens

    @property
    def total_cost_usd(self) -> float:
        """Cumulative dollar expenditure across all tracked calls."""
        with self._lock:
            return self._total_cost_usd

    @property
    def remaining_budget_usd(self) -> float:
        """Remaining dollar budget before reaching MAX_RUN_USD."""
        with self._lock:
            return max(0.0, self._max_run_usd - self._total_cost_usd)

    @property
    def records(self) -> List[CostRecord]:
        """List of all tracked cost records."""
        with self._lock:
            return list(self._records)

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        response: Optional[Any] = None,
    ) -> float:
        """Calculate dollar cost for an LLM call using LiteLLM or fallback token pricing.

        Attempts litellm.completion_cost first when response is provided.
        Falls back to per-model token pricing if calculation is unavailable.
        """
        if response is not None and litellm is not None:
            try:
                cost = litellm.completion_cost(completion_response=response)
                if cost is not None and isinstance(cost, (int, float)) and cost >= 0:
                    return float(cost)
            except Exception:
                pass

        return calculate_fallback_cost(model, prompt_tokens, completion_tokens)

    def track(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: Optional[float] = None,
        response: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostRecord:
        """Record token usage and financial cost for a single LLM invocation.

        Raises BudgetExceededError if the cumulative run cost reaches or exceeds MAX_RUN_USD.
        """
        if cost_usd is None:
            cost_usd = self.calculate_cost(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response=response,
            )

        total_tokens = prompt_tokens + completion_tokens
        record = CostRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            self._total_cost_usd += cost_usd
            current_total = self._total_cost_usd

        if current_total >= self._max_run_usd:
            raise BudgetExceededError(
                current_cost=current_total,
                max_budget=self._max_run_usd,
            )

        return record

    def track_response(
        self,
        response: Any,
        model: Optional[str] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostRecord:
        """Extract token usage from a LiteLLM response object or dictionary and track cost.

        Supports litellm.ModelResponse, standard dictionaries, or custom objects with usage.
        """
        prompt_tokens = 0
        completion_tokens = 0
        resolved_model = model or "unknown"

        # Extract usage attributes from ModelResponse or dictionary
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")

        if usage is not None:
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
            else:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
                completion_tokens = int(getattr(usage, "completion_tokens", 0))

        # Extract model identifier if not explicitly passed
        if model is None:
            resp_model = getattr(response, "model", None)
            if resp_model is None and isinstance(response, dict):
                resp_model = response.get("model")
            if resp_model:
                resolved_model = str(resp_model)

        return self.track(
            model=resolved_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            response=response,
            metadata=metadata,
        )

    def check_budget(self) -> None:
        """Check if the current cumulative cost has reached or exceeded MAX_RUN_USD.

        Raises BudgetExceededError if the budget ceiling has been reached.
        """
        with self._lock:
            current_cost = self._total_cost_usd
            limit = self._max_run_usd

        if current_cost >= limit:
            raise BudgetExceededError(
                current_cost=current_cost,
                max_budget=limit,
            )

    def reset(self) -> None:
        """Reset all cumulative counters and records to clean state."""
        with self._lock:
            self._records.clear()
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_tokens = 0
            self._total_cost_usd = 0.0

    def get_summary(self) -> CostSummary:
        """Return a snapshot of current cumulative token usage and costs."""
        with self._lock:
            return CostSummary(
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                total_tokens=self._total_tokens,
                total_cost_usd=self._total_cost_usd,
                max_run_usd=self._max_run_usd,
                remaining_budget_usd=max(0.0, self._max_run_usd - self._total_cost_usd),
                call_count=len(self._records),
            )
