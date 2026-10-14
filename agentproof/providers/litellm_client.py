"""Unified LiteLLM client wrapper with mock fallback, cost tracking, and disk caching.

Configures agent, user simulator, and judge models from environment variables.
Integrates CostTracker for budget ceiling enforcement, DiskCache for reproducible
test execution, and MockProvider for offline testing without real API credentials.
Provides resilient retry handling with exponential backoff and request timeouts.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from agentproof.providers.cache import DiskCache
from agentproof.providers.cost_tracker import CostTracker
from agentproof.providers.mock_provider import MockProvider

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


class LiteLLMClient:
    """Unified client interface wrapping LiteLLM completions with caching and cost control.

    Configured model names default to:
    - AGENT_MODEL: 'openai/gpt-4o-mini'
    - USER_SIM_MODEL: 'openai/gpt-4o-mini'
    - JUDGE_MODEL: 'anthropic/claude-3-5-sonnet-20241022'
    """

    def __init__(
        self,
        agent_model: Optional[str] = None,
        user_sim_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
        cache: Optional[DiskCache] = None,
        mock_provider: Optional[MockProvider] = None,
        use_cache: bool = True,
        force_mock: bool = False,
        timeout: float = 30.0,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        backoff_factor: float = 2.0,
    ) -> None:
        self.agent_model = agent_model or os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini")
        self.user_sim_model = user_sim_model or os.environ.get("USER_SIM_MODEL", "openai/gpt-4o-mini")
        self.judge_model = judge_model or os.environ.get(
            "JUDGE_MODEL", "anthropic/claude-3-5-sonnet-20241022"
        )

        self.cost_tracker = cost_tracker or CostTracker()
        self.cache = cache or DiskCache()
        self.mock_provider = mock_provider or MockProvider()
        self.use_cache = use_cache
        self.force_mock = force_mock
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor

    def resolve_model(self, model: Optional[str]) -> str:
        """Resolve role alias strings to concrete model identifiers."""
        if not model or model in ("agent", "agent_model"):
            return self.agent_model
        if model in ("user_sim", "user_sim_model", "user"):
            return self.user_sim_model
        if model in ("judge", "judge_model"):
            return self.judge_model
        return model

    def has_api_key_for_model(self, model: str) -> bool:
        """Determine whether required provider API credentials exist in environment."""
        lower_model = model.lower()

        if "openai" in lower_model or lower_model.startswith("gpt-"):
            return bool(os.environ.get("OPENAI_API_KEY"))

        if "anthropic" in lower_model or "claude" in lower_model:
            return bool(os.environ.get("ANTHROPIC_API_KEY"))

        if "gemini" in lower_model or "google" in lower_model:
            return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

        # Fallback check for any major provider API key
        return bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )

    def is_mock_mode(self, resolved_model: str) -> bool:
        """Determine whether mock provider should intercept the LLM call."""
        if self.force_mock:
            return True
        if os.environ.get("AGENTPROOF_MOCK_MODE") == "1":
            return True
        if "mock" in resolved_model.lower():
            return True
        if litellm is None:
            return True
        return not self.has_api_key_for_model(resolved_model)

    def completion(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute LLM completion with mock dispatch, disk caching, and cost tracking.

        Attempts cache lookup first when enabled.
        Replays mock response when running in mock mode or when API credentials are absent.
        Executes live LiteLLM completion with retries and timeout otherwise.
        """
        resolved_model = self.resolve_model(model)
        safe_messages = messages or []
        cache_key = self.cache.compute_key(
            model=resolved_model,
            messages=safe_messages,
            tools=tools,
            temperature=temperature,
            **kwargs,
        )

        # 1. Check disk cache
        if self.use_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                return cached_response

        # 2. Mock mode execution
        if self.is_mock_mode(resolved_model):
            response = self.mock_provider.get_response(
                model=resolved_model,
                messages=safe_messages,
                tools=tools,
                temperature=temperature,
                cache_key=cache_key,
                **kwargs,
            )
            self.cost_tracker.track_response(
                response,
                model=resolved_model,
                metadata=metadata,
            )
            if self.use_cache:
                self.cache.set(cache_key, response)
            return response

        # 3. Live call with retries and exponential backoff
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                call_kwargs: Dict[str, Any] = {
                    "model": resolved_model,
                    "messages": safe_messages,
                    "temperature": temperature,
                    "timeout": self.timeout,
                    **kwargs,
                }
                if tools is not None:
                    call_kwargs["tools"] = tools

                response = litellm.completion(**call_kwargs)

                # Track token usage and financial cost
                self.cost_tracker.track_response(
                    response,
                    model=resolved_model,
                    metadata=metadata,
                )

                # Record live response for mock replay if configured
                if os.environ.get("AGENTPROOF_RECORD_MOCKS") == "1":
                    request_info = {
                        "model": resolved_model,
                        "messages": safe_messages,
                        "tools": tools,
                        "temperature": temperature,
                    }
                    self.mock_provider.record(cache_key, request_info, response)

                # Persist to disk cache
                if self.use_cache:
                    self.cache.set(cache_key, response)

                return response

            except Exception as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    sleep_time = self.initial_backoff * (self.backoff_factor ** attempt)
                    time.sleep(sleep_time)

        if last_exception:
            raise last_exception

        raise RuntimeError("LLM completion failed with an unknown error.")

    async def acompletion(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronously execute LLM completion with caching and mock dispatch."""
        resolved_model = self.resolve_model(model)
        safe_messages = messages or []
        cache_key = self.cache.compute_key(
            model=resolved_model,
            messages=safe_messages,
            tools=tools,
            temperature=temperature,
            **kwargs,
        )

        # 1. Check disk cache
        if self.use_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                return cached_response

        # 2. Mock mode execution
        if self.is_mock_mode(resolved_model):
            response = self.mock_provider.get_response(
                model=resolved_model,
                messages=safe_messages,
                tools=tools,
                temperature=temperature,
                cache_key=cache_key,
                **kwargs,
            )
            self.cost_tracker.track_response(
                response,
                model=resolved_model,
                metadata=metadata,
            )
            if self.use_cache:
                self.cache.set(cache_key, response)
            return response

        # 3. Live async call with retries
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                call_kwargs: Dict[str, Any] = {
                    "model": resolved_model,
                    "messages": safe_messages,
                    "temperature": temperature,
                    "timeout": self.timeout,
                    **kwargs,
                }
                if tools is not None:
                    call_kwargs["tools"] = tools

                response = await litellm.acompletion(**call_kwargs)

                self.cost_tracker.track_response(
                    response,
                    model=resolved_model,
                    metadata=metadata,
                )

                if os.environ.get("AGENTPROOF_RECORD_MOCKS") == "1":
                    request_info = {
                        "model": resolved_model,
                        "messages": safe_messages,
                        "tools": tools,
                        "temperature": temperature,
                    }
                    self.mock_provider.record(cache_key, request_info, response)

                if self.use_cache:
                    self.cache.set(cache_key, response)

                return response

            except Exception as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    sleep_time = self.initial_backoff * (self.backoff_factor ** attempt)
                    await asyncio.sleep(sleep_time)

        if last_exception:
            raise last_exception

        raise RuntimeError("Asynchronous LLM completion failed with an unknown error.")


_DEFAULT_CLIENT: Optional[LiteLLMClient] = None


def get_client() -> LiteLLMClient:
    """Retrieve or initialize the process-wide default LiteLLMClient instance."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LiteLLMClient()
    return _DEFAULT_CLIENT


def set_default_client(client: Optional[LiteLLMClient]) -> None:
    """Override the default LiteLLMClient instance."""
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = client
