"""Unit and integration tests for providers, cost tracker, disk cache, and mock replay."""

import threading

import pytest

from agentproof.providers import (
    BudgetExceededError,
    CostTracker,
    DiskCache,
    LiteLLMClient,
    MockProvider,
    make_mock_response,
)


def test_cost_tracker_basic_and_summary() -> None:
    """Verify CostTracker tracks tokens and dollar costs accurately."""
    tracker = CostTracker(max_run_usd=1.0)
    assert tracker.total_prompt_tokens == 0
    assert tracker.total_completion_tokens == 0
    assert tracker.total_cost_usd == 0.0

    # Track simple call
    rec = tracker.track(
        model="openai/gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert rec.prompt_tokens == 1000
    assert rec.completion_tokens == 500
    assert rec.total_tokens == 1500
    assert rec.cost_usd > 0.0

    assert tracker.total_prompt_tokens == 1000
    assert tracker.total_completion_tokens == 500
    assert tracker.total_cost_usd == rec.cost_usd

    summary = tracker.get_summary()
    assert summary.call_count == 1
    assert summary.total_tokens == 1500
    assert summary.remaining_budget_usd == tracker.remaining_budget_usd

    # Reset
    tracker.reset()
    assert tracker.total_tokens == 0
    assert tracker.total_cost_usd == 0.0


def test_cost_tracker_budget_exceeded() -> None:
    """Verify CostTracker raises BudgetExceededError when MAX_RUN_USD ceiling is reached."""
    # Set a tiny budget of $0.0001
    tracker = CostTracker(max_run_usd=0.0001)

    with pytest.raises(BudgetExceededError) as exc_info:
        # 100k tokens on gpt-4o will cost > $0.0001
        tracker.track(
            model="openai/gpt-4o",
            prompt_tokens=50000,
            completion_tokens=50000,
        )

    assert exc_info.value.max_budget == 0.0001
    assert exc_info.value.current_cost >= 0.0001


def test_cost_tracker_thread_safety() -> None:
    """Verify CostTracker thread-safety across concurrent workers."""
    tracker = CostTracker(max_run_usd=100.0)
    threads = []
    calls_per_thread = 50

    def worker() -> None:
        for _ in range(calls_per_thread):
            tracker.track(
                model="gpt-4o-mini",
                prompt_tokens=10,
                completion_tokens=10,
                cost_usd=0.00001,
            )

    for _ in range(4):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    expected_calls = 4 * calls_per_thread
    assert len(tracker.records) == expected_calls
    assert tracker.total_prompt_tokens == expected_calls * 10
    assert tracker.total_completion_tokens == expected_calls * 10


def test_cost_tracker_track_response() -> None:
    """Verify CostTracker extracts usage directly from response dictionaries and objects."""
    tracker = CostTracker(max_run_usd=5.0)

    # Dictionary response
    dict_resp = {
        "model": "openai/gpt-4o-mini",
        "usage": {"prompt_tokens": 200, "completion_tokens": 100},
    }
    rec = tracker.track_response(dict_resp)
    assert rec.prompt_tokens == 200
    assert rec.completion_tokens == 100
    assert rec.model == "openai/gpt-4o-mini"

    # Synthetic ModelResponse
    model_resp = make_mock_response(
        content="Testing",
        model="openai/gpt-4o-mini",
        prompt_tokens=300,
        completion_tokens=150,
    )
    rec2 = tracker.track_response(model_resp)
    assert rec2.prompt_tokens == 300
    assert rec2.completion_tokens == 150


def test_disk_cache_operations(tmp_path) -> None:
    """Verify DiskCache computes keys, saves responses, and retrieves them reliably."""
    cache = DiskCache(cache_dir=tmp_path / "cache")
    assert cache.size() == 0

    key = cache.compute_key(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.0,
    )
    assert isinstance(key, str)
    assert len(key) == 64

    # Miss
    assert cache.get(key) is None
    assert cache.exists(key) is False

    # Store
    test_response = make_mock_response(
        content="Hello, how can I help you?",
        model="openai/gpt-4o-mini",
    )
    cache.set(key, test_response)
    assert cache.exists(key) is True
    assert cache.size() == 1

    # Hit
    retrieved = cache.get(key)
    assert retrieved is not None
    # Check content whether retrieved as ModelResponse or dict
    if hasattr(retrieved, "choices"):
        content = retrieved.choices[0].message.content
    else:
        content = retrieved["choices"][0]["message"]["content"]
    assert content == "Hello, how can I help you?"

    # Clear
    deleted = cache.clear()
    assert deleted == 1
    assert cache.size() == 0
    assert cache.get(key) is None


def test_mock_provider_scenarios(tmp_path) -> None:
    """Verify MockProvider replays responses for all 7 required domain categories."""
    mock_file = tmp_path / "mock_records.json"
    provider = MockProvider(records_path=mock_file, enable_recording=False)

    # 1. Account lookup and caller verification
    res_account = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Please find account for Apex Solutions."}],
        tools=[{"name": "find_account"}],
    )
    tool_calls = getattr(res_account.choices[0].message, "tool_calls", None)
    assert tool_calls is not None
    fn_name = tool_calls[0].function.name if hasattr(tool_calls[0], "function") else tool_calls[0]["function"]["name"]
    assert fn_name == "find_account"

    res_verify = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Verify caller John Doe for site SITE-MID-001"}],
        tools=[{"name": "verify_caller"}],
    )
    tool_calls_v = getattr(res_verify.choices[0].message, "tool_calls", None)
    fn_name_v = tool_calls_v[0].function.name if hasattr(tool_calls_v[0], "function") else tool_calls_v[0]["function"]["name"]
    assert fn_name_v == "verify_caller"

    # 2. Stock check and quoting
    res_stock = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Check stock for CHEM-CORR-01 at DEPOT-001"}],
        tools=[{"name": "check_stock"}],
    )
    tc_stock = getattr(res_stock.choices[0].message, "tool_calls", None)
    fn_stock = tc_stock[0].function.name if hasattr(tc_stock[0], "function") else tc_stock[0]["function"]["name"]
    assert fn_stock == "check_stock"

    res_quote = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Quote price for 500 gallons of corrosion inhibitor"}],
        tools=[{"name": "quote_price"}],
    )
    tc_quote = getattr(res_quote.choices[0].message, "tool_calls", None)
    fn_quote = tc_quote[0].function.name if hasattr(tc_quote[0], "function") else tc_quote[0]["function"]["name"]
    assert fn_quote == "quote_price"

    # 3. Order creation, modification, cancellation
    res_create = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Create order for 500 gallons of CHEM-CORR-01"}],
        tools=[{"name": "create_order"}],
    )
    tc_create = getattr(res_create.choices[0].message, "tool_calls", None)
    fn_create = tc_create[0].function.name if hasattr(tc_create[0], "function") else tc_create[0]["function"]["name"]
    assert fn_create == "create_order"

    res_mod = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Modify order ORD-0001 change quantity to 600"}],
        tools=[{"name": "modify_order"}],
    )
    tc_mod = getattr(res_mod.choices[0].message, "tool_calls", None)
    fn_mod = tc_mod[0].function.name if hasattr(tc_mod[0], "function") else tc_mod[0]["function"]["name"]
    assert fn_mod == "modify_order"

    res_cancel = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Cancel order ORD-0001 due to duplicate entry"}],
        tools=[{"name": "cancel_order"}],
    )
    tc_cancel = getattr(res_cancel.choices[0].message, "tool_calls", None)
    fn_cancel = tc_cancel[0].function.name if hasattr(tc_cancel[0], "function") else tc_cancel[0]["function"]["name"]
    assert fn_cancel == "cancel_order"

    # 4. Delivery slot search and booking
    res_slots = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Search delivery slot dispatch window for 2026-10-15"}],
        tools=[{"name": "list_delivery_slots"}],
    )
    tc_slots = getattr(res_slots.choices[0].message, "tool_calls", None)
    fn_slots = tc_slots[0].function.name if hasattr(tc_slots[0], "function") else tc_slots[0]["function"]["name"]
    assert fn_slots == "list_delivery_slots"

    res_book = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Book delivery slot SLOT-0001 for order ORD-0001"}],
        tools=[{"name": "book_delivery"}],
    )
    tc_book = getattr(res_book.choices[0].message, "tool_calls", None)
    fn_book = tc_book[0].function.name if hasattr(tc_book[0], "function") else tc_book[0]["function"]["name"]
    assert fn_book == "book_delivery"

    # 5. Hazmat compatibility check and certified receiver checks
    res_hazmat = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Check hazmat compatibility for CHEM-CORR-01 and CHEM-SCALE-02"}],
        tools=[{"name": "check_compatibility"}],
    )
    tc_hazmat = getattr(res_hazmat.choices[0].message, "tool_calls", None)
    fn_hazmat = tc_hazmat[0].function.name if hasattr(tc_hazmat[0], "function") else tc_hazmat[0]["function"]["name"]
    assert fn_hazmat == "check_compatibility"

    # 6. Credit checks and policy escalations
    res_credit = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Run credit check for account ACC-001"}],
        tools=[{"name": "check_credit"}],
    )
    tc_credit = getattr(res_credit.choices[0].message, "tool_calls", None)
    fn_credit = tc_credit[0].function.name if hasattr(tc_credit[0], "function") else tc_credit[0]["function"]["name"]
    assert fn_credit == "check_credit"

    res_esc = provider.get_response(
        model="mock-model",
        messages=[{"role": "user", "content": "Escalate to human desk supervisor due to policy exception"}],
        tools=[{"name": "escalate_to_human"}],
    )
    tc_esc = getattr(res_esc.choices[0].message, "tool_calls", None)
    fn_esc = tc_esc[0].function.name if hasattr(tc_esc[0], "function") else tc_esc[0]["function"]["name"]
    assert fn_esc == "escalate_to_human"

    # 7. LangGraph supervisor agent routing responses
    res_sup = provider.get_response(
        model="mock-model",
        messages=[{"role": "system", "content": "You are the LangGraph supervisor. Route to next specialist."}],
    )
    sup_content = res_sup.choices[0].message.content
    assert "order_agent" in sup_content


def test_mock_provider_recording(tmp_path) -> None:
    """Verify MockProvider records live calls when enabled and replays them by key."""
    record_file = tmp_path / "records.json"
    provider = MockProvider(records_path=record_file, enable_recording=True)

    dummy_response = make_mock_response(
        content="Recorded answer",
        model="test-model",
    )
    provider.record("test_key_123", {"prompt": "test prompt"}, dummy_response)

    # Replay
    replayed = provider.get_response(cache_key="test_key_123")
    assert replayed is not None
    if hasattr(replayed, "choices"):
        content = replayed.choices[0].message.content
    else:
        content = replayed["choices"][0]["message"]["content"]
    assert content == "Recorded answer"


def test_litellm_client_integration(tmp_path, monkeypatch) -> None:
    """Verify LiteLLMClient integrates CostTracker, DiskCache, and MockProvider seamlessly."""
    cache = DiskCache(cache_dir=tmp_path / "client_cache")
    tracker = CostTracker(max_run_usd=1.0)
    mock_prov = MockProvider(records_path=tmp_path / "mocks.json")

    # Force mock mode for deterministic testing without external API credentials
    client = LiteLLMClient(
        agent_model="openai/gpt-4o-mini",
        cost_tracker=tracker,
        cache=cache,
        mock_provider=mock_prov,
        force_mock=True,
    )

    # First call - mock response
    resp = client.completion(
        messages=[{"role": "user", "content": "Please lookup account Apex"}],
        tools=[{"name": "find_account"}],
    )
    assert resp is not None
    assert tracker.total_prompt_tokens > 0
    assert tracker.total_cost_usd > 0.0
    assert cache.size() == 1

    first_cost = tracker.total_cost_usd

    # Second identical call - hits disk cache
    resp2 = client.completion(
        messages=[{"role": "user", "content": "Please lookup account Apex"}],
        tools=[{"name": "find_account"}],
    )
    assert resp2 is not None
    # Cache hit returns without adding duplicate network cost
    assert tracker.total_cost_usd == first_cost


def test_litellm_client_role_models(monkeypatch) -> None:
    """Verify LiteLLMClient resolves AGENT_MODEL, USER_SIM_MODEL, and JUDGE_MODEL."""
    monkeypatch.setenv("AGENT_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("USER_SIM_MODEL", "anthropic/claude-3-haiku")
    monkeypatch.setenv("JUDGE_MODEL", "anthropic/claude-3-5-sonnet")

    client = LiteLLMClient()
    assert client.resolve_model("agent") == "openai/gpt-4o"
    assert client.resolve_model("user_sim") == "anthropic/claude-3-haiku"
    assert client.resolve_model("judge") == "anthropic/claude-3-5-sonnet"
    assert client.resolve_model("custom/model") == "custom/model"


def test_litellm_client_acompletion(tmp_path) -> None:
    """Verify LiteLLMClient asynchronous acompletion method."""
    import asyncio

    client = LiteLLMClient(
        cache=DiskCache(cache_dir=tmp_path / "async_cache"),
        cost_tracker=CostTracker(max_run_usd=1.0),
        mock_provider=MockProvider(records_path=tmp_path / "async_mocks.json"),
        force_mock=True,
    )

    resp = asyncio.run(
        client.acompletion(
            messages=[{"role": "user", "content": "Check stock for CHEM-CORR-01"}],
            tools=[{"name": "check_stock"}],
        )
    )
    assert resp is not None
    assert client.cost_tracker.total_tokens > 0
