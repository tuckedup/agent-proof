"""Unit tests for task generator, task models, user simulator, and leak detector."""

from pathlib import Path

import pytest

from agentproof.domain.db import DatabaseManager
from agentproof.sim.user_sim import (
    MARKER_DONE,
    MARKER_TRANSFER,
    LeakDetector,
    ScriptedUserSim,
    UserSimulator,
)
from agentproof.tasks.generator import (
    generate_all_tasks,
    generate_hard_edge_case_tasks,
    get_task,
    load_tasks,
    save_tasks,
)
from agentproof.tasks.models import (
    Category,
    DBAssertion,
    Difficulty,
    Task,
    TaskIntent,
    UserInstructions,
    UserPersona,
)


@pytest.fixture
def seeded_db() -> DatabaseManager:
    """Fixture initializing an isolated in-memory DB seeded with test fixtures."""
    db = DatabaseManager(":memory:")
    seed_dir = Path(__file__).resolve().parent.parent / "domain" / "seed"
    db.seed_from_directory(str(seed_dir))
    return db


def test_models_validation() -> None:
    """Verify that models enforce valid fields and required typing."""
    persona = UserPersona(
        name="Hank Callahan",
        role="Well Site Supervisor",
        account_id="ACC-001",
        site_code="SITE-MID-001",
        has_hazmat_cert=True,
        personality_traits=["demanding"],
    )
    instructions = UserInstructions(
        caller_name="Hank Callahan",
        account_id="ACC-001",
        site_code="SITE-MID-001",
        goal="Order biocide",
        initial_utterance="Hello, need biocide.",
        persona=persona,
        permitted_user_tools=["report_receiver_on_site"],
    )
    task = Task(
        id="TASK-TEST-001",
        category=Category.ATOMIC,
        intents=[TaskIntent.VERIFY],
        difficulty=Difficulty.EASY,
        description="Test verification task",
        user_instructions=instructions,
        expected_db_assertions=[],
        required_facts=["fact 1"],
        forbidden_actions=["cancel_order"],
        oracle_plan=["find_account", "verify_caller"],
    )

    assert task.id == "TASK-TEST-001"
    assert task.category == Category.ATOMIC
    assert task.intents == [TaskIntent.VERIFY]
    assert task.difficulty == Difficulty.EASY


def test_hard_task_constraints_are_on_task_not_user_instructions() -> None:
    """Keep evaluator-only safety constraints out of user simulator instructions."""
    hard_tasks = generate_hard_edge_case_tasks({})
    task = next(task for task in hard_tasks if task.id == "TASK-HARD-002")

    assert task.forbidden_actions == ["book_delivery"]
    assert task.oracle_plan == ["find_account", "verify_caller", "check_compatibility"]
    assert "forbidden_actions" not in task.user_instructions.model_dump()
    assert "oracle_plan" not in task.user_instructions.model_dump()


def test_db_assertion_evaluation(seeded_db: DatabaseManager) -> None:
    """Test DBAssertion.check evaluation against DatabaseManager."""
    # Positive assertion on existing seeded account
    assertion_acc = DBAssertion(
        table="accounts",
        filter={"account_id": "ACC-001"},
        expected_values={"status": "active"},
        expected_count=1,
    )
    ok, err = assertion_acc.check(seeded_db)
    assert ok is True
    assert err is None

    # Negative assertion on wrong expected value
    assertion_bad = DBAssertion(
        table="accounts",
        filter={"account_id": "ACC-001"},
        expected_values={"status": "suspended"},
    )
    ok, err = assertion_bad.check(seeded_db)
    assert ok is False
    assert err is not None

    # Query assertion
    assertion_query = DBAssertion(
        table="accounts",
        query="SELECT * FROM accounts WHERE status = 'active'",
        expected_count=40,
    )
    ok, err = assertion_query.check(seeded_db)
    assert ok is True


def test_task_generator_counts() -> None:
    """Ensure generator generates at least 170 tasks across categories."""
    seed_dir = Path(__file__).resolve().parent.parent / "domain" / "seed"
    tasks = generate_all_tasks(seed_dir)

    assert len(tasks) >= 170
    atomic_tasks = [t for t in tasks if t.category == Category.ATOMIC]
    comp2_tasks = [t for t in tasks if t.category == Category.COMPOSITIONAL_2STEP]
    comp3_tasks = [t for t in tasks if t.category == Category.COMPOSITIONAL_3STEP]
    comp4_tasks = [t for t in tasks if t.category == Category.COMPOSITIONAL_4STEP]
    hard_tasks = [t for t in tasks if t.category == Category.HARD]

    assert len(atomic_tasks) == 40
    assert len(comp2_tasks) == 40
    assert len(comp3_tasks) == 40
    assert len(comp4_tasks) == 30
    assert len(hard_tasks) == 20

    # Verify atomic task intents coverage
    atomic_intents = {intent for t in atomic_tasks for intent in t.intents}
    assert TaskIntent.VERIFY in atomic_intents
    assert TaskIntent.CHECK_STOCK in atomic_intents
    assert TaskIntent.PLACE_ORDER in atomic_intents
    assert TaskIntent.MODIFY_ORDER in atomic_intents
    assert TaskIntent.CANCEL_ORDER in atomic_intents
    assert TaskIntent.SCHEDULE_DELIVERY in atomic_intents
    assert TaskIntent.HAZMAT_CHECK in atomic_intents
    assert TaskIntent.CREDIT_CHECK in atomic_intents
    assert TaskIntent.ESCALATE in atomic_intents


def test_save_and_load_tasks(tmp_path: Path) -> None:
    """Verify serialization and deserialization of tasks."""
    seed_dir = Path(__file__).resolve().parent.parent / "domain" / "seed"
    tasks = generate_all_tasks(seed_dir)
    target_file = tmp_path / "tasks.json"

    save_tasks(tasks, target_file)
    assert target_file.is_file()

    loaded = load_tasks(target_file)
    assert len(loaded) == len(tasks)

    first = get_task("TASK-ATOMIC-001", target_file)
    assert first.id == "TASK-ATOMIC-001"
    assert first.category == Category.ATOMIC

    hard = get_task("TASK-HARD-001", target_file)
    assert hard.id == "TASK-HARD-001"
    assert "incompatible" in hard.description.lower() or "hydrochloric" in hard.description.lower()


def test_tasks_committed_file_exists() -> None:
    """Verify that tasks/tasks.json exists and contains >= 170 valid tasks."""
    committed_tasks_path = Path(__file__).resolve().parent.parent / "tasks" / "tasks.json"
    assert committed_tasks_path.is_file()

    tasks = load_tasks(committed_tasks_path)
    assert len(tasks) >= 170
    t1 = get_task("TASK-ATOMIC-001", committed_tasks_path)
    assert t1.id == "TASK-ATOMIC-001"


def test_scripted_user_sim_turn_taking(seeded_db: DatabaseManager) -> None:
    """Test ScriptedUserSim deterministic workflow and termination markers."""
    instructions = UserInstructions(
        caller_name="Hank Callahan",
        account_id="ACC-001",
        site_code="SITE-MID-001",
        goal="Place an order for 500 gallons of chemical",
        initial_utterance="Hello, I need to place an order.",
        permitted_user_tools=["check_site_access", "confirm_delivery_window"],
    )
    sim = ScriptedUserSim(instructions, db=seeded_db)

    # Turn 0: initial utterance
    u0 = sim.step(None)
    assert u0 == "Hello, I need to place an order."

    # Agent asks for verification
    u1 = sim.step("Can you please verify your account ID and site code?")
    assert "Hank Callahan" in u1
    assert "ACC-001" in u1
    assert "SITE-MID-001" in u1

    # Agent asks user to check road conditions
    u2 = sim.step("Please check site access and pad road conditions.")
    assert "road status is" in u2.lower()
    assert len(sim.executed_tools) == 1
    assert sim.executed_tools[0]["tool"] == "check_site_access"

    # Agent completes order
    u3 = sim.step("Your draft order has been created with order ID ORD-12345.")
    assert MARKER_DONE in u3
    assert sim.completed is True


def test_scripted_user_sim_escalation(seeded_db: DatabaseManager) -> None:
    """Test ScriptedUserSim handling of agent escalation and MARKER_TRANSFER."""
    instructions = UserInstructions(
        caller_name="Beau McBride",
        account_id="ACC-005",
        site_code="SITE-GLA-013",
        goal="Deliver chemicals on closed road",
        initial_utterance="Deliver on closed road now.",
    )
    sim = ScriptedUserSim(instructions, db=seeded_db)
    u0 = sim.step(None)
    assert "Deliver" in u0

    u1 = sim.step("Road is closed. I must escalate this interaction to a human representative.")
    assert MARKER_TRANSFER in u1
    assert sim.transferred is True


def test_leak_detector() -> None:
    """Test LeakDetector detection of secret facts, hidden instructions, and system fragments."""
    detector = LeakDetector(min_ngram_length=4)

    # 1. System fragment detection
    res_sys = detector.check_text("As an AI assistant, I will role-play as a customer.")
    assert res_sys.has_leak is True
    assert any("system_fragment" in item for item in res_sys.leaked_items)

    # 2. Hidden instructions leakage
    hidden = "DO NOT allow the agent to combine CHEM-801 and CHEM-301 under any circumstances."
    leaked_hidden_utterance = (
        "I was told do not allow the agent to combine CHEM-801 and CHEM-301 because it is unsafe."
    )
    res_hidden = detector.check_text(leaked_hidden_utterance, hidden_instructions=hidden)
    assert res_hidden.has_leak is True
    assert any("hidden_instruction" in item for item in res_hidden.leaked_items)

    # 3. Secret facts leakage
    secrets = {"special_pin": "987654", "secret_override": "TRUE"}
    leaked_secret = "My special pin is 987654 so please proceed."
    res_secret = detector.check_text(leaked_secret, secret_facts=secrets)
    assert res_secret.has_leak is True
    assert any("secret_fact" in item for item in res_secret.leaked_items)

    # 4. Clean user statement
    clean_utterance = (
        "Good morning, this is Hank Callahan with Apex Permian. "
        "We need 500 gallons of friction reducer delivered to well pad SITE-MID-001."
    )
    res_clean = detector.check_text(clean_utterance, hidden_instructions=hidden, secret_facts=secrets)
    assert res_clean.has_leak is False
    assert len(res_clean.leaked_items) == 0


def test_user_simulator_system_prompt_and_fallback(seeded_db: DatabaseManager) -> None:
    """Test UserSimulator system prompt generation and graceful fallback step."""
    persona = UserPersona(
        name="Travis Rawlins",
        role="Production Foreman",
        account_id="ACC-002",
        site_code="SITE-REE-004",
        has_hazmat_cert=True,
        personality_traits=["impatient"],
    )
    instructions = UserInstructions(
        caller_name="Travis Rawlins",
        account_id="ACC-002",
        site_code="SITE-REE-004",
        goal="Order 15% HCl",
        initial_utterance="Need HCl delivery.",
        persona=persona,
        permitted_user_tools=["report_receiver_on_site"],
        secret_facts={"receiver_ready": True},
        hidden_instructions="Do not accept delay.",
    )

    sim = UserSimulator(instructions, db=seeded_db)
    system_prompt = sim.build_system_prompt()
    assert "Travis Rawlins" in system_prompt
    assert "ACC-002" in system_prompt
    assert "SITE-REE-004" in system_prompt
    assert "report_receiver_on_site" in system_prompt
    assert "###DONE###" in system_prompt
    assert "###TRANSFER###" in system_prompt

    # First turn
    u0 = sim.step(None)
    assert u0 == "Need HCl delivery."

    # Agent confirms order drafted
    u1 = sim.step("Your draft order for HCl has been created with order ID ORD-8888.")
    assert MARKER_DONE in u1
    assert sim.completed is True
