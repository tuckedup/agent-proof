"""Unit tests verifying 100% of agent and user tools, and database operations."""

from pathlib import Path

import pytest

from agentproof.domain.db import DatabaseManager
from agentproof.tools.agent_tools import AGENT_TOOLS
from agentproof.tools.user_tools import USER_TOOLS


@pytest.fixture
def seeded_db() -> DatabaseManager:
    """Fixture initializing an isolated in-memory DB seeded with test fixtures."""
    db = DatabaseManager(":memory:")
    seed_dir = Path(__file__).resolve().parent.parent / "domain" / "seed"
    db.seed_from_directory(str(seed_dir))
    return db


@pytest.fixture
def acc_site(seeded_db: DatabaseManager) -> str:
    """Returns valid site code for ACC-001."""
    sites = seeded_db.execute_query("SELECT site_code FROM sites WHERE account_id = 'ACC-001'")
    return sites[0]["site_code"]


# --- Agent Tools Unit Tests ---

def test_find_account(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["find_account"]
    # Test valid query
    res = tool.execute(seeded_db, query="Apex")
    assert res.success is True
    assert res.data["count"] >= 1
    assert res.data["matches"][0]["account_id"] == "ACC-001"

    # Test query by ID
    res2 = tool.execute(seeded_db, query="ACC-002")
    assert res2.success is True
    assert res2.data["count"] == 1

    # Test not found
    res3 = tool.execute(seeded_db, query="NonExistentCompanyXYZ")
    assert res3.success is True
    assert res3.data["count"] == 0


def test_verify_caller(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["verify_caller"]
    # Lookup valid contact and site for ACC-001
    contacts = seeded_db.execute_query("SELECT full_name, has_hazmat_cert FROM contacts WHERE account_id = 'ACC-001'")
    valid_name = contacts[0]["full_name"]
    sites = seeded_db.execute_query("SELECT site_code FROM sites WHERE account_id = 'ACC-001'")
    valid_site = sites[0]["site_code"]

    # 1. Success case
    res = tool.execute(seeded_db, account_id="ACC-001", site_code=valid_site, caller_name=valid_name)
    assert res.success is True
    assert res.data["verified"] is True
    assert res.data["has_hazmat_cert"] == bool(contacts[0]["has_hazmat_cert"])

    # 2. Wrong site for account
    res_bad_site = tool.execute(seeded_db, account_id="ACC-001", site_code="SITE-NONEXISTENT", caller_name=valid_name)
    assert res_bad_site.success is False
    assert "not registered" in res_bad_site.error

    # 3. Unauthorized caller name
    res_bad_caller = tool.execute(seeded_db, account_id="ACC-001", site_code=valid_site, caller_name="Random Stranger")
    assert res_bad_caller.success is False
    assert "not listed as an authorized contact" in res_bad_caller.error

    # 4. Unknown account
    res_bad_acc = tool.execute(seeded_db, account_id="ACC-999", site_code=valid_site, caller_name=valid_name)
    assert res_bad_acc.success is False


def test_get_order(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["get_order"]
    # Success
    res = tool.execute(seeded_db, order_id="ORD-0001")
    assert res.success is True
    assert res.data["order_id"] == "ORD-0001"
    assert "lines" in res.data
    assert len(res.data["lines"]) >= 1

    # Not found
    res_none = tool.execute(seeded_db, order_id="ORD-9999")
    assert res_none.success is False
    assert "not found" in res_none.error


def test_list_orders(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["list_orders"]
    orders = seeded_db.execute_query("SELECT account_id FROM orders LIMIT 1")
    acc_id = orders[0]["account_id"]

    # List all
    res = tool.execute(seeded_db, account_id=acc_id)
    assert res.success is True
    assert res.data["count"] >= 1

    # Filter by status
    res_filtered = tool.execute(seeded_db, account_id=acc_id, status="delivered")
    assert res_filtered.success is True


def test_check_stock(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["check_stock"]
    # All depots
    res = tool.execute(seeded_db, product_id="CHEM-801")
    assert res.success is True
    assert res.data["total_available_gal"] > 0
    assert len(res.data["depots"]) == 4

    # Single depot
    res_depot = tool.execute(seeded_db, product_id="CHEM-801", depot_id="DEP-MID")
    assert res_depot.success is True
    assert res_depot.data["available_gal"] > 0

    # Non-existent product
    res_missing = tool.execute(seeded_db, product_id="CHEM-UNKNOWN")
    assert res_missing.success is False


def test_quote_price(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["quote_price"]
    # Success
    res = tool.execute(seeded_db, product_id="CHEM-801", quantity=1000.0)
    assert res.success is True
    assert res.data["subtotal"] == 4250.0 # 1000 * 4.25
    assert res.data["hazard_class"] == "8"
    assert res.data["requires_hazmat_receiver"] is True

    # Exceeding single order max limit (max for CHEM-801 is 3000)
    res_exceeded = tool.execute(seeded_db, product_id="CHEM-801", quantity=5000.0)
    assert res_exceeded.success is False
    assert "exceeds maximum single order limit" in res_exceeded.error

    # Invalid negative qty
    res_neg = tool.execute(seeded_db, product_id="CHEM-801", quantity=-100.0)
    assert res_neg.success is False


def test_create_order(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["create_order"]
    sites = seeded_db.execute_query("SELECT site_code, account_id FROM sites WHERE account_id = 'ACC-001'")
    site_code = sites[0]["site_code"]

    items = [{"product_id": "CHEM-101", "quantity": 1000.0}]
    res = tool.execute(seeded_db, account_id="ACC-001", site_code=site_code, items=items, notes="Urgent pad delivery")
    assert res.success is True
    assert res.data["status"] == "draft"
    assert res.data["total_amount"] == 7500.0 # 1000 * 7.50
    new_order_id = res.data["order_id"]

    # Verify order exists in DB
    verify_res = AGENT_TOOLS["get_order"].execute(seeded_db, order_id=new_order_id)
    assert verify_res.success is True
    assert len(verify_res.data["lines"]) == 1


def test_modify_order_policy(seeded_db: DatabaseManager, acc_site: str) -> None:
    tool = AGENT_TOOLS["modify_order"]
    # Create draft order
    create_res = AGENT_TOOLS["create_order"].execute(
        seeded_db, account_id="ACC-001", site_code=acc_site,
        items=[{"product_id": "CHEM-101", "quantity": 500.0}]
    )
    assert create_res.success is True, f"Failed to create order: {create_res.error}"
    oid = create_res.data["order_id"]

    # Modification allowed in draft status
    mod_res = tool.execute(seeded_db, order_id=oid, action="update_quantity", product_id="CHEM-101", quantity=800.0)
    assert mod_res.success is True
    assert mod_res.data["new_total_amount"] == 6000.0 # 800 * 7.50

    # Add item
    add_res = tool.execute(seeded_db, order_id=oid, action="add_item", product_id="CHEM-103", quantity=200.0)
    assert add_res.success is True

    # Transition order to picked status
    seeded_db.execute_write("UPDATE orders SET status = 'picked' WHERE order_id = ?", (oid,))

    # Policy Check: Modification must be REJECTED after picked status
    blocked_res = tool.execute(seeded_db, order_id=oid, action="update_quantity", product_id="CHEM-101", quantity=1000.0)
    assert blocked_res.success is False
    assert "modification forbidden" in blocked_res.error.lower()
    assert "picked" in blocked_res.error


def test_cancel_order_policy(seeded_db: DatabaseManager, acc_site: str) -> None:
    tool = AGENT_TOOLS["cancel_order"]

    # Case 1: Cancel in draft/submitted status -> $0 fee
    c1 = AGENT_TOOLS["create_order"].execute(
        seeded_db, account_id="ACC-001", site_code=acc_site,
        items=[{"product_id": "CHEM-101", "quantity": 500.0}]
    )
    assert c1.success is True, f"Failed to create order: {c1.error}"
    oid1 = c1.data["order_id"]
    res1 = tool.execute(seeded_db, order_id=oid1, reason="Rig schedule moved")
    assert res1.success is True
    assert res1.data["cancellation_fee"] == 0.0

    # Case 2: Cancel in picked status -> $350 restocking fee
    c2 = AGENT_TOOLS["create_order"].execute(
        seeded_db, account_id="ACC-001", site_code=acc_site,
        items=[{"product_id": "CHEM-101", "quantity": 500.0}]
    )
    assert c2.success is True, f"Failed to create order: {c2.error}"
    oid2 = c2.data["order_id"]
    seeded_db.execute_write("UPDATE orders SET status = 'picked' WHERE order_id = ?", (oid2,))
    res2 = tool.execute(seeded_db, order_id=oid2, reason="Over-ordered")
    assert res2.success is True
    assert res2.data["cancellation_fee"] == 350.0
    assert res2.data["fee_assessed"] is True

    # Check fee ledger record
    fees = seeded_db.execute_query("SELECT * FROM credit_ledger WHERE order_id = ? AND entry_type = 'fee'", (oid2,))
    assert len(fees) == 1
    assert fees[0]["amount"] == -350.0

    # Case 3: Cancel in in_transit status -> Forbidden, must escalate
    c3 = AGENT_TOOLS["create_order"].execute(
        seeded_db, account_id="ACC-001", site_code=acc_site,
        items=[{"product_id": "CHEM-101", "quantity": 500.0}]
    )
    assert c3.success is True, f"Failed to create order: {c3.error}"
    oid3 = c3.data["order_id"]
    seeded_db.execute_write("UPDATE orders SET status = 'in_transit' WHERE order_id = ?", (oid3,))
    res3 = tool.execute(seeded_db, order_id=oid3, reason="Cancelled on road")
    assert res3.success is False
    assert "Cannot cancel order" in res3.error


def test_delivery_slots_and_booking(seeded_db: DatabaseManager, acc_site: str) -> None:
    # List slots
    list_tool = AGENT_TOOLS["list_delivery_slots"]
    res_slots = list_tool.execute(seeded_db, depot_id="DEP-MID")
    assert res_slots.success is True
    assert res_slots.data["count"] >= 1
    first_slot_id = res_slots.data["available_slots"][0]["slot_id"]

    # Create order
    c = AGENT_TOOLS["create_order"].execute(
        seeded_db, account_id="ACC-001", site_code=acc_site,
        items=[{"product_id": "CHEM-101", "quantity": 500.0}]
    )
    assert c.success is True, f"Failed to create order: {c.error}"
    oid = c.data["order_id"]

    # Book slot
    book_tool = AGENT_TOOLS["book_delivery"]
    book_res = book_tool.execute(seeded_db, order_id=oid, slot_id=first_slot_id)
    assert book_res.success is True
    assert book_res.data["status"] == "booked"

    # Attempt to book the same slot again -> Fail
    book_again = book_tool.execute(seeded_db, order_id=oid, slot_id=first_slot_id)
    assert book_again.success is False
    assert "already booked" in book_again.error


def test_check_compatibility(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["check_compatibility"]
    # Compatible items
    res_comp = tool.execute(seeded_db, product_ids=["CHEM-101", "CHEM-103", "CHEM-104"])
    assert res_comp.success is True
    assert res_comp.data["compatible"] is True

    # Incompatible: Acid (CHEM-801) + Methanol solvent (CHEM-301)
    res_incomp = tool.execute(seeded_db, product_ids=["CHEM-801", "CHEM-301"])
    assert res_incomp.success is True
    assert res_incomp.data["compatible"] is False
    assert len(res_incomp.data["conflicts"]) == 1


def test_check_credit(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["check_credit"]
    # Sufficient credit
    res_suff = tool.execute(seeded_db, account_id="ACC-001", order_amount=5000.0)
    assert res_suff.success is True
    assert res_suff.data["credit_sufficient"] is True

    # Excessive credit amount
    res_excess = tool.execute(seeded_db, account_id="ACC-001", order_amount=9999999.0)
    assert res_excess.success is True
    assert res_excess.data["credit_sufficient"] is False


def test_escalate_to_human(seeded_db: DatabaseManager) -> None:
    tool = AGENT_TOOLS["escalate_to_human"]
    res = tool.execute(
        seeded_db,
        reason="credit_limit_exceeded",
        context_summary="Customer requested 5000 gal scale inhibitor but credit balance is exceeded.",
        account_id="ACC-001",
        caller_name="Hank Callahan"
    )
    assert res.success is True
    assert res.data["status"] == "escalated"
    esc_id = res.data["escalation_id"]

    # Verify in DB
    rows = seeded_db.execute_query("SELECT * FROM escalations WHERE escalation_id = ?", (esc_id,))
    assert len(rows) == 1
    assert rows[0]["reason"] == "credit_limit_exceeded"


# --- User Tools Unit Tests (Dual-Control) ---

def test_check_site_access(seeded_db: DatabaseManager, acc_site: str) -> None:
    tool = USER_TOOLS["check_site_access"]
    res = tool.execute(seeded_db, site_code=acc_site)
    assert res.success is True
    assert res.data["road_status"] in ["open", "restricted_wet", "closed"]


def test_confirm_delivery_window(seeded_db: DatabaseManager) -> None:
    tool = USER_TOOLS["confirm_delivery_window"]
    res = tool.execute(seeded_db, slot_id="SLOT-0001")
    assert res.success is True
    assert res.data["confirmed_by_site"] is True


def test_report_receiver_on_site(seeded_db: DatabaseManager, acc_site: str) -> None:
    tool = USER_TOOLS["report_receiver_on_site"]
    # Look up certified contact for ACC-001
    contacts = seeded_db.execute_query("SELECT full_name, has_hazmat_cert FROM contacts WHERE account_id = 'ACC-001'")
    name = contacts[0]["full_name"]
    has_cert = bool(contacts[0]["has_hazmat_cert"])

    res = tool.execute(seeded_db, site_code=acc_site, receiver_name=name)
    assert res.success is True
    assert res.data["certified_present"] == has_cert

    # Unrecognized person
    res_unknown = tool.execute(seeded_db, site_code=acc_site, receiver_name="Unknown Joe")
    assert res_unknown.success is True
    assert res_unknown.data["certified_present"] is False


# --- Database Snapshot and Diffing Tests ---

def test_database_snapshot_and_diff(seeded_db: DatabaseManager, acc_site: str) -> None:
    before = seeded_db.get_snapshot()

    # Perform a mutation
    c = AGENT_TOOLS["create_order"].execute(
        seeded_db, account_id="ACC-001", site_code=acc_site,
        items=[{"product_id": "CHEM-101", "quantity": 100.0}]
    )
    assert c.success is True, f"Failed to create order: {c.error}"
    oid = c.data["order_id"]

    after = seeded_db.get_snapshot()
    diff = seeded_db.diff_snapshots(before, after)

    assert "orders" in diff
    assert len(diff["orders"]["added"]) == 1
    assert diff["orders"]["added"][0]["order_id"] == oid
    assert "order_lines" in diff
    assert len(diff["order_lines"]["added"]) == 1

