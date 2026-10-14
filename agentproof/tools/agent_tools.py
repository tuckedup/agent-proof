"""Agent tools for the ChemDesk order desk assistant.

All tools adhere to Pydantic argument models, detailed docstrings explaining purpose,
constraints, and side effects without em-dashes.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentproof.domain.db import DatabaseManager
from agentproof.tools.base import ToolDefinition, ToolResult


# 1. find_account
class FindAccountArgs(BaseModel):
    query: str = Field(description="Account ID (e.g. 'ACC-001') or partial company name to search for.")


def find_account(db: DatabaseManager, query: str) -> ToolResult:
    """Search for customer accounts by ID or company name.

    Returns matching active accounts including ID, company name, credit limit, and terms.
    Does not expose sensitive ledger details directly.
    """
    cleaned = f"%{query.strip()}%"
    rows = db.execute_query(
        "SELECT account_id, company_name, status, credit_limit, payment_terms FROM accounts "
        "WHERE account_id LIKE ? OR company_name LIKE ?",
        (cleaned, cleaned)
    )
    results = [dict(r) for r in rows]
    return ToolResult(success=True, data={"matches": results, "count": len(results)})


# 2. verify_caller
class VerifyCallerArgs(BaseModel):
    account_id: str = Field(description="The customer account identifier, e.g. 'ACC-001'.")
    site_code: str = Field(description="The well site code where chemical delivery or work is occurring, e.g. 'SITE-MID-001'.")
    caller_name: str = Field(description="Full name of the caller requesting access or changes.")


def verify_caller(db: DatabaseManager, account_id: str, site_code: str, caller_name: str) -> ToolResult:
    """Verify caller identity against account contacts and registered well sites.

    Verification requires that the account exists and is active, the site code belongs
    to the account, and the caller name matches an authorized contact on file.
    Returns verified status, contact role, and whether the caller holds hazmat certifications.
    """
    acc_rows = db.execute_query("SELECT * FROM accounts WHERE account_id = ? AND status = 'active'", (account_id,))
    if not acc_rows:
        return ToolResult(success=False, error=f"Account '{account_id}' not found or inactive.")

    site_rows = db.execute_query(
        "SELECT * FROM sites WHERE site_code = ? AND account_id = ? AND is_active = 1",
        (site_code, account_id)
    )
    if not site_rows:
        return ToolResult(
            success=False,
            error=f"Site code '{site_code}' is not registered or active for account '{account_id}'."
        )

    contact_rows = db.execute_query(
        "SELECT * FROM contacts WHERE account_id = ? AND lower(full_name) = lower(?)",
        (account_id, caller_name.strip())
    )
    if not contact_rows:
        return ToolResult(
            success=False,
            error=f"Caller '{caller_name}' is not listed as an authorized contact for account '{account_id}'."
        )

    contact = dict(contact_rows[0])
    return ToolResult(
        success=True,
        data={
            "verified": True,
            "account_id": account_id,
            "site_code": site_code,
            "contact_name": contact["full_name"],
            "role": contact["role"],
            "has_hazmat_cert": bool(contact["has_hazmat_cert"]),
            "cert_expiration": contact["cert_expiration"],
        }
    )


# 3. get_order
class GetOrderArgs(BaseModel):
    order_id: str = Field(description="The unique order ID, e.g. 'ORD-0001'.")


def get_order(db: DatabaseManager, order_id: str) -> ToolResult:
    """Retrieve full details of an existing order and its line items.

    Returns order status, destination site, line item quantities, pricing,
    scheduled delivery slot, and cancellation fee if applicable.
    """
    rows = db.execute_query("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    if not rows:
        return ToolResult(success=False, error=f"Order '{order_id}' not found.")

    order = dict(rows[0])
    lines_rows = db.execute_query(
        "SELECT l.line_id, l.product_id, p.name, p.hazard_class, l.quantity, l.unit_price, l.subtotal "
        "FROM order_lines l JOIN products p ON l.product_id = p.product_id "
        "WHERE l.order_id = ?",
        (order_id,)
    )
    order["lines"] = [dict(line_row) for line_row in lines_rows]
    return ToolResult(success=True, data=order)


# 4. list_orders
class ListOrdersArgs(BaseModel):
    account_id: str = Field(description="Customer account identifier.")
    status: Optional[str] = Field(default=None, description="Optional status filter ('draft', 'submitted', 'approved', 'picked', 'loaded', 'in_transit', 'delivered', 'cancelled').")


def list_orders(db: DatabaseManager, account_id: str, status: Optional[str] = None) -> ToolResult:
    """List historical and active orders for a customer account.

    Can be filtered by order status. Returns summaries including total cost and site.
    """
    if status:
        rows = db.execute_query(
            "SELECT order_id, site_code, status, total_amount, created_at, delivery_slot_id "
            "FROM orders WHERE account_id = ? AND status = ? ORDER BY created_at DESC",
            (account_id, status)
        )
    else:
        rows = db.execute_query(
            "SELECT order_id, site_code, status, total_amount, created_at, delivery_slot_id "
            "FROM orders WHERE account_id = ? ORDER BY created_at DESC",
            (account_id,)
        )
    orders = [dict(r) for r in rows]
    return ToolResult(success=True, data={"orders": orders, "count": len(orders)})


# 5. check_stock
class CheckStockArgs(BaseModel):
    product_id: str = Field(description="Product chemical ID, e.g. 'CHEM-801'.")
    depot_id: Optional[str] = Field(default=None, description="Optional depot ID ('DEP-MID', 'DEP-PEC', 'DEP-CAR', 'DEP-ODE'). If omitted, sums across all depots.")


def check_stock(db: DatabaseManager, product_id: str, depot_id: Optional[str] = None) -> ToolResult:
    """Check inventory levels and availability for a chemical product.

    Queries warehouse records to determine available and reserved volumes in gallons.
    """
    prod_rows = db.execute_query("SELECT * FROM products WHERE product_id = ?", (product_id,))
    if not prod_rows:
        return ToolResult(success=False, error=f"Product '{product_id}' not found in catalog.")
    prod = dict(prod_rows[0])

    if depot_id:
        rows = db.execute_query(
            "SELECT i.quantity_available, i.quantity_reserved, d.depot_name "
            "FROM inventory i JOIN depots d ON i.depot_id = d.depot_id "
            "WHERE i.product_id = ? AND i.depot_id = ?",
            (product_id, depot_id)
        )
        if not rows:
            return ToolResult(success=True, data={"product_id": product_id, "depot_id": depot_id, "available": 0.0, "reserved": 0.0})
        r = dict(rows[0])
        return ToolResult(success=True, data={
            "product_id": product_id,
            "product_name": prod["name"],
            "depot_id": depot_id,
            "depot_name": r["depot_name"],
            "available_gal": r["quantity_available"],
            "reserved_gal": r["quantity_reserved"]
        })
    else:
        rows = db.execute_query(
            "SELECT i.depot_id, d.depot_name, i.quantity_available, i.quantity_reserved "
            "FROM inventory i JOIN depots d ON i.depot_id = d.depot_id "
            "WHERE i.product_id = ?",
            (product_id,)
        )
        depot_breakdown = [dict(r) for r in rows]
        total_available = sum(r["quantity_available"] for r in depot_breakdown)
        total_reserved = sum(r["quantity_reserved"] for r in depot_breakdown)
        return ToolResult(success=True, data={
            "product_id": product_id,
            "product_name": prod["name"],
            "total_available_gal": total_available,
            "total_reserved_gal": total_reserved,
            "depots": depot_breakdown
        })


# 6. quote_price
class QuotePriceArgs(BaseModel):
    product_id: str = Field(description="Product identifier, e.g. 'CHEM-801'.")
    quantity: float = Field(description="Requested volume in gallons or product units.")


def quote_price(db: DatabaseManager, product_id: str, quantity: float) -> ToolResult:
    """Calculate pricing quote, per-unit rate, and regulatory classification.

    Verifies unit price and checks if requested quantity complies with maximum order limits.
    """
    rows = db.execute_query("SELECT * FROM products WHERE product_id = ?", (product_id,))
    if not rows:
        return ToolResult(success=False, error=f"Product '{product_id}' not found.")
    prod = dict(rows[0])

    if quantity <= 0:
        return ToolResult(success=False, error="Requested quantity must be positive.")

    if quantity > prod["max_order_qty"]:
        return ToolResult(
            success=False,
            error=f"Requested quantity {quantity} gal exceeds maximum single order limit of {prod['max_order_qty']} gal for {prod['name']} (Class {prod['hazard_class']})."
        )

    subtotal = round(quantity * prod["unit_price"], 2)
    return ToolResult(success=True, data={
        "product_id": product_id,
        "name": prod["name"],
        "hazard_class": prod["hazard_class"],
        "unit": prod["unit"],
        "unit_price": prod["unit_price"],
        "quantity": quantity,
        "subtotal": subtotal,
        "requires_hazmat_receiver": prod["hazard_class"] in ["3", "8"]
    })


# 7. create_order
class OrderItem(BaseModel):
    product_id: str = Field(description="Chemical product ID.")
    quantity: float = Field(description="Volume in units/gallons.")


class CreateOrderArgs(BaseModel):
    account_id: str = Field(description="Customer account ID.")
    site_code: str = Field(description="Target well site code.")
    items: List[OrderItem] = Field(description="List of products and quantities to order.")
    notes: Optional[str] = Field(default=None, description="Optional delivery or field instructions.")


def create_order(db: DatabaseManager, account_id: str, site_code: str, items: List[Dict[str, Any]], notes: Optional[str] = None) -> ToolResult:
    """Create a new chemical order in draft status.

    Validates that the account and site exist, validates product quantities against
    regulatory caps, and creates order lines in the database.
    """
    if not items:
        return ToolResult(success=False, error="Order must contain at least one item.")

    # Check account and site
    acc_rows = db.execute_query("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
    if not acc_rows:
        return ToolResult(success=False, error=f"Account '{account_id}' does not exist.")

    site_rows = db.execute_query("SELECT * FROM sites WHERE site_code = ? AND account_id = ?", (site_code, account_id))
    if not site_rows:
        return ToolResult(success=False, error=f"Site '{site_code}' is not associated with account '{account_id}'.")

    # Generate next order id
    count_rows = db.execute_query("SELECT count(*) as cnt FROM orders")
    cnt_val = count_rows[0]["cnt"]
    next_id = int(cnt_val if cnt_val is not None else 0) + 1
    order_id = f"ORD-{next_id:04d}"

    total_amount = 0.0
    prepared_lines = []

    for item in items:
        if isinstance(item, dict):
            pid = str(item.get("product_id", ""))
            qty = float(item.get("quantity", 0.0))
        else:
            pid = str(getattr(item, "product_id", ""))
            qty = float(getattr(item, "quantity", 0.0))

        prod_rows = db.execute_query("SELECT * FROM products WHERE product_id = ?", (pid,))
        if not prod_rows:
            return ToolResult(success=False, error=f"Product '{pid}' not found.")
        prod = dict(prod_rows[0])

        if qty <= 0:
            return ToolResult(success=False, error=f"Invalid quantity {qty} for product '{pid}'.")

        max_limit = float(prod.get("max_order_qty", 8000.0))
        if qty > max_limit:
            return ToolResult(
                success=False,
                error=f"Quantity {qty} exceeds single-order limit of {max_limit} for product '{pid}'."
            )

        subtotal = round(qty * float(prod["unit_price"]), 2)
        total_amount += subtotal
        prepared_lines.append((order_id, pid, qty, float(prod["unit_price"]), subtotal))

    now_str = datetime.now(timezone.utc).isoformat()
    db.execute_write(
        "INSERT INTO orders (order_id, account_id, site_code, status, total_amount, created_at, updated_at, notes) "
        "VALUES (?, ?, ?, 'draft', ?, ?, ?, ?)",
        (order_id, account_id, site_code, round(total_amount, 2), now_str, now_str, notes)
    )

    for line in prepared_lines:
        db.execute_write(
            "INSERT INTO order_lines (order_id, product_id, quantity, unit_price, subtotal) "
            "VALUES (?, ?, ?, ?, ?)",
            line
        )

    return ToolResult(success=True, data={
        "order_id": order_id,
        "account_id": account_id,
        "site_code": site_code,
        "status": "draft",
        "total_amount": round(total_amount, 2),
        "items_count": len(prepared_lines)
    })


# 8. modify_order
class ModifyOrderArgs(BaseModel):
    order_id: str = Field(description="Order ID to modify.")
    action: str = Field(description="Action to perform: 'update_quantity', 'add_item', or 'remove_item'.")
    product_id: Optional[str] = Field(default=None, description="Product ID being modified or added.")
    quantity: Optional[float] = Field(default=None, description="New quantity for update_quantity or add_item.")


def modify_order(db: DatabaseManager, order_id: str, action: str, product_id: Optional[str] = None, quantity: Optional[float] = None) -> ToolResult:
    """Modify an existing order before staging.

    Policy Rule: Order modifications are permitted ONLY while status is 'draft', 'submitted',
    or 'approved'. Once an order reaches 'picked' status or later, modifications are strictly
    forbidden by warehouse operating policy.
    """
    rows = db.execute_query("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    if not rows:
        return ToolResult(success=False, error=f"Order '{order_id}' not found.")
    order = dict(rows[0])

    # Check status cutoff
    if order["status"] in ["picked", "loaded", "in_transit", "delivered", "cancelled"]:
        return ToolResult(
            success=False,
            error=f"Order modification forbidden: order '{order_id}' is in '{order['status']}' status. Modifications are only permitted prior to 'picked' status."
        )

    if action == "update_quantity":
        if not product_id or quantity is None:
            return ToolResult(success=False, error="product_id and quantity are required for update_quantity.")
        if quantity <= 0:
            return ToolResult(success=False, error="Quantity must be greater than 0.")

        prod_rows = db.execute_query("SELECT * FROM products WHERE product_id = ?", (product_id,))
        if not prod_rows:
            return ToolResult(success=False, error=f"Product '{product_id}' not found.")
        prod = dict(prod_rows[0])

        if quantity > prod["max_order_qty"]:
            return ToolResult(success=False, error=f"Quantity exceeds max order limit of {prod['max_order_qty']}.")

        new_subtotal = round(quantity * prod["unit_price"], 2)
        db.execute_write(
            "UPDATE order_lines SET quantity = ?, subtotal = ? WHERE order_id = ? AND product_id = ?",
            (quantity, new_subtotal, order_id, product_id)
        )
    elif action == "add_item":
        if not product_id or quantity is None:
            return ToolResult(success=False, error="product_id and quantity are required for add_item.")
        prod_rows = db.execute_query("SELECT * FROM products WHERE product_id = ?", (product_id,))
        if not prod_rows:
            return ToolResult(success=False, error=f"Product '{product_id}' not found.")
        prod = dict(prod_rows[0])

        if quantity > prod["max_order_qty"]:
            return ToolResult(success=False, error=f"Quantity exceeds max limit of {prod['max_order_qty']}.")

        subtotal = round(quantity * prod["unit_price"], 2)
        db.execute_write(
            "INSERT INTO order_lines (order_id, product_id, quantity, unit_price, subtotal) VALUES (?, ?, ?, ?, ?)",
            (order_id, product_id, quantity, prod["unit_price"], subtotal)
        )
    elif action == "remove_item":
        if not product_id:
            return ToolResult(success=False, error="product_id required for remove_item.")
        db.execute_write("DELETE FROM order_lines WHERE order_id = ? AND product_id = ?", (order_id, product_id))
    else:
        return ToolResult(success=False, error=f"Unrecognized action '{action}'.")

    # Recalculate order total
    sum_rows = db.execute_query("SELECT coalesce(sum(subtotal), 0.0) as total FROM order_lines WHERE order_id = ?", (order_id,))
    new_total = round(sum_rows[0]["total"], 2)
    now_str = datetime.now(timezone.utc).isoformat()
    db.execute_write("UPDATE orders SET total_amount = ?, updated_at = ? WHERE order_id = ?", (new_total, now_str, order_id))

    return ToolResult(success=True, data={"order_id": order_id, "action": action, "new_total_amount": new_total})


# 9. cancel_order
class CancelOrderArgs(BaseModel):
    order_id: str = Field(description="Order ID to cancel.")
    reason: str = Field(description="Reason for cancellation.")


def cancel_order(db: DatabaseManager, order_id: str, reason: str) -> ToolResult:
    """Cancel an order and assess cancellation fees according to policy.

    Policy Rules:
    - Orders in 'draft', 'submitted', or 'approved' status incur a $0 cancellation fee.
    - Orders in 'picked' or 'loaded' status incur a mandatory $350 restocking and staging fee.
    - Orders in 'in_transit' or 'delivered' status cannot be cancelled via the order desk and must escalate.
    """
    rows = db.execute_query("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    if not rows:
        return ToolResult(success=False, error=f"Order '{order_id}' not found.")
    order = dict(rows[0])

    if order["status"] == "cancelled":
        return ToolResult(success=False, error=f"Order '{order_id}' is already cancelled.")

    if order["status"] in ["in_transit", "delivered"]:
        return ToolResult(
            success=False,
            error=f"Cannot cancel order '{order_id}' in '{order['status']}' status. Dispatch in progress; request must be escalated to logistics dispatch."
        )

    now_str = datetime.now(timezone.utc).isoformat()
    fee = 0.0

    if order["status"] in ["picked", "loaded"]:
        fee = 350.0
        # Post fee to credit ledger
        db.execute_write(
            "INSERT INTO credit_ledger (account_id, entry_type, amount, order_id, timestamp, description) "
            "VALUES (?, 'fee', ?, ?, ?, ?)",
            (order["account_id"], -fee, order_id, now_str, f"Restocking fee for cancelled order {order_id} in {order['status']} status.")
        )

    # Release any booked slot
    if order["delivery_slot_id"]:
        db.execute_write(
            "UPDATE delivery_slots SET status = 'open', order_id = NULL WHERE slot_id = ?",
            (order["delivery_slot_id"],)
        )

    db.execute_write(
        "UPDATE orders SET status = 'cancelled', cancellation_fee = ?, updated_at = ?, notes = ? WHERE order_id = ?",
        (fee, now_str, f"Cancelled: {reason}", order_id)
    )

    return ToolResult(success=True, data={
        "order_id": order_id,
        "previous_status": order["status"],
        "status": "cancelled",
        "cancellation_fee": fee,
        "fee_assessed": fee > 0
    })


# 10. list_delivery_slots
class ListDeliverySlotsArgs(BaseModel):
    depot_id: Optional[str] = Field(default=None, description="Optional depot ID filter ('DEP-MID', 'DEP-PEC', etc.).")
    delivery_date: Optional[str] = Field(default=None, description="Optional target date filter, e.g. '2026-10-16'.")


def list_delivery_slots(db: DatabaseManager, depot_id: Optional[str] = None, delivery_date: Optional[str] = None) -> ToolResult:
    """List open delivery slots available for truck dispatch.

    Returns available dates, shifts (morning, afternoon, night), and truck capacities.
    """
    query = (
        "SELECT s.slot_id, s.depot_id, s.truck_id, s.delivery_date, s.shift, t.truck_type, t.hazmat_certified_classes "
        "FROM delivery_slots s JOIN trucks t ON s.truck_id = t.truck_id "
        "WHERE s.status = 'open' "
    )
    params = []
    if depot_id:
        query += "AND s.depot_id = ? "
        params.append(depot_id)
    if delivery_date:
        query += "AND s.delivery_date = ? "
        params.append(delivery_date)
    query += "ORDER BY s.delivery_date ASC, s.shift ASC"

    rows = db.execute_query(query, tuple(params))
    slots = [dict(r) for r in rows]
    return ToolResult(success=True, data={"available_slots": slots, "count": len(slots)})


# 11. book_delivery
class BookDeliveryArgs(BaseModel):
    order_id: str = Field(description="Order ID to schedule.")
    slot_id: str = Field(description="Target delivery slot ID.")


def book_delivery(db: DatabaseManager, order_id: str, slot_id: str) -> ToolResult:
    """Assign an open delivery slot to an order and reserve the truck dispatch.

    Validates that the order exists, the slot is open, and transitions order status
    to 'approved' or 'submitted'.
    """
    order_rows = db.execute_query("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    if not order_rows:
        return ToolResult(success=False, error=f"Order '{order_id}' not found.")

    slot_rows = db.execute_query("SELECT * FROM delivery_slots WHERE slot_id = ?", (slot_id,))
    if not slot_rows:
        return ToolResult(success=False, error=f"Delivery slot '{slot_id}' not found.")
    slot = dict(slot_rows[0])

    if slot["status"] != "open":
        return ToolResult(success=False, error=f"Slot '{slot_id}' is already {slot['status']}.")

    now_str = datetime.now(timezone.utc).isoformat()
    db.execute_write(
        "UPDATE delivery_slots SET status = 'booked', order_id = ? WHERE slot_id = ?",
        (order_id, slot_id)
    )
    db.execute_write(
        "UPDATE orders SET delivery_slot_id = ?, status = 'approved', updated_at = ? WHERE order_id = ?",
        (slot_id, now_str, order_id)
    )

    return ToolResult(success=True, data={
        "order_id": order_id,
        "slot_id": slot_id,
        "delivery_date": slot["delivery_date"],
        "shift": slot["shift"],
        "status": "booked"
    })


# 12. check_compatibility
class CheckCompatibilityArgs(BaseModel):
    product_ids: List[str] = Field(description="List of product IDs to check for multi-product transport compatibility.")


def check_compatibility(db: DatabaseManager, product_ids: List[str]) -> ToolResult:
    """Check pairwise transport compatibility for multiple chemicals.

    Returns whether all chemical pairs can safely ship together on the same vehicle.
    Flags hazardous combinations such as strong mineral acids with flammable solvents or caustics.
    """
    if len(product_ids) <= 1:
        return ToolResult(success=True, data={"compatible": True, "conflicts": []})

    conflicts = []
    for i in range(len(product_ids)):
        for j in range(i + 1, len(product_ids)):
            p_a, p_b = product_ids[i], product_ids[j]
            rows = db.execute_query(
                "SELECT * FROM incompatibility WHERE (product_id_a = ? AND product_id_b = ?) "
                "OR (product_id_a = ? AND product_id_b = ?)",
                (p_a, p_b, p_b, p_a)
            )
            if rows:
                conflicts.append({
                    "product_a": p_a,
                    "product_b": p_b,
                    "reason": rows[0]["reason"]
                })

    is_compatible = len(conflicts) == 0
    return ToolResult(success=True, data={
        "compatible": is_compatible,
        "conflicts": conflicts
    })


# 13. check_credit
class CheckCreditArgs(BaseModel):
    account_id: str = Field(description="Customer account identifier.")
    order_amount: float = Field(description="Total dollar amount of the proposed order.")


def check_credit(db: DatabaseManager, account_id: str, order_amount: float) -> ToolResult:
    """Evaluate customer credit availability against outstanding commitments.

    Computes current ledger balance, committed pending orders, and compares against
    account credit ceiling. If order_amount exceeds available credit, returns credit_sufficient = False.
    """
    acc_rows = db.execute_query("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
    if not acc_rows:
        return ToolResult(success=False, error=f"Account '{account_id}' not found.")
    acc = dict(acc_rows[0])
    credit_limit = acc["credit_limit"]

    # Sum ledger balance (positive = payment/credit, negative = invoice/fee)
    ledger_rows = db.execute_query(
        "SELECT coalesce(sum(amount), 0.0) as balance FROM credit_ledger WHERE account_id = ?",
        (account_id,)
    )
    current_ledger_balance = ledger_rows[0]["balance"]

    # Sum pending unbilled orders
    pending_rows = db.execute_query(
        "SELECT coalesce(sum(total_amount), 0.0) as pending_total FROM orders "
        "WHERE account_id = ? AND status IN ('draft', 'submitted', 'approved', 'picked', 'loaded')",
        (account_id,)
    )
    pending_orders_amount = pending_rows[0]["pending_total"]

    # Available credit formula
    available_credit = max(0.0, credit_limit + current_ledger_balance - pending_orders_amount)
    sufficient = available_credit >= order_amount

    return ToolResult(success=True, data={
        "account_id": account_id,
        "credit_limit": credit_limit,
        "current_balance": current_ledger_balance,
        "pending_orders": pending_orders_amount,
        "available_credit": available_credit,
        "order_amount": order_amount,
        "credit_sufficient": sufficient
    })


# 14. escalate_to_human
class EscalateToHumanArgs(BaseModel):
    reason: str = Field(description="Primary reason for escalation ('credit_limit_exceeded', 'road_closed', 'hazmat_unverified', 'unresolved_caller', 'policy_forbidden').")
    context_summary: str = Field(description="Concise factual summary of the customer request and why the agent cannot resolve it.")
    account_id: Optional[str] = Field(default=None, description="Associated account ID if known.")
    caller_name: Optional[str] = Field(default=None, description="Name of the caller if known.")


def escalate_to_human(db: DatabaseManager, reason: str, context_summary: str, account_id: Optional[str] = None, caller_name: Optional[str] = None) -> ToolResult:
    """Escalate interaction to a human representative when policy limits are reached.

    Records the escalation event in the database for human follow-up.
    Must be called after two unresolved clarification attempts, for credit overrides,
    or upon unresolvable road/hazmat restrictions.
    """
    count_rows = db.execute_query("SELECT count(*) as cnt FROM escalations")
    esc_id = f"ESC-{(count_rows[0]['cnt'] + 1):04d}"
    now_str = datetime.now(timezone.utc).isoformat()

    db.execute_write(
        "INSERT INTO escalations (escalation_id, account_id, caller_name, reason, context_summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (esc_id, account_id, caller_name, reason, context_summary, now_str)
    )

    return ToolResult(success=True, data={
        "escalation_id": esc_id,
        "status": "escalated",
        "reason": reason,
        "message": f"Interaction escalated to human desk with reference ID {esc_id}."
    })


# Registry of all 14 Agent Tools
AGENT_TOOLS: Dict[str, ToolDefinition] = {
    "find_account": ToolDefinition(
        name="find_account",
        description="Search for customer accounts by ID or company name. Returns active account records.",
        args_model=FindAccountArgs,
        func=find_account,
        is_write=False,
        requestor="assistant"
    ),
    "verify_caller": ToolDefinition(
        name="verify_caller",
        description="Verify caller identity against account contacts and registered well sites. Requires account_id, site_code, and caller_name.",
        args_model=VerifyCallerArgs,
        func=verify_caller,
        is_write=False,
        requestor="assistant"
    ),
    "get_order": ToolDefinition(
        name="get_order",
        description="Retrieve full details of an existing order and its line items.",
        args_model=GetOrderArgs,
        func=get_order,
        is_write=False,
        requestor="assistant"
    ),
    "list_orders": ToolDefinition(
        name="list_orders",
        description="List orders for a customer account with optional status filter.",
        args_model=ListOrdersArgs,
        func=list_orders,
        is_write=False,
        requestor="assistant"
    ),
    "check_stock": ToolDefinition(
        name="check_stock",
        description="Check chemical inventory levels and availability across depots or at a specific depot.",
        args_model=CheckStockArgs,
        func=check_stock,
        is_write=False,
        requestor="assistant"
    ),
    "quote_price": ToolDefinition(
        name="quote_price",
        description="Calculate pricing quote and check product volume limit compliance for a chemical.",
        args_model=QuotePriceArgs,
        func=quote_price,
        is_write=False,
        requestor="assistant"
    ),
    "create_order": ToolDefinition(
        name="create_order",
        description="Create a new chemical order in draft status with specified products and quantities. Mutates database.",
        args_model=CreateOrderArgs,
        func=create_order,
        is_write=True,
        requestor="assistant"
    ),
    "modify_order": ToolDefinition(
        name="modify_order",
        description="Modify line items on an existing order. Allowed only before 'picked' status. Mutates database.",
        args_model=ModifyOrderArgs,
        func=modify_order,
        is_write=True,
        requestor="assistant"
    ),
    "cancel_order": ToolDefinition(
        name="cancel_order",
        description="Cancel an existing order. Assesses $0 fee before 'picked', $350 fee for 'picked' or 'loaded'. Mutates database.",
        args_model=CancelOrderArgs,
        func=cancel_order,
        is_write=True,
        requestor="assistant"
    ),
    "list_delivery_slots": ToolDefinition(
        name="list_delivery_slots",
        description="List open delivery slots available for truck dispatch.",
        args_model=ListDeliverySlotsArgs,
        func=list_delivery_slots,
        is_write=False,
        requestor="assistant"
    ),
    "book_delivery": ToolDefinition(
        name="book_delivery",
        description="Book an open delivery slot for an order and assign truck dispatch. Mutates database.",
        args_model=BookDeliveryArgs,
        func=book_delivery,
        is_write=True,
        requestor="assistant"
    ),
    "check_compatibility": ToolDefinition(
        name="check_compatibility",
        description="Check pairwise chemical compatibility for transporting multiple products on the same vehicle.",
        args_model=CheckCompatibilityArgs,
        func=check_compatibility,
        is_write=False,
        requestor="assistant"
    ),
    "check_credit": ToolDefinition(
        name="check_credit",
        description="Check customer credit limit and available balance against a proposed order amount.",
        args_model=CheckCreditArgs,
        func=check_credit,
        is_write=False,
        requestor="assistant"
    ),
    "escalate_to_human": ToolDefinition(
        name="escalate_to_human",
        description="Escalate interaction to a human representative when policy thresholds are breached or clarification fails twice. Mutates database.",
        args_model=EscalateToHumanArgs,
        func=escalate_to_human,
        is_write=True,
        requestor="assistant"
    ),
}
