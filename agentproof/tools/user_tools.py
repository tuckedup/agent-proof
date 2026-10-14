"""Site-side user tools for dual-control simulation.

These tools represent ground truth checks on the well site that only the user simulator
can execute. The agent cannot observe these results directly; it must query the user
and rely on user-reported information.
"""

from typing import Dict

from pydantic import BaseModel, Field

from agentproof.domain.db import DatabaseManager
from agentproof.tools.base import ToolDefinition, ToolResult


# 1. check_site_access
class CheckSiteAccessArgs(BaseModel):
    site_code: str = Field(description="The well site code, e.g. 'SITE-MID-001'.")


def check_site_access(db: DatabaseManager, site_code: str) -> ToolResult:
    """Check physical road access and county weather restrictions at the well site.

    User tool executed on the site side. Returns 'open', 'restricted_wet', or 'closed'.
    """
    rows = db.execute_query("SELECT road_status, updated_at FROM site_access WHERE site_code = ?", (site_code,))
    if not rows:
        return ToolResult(
            success=False,
            error=f"Well site '{site_code}' has no recorded road status."
        )

    status = rows[0]["road_status"]
    return ToolResult(
        success=True,
        data={
            "site_code": site_code,
            "road_status": status,
            "is_passable": status == "open",
            "requires_low_ground_pressure": status == "restricted_wet"
        }
    )


# 2. confirm_delivery_window
class ConfirmDeliveryWindowArgs(BaseModel):
    slot_id: str = Field(description="The delivery slot proposed by the agent, e.g. 'SLOT-0001'.")


def confirm_delivery_window(db: DatabaseManager, slot_id: str) -> ToolResult:
    """Crew lead confirms that the well site crew can accept delivery during the proposed slot.

    User tool executed on the site side to validate on-pad operations and unloading capability.
    """
    rows = db.execute_query(
        "SELECT slot_id, delivery_date, shift, status FROM delivery_slots WHERE slot_id = ?",
        (slot_id,)
    )
    if not rows:
        return ToolResult(success=False, error=f"Delivery slot '{slot_id}' not found.")

    slot = dict(rows[0])
    return ToolResult(
        success=True,
        data={
            "slot_id": slot_id,
            "delivery_date": slot["delivery_date"],
            "shift": slot["shift"],
            "confirmed_by_site": True
        }
    )


# 3. report_receiver_on_site
class ReportReceiverOnSiteArgs(BaseModel):
    site_code: str = Field(description="The well site code where delivery is scheduled.")
    receiver_name: str = Field(description="Name of the person designated to receive and sign for chemicals.")


def report_receiver_on_site(db: DatabaseManager, site_code: str, receiver_name: str) -> ToolResult:
    """Verify and report whether a certified hazmat receiver will be on site.

    User tool executed on the site side. Validates whether the designated receiver
    holds an active DOT hazardous materials certification required for Class 3 and Class 8 shipments.
    """
    site_rows = db.execute_query("SELECT account_id FROM sites WHERE site_code = ?", (site_code,))
    if not site_rows:
        return ToolResult(success=False, error=f"Site '{site_code}' not recognized.")
    account_id = site_rows[0]["account_id"]

    contact_rows = db.execute_query(
        "SELECT full_name, has_hazmat_cert, cert_expiration FROM contacts "
        "WHERE account_id = ? AND lower(full_name) = lower(?)",
        (account_id, receiver_name.strip())
    )
    if not contact_rows:
        return ToolResult(
            success=True,
            data={
                "site_code": site_code,
                "receiver_name": receiver_name,
                "certified_present": False,
                "reason": f"Person '{receiver_name}' is not listed in registered personnel for this site."
            }
        )

    contact = dict(contact_rows[0])
    is_certified = bool(contact["has_hazmat_cert"])

    return ToolResult(
        success=True,
        data={
            "site_code": site_code,
            "receiver_name": contact["full_name"],
            "certified_present": is_certified,
            "cert_expiration": contact["cert_expiration"] if is_certified else None
        }
    )


# Registry of all 3 User Tools
USER_TOOLS: Dict[str, ToolDefinition] = {
    "check_site_access": ToolDefinition(
        name="check_site_access",
        description="Check road conditions and access restrictions for a well site on the customer side.",
        args_model=CheckSiteAccessArgs,
        func=check_site_access,
        is_write=False,
        requestor="user"
    ),
    "confirm_delivery_window": ToolDefinition(
        name="confirm_delivery_window",
        description="Confirm availability of site crew to receive delivery during the proposed slot.",
        args_model=ConfirmDeliveryWindowArgs,
        func=confirm_delivery_window,
        is_write=False,
        requestor="user"
    ),
    "report_receiver_on_site": ToolDefinition(
        name="report_receiver_on_site",
        description="Verify whether a certified receiver is present at the well site to accept hazmat deliveries.",
        args_model=ReportReceiverOnSiteArgs,
        func=report_receiver_on_site,
        is_write=False,
        requestor="user"
    ),
}
