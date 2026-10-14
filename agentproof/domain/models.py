"""Pydantic domain models for ChemDesk oilfield chemical order desk."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    PICKED = "picked"
    LOADED = "loaded"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class HazardClass(str, Enum):
    CLASS_3 = "3"
    CLASS_8 = "8"
    CLASS_9 = "9"
    NON_REGULATED = "non_regulated"


class ShiftType(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"


class RoadStatus(str, Enum):
    OPEN = "open"
    RESTRICTED_WET = "restricted_wet"
    CLOSED = "closed"


class Account(BaseModel):
    account_id: str
    company_name: str
    status: str = "active"
    credit_limit: float
    payment_terms: str = "Net 30"
    created_at: str


class Site(BaseModel):
    site_code: str
    account_id: str
    site_name: str
    basin: str = "Permian"
    county: str
    state: str = "TX"
    road_type: str = "caliche"
    is_active: bool = True


class Contact(BaseModel):
    contact_id: str
    account_id: str
    full_name: str
    phone: str
    email: str
    role: str
    has_hazmat_cert: bool = False
    cert_expiration: Optional[str] = None


class Product(BaseModel):
    product_id: str
    sku: str
    name: str
    category: str
    hazard_class: str = "non_regulated"
    chemical_type: str
    unit: str = "gal"
    unit_price: float
    max_order_qty: float = 8000.0
    is_active: bool = True


class Depot(BaseModel):
    depot_id: str
    depot_name: str
    location: str
    capacity_gal: float


class InventoryRecord(BaseModel):
    inventory_id: int
    depot_id: str
    product_id: str
    quantity_available: float
    quantity_reserved: float = 0.0


class OrderLine(BaseModel):
    line_id: Optional[int] = None
    order_id: str
    product_id: str
    quantity: float
    unit_price: float
    subtotal: float


class Order(BaseModel):
    order_id: str
    account_id: str
    site_code: str
    status: OrderStatus = OrderStatus.DRAFT
    total_amount: float = 0.0
    cancellation_fee: float = 0.0
    delivery_slot_id: Optional[str] = None
    created_at: str
    updated_at: str
    notes: Optional[str] = None
    lines: list[OrderLine] = Field(default_factory=list)


class DeliverySlot(BaseModel):
    slot_id: str
    depot_id: str
    truck_id: str
    delivery_date: str
    shift: ShiftType
    status: str = "open"
    order_id: Optional[str] = None


class ChemicalIncompatibility(BaseModel):
    product_id_a: str
    product_id_b: str
    reason: str


class CreditEntry(BaseModel):
    entry_id: Optional[int] = None
    account_id: str
    entry_type: str
    amount: float
    order_id: Optional[str] = None
    timestamp: str
    description: str
