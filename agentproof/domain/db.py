"""SQLite database management, schema creation, seeding, and diffing for ChemDesk."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    credit_limit REAL NOT NULL,
    payment_terms TEXT NOT NULL DEFAULT 'Net 30',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sites (
    site_code TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    site_name TEXT NOT NULL,
    basin TEXT NOT NULL DEFAULT 'Permian',
    county TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'TX',
    road_type TEXT NOT NULL DEFAULT 'caliche',
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE IF NOT EXISTS contacts (
    contact_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    full_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL,
    has_hazmat_cert INTEGER NOT NULL DEFAULT 0,
    cert_expiration TEXT,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    hazard_class TEXT NOT NULL DEFAULT 'non_regulated',
    chemical_type TEXT NOT NULL,
    unit TEXT NOT NULL DEFAULT 'gal',
    unit_price REAL NOT NULL,
    max_order_qty REAL NOT NULL DEFAULT 8000.0,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS depots (
    depot_id TEXT PRIMARY KEY,
    depot_name TEXT NOT NULL,
    location TEXT NOT NULL,
    capacity_gal REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    depot_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity_available REAL NOT NULL,
    quantity_reserved REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (depot_id) REFERENCES depots(depot_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    UNIQUE(depot_id, product_id)
);

CREATE TABLE IF NOT EXISTS trucks (
    truck_id TEXT PRIMARY KEY,
    license_plate TEXT NOT NULL,
    truck_type TEXT NOT NULL,
    capacity_gal REAL NOT NULL,
    hazmat_certified_classes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS delivery_slots (
    slot_id TEXT PRIMARY KEY,
    depot_id TEXT NOT NULL,
    truck_id TEXT NOT NULL,
    delivery_date TEXT NOT NULL,
    shift TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    order_id TEXT,
    FOREIGN KEY (depot_id) REFERENCES depots(depot_id),
    FOREIGN KEY (truck_id) REFERENCES trucks(truck_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    site_code TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    total_amount REAL NOT NULL DEFAULT 0.0,
    cancellation_fee REAL NOT NULL DEFAULT 0.0,
    delivery_slot_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id),
    FOREIGN KEY (site_code) REFERENCES sites(site_code),
    FOREIGN KEY (delivery_slot_id) REFERENCES delivery_slots(slot_id)
);

CREATE TABLE IF NOT EXISTS order_lines (
    line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity REAL NOT NULL,
    unit_price REAL NOT NULL,
    subtotal REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS incompatibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id_a TEXT NOT NULL,
    product_id_b TEXT NOT NULL,
    reason TEXT NOT NULL,
    FOREIGN KEY (product_id_a) REFERENCES products(product_id),
    FOREIGN KEY (product_id_b) REFERENCES products(product_id),
    UNIQUE(product_id_a, product_id_b)
);

CREATE TABLE IF NOT EXISTS credit_ledger (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    amount REAL NOT NULL,
    order_id TEXT,
    timestamp TEXT NOT NULL,
    description TEXT NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

CREATE TABLE IF NOT EXISTS site_access (
    site_code TEXT PRIMARY KEY,
    road_status TEXT NOT NULL DEFAULT 'open',
    updated_at TEXT NOT NULL,
    FOREIGN KEY (site_code) REFERENCES sites(site_code)
);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id TEXT PRIMARY KEY,
    account_id TEXT,
    caller_name TEXT,
    reason TEXT NOT NULL,
    context_summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class DatabaseManager:
    """Manages an isolated SQLite instance for an evaluation episode."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None or db_path == ":memory:":
            self.db_path = ":memory:"
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self.db_path = db_path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(db_path, check_same_thread=False)

        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.executescript(SCHEMA_SQL)
        self.conn.commit()

    def seed_from_data(self, seed_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Seed tables from in-memory dictionary of rows."""
        cursor = self.conn.cursor()
        for table, rows in seed_data.items():
            if not rows:
                continue
            first_row = rows[0]
            columns = list(first_row.keys())
            placeholders = ", ".join(["?"] * len(columns))
            col_names = ", ".join(columns)
            query = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"
            data = [tuple(r.get(c) for c in columns) for r in rows]
            cursor.executemany(query, data)
        self.conn.commit()

    def seed_from_directory(self, seed_dir: str) -> None:
        """Seed all tables from JSON files in the specified directory."""
        path = Path(seed_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

        seed_data: Dict[str, List[Dict[str, Any]]] = {}
        for json_file in path.glob("*.json"):
            table_name = json_file.stem
            with open(json_file, "r", encoding="utf-8") as f:
                seed_data[table_name] = json.load(f)

        self.seed_from_data(seed_data)

    def execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        """Execute a read-only query and return rows."""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def execute_write(self, query: str, params: Tuple[Any, ...] = ()) -> int:
        """Execute an insert/update/delete query and commit changes."""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor.lastrowid or cursor.rowcount

    def get_snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        """Capture entire DB state as a dictionary of table name to rows."""
        tables = [
            "accounts", "sites", "contacts", "products", "depots",
            "inventory", "trucks", "delivery_slots", "orders",
            "order_lines", "incompatibility", "credit_ledger", "site_access", "escalations"
        ]
        snapshot: Dict[str, List[Dict[str, Any]]] = {}
        cursor = self.conn.cursor()
        for t in tables:
            cursor.execute(f"SELECT * FROM {t}")
            rows = cursor.fetchall()
            snapshot[t] = [dict(r) for r in rows]
        return snapshot

    def diff_snapshots(
        self, before: Dict[str, List[Dict[str, Any]]], after: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute state difference between before and after snapshots."""
        diff: Dict[str, Dict[str, Any]] = {}

        for table in after.keys():
            before_rows = {json.dumps(r, sort_keys=True): r for r in before.get(table, [])}
            after_rows = {json.dumps(r, sort_keys=True): r for r in after.get(table, [])}

            added = [r for k, r in after_rows.items() if k not in before_rows]
            removed = [r for k, r in before_rows.items() if k not in after_rows]

            if added or removed:
                diff[table] = {
                    "added": added,
                    "removed": removed,
                }

        return diff

    def close(self) -> None:
        self.conn.close()
