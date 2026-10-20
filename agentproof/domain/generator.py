"""Deterministic synthetic seed data generator for ChemDesk domain."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List


def generate_seed_dataset(output_dir: str = "domain/seed", seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Generates realistic oilfield chemical supplier seed datasets with seeded RNG."""
    rng = random.Random(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Depots (4 major Permian depots)
    depots = [
        {"depot_id": "DEP-MID", "depot_name": "Midland Central Bulk Terminal", "location": "Midland, TX", "capacity_gal": 250000.0},
        {"depot_id": "DEP-PEC", "depot_name": "Pecos Valley Chemical Depot", "location": "Pecos, TX", "capacity_gal": 180000.0},
        {"depot_id": "DEP-CAR", "depot_name": "Carlsbad Delaware Basin Hub", "location": "Carlsbad, NM", "capacity_gal": 200000.0},
        {"depot_id": "DEP-ODE", "depot_name": "Odessa Chemical Staging Facility", "location": "Odessa, TX", "capacity_gal": 150000.0},
    ]

    # 2. Accounts (40 accounts)
    operator_names = [
        "Apex Permian Resources", "Permian Deep Basin Operating", "Cimarron E&P", "Lone Star Energy Partners",
        "Midland Basin Petroleum", "Red Dirt Production Co", "Cholla Creek Exploration", "Black Gold Energy",
        "Caprock Natural Resources", "West Texas Crude Operating", "Delaware Basin Energy Corp", "Eagle Claw Exploration",
        "Pecos River Operating LLC", "Iron Mountain Resources", "Roughneck Operating Co", "Guadalupe Peak Energy",
        "Coyote Draw Production", "Mesa Verde Petroleum", "Pioneer Valley E&P", "Silver Spur Operating",
        "Brazos Permian Partners", "High Plains Energy Group", "Caliche Ridge Oil & Gas", "Sundown Operating Co",
        "Vanguard Permian Resources", "Falcon Ridge Energy", "Tall City Exploration", "Dusty Trail Oil Corp",
        "Blue Quail Production", "Centennial Basin Operating", "Ocotillo Energy LLC", "Rio Pecos Resources",
        "Mustang Draw Petroleum", "Wildcat Creek Exploration", "Shale Rock Operating", "Buffalo Wallow Energy",
        "Frontier Basin Petroleum", "Chaparral Energy Partners", "Yucca Flat Production", "Southwestern Permian LLC"
    ]

    accounts: List[Dict[str, Any]] = []
    for i, name in enumerate(operator_names, 1):
        acc_id = f"ACC-{i:03d}"
        credit_limit = float(rng.choice([50000.0, 75000.0, 100000.0, 150000.0, 200000.0, 250000.0]))
        accounts.append({
            "account_id": acc_id,
            "company_name": name,
            "status": "active",
            "credit_limit": credit_limit,
            "payment_terms": str(rng.choice(["Net 30", "Net 60", "Net 15"])),
            "created_at": f"2024-01-{(i % 28) + 1:02d}T08:00:00Z"
        })

    # 3. Sites (120 well sites: 3 per account)
    counties_tx = ["Midland", "Martin", "Reeves", "Loving", "Upton", "Glasscock", "Ward", "Howard", "Andrews", "Culberson"]
    sites: List[Dict[str, Any]] = []
    site_counter = 1
    for acc in accounts:
        comp_prefix = str(acc["company_name"]).split()[0]
        for s in range(3):
            county = rng.choice(counties_tx)
            prefix = county[:3].upper()
            code = f"SITE-{prefix}-{site_counter:03d}"
            sites.append({
                "site_code": code,
                "account_id": acc["account_id"],
                "site_name": f"{comp_prefix} {county} Well Pad #{s + 1}",
                "basin": "Permian",
                "county": county,
                "state": "TX",
                "road_type": rng.choice(["caliche", "paved", "unimproved_dirt"]),
                "is_active": 1
            })
            site_counter += 1

    # 4. Contacts (120 contacts: 1 primary contact per site/account)
    first_names = ["Hank", "Wayne", "Travis", "Dale", "Cody", "Wyatt", "Clayton", "Dustin", "Colt", "Beau", "Austin", "Garrett", "Brett", "Clint", "Jesse", "Luke"]
    last_names = ["Callahan", "Dutton", "Miller", "Vance", "Rawlins", "McBride", "Hawkins", "Hensley", "Garrison", "Collier", "Bledsoe", "Holt"]
    roles = ["Company Man", "Well Site Supervisor", "Production Foreman", "Chemical Coordinator", "Frac Lead"]

    contacts: List[Dict[str, Any]] = []
    contact_counter = 1
    for acc in accounts:
        comp_clean = str(acc["company_name"]).split()[0].lower()
        # Create 2-3 contacts per account
        num_contacts = rng.randint(2, 3)
        for c in range(num_contacts):
            full_name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
            has_hazmat = 1 if rng.random() < 0.65 else 0
            contacts.append({
                "contact_id": f"CON-{contact_counter:03d}",
                "account_id": acc["account_id"],
                "full_name": full_name,
                "phone": f"432-555-{rng.randint(1000, 9999)}",
                "email": f"{full_name.lower().replace(' ', '.')}@{comp_clean}energy.com",
                "role": rng.choice(roles),
                "has_hazmat_cert": has_hazmat,
                "cert_expiration": "2027-12-31" if has_hazmat else None
            })
            contact_counter += 1

    # 5. Products (60 chemicals across classes)
    chemicals = [
        # Class 8 Corrosive - Acids & Bases
        ("CHEM-801", "SKU-HCL-15", "Hydrochloric Acid 15% Inhibited", "Acidizing", "8", "acid", "gal", 4.25, 3000.0),
        ("CHEM-802", "SKU-HCL-28", "Hydrochloric Acid 28% Concentrated", "Acidizing", "8", "acid", "gal", 6.80, 2500.0),
        ("CHEM-803", "SKU-HF-ACID", "Hydrofluoric Mud Acid Blend", "Acidizing", "8", "acid", "gal", 12.50, 1500.0),
        ("CHEM-804", "SKU-ACETIC-50", "Glacial Acetic Acid 50% Buffer", "Buffering", "8", "acid", "gal", 5.10, 3000.0),
        ("CHEM-805", "SKU-SULF-60", "Sulfamic Acid Descaler Liquid", "Scale Removal", "8", "acid", "gal", 7.40, 3000.0),
        ("CHEM-806", "SKU-NAOH-25", "Sodium Hydroxide Caustic Solution 25%", "pH Control", "8", "base", "gal", 3.90, 3000.0),
        ("CHEM-807", "SKU-KOH-45", "Potassium Hydroxide Liquid 45%", "Clay Control", "8", "base", "gal", 5.50, 3000.0),
        ("CHEM-808", "SKU-SCALE-PH", "Phosphonate Scale Inhibitor Acidic", "Scale Control", "8", "acid", "gal", 8.20, 3000.0),
        ("CHEM-809", "SKU-CITRIC-50", "Liquid Citric Acid Iron Chelant", "Iron Control", "8", "acid", "gal", 4.80, 3000.0),
        ("CHEM-810", "SKU-CORR-INH8", "Amine Bisulfite Corrosion Inhibitor", "Corrosion Control", "8", "acid", "gal", 14.20, 2000.0),

        # Class 3 Flammable Liquids - Solvents & Scavengers
        ("CHEM-301", "SKU-MEOH-PUR", "Methanol Technical Grade 99%", "Hydrate Inhibitor", "3", "solvent", "gal", 3.10, 4000.0),
        ("CHEM-302", "SKU-ISOP-SOL", "Isopropanol Solvent Blend 70/30", "Solvent Wash", "3", "solvent", "gal", 4.50, 4000.0),
        ("CHEM-303", "SKU-XYL-FLM", "Xylene Heavy Aromatic Solvent", "Paraffin Solvent", "3", "solvent", "gal", 7.90, 3500.0),
        ("CHEM-304", "SKU-TOL-IND", "Toluene Industrial Solvent", "Wax Solvent", "3", "solvent", "gal", 7.20, 3500.0),
        ("CHEM-305", "SKU-H2S-TRZ3", "Triazine H2S Scavenger High-Flash", "Scavenger", "3", "solvent", "gal", 9.50, 4000.0),
        ("CHEM-306", "SKU-FR-DIESEL", "Polyacrylamide Friction Reducer Oil Carrier", "Frac Additive", "3", "solvent", "gal", 8.80, 4000.0),
        ("CHEM-307", "SKU-PARA-DIS", "Heavy Paraffin Dispersant Flammable", "Paraffin Control", "3", "solvent", "gal", 11.20, 3000.0),
        ("CHEM-308", "SKU-COND-INH", "Gas Condensate Corrosion Inhibitor Class 3", "Corrosion Control", "3", "solvent", "gal", 16.50, 2500.0),
        ("CHEM-309", "SKU-SURF-FLM", "Microemulsion Surfactant Alcohol Base", "Flowback Aid", "3", "solvent", "gal", 18.00, 2500.0),
        ("CHEM-310", "SKU-DEFOAM-FL", "Alcohol-Based Defoaming Agent", "Defoamer", "3", "solvent", "gal", 6.70, 3000.0),

        # Class 9 & Non-Regulated - Biocides, Salts, Friction Reducers, Surfactants, Scavengers
        ("CHEM-901", "SKU-GLUT-25", "Glutaraldehyde Biocide 25% Active", "Biocide", "9", "biocide", "gal", 15.40, 5000.0),
        ("CHEM-902", "SKU-THPS-75", "THPS Biocide 75% Non-Flammable", "Biocide", "9", "biocide", "gal", 22.00, 4000.0),
        ("CHEM-903", "SKU-DBNPA-20", "Fast-Acting DBNPA Biocide Liquid", "Biocide", "9", "biocide", "gal", 19.50, 4000.0),
        ("CHEM-101", "SKU-FR-HVY", "High-Viscosity Friction Reducer Aqueous", "Friction Reducer", "non_regulated", "friction_reducer", "gal", 7.50, 8000.0),
        ("CHEM-102", "SKU-FR-BRINE", "High-Brine Tolerant Friction Reducer", "Friction Reducer", "non_regulated", "friction_reducer", "gal", 8.90, 8000.0),
        ("CHEM-103", "SKU-KCL-SUB", "Choline Chloride 70% Clay Stabilizer", "Clay Control", "non_regulated", "clay_stabilizer", "gal", 4.10, 8000.0),
        ("CHEM-104", "SKU-SCALE-POLY", "Polyacrylate Scale Inhibitor Neutral", "Scale Control", "non_regulated", "scale_inhibitor", "gal", 6.30, 8000.0),
        ("CHEM-105", "SKU-H2S-AQ", "Water-Soluble Non-Flammable H2S Scavenger", "Scavenger", "non_regulated", "scavenger", "gal", 10.50, 8000.0),
        ("CHEM-106", "SKU-O2-SCAV", "Ammonium Bisulfite Oxygen Scavenger", "Scavenger", "non_regulated", "scavenger", "gal", 5.20, 8000.0),
        ("CHEM-107", "SKU-SURF-NEUT", "Non-Ionic Eco-Surfactant Flowback", "Surfactant", "non_regulated", "surfactant", "gal", 12.00, 8000.0),
        ("CHEM-108", "SKU-DEMUL-OIL", "Heavy Crude Demulsifier Emulsion Breaker", "Demulsifier", "non_regulated", "demulsifier", "gal", 14.80, 6000.0),
        ("CHEM-109", "SKU-FOAM-AER", "Foaming Agent for Gas Well Deliquification", "Foamer", "non_regulated", "foamer", "gal", 8.40, 6000.0),
        ("CHEM-110", "SKU-CLAY-TMAC", "Tetramethylammonium Chloride 50%", "Clay Control", "non_regulated", "clay_stabilizer", "gal", 6.90, 8000.0),
        ("CHEM-111", "SKU-BIO-ENZ", "Enzyme Breaker Gel Deactivator", "Breaker", "non_regulated", "breaker", "gal", 9.10, 5000.0),
        ("CHEM-112", "SKU-GEL-GUAR", "Slurried Guar Gum Slurry Concentrate", "Gelling Agent", "non_regulated", "gelling_agent", "gal", 5.75, 8000.0),
        ("CHEM-113", "SKU-XLINK-BOR", "Borate Crosslinker Aqueous Slurry", "Crosslinker", "non_regulated", "crosslinker", "gal", 11.80, 5000.0),
        ("CHEM-114", "SKU-XLINK-ZR", "Zirconium Crosslinker Thermal Stable", "Crosslinker", "non_regulated", "crosslinker", "gal", 24.50, 4000.0),
        ("CHEM-115", "SKU-IRON-RED", "Erythorbic Acid Iron Reducing Agent", "Iron Control", "non_regulated", "iron_control", "gal", 7.20, 6000.0),
        ("CHEM-116", "SKU-BACT-DET", "Bacteriological ATP Detection Reagent", "Testing", "non_regulated", "testing", "gal", 45.00, 500.0),
        ("CHEM-117", "SKU-TRACE-FL", "Water-Tracing Fluorophore Dye Red", "Tracer", "non_regulated", "tracer", "gal", 65.00, 500.0),
        ("CHEM-118", "SKU-TRACE-GRN", "Water-Tracing Fluorophore Dye Green", "Tracer", "non_regulated", "tracer", "gal", 65.00, 500.0),
        ("CHEM-119", "SKU-POW-SALT", "Potassium Formate Brine Weighting Salt", "Brine", "non_regulated", "brine", "gal", 8.50, 8000.0),
        ("CHEM-120", "SKU-CACL-BRN", "Calcium Chloride 38% Heavy Brine", "Completion Brine", "non_regulated", "brine", "gal", 3.20, 8000.0),
        ("CHEM-121", "SKU-POLY-VIS", "Liquid Polymeric Viscosifier", "Workover", "non_regulated", "viscosifier", "gal", 11.20, 6000.0),
        ("CHEM-122", "SKU-LUB-FLUID", "Environmentally Safe Drilling Lubricant", "Lubricant", "non_regulated", "lubricant", "gal", 13.40, 6000.0),
        ("CHEM-123", "SKU-CORR-NEUT", "Neutral Film-Forming Corrosion Amine", "Corrosion Control", "non_regulated", "corrosion_inhibitor", "gal", 15.00, 6000.0),
        ("CHEM-124", "SKU-ASPH-DIS", "Polymeric Asphaltene Inhibitor", "Asphaltene Control", "non_regulated", "asphaltene_inhibitor", "gal", 17.50, 5000.0),
        ("CHEM-125", "SKU-RIG-WASH", "Heavy Duty Alkaline Rig Wash Cleaner", "Cleaning", "non_regulated", "cleaner", "gal", 4.50, 8000.0),
        ("CHEM-126", "SKU-PIPE-COAT", "Tubing Internal Protector Coating", "Maintenance", "non_regulated", "coating", "gal", 28.00, 2000.0),
        ("CHEM-127", "SKU-WELL-STIM", "Mutual Solvent Surfactant Package", "Stimulation", "non_regulated", "solvent", "gal", 16.20, 6000.0),
        ("CHEM-128", "SKU-CORR-SWEET", "CO2 Sweet Gas Corrosion Inhibitor", "Corrosion Control", "non_regulated", "corrosion_inhibitor", "gal", 14.50, 6000.0),
        ("CHEM-129", "SKU-CORR-SOUR", "H2S Sour Well Corrosion Inhibitor", "Corrosion Control", "non_regulated", "corrosion_inhibitor", "gal", 18.20, 5000.0),
        ("CHEM-130", "SKU-PARA-CRY", "Crystal Modifier Paraffin Inhibitor", "Paraffin Control", "non_regulated", "paraffin_inhibitor", "gal", 13.90, 6000.0),
        ("CHEM-131", "SKU-SCALE-SULF", "Barium & Strontium Scale Dissolver", "Scale Removal", "non_regulated", "scale_inhibitor", "gal", 19.80, 5000.0),
        ("CHEM-132", "SKU-WATER-CLAR", "Cationic Water Clarifier Coagulant", "Water Treatment", "non_regulated", "water_treatment", "gal", 8.70, 8000.0),
        ("CHEM-133", "SKU-SULF-SCAV", "Solid-Form Zinc Carbonate Scavenger", "Scavenger", "non_regulated", "scavenger", "gal", 9.80, 5000.0),
        ("CHEM-134", "SKU-BIO-DISP", "Enzymatic Slime & Biofilm Dispersant", "Biocide", "non_regulated", "biocide", "gal", 14.00, 5000.0),
        ("CHEM-135", "SKU-FRAC-OIL", "Mineral Seal Oil Carrier Base", "Base Oil", "non_regulated", "solvent", "gal", 4.80, 8000.0),
        ("CHEM-136", "SKU-MUD-THIN", "Polyacrylic Mud Thinner Deflocculant", "Drilling", "non_regulated", "deflocculant", "gal", 6.50, 8000.0),
        ("CHEM-137", "SKU-SILICA-SCALE", "Specialty Geothermal Silica Inhibitor", "Scale Control", "non_regulated", "scale_inhibitor", "gal", 21.00, 4000.0),
    ]

    products: List[Dict[str, Any]] = []
    for pid, sku, name, cat, hclass, ctype, unit, price, max_qty in chemicals:
        products.append({
            "product_id": pid,
            "sku": sku,
            "name": name,
            "category": cat,
            "hazard_class": hclass,
            "chemical_type": ctype,
            "unit": unit,
            "unit_price": price,
            "max_order_qty": max_qty,
            "is_active": 1
        })

    # 6. Inventory (every product stocked across all 4 depots with realistic variation)
    inventory = []
    inv_id = 1
    for depot in depots:
        for prod in products:
            qty = rng.choice([5000.0, 10000.0, 15000.0, 25000.0, 40000.0])
            inventory.append({
                "inventory_id": inv_id,
                "depot_id": depot["depot_id"],
                "product_id": prod["product_id"],
                "quantity_available": qty,
                "quantity_reserved": 0.0
            })
            inv_id += 1

    # 7. Chemical Incompatibilities
    incompatibility = [
        {"product_id_a": "CHEM-801", "product_id_b": "CHEM-301", "reason": "Severe reaction: Hydrochloric Acid with Methammable solvent."},
        {"product_id_a": "CHEM-801", "product_id_b": "CHEM-806", "reason": "Violent exothermic neutralization: Strong Acid (HCl) with Strong Base (NaOH)."},
        {"product_id_a": "CHEM-802", "product_id_b": "CHEM-301", "reason": "Severe reaction: Concentrated Acid with Flammable solvent."},
        {"product_id_a": "CHEM-802", "product_id_b": "CHEM-806", "reason": "Violent exothermic neutralization: Acid (HCl 28%) with Base (NaOH 25%)."},
        {"product_id_a": "CHEM-803", "product_id_b": "CHEM-302", "reason": "Toxic gas generation: Hydrofluoric acid with Isopropanol."},
        {"product_id_a": "CHEM-806", "product_id_b": "CHEM-301", "reason": "Incompatible caustic with alcohol solvent in shared transport."},
    ]

    # 8. Trucks (16 trucks: 4 per depot)
    trucks = []
    truck_id = 1
    for depot in depots:
        for t in range(4):
            tid = f"TRK-{truck_id:03d}"
            ttype = "Acid Tanker" if t == 0 else ("Flammable Transport" if t == 1 else "Bulk Chemical Transport")
            hclasses = "8" if t == 0 else ("3" if t == 1 else "non_regulated,9")
            trucks.append({
                "truck_id": tid,
                "license_plate": f"TX-{rng.randint(100, 999)}-{rng.choice(['AB', 'CD', 'EF', 'GH'])}",
                "truck_type": ttype,
                "capacity_gal": 5000.0,
                "hazmat_certified_classes": hclasses
            })
            truck_id += 1

    # 9. Delivery Slots (available dates across shifts)
    shifts = ["morning", "afternoon", "night"]
    delivery_slots: List[Dict[str, Any]] = []
    slot_id = 1
    dates = ["2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19", "2026-10-20"]
    for d in dates:
        for truck in trucks[:8]:  # 8 primary active trucks
            for shift in shifts:
                delivery_slots.append({
                    "slot_id": f"SLOT-{slot_id:04d}",
                    "depot_id": "DEP-MID" if "TRK-001" <= str(truck["truck_id"]) <= "TRK-004" else "DEP-PEC",
                    "truck_id": truck["truck_id"],
                    "delivery_date": d,
                    "shift": shift,
                    "status": "open",
                    "order_id": None
                })
                slot_id += 1

    # 10. Historical Orders & Order Lines (~300 orders)
    orders: List[Dict[str, Any]] = []
    order_lines: List[Dict[str, Any]] = []
    credit_ledger: List[Dict[str, Any]] = []
    line_id = 1
    ledger_id = 1

    # Initialize account credit ledger
    for acc in accounts:
        credit_ledger.append({
            "entry_id": ledger_id,
            "account_id": acc["account_id"],
            "entry_type": "credit_grant",
            "amount": float(acc["credit_limit"]),
            "order_id": None,
            "timestamp": "2024-01-01T00:00:00Z",
            "description": "Initial credit facility approved."
        })
        ledger_id += 1

    order_statuses = [
        "delivered", "delivered", "delivered", "delivered", "delivered",
        "in_transit", "loaded", "picked", "approved", "submitted", "cancelled"
    ]

    for o_idx in range(1, 301):
        oid = f"ORD-{o_idx:04d}"
        acc = rng.choice(accounts)
        acc_sites = [s for s in sites if s["account_id"] == acc["account_id"]]
        site = rng.choice(acc_sites) if acc_sites else sites[0]
        status = rng.choice(order_statuses)

        # 1 to 3 items per order
        num_items = rng.randint(1, 3)
        order_prods = rng.sample(products, num_items)
        order_total = 0.0

        for prod in order_prods:
            qty = float(rng.randint(2, 20) * 100)  # 200 to 2000 gal
            subtotal = round(qty * float(prod["unit_price"]), 2)
            order_total += subtotal

            order_lines.append({
                "line_id": line_id,
                "order_id": oid,
                "product_id": prod["product_id"],
                "quantity": qty,
                "unit_price": float(prod["unit_price"]),
                "subtotal": subtotal
            })
            line_id += 1

        cancellation_fee = 350.0 if status == "cancelled" and rng.random() < 0.4 else 0.0

        orders.append({
            "order_id": oid,
            "account_id": acc["account_id"],
            "site_code": site["site_code"],
            "status": status,
            "total_amount": round(order_total, 2),
            "cancellation_fee": cancellation_fee,
            "delivery_slot_id": None,
            "created_at": f"2026-09-{(o_idx % 28) + 1:02d}T10:00:00Z",
            "updated_at": f"2026-09-{(o_idx % 28) + 1:02d}T14:30:00Z",
            "notes": "Historical routine delivery."
        })

        if status in ["delivered", "in_transit", "loaded", "picked"]:
            credit_ledger.append({
                "entry_id": ledger_id,
                "account_id": acc["account_id"],
                "entry_type": "invoice",
                "amount": -round(order_total, 2),
                "order_id": oid,
                "timestamp": f"2026-09-{(o_idx % 28) + 1:02d}T15:00:00Z",
                "description": f"Invoice commitment for order {oid}"
            })
            ledger_id += 1

    # 11. Site Access (road conditions per site)
    site_access: List[Dict[str, Any]] = []
    for site in sites:
        road_status = rng.choices(["open", "restricted_wet", "closed"], weights=[0.80, 0.15, 0.05])[0]
        site_access.append({
            "site_code": site["site_code"],
            "road_status": road_status,
            "updated_at": "2026-10-14T06:00:00Z"
        })

    # 12. Escalations (empty initially)
    escalations: List[Dict[str, Any]] = []

    dataset: Dict[str, List[Dict[str, Any]]] = {
        "depots": depots,
        "accounts": accounts,
        "sites": sites,
        "contacts": contacts,
        "products": products,
        "inventory": inventory,
        "incompatibility": incompatibility,
        "trucks": trucks,
        "delivery_slots": delivery_slots,
        "orders": orders,
        "order_lines": order_lines,
        "credit_ledger": credit_ledger,
        "site_access": site_access,
        "escalations": escalations,
    }

    # Write JSON files to output directory
    for key, data in dataset.items():
        with open(out_path / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return dataset


if __name__ == "__main__":
    generate_seed_dataset()
    print("Seed dataset generated successfully.")
