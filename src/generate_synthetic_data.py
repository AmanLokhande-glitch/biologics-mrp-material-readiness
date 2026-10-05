"""
generate_synthetic_data.py

Build step 1: create the master data for the fictional company
"Northstar Biologics Manufacturing".

This script writes two CSV files:
    data/raw/supplier_master.csv  -> 14 fictional suppliers
    data/raw/material_master.csv  -> 60 materials (22 raw/process,
                                     23 single-use, 15 packaging)

The columns follow section 6 of the project plan. They mirror SAP-style
material planning parameters (procurement type, planned delivery time,
goods-receipt processing time, safety stock, minimum lot size, rounding value,
lot-sizing procedure). This is a synthetic model of those concepts, not SAP.

All names and numbers are synthetic. A fixed random seed is used, so the
output is exactly the same every time the script runs.

Run from the project folder:
    python src/generate_synthetic_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

# Fixed seed -> the "random" numbers are identical on every run.
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Output folder: <project root>/data/raw
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

# Final column order for each file (section 6 of the plan, then extras).
SUPPLIER_COLUMNS = [
    "supplier_id", "supplier_name", "supplier_region", "standard_lead_time",
    "sole_source_flag", "risk_rating", "historical_otd_pct",
    "quality_score",  # extra column
]
MATERIAL_COLUMNS = [
    "material_id", "material_description", "material_category", "base_uom",
    "supplier_id", "procurement_type", "planned_delivery_days",
    "gr_processing_days", "safety_stock_qty", "minimum_order_qty",
    "rounding_value", "lot_size_rule", "minimum_remaining_shelf_life_days",
    "criticality", "sole_source_flag", "unit_cost",
    "shelf_life_days", "storage_condition",  # extra columns
]

# Allowed values for the coded fields.
ALLOWED_REGIONS = {"NORTH_AMERICA", "EUROPE", "ASIA_PACIFIC"}
ALLOWED_RISK_RATINGS = {"HIGH", "MEDIUM", "LOW"}
ALLOWED_FLAGS = {"Y", "N"}
ALLOWED_CRITICALITY = {"HIGH", "MEDIUM"}

# Procurement type (SAP-style code). Every material here is bought from a
# supplier, so all use "F" = external procurement. ("E" would mean made in-house.)
ALLOWED_PROCUREMENT_TYPES = {"F"}

# Lot-size rule (SAP-style codes) = how MRP decides how much to order:
#   EX = lot-for-lot: order exactly the quantity that is short
#   FX = fixed lot: always order the same fixed quantity (here: the MOQ)
#   MB = monthly lot: combine about 4 weeks of requirements into one order
ALLOWED_LOT_SIZE_RULES = {"EX", "FX", "MB"}

# Material planned delivery time may differ from the supplier's standard
# lead time by at most this many days.
MAX_LEAD_TIME_GAP_DAYS = 14


# ---------------------------------------------------------------------------
# 2. Supplier master data
# ---------------------------------------------------------------------------

# Each row: (supplier_id, supplier_name, supplier_category, supplier_region,
#            sole_source_flag)
# Every name here is invented for this project.
# supplier_category is only used inside this script (to pick a lead-time range
# and to check that each material goes to the right kind of supplier). It is
# not written to the CSV.
# sole_source_flag = "Y" means this supplier is the only qualified source for
# at least one of our materials (for example a specialised resin or sensor).
SUPPLIERS = [
    # Raw / process material suppliers
    ("SUP-001", "Harborline Bioprocess Media", "RAW_PROCESS", "NORTH_AMERICA", "Y"),
    ("SUP-002", "Cedarvale Fine Chemicals", "RAW_PROCESS", "EUROPE", "N"),
    ("SUP-003", "Northgate Resin Technologies", "RAW_PROCESS", "EUROPE", "Y"),
    ("SUP-004", "Willowmere Biochem", "RAW_PROCESS", "EUROPE", "N"),
    ("SUP-005", "Redfern Excipient Supply", "RAW_PROCESS", "NORTH_AMERICA", "N"),
    # Single-use component suppliers
    ("SUP-006", "Ironwood Single-Use Systems", "SINGLE_USE", "NORTH_AMERICA", "Y"),
    ("SUP-007", "Brightwater Filtration Co.", "SINGLE_USE", "EUROPE", "Y"),
    ("SUP-008", "Kestrel Fluid Path Assemblies", "SINGLE_USE", "EUROPE", "N"),
    ("SUP-009", "Lumenbridge Sensor Works", "SINGLE_USE", "ASIA_PACIFIC", "Y"),
    ("SUP-010", "Granite Peak Polymers", "SINGLE_USE", "ASIA_PACIFIC", "N"),
    # Packaging suppliers
    ("SUP-011", "Silverbirch Glassworks", "PACKAGING", "EUROPE", "Y"),
    ("SUP-012", "Marlowe Closure Components", "PACKAGING", "NORTH_AMERICA", "N"),
    ("SUP-013", "Tidewell Label & Print", "PACKAGING", "NORTH_AMERICA", "N"),
    ("SUP-014", "Oakhaven Cartons & Cold Chain", "PACKAGING", "NORTH_AMERICA", "N"),
]

# Quick lookup: supplier_id -> supplier_category (used in the checks).
SUPPLIER_CATEGORY = {row[0]: row[2] for row in SUPPLIERS}


def assign_risk_rating(otd_pct, quality_score):
    """
    Simple rule for supplier risk, based only on supplier performance
    (not on how critical our materials are):
      HIGH   -> on-time delivery below 85% OR quality score below 84
      LOW    -> on-time delivery 92% or better AND quality score 90 or better
      MEDIUM -> everything in between
    """
    if otd_pct < 85 or quality_score < 84:
        return "HIGH"
    if otd_pct >= 92 and quality_score >= 90:
        return "LOW"
    return "MEDIUM"


def build_supplier_master():
    """Turn the supplier list into a table and add performance fields."""
    df = pd.DataFrame(
        SUPPLIERS,
        columns=["supplier_id", "supplier_name", "supplier_category",
                 "supplier_region", "sole_source_flag"],
    )
    n = len(df)

    # Standard lead time in days (order date -> delivery), rounded to whole
    # weeks. The range depends on what the supplier makes.
    lead_times = []
    for category in df["supplier_category"]:
        low, high = CATEGORY_SETTINGS[category]["lead_time"]
        days = int(rng.integers(low, high, endpoint=True))
        lead_times.append(int(round(days / 7) * 7))
    df["standard_lead_time"] = lead_times

    # Historical on-time delivery % (share of PO lines delivered on time).
    df["historical_otd_pct"] = rng.uniform(78, 98, size=n).round(1)

    # Quality score out of 100 (based on incoming inspection results).
    df["quality_score"] = rng.integers(80, 100, size=n, endpoint=True)

    # Risk rating comes from the two performance numbers above.
    df["risk_rating"] = [
        assign_risk_rating(otd, q)
        for otd, q in zip(df["historical_otd_pct"], df["quality_score"])
    ]

    # Drop the internal-only category column and put columns in plan order.
    return df[SUPPLIER_COLUMNS]


# ---------------------------------------------------------------------------
# 3. Material master data
# ---------------------------------------------------------------------------

# Each row: (description, base_uom, storage_condition, unit_cost,
#            supplier_id, minimum_order_qty, rounding_value)
# unit_cost is in USD per base unit of measure.
# rounding_value is the pack size: order quantities must be a multiple of it
# (for example, a 25 kg bag means you order 25, 50, 75 ... kg).
# Descriptions are generic and fictional (no brand names).

RAW_PROCESS_MATERIALS = [
    ("Chemically Defined CHO Basal Medium, Powder, 10 kg", "KG", "2-8 C", 410.00, "SUP-001", 100, 10),
    ("CHO Feed Supplement A, Liquid, 20 L", "L", "2-8 C", 95.00, "SUP-001", 200, 20),
    ("CHO Feed Supplement B, Liquid, 20 L", "L", "2-8 C", 120.00, "SUP-001", 200, 20),
    ("L-Glutamine Solution 200 mM, 1 L", "L", "2-8 C", 38.00, "SUP-001", 50, 10),
    ("Poloxamer 188, Cell Culture Grade, 1 kg", "KG", "15-25 C", 145.00, "SUP-004", 20, 5),
    ("Antifoam Emulsion, Low-Silicone, 1 kg", "KG", "15-25 C", 88.00, "SUP-004", 10, 5),
    ("Protein A Affinity Resin, 5 L", "L", "2-8 C", 9800.00, "SUP-003", 20, 5),
    ("Cation Exchange Resin, 10 L", "L", "2-8 C", 1650.00, "SUP-003", 40, 10),
    ("Anion Exchange Resin, 10 L", "L", "2-8 C", 1420.00, "SUP-003", 40, 10),
    ("Sodium Chloride, Multi-Compendial, 25 kg", "KG", "15-25 C", 4.50, "SUP-002", 500, 25),
    ("Tris Base, USP/EP, 10 kg", "KG", "15-25 C", 32.00, "SUP-002", 200, 10),
    ("Sodium Acetate Trihydrate, USP, 25 kg", "KG", "15-25 C", 11.00, "SUP-002", 250, 25),
    ("Glacial Acetic Acid, Multi-Compendial, 2.5 L", "L", "15-25 C", 26.00, "SUP-002", 100, 10),
    ("Sodium Hydroxide 50% Solution, 20 L", "L", "15-25 C", 7.50, "SUP-002", 400, 20),
    ("Sodium Citrate Dihydrate, USP, 25 kg", "KG", "15-25 C", 9.00, "SUP-002", 250, 25),
    ("Sodium Phosphate Monobasic, USP, 25 kg", "KG", "15-25 C", 14.00, "SUP-002", 250, 25),
    ("L-Histidine, Low-Endotoxin, 5 kg", "KG", "15-25 C", 210.00, "SUP-005", 25, 5),
    ("L-Arginine Hydrochloride, Low-Endotoxin, 5 kg", "KG", "15-25 C", 165.00, "SUP-005", 25, 5),
    ("Sucrose, Low-Endotoxin, 25 kg", "KG", "15-25 C", 18.00, "SUP-005", 250, 25),
    ("Polysorbate 80, High-Purity Grade, 1 kg", "KG", "2-8 C", 520.00, "SUP-005", 5, 1),
    ("Glycine, Multi-Compendial, 25 kg", "KG", "15-25 C", 12.00, "SUP-005", 250, 25),
    ("Hydrochloric Acid 1 N Solution, 10 L", "L", "15-25 C", 9.50, "SUP-004", 200, 10),
]

SINGLE_USE_COMPONENTS = [
    ("Single-Use Bioreactor Bag, 2000 L", "EA", "15-25 C", 14500.00, "SUP-006", 4, 1),
    ("Single-Use Bioreactor Bag, 500 L", "EA", "15-25 C", 6200.00, "SUP-006", 4, 1),
    ("Seed Bioreactor Bag, 50 L", "EA", "15-25 C", 1850.00, "SUP-006", 6, 2),
    ("Single-Use Mixer Bag, 1000 L", "EA", "15-25 C", 3900.00, "SUP-006", 6, 2),
    ("Media Storage Bag, 3D, 200 L", "EA", "15-25 C", 780.00, "SUP-010", 12, 4),
    ("Buffer Storage Bag, 3D, 500 L", "EA", "15-25 C", 1150.00, "SUP-010", 10, 2),
    ("Harvest Collection Bag, 2D, 50 L", "EA", "15-25 C", 310.00, "SUP-010", 20, 10),
    ("Freeze-Thaw Bag, 6 L", "EA", "15-25 C", 265.00, "SUP-010", 24, 12),
    ("Sterilizing-Grade Filter Capsule, 0.2 um, 10 in", "EA", "15-25 C", 690.00, "SUP-007", 24, 6),
    ("Sterilizing-Grade Filter Capsule, 0.2 um, 30 in", "EA", "15-25 C", 1540.00, "SUP-007", 12, 3),
    ("Depth Filter Capsule, Primary Clarification, 1.1 m2", "EA", "15-25 C", 1280.00, "SUP-007", 12, 3),
    ("Virus Retentive Filter, 20 nm, 1 m2", "EA", "15-25 C", 5600.00, "SUP-007", 6, 1),
    ("TFF Cassette, 30 kDa PES, 2.5 m2", "EA", "2-8 C", 4100.00, "SUP-007", 6, 1),
    ("Aseptic Connector, 1/2 in Hose Barb", "EA", "15-25 C", 145.00, "SUP-008", 100, 25),
    ("Platinum-Cured Silicone Tubing Assembly, 3/8 in, 3 m", "EA", "15-25 C", 210.00, "SUP-008", 50, 10),
    ("Thermoplastic Weldable Tubing Assembly, 1/2 in, 2 m", "EA", "15-25 C", 175.00, "SUP-008", 50, 10),
    ("Sampling Manifold, 6-Port, Gamma Irradiated", "EA", "15-25 C", 330.00, "SUP-008", 40, 10),
    ("PETG Media Bottle Assembly, 1 L", "EA", "15-25 C", 48.00, "SUP-008", 200, 20),
    ("Transfer Line Assembly, 1 in, with Sterile Connectors", "EA", "15-25 C", 460.00, "SUP-008", 30, 6),
    ("Single-Use pH Sensor", "EA", "15-25 C", 395.00, "SUP-009", 24, 6),
    ("Single-Use Dissolved Oxygen Sensor", "EA", "15-25 C", 420.00, "SUP-009", 24, 6),
    ("Single-Use Pressure Sensor", "EA", "15-25 C", 185.00, "SUP-009", 40, 10),
    ("Single-Use Conductivity Sensor", "EA", "15-25 C", 240.00, "SUP-009", 30, 6),
]

PACKAGING_ITEMS = [
    ("Type I Borosilicate Glass Vial, 10 mL, 20 mm Finish", "EA", "15-25 C", 0.42, "SUP-011", 50000, 5000),
    ("Type I Borosilicate Glass Vial, 2R", "EA", "15-25 C", 0.31, "SUP-011", 50000, 5000),
    ("Prefilled Syringe Barrel, 1 mL Long, Staked Needle", "EA", "15-25 C", 0.95, "SUP-011", 40000, 2000),
    ("Chlorobutyl Lyophilization Stopper, 20 mm, Ready-to-Use", "EA", "15-25 C", 0.18, "SUP-012", 50000, 5000),
    ("Bromobutyl Serum Stopper, 13 mm, Ready-to-Use", "EA", "15-25 C", 0.14, "SUP-012", 50000, 5000),
    ("Syringe Plunger Stopper, 1 mL, Coated", "EA", "15-25 C", 0.22, "SUP-012", 40000, 2000),
    ("Aluminum Flip-Off Seal, 20 mm, Blue", "EA", "15-25 C", 0.06, "SUP-012", 50000, 5000),
    ("Rigid Needle Shield, 1 mL Syringe", "EA", "15-25 C", 0.12, "SUP-012", 40000, 2000),
    ("Printed Vial Label, Product Strength A", "EA", "15-25 C", 0.04, "SUP-013", 100000, 5000),
    ("Printed Vial Label, Product Strength B", "EA", "15-25 C", 0.04, "SUP-013", 100000, 5000),
    ("Tamper-Evident Carton Seal Label", "EA", "15-25 C", 0.02, "SUP-013", 100000, 10000),
    ("Patient Information Leaflet, Folded", "EA", "15-25 C", 0.03, "SUP-013", 100000, 10000),
    ("Folding Carton, 1-Vial Pack", "EA", "15-25 C", 0.21, "SUP-014", 25000, 1000),
    ("Thermoformed Vial Tray, 10-Count", "EA", "15-25 C", 0.35, "SUP-014", 20000, 1000),
    ("Insulated Cold-Chain Shipper, 2-8 C, 48 h", "EA", "15-25 C", 38.00, "SUP-014", 500, 20),
]

# Category settings:
#   prefix          -> material ID prefix (RM / SU / PK)
#   lead_time       -> supplier standard lead-time range (days)
#   gr_processing   -> goods-receipt processing time range (days). Raw
#                      materials need lab testing, so they take longest.
#   shelf_life      -> total shelf-life range (days)
#   lot_size_odds   -> chance of each lot-size rule (EX, FX, MB)
#   p_single_source -> chance a material is sole-source, IF its supplier is
#                      flagged as a sole-source supplier
#   p_high_crit     -> chance a material is HIGH criticality
CATEGORY_SETTINGS = {
    "RAW_PROCESS": {
        "prefix": "RM", "items": RAW_PROCESS_MATERIALS,
        "lead_time": (21, 84), "gr_processing": (7, 14),
        "shelf_life": (365, 1095),
        "lot_size_odds": {"EX": 0.6, "FX": 0.2, "MB": 0.2},
        "p_single_source": 0.6, "p_high_crit": 0.45,
    },
    "SINGLE_USE": {
        "prefix": "SU", "items": SINGLE_USE_COMPONENTS,
        "lead_time": (42, 140), "gr_processing": (2, 5),
        "shelf_life": (730, 1825),
        "lot_size_odds": {"EX": 0.7, "FX": 0.1, "MB": 0.2},
        "p_single_source": 0.6, "p_high_crit": 0.60,
    },
    "PACKAGING": {
        "prefix": "PK", "items": PACKAGING_ITEMS,
        "lead_time": (28, 98), "gr_processing": (3, 7),
        "shelf_life": (1095, 1825),
        "lot_size_odds": {"EX": 0.2, "FX": 0.6, "MB": 0.2},
        "p_single_source": 0.6, "p_high_crit": 0.30,
    },
}


def build_material_master(suppliers):
    """Create one row per material, category by category."""
    # Lookups from the supplier table built in step 2.
    supplier_lead_time = dict(zip(suppliers["supplier_id"], suppliers["standard_lead_time"]))
    supplier_sole_source = dict(zip(suppliers["supplier_id"], suppliers["sole_source_flag"]))

    rows = []

    for category, cfg in CATEGORY_SETTINGS.items():
        lot_rules = list(cfg["lot_size_odds"].keys())
        lot_odds = list(cfg["lot_size_odds"].values())

        for i, item in enumerate(cfg["items"], start=1):
            description, uom, storage, unit_cost, supplier_id, moq, rounding = item

            # Planned delivery time = supplier's standard lead time, adjusted
            # by -1 to +2 weeks for this specific material.
            adjustment = int(rng.choice([-7, 0, 0, 7, 14]))
            planned_delivery_days = int(supplier_lead_time[supplier_id]) + adjustment

            # Goods-receipt processing time: days after delivery before the
            # material is ready to use (inspection / testing).
            gr_processing_days = int(rng.integers(*cfg["gr_processing"], endpoint=True))

            # Safety stock = 30% to 100% of the minimum order quantity,
            # rounded to a whole number.
            safety_stock_qty = int(round(moq * rng.uniform(0.3, 1.0)))

            # Lot-size rule picked using the category odds.
            lot_size_rule = str(rng.choice(lot_rules, p=lot_odds))

            # Total shelf life, rounded to 30-day months.
            shelf_life_days = int(rng.integers(*cfg["shelf_life"], endpoint=True))
            shelf_life_days = int(round(shelf_life_days / 30) * 30)

            # Batches must have at least ~1/3 of their shelf life left to be usable.
            min_remaining_shelf_life_days = int(round(shelf_life_days / 3 / 30) * 30)

            criticality = "HIGH" if rng.random() < cfg["p_high_crit"] else "MEDIUM"

            # A material can only be sole-source if its supplier is a
            # sole-source supplier.
            if supplier_sole_source[supplier_id] == "Y" and rng.random() < cfg["p_single_source"]:
                sole_source_flag = "Y"
            else:
                sole_source_flag = "N"

            rows.append({
                "material_id": f"{cfg['prefix']}-{i:03d}",
                "material_description": description,
                "material_category": category,
                "base_uom": uom,
                "supplier_id": supplier_id,
                "procurement_type": "F",
                "planned_delivery_days": planned_delivery_days,
                "gr_processing_days": gr_processing_days,
                "safety_stock_qty": safety_stock_qty,
                "minimum_order_qty": moq,
                "rounding_value": rounding,
                "lot_size_rule": lot_size_rule,
                "minimum_remaining_shelf_life_days": min_remaining_shelf_life_days,
                "criticality": criticality,
                "sole_source_flag": sole_source_flag,
                "unit_cost": unit_cost,
                "shelf_life_days": shelf_life_days,
                "storage_condition": storage,
            })

    return pd.DataFrame(rows)[MATERIAL_COLUMNS]


# ---------------------------------------------------------------------------
# 4. Checks: make sure the data is valid before saving
# ---------------------------------------------------------------------------

def validate(suppliers, materials):
    """Stop with an error if the master data breaks a basic rule."""

    # --- Supplier checks ---------------------------------------------------
    assert len(suppliers) == 14, "Expected 14 suppliers"
    assert suppliers["supplier_id"].is_unique, "Duplicate supplier_id"
    assert list(suppliers.columns) == SUPPLIER_COLUMNS, "Supplier columns do not match the plan"
    assert not suppliers.isna().any().any(), "Missing values in supplier_master"

    assert set(suppliers["supplier_region"]) <= ALLOWED_REGIONS, "Invalid supplier_region"
    assert (suppliers["standard_lead_time"] > 0).all(), "standard_lead_time must be > 0"
    assert set(suppliers["sole_source_flag"]) <= ALLOWED_FLAGS, "sole_source_flag must be Y or N"
    assert set(suppliers["risk_rating"]) <= ALLOWED_RISK_RATINGS, "risk_rating must be HIGH, MEDIUM or LOW"
    assert suppliers["historical_otd_pct"].between(0, 100).all(), "historical_otd_pct must be 0-100"
    assert suppliers["quality_score"].between(0, 100).all(), "quality_score must be 0-100"

    # --- Material checks ---------------------------------------------------
    assert len(materials) == 60, "Expected 60 materials"
    assert materials["material_id"].is_unique, "Duplicate material_id"
    assert list(materials.columns) == MATERIAL_COLUMNS, "Material columns do not match the plan"
    assert not materials.isna().any().any(), "Missing values in material_master"

    expected_counts = {"RAW_PROCESS": 22, "SINGLE_USE": 23, "PACKAGING": 15}
    actual_counts = materials["material_category"].value_counts().to_dict()
    assert actual_counts == expected_counts, f"Wrong category counts: {actual_counts}"

    # Every material must point to a supplier that exists...
    unknown = set(materials["supplier_id"]) - set(suppliers["supplier_id"])
    assert not unknown, f"Unknown supplier_id(s): {unknown}"

    # ...and that supplier must serve the same category as the material.
    supplier_cat = materials["supplier_id"].map(SUPPLIER_CATEGORY)
    mismatch = materials[materials["material_category"] != supplier_cat]
    assert mismatch.empty, f"Category mismatch for: {list(mismatch['material_id'])}"

    # Planning parameters must make sense.
    assert set(materials["procurement_type"]) <= ALLOWED_PROCUREMENT_TYPES, "Invalid procurement_type"
    assert (materials["planned_delivery_days"] > 0).all(), "planned_delivery_days must be > 0"
    assert (materials["gr_processing_days"] > 0).all(), "gr_processing_days must be > 0"
    assert (materials["safety_stock_qty"] >= 0).all(), "safety_stock_qty must be >= 0"
    assert (materials["minimum_order_qty"] > 0).all(), "minimum_order_qty must be > 0"
    assert (materials["rounding_value"] > 0).all(), "rounding_value must be > 0"
    assert set(materials["lot_size_rule"]) <= ALLOWED_LOT_SIZE_RULES, "Invalid lot_size_rule"
    assert (materials["unit_cost"] > 0).all(), "unit_cost must be > 0"
    assert set(materials["criticality"]) <= ALLOWED_CRITICALITY, "Invalid criticality"

    # The MOQ must be a whole number of packs (a multiple of the rounding value).
    bad_rounding = materials[materials["minimum_order_qty"] % materials["rounding_value"] != 0]
    assert bad_rounding.empty, f"MOQ not a multiple of rounding_value: {list(bad_rounding['material_id'])}"

    # Required remaining shelf life must be positive and shorter than total shelf life.
    min_life = materials["minimum_remaining_shelf_life_days"]
    assert (min_life > 0).all(), "minimum_remaining_shelf_life_days must be > 0"
    assert (min_life < materials["shelf_life_days"]).all(), \
        "minimum_remaining_shelf_life_days must be less than shelf_life_days"

    # Material planned delivery time should be close to the supplier's
    # standard lead time.
    supplier_lead = materials["supplier_id"].map(
        dict(zip(suppliers["supplier_id"], suppliers["standard_lead_time"]))
    )
    gap = (materials["planned_delivery_days"] - supplier_lead).abs()
    far_off = materials[gap > MAX_LEAD_TIME_GAP_DAYS]
    assert far_off.empty, f"Planned delivery time far from supplier lead time: {list(far_off['material_id'])}"

    # Sole-source flags must agree between the two files:
    #   - a sole-source material must come from a sole-source supplier
    #   - a sole-source supplier must be the sole source for at least one material
    assert set(materials["sole_source_flag"]) <= ALLOWED_FLAGS, "sole_source_flag must be Y or N"
    supplier_flag = materials["supplier_id"].map(
        dict(zip(suppliers["supplier_id"], suppliers["sole_source_flag"]))
    )
    wrong = materials[(materials["sole_source_flag"] == "Y") & (supplier_flag != "Y")]
    assert wrong.empty, f"Sole-source material from a non-sole-source supplier: {list(wrong['material_id'])}"

    sole_suppliers = set(suppliers.loc[suppliers["sole_source_flag"] == "Y", "supplier_id"])
    suppliers_with_sole_material = set(materials.loc[materials["sole_source_flag"] == "Y", "supplier_id"])
    missing = sole_suppliers - suppliers_with_sole_material
    assert not missing, f"Sole-source supplier(s) with no sole-source material: {missing}"


# ---------------------------------------------------------------------------
# 5. Main: build, check, save
# ---------------------------------------------------------------------------

def main():
    suppliers = build_supplier_master()
    materials = build_material_master(suppliers)
    validate(suppliers, materials)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suppliers.to_csv(OUTPUT_DIR / "supplier_master.csv", index=False)
    materials.to_csv(OUTPUT_DIR / "material_master.csv", index=False)

    print(f"Wrote {len(suppliers)} suppliers -> {OUTPUT_DIR / 'supplier_master.csv'}")
    print(f"Wrote {len(materials)} materials -> {OUTPUT_DIR / 'material_master.csv'}")
    print("All validation checks passed.")


if __name__ == "__main__":
    main()
