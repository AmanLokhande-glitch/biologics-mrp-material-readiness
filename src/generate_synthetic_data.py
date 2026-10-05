"""
generate_synthetic_data.py

Build steps 1-3: create the master data, the weekly manufacturing
requirements, and the opening batch inventory for the fictional company
"Northstar Biologics Manufacturing".

This script writes four CSV files:
    data/raw/supplier_master.csv     -> 14 fictional suppliers
    data/raw/material_master.csv     -> 60 materials (22 raw/process,
                                        23 single-use, 15 packaging)
    data/raw/weekly_requirements.csv -> 60 materials x 26 weeks = 1,560 rows
                                        of gross requirements, driven by a
                                        campaign production schedule
    data/raw/inventory_batches.csv   -> 130-180 stock batches on hand at the
                                        start of the plan, each with a
                                        quality status (stock type)

The columns follow section 6 of the project plan. They mirror SAP-style
material planning parameters (procurement type, planned delivery time,
goods-receipt processing time, safety stock, minimum lot size, rounding value,
lot-sizing procedure). This is a synthetic model of those concepts, not SAP.

All names and numbers are synthetic. A fixed random seed is used, so the
output is exactly the same every time the script runs.

Run from the project folder:
    python src/generate_synthetic_data.py
"""

import math
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
REQUIREMENT_COLUMNS = [
    "material_id", "week_start", "gross_requirement_qty", "requirement_type",
    "production_program", "priority",
]
INVENTORY_COLUMNS = [
    "material_id", "batch_id", "stock_type", "quantity", "goods_receipt_date",
    "manufacture_date", "expiration_date", "expected_quality_release_date",
    "storage_location",
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

# Requirement types:
#   PRODUCTION       = demand from firm production inside the firm horizon
#   PLANNED_CAMPAIGN = demand from planned campaigns further out (may change)
#   SAFETY_STOCK     = no production demand this week (quantity 0); the only
#                      planning need is to keep safety stock on hand
ALLOWED_REQUIREMENT_TYPES = {"PRODUCTION", "SAFETY_STOCK", "PLANNED_CAMPAIGN"}
ALLOWED_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}

# Stock types (SAP-style quality status of a batch):
#   UNRESTRICTED       = released by Quality, free to use
#   QUALITY_INSPECTION = received but still being tested; not usable until
#                        its expected_quality_release_date
#   BLOCKED            = on hold (e.g. failed inspection, damaged); never usable
ALLOWED_STOCK_TYPES = {"UNRESTRICTED", "QUALITY_INSPECTION", "BLOCKED"}


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
# 4. Weekly manufacturing requirements (build step 2)
# ---------------------------------------------------------------------------

# Planning horizon: 26 weekly buckets. Each week starts on a Monday.
PLANNING_START = pd.Timestamp("2026-01-05")  # a Monday
N_WEEKS = 26

# Weeks 1-8 are the "firm" horizon: that production is fixed, so its demand
# is PRODUCTION. Weeks 9-26 are still a plan, so their demand is
# PLANNED_CAMPAIGN.
FIRM_HORIZON_WEEKS = 8

# Four fictional production programs (neutral codes, no product names).
#   stage       -> COMMERCIAL programs get higher priority than CLINICAL ones
#   scale_l     -> bioreactor size in litres
#   fill_units  -> units filled in a normal fill/finish week (0 = this program
#                  is drug substance only and is not filled on site)
PROGRAMS = {
    "PROG-01": {"stage": "COMMERCIAL", "scale_l": 2000, "fill_units": 24000},  # lyophilised vial, 10 mL
    "PROG-02": {"stage": "COMMERCIAL", "scale_l": 2000, "fill_units": 30000},  # prefilled syringe, 1 mL
    "PROG-03": {"stage": "CLINICAL",   "scale_l": 500,  "fill_units": 18000},  # liquid vial, 2R
    "PROG-04": {"stage": "CLINICAL",   "scale_l": 500,  "fill_units": 0},      # drug substance only
}

# Campaign calendar: (program, first week, last week). Week 1 = Jan 5.
# Biologics plants make one product at a time in long "campaigns", with a
# cleaning / changeover week between products. So:
#   - the drug substance (DS) suite runs only one program in any week
#   - the fill/finish line also runs only one program in any week
# Week 1 is the start-up week after the year-end shutdown (no production).
DS_CAMPAIGNS = [
    ("PROG-01", 2, 6),
    ("PROG-03", 8, 11),    # week 7 = changeover
    ("PROG-02", 13, 17),   # week 12 = changeover
    ("PROG-04", 19, 21),   # week 18 = changeover
    ("PROG-01", 23, 26),   # week 22 = changeover
]
# Filling happens about 3-4 weeks after the drug substance is made (time for
# QC testing and release). The first PROG-02 fill uses DS made in late 2025.
FILL_CAMPAIGNS = [
    ("PROG-02", 3, 4),
    ("PROG-01", 9, 11),
    ("PROG-03", 15, 16),
    ("PROG-02", 21, 23),
]

ALL_PROGRAMS = ["PROG-01", "PROG-02", "PROG-03", "PROG-04"]
LARGE_SCALE = ["PROG-01", "PROG-02"]          # 2000 L programs
SMALL_SCALE = ["PROG-03", "PROG-04"]          # 500 L programs
FILLED_ON_SITE = ["PROG-01", "PROG-02", "PROG-03"]
VIAL_PROGRAMS = ["PROG-01", "PROG-03"]

# Simple bill of materials: material_id -> (programs that use it, qty, basis)
# basis says what the quantity is "per":
#   PER_BATCH    -> per bioreactor batch, used every DS week
#   PER_CAMPAIGN -> once, in the first week of a DS campaign (chromatography
#                   resin is packed into a fresh column per campaign and then
#                   re-used for every batch in that campaign)
#   PER_UNIT     -> per unit filled on the fill/finish line (packaging)
# Raw/process quantities are written for a 2000 L batch and are scaled down
# for 500 L programs. Single-use parts and packaging are simple counts.
REQUIREMENT_BOM = {
    # --- Raw / process materials (per 2000 L batch) ---
    "RM-001": (ALL_PROGRAMS, 40, "PER_BATCH"),        # basal medium
    "RM-002": (ALL_PROGRAMS, 120, "PER_BATCH"),       # feed A
    "RM-003": (LARGE_SCALE, 80, "PER_BATCH"),         # feed B (high-titer processes)
    "RM-004": (SMALL_SCALE, 40, "PER_BATCH"),         # glutamine (non-GS cell lines)
    "RM-005": (ALL_PROGRAMS, 2, "PER_BATCH"),         # poloxamer
    "RM-006": (ALL_PROGRAMS, 1, "PER_BATCH"),         # antifoam
    "RM-007": (ALL_PROGRAMS, 30, "PER_CAMPAIGN"),     # Protein A resin
    "RM-008": (["PROG-01", "PROG-02", "PROG-03"], 40, "PER_CAMPAIGN"),  # cation exchange resin
    "RM-009": (ALL_PROGRAMS, 30, "PER_CAMPAIGN"),     # anion exchange resin
    "RM-010": (ALL_PROGRAMS, 150, "PER_BATCH"),       # sodium chloride
    "RM-011": (ALL_PROGRAMS, 60, "PER_BATCH"),        # Tris
    "RM-012": (ALL_PROGRAMS, 50, "PER_BATCH"),        # sodium acetate
    "RM-013": (ALL_PROGRAMS, 20, "PER_BATCH"),        # acetic acid
    "RM-014": (ALL_PROGRAMS, 120, "PER_BATCH"),       # sodium hydroxide (cleaning)
    "RM-015": (["PROG-02", "PROG-04"], 40, "PER_BATCH"),  # sodium citrate
    "RM-016": (["PROG-01", "PROG-03"], 50, "PER_BATCH"),  # sodium phosphate
    "RM-017": (["PROG-01", "PROG-02", "PROG-03"], 6, "PER_BATCH"),  # histidine (formulation)
    "RM-018": (["PROG-02"], 8, "PER_BATCH"),          # arginine (formulation)
    "RM-019": (VIAL_PROGRAMS, 60, "PER_BATCH"),       # sucrose (formulation)
    "RM-020": (ALL_PROGRAMS, 1, "PER_BATCH"),         # polysorbate 80 (formulation)
    "RM-021": (["PROG-04"], 120, "PER_BATCH"),        # glycine (formulation)
    "RM-022": (ALL_PROGRAMS, 60, "PER_BATCH"),        # hydrochloric acid
    # --- Single-use components (count per batch) ---
    "SU-001": (LARGE_SCALE, 1, "PER_BATCH"),          # 2000 L bioreactor bag
    "SU-002": (SMALL_SCALE, 1, "PER_BATCH"),          # 500 L bioreactor bag
    "SU-003": (ALL_PROGRAMS, 1, "PER_BATCH"),         # seed bioreactor bag
    "SU-004": (ALL_PROGRAMS, 2, "PER_BATCH"),         # mixer bag (media + buffer prep)
    "SU-005": (ALL_PROGRAMS, 4, "PER_BATCH"),         # media storage bag
    "SU-006": (ALL_PROGRAMS, 3, "PER_BATCH"),         # buffer storage bag
    "SU-007": (ALL_PROGRAMS, 6, "PER_BATCH"),         # harvest collection bag
    "SU-008": (ALL_PROGRAMS, 8, "PER_BATCH"),         # freeze-thaw bag (DS storage)
    "SU-009": (ALL_PROGRAMS, 6, "PER_BATCH"),         # 0.2 um filter, 10 in
    "SU-010": (LARGE_SCALE, 3, "PER_BATCH"),          # 0.2 um filter, 30 in
    "SU-011": (ALL_PROGRAMS, 4, "PER_BATCH"),         # depth filter
    "SU-012": (ALL_PROGRAMS, 1, "PER_BATCH"),         # virus filter
    "SU-013": (ALL_PROGRAMS, 2, "PER_BATCH"),         # TFF cassette
    "SU-014": (ALL_PROGRAMS, 20, "PER_BATCH"),        # aseptic connector
    "SU-015": (ALL_PROGRAMS, 10, "PER_BATCH"),        # silicone tubing assembly
    "SU-016": (ALL_PROGRAMS, 8, "PER_BATCH"),         # weldable tubing assembly
    "SU-017": (ALL_PROGRAMS, 6, "PER_BATCH"),         # sampling manifold
    "SU-018": (ALL_PROGRAMS, 30, "PER_BATCH"),        # media bottle
    "SU-019": (ALL_PROGRAMS, 6, "PER_BATCH"),         # transfer line
    "SU-020": (ALL_PROGRAMS, 3, "PER_BATCH"),         # pH sensor
    "SU-021": (ALL_PROGRAMS, 3, "PER_BATCH"),         # dissolved oxygen sensor
    "SU-022": (ALL_PROGRAMS, 6, "PER_BATCH"),         # pressure sensor
    "SU-023": (ALL_PROGRAMS, 4, "PER_BATCH"),         # conductivity sensor
    # --- Packaging (per unit filled; 1.02 = 2% line loss, 1.05 = 5% label overage) ---
    "PK-001": (["PROG-01"], 1.02, "PER_UNIT"),        # 10 mL vial
    "PK-002": (["PROG-03"], 1.02, "PER_UNIT"),        # 2R vial
    "PK-003": (["PROG-02"], 1.02, "PER_UNIT"),        # syringe barrel
    "PK-004": (["PROG-01"], 1.02, "PER_UNIT"),        # 20 mm lyo stopper
    "PK-005": (["PROG-03"], 1.02, "PER_UNIT"),        # 13 mm stopper
    "PK-006": (["PROG-02"], 1.02, "PER_UNIT"),        # plunger stopper
    "PK-007": (["PROG-01"], 1.02, "PER_UNIT"),        # 20 mm seal
    "PK-008": (["PROG-02"], 1.02, "PER_UNIT"),        # needle shield
    "PK-009": (["PROG-01"], 1.05, "PER_UNIT"),        # label, strength A
    "PK-010": (["PROG-03"], 1.05, "PER_UNIT"),        # label, strength B
    "PK-011": (VIAL_PROGRAMS, 2, "PER_UNIT"),         # carton seal label (2 per carton)
    "PK-012": (FILLED_ON_SITE, 1, "PER_UNIT"),        # leaflet
    "PK-013": (VIAL_PROGRAMS, 1, "PER_UNIT"),         # folding carton
    "PK-014": (VIAL_PROGRAMS, 0.1, "PER_UNIT"),       # vial tray (holds 10)
    "PK-015": (FILLED_ON_SITE, 1 / 300, "PER_UNIT"),  # cold-chain shipper (holds 300)
}


def build_production_schedule():
    """
    Turn the campaign calendar into a week-by-week schedule.

    Returns a dict: week_no -> {
        "ds_program", "ds_batches", "ds_campaign_start",
        "fill_program", "fill_units" }
    """
    schedule = {
        week_no: {"ds_program": None, "ds_batches": 0, "ds_campaign_start": False,
                  "fill_program": None, "fill_units": 0}
        for week_no in range(1, N_WEEKS + 1)
    }

    # Drug substance suite: the first week of a campaign runs 1 batch
    # (ramp-up), later weeks run 1 or 2 batches.
    for program, first, last in DS_CAMPAIGNS:
        for week_no in range(first, last + 1):
            week = schedule[week_no]
            assert week["ds_program"] is None, f"Two DS campaigns overlap in week {week_no}"
            week["ds_program"] = program
            week["ds_campaign_start"] = (week_no == first)
            week["ds_batches"] = 1 if week_no == first else int(rng.integers(1, 2, endpoint=True))

    # Fill/finish line: the normal weekly output, +/- 10%, rounded to 500 units.
    for program, first, last in FILL_CAMPAIGNS:
        for week_no in range(first, last + 1):
            week = schedule[week_no]
            assert week["fill_program"] is None, f"Two fill campaigns overlap in week {week_no}"
            units = PROGRAMS[program]["fill_units"] * rng.uniform(0.9, 1.1)
            week["fill_program"] = program
            week["fill_units"] = int(round(units / 500) * 500)

    return schedule


def assign_priority(requirement_type, program, criticality):
    """
    Priority of one weekly requirement:
      - a week with no demand (SAFETY_STOCK) is LOW
      - otherwise score 1 point for a COMMERCIAL program and 1 point for a
        HIGH-criticality material: 2 points = HIGH, 1 = MEDIUM, 0 = LOW
    """
    if requirement_type == "SAFETY_STOCK":
        return "LOW"
    points = 0
    if PROGRAMS[program]["stage"] == "COMMERCIAL":
        points += 1
    if criticality == "HIGH":
        points += 1
    return {2: "HIGH", 1: "MEDIUM", 0: "LOW"}[points]


def build_weekly_requirements(materials, schedule):
    """Create one row per material per week (60 x 26 = 1,560 rows)."""
    rows = []

    for material in materials.itertuples():
        programs, qty, basis = REQUIREMENT_BOM[material.material_id]

        for week_no, week in schedule.items():
            program = None
            amount = 0.0

            if basis in ("PER_BATCH", "PER_CAMPAIGN") and week["ds_program"] in programs:
                program = week["ds_program"]
                # Raw/process materials scale with bioreactor size
                # (a 500 L batch uses 1/4 of a 2000 L batch).
                scale = 1.0
                if material.material_category == "RAW_PROCESS":
                    scale = PROGRAMS[program]["scale_l"] / 2000
                if basis == "PER_BATCH":
                    amount = qty * week["ds_batches"] * scale
                elif week["ds_campaign_start"]:
                    amount = qty * scale

            elif basis == "PER_UNIT" and week["fill_program"] in programs:
                program = week["fill_program"]
                amount = qty * week["fill_units"]

            # Round up to a whole unit. (round(..., 6) first removes tiny
            # floating-point errors, e.g. 24480.000000001 -> 24480.)
            gross_qty = int(math.ceil(round(amount, 6)))

            if gross_qty == 0:
                requirement_type = "SAFETY_STOCK"
                program = None  # filled in below
            elif week_no <= FIRM_HORIZON_WEEKS:
                requirement_type = "PRODUCTION"
            else:
                requirement_type = "PLANNED_CAMPAIGN"

            rows.append({
                "material_id": material.material_id,
                "week_start": PLANNING_START + pd.Timedelta(weeks=week_no - 1),
                "gross_requirement_qty": gross_qty,
                "requirement_type": requirement_type,
                "production_program": program,
                "criticality": material.criticality,  # helper, dropped below
            })

    df = pd.DataFrame(rows)

    # Weeks with no demand still need a program code. Use the program whose
    # campaign this material is waiting for next (that is what its safety
    # stock is protecting). After the material's last campaign, use the last
    # program that used it.
    df["production_program"] = (
        df.groupby("material_id")["production_program"]
          .transform(lambda s: s.bfill().ffill())
    )

    df["priority"] = [
        assign_priority(rt, prog, crit)
        for rt, prog, crit in zip(df["requirement_type"], df["production_program"], df["criticality"])
    ]

    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    return df[REQUIREMENT_COLUMNS]


# ---------------------------------------------------------------------------
# 5. Opening inventory batches (build step 3)
# ---------------------------------------------------------------------------

# Opening inventory = the stock on hand on the first day of the plan.
# Later steps (MRP, exception detection) reuse this date.
INVENTORY_SNAPSHOT_DATE = PLANNING_START  # 2026-01-05

# Last day of the planning horizon (the Sunday of week 26 = 2026-07-05).
PLANNING_END = PLANNING_START + pd.Timedelta(weeks=N_WEEKS) - pd.Timedelta(days=1)

# How many batches a material has on hand (1 to 5), and the chance of each.
BATCH_COUNT_ODDS = {1: 0.15, 2: 0.30, 3: 0.30, 4: 0.15, 5: 0.10}

# Released (UNRESTRICTED) stock covers safety stock + the demand of the next
# 6 to 12 weeks (picked per material). 6 weeks = a tighter material,
# 12 weeks = a comfortable one. Purchase orders (build step 4) cover the
# weeks after that.
COVERAGE_WEEKS_RANGE = (6, 12)

# Chance that a material also holds one extra batch that is not usable yet.
# Only materials with 2+ batches can have a QUALITY_INSPECTION batch, and only
# materials with 3+ batches can have a BLOCKED batch, so every material keeps
# at least one UNRESTRICTED batch.
P_QUALITY_INSPECTION = 0.45
P_BLOCKED = 0.20

# Released batches were received at most this many days before the snapshot.
MAX_DAYS_SINCE_RECEIPT = 120
# Days between the supplier making a batch and us receiving it (min, max).
MANUFACTURE_TO_RECEIPT_DAYS = (7, 60)

# Fictional storage locations (4-character codes) and the storage condition
# each one provides. CR = cold room, WH = ambient warehouse.
STORAGE_LOCATIONS = {
    "CR01": "2-8 C",    # cold room, raw/process materials
    "CR02": "2-8 C",    # cold room, single-use components
    "WH01": "15-25 C",  # ambient warehouse, raw/process materials
    "WH02": "15-25 C",  # ambient warehouse, single-use components
    "WH03": "15-25 C",  # ambient warehouse, packaging
}
# (storage_condition, material_category) -> storage location
LOCATION_FOR = {
    ("2-8 C", "RAW_PROCESS"): "CR01",
    ("2-8 C", "SINGLE_USE"): "CR02",
    ("15-25 C", "RAW_PROCESS"): "WH01",
    ("15-25 C", "SINGLE_USE"): "WH02",
    ("15-25 C", "PACKAGING"): "WH03",
}

# Columns that hold dates (written to the CSV as YYYY-MM-DD).
INVENTORY_DATE_COLUMNS = [
    "goods_receipt_date", "manufacture_date", "expiration_date",
    "expected_quality_release_date",
]


def round_up_to_pack(qty, pack_size):
    """Round a quantity up to a whole number of packs (at least one pack)."""
    packs = max(1, math.ceil(round(qty, 6) / pack_size))
    return int(packs * pack_size)


def max_batch_age_days(material):
    """
    The oldest a batch may be (days from manufacture to the snapshot) and
    still have its minimum remaining shelf life on the last day of the plan.
    This keeps step 3 free of expiry problems; those are seeded in step 5.
    """
    days_to_plan_end = (PLANNING_END - INVENTORY_SNAPSHOT_DATE).days
    return (material.shelf_life_days
            - material.minimum_remaining_shelf_life_days
            - days_to_plan_end)


def build_inventory_batches(materials, requirements):
    """Create the batches on hand on INVENTORY_SNAPSHOT_DATE (1-5 per material)."""
    # material_id -> list of weekly demand, week 1 first.
    weekly_demand = requirements.groupby("material_id")["gross_requirement_qty"].apply(list).to_dict()

    rows = []

    for material in materials.itertuples():
        pack = material.rounding_value
        location = LOCATION_FOR[(material.storage_condition, material.material_category)]

        # --- 1. How many batches, and which ones are not usable yet --------
        n_batches = int(rng.choice(list(BATCH_COUNT_ODDS), p=list(BATCH_COUNT_ODDS.values())))
        has_qi = n_batches >= 2 and rng.random() < P_QUALITY_INSPECTION
        has_blocked = n_batches >= 3 and rng.random() < P_BLOCKED
        n_unrestricted = n_batches - int(has_qi) - int(has_blocked)

        # --- 2. Released stock: safety stock + demand of the next N weeks,
        #        split at random over the UNRESTRICTED batches ---------------
        coverage_weeks = int(rng.integers(*COVERAGE_WEEKS_RANGE, endpoint=True))
        target_qty = material.safety_stock_qty + sum(weekly_demand[material.material_id][:coverage_weeks])
        shares = rng.dirichlet([2.0] * n_unrestricted)  # random shares that add up to 1
        stock_types = ["UNRESTRICTED"] * n_unrestricted
        quantities = [round_up_to_pack(target_qty * share, pack) for share in shares]

        # --- 3. Extra stock on top of that, not usable on the snapshot date -
        if has_qi:
            # A recent delivery still being tested: half to one MOQ.
            quantities.append(round_up_to_pack(material.minimum_order_qty * rng.uniform(0.5, 1.0), pack))
            stock_types.append("QUALITY_INSPECTION")
        if has_blocked:
            # A small lot on hold: 10% to 30% of an MOQ.
            quantities.append(round_up_to_pack(material.minimum_order_qty * rng.uniform(0.1, 0.3), pack))
            stock_types.append("BLOCKED")

        # --- 4. Dates for each batch -----------------------------------------
        max_age = max_batch_age_days(material)
        batches = []
        for stock_type, qty in zip(stock_types, quantities):
            if stock_type == "QUALITY_INSPECTION":
                # Received in the last few days, so goods-receipt testing
                # (gr_processing_days) is not finished on the snapshot date.
                days_since_receipt = int(rng.integers(0, material.gr_processing_days - 1, endpoint=True))
            else:
                # Received long enough ago for testing to be finished.
                latest = min(MAX_DAYS_SINCE_RECEIPT, max_age - MANUFACTURE_TO_RECEIPT_DAYS[0])
                days_since_receipt = int(rng.integers(material.gr_processing_days, latest, endpoint=True))

            # The supplier made the batch 7-60 days before we received it,
            # but never so early that it would be too old by the plan end.
            lag_max = min(MANUFACTURE_TO_RECEIPT_DAYS[1], max_age - days_since_receipt)
            lag = int(rng.integers(MANUFACTURE_TO_RECEIPT_DAYS[0], lag_max, endpoint=True))

            gr_date = INVENTORY_SNAPSHOT_DATE - pd.Timedelta(days=days_since_receipt)
            manufacture_date = gr_date - pd.Timedelta(days=lag)

            if stock_type == "QUALITY_INSPECTION":
                release_date = gr_date + pd.Timedelta(days=material.gr_processing_days)
            else:
                release_date = pd.NaT  # left blank in the CSV

            batches.append({
                "material_id": material.material_id,
                "stock_type": stock_type,
                "quantity": qty,
                "goods_receipt_date": gr_date,
                "manufacture_date": manufacture_date,
                "expiration_date": manufacture_date + pd.Timedelta(days=material.shelf_life_days),
                "expected_quality_release_date": release_date,
                "storage_location": location,
            })

        # --- 5. Number the batches oldest first: RM-001-B01, RM-001-B02 ... --
        batches.sort(key=lambda batch: batch["goods_receipt_date"])
        for seq, batch in enumerate(batches, start=1):
            batch["batch_id"] = f"{material.material_id}-B{seq:02d}"
        rows.extend(batches)

    df = pd.DataFrame(rows)
    for col in INVENTORY_DATE_COLUMNS:
        df[col] = df[col].dt.strftime("%Y-%m-%d")  # NaT stays empty
    return df[INVENTORY_COLUMNS]


# ---------------------------------------------------------------------------
# 6. Checks: make sure the data is valid before saving
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


def validate_requirements(requirements, materials):
    """Stop with an error if the weekly requirements break a basic rule."""
    req = requirements

    # Every material in the master needs a bill-of-materials entry.
    assert set(REQUIREMENT_BOM) == set(materials["material_id"]), \
        "REQUIREMENT_BOM must list exactly the 60 materials"

    # Shape: exactly 60 x 26 rows, the planned columns, nothing missing.
    assert len(req) == 1560, f"Expected 1,560 rows, got {len(req)}"
    assert list(req.columns) == REQUIREMENT_COLUMNS, "Requirement columns do not match the plan"
    assert not req.isna().any().any(), "Missing values in weekly_requirements"

    # One row per material per week.
    dupes = req.duplicated(subset=["material_id", "week_start"])
    assert not dupes.any(), f"{dupes.sum()} duplicate material/week pairs"

    # Every material_id exists in the material master, and all 60 appear.
    unknown = set(req["material_id"]) - set(materials["material_id"])
    assert not unknown, f"Unknown material_id(s): {unknown}"
    assert req["material_id"].nunique() == 60, "Not all 60 materials have requirements"

    # The 26 week_start dates are the expected Mondays.
    expected_weeks = {
        (PLANNING_START + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(N_WEEKS)
    }
    assert set(req["week_start"]) == expected_weeks, "week_start dates are not the 26 planning Mondays"
    assert (pd.to_datetime(req["week_start"]).dt.dayofweek == 0).all(), "week_start must be a Monday"

    # Quantities are whole numbers and never negative.
    assert (req["gross_requirement_qty"] >= 0).all(), "Negative gross_requirement_qty"

    # Only valid codes.
    assert set(req["requirement_type"]) <= ALLOWED_REQUIREMENT_TYPES, "Invalid requirement_type"
    assert set(req["priority"]) <= ALLOWED_PRIORITIES, "Invalid priority"
    assert set(req["production_program"]) <= set(PROGRAMS), "Invalid production_program"

    # SAFETY_STOCK rows (and only those) have zero quantity.
    is_zero = req["gross_requirement_qty"] == 0
    is_ss = req["requirement_type"] == "SAFETY_STOCK"
    assert (is_zero == is_ss).all(), "SAFETY_STOCK must mean zero quantity, and zero quantity must be SAFETY_STOCK"

    # A program can only create demand for materials in its bill of materials.
    demand = req[~is_zero]
    wrong_program = [
        (m, p) for m, p in zip(demand["material_id"], demand["production_program"])
        if p not in REQUIREMENT_BOM[m][0]
    ]
    assert not wrong_program, f"Demand from a program that does not use the material: {wrong_program[:5]}"

    # Every material is used at least once in the horizon.
    no_demand = set(materials["material_id"]) - set(demand["material_id"])
    assert not no_demand, f"Materials with no demand at all: {no_demand}"


def validate_inventory(batches, materials, requirements):
    """Stop with an error if the opening inventory breaks a basic rule."""
    b = batches
    snapshot = INVENTORY_SNAPSHOT_DATE

    # Material master fields, lined up with each batch row.
    master = materials.set_index("material_id")
    shelf_life = pd.to_timedelta(b["material_id"].map(master["shelf_life_days"]), unit="D")
    gr_processing = pd.to_timedelta(b["material_id"].map(master["gr_processing_days"]), unit="D")
    min_life_days = b["material_id"].map(master["minimum_remaining_shelf_life_days"])

    # Shape: planned columns, 130-180 batches, unique IDs.
    assert list(b.columns) == INVENTORY_COLUMNS, "Inventory columns do not match the plan"
    assert 130 <= len(b) <= 180, f"Expected 130-180 batches, got {len(b)}"
    assert b["batch_id"].is_unique, "Duplicate batch_id"

    # Every material_id exists in the master; every material has 1-5 batches.
    unknown = set(b["material_id"]) - set(materials["material_id"])
    assert not unknown, f"Unknown material_id(s): {unknown}"
    per_material = b.groupby("material_id").size()
    assert len(per_material) == 60, "Every material needs at least one batch"
    assert per_material.between(1, 5).all(), "Each material must have 1-5 batches"

    # Valid codes, positive quantities, nothing missing (except release date).
    assert set(b["stock_type"]) <= ALLOWED_STOCK_TYPES, "Invalid stock_type"
    assert (b["quantity"] > 0).all(), "Batch quantity must be > 0"
    assert not b.drop(columns="expected_quality_release_date").isna().any().any(), \
        "Missing values in inventory_batches"

    # Most batches are released, fewer are in testing, only a few are blocked.
    counts = b["stock_type"].value_counts()
    assert counts.get("UNRESTRICTED", 0) > len(b) / 2, "Most batches should be UNRESTRICTED"
    assert counts.get("QUALITY_INSPECTION", 0) > counts.get("BLOCKED", 0) >= 1, \
        "Expect more QUALITY_INSPECTION batches than BLOCKED ones, and at least one BLOCKED"

    # --- Date rules ---------------------------------------------------------
    gr_date = pd.to_datetime(b["goods_receipt_date"])
    mfg_date = pd.to_datetime(b["manufacture_date"])
    exp_date = pd.to_datetime(b["expiration_date"])
    release_date = pd.to_datetime(b["expected_quality_release_date"])
    is_qi = b["stock_type"] == "QUALITY_INSPECTION"

    assert (mfg_date < gr_date).all(), "manufacture_date must be before goods_receipt_date"
    assert (gr_date <= snapshot).all(), "goods_receipt_date must be on or before the snapshot date"
    assert (exp_date == mfg_date + shelf_life).all(), "expiration_date must be manufacture_date + shelf_life_days"

    # QUALITY_INSPECTION: release date = receipt + GR processing time, after the snapshot.
    assert release_date[is_qi].notna().all(), "QUALITY_INSPECTION batch missing release date"
    assert (release_date[is_qi] == gr_date[is_qi] + gr_processing[is_qi]).all(), \
        "Release date must be goods_receipt_date + gr_processing_days"
    assert (release_date[is_qi] > snapshot).all(), "QUALITY_INSPECTION release date must be after the snapshot"

    # UNRESTRICTED / BLOCKED: no release date, and testing finished before the snapshot.
    assert release_date[~is_qi].isna().all(), "Only QUALITY_INSPECTION batches may have a release date"
    assert (gr_date[~is_qi] + gr_processing[~is_qi] <= snapshot).all(), \
        "UNRESTRICTED/BLOCKED batches must have finished goods-receipt processing"

    # Storage location must provide the material's storage condition.
    location_condition = b["storage_location"].map(STORAGE_LOCATIONS)
    material_condition = b["material_id"].map(master["storage_condition"])
    wrong_place = b[location_condition != material_condition]
    assert wrong_place.empty, f"Wrong storage location for: {list(wrong_place['batch_id'])}"

    # --- No problems seeded yet (that is build step 5) ----------------------
    # No expiry problem: every batch still has its minimum remaining shelf
    # life on the last day of the plan.
    life_left_at_end = (exp_date - PLANNING_END).dt.days
    assert (life_left_at_end >= min_life_days).all(), "A batch would breach minimum remaining shelf life"

    # No shortage, safety-stock breach, or quality hold early on: released
    # stock alone covers safety stock + demand for the shortest coverage
    # window. (Quality-inspection stock releases within 14 days, inside this
    # window, so it can not hold up demand either.)
    first_weeks_end = PLANNING_START + pd.Timedelta(weeks=COVERAGE_WEEKS_RANGE[0])
    early = requirements[pd.to_datetime(requirements["week_start"]) < first_weeks_end]
    early_demand = early.groupby("material_id")["gross_requirement_qty"].sum()
    released = b[b["stock_type"] == "UNRESTRICTED"].groupby("material_id")["quantity"].sum()
    needed = master["safety_stock_qty"] + early_demand.reindex(master.index, fill_value=0)
    short = needed[released.reindex(master.index, fill_value=0) < needed]
    assert short.empty, f"Opening stock does not cover the first weeks for: {list(short.index)}"
    assert (release_date[is_qi] < first_weeks_end).all(), "Quality release falls outside the covered weeks"


# ---------------------------------------------------------------------------
# 7. Main: build, check, save
# ---------------------------------------------------------------------------

def main():
    # Order matters: everything shares one random number generator, so the
    # master data is built first (exactly as in step 1), then requirements
    # (step 2), then inventory (step 3). New steps go at the end, so the
    # files from earlier steps stay exactly the same.
    suppliers = build_supplier_master()
    materials = build_material_master(suppliers)
    validate(suppliers, materials)

    schedule = build_production_schedule()
    requirements = build_weekly_requirements(materials, schedule)
    validate_requirements(requirements, materials)

    inventory = build_inventory_batches(materials, requirements)
    validate_inventory(inventory, materials, requirements)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suppliers.to_csv(OUTPUT_DIR / "supplier_master.csv", index=False)
    materials.to_csv(OUTPUT_DIR / "material_master.csv", index=False)
    requirements.to_csv(OUTPUT_DIR / "weekly_requirements.csv", index=False)
    inventory.to_csv(OUTPUT_DIR / "inventory_batches.csv", index=False)

    print(f"Wrote {len(suppliers)} suppliers -> {OUTPUT_DIR / 'supplier_master.csv'}")
    print(f"Wrote {len(materials)} materials -> {OUTPUT_DIR / 'material_master.csv'}")
    print(f"Wrote {len(requirements)} weekly requirements -> {OUTPUT_DIR / 'weekly_requirements.csv'}")
    print(f"Wrote {len(inventory)} inventory batches -> {OUTPUT_DIR / 'inventory_batches.csv'}")
    print("All validation checks passed.")


if __name__ == "__main__":
    main()
