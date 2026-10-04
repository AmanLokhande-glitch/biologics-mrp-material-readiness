# Biologics Material Readiness & MRP Exception Control Tower

> **Portfolio & Confidentiality Notice**
>
> This repository is an independent portfolio project built entirely with synthetic data. It demonstrates general material-planning, MRP-exception-management, inventory, procurement, and analytics concepts used in manufacturing and biopharmaceutical supply chains.
>
> No employer data, supplier information, material numbers, production schedules, system extracts, screenshots, internal business rules, proprietary calculations, confidential documents, or non-public operational information are included. All organizations, suppliers, materials, quantities, dates, transactions, thresholds, scenarios, and results shown in this repository are fictional.

## Overview

This project simulates the weekly material-planning workflow of a cGMP biologics manufacturing environment. The goal is to identify material shortages and planning exceptions early enough for planners to act before they affect manufacturing schedules.

## Business Problem

Material planners don't just need to know how much inventory exists. They need to know whether the right quantity will be usable at the right time. A material can look well stocked and still put production at risk because inventory is in quality inspection, a purchase order arrives after the requirement date, a batch won't meet remaining shelf-life requirements, or demand has increased.

This project answers a more useful question: **which materials need planner action now, why are they at risk, and what action should be taken?**

## Planned Scope

- 60 direct materials across raw/process materials, single-use components, and packaging
- 14 fictional suppliers and 26 weeks of manufacturing requirements
- Batch-level inventory with unrestricted, quality-inspection, and blocked stock
- Open purchase orders with supplier-confirmed delivery dates
- Rule-based MRP exception detection, priority scoring, and recommended planner actions
- A Power BI material-readiness dashboard

## Tools

SAP-style MRP concepts · Python · SQL (SQLite) · Power BI · Excel

## Project Status

In progress. This README will be updated as each component is built.