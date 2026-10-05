   # Project rules for Claude

   ## About this project
   Portfolio project: Biologics Material Readiness & MRP Exception Control Tower.
   A synthetic, SAP-style material-planning case study for a fictional cGMP
   biologics manufacturer called "Northstar Biologics Manufacturing."
   Tools: Python (pandas, numpy), SQL (SQLite), Power BI, Excel.

   ## Hard rules
   - All data is synthetic. Never use real company, supplier, product, or person names.
   - The private/ folder holds my planning notes. You may read them for guidance,
     but never copy company names, research, or links from them into project files.
   - The private/ folder must always be listed in .gitignore.
   - Describe the work as "SAP-style MRP concepts." Never claim this runs SAP S/4HANA.
   - Never commit or push without first showing me the list of files and waiting for my OK.
   - If a file I point you to is missing or empty, stop and ask me before continuing.

   ## How to work with me
   - I am learning. After each task, explain what you did in simple words.
   - Keep code simple, readable, and well-commented. No machine learning.
   - Work on one build step at a time. Don't jump ahead.
   - Use a fixed random seed so the synthetic data is the same every run.

   ## Synthetic scope
   - 60 materials (22 raw/process, 23 single-use, 15 packaging), 14 suppliers, 4 production programs
   - Planning period Jan 5 - Jun 28, 2026, in 26 weekly buckets
   - About 150 PO lines and 130-180 inventory batches
   - Stock types: UNRESTRICTED, QUALITY_INSPECTION, BLOCKED