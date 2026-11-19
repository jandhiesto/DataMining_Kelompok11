import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"C:\Users\LENOVO\Downloads\DataMining_Kelompok11\ProjectDataMining_dblpDec2025.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 3
nb["cells"][3]["source"] = [
    "# Checking file existence\n",
    "XML_PATH = Path(r\"C:\\Users\\Mark\\Documents\\ITB\\DataMining\\TugasKelompok\\dblp-2025-12-01.xml\")\n",
    "if not XML_PATH.exists():\n",
    "    for alt in [Path(\"dblp-2025-12-01.xml\"), Path(\"data/dblp-2025-12-01.xml\")]:\n",
    "        if alt.exists():\n",
    "            XML_PATH = alt\n",
    "            break\n",
    "if XML_PATH.exists():\n",
    "    print(XML_PATH.exists(), XML_PATH.stat().st_size / 1e9, \"GB\")\n",
    "else:\n",
    "    print(f\"File XML tidak ditemukan di {XML_PATH}. (Ekstraksi dilewati jika data/parquet sudah ada)\")\n"
]

# Cell 20
nb["cells"][20]["source"] = [
    "# Cek ketersediaan file parquet mentah atau sampel\n",
    "SAMPLE_PARQUET = OUT_DIR / \"clean_inproceedings_sample.parquet\"\n",
    "if RAW_PARQUET.exists():\n",
    "    df = pd.read_parquet(RAW_PARQUET)\n",
    "elif (OUT_DIR / \"clean_inproceedings.parquet\").exists():\n",
    "    df = pd.read_parquet(OUT_DIR / \"clean_inproceedings.parquet\")\n",
    "elif SAMPLE_PARQUET.exists():\n",
    "    print(f\"Info: {RAW_PARQUET} belum ada di disk lokal. Memuat data sampel: {SAMPLE_PARQUET}\")\n",
    "    df = pd.read_parquet(SAMPLE_PARQUET)\n",
    "else:\n",
    "    import sys\n",
    "    sys.path.append(\"src\")\n",
    "    from data_loader import load_clean_data\n",
    "    df = load_clean_data()\n",
    "\n",
    "funnel = []\n",
    "def log_step(step, before, after, note=\"\"):\n",
    "    funnel.append({\"step\": step, \"rows_before\": before, \"rows_after\": after,\n",
    "                   \"removed\": before - after, \"note\": note})\n",
    "\n",
    "log_step(\"Raw extract: inproceedings, 2015-2025\", len(df), len(df))\n",
    "display(pd.DataFrame(funnel))\n"
]

# Cell 22
nb["cells"][22]["source"] = [
    "# Step 1: select conference and workshop papers\n",
    "KEEP_PREFIXES = {\"conf\"}      # decision point: see note below\n",
    "\n",
    "if \"key_prefix\" in df.columns:\n",
    "    print(\"Key prefixes before:\")\n",
    "    print(df[\"key_prefix\"].value_counts(), \"\\n\")\n",
    "\n",
    "    mask = df[\"key_prefix\"].isin(KEEP_PREFIXES)\n",
    "    print(\"Examples of rows that would be dropped:\")\n",
    "    display(df.loc[~mask, [\"paper_key\", \"booktitle\"]].head(10))\n",
    "\n",
    "    before = len(df)\n",
    "    df = df.loc[mask].copy()\n",
    "    log_step(\"Keep conference/workshop keys\", before, len(df),\n",
    "             f\"prefixes kept: {sorted(KEEP_PREFIXES)}\")\n",
    "    display(pd.DataFrame(funnel))\n",
    "else:\n",
    "    print(\"Dataset sudah dalam format conference/workshop terfilter.\")\n"
]

# Cell 24
nb["cells"][24]["source"] = [
    "def blank(s):\n",
    "    \"\"\"True where a value is missing or only whitespace.\"\"\"\n",
    "    return s.isna() | s.astype(str).str.strip().eq(\"\")\n",
    "\n",
    "if \"booktitle\" in df.columns and \"venue_key\" in df.columns:\n",
    "    # drop empty/whitespace-only author entries; strip stray spaces\n",
    "    df[\"authors\"] = df[\"authors\"].map(lambda L: [a.strip() for a in L if a and str(a).strip()])\n",
    "\n",
    "    drop_rules = {\n",
    "        \"no author\":       df[\"authors\"].str.len().eq(0),\n",
    "        \"blank booktitle\": blank(df[\"booktitle\"]),\n",
    "        \"blank venue_key\": blank(df[\"venue_key\"]),\n",
    "        \"blank year\":      blank(df[\"year_raw\"] if \"year_raw\" in df.columns else df[\"year\"]),\n",
    "    }\n",
    "    flags = pd.concat(drop_rules, axis=1)\n",
    "\n",
    "    print(\"Rows failing each rule (rules can overlap):\")\n",
    "    print(flags.sum(), \"\\n\")\n",
    "    print(\"Rows failing at least one rule:\", int(flags.any(axis=1).sum()))\n",
    "    print(\"Title blank (kept, reported only):\", int(blank(df[\"title\"]).sum()))\n",
    "\n",
    "    before = len(df)\n",
    "    df = df.loc[~flags.any(axis=1)].copy()\n",
    "    log_step(\"Drop rows missing author/booktitle/venue/year\", before, len(df),\n",
    "             \"blank titles kept\")\n",
    "    display(pd.DataFrame(funnel))\n",
    "else:\n",
    "    print(\"Dataset sudah melewati tahap pembersihan missing values.\")\n"
]

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook resilient updates applied successfully!")
