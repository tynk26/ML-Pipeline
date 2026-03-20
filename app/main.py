import sqlite3
import os
import json
import pandas as pd
from fastapi import FastAPI, Body, HTTPException, Query
from typing import Dict, List, Optional

from app.ingestion.loader import load_all
from app.preprocessing.preprocessing import (
    normalize_selections,
    merge_selections_with_odds,
    merge_with_labels
)

app = FastAPI(title="ML Data SQL API")
DB_PATH = "ml_data.db"

@app.post("/analyze")
async def analyze():
    """
    Strict Data Pipeline with categorization for rejections.
    Stages: odd_tagging_step, auto_labeling_step
    """
    if os.path.exists(DB_PATH):
        return {"message": "Database already exists. Ingestion skipped."}

    try:
        # 1. Load and Normalize
        selections_raw, odds_df, labels_df = load_all()
        selections_df = normalize_selections(selections_raw)

        # --- STAGE 1: ODD Step ---
        # FIX: Unpack all 3 values returned by your helper function
        merged_odds_df, missing_odd_ids, duplicate_odd_ids = merge_selections_with_odds(selections_df, odds_df)
        
        # A. REJECTION: Missing ODD Metadata
        rejection_missing_odds = selections_df[selections_df["video_id"].isin(missing_odd_ids)].copy()
        rejection_missing_odds["reason"] = "missing_odd_metadata"
        rejection_missing_odds["stage"] = "odd_tagging_step"

        # B. REJECTION: Duplicate ODD Metadata (e.g., ID 4938)
        # We look back at the original selections_df to get the records for these duplicate IDs
        rejection_duplicates = selections_df[selections_df["video_id"].isin(duplicate_odd_ids)].copy()
        rejection_duplicates["reason"] = "duplicate_odd_metadata"
        rejection_duplicates["stage"] = "odd_tagging_step"

        # --- STAGE 2: Labeling Step ---
        # The merged_odds_df is already "clean" (duplicates removed by your helper)
        final_df, label_stats = merge_with_labels(merged_odds_df, labels_df)
        
        # C. REJECTION: Missing Label Data
        missing_label_ids = label_stats.get("missing_ids", [])
        rejection_missing_labels = merged_odds_df[merged_odds_df["video_id"].isin(missing_label_ids)].copy()
        rejection_missing_labels["reason"] = "missing_label_data"
        rejection_missing_labels["stage"] = "auto_labeling_step"

        # D. REJECTION: Integrity/Noise (from error_map)
        error_map = label_stats.get("error_map", {})
        rejection_errors = merged_odds_df[merged_odds_df["video_id"].isin(error_map.keys())].copy()
        rejection_errors["reason"] = rejection_errors["video_id"].map(error_map)
        rejection_errors["stage"] = "auto_labeling_step"

        # --- COMBINE AND SAVE ---
        frames = [rejection_missing_odds, rejection_duplicates, rejection_missing_labels, rejection_errors]
        all_rejections_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        
        conn = sqlite3.connect(DB_PATH)
        
        # Save Rejections
        if not all_rejections_df.empty:
            save_cols = ["video_id", "reason", "stage", "raw_data"]
            existing = [c for c in save_cols if c in all_rejections_df.columns]
            all_rejections_df[existing].to_sql("rejections", conn, if_exists="replace", index=False)

        # Save Integrated Data
        if "raw_data" in final_df.columns:
            final_df = final_df.drop(columns=["raw_data"])
        final_df.to_sql("integrated_data", conn, if_exists="replace", index=False)

        # Performance Indexing
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rej_reason ON rejections(reason)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rej_stage ON rejections(stage)")
        
        conn.close()
        return {
            "status": "success", 
            "integrated": len(final_df), 
            "rejected": len(all_rejections_df)
        }

    except Exception as e:
        if os.path.exists(DB_PATH): 
            os.remove(DB_PATH)
        # Re-raising the error with more context for debugging
        raise HTTPException(status_code=500, detail=f"Pipeline Failure: {str(e)}")

@app.get("/rejections")
def get_rejections(
    reason: Optional[str] = None, 
    stage: Optional[str] = None, 
    page: int = 1, 
    size: int = 50
):
    """
    Requirement 2-2: Get rejected data with filters for reason and stage.
    """
    if not os.path.exists(DB_PATH):
        return {"total": 0, "items": []}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build filtered query
    base_query = "FROM rejections WHERE 1=1"
    params = []
    if reason:
        base_query += " AND reason = ?"; params.append(reason)
    if stage:
        base_query += " AND stage = ?"; params.append(stage)

    # Get total count for metadata
    cursor.execute(f"SELECT COUNT(*) {base_query}", params)
    total_count = cursor.fetchone()[0]

    # Get paginated data
    query = f"SELECT * {base_query} LIMIT ? OFFSET ?"
    cursor.execute(query, params + [size, (page - 1) * size])
    rows = [dict(row) for row in cursor.fetchall()]
    
    for r in rows:
        if r.get("raw_data"): r["raw_data"] = json.loads(r["raw_data"])

    conn.close()
    return {"total": total_count, "page": page, "items": rows}

@app.post("/search")
def search_data(filters: dict = Body(...)):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="DB not initialized.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM integrated_data WHERE 1=1"
    params = []

    for col in ["weather", "time_of_day", "road_surface"]:
        if col in filters and filters[col]:
            vals = [v.strip() for v in str(filters[col]).split(",")]
            query += f" AND {col} IN ({', '.join(['?']*len(vals))})"
            params.extend(vals)

    for key, val in filters.items():
        if key.startswith("label_") and key.endswith("_min"):
            obj = key.replace("label_", "").replace("_min", "")
            query += f" AND label_{obj}_count >= ?"
            params.append(int(val))

    if "temperature_fahrenheit_min" in filters:
        query += " AND temperature_fahrenheit >= ?"; params.append(float(filters["temperature_fahrenheit_min"]))

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    for r in rows:
        if r.get("labels"): r["labels"] = json.loads(r["labels"])

    conn.close()
    return {"total_found": len(rows), "results": rows[:100]}

@app.get("/joined_data")
def get_joined_data():
    if not os.path.exists(DB_PATH): raise HTTPException(status_code=503, detail="DB not initialized.")
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM integrated_data LIMIT 50")
    rows = [dict(row) for row in cursor.fetchall()]
    for r in rows:
        if r.get("labels"): r["labels"] = json.loads(r["labels"])
    conn.close()
    return {"count": len(rows), "data": rows}