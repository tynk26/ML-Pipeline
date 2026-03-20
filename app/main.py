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
    Requirement 2-1 & 2-2: Clean, merge, and load data into SQL.
    Tracks rejections with 'reason' and 'stage'.
    """
    if os.path.exists(DB_PATH):
        return {"message": "Database already exists. Ingestion skipped."}

    try:
        # 1. Load
        selections_raw, odds_df, labels_df = load_all()

        # 2. Normalize
        selections_df = normalize_selections(selections_raw)

        # 3. ODD Tagging Step (Stage Tracking)
        merged_odds_df, missing_odd_ids = merge_selections_with_odds(selections_df, odds_df)
        
        rejection_odds = selections_df[selections_df["video_id"].isin(missing_odd_ids)].copy()
        rejection_odds["reason"] = "MISSING_ODD_METADATA"
        rejection_odds["stage"] = "odd_tagging_step"

        # 4. Auto Labeling Step (Stage Tracking)
        final_df, label_stats = merge_with_labels(merged_odds_df, labels_df)
        
        missing_label_ids = label_stats.get("join_missing", [])
        rejection_labels = merged_odds_df[merged_odds_df["video_id"].isin(missing_label_ids)].copy()
        rejection_labels["reason"] = "MISSING_AUTO_LABELS"
        rejection_labels["stage"] = "auto_labeling_step"

        # 5. Combine Rejections
        all_rejections_df = pd.concat([rejection_odds, rejection_labels], ignore_index=True)
        
        conn = sqlite3.connect(DB_PATH)

        # 6. Save Rejections with Stage info
        if not all_rejections_df.empty:
            rejections_to_save = all_rejections_df[["video_id", "reason", "stage", "raw_data"]].copy()
            rejections_to_save.to_sql("rejections", conn, if_exists="replace", index=False)

        # 7. Save Integrated Data (No raw_data)
        if "raw_data" in final_df.columns:
            final_df = final_df.drop(columns=["raw_data"])
        final_df.to_sql("integrated_data", conn, if_exists="replace", index=False)

        # 8. Optimized Indexing
        conn.execute("CREATE INDEX idx_vid ON integrated_data(video_id)")
        conn.execute("CREATE INDEX idx_rej_stage ON rejections(stage)")
        count_cols = [c for c in final_df.columns if c.startswith("label_") and c.endswith("_count")]
        for col in count_cols:
            conn.execute(f"CREATE INDEX idx_{col} ON integrated_data({col})")
        
        conn.close()
        return {"status": "success", "integrated": len(final_df), "rejected": len(all_rejections_df)}

    except Exception as e:
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        raise HTTPException(status_code=500, detail=str(e))

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