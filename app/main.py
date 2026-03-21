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
        merged_odds_df, missing_odd_ids, duplicate_odd_ids = merge_selections_with_odds(selections_df, odds_df)
        
        # A. REJECTION: Missing ODD Metadata
        rejection_missing_odds = selections_df[selections_df["video_id"].isin(missing_odd_ids)].copy()
        rejection_missing_odds["reason"] = "missing_odd_metadata"
        rejection_missing_odds["stage"] = "odd_tagging_step"

        # B. REJECTION: Duplicate ODD Metadata (e.g., ID 4938)
        rejection_duplicates = selections_df[selections_df["video_id"].isin(duplicate_odd_ids)].copy()
        rejection_duplicates["reason"] = "duplicate_odd_metadata"
        rejection_duplicates["stage"] = "odd_tagging_step"

        # --- STAGE 2: Labeling Step ---
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

        # --- COMBINE ---
        frames = [rejection_missing_odds, rejection_duplicates, rejection_missing_labels, rejection_errors]
        all_rejections_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        
        # --- GENERATE SUMMARY STATS ---
        # This creates a dictionary of counts per stage and reason
        rejection_summary = {}
        if not all_rejections_df.empty:
            # Summary by Stage
            stage_counts = all_rejections_df["stage"].value_counts().to_dict()
            # Summary by Reason
            reason_counts = all_rejections_df["reason"].value_counts().to_dict()
            
            rejection_summary = {
                "by_stage": stage_counts,
                "by_reason": reason_counts
            }

        # --- SAVE TO DATABASE ---
        conn = sqlite3.connect(DB_PATH)
        
        if not all_rejections_df.empty:
            save_cols = ["video_id", "reason", "stage", "raw_data"]
            existing = [c for c in save_cols if c in all_rejections_df.columns]
            all_rejections_df[existing].to_sql("rejections", conn, if_exists="replace", index=False)

        if "raw_data" in final_df.columns:
            final_df = final_df.drop(columns=["raw_data"])
        final_df.to_sql("integrated_data", conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_rej_reason ON rejections(reason)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rej_stage ON rejections(stage)")
        conn.close()

        return {
            "status": "success", 
            "counts": {
                "integrated": len(final_df), 
                "total_rejected": len(all_rejections_df)
            },
            "rejection_breakdown": rejection_summary
        }

    except Exception as e:
        if os.path.exists(DB_PATH): 
            os.remove(DB_PATH)
        raise HTTPException(status_code=500, detail=f"Pipeline Failure: {str(e)}")
    
@app.get("/rejections")
def get_rejections(
    reason: Optional[str] = Query(
        None, 
        description="거부 사유 (예: missing_odd_metadata, duplicate_odd_metadata, missing_label_data)",
        openapi_examples={
            "missing_odd": {"value": "missing_odd_metadata"},
            "duplicate": {"value": "duplicate_odd_metadata"},
            "missing_label": {"value": "missing_label_data"}
        }
    ), 
    stage: Optional[str] = Query(
        None, 
        description="발생 단계 (예: odd_tagging_step, auto_labeling_step)",
        openapi_examples={
            "odd_stage": {"value": "odd_tagging_step"},
            "label_stage": {"value": "auto_labeling_step"}
        }
    ), 
    page: int = Query(1, ge=1, description="페이지 번호"), 
    size: int = Query(50, ge=1, le=100, description="페이지당 아이템 수")
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
def search_data(
    filters: dict = Body(
        ...,
        openapi_examples={
            "full_video_3_profile": {
                "summary": "Full Search Profile: Video ID 3",
                "value": {
                    "video_id_min": 3,
                    "video_id_max": 3,
                    "recorded_at_min": "2026-01-10",
                    "weather": "sunny",
                    "time_of_day": "night",
                    "road_surface": "dry",
                    "temperature_fahrenheit_min": 58,
                    "temperature_fahrenheit_max": 59,
                    "wiper_on": 1,
                    "headlights_on": 1,
                    "wiper_level_min": 3,
                    "wiper_level_max": 3,
                    "label_car_min": 31,
                    "label_car_max": 31,
                    "label_pedestrian_min": 11
                }
            }
        }
    )
):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="DB not initialized.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM integrated_data WHERE 1=1"
    params = []

    # --- 1. DIRECT MATCHES (Case-Insensitive) ---
    direct_cols = ["weather", "time_of_day", "road_surface", "headlights_on", "wiper_on"]
    for col in direct_cols:
        if col in filters and filters[col] is not None and filters[col] != "":
            val = filters[col]
            if col in ["headlights_on", "wiper_on"]:
                query += f" AND {col} = ?"
                params.append(int(val))
            elif isinstance(val, str) and "," in val:
                vals = [v.strip().lower() for v in val.split(",")]
                query += f" AND LOWER({col}) IN ({', '.join(['?']*len(vals))})"
                params.extend(vals)
            else:
                query += f" AND LOWER({col}) = LOWER(?)"
                params.append(str(val))

    # Partial Path
    if "source_path" in filters and filters["source_path"]:
        query += " AND source_path LIKE ?"
        params.append(f"%{filters['source_path']}%")

    # --- 2. NUMERIC & DATE SANDWICHES ---
    range_fields = ["video_id", "id", "temperature_fahrenheit", "temperature_celsius", "wiper_level", "recorded_at"]

    for key, val in filters.items():
        if val is None or val == "": continue
        
        target_col = None
        if key.startswith("label_"):
            obj = key.replace("label_", "").replace("_min", "").replace("_max", "")
            target_col = f"label_{obj}_count"
        else:
            for field in range_fields:
                if key.startswith(field):
                    target_col = field
                    break
        
        if target_col:
            # Casting for comparison
            try:
                clean_val = val if target_col == "recorded_at" else float(val)
                if key.endswith("_min"):
                    query += f" AND {target_col} >= ?"
                    params.append(clean_val)
                elif key.endswith("_max"):
                    query += f" AND {target_col} <= ?"
                    params.append(clean_val)
            except ValueError:
                continue

    try:
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        for r in rows:
            if r.get("labels"):
                r["labels"] = json.loads(r["labels"])
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")

    conn.close()
    return {"total_found": len(rows), "results": rows}

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