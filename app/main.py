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
    Requirement 2-1 & 2-2: 정제, 통합, 적재 프로세스.
    - 거부 사유(Reason)와 발생 단계(Stage)를 명확히 구분하여 저장합니다.
    - 음수 객체 카운트 등 노이즈 데이터를 탐지하여 Rejection 처리합니다.
    """
    if os.path.exists(DB_PATH):
        return {"message": "Database already exists. Ingestion skipped."}

    print("[analyze] Starting optimized data pipeline with noise detection...")
    
    try:
        # 1. 원본 데이터 로드
        selections_raw, odds_df, labels_df = load_all()

        # 2. 정규화 (Selection 단계)
        selections_df = normalize_selections(selections_raw)

        # 3. ODD Tagging 단계 조인 및 Rejection 추출
        merged_odds_df, missing_odd_ids = merge_selections_with_odds(selections_df, odds_df)
        
        # ODD 단계 거부 데이터 (Missing metadata)
        rejection_odds = selections_df[selections_df["video_id"].isin(missing_odd_ids)].copy()
        rejection_odds["reason"] = "MISSING_ODD_METADATA"
        rejection_odds["stage"] = "odd_tagging_step"

        # 4. Auto Labeling 단계 조인 및 노이즈 필터링
        # merge_with_labels 함수에서 음수 카운트(Noise)를 탐지하여 반환하도록 설계됨
        final_df, label_stats = merge_with_labels(merged_odds_df, labels_df)
        
        # A. 레이블 데이터 자체가 없는 경우
        missing_ids = label_stats.get("missing_ids", [])
        rejection_missing = merged_odds_df[merged_odds_df["video_id"].isin(missing_ids)].copy()
        rejection_missing["reason"] = "MISSING_AUTO_LABELS"
        rejection_missing["stage"] = "auto_labeling_step"

        # B. 음수 카운트 등 노이즈 데이터인 경우 (Specific Reason 포함)
        neg_map = label_stats.get("negative_count_map", {})
        rejection_noise = merged_odds_df[merged_odds_df["video_id"].isin(neg_map.keys())].copy()
        rejection_noise["reason"] = rejection_noise["video_id"].map(neg_map) # 예: "INVALID_NEGATIVE_COUNT: car"
        rejection_noise["stage"] = "auto_labeling_step"

        # 5. 모든 Rejection 통합
        all_rejections_df = pd.concat([rejection_odds, rejection_missing, rejection_noise], ignore_index=True)
        
        # 6. SQLite 연결 및 데이터 적재
        conn = sqlite3.connect(DB_PATH)

        # Rejection 테이블 저장 (Requirement 2-2 필터링을 위해 stage, reason 포함)
        if not all_rejections_df.empty:
            rejections_to_save = all_rejections_df[["video_id", "reason", "stage", "raw_data"]].copy()
            rejections_to_save.to_sql("rejections", conn, if_exists="replace", index=False)

        # Integrated Data 저장 (raw_data 제거하여 경량화)
        if "raw_data" in final_df.columns:
            final_df = final_df.drop(columns=["raw_data"])
        final_df.to_sql("integrated_data", conn, if_exists="replace", index=False)

        # 7. 성능 최적화를 위한 인덱스 생성 (Requirement 2-3 검색 성능 최적화)
        conn.execute("CREATE INDEX idx_vid ON integrated_data(video_id)")
        conn.execute("CREATE INDEX idx_weather ON integrated_data(weather)")
        conn.execute("CREATE INDEX idx_rej_stage ON rejections(stage)")
        conn.execute("CREATE INDEX idx_rej_reason ON rejections(reason)")
        
        # 동적으로 생성된 label_count 컬럼들에 대해 인덱스 생성
        count_cols = [c for c in final_df.columns if c.startswith("label_") and c.endswith("_count")]
        for col in count_cols:
            conn.execute(f"CREATE INDEX idx_{col} ON integrated_data({col})")
        
        conn.close()

        print(f"[analyze] Complete. Integrated: {len(final_df)}, Rejected: {len(all_rejections_df)}")
        
        return {
            "status": "success",
            "counts": {
                "integrated": len(final_df),
                "rejected_total": len(all_rejections_df),
                "rejected_odd_step": len(rejection_odds),
                "rejected_label_step": len(rejection_missing) + len(rejection_noise)
            }
        }

    except Exception as e:
        if os.path.exists(DB_PATH): 
            os.remove(DB_PATH)
        print(f"[analyze] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data pipeline failed: {str(e)}")

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