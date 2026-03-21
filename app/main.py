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

# API 문서의 메타데이터 설정
app = FastAPI(
    title="🚗 자율주행 학습 데이터 통합 및 검증 API",
    description="""
    본 API는 자율주행 차량 센서 데이터(ODD)와 객체 인식 결과(Labels)를 통합하고, 
    데이터 무결성을 검증하여 최적의 학습셋을 구축하는 시스템입니다.
    
    ### 주요 기능:
    * **데이터 파이프라인 (/analyze)**: 원본 데이터를 정제하여 통합 DB 구축
    * **거절 데이터 관리 (/rejections)**: 중복, 음수, 누락 등 오류 데이터 추적 및 필터링
    * **정밀 검색 (/search)**: 다양한 환경 조건과 객체 카운트 기반의 샌드위치 검색
    """,
    version="1.0.0"
)

DB_PATH = "ml_data.db"

@app.post("/analyze", tags=["Pipeline"])
async def analyze():
    """
    ### ⚙️ 데이터 파이프라인 가동 (Ingestion & Cleaning)
    1. 원본 JSON 및 CSV 데이터를 로드하고 정규화합니다.
    2. **1단계(ODD)**: 메타데이터 누락 및 비디오 ID 중복 여부를 체크하여 거절합니다.
    3. **2단계(Labeling)**: 객체 수의 음수/실수 여부, 클래스 중복을 체크하여 거절합니다.
    4. 정상 데이터는 `integrated_data` 테이블에, 오류 데이터는 `rejections` 테이블에 저장합니다.
    """
    if os.path.exists(DB_PATH):
        return {"message": "데이터베이스가 이미 존재합니다. 통합 과정을 건너뜁니다."}

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

        # B. REJECTION: Duplicate ODD Metadata
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
        rejection_summary = {}
        if not all_rejections_df.empty:
            stage_counts = all_rejections_df["stage"].value_counts().to_dict()
            reason_counts = all_rejections_df["reason"].value_counts().to_dict()
            rejection_summary = {"by_stage": stage_counts, "by_reason": reason_counts}

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
            "counts": {"integrated": len(final_df), "total_rejected": len(all_rejections_df)},
            "rejection_breakdown": rejection_summary
        }

    except Exception as e:
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        raise HTTPException(status_code=500, detail=f"Pipeline Failure: {str(e)}")

  
@app.get("/rejections")
def get_rejections(
    stage: Optional[str] = Query(
        None, 
        description="데이터 처리 단계 (Step 1: ODD 또는 Step 2: Labeling)",
        openapi_examples={
            "Step 1: ODD Tagging": {
                "summary": "1단계: ODD 매칭 단계",
                "value": "odd_tagging_step"
            },
            "Step 2: Auto Labeling": {
                "summary": "2단계: 라벨 데이터 검증 단계",
                "value": "auto_labeling_step"
            }
        }
    ),
    reason: Optional[str] = Query(
        None, 
        description="해당 단계에 발생하는 구체적인 사유",
        openapi_examples={
            "ODD: Missing Metadata": {
                "summary": "[ODD 전용] ODD 데이터 없음",
                "value": "missing_odd_metadata"
            },
            "ODD: Duplicate ID": {
                "summary": "[ODD 전용] 중복된 ODD 비디오 ID",
                "value": "duplicate_odd_metadata"
            },
            "Label: Missing Labels": {
                "summary": "[Labeling 전용] LABELING 데이터 없음",
                "value": "missing_label_data"
            },
            "Label: Car Duplicates": {
                "summary": "[Labeling 전용] LABELING object 중복",
                "value": "duplicate_label"
            },
            "Label: Pedestrian Negatives": {
                "summary": "[Labeling 전용] LABELING object 음수 오류",
                "value": "negative_obj_count"
            }
        }
    ), 
    page: int = Query(1, ge=1), 
    size: int = Query(50, ge=1, le=100)
):
    """
    ### 🛡️ 거절 데이터 필터링 가이드
    Swagger UI의 각 파라미터 드롭다운에서 **단계(Stage)**와 **사유(Reason)** 조합을 선택하여 테스트할 수 있습니다.
    
    * **Step 1 (ODD)** 선택 시 -> `missing_odd_metadata` 등을 조합하세요.
    * **Step 2 (Labeling)** 선택 시 -> `duplicate_label`이나 `negative_obj_count` 등을 조합하세요.
    """
    if not os.path.exists(DB_PATH):
        return {"total": 0, "items": [], "summary_by_categories": {}}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. 동적 요약 (Summary) 생성 - 현재 DB에 실시간으로 존재하는 카테고리 노출
    summary_query = "SELECT stage, reason, COUNT(*) as count FROM rejections GROUP BY stage, reason"
    cursor.execute(summary_query)
    summary_rows = cursor.fetchall()
    summary = {}
    for row in summary_rows:
        s, r, c = row["stage"], row["reason"], row["count"]
        if s not in summary: summary[s] = {}
        summary[s][r] = c

    # 2. 필터링 로직
    base_query = "FROM rejections WHERE 1=1"
    query_params = []
    
    if stage:
        base_query += " AND stage = ?"
        query_params.append(stage)
    if reason:
        # LIKE를 사용하여 'car'만 입력해도 'duplicate_label: car'를 찾도록 유연성 유지
        base_query += " AND reason LIKE ?"
        query_params.append(f"%{reason}%")

    # 페이징 및 결과 반환
    cursor.execute(f"SELECT COUNT(*) {base_query}", query_params)
    total_count = cursor.fetchone()[0]

    query = f"SELECT * {base_query} LIMIT ? OFFSET ?"
    cursor.execute(query, query_params + [size, (page - 1) * size])
    rows = [dict(row) for row in cursor.fetchall()]
    
    for r in rows:
        if r.get("raw_data"):
            try: r["raw_data"] = json.loads(r["raw_data"])
            except: pass

    conn.close()
    
    return {
        "status": "success",
        "total": total_count,
        "summary_by_categories": summary,
        "items": rows
    }

@app.post("/search", tags=["Search"])
def search_data(
    filters: dict = Body(..., openapi_examples={
        "Video_3_Full_Confidence": {
            "summary": "Video 3 마스터 검색 (Confidence 필터 포함)",
            "value": {
                "video_id_min": 3, "video_id_max": 3,
                "weather": "sunny", "time_of_day": "night", "road_surface": "dry",
                "wiper_on": 1, "headlights_on": 1,
                "wiper_level_min": 3, "wiper_level_max": 3,
                "label_car_min": 31, "label_car_max": 31,
                "label_car_confidence_min": 0.8, "label_car_confidence_max": 0.9,
                "label_pedestrian_min": 11, "label_pedestrian_confidence_min": 0.7
            }
        }
    })
):
    if not os.path.exists(DB_PATH): raise HTTPException(status_code=503, detail="DB 미초기화")
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()

    query = "SELECT * FROM integrated_data WHERE 1=1"
    params = []

    # 1. DIRECT MATCHES
    direct_cols = ["weather", "time_of_day", "road_surface", "headlights_on", "wiper_on"]
    for col in direct_cols:
        if col in filters and filters[col] is not None:
            query += f" AND {col} = ?"; params.append(filters[col])

    # 2. RANGE MATCHES
    range_fields = ["video_id", "id", "temperature_fahrenheit", "temperature_celsius", "wiper_level", "recorded_at"]
    for key, val in filters.items():
        if val is None or val == "": continue
        target_col = None

        if key.startswith("label_"):
            base_key = key.replace("_min", "").replace("_max", "")
            target_col = base_key if "confidence" in base_key else f"label_{base_key.replace('label_', '')}_count"
        else:
            for field in range_fields:
                if key.startswith(field):
                    target_col = field; break

        if target_col:
            try:
                clean_val = val if target_col == "recorded_at" else float(val)
                if key.endswith("_min"): query += f" AND {target_col} >= ?"; params.append(clean_val)
                elif key.endswith("_max"): query += f" AND {target_col} <= ?"; params.append(clean_val)
            except ValueError: continue

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    for r in rows:
        if r.get("labels"): r["labels"] = json.loads(r["labels"])
    conn.close()
    return {"total_found": len(rows), "results": rows}

@app.get("/joined_data", tags=["View"])
def get_joined_data():
    """### 📂 통합 데이터 미리보기 (Top 50)"""
    if not os.path.exists(DB_PATH): raise HTTPException(status_code=503, detail="DB 초기화 필요")
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM integrated_data LIMIT 50")
    rows = [dict(row) for row in cursor.fetchall()]
    for r in rows:
        if r.get("labels"): r["labels"] = json.loads(r["labels"])
    conn.close()
    return {"count": len(rows), "data": rows}