import sqlite3
import os
import json
import pandas as pd
from fastapi import FastAPI, Body, HTTPException
from typing import Dict, List, Optional

# 기존 로더 및 전처리 함수 임포트
from app.ingestion.loader import load_all
from app.preprocessing.preprocessing import (
    normalize_selections,
    merge_selections_with_odds,
    merge_with_labels,
    safe_json
)

app = FastAPI(title="ML Data SQL API")
DB_PATH = "ml_data.db"  # 로컬 테스트를 위해 경로 수정 (필요시 /data/로 변경)

@app.post("/post_to_db")
async def post_to_db():
    if os.path.exists(DB_PATH):
        return {"message": "Database already exists. Ingestion skipped."}

    print("[post_to_db] Initializing Data Pipeline (No Raw Data in Integrated)...")
    try:
        # 1. 원본 파일 로드
        selections_raw, odds_df, labels_df = load_all()

        # 2. 정규화 (이때 raw_data가 포함될 수 있음)
        selections_df = normalize_selections(selections_raw)

        # 3. Odds 조인 및 Rejections 추적
        merged_odds_df, missing_odd_ids = merge_selections_with_odds(selections_df, odds_df)
        
        rejections_list = []
        for vid in missing_odd_ids:
            raw_row = selections_df[selections_df["video_id"] == vid]
            if not raw_row.empty:
                rejections_list.append({
                    "video_id": int(vid),
                    "reason": "Missing entry in odds.csv",
                    "raw_data": raw_row.iloc[0].to_json(date_format='iso') 
                })

        # 4. Labels 조인 및 동적 플래트닝
        final_df, label_stats = merge_with_labels(merged_odds_df, labels_df)
        
        for vid in label_stats.get("join_missing", []):
            raw_row = merged_odds_df[merged_odds_df["video_id"] == vid]
            if not raw_row.empty:
                rejections_list.append({
                    "video_id": int(vid),
                    "reason": "Missing entry in labels.csv",
                    "raw_data": raw_row.iloc[0].to_json(date_format='iso')
                })

        # --- 핵심 수정 사항 ---
        # 5. integrated_data로 들어갈 데이터에서 raw_data 컬럼 제거
        if "raw_data" in final_df.columns:
            final_df = final_df.drop(columns=["raw_data"])
        # ----------------------

        # 6. SQLite 적재
        conn = sqlite3.connect(DB_PATH)
        
        # 메인 데이터 저장 (raw_data 없음, 가벼움)
        final_df.to_sql("integrated_data", conn, if_exists="replace", index=False)
        
        # Rejections 데이터 저장 (raw_data 포함, 추적용)
        rejections_df = pd.DataFrame(rejections_list)
        if not rejections_df.empty:
            rejections_df.to_sql("rejections", conn, if_exists="replace", index=False)

        # 7. 인덱스 생성
        conn.execute("CREATE INDEX idx_vid ON integrated_data(video_id)")
        count_cols = [c for c in final_df.columns if c.startswith("label_") and c.endswith("_count")]
        for col in count_cols:
            conn.execute(f"CREATE INDEX idx_{col} ON integrated_data({col})")
        
        conn.close()
        
        return {
            "status": "success",
            "message": "Raw data removed from integrated_data, preserved in rejections.",
            "final_count": len(final_df),
            "rejection_count": len(rejections_df)
        }

    except Exception as e:
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_data(filters: dict = Body(...)):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="DB not initialized. Call /post_to_db first.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM integrated_data WHERE 1=1"
    params = []

    # 1. 텍스트 필터 (weather, time_of_day 등)
    for col in ["weather", "time_of_day", "road_surface"]:
        if col in filters and filters[col]:
            vals = [v.strip() for v in str(filters[col]).split(",")]
            placeholders = ', '.join(['?'] * len(vals))
            query += f" AND {col} IN ({placeholders})"
            params.extend(vals)

    # 2. 동적 플래트닝 레이블 필터 ( label_car_min -> label_car_count >= X )
    for key, val in filters.items():
        if key.startswith("label_") and key.endswith("_min"):
            obj = key.replace("label_", "").replace("_min", "")
            # DB 컬럼명 규칙: label_{obj}_count
            query += f" AND label_{obj}_count >= ?"
            params.append(int(val))

    # 3. 범위 필터 (Temperature)
    if "temperature_fahrenheit_min" in filters:
        query += " AND temperature_fahrenheit >= ?"
        params.append(float(filters["temperature_fahrenheit_min"]))
    if "temperature_fahrenheit_max" in filters:
        query += " AND temperature_fahrenheit <= ?"
        params.append(float(filters["temperature_fahrenheit_max"]))

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    
    # JSON 복구
    for r in rows:
        if r.get("labels"):
            r["labels"] = json.loads(r["labels"])

    conn.close()
    return {"total_found": len(rows), "results": rows[:100]}

@app.get("/rejections")
def get_rejections():
    if not os.path.exists(DB_PATH): return {"items": []}
    conn = sqlite3.connect(DB_PATH)
    # raw_data가 포함된 rejections 테이블 조회
    df = pd.read_sql("SELECT * FROM rejections", conn)
    conn.close()
    
    results = df.to_dict(orient="records")
    for r in results:
        if r.get("raw_data"):
            r["raw_data"] = json.loads(r["raw_data"])
    return {"total": len(results), "items": results}

@app.get("/joined_data")
def get_joined_data():
    """
    DB에 적재된 최종 통합 데이터 중 상위 10개를 반환합니다.
    (Flattening된 컬럼과 원본 labels 객체를 모두 포함)
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="DB not initialized. Call /post_to_db first.")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 결과를 딕셔너리 형태로 받기 위해 설정
        cursor = conn.cursor()

        # 상위 10개 데이터 조회
        cursor.execute("SELECT * FROM integrated_data LIMIT 10")
        rows = cursor.fetchall()
        
        # Row 객체를 dict로 변환하고, 문자열로 저장된 labels를 JSON 객체로 파싱
        results = []
        for row in rows:
            item = dict(row)
            if item.get("labels"):
                try:
                    item["labels"] = json.loads(item["labels"])
                except:
                    pass # 파싱 실패 시 문자열 그대로 유지
            results.append(item)

        conn.close()

        return {
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")