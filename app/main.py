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
import pandas as pd
import sqlite3
import os
import json
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter()

@app.post("/analyze", tags=["Pipeline"])
async def analyze():
    """
    ### 데이터 분석 및 DB 적재 
    
    ### 이 엔드포인트는 원본 데이터셋을 전수 조사하여 학습 가용 데이터를 추출하고 품질 미달 및 오류 데이터를 격리합니다.
    
    **1. 데이터 통합 및 정제 요약 (Processing Summary):** 전체 입력 데이터 중 결함을 걸러내고 최종 학습에 투입 가능한 유효 데이터의 총량과 정제 효율을 정량적으로 증명하는 지표입니다.**
    * **Total Input Videos:** 원본 데이터셋에 존재하는 총 영상 수입니다.
    * **Integrated Videos:** ODD와 LABELING데이터가 모두 존재하고, 데이터 무결성 검사를 통과한 최종 학습용 영상 수입니다.
    * **Integration Rate:** 전체 입력 대비 최종 통합된 영상의 비율로, 데이터 품질과 정제 효율을 나타냅니다.
    * **Total Rejections:** ODD 매칭 실패, LABELING 누락, 객체 수 오류 등으로 거절된 영상의 총 수입니다.

    **2. 단계별 격리 처리 (Rejection by Stage):**  각 처리 단계(ODD 매칭, 라벨링 검증)별로 거절된 영상 수를 집계하여 어느 단계에서 문제가 발생하는지 파악합니다.
    * 모든 거절 데이터는 발생 지점에 따라 `odd_tagging_step` 또는 `auto_labeling_step`으로 분류되어 기록됩니다.

    **3. 캡처하는 예외 케이스 (Rejection By Reason):** 거절 사유별로 집계하여 어떤 유형의 오류가 가장 빈번한지 분석합니다.
    * **Stage 1 (ODD DATA):** 메타데이터 누락(`missing_odd_metadata`).
    * **Stage 1 (ODD INTEGRITY):** ODD ID 중복(`duplicate_odd_metadata`).
    * **Stage 2 (LABELING DATA):** 라벨 파일 누락(`missing_label_data`).
    * **Stage 2 (LABELING INTEGRITY):** 객체 수 0(`zero_obj_count`), 음수(`negative_obj_count`), 실수(`non_integer_obj_count`), 클래스 중복(`duplicate_label_class`).
    
    **4. 거부 사유 병합 논리:** 한 비디오가 여러 단계에서 중복 오류를 가질 경우, `rejections` 테이블에는 `&`로 연결된 단일 문자열로 저장됩니다.
    * 예: `duplicate_odd_metadata & missing_label_data`
    
    **5. 통계 분석 (Statistical Report):** 최종 통합된 데이터셋에 대한 통계 분석을 통해 학습 데이터의 특성과 편향성을 파악합니다.
    * **Label Class Distribution:** 각 객체 클래스가 전체 영상 중 몇 퍼센트의 영상에 출현하는지 분석합니다. 특정 배경에만 객체가 편중되어 학습되는 '배경 편향성'을 탐지하는 데 사용됩니다.
    * **Scene Complexity Distribution:** 영상 내 총 객체 수를 기준으로 저/중/고밀도 상황을 분류합니다. 모델이 혼잡한 환경에서 성능이 얼마나 유지되는지 테스트하기 위한 벤치마크 데이터셋 구성의 근거가 됩니다.
    * **Environment Report:** 기상, 시간대, 노면 상태별 비중(%)을 계산하여 학습 데이터의 편향성을 수치화합니다.
    * **Label Density Analysis:** 영상당 평균 객체 수를 산출하여 데이터의 복잡도(Complexity)를 파악합니다.
    """
    if os.path.exists(DB_PATH):
        return {"message": "데이터베이스가 이미 존재합니다. 통합 과정을 건너뜁니다."}

    try:
        # [1] 데이터 로드 및 정규화
        selections_raw, odds_df, labels_df = load_all()
        sel_df = normalize_selections(selections_raw)
        all_ids = set(sel_df["video_id"])

        # --- STAGE 1: ODD Step (Metadata Integrity) ---
        merged_odds_df, missing_odd_ids, duplicate_odd_ids = merge_selections_with_odds(sel_df, odds_df)
        rejection_frames = []

        if len(missing_odd_ids) > 0:
            tmp = sel_df[sel_df["video_id"].isin(list(missing_odd_ids))].copy()
            tmp["reason"], tmp["stage"] = "missing_odd_metadata", "odd_tagging_step"
            rejection_frames.append(tmp)

        if len(duplicate_odd_ids) > 0:
            tmp = sel_df[sel_df["video_id"].isin(list(duplicate_odd_ids))].copy()
            tmp["reason"], tmp["stage"] = "duplicate_odd_metadata", "odd_tagging_step"
            rejection_frames.append(tmp)

        # --- STAGE 2: Labeling Integrity Check (Detailed Errors) ---
        error_map = {}
        for vid, group in labels_df.groupby("video_id"):
            reasons = []
            total_obj = group["obj_count"].sum()
            
            if total_obj == 0: reasons.append("zero_obj_count")
            if (group["obj_count"] < 0).any(): reasons.append("negative_obj_count")
            if (group["obj_count"] % 1 != 0).any(): reasons.append("non_integer_obj_count")
            if group.duplicated("object_class").any(): reasons.append("duplicate_label_class")
            
            if reasons:
                error_map[vid] = " & ".join(sorted(reasons))

        # C. Missing Label
        missing_label_ids = all_ids - set(labels_df["video_id"])
        if missing_label_ids:
            tmp = sel_df[sel_df["video_id"].isin(list(missing_label_ids))].copy()
            tmp["reason"], tmp["stage"] = "missing_label_data", "auto_labeling_step"
            rejection_frames.append(tmp)

        # D. Integrity Errors
        if error_map:
            tmp = sel_df[sel_df["video_id"].isin(error_map.keys())].copy()
            tmp["reason"] = tmp["video_id"].map(error_map)
            tmp["stage"] = "auto_labeling_step"
            rejection_frames.append(tmp)

        # --- STAGE 3: Final Aggregation & DB Store ---
        if rejection_frames:
            raw_rejections_df = pd.concat(rejection_frames, ignore_index=True)
            all_rejections_df = raw_rejections_df.groupby("video_id").agg({
                "reason": lambda x: " & ".join(sorted(set(" & ".join(x).split(" & ")))),
                "stage": "first",
                "raw_data": "first" if "raw_data" in raw_rejections_df.columns else lambda x: None
            }).reset_index()
        else:
            all_rejections_df = pd.DataFrame(columns=["video_id", "reason", "stage"])

        conn = sqlite3.connect(DB_PATH)
        all_rejections_df.to_sql("rejections", conn, if_exists="replace", index=False)
        
        rejected_ids = set(all_rejections_df["video_id"])
        # 최종 통합 (학습 효율을 위한 1:N 역정규화 구조)
        final_df = merged_odds_df[~merged_odds_df["video_id"].isin(rejected_ids)].merge(labels_df, on="video_id")
        final_df.to_sql("integrated_data", conn, if_exists="replace", index=False)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rej_reason ON rejections(reason)")
        conn.close()

        # --- STAGE 4: ML Pipeline Augmented Analysis ---
        unique_final_vids = final_df.drop_duplicates('video_id')
        total_vids_count = len(unique_final_vids)

        # 1. 클래스별 포함 영상 비율 (Sparsity & Background Bias Detection)
        class_presence = {}
        for cls in final_df['object_class'].unique():
            presence_count = final_df[final_df['object_class'] == cls]['video_id'].nunique()
            class_presence[cls] = f"{(presence_count / total_vids_count) * 100:.2f}%"

        # 2. 장면 복잡도 분포 (Scene Complexity Distribution)
        obj_counts_per_vid = final_df.groupby('video_id')['obj_count'].sum()
        complexity_stats = {
            "low_complexity (1-5 objects)": int((obj_counts_per_vid <= 5).sum()),
            "mid_complexity (6-15 objects)": int(((obj_counts_per_vid > 5) & (obj_counts_per_vid <= 15)).sum()),
            "high_complexity (16+ objects)": int((obj_counts_per_vid > 15).sum())
        }

        # 3. 환경 조합 교차 분석 (Critical Scenario Mapping)
        env_combos = unique_final_vids.groupby(['weather', 'time_of_day']).size().to_dict()
        formatted_combos = {f"{k[0]} | {k[1]}": v for k, v in env_combos.items()}

        # 4. 기본 분포 비율 (%)
        weather_pct = unique_final_vids['weather'].value_counts(normalize=True).mul(100).round(2).to_dict()
        tod_pct = unique_final_vids['time_of_day'].value_counts(normalize=True).mul(100).round(2).to_dict()

        return {
            "status": "success",
            "analysis_report": {
                "total_input_videos": len(sel_df),
                "integrated_videos": total_vids_count,
                "integration_rate": f"{(total_vids_count/len(sel_df))*100:.2f}%",
                "total_rejections": len(all_rejections_df),
                "rejection_by_stage": all_rejections_df["stage"].value_counts().to_dict() if not all_rejections_df.empty else {},
                "rejection_by_reason": all_rejections_df["reason"].value_counts().to_dict() if not all_rejections_df.empty else {},
                "statistical_report": {
                    "object_class_frequency": final_df['object_class'].value_counts().to_dict(),
                    "label_class_distribution": class_presence,
                    "scene_complexity_distribution": complexity_stats,
                    "environment_report": {
                        "weather_distribution": weather_pct,
                        "time_of_day_distribution": tod_pct,
                        "scenario_distribution": formatted_combos
                    },
                    "avg_labels_per_video": round(len(final_df) / total_vids_count, 2) if total_vids_count > 0 else 0
                }
            }
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

@app.get("/rejections", tags=["Pipeline"])
def get_rejections(
    stage: Optional[str] = Query(
        None, 
        description="특정 단계 필터 (미선택 시 전체 단계 조회)",
        openapi_examples={
            "All": {"summary": "전체 보기", "value": None},
            "Step 1: ODD": {"value": "odd_tagging_step"},
            "Step 2: Labeling": {"value": "auto_labeling_step"}
        }
    ),
    reason: Optional[str] = Query(
        None, 
        description="특정 사유 필터 (미선택 시 전체 사유 조회)",
        openapi_examples={
            "All": {"summary": "전체 보기", "value": None},
            "ODD: Missing Metadata": {"value": "missing_odd_metadata"},
            "ODD: Duplicate ID": {"value": "duplicate_odd_metadata"},
            "Label: Missing Data": {"value": "missing_label_data"},
            "Label: Zero Objects": {"value": "zero_obj_count"},
            "Label: Negative Count": {"value": "negative_obj_count"},
            "Label: Non-Integer": {"value": "non_integer_obj_count"},
            "Label: Duplicate Class": {"value": "duplicate_label_class"}
        }
    ), 
    page: int = Query(1, ge=1), 
    size: int = Query(50, ge=1, le=100)
):
    """
    ### 🛡️ 리젝션 통합 조회 (Waterfall 구조 적용)
    
    1단계(ODD) 결함 발견 시 2단계(Labeling) 검증을 건너뛰는 Waterfall 원칙이 적용되었습니다.
    이제 각 단계별 사유가 논리적으로 엄격히 분리되어 표시됩니다.
    """
    if not os.path.exists(DB_PATH):
        return {"status": "error", "message": "데이터베이스가 존재하지 않습니다."}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # --- [1] Overall Stats: 시스템 전체 현황 (항상 노출) ---
    cursor.execute("SELECT stage, reason FROM rejections")
    all_rows = cursor.fetchall()
    
    overall_stats = {
        "total_quarantined": len(all_rows),
        "by_stage": {"odd_tagging_step": 0, "auto_labeling_step": 0},
        "by_reason": {} # 7대 사유별 개별 카운트
    }
    
    for row in all_rows:
        s, r_str = row["stage"], row["reason"]
        overall_stats["by_stage"][s] = overall_stats["by_stage"].get(s, 0) + 1
        
        # 복합 사유가 발생하더라도(동일 단계 내 중복 오류) 개별 집계
        for r in [x.strip() for x in r_str.split("&")]:
            overall_stats["by_reason"][r] = overall_stats["by_reason"].get(r, 0) + 1

    # --- [2] Independent Filtering Logic (Smart Toggle) ---
    conditions = ["1=1"]
    params = []
    
    # Stage 단독 검색 가능
    if stage:
        conditions.append("stage = ?")
        params.append(stage)
    
    # Reason 단독 검색 가능 (복합 사유 포함)
    if reason:
        conditions.append("reason LIKE ?")
        params.append(f"%{reason}%")

    where_clause = " AND ".join(conditions)

    # --- [3] Pagination & Items ---
    cursor.execute(f"SELECT COUNT(*) FROM rejections WHERE {where_clause}", params)
    filtered_total = cursor.fetchone()[0]

    cursor.execute(f"SELECT * FROM rejections WHERE {where_clause} LIMIT ? OFFSET ?", params + [size, (page - 1) * size])
    
    items = []
    for row in cursor.fetchall():
        item = dict(row)
        if item.get("raw_data"):
            try: item["raw_data"] = json.loads(item["raw_data"])
            except: pass
        items.append(item)

    conn.close()
    
    return {
        "status": "success",
        "overall_stats": overall_stats,
        "metadata": {
            "filtered_total": filtered_total,
            "page": page,
            "size": size,
            "total_pages": (filtered_total + size - 1) // size if filtered_total > 0 else 0
        },
        "items": items
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