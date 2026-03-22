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

@app.post("/analyze")
async def analyze():
    """
    ## `POST /analyze` 데이터 분석 및 DB 적재: 이 엔드포인트는 원본 데이터셋을 전수 조사하여 학습 가용 데이터를 추출하고 품질 미달 및 오류 데이터를 격리합니다.

    ## 1. RESPONSE

    **1-1. Status:** 데이터 처리의 성공 여부를 나타내는 필드입니다. "success" 또는 "error"로 반환됩니다.
    
    **1-2. 데이터 통합 및 정제 요약 (Processing Summary):** 전체 입력 데이터 중 결함을 걸러내고 최종 학습에 투입 가능한 유효 데이터의 총량과 정제 효율을 정량적으로 증명하는 지표입니다.
    * **Total Input Videos:** 원본 데이터셋에 존재하는 총 영상 수입니다.
    * **Integrated Videos:** ODD와 LABELING데이터가 모두 존재하고, 데이터 무결성 검사를 통과한 최종 학습용 영상 수입니다.
    * **Integration Rate:** 전체 입력 대비 최종 통합된 영상의 비율로, 데이터 품질과 정제 효율을 나타냅니다.
    * **Total Rejections:** ODD 매칭 실패, LABELING 누락, 객체 수 오류 등으로 거절된 영상의 총 수입니다.

    **1-3. 단계별 거절(Rejection by Stage):** 각 처리 단계(ODD 매칭, 라벨링 검증)별로 거절된 영상 수를 집계하여 어느 단계에서 문제가 발생하는지 파악합니다.
    * 모든 거절 데이터는 발생 지점에 따라 `odd_tagging_step` 또는 `auto_labeling_step`으로 분류되어 기록됩니다.

    **1-4. 사유별 거절(Rejection By Reason):** 거절 사유별로 집계하여 어떤 유형의 오류가 가장 빈번한지 분석합니다.
    * **Stage 1 (ODD DATA):** ODD 데이터 누락(`missing_odd_metadata`).
    * **Stage 1 (ODD INTEGRITY):** ODD ID 중복(`duplicate_odd_metadata`).
    * **Stage 2 (LABELING DATA):** LABEL 파일 누락(`missing_label_data`).
    * **Stage 2 (LABELING INTEGRITY):** LABEL 객체 수 0(`zero_obj_count`), 음수(`negative_obj_count`), 실수(`non_integer_obj_count`), 클래스 중복(`duplicate_label_class`).
    
    **1-5. 통계 분석 (Statistical Report):** 최종 통합된 데이터셋에 대한 통계 분석을 통해 학습 데이터의 특성과 편향성을 파악합니다.
    * **Object Class Frequency:** 각 객체 클래스(예: 자동차, 보행자 등)가 전체 영상에서 얼마나 자주 등장하는지 분석하여 클래스 불균형 문제를 탐지합니다.
    * **Label Class Distribution:** 각 객체 클래스가 전체 영상 중 몇 퍼센트의 영상에 출현하는지 분석합니다. 특정 배경에만 객체가 편중되어 학습되는 '배경 편향성'을 탐지하는 데 사용됩니다.
    * **Scene Complexity Distribution:** 영상 내 총 객체 수를 기준으로 저/중/고밀도 상황을 분류합니다. 모델이 혼잡한 환경에서 성능이 얼마나 유지되는지 테스트하기 위한 벤치마크 데이터셋 구성의 근거가 됩니다.
    * **Environment Report:** 기상, 시간대, 노면 상태별 비중(%)을 계산하여 학습 데이터의 편향성을 수치화합니다.
        - weather_distribution: 맑음, 비, 눈 등 다양한 기상 조건이 학습 데이터에 어떻게 분포되어 있는지 분석합니다.
        - time_of_day_distribution: 낮, 밤 등 시간대별로 학습 데이터가 어떻게 분포되어 있는지 분석합니다.
        - scenario_distribution: 기상과 시간대의 조합별로 학습 데이터가 어떻게 분포되어 있는지 분석합니다. 예를 들어, '맑은 날의 낮'과 '비 오는 날의 밤'이 각각 전체 데이터에서 몇 퍼센트를 차지하는지 분석하여 모델이 다양한 시나리오에서 학습될 수 있도록 합니다.
    * **Label Density Analysis (avg_labels_per_video):** 영상당 평균 객체 수를 산출하여 데이터의 복잡도(Complexity)를 파악합니다.
    """
    if os.path.exists(DB_PATH):
        return {"message": "데이터베이스가 이미 존재합니다."}

    try:
        selections_raw, odds_df, labels_df = load_all()
        sel_df = normalize_selections(selections_raw)
        
        # Waterfall 제어를 위한 생존자 명단
        current_survivor_ids = set(sel_df["video_id"])
        rejection_frames = []

        # --- STAGE 1: ODD Step ---
        odds_ids = set(odds_df["video_id"])
        
        # 1-A. Missing ODD
        missing_odd_ids = current_survivor_ids - odds_ids
        if missing_odd_ids:
            tmp = sel_df[sel_df["video_id"].isin(missing_odd_ids)].copy()
            tmp["reason"], tmp["stage"] = "missing_odd_metadata", "odd_tagging_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]]) # 필요한 컬럼만 추출
            current_survivor_ids -= missing_odd_ids

        # 1-B. Duplicate ODD
        duplicate_odd_ids = set(odds_df[odds_df.duplicated("video_id")]["video_id"]) & current_survivor_ids
        if duplicate_odd_ids:
            tmp = sel_df[sel_df["video_id"].isin(duplicate_odd_ids)].copy()
            tmp["reason"], tmp["stage"] = "duplicate_odd_metadata", "odd_tagging_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]])
            current_survivor_ids -= duplicate_odd_ids

        # --- STAGE 2: Labeling Step ---
        # 2-A. Missing Label
        labels_ids = set(labels_df["video_id"])
        missing_label_ids = current_survivor_ids - labels_ids
        if missing_label_ids:
            tmp = sel_df[sel_df["video_id"].isin(missing_label_ids)].copy()
            tmp["reason"], tmp["stage"] = "missing_label_data", "auto_labeling_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]])
            current_survivor_ids -= missing_label_ids

        # 2-B. Labeling Integrity
        survivor_labels = labels_df[labels_df["video_id"].isin(current_survivor_ids)]
        error_map = {}
        for vid, group in survivor_labels.groupby("video_id"):
            reasons = []
            if group["obj_count"].sum() == 0: reasons.append("zero_obj_count")
            if (group["obj_count"] < 0).any(): reasons.append("negative_obj_count")
            if (group["obj_count"] % 1 != 0).any(): reasons.append("non_integer_obj_count")
            if group.duplicated("object_class").any(): reasons.append("duplicate_label_class")
            
            if reasons:
                error_map[vid] = " & ".join(sorted(reasons))

        if error_map:
            tmp = sel_df[sel_df["video_id"].isin(error_map.keys())].copy()
            tmp["reason"] = tmp["video_id"].map(error_map)
            tmp["stage"] = "auto_labeling_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]])
            current_survivor_ids -= set(error_map.keys())

        # --- STAGE 3: DB 적재 (Minimal Columns) ---
        conn = sqlite3.connect(DB_PATH)
        
        if rejection_frames:
            # Waterfall에 의해 중복이 제거되었으므로 단순 병합
            all_rejections_df = pd.concat(rejection_frames, ignore_index=True)
            # 최종 확인: video_id, stage, reason, raw_data 4개 컬럼만 저장
            all_rejections_df.to_sql("rejections", conn, if_exists="replace", index=False)
        else:
            # 데이터가 없을 경우 스키마만 생성
            pd.DataFrame(columns=["video_id", "stage", "reason", "raw_data"]).to_sql("rejections", conn, if_exists="replace", index=False)

        # Integrated Data 저장 (ODD + Labels 합본)
        final_df = odds_df[odds_df["video_id"].isin(current_survivor_ids)].merge(labels_df, on="video_id")
        final_df.to_sql("integrated_data", conn, if_exists="replace", index=False)
        
        conn.close()

        # --- STAGE 4: ML Pipeline Augmented Analysis ---
        unique_final_vids = final_df.drop_duplicates('video_id')
        total_vids_count = len(unique_final_vids)

        class_presence = {}
        if total_vids_count > 0:
            for cls in final_df['object_class'].unique():
                presence_count = final_df[final_df['object_class'] == cls]['video_id'].nunique()
                class_presence[cls] = f"{(presence_count / total_vids_count) * 100:.2f}%"

        obj_counts_per_vid = final_df.groupby('video_id')['obj_count'].sum()
        complexity_stats = {
            "low_complexity (1-5 objects)": int((obj_counts_per_vid <= 5).sum()),
            "mid_complexity (6-15 objects)": int(((obj_counts_per_vid > 5) & (obj_counts_per_vid <= 15)).sum()),
            "high_complexity (16+ objects)": int((obj_counts_per_vid > 15).sum())
        }

        env_combos = unique_final_vids.groupby(['weather', 'time_of_day']).size().to_dict()
        formatted_combos = {f"{k[0]} | {k[1]}": v for k, v in env_combos.items()}

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
        description="특정 단계 필터 (미선택 시 전체 단계)",
        openapi_examples={
            "All": {"summary": "전체 단계", "value": None},
            "Step 1: ODD": {"value": "odd_tagging_step"},
            "Step 2: Labeling": {"value": "auto_labeling_step"}
        }
    ),
    reason: Optional[str] = Query(
        None, 
        description="7가지 사유 중 필터 (미선택 시 전체 사유)",
        openapi_examples={
            "All": {"summary": "전체 사유", "value": None},
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
    ## `GET /rejections` 정제 과정에서 거부된 데이터 목록을 조회합니다. 거부 사유**와** 문제가 발생한 단계로 필터링할 수 있으며, 페이지네이션을 통해 대규모 거부 데이터셋을 효율적으로 탐색할 수 있습니다.
    ## 1. Feature: Multidimensional Filtering
    - 특정 검증 단계(stage)나 세부 결함 사유(Reason)를 조합하여 조회가 가능합니다.
    - 페이지네이션을 통해 대규모 거부 데이터셋을 효율적으로 탐색할 수 있습니다.
    - 전체 거절 항목 조회시, 전체 단계, 전체 사유를 선택해야합니다

    ## 2. Arguments
    - stage (str, optional): 'odd_tagging_step' 또는 'auto_labeling_step'으로 필터링.
    - reason (str, optional): 'missing_odd_metadata', 'zero_obj_count' 등 구체적 사유로 필터링.
    - page (int): 조회할 페이지 번호 (기본값: 1).
    - size (int): 페이지당 레코드 수 (기본값: 50).

    ## 3. Response
    - status (str): 요청 처리 성공 여부 (success/error).
    - overall_stats (dict): 필터링 조건과 관계없는 전체 격리 데이터의 거시적 통계.
        * total_rejections: 파이프라인에서 제외된 총 영상 수.
        * by_stage: 각 단계(Stage 1 & 2)에서 발생한 탈락 수.
        * by_reason: 세부 결함 사유별 발생 빈도(Distribution).
    - metadata (dict): 현재 필터링된 결과에 대한 페이지네이션 정보 (filtered_total, total_pages 등).
    - items (list): 거절된 데이터의 상세 객체 리스트.
        * video_id: 영상 식별자.
        * stage: 거절 단계 
        * reason: 거절 상세 사유.
        * raw_data: 문제 발생 당시의 원본 selections JSON/CSV 레코드 보존 데이터.
    """
    if not os.path.exists(DB_PATH):
        return {"status": "error", "message": "DB 파일이 없습니다."}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Overall Stats (전체 대시보드 현황)
    cursor.execute("SELECT stage, reason FROM rejections")
    all_rejections = cursor.fetchall()
    
    overall_stats = {
        "total_rejections": len(all_rejections),
        "by_stage": {"odd_tagging_step": 0, "auto_labeling_step": 0},
        "by_reason": {
            "missing_odd_metadata": 0, "duplicate_odd_metadata": 0,
            "missing_label_data": 0, "zero_obj_count": 0,
            "negative_obj_count": 0, "non_integer_obj_count": 0,
            "duplicate_label_class": 0
        }
    }
    
    for row in all_rejections:
        overall_stats["by_stage"][row["stage"]] = overall_stats["by_stage"].get(row["stage"], 0) + 1
        for r in [x.strip() for x in row["reason"].split("&")]:
            if r in overall_stats["by_reason"]:
                overall_stats["by_reason"][r] += 1

    # 2. 독립 검색 및 필터 로직 구성
    conditions = ["1=1"]
    params = []
    
    if stage:
        conditions.append("stage = ?")
        params.append(stage)
    
    if reason:
        # % 기호를 포함하여 LIKE 검색 수행 (복합 사유 대응)
        conditions.append("reason LIKE ?")
        params.append(f"%{reason.strip()}%")

    where_clause = " AND ".join(conditions)

    # 3. 데이터 조회 (Count와 Items의 일치 확인)
    cursor.execute(f"SELECT COUNT(*) FROM rejections WHERE {where_clause}", params)
    filtered_total = cursor.fetchone()[0]

    # 중요: LIMIT/OFFSET 파라미터는 검색 조건(params) 뒤에 추가되어야 함
    final_query = f"SELECT * FROM rejections WHERE {where_clause} LIMIT ? OFFSET ?"
    paging_params = params + [size, (page - 1) * size]
    
    cursor.execute(final_query, paging_params)
    rows = cursor.fetchall()
    
    items = []
    for row in rows:
        item = dict(row)
        if item.get("raw_data"):
            try:
                item["raw_data"] = json.loads(item["raw_data"])
            except:
                pass
        items.append(item)

    conn.close()
    
    return {
        "status": "success",
        "overall_stats": overall_stats,
        "filtered_rejections": {
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