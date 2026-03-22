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
    * **Stage 2 (LABELING INTEGRITY):** LABEL 객체 수 음수(`negative_obj_count`), 실수(`non_integer_obj_count`), 클래스 중복(`duplicate_label_class`).
    
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
        os.remove(DB_PATH)

    try:
        selections_raw, odds_raw_df, labels_raw_df = load_all()
        sel_df = normalize_selections(selections_raw)
        
        current_survivor_ids = set(sel_df["video_id"])
        rejection_frames = []

        # --- STAGE 1: ODD Step (Waterfall) ---
        odds_ids = set(odds_raw_df["video_id"])
        
        missing_odd_ids = current_survivor_ids - odds_ids
        if missing_odd_ids:
            tmp = sel_df[sel_df["video_id"].isin(missing_odd_ids)].copy()
            tmp["reason"], tmp["stage"] = "missing_odd_metadata", "odd_tagging_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]])
            current_survivor_ids -= missing_odd_ids

        duplicate_odd_ids = set(odds_raw_df[odds_raw_df.duplicated("video_id")]["video_id"]) & current_survivor_ids
        if duplicate_odd_ids:
            tmp = sel_df[sel_df["video_id"].isin(duplicate_odd_ids)].copy()
            tmp["reason"], tmp["stage"] = "duplicate_odd_metadata", "odd_tagging_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]])
            current_survivor_ids -= duplicate_odd_ids

        # --- STAGE 2: Labeling Step ---
        labels_ids = set(labels_raw_df["video_id"])
        missing_label_ids = current_survivor_ids - labels_ids
        if missing_label_ids:
            tmp = sel_df[sel_df["video_id"].isin(missing_label_ids)].copy()
            tmp["reason"], tmp["stage"] = "missing_label_data", "auto_labeling_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]])
            current_survivor_ids -= missing_label_ids

        survivor_labels = labels_raw_df[labels_raw_df["video_id"].isin(current_survivor_ids)]
        
        clean_odds = odds_raw_df[odds_raw_df["video_id"].isin(current_survivor_ids)]
        merged_odd_sel = sel_df[sel_df["video_id"].isin(current_survivor_ids)].merge(clean_odds, on="video_id")
        
        # [중요] merge_with_labels 내부에서 이미 CSV의 labeled_at을 가져옵니다.
        integrated_df, label_stats = merge_with_labels(merged_odd_sel, survivor_labels)

        if label_stats["error_map"]:
            bad_vids = label_stats["error_map"].keys()
            tmp = sel_df[sel_df["video_id"].isin(bad_vids)].copy()
            tmp["reason"] = tmp["video_id"].map(label_stats["error_map"])
            tmp["stage"] = "auto_labeling_step"
            rejection_frames.append(tmp[["video_id", "stage", "reason", "raw_data"]])

        # --- STAGE 3: DB 적재 ---
        conn = sqlite3.connect(DB_PATH)
        
        if rejection_frames:
            all_rejections_df = pd.concat(rejection_frames, ignore_index=True)
            all_rejections_df.to_sql("rejections", conn, if_exists="replace", index=False)
        else:
            pd.DataFrame(columns=["video_id", "stage", "reason", "raw_data"]).to_sql("rejections", conn, if_exists="replace", index=False)

        # 2. Integrated Data 테이블
        if not integrated_df.empty:
            final_storage_df = integrated_df.drop(columns=['raw_data'], errors='ignore')

            # [FIX] 아래의 datetime.now() 관련 코드를 제거하여 
            # preprocessing.py에서 가져온 원본 labeled_at이 유지되도록 합니다.
            # (기존의 덮어쓰기 로직 삭제)

            final_storage_df.to_sql("integrated_data", conn, if_exists="replace", index=False)
        else:
            cols = [c for c in integrated_df.columns if c != 'raw_data']
            pd.DataFrame(columns=cols).to_sql("integrated_data", conn, if_exists="replace", index=False)

        conn.close()

        # --- STAGE 4: Statistical Report ---
        # ... (이하 동일) ...
        total_vids_count = len(integrated_df)
        
        count_cols = [c for c in integrated_df.columns if c.startswith("label_") and c.endswith("_count")]
        class_presence = {}
        if total_vids_count > 0:
            for col in count_cols:
                cls_name = col.replace("label_", "").replace("_count", "")
                presence_count = (integrated_df[col] > 0).sum()
                class_presence[cls_name] = f"{(presence_count / total_vids_count) * 100:.2f}%"

        obj_counts_per_vid = integrated_df[count_cols].sum(axis=1) if not integrated_df.empty else pd.Series()
        complexity_stats = {
            "low_complexity (1-5 objects)": int((obj_counts_per_vid <= 5).sum()),
            "mid_complexity (6-15 objects)": int(((obj_counts_per_vid > 5) & (obj_counts_per_vid <= 15)).sum()),
            "high_complexity (16+ objects)": int((obj_counts_per_vid > 15).sum())
        }

        weather_pct = integrated_df['weather'].value_counts(normalize=True).mul(100).round(2).to_dict() if not integrated_df.empty else {}
        tod_pct = integrated_df['time_of_day'].value_counts(normalize=True).mul(100).round(2).to_dict() if not integrated_df.empty else {}
        
        env_combos = integrated_df.groupby(['weather', 'time_of_day']).size().to_dict() if not integrated_df.empty else {}
        formatted_combos = {f"{k[0]} | {k[1]}": f"{(v/total_vids_count)*100:.2f}%" for k, v in env_combos.items()}

        return {
            "status": "success",
            "analysis_report": {
                "total_input_videos": len(sel_df),
                "integrated_videos": total_vids_count,
                "integration_rate": f"{(total_vids_count/len(sel_df))*100:.2f}%" if len(sel_df) > 0 else "0.00%",
                "total_rejections": len(all_rejections_df) if rejection_frames else 0,
                "rejection_by_stage": all_rejections_df["stage"].value_counts().to_dict() if rejection_frames else {},
                "rejection_by_reason": all_rejections_df["reason"].value_counts().to_dict() if rejection_frames else {},
                "statistical_report": {
                    "object_class_frequency": class_presence,
                    "label_class_distribution": class_presence,
                    "scene_complexity_distribution": complexity_stats,
                    "environment_report": {
                        "weather_distribution": weather_pct,
                        "time_of_day_distribution": tod_pct,
                        "scenario_distribution": formatted_combos
                    },
                    "avg_labels_per_video": round(obj_counts_per_vid.mean(), 2) if total_vids_count > 0 else 0
                }
            }
        }

    except Exception as e:
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
            "Label: Missing Label Data": {"value": "missing_label_data"},
            "Label: Negative Object Count": {"value": "negative_obj_count"},
            "Label: Non-Integer Object Count": {"value": "non_integer_obj_count"},
            "Label: Duplicate Object Class": {"value": "duplicate_label_class"}
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
    - reason (str, optional): 'missing_odd_metadata', 'negative_obj_count' 등 구체적 사유로 필터링.
    - page (int): 조회할 페이지 번호 (기본값: 1).
    - size (int): 페이지당 레코드 수 (기본값: 50).

    ## 3. Response
    - status (str): 요청 처리 성공 여부 (success/error).
    - rejection_stats (dict): 필터링 조건과 관계없는 전체 격리 데이터의 거시적 통계.
        * total_rejections: 파이프라인에서 제외된 총 영상 수.
        * by_stage: 각 단계(Odd & Labeling)에서 발생한 탈락 수.
        * by_reason: 각 사유별로 발생한 탈락 수.
    - filtered_rejections (dict): 현재 필터링된 결과에 대한 페이지네이션 정보 (filtered_total, total_pages 등).
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
            "missing_label_data": 0, 
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
        "rejection_stats": overall_stats,
        "filtered_rejections": {
            "filtered_total": filtered_total,
            "page": page,
            "size": size,
            "total_pages": (filtered_total + size - 1) // size if filtered_total > 0 else 0
        },
        "items": items
    }


@app.post("/search")
async def search_data(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=100),
    filters: dict = Body(..., openapi_examples={
        "Comprehensive_Video3_Master": {
            "summary": "Video #3 전수 컬럼 검색 (참조용)",
            "description": "Video #3의 모든 컬럼명을 노출하며, 정밀 일치와 범위 검색을 조합한 마스터 샘플입니다.",
            "value": {
                # 1. 정밀 일치 필터 (Equivalence Values)
                # ODD 및 센서 데이터 (DB 저장 타입인 0/1 정수 권장)
                "weather": "sunny",
                "time_of_day": "night",
                "road_surface": "dry",
                "wiper_on": 1, 
                "headlights_on": 1,

                # 2. 범위 및 샌드위치 필터 (Ranged Values)
                # 메타데이터 범위
                "video_id_min": 3,
                "video_id_max": 3,
                "id_min": 3,
                "id_max": 3,
                "wiper_level_min": 3,
                "wiper_level_max": 3,
                
                # 소수점 데이터 샌드위치 (부동 소수점 정밀도 대응)
                "temperature_celsius_min": 14.5,
                "temperature_celsius_max": 14.6,
                "temperature_fahrenheit_min": 58.1,
                "temperature_fahrenheit_max": 58.3,

                # 3. 언패킹된 레이블 개수 필터 (Object Counts)
                # 형식: label_{class}_min / label_{class}_max
                "label_car_min": 31,
                "label_car_max": 31,
                "label_pedestrian_min": 26,
                "label_pedestrian_max": 26,
                "label_traffic_sign_min": 11,
                "label_traffic_sign_max": 11,
                "label_truck_min": 0,
                "label_truck_max": 0,
                "label_bus_min": 0,
                "label_bus_max": 0,
                "label_cyclist_min": 0,
                "label_motorcycle_min": 0,
                "label_traffic_light_min": 0,

                # 4. 언패킹된 레이블 신뢰도 필터 (Confidence Ranges)
                # 형식: label_{class}_confidence_min / _max
                "label_car_confidence_min": 0.83,
                "label_car_confidence_max": 0.84,
                "label_pedestrian_confidence_min": 0.73,
                "label_pedestrian_confidence_max": 0.74,
                "label_traffic_sign_confidence_min": 0.90,
                "label_traffic_sign_confidence_max": 0.92,
                "label_truck_confidence_min": 0.94,
                "label_truck_confidence_max": 0.95,

                #5. [중요] KST 시간 범위 필터 
                "recorded_at_min": "2026-01-10T00:00:00",
                "recorded_at_max": "2026-01-12T23:59:59"
            }
        }
    })
):
    """
    ## `POST /search` 통합 데이터셋 필터 검색 (Search & Filtering):  정제·통합 완료된 데이터를 다양한 조건으로 필터링하여 조회할 수 있는 엔드포인트입니다.
    이 엔드포인트는 `/analyze` 프로세스를 거쳐 `integrated_data` 테이블에 적재된 데이터를 필터링하여 제공합니다.

    ### 1. 필터링 원칙 (Filtering Rules)
    모든 필터는 `AND` 조건으로 결합되며, 크게 세 가지 매칭 방식을 지원합니다.

    * **정밀 일치 (Direct Match)**: 
        - 대상: `weather`, `time_of_day`, `road_surface`, `headlights_on`, `wiper_on`
        - 사용법: 컬럼명과 값을 그대로 전달 (예: `"weather": "sunny"`)
        - 비고: `wiper_on` 같은 Boolean 성격의 필드는 `0` 또는 `1` 전송을 권장합니다.

    * **수치 범위 (Numeric Range)**:
        - 대상: `video_id`, `id`, `temperature_celsius`, `wiper_level` 및 모든 라벨 통계
        - 사용법: `{컬럼명}_min` 또는 `{컬럼명}_max` 접미사 사용 (예: `"video_id_min": 100`)
        - 부동 소수점(`temperature`)은 `_min`과 `_max`를 모두 사용하여 오차 범위를 잡는 **샌드위치 필터링**을 권장합니다.
        - 라벨 개수 필터는 `label_{class}_min` / `label_{class}_max` 형식으로, 신뢰도 필터는 `label_{class}_confidence_min` / `_max` 형식으로 전달합니다.
        - video_id_min = 3, video_id_max = 3과 같이 동일한 값을 min/max로 전달하면 해당 값과 정확히 일치하는 레코드만 조회됩니다. 이는 정밀 일치와 유사한 효과를 낼 수 있습니다.
        - 예시: `"label_car_min": 5`는 자동차가 최소 5개 이상인 영상 검색, `"label_pedestrian_confidence_min": 0.8`는 보행자 신뢰도가 최소 0.8 이상인 영상 검색.

    * **언패킹된 라벨 필터 (Unpacked Labels)**:
        - **객체 수**: `label_{class}_min` 형식 (예: `"label_car_min": 5`) -> `label_car_count` 컬럼 매핑
        - **신뢰도**: `label_{class}_confidence_min` 형식 (예: `"label_pedestrian_confidence_min": 0.8`)

    ### 2. 시간 데이터 검색 특화 (Time-Series Matching)
    `recorded_at` 및 `labeled_at` 필드는 SQLite의 `substr` 함수를 이용한 **유연한 부분 일치**를 지원합니다.
    - **동작 방식**: 사용자가 입력한 문자열 길이만큼 DB 값을 잘라서 비교합니다.
    - **예시**: 
        - `"recorded_at_min": "2026-03-15"` 입력 시 -> 2026년 3월 15일 00시 이후 데이터 전체 검색
        - `"recorded_at_min": "2026-03-15T09:00"` 입력 시 -> 해당 날짜 오전 9시 이후 데이터 검색

    ### 3. 페이지네이션 (Pagination)
    - `page`: 조회할 페이지 번호 (1부터 시작)
    - `size`: 페이지당 결과 수 (최대 100개)
    - Response에는 `total_found`가 포함되어 전체 검색 규모를 파악할 수 있습니다.
    
    ### 4. 요청 예시 (Request Sample)
    ```json
    {
      "weather": "sunny", "time_of_day": "night",
      "video_id_min": 3, "video_id_max": 3,
      "temperature_celsius_min": 14.5, "temperature_celsius_max": 14.6,
      "label_car_min": 31, "label_car_confidence_min": 0.83,
      "recorded_at_min": "2026-01-10T00:00:00"
    }
    ```

    ### 5. 응답 예시 (Response Sample)
    ```json
    {
      "status": "success",
      "pagination": { "page": 1, "size": 50, "total_found": 1 },
      "results": [
        {
          "video_id": 3, "weather": "sunny", "time_of_day": "night",
          "temperature_celsius": 14.55, "wiper_on": 1,
          "labels": { "car": { "count": 31, "avg_confidence": 0.831 } },
          "label_car_count": 31, "label_car_confidence": 0.831,
          "recorded_at": "2026-01-10T19:44:32+0900"
        }
      ]
    }
    ```
    """
   
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=503, detail="DB 미초기화")

    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    where_clauses = ["1=1"]; params = []

    for key, val in filters.items():
        if val is None or val == "": continue

        # 1. 직접 일치 (Direct Match)
        direct_cols = ["weather", "time_of_day", "road_surface", "headlights_on", "wiper_on"]
        if key in direct_cols:
            # DB가 0/1 정수형이므로 불리언/문자열을 정수로 변환
            clean_val = int(val) if isinstance(val, (bool, int)) or (isinstance(val, str) and val.isdigit()) else val
            where_clauses.append(f"{key} = ?"); params.append(clean_val)

        # 2. 범위/샌드위치 매칭 (Range Match)
        elif key.endswith(("_min", "_max")):
            is_min = key.endswith("_min")
            base_key = key.replace("_min", "").replace("_max", "")
            
            # 컬럼명 결정 (기존 로직 유지)
            target_col = base_key
            if base_key.startswith("label_"):
                target_col = base_key if "confidence" in base_key else f"{base_key}_count"

            # [시간 검색 핵심 수정]
            if target_col in ["recorded_at", "labeled_at"] and isinstance(val, str):
                # 1. 입력값 정규화 (공백을 T로 치환)
                clean_val = val.replace(" ", "T")
                
                # 2. 입력값의 길이를 계산 (예: '2026-01-11T06:58:00' 이면 19자)
                val_len = len(clean_val)
                
                # 3. DB의 데이터도 동일한 길이만큼 잘라서 비교 (substr 사용)
                # 이렇게 하면 DB의 '+0900' 부분을 무시하고 앞부분 시간만 비교합니다.
                op = ">=" if is_min else "<="
                where_clauses.append(f"substr({target_col}, 1, {val_len}) {op} ?")
                params.append(clean_val)
                
            else:
                # 일반 숫자형 데이터 처리
                op = ">=" if is_min else "<="
                where_clauses.append(f"{target_col} {op} ?")
                params.append(val)
    # 쿼리 조립 및 실행
    where_stmt = " AND ".join(where_clauses)
    count_query = f"SELECT COUNT(*) FROM integrated_data WHERE {where_stmt}"
    cursor.execute(count_query, params)
    total_found = cursor.fetchone()[0]

    offset = (page - 1) * size
    search_query = f"SELECT * FROM integrated_data WHERE {where_stmt} LIMIT ? OFFSET ?"
    cursor.execute(search_query, params + [size, offset])
    
    rows = [dict(row) for row in cursor.fetchall()]
    for r in rows:
        if r.get("labels"): r["labels"] = json.loads(r["labels"])
    
    conn.close()
    return {"status": "success", "pagination": {"page": page, "size": size, "total_found": total_found}, "results": rows}

# @app.get("/joined_data", tags=["View"])
# def get_joined_data():
#     """### 📂 통합 데이터 미리보기 (Top 50)"""
#     if not os.path.exists(DB_PATH): raise HTTPException(status_code=503, detail="DB 초기화 필요")
#     conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM integrated_data LIMIT 50")
#     rows = [dict(row) for row in cursor.fetchall()]
#     for r in rows:
#         if r.get("labels"): r["labels"] = json.loads(r["labels"])
#     conn.close()
#     return {"count": len(rows), "data": rows}