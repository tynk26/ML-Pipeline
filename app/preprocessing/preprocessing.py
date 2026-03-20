import pandas as pd
import numpy as np
import json
from typing import Tuple, List, Dict

# ... f_to_c, c_to_f 함수는 기존과 동일 ...

def f_to_c(f):
    return (f - 32) * 5.0 / 9.0 if f is not None else None

def c_to_f(c):
    return (c * 9.0 / 5.0) + 32 if c is not None else None
def normalize_selections(selections: list) -> pd.DataFrame:
    # 기존 로직 유지
    records = []
    for item in selections:
        record = {
            "video_id": item.get("id"),
            "recorded_at": item.get("recordedAt"),
            "source_path": item.get("sourcePath"),
            "raw_data": json.dumps(item) # Rejection 시 원본 확인을 위해 추가 권장
        }
        sensor = item.get("sensor")
        if sensor:
            temp_f = sensor.get("temperature", {}).get("value")
            record.update({
                "temperature_fahrenheit": temp_f,
                "temperature_celsius": f_to_c(temp_f),
                "wiper_on": sensor.get("wiper", {}).get("isActive"),
                "wiper_level": sensor.get("wiper", {}).get("level"),
                "headlights_on": sensor.get("headlights")
            })
        else:
            temp_c = item.get("temperature")
            record.update({
                "temperature_celsius": temp_c,
                "temperature_fahrenheit": c_to_f(temp_c),
                "wiper_on": item.get("isWiperOn"),
                "wiper_level": 0,
                "headlights_on": item.get("headlightsOn")
            })
        records.append(record)

    df = pd.DataFrame(records)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce", utc=True)
    df["wiper_on"] = df["wiper_on"].astype("boolean")
    df["headlights_on"] = df["headlights_on"].astype("boolean")
    df["wiper_level"] = df["wiper_level"].fillna(0).astype(int)
    return df

def merge_selections_with_odds(selections_df, odds_df) -> Tuple[pd.DataFrame, List[int]]:
    # 기존 로직 유지 (Inner Join)
    selections_df["video_id"] = selections_df["video_id"].astype(int)
    odds_df["video_id"] = odds_df["video_id"].astype(int)
    
    existing_ids = set(odds_df["video_id"].unique())
    all_ids = set(selections_df["video_id"].unique())
    rejected_ids = sorted(all_ids - existing_ids)
    
    merged = pd.merge(selections_df, odds_df, on="video_id", how="inner")
    return merged, rejected_ids

def merge_with_labels(merged_df, labels_df) -> Tuple[pd.DataFrame, Dict]:
    labels_df["video_id"] = labels_df["video_id"].astype(int)
    
    # 1. 동적 플래트닝을 위한 준비: 모든 유니크한 객체 클래스 추출
    all_classes = labels_df["object_class"].unique().tolist()
    
    # 2. 피벗 테이블 생성 (video_id별 각 클래스의 count를 컬럼으로)
    # fillna(0)를 통해 해당 비디오에 없는 객체는 0으로 처리
    label_counts_pivot = labels_df.pivot_table(
        index="video_id", 
        columns="object_class", 
        values="obj_count", 
        aggfunc="first"
    ).fillna(0).astype(int)
    
    # 컬럼명 명확화 (예: car -> label_car_count)
    label_counts_pivot.columns = [f"label_{col}_count" for col in label_counts_pivot.columns]
    label_counts_pivot = label_counts_pivot.reset_index()

    # 3. 기존의 dict 구조(labels 컬럼)도 유지 (상세 정보용)
    labels_dict_pivot = labels_df.groupby("video_id").apply(
        lambda x: {row.object_class: {"count": row.obj_count, "avg_confidence": row.avg_confidence} 
                   for row in x.itertuples()}
    ).reset_index(name="labels")
    
    # 4. 최종 병합 (Merged + Counts Pivot + Dict Pivot)
    final_df = pd.merge(merged_df, labels_dict_pivot, on="video_id", how="inner")
    final_df = pd.merge(final_df, label_counts_pivot, on="video_id", how="left")
    
    # 병합 후 NaN이 된 카운트 컬럼들은 0으로 채움
    count_cols = [c for c in final_df.columns if c.startswith("label_") and c.endswith("_count")]
    final_df[count_cols] = final_df[count_cols].fillna(0).astype(int)

    stats = {"join_missing": sorted(set(merged_df["video_id"]) - set(labels_dict_pivot["video_id"]))}
    
    # SQL 저장을 위해 dict 타입은 string으로 변환
    final_df["labels"] = final_df["labels"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    
    return final_df, stats

def safe_json(df: pd.DataFrame):
    return df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")