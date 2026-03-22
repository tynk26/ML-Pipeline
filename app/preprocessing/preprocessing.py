import pandas as pd
import numpy as np
import json
from typing import Tuple, List, Dict

def f_to_c(f):
    return (f - 32) * 5.0 / 9.0 if f is not None else None

def c_to_f(c):
    return (c * 9.0 / 5.0) + 32 if c is not None else None

def normalize_selections(selections: list) -> pd.DataFrame:
    records = []
    for item in selections:
        record = {
            "video_id": item.get("id"),
            "recorded_at": item.get("recordedAt"),
            "source_path": item.get("sourcePath"),
            "raw_data": json.dumps(item) 
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
    temp_dt = pd.to_datetime(df["recorded_at"], errors="coerce")
    
    df["recorded_at"] = temp_dt.apply(
        lambda x: x.strftime('%Y-%m-%dT%H:%M:%S%z') if pd.notnull(x) else None
    )
    df["wiper_on"] = df["wiper_on"].astype("boolean")
    df["headlights_on"] = df["headlights_on"].astype("boolean")
    df["wiper_level"] = df["wiper_level"].fillna(0).astype(int)
    return df

def merge_selections_with_odds(selections_df, odds_df) -> Tuple[pd.DataFrame, list, list]:
    duplicate_mask = odds_df.duplicated(subset=["video_id"], keep=False)
    duplicate_odd_vids = odds_df[duplicate_mask]["video_id"].unique().tolist()
    
    selection_ids = set(selections_df["video_id"])
    odd_ids = set(odds_df["video_id"])
    missing_odd_ids = list(selection_ids - odd_ids)
    
    clean_odds_df = odds_df[~odds_df["video_id"].isin(duplicate_odd_vids)]
    merged_df = selections_df.merge(clean_odds_df, on="video_id", how="inner")
    
    return merged_df, missing_odd_ids, duplicate_odd_vids

def merge_with_labels(merged_df, labels_df) -> Tuple[pd.DataFrame, Dict]:
    """
    라벨 데이터를 언패킹하고 알파벳 순으로 정렬합니다.
    - 정렬 순서: label_{object}_count, label_{object}_confidence 순서
    """
    # 1. Rejection Logic (기존 유지)
    invalid_class_mask = (
        labels_df["object_class"].isna() | 
        (labels_df["object_class"].astype(str).str.strip() == "") |
        (labels_df["object_class"].astype(str).str.lower().isin(["unknown", "null"]))
    )
    invalid_class_vids = labels_df[invalid_class_mask]["video_id"].unique()
    class_err_map = {vid: "invalid_object_class" for vid in invalid_class_vids}

    non_int_mask = (labels_df["obj_count"] % 1 != 0)
    non_int_vids = labels_df[non_int_mask]["video_id"].unique()
    non_int_err_map = {vid: "non_integer_obj_count" for vid in non_int_vids}

    negative_mask = (labels_df["obj_count"] < 0)
    neg_vids = labels_df[negative_mask]["video_id"].unique()
    neg_err_map = {vid: "negative_obj_count" for vid in neg_vids}

    dup_mask = labels_df.duplicated(subset=["video_id", "object_class"], keep=False)
    dup_vids = labels_df[dup_mask]["video_id"].unique()
    dup_err_map = {vid: "duplicate_label" for vid in dup_vids}

    bad_vids = set(invalid_class_vids) | set(non_int_vids) | set(neg_vids) | set(dup_vids)
    error_map = {**dup_err_map, **neg_err_map, **non_int_err_map, **class_err_map}

    # 2. Cleaning
    clean_labels_df = labels_df[~labels_df["video_id"].isin(bad_vids)].copy()
    clean_labels_df["obj_count"] = clean_labels_df["obj_count"].astype(int)
    
    # 3. Pivot 및 정렬 로직
    # 카운트와 신뢰도를 각각 피벗
    counts_pivot = clean_labels_df.pivot(index="video_id", columns="object_class", values="obj_count").fillna(0).astype(int)
    conf_pivot = clean_labels_df.pivot(index="video_id", columns="object_class", values="avg_confidence")

    # 컬럼을 알파벳 순으로 생성 (Count -> Confidence 쌍)
    unique_objects = sorted(clean_labels_df["object_class"].unique())
    ordered_label_cols = []
    
    for obj in unique_objects:
        count_col = f"label_{obj}_count"
        conf_col = f"label_{obj}_confidence"
        
        # 실제 컬럼명 변경 적용
        counts_pivot.rename(columns={obj: count_col}, inplace=True)
        conf_pivot.rename(columns={obj: conf_col}, inplace=True)
        
        # 순서 리스트에 추가
        ordered_label_cols.extend([count_col, conf_col])

    # 피벗 결과 병합 및 컬럼 순서 재배치
    combined_pivot = pd.concat([counts_pivot, conf_pivot], axis=1)
    combined_pivot = combined_pivot[ordered_label_cols]
    
    # JSON 요약 필드 생성
    labels_summary = clean_labels_df.groupby("video_id").apply(
        lambda x: {row.object_class: {"count": row.obj_count, "avg_confidence": row.avg_confidence} 
                   for row in x.itertuples()},
        include_groups=False
    ).reset_index(name="labels")
    
    # 4. 최종 병합
    final_df = merged_df.merge(labels_summary, on="video_id", how="inner")
    final_df = final_df.merge(combined_pivot.reset_index(), on="video_id", how="left")
    
    missing_ids = set(merged_df["video_id"]) - set(labels_summary["video_id"])
    stats = {
        "missing_ids": list(missing_ids - bad_vids),
        "error_map": error_map 
    }
    
    final_df["labels"] = final_df["labels"].apply(json.dumps)
    return final_df, stats

def safe_json(df: pd.DataFrame):
    return df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")