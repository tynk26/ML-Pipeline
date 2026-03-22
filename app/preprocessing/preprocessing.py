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
    라벨 데이터를 언패킹하고 CSV의 labeled_at을 보존하며 병합합니다.
    - zero_obj_count 검증 로직을 제거하여 0개 객체 데이터도 허용합니다.
    """
    # 1. Rejection Logic (0개 카운트 제외)
    
    # 1-A. Duplicate Label Class
    dup_mask = labels_df.duplicated(subset=["video_id", "object_class"], keep=False)
    dup_vids = labels_df[dup_mask]["video_id"].unique()
    dup_err_map = {vid: "duplicate_label_class" for vid in dup_vids}

    # 1-B. Negative Object Count (음수는 여전히 오류로 처리)
    negative_mask = (labels_df["obj_count"] < 0)
    neg_vids = labels_df[negative_mask]["video_id"].unique()
    neg_err_map = {vid: "negative_obj_count" for vid in neg_vids}

    # 1-C. Non-integer Object Count
    non_int_mask = (labels_df["obj_count"] % 1 != 0)
    non_int_vids = labels_df[non_int_mask]["video_id"].unique()
    non_int_err_map = {vid: "non_integer_obj_count" for vid in non_int_vids}

    # 1-D. Invalid Object Class
    invalid_class_mask = (
        labels_df["object_class"].isna() | 
        (labels_df["object_class"].astype(str).str.strip() == "") |
        (labels_df["object_class"].astype(str).str.lower().isin(["unknown", "null"]))
    )
    invalid_class_vids = labels_df[invalid_class_mask]["video_id"].unique()
    class_err_map = {vid: "invalid_object_class" for vid in invalid_class_vids}

    # 에러 맵 통합 (zero_obj_count 관련 로직 삭제됨)
    error_map = {
        **class_err_map,
        **non_int_err_map,
        **neg_err_map,
        **dup_err_map
    }
    
    bad_vids = set(error_map.keys())

    # 2. Cleaning
    clean_labels_df = labels_df[~labels_df["video_id"].isin(bad_vids)].copy()
    clean_labels_df["obj_count"] = clean_labels_df["obj_count"].astype(int)
    
    # CSV 원본 시간 보존용 맵핑
    csv_time_map = clean_labels_df.groupby("video_id")["labeled_at"].first().reset_index()
    
    # 3. Pivot 및 정렬
    counts_pivot = clean_labels_df.pivot(index="video_id", columns="object_class", values="obj_count").fillna(0).astype(int)
    conf_pivot = clean_labels_df.pivot(index="video_id", columns="object_class", values="avg_confidence")

    unique_objects = sorted(clean_labels_df["object_class"].unique())
    ordered_label_cols = []
    
    for obj in unique_objects:
        count_col = f"label_{obj}_count"
        conf_col = f"label_{obj}_confidence"
        counts_pivot.rename(columns={obj: count_col}, inplace=True)
        conf_pivot.rename(columns={obj: conf_col}, inplace=True)
        ordered_label_cols.extend([count_col, conf_col])

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
    
    # labeled_at 복구 (CSV 원본 값 유지)
    if "labeled_at" in final_df.columns:
        final_df = final_df.drop(columns=["labeled_at"])
    final_df = final_df.merge(csv_time_map, on="video_id", how="left")
    
    missing_ids = set(merged_df["video_id"]) - set(labels_summary["video_id"])
    stats = {
        "missing_ids": list(missing_ids - bad_vids),
        "error_map": error_map 
    }
    
    final_df["labels"] = final_df["labels"].apply(json.dumps)
    return final_df, stats

def safe_json(df: pd.DataFrame):
    return df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")