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
    # --- STEP 1: AGGREGATE DUPLICATES (Handling your concern) ---
    # video_id와 object_class가 같은 행이 여러 개일 경우 하나로 합칩니다.
    # Count는 합산(sum), Confidence는 평균(mean)을 내는 것이 일반적입니다.
    labels_df = labels_df.groupby(["video_id", "object_class"]).agg({
        "obj_count": "sum",
        "avg_confidence": "mean"
    }).reset_index()

    # --- STEP 2: NOISE DETECTION (Negative Counts) ---
    negative_rows = labels_df[labels_df["obj_count"] < 0].copy()
    invalid_vids = negative_rows["video_id"].unique()
    
    neg_reasons = negative_rows.groupby("video_id").apply(
        lambda x: f"INVALID_NEGATIVE_COUNT: {', '.join(x['object_class'].unique())}",
        include_groups=False
    ).to_dict()

    # --- STEP 3: CLEANING ---
    clean_labels_df = labels_df[~labels_df["video_id"].isin(invalid_vids)].copy()
    
    # --- STEP 4: PIVOTING (Now safe from "Duplicate Entry" errors) ---
    counts_pivot = clean_labels_df.pivot(
        index="video_id", 
        columns="object_class", 
        values="obj_count"
    ).fillna(0).astype(int)
    counts_pivot.columns = [f"label_{col}_count" for col in counts_pivot.columns]
    
    # --- STEP 5: NESTED JSON CREATION ---
    labels_pivot = clean_labels_df.groupby("video_id").apply(
        lambda x: {row.object_class: {"count": row.obj_count, "avg_confidence": row.avg_confidence} 
                   for row in x.itertuples()},
        include_groups=False
    ).reset_index(name="labels")
    
    # --- STEP 6: FINAL MERGE ---
    final_df = merged_df.merge(labels_pivot, on="video_id", how="inner")
    final_df = final_df.merge(counts_pivot.reset_index(), on="video_id", how="left")
    
    # Fill any NaNs in the count columns
    count_cols = [c for c in final_df.columns if c.startswith("label_") and c.endswith("_count")]
    final_df[count_cols] = final_df[count_cols].fillna(0).astype(int)

    # --- STEP 7: STATS FOR REJECTIONS ---
    missing_ids = set(merged_df["video_id"]) - set(labels_pivot["video_id"])
    stats = {
        "missing_ids": list(missing_ids - set(invalid_vids)),
        "negative_count_map": neg_reasons
    }
    
    # Vectorized JSON string conversion for SQL
    final_df["labels"] = final_df["labels"].apply(json.dumps)
    
    return final_df, stats

def safe_json(df: pd.DataFrame):
    return df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")