import pandas as pd
import numpy as np
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
    selections_df["video_id"] = selections_df["video_id"].astype(int)
    odds_df["video_id"] = odds_df["video_id"].astype(int)
    
    # Simple gap detection for the rejection requirement
    existing_ids = set(odds_df["video_id"].unique())
    all_ids = set(selections_df["video_id"].unique())
    rejected_ids = sorted(all_ids - existing_ids)
    
    merged = pd.merge(selections_df, odds_df, on="video_id", how="inner")
    return merged, rejected_ids

def merge_with_labels(merged_df, labels_df) -> Tuple[pd.DataFrame, Dict]:
    labels_df["video_id"] = labels_df["video_id"].astype(int)
    
    # Pivot logic
    labels_pivot = labels_df.groupby("video_id").apply(
        lambda x: {row.object_class: {"count": row.obj_count, "avg_confidence": row.avg_confidence} 
                   for row in x.itertuples()}
    ).reset_index(name="labels")
    
    final_df = pd.merge(merged_df, labels_pivot, on="video_id", how="inner")
    stats = {"join_missing": sorted(set(merged_df["video_id"]) - set(labels_pivot["video_id"]))}
    return final_df, stats

def safe_json(df: pd.DataFrame):
    return df.replace([np.nan, np.inf, -np.inf], None).to_dict(orient="records")