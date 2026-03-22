import pytest
import pandas as pd
import numpy as np
import json
from app.preprocessing.preprocessing import (
    f_to_c, 
    normalize_selections, 
    merge_selections_with_odds, 
    merge_with_labels
)

# --- 1. Unit Tests: Temperature Logic ---

def test_f_to_c_freezing():
    assert f_to_c(32) == 0

def test_f_to_c_precision():
    assert abs(f_to_c(50) - 10.0) < 1e-9

def test_f_to_c_none():
    assert f_to_c(None) is None

# --- 2. Unit Tests: Normalization ---

def test_normalize_time_format_iso():
    mixed_data = [{
        "id": 1,
        "recordedAt": "2026-01-01 10:00:00",
        "sensor": {"temperature": {"value": 50}, "wiper": {"level": 2}},
        "headlights": True
    }]
    df = normalize_selections(mixed_data)
    recorded_val = df.iloc[0]["recorded_at"]
    assert " " not in recorded_val
    assert "T" in recorded_val

def test_normalize_temperature_celsius():
    mixed_data = [{
        "id": 1,
        "recordedAt": "2026-01-01 10:00:00",
        "sensor": {"temperature": {"value": 50}}
    }]
    df = normalize_selections(mixed_data)
    assert df.iloc[0]["temperature_celsius"] == 10.0

# --- 3. Stage 2: Label Rejection (Modularized) ---

@pytest.fixture
def rejection_data():
    merged_df = pd.DataFrame([{"video_id": i} for i in range(1, 6)])
    labels_df = pd.DataFrame([
        {"video_id": 1, "object_class": "car", "obj_count": 5, "avg_confidence": 0.9},
        {"video_id": 2, "object_class": "car", "obj_count": 5, "avg_confidence": 0.9},
        {"video_id": 2, "object_class": "car", "obj_count": 2, "avg_confidence": 0.8},
        {"video_id": 3, "object_class": "bus", "obj_count": -1, "avg_confidence": 0.7},
        {"video_id": 4, "object_class": "ped", "obj_count": 12.5, "avg_confidence": 0.6},
    ])
    labels_df["labeled_at"] = "2026-03-22T12:00:00"
    return merged_df, labels_df

def test_reject_duplicate_class(rejection_data):
    merged_df, labels_df = rejection_data
    _, stats = merge_with_labels(merged_df, labels_df)
    assert stats["error_map"][2] == "duplicate_label_class"

def test_reject_negative_count(rejection_data):
    merged_df, labels_df = rejection_data
    _, stats = merge_with_labels(merged_df, labels_df)
    assert stats["error_map"][3] == "negative_obj_count"

def test_reject_non_integer_count(rejection_data):
    merged_df, labels_df = rejection_data
    _, stats = merge_with_labels(merged_df, labels_df)
    assert stats["error_map"][4] == "non_integer_obj_count"

# --- 4. Stage 2: Label Survival & Flattening ---

def test_missing_label_exclusion():
    merged_df = pd.DataFrame([{"video_id": 10}, {"video_id": 20}])
    labels_df = pd.DataFrame([
        {"video_id": 10, "object_class": "car", "obj_count": 1, "avg_confidence": 0.9, "labeled_at": "now"}
    ])
    final_df, _ = merge_with_labels(merged_df, labels_df)
    assert 10 in final_df["video_id"].values
    assert 20 not in final_df["video_id"].values

def test_dynamic_column_flattening():
    merged_df = pd.DataFrame([{"video_id": 1}])
    labels_df = pd.DataFrame([
        {"video_id": 1, "object_class": "car", "obj_count": 10, "avg_confidence": 0.95, "labeled_at": "T1"},
        {"video_id": 1, "object_class": "truck", "obj_count": 2, "avg_confidence": 0.80, "labeled_at": "T1"}
    ])
    final_df, _ = merge_with_labels(merged_df, labels_df)
    assert final_df.iloc[0]["label_car_count"] == 10
    assert final_df.iloc[0]["label_truck_count"] == 2

def test_json_summary_structure():
    merged_df = pd.DataFrame([{"video_id": 1}])
    labels_df = pd.DataFrame([
        {"video_id": 1, "object_class": "car", "obj_count": 10, "avg_confidence": 0.95, "labeled_at": "T1"}
    ])
    final_df, _ = merge_with_labels(merged_df, labels_df)
    labels_dict = json.loads(final_df.iloc[0]["labels"])
    assert labels_dict["car"]["count"] == 10
    # Implementation key is avg_confidence
    assert labels_dict["car"]["avg_confidence"] == 0.95

# --- 5. Stage 1: ODD Processing ---

def test_odd_missing_ids():
    sel = pd.DataFrame([{"video_id": 1}, {"video_id": 2}, {"video_id": 3}])
    odd = pd.DataFrame([{"video_id": 1, "weather": "sunny"}])
    _, missing, _ = merge_selections_with_odds(sel, odd)
    assert 2 in missing
    assert 3 in missing

def test_odd_duplicate_ids():
    sel = pd.DataFrame([{"video_id": 1}, {"video_id": 2}])
    odd = pd.DataFrame([
        {"video_id": 2, "weather": "rainy"},
        {"video_id": 2, "weather": "snowy"}
    ])
    _, _, dups = merge_selections_with_odds(sel, odd)
    assert 2 in dups

# --- 6. Full Pipeline Handover ---

def test_full_pipeline_flow():
    sel = pd.DataFrame([{"video_id": 1, "raw": "{}"}, {"video_id": 2, "raw": "{}"}])
    odd = pd.DataFrame([{"video_id": 1, "weather": "clear"}])
    lab = pd.DataFrame([
        {"video_id": 1, "object_class": "car", "obj_count": 1, "avg_confidence": 0.9, "labeled_at": "T1"},
        {"video_id": 2, "object_class": "car", "obj_count": 1, "avg_confidence": 0.9, "labeled_at": "T1"}
    ])
    
    merged_odd, _, _ = merge_selections_with_odds(sel, odd)
    final_df, stats = merge_with_labels(merged_odd, lab)
    
    assert len(final_df) == 1
    assert final_df.iloc[0]["video_id"] == 1
    assert 2 not in stats["error_map"]