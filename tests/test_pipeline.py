import pytest
import pandas as pd
import numpy as np
import json
import requests
import os

# Silence the specific Pandas FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# API Configuration
BASE_URL = "http://127.0.0.1:8000"

from app.preprocessing.preprocessing import (
    f_to_c, 
    normalize_selections, 
    merge_selections_with_odds, 
    merge_with_labels
)



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

# --- 3. Stage 2: Label Rejection ---

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

# --- 6. Integration Tests (Full Pipeline Suite) ---

def test_full_pipeline_flow():
    # Trigger /analyze
    resp = requests.post(f"{BASE_URL}/analyze")
    assert resp.status_code == 200
    assert "analysis_report" in resp.json()

def test_search_extreme_env():
    # Case SEARCH-01
    payload = {"weather": "rainy", "time_of_day": "night", "wiper_on": 1}
    resp = requests.post(f"{BASE_URL}/search", json=payload)
    assert resp.status_code == 200

def test_search_time_substr():
    # Case SEARCH-02
    payload = {"recorded_at_min": "2026-01-10T19", "recorded_at_max": "2026-01-10T19"}
    resp = requests.post(f"{BASE_URL}/search", json=payload)
    assert resp.status_code == 200

def test_search_confidence_sandwich():
    # Case SEARCH-03
    payload = {"label_car_min": 10, "label_car_confidence_min": 0.8, "label_car_confidence_max": 0.95}
    resp = requests.post(f"{BASE_URL}/search", json=payload)
    assert resp.status_code == 200

def test_search_exact_id():
    # Case SEARCH-04
    payload = {"video_id_min": 3, "video_id_max": 3}
    resp = requests.post(f"{BASE_URL}/search", json=payload)
    assert resp.status_code == 200

# --- 6. Integration Tests: Scenario-Based Search & Rejections ---

@pytest.fixture(scope="module", autouse=True)
def prepare_database():
    """Triggers analyze to populate DB and ensures we start fresh."""
    # Optional: If you have a reset endpoint, call it here.
    resp = requests.post(f"{BASE_URL}/analyze")
    assert resp.status_code == 200
    return resp.json()

# --- Search Functionality Scenarios ---
@pytest.mark.parametrize("scenario, payload, expect_to_find_vid_3", [
    # --- A. The "Sandwich" Stress Tests ---
    ("TEMP_EXACT_BOUNDARY", {"temperature_celsius_min": 14.55, "temperature_celsius_max": 14.56}, True),
    ("MULTI_OBJECT_SANDWICH", {
        "label_car_min": 20, 
        "label_pedestrian_min": 20, 
        "label_traffic_sign_min": 5
    }, True),
    ("CONFIDENCE_BOUNDARY_HIGH", {"label_car_confidence_min": 0.83, "label_car_confidence_max": 0.84}, True),
    ("CONFIDENCE_BOUNDARY_LOW", {"label_car_confidence_min": 0.90}, False),

    # --- B. Environmental & Hardware State ---
    ("HARDWARE_MATCH", {"wiper_on": 1, "headlights_on": 1, "wiper_level": 3}, True),
    ("ROAD_SURFACE_STRICT", {"road_surface": "dry", "weather": "sunny"}, True),
    ("ROAD_SURFACE_MISS", {"road_surface": "wet"}, False),

    # --- C. Time String "LIKE" Logic ---
    ("TIME_SUBSTR_HOUR", {"recorded_at_min": "2026-01-10T19:44"}, True),
    ("TIME_SUBSTR_SECOND_MISS", {"recorded_at_min": "2026-01-10T19:44:99"}, False),

    # --- D. Null & Empty Handling ---
    ("NON_EXISTENT_CLASS_FILTER", {"label_motorcycle_min": 1}, False),
    ("EMPTY_PAYLOAD", {}, True), # Should return everything, including Vid 3
    
    # --- E. ID Range (Video 3) ---
    ("ID_SANDWICH_HIT", {"video_id_min": 1, "video_id_max": 10}, True),
    ("ID_SANDWICH_MISS", {"video_id_min": 4, "video_id_max": 100}, False),
])
def test_search_scenarios(scenario, payload, expect_to_find_vid_3):
    """Checks various filter combinations against the known sample data."""
    resp = requests.post(f"{BASE_URL}/search", json=payload)
    assert resp.status_code == 200
    
    results = resp.json().get("results", [])
    found_ids = [r["video_id"] for r in results]
    
    if expect_to_find_vid_3:
        assert 3 in found_ids, f"Scenario '{scenario}' failed: Video 3 should have been found."
    else:
        assert 3 not in found_ids, f"Scenario '{scenario}' failed: Video 3 should NOT have been found."
# --- Rejection Endpoint Scenarios (Dynamic ID Version) ---
@pytest.mark.parametrize("scenario, params", [
    # 1. Stage Isolation
    ("STAGE_ODD", {"stage": "odd_tagging_step"}),
    ("STAGE_LABEL", {"stage": "auto_labeling_step"}),
    
    # 2. Specific Error Reason Substrings
    ("REASON_MISSING_ODD", {"reason": "missing_odd"}),
    ("REASON_DUPLICATE", {"reason": "duplicate"}),
    ("REASON_NEGATIVE", {"reason": "negative"}),
    ("REASON_NON_INT", {"reason": "non_integer"}),
    
    # 3. Combined Filters (If your API supports both at once)
    ("STAGE_AND_REASON", {"stage": "auto_labeling_step", "reason": "non_integer"}),
])
def test_rejection_scenarios(scenario, params):
    """
    Validates that the rejection endpoint correctly filters data.
    Instead of hardcoded IDs, it verifies the 'integrity' of returned items.
    """
    resp = requests.get(f"{BASE_URL}/rejections", params=params)
    assert resp.status_code == 200
    
    data = resp.json()
    items = data.get("items", [])
    
    # Check 1: Ensure we actually got data back from the DB
    assert len(items) > 0, f"No rejections found for {params}. Run /analyze first!"

    # Check 2: Validate the logic for every returned item
    for rej in items:
        # If we filtered by stage, every item must be from that stage
        if "stage" in params:
            assert rej["stage"] == params["stage"], \
                f"ID {rej['video_id']} has wrong stage: {rej['stage']}"
        
        # If we filtered by reason (LIKE), every reason must contain the substring
        if "reason" in params:
            expected_sub = params["reason"].lower()
            assert expected_sub in rej["reason"].lower(), \
                f"ID {rej['video_id']} reason '{rej['reason']}' misses '{expected_sub}'"

def test_rejection_empty_case():
    """Ensures garbage filters return 0 results instead of crashing."""
    resp = requests.get(f"{BASE_URL}/rejections", params={"reason": "non_existent_error_999"})
    assert resp.status_code == 200
    assert len(resp.json().get("items", [])) == 0