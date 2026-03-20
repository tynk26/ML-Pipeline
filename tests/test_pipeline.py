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

# --- 1. Unit Tests for Math & Conversion ---
def test_temperature_edge_cases():
    """Test extreme and null values for temperature."""
    assert f_to_c(32) == 0
    assert f_to_c(-40) == -40  # Point where F and C meet
    assert f_to_c(None) is None
    assert f_to_c(999) == (999 - 32) * 5/9  # High heat

# --- 2. Tests for Data Normalization (Requirement 2-1) ---
def test_normalize_mixed_software_versions():
    """
    Test Requirement: 'Software versions differ, so collected info varies.'
    One item has 'sensor' dict, another has flat keys.
    """
    mixed_data = [
        {
            "id": 1,
            "recordedAt": "2026-01-01T10:00:00Z",
            "sensor": {"temperature": {"value": 50}, "wiper": {"isActive": True, "level": 2}},
            "headlights": True
        },
        {
            "id": 2,
            "recordedAt": "2026-01-01T11:00:00Z",
            "temperature": 10, # Celsius
            "isWiperOn": False,
            "headlightsOn": False
        }
    ]
    df = normalize_selections(mixed_data)
    
    # ID 1 (Fahrenheit conversion)
    assert df.loc[df['video_id'] == 1, 'temperature_celsius'].iloc[0] == f_to_c(50)
    # ID 2 (Direct Celsius)
    assert df.loc[df['video_id'] == 2, 'temperature_celsius'].iloc[0] == 10
    assert df.loc[df['video_id'] == 1, 'wiper_level'].iloc[0] == 2
    assert df.loc[df['video_id'] == 2, 'wiper_level'].iloc[0] == 0

def test_normalize_malformed_dates():
    """Test how the pipeline handles invalid date strings."""
    bad_date_data = [{
        "id": 99,
        "recordedAt": "not-a-date",
        "temperature": 20
    }]
    df = normalize_selections(bad_date_data)
    # pd.to_datetime with errors="coerce" should result in NaT
    assert pd.isna(df.iloc[0]["recorded_at"])

# --- 3. Tests for Integration & Rejection (Requirement 2-2) ---
def test_rejection_logic_flow():
    """Verify that IDs missing in ODDs are caught for the rejection table."""
    sel_df = pd.DataFrame([{"video_id": 101}, {"video_id": 102}])
    odds_df = pd.DataFrame([{"video_id": 101, "weather": "sunny"}]) # 102 is missing
    
    merged, rejected_ids = merge_selections_with_odds(sel_df, odds_df)
    assert 102 in rejected_ids
    assert len(merged) == 1

def test_dynamic_flattening_edge_cases():
    """Test labeling when some videos have zero objects of a certain class."""
    merged_df = pd.DataFrame([{"video_id": 1}, {"video_id": 2}])
    labels_df = pd.DataFrame([
        {"video_id": 1, "object_class": "car", "obj_count": 5, "avg_confidence": 0.9},
        {"video_id": 1, "object_class": "bus", "obj_count": 1, "avg_confidence": 0.8}
        # Video 2 has NO labels in labels_df
    ])
    
    final_df, stats = merge_with_labels(merged_df, labels_df)
    
    # Video 2 should be in join_missing stats (Requirement 2-2)
    assert 2 in stats["join_missing"]
    # Check flattening for Video 1
    assert "label_car_count" in final_df.columns
    assert final_df.loc[final_df['video_id'] == 1, 'label_car_count'].iloc[0] == 5
    assert final_df.loc[final_df['video_id'] == 1, 'label_bus_count'].iloc[0] == 1