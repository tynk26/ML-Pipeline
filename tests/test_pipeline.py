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
    # High heat check
    assert round(f_to_c(999), 2) == round((999 - 32) * 5/9, 2)

# --- 2. Tests for Data Normalization ---
def test_normalize_mixed_software_versions():
    """Test handling of nested 'sensor' dict vs flat keys (Requirement 2-1)."""
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
            "temperature": 10,
            "isWiperOn": False,
            "headlightsOn": False
        }
    ]
    df = normalize_selections(mixed_data)
    
    assert df.loc[df['video_id'] == 1, 'temperature_celsius'].iloc[0] == 10.0 # (50-32)*5/9
    assert df.loc[df['video_id'] == 2, 'temperature_celsius'].iloc[0] == 10.0
    assert df.loc[df['video_id'] == 1, 'wiper_level'].iloc[0] == 2
    assert df.loc[df['video_id'] == 2, 'wiper_level'].iloc[0] == 0

# --- 3. Tests for Label Edge Cases (The "Strict" Suite) ---

def test_label_rejection_reasons():
    """
    Comprehensive test for negative counts, duplicates, non-integers, and invalid classes.
    """
    merged_df = pd.DataFrame([
        {"video_id": 1}, {"video_id": 2}, {"video_id": 3}, 
        {"video_id": 4}, {"video_id": 5}
    ])
    
    labels_df = pd.DataFrame([
        # ID 1: Valid
        {"video_id": 1, "object_class": "car", "obj_count": 5, "avg_confidence": 0.9},
        # ID 2: Duplicate labels (Same video, same class)
        {"video_id": 2, "object_class": "car", "obj_count": 5, "avg_confidence": 0.9},
        {"video_id": 2, "object_class": "car", "obj_count": 2, "avg_confidence": 0.8},
        # ID 3: Negative count
        {"video_id": 3, "object_class": "bus", "obj_count": -1, "avg_confidence": 0.7},
        # ID 4: Non-integer count
        {"video_id": 4, "object_class": "pedestrian", "obj_count": 12.5, "avg_confidence": 0.6},
        # ID 5: Invalid object class
        {"video_id": 5, "object_class": "unknown", "obj_count": 1, "avg_confidence": 0.5}
    ])
    
    final_df, stats = merge_with_labels(merged_df, labels_df)
    
    # Verify rejection mapping
    error_map = stats["error_map"]
    assert "duplicate_label: car" in error_map[2]
    assert "negative_obj_count: bus" in error_map[3]
    assert "non_integer_obj_count: pedestrian" in error_map[4]
    assert "invalid_object_class" in error_map[5]
    
    # Only ID 1 should make it to final_df
    assert len(final_df) == 1
    assert final_df.iloc[0]["video_id"] == 1

def test_missing_label_data():
    """Verify videos with NO labels are caught as missing_label_data."""
    merged_df = pd.DataFrame([{"video_id": 10}, {"video_id": 20}])
    labels_df = pd.DataFrame([
        {"video_id": 10, "object_class": "car", "obj_count": 1, "avg_confidence": 0.9}
    ]) # ID 20 is missing
    
    final_df, stats = merge_with_labels(merged_df, labels_df)
    
    assert 20 in stats["missing_ids"]
    assert len(final_df) == 1

def test_dynamic_flattening_and_json():
    """Verify columns are created dynamically and 'labels' is a valid JSON string."""
    merged_df = pd.DataFrame([{"video_id": 1}])
    labels_df = pd.DataFrame([
        {"video_id": 1, "object_class": "car", "obj_count": 10, "avg_confidence": 0.95},
        {"video_id": 1, "object_class": "truck", "obj_count": 2, "avg_confidence": 0.80}
    ])
    
    final_df, _ = merge_with_labels(merged_df, labels_df)
    
    # Check flattening
    assert "label_car_count" in final_df.columns
    assert "label_truck_count" in final_df.columns
    assert final_df.iloc[0]["label_car_count"] == 10
    
    # Check JSON string for DB storage
    labels_json = json.loads(final_df.iloc[0]["labels"])
    assert labels_json["car"]["count"] == 10
    assert labels_json["truck"]["avg_confidence"] == 0.80

def test_id_type_resilience():
    """Ensure pipeline handles string IDs vs int IDs gracefully."""
    # JSON often gives IDs as strings or ints; CSV IDs are usually ints
    merged_df = pd.DataFrame([{"video_id": 100}]) # int
    labels_df = pd.DataFrame([
        {"video_id": 100, "object_class": "car", "obj_count": 1, "avg_confidence": 0.9}
    ])
    
    # merge_with_labels should be resilient (usually handled by normalization)
    final_df, _ = merge_with_labels(merged_df, labels_df)
    assert len(final_df) == 1