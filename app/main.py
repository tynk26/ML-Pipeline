import pandas as pd
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional

# Internal imports from your structured folders
from app.ingestion.loader import load_all
from app.preprocessing.preprocessing import (
    normalize_selections,
    merge_selections_with_odds,
    merge_with_labels,
    safe_json
)

app = FastAPI(title="ML Data Ingestion API")

# In-memory storage for the processed dataset and rejections
DATA_STORE = {
    "merged_df": None,
    "rejections": []  # List of dicts: {"video_id": int, "stage": str, "reason": str}
}

@app.on_event("startup")
def startup_event():
    """
    Initial data load and processing on server start.
    """
    print("[startup] Initializing Data Pipeline...")
    try:
        # 1. Load raw files
        selections_raw, odds_df, labels_df = load_all()

        # 2. Normalize selections (Handling the nested/flat JSON schemas)
        selections_df = normalize_selections(selections_raw)

        # 3. Merge Step 1: Odds (Track rejections for missing ODD)
        merged_odds_df, missing_odd_ids = merge_selections_with_odds(selections_df, odds_df)
        for vid in missing_odd_ids:
            DATA_STORE["rejections"].append({
                "video_id": int(vid),
                "stage": "ODD_TAGGING",
                "reason": "Missing entry in odds.csv"
            })

        # 4. Merge Step 2: Labels (Track rejections for missing labels)
        final_df, label_stats = merge_with_labels(merged_odds_df, labels_df)
        for vid in label_stats.get("join_missing", []):
            DATA_STORE["rejections"].append({
                "video_id": int(vid),
                "stage": "AUTO_LABELING",
                "reason": "Missing entry in labels.csv"
            })

        # 5. Store final clean dataset
        DATA_STORE["merged_df"] = final_df
        print(f"[startup] Pipeline Complete. {len(final_df)} records ready.")

    except Exception as e:
        print(f"[startup] ERROR: {e}")

@app.get("/rejections")
def get_rejections(
    stage: Optional[str] = None, 
    reason: Optional[str] = None
):
    """
    Requirement 2-2: Retrieve rejected data.
    Supports filtering across multiple stages/reasons per video.
    """
    rejections = DATA_STORE["rejections"]

    # 1. Filter by Stage (Checks if the requested stage is in the list)
    if stage:
        target_stage = stage.upper()
        rejections = [r for r in rejections if target_stage in r["stages"]]
    
    # 2. Filter by Reason (Partial match across the list of reasons)
    if reason:
        query = reason.lower()
        rejections = [
            r for r in rejections 
            if any(query in res.lower() for res in r["reasons"])
        ]

    # 3. Sort by video_id
    sorted_rejections = sorted(rejections, key=lambda x: x["video_id"])

    return {
        "total_rejected_videos": len(sorted_rejections),
        "items": sorted_rejections
    }
@app.get("/merged_data")
def get_all_merged_data():
    """
    Returns the entire cleaned and integrated dataset.
    This fulfills the 'Integrated Data' portion of the assignment.
    """
    df = DATA_STORE["merged_df"]
    
    if df is None:
        raise HTTPException(
            status_code=503, 
            detail="Data pipeline has not been initialized. Please check server logs."
        )
    
    # We use safe_json to handle NaN values which standard JSON cannot parse
    return {
        "total_records": len(df),
        "data": safe_json(df.head(50))
    }

@app.post("/search")
def search_data(filters: Dict = Body(...)):
    df = DATA_STORE["merged_df"]
    if df is None:
        raise HTTPException(status_code=503, detail="Data not initialized")

    results_df = df.copy()

    # 1. Categorical / String Filters (Supports comma-separated: "sunny,rainy")
    # Columns: weather, time_of_day, road_surface, source_path
    text_cols = ["weather", "time_of_day", "road_surface", "source_path"]
    for col in text_cols:
        if col in filters and filters[col]:
            # Normalize to list and lowercase for comparison
            allowed = [v.strip().lower() for v in str(filters[col]).split(",")]
            results_df = results_df[results_df[col].astype(str).str.lower().isin(allowed)]

    # 2. Boolean Filters (wiper_on, headlights_on)
    # Handles inputs like true, "true", 1, "yes"
    for col in ["wiper_on", "headlights_on"]:
        if col in filters and filters[col] is not None:
            val = str(filters[col]).lower() in ["true", "1", "yes"]
            results_df = results_df[results_df[col] == val]

    # 3. Numeric Range Filters (Temperature, Wiper Level, Video ID)
    # Usage: { "temperature_celsius_min": 5.5, "temperature_celsius_max": 20 }
    range_map = {
        "temperature_celsius": "temperature_celsius",
        "temperature_fahrenheit": "temperature_fahrenheit",
        "wiper_level": "wiper_level",
        "video_id": "video_id"
    }
    for filter_prefix, col_name in range_map.items():
        if f"{filter_prefix}_min" in filters:
            results_df = results_df[results_df[col_name] >= float(filters[f"{filter_prefix}_min"])]
        if f"{filter_prefix}_max" in filters:
            results_df = results_df[results_df[col_name] <= float(filters[f"{filter_prefix}_max"])]

    # 4. Dynamic Nested Label Filters
    # Pattern: label_{object_class}_min
    # Example: "label_bus_min": 2 will look into the 'labels' dict for 'bus' count >= 2
    for key, value in filters.items():
        if key.startswith("label_") and key.endswith("_min"):
            obj_class = key.replace("label_", "").replace("_min", "")
            
            # Using a lambda to safely traverse the nested JSON dictionary
            results_df = results_df[
                results_df["labels"].apply(
                    lambda x: isinstance(x, dict) and 
                              x.get(obj_class, {}).get("count", 0) >= int(value)
                )
            ]

    return {
        "query_criteria": filters,
        "total_found": len(results_df),
        "results": safe_json(results_df.head(100)) # Return first 100 matches
    }

@app.get("/search-ui", response_class=HTMLResponse)
def search_ui():
    """HTML UI updated to specifically target the attributes of Video ID 3."""
    
    # This JSON matches the specific conditions of video_id 3:
    # 58.2°F, Wipers Level 3, Night, 31 Cars, 26 Pedestrians
    sample_json = {
        "weather": "sunny",
        "time_of_day": "night",
        "road_surface": "dry",
        "wiper_on": "true",
        "wiper_level_min": 3,
        "headlights_on": "true",
        "temperature_fahrenheit_min": 58,
        "temperature_fahrenheit_max": 60,
        "label_car_min": 30,
        "label_pedestrian_min": 25,
        "label_traffic_sign_min": 10
    }
    
    import json
    json_str = json.dumps(sample_json, indent=2)

    return f"""
    <html>
        <head>
            <title>ML Search Tester - Video ID 3 Target</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f0f2f5; }}
                .container {{ max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                textarea {{ 
                    width: 100%; 
                    height: 350px; 
                    font-family: 'Consolas', 'Monaco', monospace; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 4px; 
                    background-color: #f8f9fa;
                    font-size: 14px;
                    line-height: 1.5;
                }}
                button {{ 
                    padding: 12px 24px; 
                    background-color: #28a745; 
                    color: white; 
                    border: none; 
                    border-radius: 4px; 
                    cursor: pointer; 
                    margin-top: 15px; 
                    font-size: 16px; 
                    font-weight: bold;
                }}
                button:hover {{ background-color: #218838; }}
                pre {{ 
                    background: #2b2b2b; 
                    color: #e6e6e6; 
                    padding: 20px; 
                    border-radius: 6px; 
                    overflow-x: auto; 
                    margin-top: 20px;
                    border-left: 5px solid #28a745;
                }}
                .error {{ color: #dc3545; border-left-color: #dc3545; }}
                h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>ML Dataset Search API</h2>
                <p>아래 JSON 형식으로 필터를 입련한 후 검색해보세요</p>
                
                <textarea id="filters">{json_str}</textarea><br>
                
                <button onclick="doSearch()">Run Search Query</button>
                
                <h3>API Response:</h3>
                <pre id="out">// Click 'Run Search Query' to see results...</pre>
            </div>

            <script>
                async function doSearch() {{
                    const out = document.getElementById('out');
                    out.innerText = "Processing request...";
                    out.classList.remove('error');

                    try {{
                        const filterData = JSON.parse(document.getElementById('filters').value);
                        
                        const res = await fetch('/search', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(filterData)
                        }});

                        if (!res.ok) {{
                            const error = await res.json();
                            throw new Error(`Server returned ${{res.status}}: ${{JSON.stringify(error)}}`);
                        }}

                        const data = await res.json();
                        out.innerText = JSON.stringify(data, null, 2);
                    }} catch (err) {{
                        out.classList.add('error');
                        out.innerText = "Error: " + err.message;
                    }}
                }}
            </script>
        </body>
    </html>
    """
# @app.get("/search-ui", response_class=HTMLResponse)
# def search_ui():
#     """Simple HTML wrapper to test the search functionality."""
#     return """
#     <html>
#         <body>
#             <h1>ML Dataset Explorer</h1>
#             <p>Enter filters (e.g., weather: sunny, label_car_min: 5)</p>
#             <textarea id="filters" style="width:300px; height:100px">{"weather": "sunny", "label_car_min": 10}</textarea><br>
#             <button onclick="doSearch()">Search</button>
#             <pre id="out"></pre>
#             <script>
#                 async function doSearch() {
#                     const filters = JSON.parse(document.getElementById('filters').value);
#                     const res = await fetch('/search', {
#                         method: 'POST',
#                         headers: {'Content-Type': 'application/json'},
#                         body: JSON.stringify(filters)
#                     });
#                     const data = await res.json();
#                     document.getElementById('out').innerText = JSON.stringify(data, null, 2);
#                 }
#             </script>
#         </body>
#     </html>
#     """