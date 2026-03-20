import os
import json
import pandas as pd

def get_data_dir():
    # backend/app/ingestion/loader.py → up 2 levels = root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", "..")) 
    data_dir = os.path.join(project_root, "data")
    return data_dir

def load_json(file_path):
    if not os.path.exists(file_path): raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[loader] Loaded JSON: {len(data)} records")
    return data

def load_csv(file_path):
    if not os.path.exists(file_path): raise FileNotFoundError(f"{file_path} not found")
    df = pd.read_csv(file_path)
    print(f"[loader] Loaded CSV: {len(df)} rows")
    return df

def load_all():
    data_dir = get_data_dir()
    selections = load_json(os.path.join(data_dir, "selections.json"))
    odds = load_csv(os.path.join(data_dir, "odds.csv"))
    labels = load_csv(os.path.join(data_dir, "labels.csv"))
    return selections, odds, labels