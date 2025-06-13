import pandas as pd
import json
from model.config import DATA_PATH, SYNONYM_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["감성점수"] >= 53].reset_index(drop=True)
    return df

def load_synonym_dict():
    with open(SYNONYM_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
