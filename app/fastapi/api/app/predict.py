import pandas as pd
import joblib
from typing import List, Dict, Any
from .ml_train import handle_missing_values, extract_first_3_words

MODEL_PATH = "/tmp/baseline_model.pkl"
FEATURES_PATH = "/tmp/feature_columns.txt"
FREQ_MAPS_PATH = "/tmp/freq_maps.pkl"


def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    freq_maps = joblib.load(FREQ_MAPS_PATH)
    return model, feature_cols, freq_maps


def apply_frequency_encoding(df, categorical_cols, freq_maps):
    for col in categorical_cols:
        if col in df.columns and col in freq_maps:
            df[col] = df[col].map(freq_maps[col]).fillna(0)
    return df


def preprocess_prediction(df, feature_cols, freq_maps, numeric_cols, categorical_cols):
    df = handle_missing_values(df, numeric_cols, categorical_cols)
    df = apply_frequency_encoding(df, categorical_cols, freq_maps)
    df = df[feature_cols]
    return df


def predict_model(requests: List[Dict[str, Any]]) -> List[float]:
    model, feature_cols, freq_maps = load_model_and_features()

    numeric_cols = [
        'quantity', 'price', 'capacity', 'product_height',
        'product_length', 'product_width', 'actual_length',
        'actual_width', 'number_of_operations', 'adjustment', 'manufacturing'
    ]
    categorical_cols = [
        'currency', 'customer_name', 'product_type', 'ntk_operation',
        'product_type_order', 'material', 'resource_enterprise',
        'project', 'item_short', 'test_plate'
    ]

    df = pd.DataFrame(requests)
    required_cols = numeric_cols + categorical_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if 'item_short' not in df.columns and 'item_name' in df.columns:
        df['item_short'] = df['item_name'].astype(str).apply(extract_first_3_words)
        categorical_cols.append('item_short')

    df_processed = preprocess_prediction(df, feature_cols, freq_maps, numeric_cols, categorical_cols)
    predictions = model.predict(df_processed)
    return predictions.tolist()