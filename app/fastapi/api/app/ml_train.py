import pandas as pd
import numpy as np
import re
import json
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .config import DB_CONNECTION_PARAMS

MODEL_PATH = "/tmp/baseline_model.pkl"
FEATURES_PATH = "/tmp/feature_columns.txt"

COLUMN_MAPPING = {
    "Ссылка_order": "id",
    "Дата_order": "date",
    "ВалютаДокумента": "currency",
    "Контрагент": "customer_name",
    "Курс": "currency_rate",
    "Проект": "project",
    "ВидПродукции": "product_type",
    "Номенклатура": "item_name",
    "ЕдиницаИзмерения": "unit",
    "Количество_order": "quantity",
    "Цена": "price",
    "Объем": "capacity",
    "ОперацияНТК": "ntk_operation",
    "ВидПродукцииЗаказ": "product_type_order",
    "ВысотаПродукции": "product_height",
    "ДлинаПродукции": "product_length",
    "ШиринаПродукции": "product_width",
    "Материал": "material",
    "ДлинаРеальная": "actual_length",
    "ШиринаРеальная": "actual_width",
    "cnt_order": "number_of_operations",
    "СостояниеЗаказа_order": "status",
    "Ссылка_plant": "production_id",
    "Дата_plant": "production_date",
    "ВидОперации": "operation_type",
    "ЗаказПокупателя": "customer_order_id",
    "СостояниеЗаказа_plant": "production_status",
    "cnt_plant": "product_count",
    "Количество_plant": "prod_quantity",
    "РесурсПредприятия": "resource_enterprise",
    "Тип": "power_type",
    "Старт": "start_time",
    "Финиш": "end_time",
    "ВремяArtCAM": "artcam_time",
    "Корректировка": "adjustment",
    "Времяработы": "working_time",
    "Брак": "manufacturing",
    "Пробноеклише": "test_plate",
    "ПутьКфайлам": "path_to_files",
    "Необходимытесты": "required_tests",
    "ТестыВыполнены": "tests_completed",
}


def get_db_engine():
    params = DB_CONNECTION_PARAMS
    url = f"postgresql+pg8000://{params['user']}:{params['password']}@{params['host']}/{params['database']}"
    return create_engine(url)


def load_data_from_buffer():
    engine = get_db_engine()
    df_raw = pd.read_sql("SELECT data FROM buffer_merged;", engine)
    if df_raw.empty:
        raise ValueError("No data in buffer_merged")
    records = [json.loads(row['data']) for _, row in df_raw.iterrows()]
    df = pd.DataFrame(records)
    df = df.rename(columns=COLUMN_MAPPING)
    return df


def filter_dataframe(df):
    df_filtered = df.copy()
    mask_resource = df_filtered['resource_enterprise'].str.contains('Датрон|ПЛ', na=False, case=False)
    mask_material = df_filtered['material'].str.contains('Латунь|Алюминий|Стеклотекстолит', na=False, case=False)
    mask_unit = ~df_filtered['unit'].str.contains('усл|ч', na=False, case=False)
    return df_filtered[mask_resource & mask_material & mask_unit]


def fix_date_parsing(df):
    df_fixed = df.copy()
    for col in ['start_time', 'end_time']:
        if col in df_fixed.columns:
            df_fixed[col] = pd.to_datetime(df_fixed[col], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    return df_fixed


def extract_first_3_words(text):
    if pd.isna(text):
        return None
    text = re.sub(r'\s{2,}', ' ', text).replace('_', ' ')
    return " ".join(text.split()[:3])


def to_numeric_safe(series):
    return pd.to_numeric(series.astype(str).str.replace(' ', '').str.replace(',', '.'), errors='coerce')


def handle_missing_values(df, numeric_features, categorical_features):
    df_processed = df.copy()
    for col in numeric_features:
        if col in df_processed.columns:
            if df_processed[col].isnull().all():
                df_processed[col] = 0
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    for col in categorical_features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
    return df_processed


def frequency_encode(train_df, test_df, categorical_columns):
    train_enc = train_df.copy()
    test_enc = test_df.copy()
    for col in categorical_columns:
        if col in train_enc.columns:
            freq = train_enc[col].value_counts().to_dict()
            train_enc[col] = train_enc[col].map(freq)
            test_enc[col] = test_enc[col].map(freq).fillna(0)
    return train_enc, test_enc


def run_training():
    df = load_data_from_buffer()
    print(f"Loaded {len(df)} rows")

    df = filter_dataframe(df)
    print(f"After filtering: {len(df)} rows")

    df = fix_date_parsing(df)
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].abs()
    df = df[df['duration_minutes'].notna() & (df['duration_minutes'] > 0)].copy()
    print(f"After removing invalid duration: {len(df)} rows")

    df['item_short'] = df['item_name'].astype(str).apply(extract_first_3_words)

    categorical_cols = [
        'currency', 'customer_name', 'product_type', 'ntk_operation',
        'product_type_order', 'material', 'resource_enterprise',
        'project', 'item_short', 'test_plate'
    ]
    numeric_cols = [
        'quantity', 'price', 'capacity', 'product_height',
        'product_length', 'product_width', 'actual_length',
        'actual_width', 'number_of_operations', 'adjustment', 'manufacturing'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    X = df[numeric_cols + categorical_cols].copy()
    y = df['duration_minutes']

    X = handle_missing_values(X, numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_enc, X_test_enc = frequency_encode(X_train, X_test, categorical_cols)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        f.write("\n".join(X_train_enc.columns.tolist()))

    print(f"Model saved to {MODEL_PATH}")
    print(f"MAE: {mae:.2f}, R²: {r2:.4f}")

    freq_maps = {}
    for col in categorical_cols:
        if col in X_train.columns:
            freq_maps[col] = X_train[col].value_counts().to_dict()
    joblib.dump(freq_maps, "/tmp/freq_maps.pkl")

    with open(FEATURES_PATH, "w") as f:
        f.write("\n".join(X_train_enc.columns.tolist()))

    return {
        "mae": mae,
        "r2": r2,
        "model_path": MODEL_PATH,
        "n_samples": len(df),
        "features_count": X_train_enc.shape[1]
    }


if __name__ == "__main__":
    print(run_training())