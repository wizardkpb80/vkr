from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import mlflow
import pandas as pd
import numpy as np
import torch
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import create_engine

DB_HOST = Variable.get("DB_HOST", "postgres")
DB_NAME = Variable.get("DB_NAME", "postgres")
DB_USER = Variable.get("DB_USER", "postgres")
DB_PASSWORD = Variable.get("DB_PASSWORD", "postgres")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = Variable.get("MLFLOW_EXPERIMENT", "qwen_clishe_regression")
VERSION = Variable.get("VERSION", "v_3_1")
PROJECT_NAME = Variable.get("PROJECT_NAME", "cliche_ai")
BASE_IMAGE_DIR = Variable.get("BASE_IMAGE_DIR", "/tmp/images")

TRAINING_VARIANTS = [
    {"name": "lora_r8_lr1e-4", "lora_r": 8, "lora_alpha": 16, "learning_rate": 1e-4},
    {"name": "lora_r16_lr2e-4", "lora_r": 16, "lora_alpha": 32, "learning_rate": 2e-4},
    {"name": "lora_r32_lr3e-4", "lora_r": 32, "lora_alpha": 64, "learning_rate": 3e-4},
]


def get_db_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    return create_engine(url)


def load_training_data() -> pd.DataFrame:
    engine = get_db_engine()
    sql = """
    SELECT b.data, oi.file_path, oi.order_number
    FROM buffer_merged b
    JOIN order_images oi ON (b.data->>'id') = oi.order_uuid::text
    """
    df_raw = pd.read_sql(sql, engine)
    if df_raw.empty:
        raise ValueError("No data in buffer_merged with matching images")

    records = [json.loads(row['data']) for _, row in df_raw.iterrows()]
    df_tab = pd.DataFrame(records)

    df_tab['file_path'] = df_raw['file_path'].values

    from app.ml_train import COLUMN_MAPPING
    df_tab = df_tab.rename(columns=COLUMN_MAPPING)

    from app.ml_train import filter_dataframe, fix_date_parsing, extract_first_3_words
    df = filter_dataframe(df_tab)
    df = fix_date_parsing(df)
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].abs()
    df = df[df['duration_minutes'].notna() & (df['duration_minutes'] > 0)].copy()
    df['item_short'] = df['item_name'].astype(str).apply(extract_first_3_words)

    from app.dl_train import create_smart_context
    df['text_context'] = df.apply(create_smart_context, axis=1)

    df['local_image_path'] = df['file_path'].apply(
        lambda p: str(BASE_IMAGE_DIR / p.replace('\\', '/')) if p else None
    )
    df = df[df['local_image_path'].notna() & df['local_image_path'].apply(os.path.exists)]
    return df


def train_qwen_variant(params: Dict[str, Any], train_df: pd.DataFrame,
                       test_df: pd.DataFrame, run_name: str):
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments
    )
    from peft import LoraConfig, get_peft_model
    from app.dl_train import QwenRegressionDataset, QwenRegressionCollator, predict_qwen_model
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    lora_config = LoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_ds = QwenRegressionDataset(train_df, image_col="local_image_path",
                                     target_col="duration_minutes")
    test_ds = QwenRegressionDataset(test_df, image_col="local_image_path",
                                    target_col="duration_minutes")
    collator = QwenRegressionCollator(processor, device=model.device)

    training_args = TrainingArguments(
        output_dir=f"/tmp/qwen_{run_name}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        num_train_epochs=3,
        learning_rate=params["learning_rate"],
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )
    trainer.train()

    pred_df = predict_qwen_model(test_df, model, processor)
    mae = mean_absolute_error(pred_df["true_minutes"], pred_df["pred_minutes"])
    r2 = r2_score(pred_df["true_minutes"], pred_df["pred_minutes"])

    mlflow.log_params(params)
    mlflow.log_metrics({"mae": mae, "r2": r2})

    local_path = Path(f"/tmp/qwen_adapter_{run_name}")
    model.save_pretrained(local_path)
    processor.save_pretrained(local_path / "processor")
    mlflow.log_artifacts(str(local_path), artifact_path="model")

    from app.dl_train import QwenModelWrapper
    mlflow.pyfunc.log_model(
        artifact_path="pyfunc",
        python_model=QwenModelWrapper(processor_path=local_path / "processor",
                                      adapter_path=local_path),
        input_example=test_df.head(2)[["local_image_path", "text_context"]].to_dict(orient="records")
    )
    return mae, r2, local_path


def train_and_log_qwen(**kwargs):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_training_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    best_mae = float('inf')
    best_run_info = None

    with mlflow.start_run(run_name=PROJECT_NAME) as parent_run:
        for variant in TRAINING_VARIANTS:
            with mlflow.start_run(run_name=variant["name"], nested=True):
                mae, r2, model_path = train_qwen_variant(variant, train_df, test_df, variant["name"])
                if mae < best_mae:
                    best_mae = mae
                    best_run_info = (variant, mlflow.active_run().info.run_id, model_path)

    if best_run_info:
        best_variant, best_run_id, best_model_path = best_run_info
        with mlflow.start_run(run_id=best_run_id):
            registered_model_name = f"QwenRegressor_{VERSION}"
            from app.dl_train import QwenModelWrapper
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=QwenModelWrapper(
                    processor_path=best_model_path / "processor",
                    adapter_path=best_model_path
                ),
                registered_model_name=registered_model_name,
                input_example=test_df.head(2)[["local_image_path", "text_context"]].to_dict(orient="records")
            )
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(registered_model_name, stages=["None"])[0].version
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version,
                stage="Staging"
            )
            print(f"Registered best model: {registered_model_name} (MAE={best_mae})")

    return "done"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="qwen_clishe_training",
    default_args=default_args,
    description="Train Qwen2.5-VL for duration prediction and log to MLflow",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["qwen", "mlflow", "vision"],
) as dag:
    train_qwen = PythonOperator(
        task_id="train_qwen_models",
        python_callable=train_and_log_qwen,
        provide_context=True,
    )