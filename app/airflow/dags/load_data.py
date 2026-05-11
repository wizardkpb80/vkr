from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger(__name__)

API_BASE_URL = "http://host.docker.internal:8000"
DATE_START = datetime(2023, 1, 1)
DATE_END = datetime(2027, 1, 1)

PREDICT_CONFIG = {
    1: {"table": "customer_orders", "pk": "id", "endpoint": "/1c/get_predict_data"},
    2: {"table": "production_orders", "pk": "id", "endpoint": "/1c/get_predict_data"},
    3: {"table": "production_operations", "pk": "id", "endpoint": "/1c/get_predict_data"},
}
IMAGES_CONFIG = {
    "table": "order_images",
    "unique_keys": ["order_uuid", "file_path"],
    "endpoint": "/1c/get_images",
}


def fetch_predict_data(request_number: int) -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}{PREDICT_CONFIG[request_number]['endpoint']}"
    payload = {
        "date_start": DATE_START.isoformat(),
        "date_end": DATE_END.isoformat(),
        "request_number": request_number
    }
    logger.info(f"Fetching predict data for request_number={request_number}")
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("items", [])


def fetch_images_data() -> List[Dict[str, Any]]:
    url = f"{API_BASE_URL}{IMAGES_CONFIG['endpoint']}"
    payload = {
        "date_start": DATE_START.isoformat(),
        "date_end": DATE_END.isoformat(),
        "file_id": None
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("items", [])


def insert_new_predict_data(request_number: int, **kwargs):
    items = fetch_predict_data(request_number)
    if not items:
        logger.info(f"No data for request_number={request_number}")
        return

    df = pd.DataFrame(items)
    table_name = PREDICT_CONFIG[request_number]["table"]
    pk = PREDICT_CONFIG[request_number]["pk"]

    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = pg_hook.get_sqlalchemy_engine()

    with engine.connect() as conn:
        existing_ids = pd.read_sql(f"SELECT {pk} FROM {table_name}", conn)

    if not existing_ids.empty:
        df = df[~df[pk].isin(existing_ids[pk])]

    if df.empty:
        logger.info(f"No new records for {table_name}")
        return

    df.to_sql(table_name, engine, if_exists='append', index=False)
    logger.info(f"Inserted {len(df)} new rows into {table_name}")


def insert_new_images_data(**kwargs):
    items = fetch_images_data()
    if not items:
        logger.info("No images data")
        return

    df = pd.DataFrame(items)
    table_name = IMAGES_CONFIG["table"]
    uk = IMAGES_CONFIG["unique_keys"]

    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = pg_hook.get_sqlalchemy_engine()

    with engine.connect() as conn:
        existing_df = pd.read_sql(f"SELECT {','.join(uk)} FROM {table_name}", conn)

    if not existing_df.empty:
        existing_df['_key'] = existing_df[uk[0]].astype(str) + "|" + existing_df[uk[1]].astype(str)
        df['_key'] = df[uk[0]].astype(str) + "|" + df[uk[1]].astype(str)
        df = df[~df['_key'].isin(existing_df['_key'])]
        df = df.drop(columns=['_key'])

    if df.empty:
        logger.info("No new images")
        return

    df.to_sql(table_name, engine, if_exists='append', index=False)
    logger.info(f"Inserted {len(df)} new rows into {table_name}")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='load_1c_incremental',
    default_args=default_args,
    description='Incremental load from 1C API to PostgreSQL',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['1c', 'postgres', 'incremental'],
) as dag:
    tasks = []
    for req_num in [1, 2, 3]:
        task = PythonOperator(
            task_id=f'insert_predict_req_{req_num}',
            python_callable=insert_new_predict_data,
            op_kwargs={'request_number': req_num},
        )
        tasks.append(task)

    images_task = PythonOperator(
        task_id='insert_images',
        python_callable=insert_new_images_data,
    )
    tasks.append(images_task)