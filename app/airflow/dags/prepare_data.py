from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import json

CREATE_BUFFER_MERGED_SQL = """
CREATE TABLE IF NOT EXISTS buffer_merged (
    id                  SERIAL PRIMARY KEY,
    data                JSONB NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def merge_and_store_buffer(**context):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = pg_hook.get_sqlalchemy_engine()

    df_order = pd.read_sql("SELECT * FROM customer_orders", engine)
    df_plant = pd.read_sql("SELECT * FROM production_orders", engine)

    print(f"customer_orders shape: {df_order.shape}")
    print(f"production_orders shape: {df_plant.shape}")

    df_order['id_str'] = df_order['id'].astype(str)

    df_merged = df_order.merge(
        df_plant,
        left_on=['id_str', 'item_name'],
        right_on=['customer_order_id', 'item_name'],
        how='inner',
        suffixes=('_order', '_plant')
    )

    print(f"Merged shape: {df_merged.shape}")

    if df_merged.empty:
        raise ValueError("No matching rows found between customer_orders and production_orders")

    records = df_merged.to_dict(orient='records')
    json_records = [json.dumps(rec, ensure_ascii=False) for rec in records]

    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE buffer_merged;")
    for rec_json in json_records:
        cursor.execute("INSERT INTO buffer_merged (data) VALUES (%s);", (rec_json,))
    conn.commit()
    cursor.close()
    conn.close()

    print(f"Inserted {len(json_records)} rows into buffer_merged")


def create_buffer_table():
    hook = PostgresHook(postgres_conn_id='postgres_default')
    hook.run(CREATE_BUFFER_MERGED_SQL)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='prepare_data',
    default_args=default_args,
    description='Merge customer_orders and production_orders into buffer_merged',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['merge', 'buffer'],
) as dag:
    create_buffer = PythonOperator(
        task_id='create_buffer_table',
        python_callable=create_buffer_table,
    )

    merge_load = PythonOperator(
        task_id='merge_and_store_buffer',
        python_callable=merge_and_store_buffer,
    )

    create_buffer >> merge_load