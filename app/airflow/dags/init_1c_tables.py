from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from datetime import datetime

CREATE_CUSTOMER_ORDERS_SQL = """
CREATE TABLE IF NOT EXISTS customer_orders (
    id                  UUID PRIMARY KEY,
    date                TIMESTAMP,
    currency            VARCHAR(10),
    counterparty        TEXT,
    rate                NUMERIC,
    project             TEXT,
    status              TEXT,
    product_type        TEXT,
    item_name           TEXT,
    unit                VARCHAR(20),
    quantity            NUMERIC,
    price               NUMERIC,
    volume              NUMERIC,
    ntk_operation       TEXT,
    product_height      TEXT,
    product_length      TEXT,
    product_width       TEXT,
    material            TEXT,
    actual_length       NUMERIC,
    actual_width        NUMERIC,
    number_of_operations INTEGER,
    number_of_specifications INTEGER
);
"""

CREATE_PRODUCTION_ORDERS_SQL = """
CREATE TABLE IF NOT EXISTS production_orders (
    id                  UUID PRIMARY KEY,
    date                TIMESTAMP,
    operation_type      TEXT,
    customer_order_id   TEXT,
    status              TEXT,
    product_count       INTEGER,
    item_name           TEXT,
    quantity            NUMERIC,
    resource_enterprise TEXT,
    power_type          TEXT,
    start_time          TIMESTAMP,
    end_time            TIMESTAMP,
    artcam_time         TEXT,
    adjustment          TEXT,
    working_time        TEXT,
    manufacturing       TEXT,
    test_plate          TEXT,
    path_to_files       TEXT,
    required_tests      TEXT,
    tests_completed     TEXT
);
"""

CREATE_PRODUCTION_OPERATIONS_SQL = """
CREATE TABLE IF NOT EXISTS production_operations (
    id                  UUID PRIMARY KEY,
    operation           TEXT,
    standard_hours      NUMERIC,
    collet              TEXT,
    order_id            UUID
);
"""

CREATE_ORDER_IMAGES_SQL = """
CREATE TABLE IF NOT EXISTS order_images (
    id                  SERIAL PRIMARY KEY,
    order_uuid          UUID,
    order_number        VARCHAR(50),
    order_date          TIMESTAMP,
    file_path           TEXT,
    file_name           TEXT,
    extension           VARCHAR(10)
);
"""


def create_tables():
    hook = PostgresHook(postgres_conn_id='postgres_default')
    hook.run(CREATE_CUSTOMER_ORDERS_SQL)
    hook.run(CREATE_PRODUCTION_ORDERS_SQL)
    hook.run(CREATE_PRODUCTION_OPERATIONS_SQL)
    hook.run(CREATE_ORDER_IMAGES_SQL)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='init_1c_sql_tables',
    default_args=default_args,
    description='Create tables for 1C data',
    schedule_interval=None,
    start_date=datetime(2026, 5, 1),
    catchup=False,
    tags=['1c', 'init', 'postgres'],
) as dag:
    init_tables = PythonOperator(
        task_id='create_tables',
        python_callable=create_tables,
    )