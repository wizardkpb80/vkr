airflow setting:
DATE_START = Variable.get("LOAD_START_DATE", default_var="2023-01-01")
DATE_END = Variable.get("LOAD_END_DATE", default_var="2027-01-01")

airflow jobs:
- init_1c_tables.py - инициализация таблиц
- load_data.py - загрузка и обновление данных в БД
- prepare_data.py - подготовка данных для обработки

fastapi :
- get_predict_data
- get_images
- fit_baseline
- predict_baseline

feature_columns:
quantity
price
capacity
product_height
product_length
product_width
actual_length
actual_width
number_of_operations
adjustment
manufacturing
currency
customer_name
product_type
ntk_operation
product_type_order
material
resource_enterprise
project
item_short
test_plate

JSON Example:

[
  {
    "data": {
      "quantity": 1,
      "price": 1443,
      "capacity": 15.6,
      "product_height": "52",
      "product_length": "52",
      "product_width": "42",
      "actual_length": 52.0,
      "actual_width": 30.0,
      "number_of_operations": 1,
      "adjustment": 0,
      "manufacturing": 0,
      "currency": "руб.",
      "customer_name": "Client_1",
      "product_type": "Упаковка",
      "ntk_operation": "КТ",
      "product_type_order": "Упаковка",
      "material": "Латунь 7 мм",
      "resource_enterprise": "Датрон 3",
      "project": "Multi Tabak",
      "item_short": "Клише A 1_23 TEST_MT_H+K-59119S0059-SSP-M-ARM_final",
      "test_plate": ""
    }
  }
]
Response:
[
  {
    "predicted_duration_minutes": 42.3
  }
]

- predict_qwen_mlflow
model_name=QwenRegressor_Datsiuk&stage=Staging" \
  -H "Content-Type: application/json" \
  -d '[
    {"image_path": "/tmp/images/1.png", "text_context": "Resource: Datron, Material: Brass, Quantity: 1"},
    {"image_path": "/tmp/images/2.png", "text_context": "Resource: PL, Material: Aluminium, Quantity: 2"}
  ]'
Response:
{"predictions": [12.5, 23.7]}
