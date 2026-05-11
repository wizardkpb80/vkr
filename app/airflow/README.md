# How to use Airflow locally
## Before Installation
- Install Docker
- Install Docker Compose
## Installation
- Execute `docker compose up`
- Open http://localhost:8080 in browser
- Login using airflow/airflow

## MLflow Integration

This project uses **MLflow** for experiment tracking, model registry, and artifact storage.

### MLflow Server (via Docker)
- The `docker-compose.yml` includes a dedicated MLflow service.
- Backend: PostgreSQL (`mlflow` database)
- Artifact root: mounted volume `./mlflow_artifacts`
- UI available at `http://localhost:5000`

### What is logged automatically?
- **Parameters** – learning rate, LoRA rank, batch size, number of epochs, etc.
- **Metrics** – MAE, R² on the test set.
- **Artifacts** – LoRA adapter weights, processor config, and a `pyfunc` model wrapper.

### Best Model Registration
After training all hyperparameter variants, the DAG:
1. Selects the model with the lowest MAE.
2. Registers it under the name `QwenRegressor_{SURNAME}` (e.g., `QwenRegressor_Datsiuk`).
3. Promotes it to the **`Staging`** stage in the MLflow Model Registry.

### How to use the registered model for inference
```python
import mlflow
model = mlflow.pyfunc.load_model("models:/QwenRegressor_Datsiuk/Staging")
predictions = model.predict(input_dataframe)