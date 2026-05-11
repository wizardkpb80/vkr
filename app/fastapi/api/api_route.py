import pandas as pd
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient

from api.schemas import (
    PredictionRequest, PredictionResponse, ImageRequest,
    ImageResponse, PredictRequest, PredictResponse
)
from .app.manager import get_prediction_data, get_images_data
from .app.logging import logger
from .app.ml_train import run_training, COLUMN_MAPPING, extract_first_3_words
from .app.predict import predict_model
from .app.dl_train import (
    train_qwen_model, predict_qwen_model as qwen_predict,
    create_smart_context, get_features_by_order_id,
    get_qwen_features_by_order_id, predict_qwen_model
)
from app.config import MLFLOW_TRACKING_URI

router = APIRouter(prefix="/1c", tags=["1c"])
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@router.post("/get_predict_data", response_model=PredictionResponse)
async def get_predict_data(request: PredictionRequest):
    try:
        items = get_prediction_data(
            date_start=request.date_start,
            date_end=request.date_end,
            request_number=request.request_number
        )
        return PredictionResponse(items=items)
    except Exception as e:
        logger.exception("Error in /predict")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/get_images", response_model=ImageResponse)
async def get_images(request: ImageRequest):
    try:
        items = get_images_data(
            date_start=request.date_start,
            date_end=request.date_end,
            file_id=request.file_id
        )
        return ImageResponse(items=items)
    except Exception as e:
        logger.exception("Error in /images")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/fit_baseline")
async def fit_baseline():
    try:
        metrics = run_training()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_baseline", response_model=List[PredictResponse])
async def predict_baseline(request: List[PredictRequest]):
    try:
        input_data = [item.dict() for item in request]
        predictions = predict_model(input_data)
        return [{"predicted_duration_minutes": pred} for pred in predictions]
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_baseline_by_order_id")
async def predict_baseline_by_order_id(request: dict):
    try:
        order_id = request.get("order_id")
        if not order_id:
            raise HTTPException(status_code=400, detail="Missing order_id")
        features = get_features_by_order_id(order_id)
        input_df = pd.DataFrame([features])
        prediction = predict_model(input_df)
        if isinstance(prediction, (list, np.ndarray)):
            pred_value = prediction[0]
        else:
            pred_value = float(prediction)
        return {"order_id": order_id, "predicted_duration_minutes": pred_value}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Prediction by order_id failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fit_qwen")
async def fit_qwen(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_qwen_model)
    return {"status": "training_started", "message": "Qwen training started in background"}


@router.post("/predict_qwen")
async def predict_qwen(df: dict):
    try:
        records = df.get("data", [])
        if not records:
            raise HTTPException(400, "Empty data")
        input_df = pd.DataFrame(records)
        inv_map = {v: k for k, v in COLUMN_MAPPING.items()}
        input_df = input_df.rename(columns=inv_map)
        input_df['item_short'] = input_df['item_name'].astype(str).apply(extract_first_3_words)
        input_df['text_context'] = input_df.apply(create_smart_context, axis=1)
        result_df = qwen_predict(input_df)
        predictions = result_df[["image_path", "pred_minutes"]].to_dict(orient="records")
        return {"predictions": predictions}
    except Exception as e:
        logger.exception("Qwen prediction failed")
        raise HTTPException(500, detail=str(e))


@router.post("/predict_qwen_by_order_id")
async def predict_qwen_by_order_id(request: dict):
    try:
        order_id = request.get("order_id")
        if not order_id:
            raise HTTPException(status_code=400, detail="Missing order_id")
        features = get_qwen_features_by_order_id(order_id)
        input_df = pd.DataFrame([features])
        result_df = predict_qwen_model(input_df)
        pred_value = result_df.iloc[0]['pred_minutes']
        return {"order_id": order_id, "predicted_duration_minutes": pred_value}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Qwen prediction by order_id failed")
        raise HTTPException(status_code=500, detail=str(e))


def get_qwen_models() -> List[Dict[str, Any]]:
    client = MlflowClient()
    all_models = client.search_registered_models()
    qwen_models = []
    for model in all_models:
        if model.name.startswith("QwenRegressor_"):
            latest_versions = client.get_latest_versions(model.name)
            qwen_models.append({
                "name": model.name,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.stage,
                        "run_id": v.run_id,
                        "status": v.status,
                        "created_at": v.creation_timestamp
                    } for v in latest_versions
                ],
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp
            })
    return qwen_models


def get_model_params(model_name: str, version: Optional[str] = None,
                     stage: Optional[str] = None) -> Dict[str, Any]:
    client = MlflowClient()
    if stage:
        latest = client.get_latest_versions(model_name, stages=[stage])
        if not latest:
            raise ValueError(f"No model found for name {model_name} in stage {stage}")
        version = latest[0].version
    elif version is None:
        raise ValueError("Either 'version' or 'stage' must be provided")

    model_version = client.get_model_version(model_name, version)
    run_id = model_version.run_id
    run = client.get_run(run_id)
    return {
        "model_name": model_name,
        "version": version,
        "stage": model_version.stage,
        "run_id": run_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags
    }


def load_qwen_model(model_name: str, stage: str = "Staging"):
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)


@router.get("/list_qwen_models")
async def list_qwen_models():
    try:
        models = get_qwen_models()
        return {"models": models}
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_qwen_params")
async def get_qwen_params(
        model_name: str = Query(..., description="Registered model name"),
        version: Optional[str] = Query(None, description="Specific version number"),
        stage: Optional[str] = Query(None, description="Stage: Staging, Production, Archived")
):
    try:
        params_info = get_model_params(model_name, version, stage)
        return params_info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to get model parameters")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_qwen_mlflow")
async def predict_qwen_mlflow(
        data: List[Dict[str, Any]],
        model_name: str = Query("QwenRegressor_Datsiuk", description="Registered model name"),
        stage: str = Query("Staging", description="Model stage")
):
    try:
        model = load_qwen_model(model_name, stage)
        input_df = pd.DataFrame(data)
        required_cols = ["image_path", "text_context"]
        missing = [c for c in required_cols if c not in input_df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        predictions = model.predict(input_df)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        return {"predictions": predictions}
    except Exception as e:
        logger.exception("Prediction via MLflow failed")
        raise HTTPException(status_code=500, detail=str(e))