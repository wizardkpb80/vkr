from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    date_start: datetime
    date_end: datetime
    request_number: int


class PredictionResponse(BaseModel):
    items: List[dict]


class ImageRequest(BaseModel):
    date_start: datetime
    date_end: datetime
    file_id: Optional[str] = None


class ImageResponse(BaseModel):
    items: List[dict]


class PredictRequest(BaseModel):
    data: Dict[str, Any]


class PredictResponse(BaseModel):
    predicted_duration_minutes: float