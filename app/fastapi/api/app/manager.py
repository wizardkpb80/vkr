from . import api
from .logging import logger
from datetime import datetime
from typing import Optional, List, Dict, Any

def get_prediction_data(date_start: datetime, date_end: datetime, request_number: int) -> List[Dict[str, Any]]:
    logger.info(f"Fetching prediction data from {date_start} to {date_end}, request №{request_number}")
    return api.call_predict(date_start, date_end, request_number)

def get_images_data(date_start: datetime, date_end: datetime, file_id: Optional[str] = None) -> List[Dict[str, Any]]:
    logger.info(f"Fetching images from {date_start} to {date_end}, file_id={file_id}")
    return api.call_get_images(date_start, date_end, file_id)