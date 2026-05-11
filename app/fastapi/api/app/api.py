from datetime import datetime
from typing import Optional, List, Dict, Any

from zeep import Client, Settings, Transport
from zeep.exceptions import Fault, TransportError

from .config import SERVICE_URL, SOAP_TIMEOUT
from .logging import logger

settings = Settings(strict=False, xml_huge_tree=True)
client = Client(SERVICE_URL, settings=settings, transport=Transport(timeout=SOAP_TIMEOUT))


def call_predict(datetime_start: datetime, datetime_end: datetime,
                 request_number: int) -> List[Dict[str, Any]]:
    date_start_str = datetime_start.strftime("%Y-%m-%d")
    date_end_str = datetime_end.strftime("%Y-%m-%d")
    try:
        result = client.service.ПрогнозированиеИИ(
            ДатаС=date_start_str,
            ДатаПо=date_end_str,
            НомерЗапроса=request_number
        )
        return parse_1c_object(result)
    except (Fault, TransportError) as e:
        logger.error(f"SOAP call ПрогнозированиеИИ failed: {e}")
        raise


def call_get_images(datetime_start: datetime, datetime_end: datetime,
                    file_id: Optional[str] = None) -> List[Dict[str, Any]]:
    date_start_str = datetime_start.strftime("%Y-%m-%d")
    date_end_str = datetime_end.strftime("%Y-%m-%d")
    try:
        result = client.service.ПолучитьФайлыИзображений(
            ДатаС=date_start_str,
            ДатаПо=date_end_str,
            ФайлКартинки=file_id if file_id else None
        )
        return parse_1c_object(result)
    except (Fault, TransportError) as e:
        logger.error(f"SOAP call ПолучитьФайлыИзображений failed: {e}")
        raise


def parse_1c_object(xdto_obj) -> List[Dict[str, Any]]:
    items = []
    if hasattr(xdto_obj, 'Item') and isinstance(xdto_obj.Item, list):
        for row in xdto_obj.Item:
            row_dict = {}
            for attr_name in dir(row):
                if not attr_name.startswith('_') and not callable(getattr(row, attr_name)):
                    val = getattr(row, attr_name)
                    if hasattr(val, '__dict__'):
                        val = str(val)
                    row_dict[attr_name] = val
            items.append(row_dict)
    else:
        items.append({"message": str(xdto_obj)})
    return items