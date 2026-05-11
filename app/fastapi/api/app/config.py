from dotenv import load_dotenv
import os

load_dotenv()

DB_CONNECTION_PARAMS = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

SERVICE_URL = os.getenv("ONEC_SERVICE_URL", "http://server-iis/unf/ws/AIWebServices?wsdl")
# Таймауты (секунды)
SOAP_TIMEOUT = int(os.getenv("SOAP_TIMEOUT", 30))
# Настройки логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- SMB (Windows network share) configuration ---
SMB_SERVER = os.getenv("SMB_SERVER", "server-sql")
SMB_USERNAME = os.getenv("SMB_USERNAME", "user1")
SMB_PASSWORD = os.getenv("SMB_PASSWORD", "pass1")
SMB_SHARE = os.getenv("SMB_SHARE", "UNF_Share")
SMB_PORT = int(os.getenv("SMB_PORT", "445"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

gShowMessage = 1
