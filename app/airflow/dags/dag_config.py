import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Optional

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", TEMP_DIR / "models"))
IMAGES_DIR = Path(os.getenv("QWEN_IMAGE_DIR", TEMP_DIR / "images"))

MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QwenConfig:
    output_dir: Path = Path(
        os.getenv("QWEN_OUT_DIR", MODELS_DIR / "qwen_model")
    )
    lora_adapter_dir: Path = Path(
        os.getenv("QWEN_LORA_DIR", MODELS_DIR / "qwen_model/lora_adapter")
    )
    processor_dir: Path = Path(
        os.getenv("QWEN_PROCESSOR_DIR", MODELS_DIR / "qwen_model/processor")
    )

    model_name: str = os.getenv(
        "QWEN_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    use_4bit: bool = os.getenv("QWEN_USE_4BIT", "true").lower() == "true"
    torch_dtype: str = os.getenv("TORCH_DTYPE", "float16")

    min_pixels: int = int(os.getenv("QWEN_MIN_PIXELS", "3584"))
    max_pixels: int = int(os.getenv("QWEN_MAX_PIXELS", "200704"))

    batch_size: int = int(os.getenv("QWEN_BATCH_SIZE", "1"))
    gradient_accumulation_steps: int = int(
        os.getenv("QWEN_GRADIENT_ACCUMULATION", "16")
    )
    num_epochs: int = int(os.getenv("QWEN_EPOCHS", "3"))
    learning_rate: float = float(os.getenv("QWEN_LEARNING_RATE", "2e-4"))
    warmup_steps: int = int(os.getenv("QWEN_WARMUP_STEPS", "50"))
    logging_steps: int = int(os.getenv("QWEN_LOGGING_STEPS", "10"))
    save_steps: int = int(os.getenv("QWEN_SAVE_STEPS", "100"))

    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_target_modules: List[str] = os.getenv(
        "LORA_TARGET_MODULES", "q_proj,v_proj,k_proj,o_proj"
    ).split(",")

    use_smb: bool = os.getenv("USE_SMB", "true").lower() == "true"
    use_gpu: bool = os.getenv("USE_GPU", "true").lower() == "true"

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_adapter_dir.mkdir(parents=True, exist_ok=True)
        self.processor_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class BaselineConfig:
    model_path: Path = Path(
        os.getenv("BASELINE_MODEL_PATH", MODELS_DIR / "baseline_model.pkl")
    )
    features_path: Path = Path(
        os.getenv("FEATURES_PATH", TEMP_DIR / "feature_columns.txt")
    )
    freq_maps_path: Path = Path(
        os.getenv("FREQ_MAPS_PATH", TEMP_DIR / "freq_maps.pkl")
    )

    n_estimators: int = int(os.getenv("BASELINE_N_ESTIMATORS", "100"))
    random_state: int = int(os.getenv("BASELINE_RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("BASELINE_TEST_SIZE", "0.2"))

    categorical_features: List[str] = os.getenv(
        "CATEGORICAL_FEATURES",
        "currency,customer_name,product_type,ntk_operation,"
        "product_type_order,material,resource_enterprise,project,"
        "item_short,test_plate"
    ).split(",")

    numeric_features: List[str] = os.getenv(
        "NUMERIC_FEATURES",
        "quantity,price,capacity,product_height,product_length,"
        "product_width,actual_length,actual_width,number_of_operations,"
        "adjustment,manufacturing"
    ).split(",")

    def __post_init__(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    training_query: str = os.getenv(
        "TRAINING_SQL_QUERY",
        """
        SELECT b.data, oi.file_path, oi.order_uuid
        FROM buffer_merged b
        JOIN order_images oi ON (b.data->>'id') = oi.order_uuid::text
        """
    )

    order_query: str = os.getenv(
        "ORDER_SQL_QUERY",
        "SELECT data FROM buffer_merged WHERE data->>'id' = %s LIMIT 1"
    )

    resource_filter: str = os.getenv("RESOURCE_FILTER", "Датрон|ПЛ")
    material_filter: str = os.getenv("MATERIAL_FILTER",
                                     "Латунь|Алюминий|Стеклотекстолит")
    exclude_units: str = os.getenv("EXCLUDE_UNITS", "усл|ч")

    min_duration_minutes: float = float(
        os.getenv("MIN_DURATION_MINUTES", "1")
    )
    max_duration_minutes: float = float(
        os.getenv("MAX_DURATION_MINUTES", "1440")
    )

    date_format: str = os.getenv("DATE_FORMAT", "%d.%m.%Y %H:%M:%S")


@dataclass
class SMBConfig:
    server: str = os.getenv("SMB_SERVER", "server-sql")
    username: str = os.getenv("SMB_USERNAME", "user1")
    password: str = os.getenv("SMB_PASSWORD", "pass1")
    share: str = os.getenv("SMB_SHARE", "UNF_Share")
    port: int = int(os.getenv("SMB_PORT", "445"))


@dataclass
class LoggingConfig:
    level: str = os.getenv("DAG_LOG_LEVEL", "INFO")
    format: str = os.getenv(
        "DAG_LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_path: Optional[Path] = (
        Path(os.getenv("DAG_LOG_FILE", TEMP_DIR / "dag_training.log"))
        if os.getenv("DAG_LOG_FILE") else None
    )


qwen_config = QwenConfig()
baseline_config = BaselineConfig()
data_config = DataConfig()
smb_config = SMBConfig()
logging_config = LoggingConfig()


def get_device() -> str:
    if qwen_config.use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_dtype():
    if qwen_config.torch_dtype == "float16":
        return torch.float16
    return torch.float32


def validate_configs():
    assert data_config.min_duration_minutes > 0, (
        "min_duration_minutes must be positive"
    )
    assert data_config.max_duration_minutes > data_config.min_duration_minutes, (
        "max_duration_minutes must be greater than min"
    )
    assert 0 < baseline_config.test_size < 1, (
        "test_size must be between 0 and 1"
    )
    assert qwen_config.learning_rate > 0, (
        "learning_rate must be positive"
    )


try:
    import torch
except ImportError:
    torch = None