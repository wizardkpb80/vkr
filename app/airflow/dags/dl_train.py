import os
import re
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sqlalchemy import create_engine
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from PIL import Image
from .config import DB_CONNECTION_PARAMS, SMB_SHARE
from .ml_train import filter_dataframe, fix_date_parsing, COLUMN_MAPPING, extract_first_3_words
from .utils import init_smb_session, read_file_smb
from .logging import logger

OUT_DIR = Path(os.getenv("QWEN_OUT_DIR", "/tmp/qwen_model"))
BASE_IMAGE_DIR = Path(os.getenv("QWEN_IMAGE_DIR", "/tmp/images"))
USE_SMB = os.getenv("USE_SMB", "true").lower() == "true"
MIN_PIXELS = 128 * 28
MAX_PIXELS = 28 * 28 * 256

FEATURE_COLS = [
    'resource_enterprise', 'capacity', 'price', 'actual_length', 'ntk_operation',
    'actual_width', 'project', 'quantity', 'customer_name',
    'item_short', 'product_length', 'product_width', 'material'
]


def create_smart_context(row: pd.Series) -> str:
    glossary = (
        "Industrial Context Definitions:\n"
        "- 'resource_enterprise': Machinery/Workstation ID.\n"
        "- 'ntk_operation': Specific technological step.\n"
        "- 'capacity', 'actual_length': Physical workload metrics.\n\n"
    )
    context = glossary + "Current Job Characteristics:\n"
    for col in FEATURE_COLS:
        if col in row and pd.notna(row[col]):
            context += f"- {col}: {row[col]}\n"
    return context


class QwenRegressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str = "file_path",
                 target_col: str = "duration_minutes"):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "image_path": row[self.image_col],
            "context": row.get("text_context", create_smart_context(row)),
            "duration": row[self.target_col] if self.target_col in row else None
        }


@dataclass
class QwenRegressionCollator:
    processor: any
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_image(self, rel_path: str) -> Image.Image:
        clean = rel_path.replace('\\', '/')
        local_path = BASE_IMAGE_DIR / clean
        if local_path.exists():
            return Image.open(local_path).convert("RGB")
        if USE_SMB:
            init_smb_session()
            try:
                data = read_file_smb(clean, share=SMB_SHARE, binary=True)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
                return Image.open(local_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load from SMB: {e}")
        return Image.new('RGB', (224, 224), color=(255, 255, 255))

    def __call__(self, batch):
        image_inputs = []
        messages = []
        for item in batch:
            img = self._load_image(item["image_path"])
            w, h = img.size
            if w / h > 100 or h / w > 100:
                img = Image.new('RGB', (224, 224), color=(255, 255, 255))
            image_inputs.append(img)
            msg = build_qwen_message(item["image_path"], item["context"],
                                     duration=item.get("duration"))
            messages.append(msg)

        visions = [process_vision_info(m) for m in messages]
        for i, v in enumerate(visions):
            v[0] = image_inputs[i]
        curr_images = [v[0] for v in visions]

        texts = [self.processor.apply_chat_template(m, tokenize=False,
                                                    add_generation_prompt=False)
                 for m in messages]

        inputs = self.processor(
            text=texts,
            images=curr_images,
            padding=True,
            return_tensors="pt",
            min_pixels=28 * 28,
            max_pixels=28 * 28 * 512
        ).to(self.device)

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100

        prompt_msgs = [build_qwen_message(item["image_path"], item["context"],
                                          duration=None) for item in batch]
        prompt_texts = [self.processor.apply_chat_template(m, tokenize=False,
                                                           add_generation_prompt=True)
                        for m in prompt_msgs]
        prompt_inputs = self.processor(text=prompt_texts, images=curr_images,
                                       padding=True, return_tensors="pt")
        p_lens = prompt_inputs["attention_mask"].sum(dim=1)

        for i, length in enumerate(p_lens):
            safe_len = min(length, labels.shape[1])
            labels[i, :safe_len] = -100

        inputs["labels"] = labels
        return inputs


def build_qwen_message(image_relative_path: str, context: str, duration: float = None):
    prompt = (f"{context}\nTask: Predict 'duration_minutes' for this manufacturing job. "
              "Output ONLY the numerical value in minutes.")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file://placeholder",
                 "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if duration is not None:
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": str(duration)}]})
    return messages


def load_training_data() -> pd.DataFrame:
    params = DB_CONNECTION_PARAMS
    engine = create_engine(
        f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['database']}"
    )
    sql = """
    SELECT b.data, oi.file_path, oi.order_uuid
    FROM buffer_merged b
    JOIN order_images oi ON (b.data->>'id') = oi.order_uuid::text
    """
    df_raw = pd.read_sql(sql, engine)
    if df_raw.empty:
        raise ValueError("No data found for training")

    records = [json.loads(row['data']) for _, row in df_raw.iterrows()]
    df_tab = pd.DataFrame(records)
    df_tab = df_tab.rename(columns=COLUMN_MAPPING)
    df_tab['file_path'] = df_raw['file_path'].values

    df = filter_dataframe(df_tab)
    df = fix_date_parsing(df)
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].abs()
    df = df[(df['duration_minutes'].notna()) & (df['duration_minutes'] > 0)].copy()

    df['item_short'] = df['item_name'].astype(str).apply(extract_first_3_words)
    df['text_context'] = df.apply(create_smart_context, axis=1)
    df = df[df['file_path'].notna()].reset_index(drop=True)
    return df


def train_qwen_model():
    logger.info("Loading training data...")
    df = load_training_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_ds = QwenRegressionDataset(train_df)
    test_ds = QwenRegressionDataset(test_df)
    collator = QwenRegressionCollator(processor, device=model.device)

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )
    logger.info("Starting Qwen training...")
    trainer.train()
    model.save_pretrained(OUT_DIR / "lora_adapter")
    processor.save_pretrained(OUT_DIR / "processor")
    logger.info(f"Model saved to {OUT_DIR}")

    pred_df = predict_qwen_model(test_df, model, processor)
    mae = np.mean(np.abs(pred_df["true_minutes"] - pred_df["pred_minutes"]))
    logger.info(f"Test MAE: {mae:.2f} minutes")
    return {"mae": mae, "model_path": str(OUT_DIR)}


def predict_qwen_model(df: pd.DataFrame = None, model=None, processor=None):
    if model is None or processor is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            OUT_DIR / "lora_adapter",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        processor = AutoProcessor.from_pretrained(OUT_DIR / "processor")
    model.eval()
    results = []
    ds = QwenRegressionDataset(df)
    collator = QwenRegressionCollator(processor, device=model.device)
    batch_size = 2
    for start in range(0, len(ds), batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        batch_inputs = collator(batch)
        with torch.no_grad():
            generated_ids = model.generate(
                **{k: v for k, v in batch_inputs.items() if k != "labels"},
                max_new_tokens=16,
                do_sample=False
            )
        prompt_len = batch_inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, prompt_len:]
        outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        for item, raw in zip(batch, outputs):
            match = re.search(r"[-+]?\d*\.\d+|\d+", raw)
            pred = float(match.group()) if match else 0.0
            results.append({
                "image_path": item["image_path"],
                "true_minutes": item["duration"],
                "pred_minutes": pred,
            })
    return pd.DataFrame(results)


def get_features_by_order_id(order_id: str) -> dict:
    engine = create_engine(
        f"postgresql+psycopg2://{DB_CONNECTION_PARAMS['user']}:{DB_CONNECTION_PARAMS['password']}@{DB_CONNECTION_PARAMS['host']}/{DB_CONNECTION_PARAMS['database']}"
    )
    sql = "SELECT data FROM buffer_merged WHERE data->>'id' = %s LIMIT 1"
    df = pd.read_sql(sql, engine, params=(order_id,))
    if df.empty:
        raise ValueError(f"Order {order_id} not found in buffer_merged")
    record = json.loads(df.iloc[0]['data'])
    df_row = pd.DataFrame([record]).rename(columns=COLUMN_MAPPING)

    df_row = filter_dataframe(df_row)
    df_row = fix_date_parsing(df_row)
    df_row['duration_minutes'] = (df_row['end_time'] - df_row['start_time']).dt.total_seconds() / 60
    df_row['duration_minutes'] = df_row['duration_minutes'].abs()
    df_row = df_row[(df_row['duration_minutes'].notna()) & (df_row['duration_minutes'] > 0)]
    if df_row.empty:
        raise ValueError(f"Order {order_id} failed filtering or has invalid duration")

    df_row['item_short'] = df_row['item_name'].astype(str).apply(extract_first_3_words)
    numeric_cols = ['quantity', 'price', 'capacity', 'product_height',
                    'product_length', 'product_width', 'actual_length',
                    'actual_width', 'number_of_operations', 'adjustment', 'manufacturing']
    categorical_cols = ['currency', 'customer_name', 'product_type', 'ntk_operation',
                        'product_type_order', 'material', 'resource_enterprise',
                        'project', 'item_short', 'test_plate']
    from .ml_train import handle_missing_values
    X = handle_missing_values(df_row[numeric_cols + categorical_cols],
                              numeric_cols, categorical_cols)
    import joblib
    try:
        freq_maps = joblib.load("/tmp/freq_maps.pkl")
        for col in categorical_cols:
            if col in freq_maps and col in X.columns:
                X[col] = X[col].map(freq_maps[col]).fillna(0)
    except FileNotFoundError:
        pass
    feature_cols = joblib.load("/tmp/feature_columns.txt") if open("/tmp/feature_columns.txt") else X.columns.tolist()
    X = X[feature_cols]
    return X.iloc[0].to_dict()


def get_qwen_features_by_order_id(order_id: str) -> dict:
    engine = create_engine(
        f"postgresql+psycopg2://{DB_CONNECTION_PARAMS['user']}:{DB_CONNECTION_PARAMS['password']}@{DB_CONNECTION_PARAMS['host']}/{DB_CONNECTION_PARAMS['database']}"
    )
    sql = """
    SELECT b.data, oi.file_path
    FROM buffer_merged b
    JOIN order_images oi ON (b.data->>'id') = oi.order_uuid::text
    WHERE b.data->>'id' = %s
    LIMIT 1
    """
    df = pd.read_sql(sql, engine, params=(order_id,))
    if df.empty:
        raise ValueError(f"Order {order_id} not found in buffer_merged or order_images")
    record = json.loads(df.iloc[0]['data'])
    df_row = pd.DataFrame([record]).rename(columns=COLUMN_MAPPING)
    df_row = filter_dataframe(df_row)
    df_row = fix_date_parsing(df_row)
    df_row['item_short'] = df_row['item_name'].astype(str).apply(extract_first_3_words)
    df_row['text_context'] = df_row.apply(create_smart_context, axis=1)
    image_path = df.iloc[0]['file_path']
    context = df_row.iloc[0]['text_context']
    return {"image_path": image_path, "text_context": context}