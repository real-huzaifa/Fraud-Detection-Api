from pathlib import Path

# Project root = one level up from this file's folder (src/)
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
MERGED_FILE = PROCESSED_DIR / "merged.parquet"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

RANDOM_STATE = 42
OPERATING_THRESHOLD = 0.50